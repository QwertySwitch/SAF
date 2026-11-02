import logging
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model, TaskType
from typing import Any
logger = logging.getLogger('fedfl')

@dataclass
class ClientUpdate:
    client_id: Any
    state_dict: Any
    num_samples: Any
    loss: Any
    metrics: Any = None

class FFALoRA:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        from src.utils.config import get_method_params
        params = get_method_params(config)
        self.rank = params.get('rank', 8)
        self.alpha = params.get('alpha', params.get('lora_alpha', 16))
        self.target_modules = params.get('target_modules', ['query', 'value'])
        self.task_type_str = params.get('task_type', 'SEQ_CLS')
        self.lr = float(params.get('lr', 0.0003))
        self.lr_cls = float(params.get('lr_cls', 3e-05))
        self.local_epochs = params.get('local_epochs', 3)
        self.weight_decay = params.get('weight_decay', 0.01)
        self.grad_clip = params.get('grad_clip', 1.0)
        self.use_bf16 = params.get('use_bf16', True)
        if self.use_bf16 and (not torch.cuda.is_bf16_supported()):
            logger.warning('BF16 not supported, falling back to FP32')
            self.use_bf16 = False
        self.warmup = params.get('warmup', 'linear')
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self._method_logger = None
        self._current_round = 0

    def set_method_logger(self, method_logger):
        self._method_logger = method_logger

    def setup_model(self, model):
        lora_config = LoraConfig(r=self.rank, lora_alpha=self.alpha, target_modules=self.target_modules, lora_dropout=0.1, bias='none', task_type=getattr(TaskType, self.task_type_str, TaskType.SEQ_CLS))
        peft_model = get_peft_model(model, lora_config)
        for (name, param) in peft_model.named_parameters():
            if 'lora_A' in name:
                param.requires_grad = False
            elif 'lora_B' in name:
                param.requires_grad = True
            elif ('classifier' in name or 'qa_outputs' in name) and 'modules_to_save' in name or name.endswith('.head.weight') or name.endswith('.head.bias'):
                param.requires_grad = True
        return peft_model

    def local_train(self, model, train_loader, client_id, **kwargs):
        model = model.to(self.device)
        model.train()
        lr_factor = kwargs.get('lr_factor', 1.0)
        trainable_params = self.get_trainable_params(model)
        params_B = []
        params_classifier = []
        for (name, param) in trainable_params.items():
            if 'lora_B' in name:
                params_B.append(param)
            elif 'classifier' in name or 'qa_outputs' in name or name.endswith('.head.weight') or name.endswith('.head.bias'):
                params_classifier.append(param)
        param_groups = []
        if params_B:
            param_groups.append({'params': params_B, 'lr': self.lr * lr_factor})
        if params_classifier:
            param_groups.append({'params': params_classifier, 'lr': self.lr_cls * lr_factor})
        optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)
        total_loss = 0.0
        num_batches = 0
        for _ in range(self.local_epochs):
            for batch in train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for (k, v) in batch.items()}
                optimizer.zero_grad()
                if self.use_bf16:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(**batch)
                        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                else:
                    outputs = model(**batch)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                if num_batches == 0 and self._method_logger:
                    (_gb, _gc, _bc, _cc) = (0.0, 0.0, 0, 0)
                    for (name, param) in model.named_parameters():
                        if param.grad is not None:
                            g = param.grad.float().norm().item()
                            if 'lora_B' in name:
                                _gb += g
                                _bc += 1
                            elif 'classifier' in name or 'qa_outputs' in name or name.endswith('.head.weight') or name.endswith('.head.bias'):
                                _gc += g
                                _cc += 1
                    self._method_logger.log_gradient_stats(round_num=self._current_round, client_id=client_id, grad_a_norm=0.0, grad_b_norm=_gb / max(_bc, 1), grad_m_norm=0.0, grad_cls_norm=_gc / max(_cc, 1))
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)
        lora_b_state_dict = {name: param.data.detach().clone().cpu() for (name, param) in trainable_params.items()}
        return ClientUpdate(client_id=client_id, state_dict=lora_b_state_dict, num_samples=len(train_loader.dataset), loss=avg_loss, metrics={'train_loss': avg_loss})

    def aggregate(self, global_model, client_updates, round_idx=-1, **kwargs):
        if not client_updates:
            raise ValueError('No client updates provided')
        if round_idx >= 0:
            self._current_round = round_idx
        total_samples = sum((update.num_samples for update in client_updates))
        if total_samples == 0:
            raise ValueError('Total samples is zero - all clients have empty data')
        global_state = global_model.state_dict()
        param_names = client_updates[0].state_dict.keys()
        for param_name in param_names:
            aggregated_param = torch.zeros_like(client_updates[0].state_dict[param_name])
            for update in client_updates:
                weight = update.num_samples / total_samples
                aggregated_param += weight * update.state_dict[param_name]
            global_state[param_name] = aggregated_param
        global_model.load_state_dict(global_state, strict=False)
        if self._method_logger is not None:
            avg_metrics = {}
            if client_updates:
                all_keys = set()
                for u in client_updates:
                    if u.metrics:
                        all_keys.update(u.metrics.keys())
                for key in all_keys:
                    values = [u.metrics[key] for u in client_updates if u.metrics and key in u.metrics]
                    if values:
                        avg_metrics[f'avg_{key}'] = sum(values) / len(values)
            self._method_logger.log_aggregation(round_num=self._current_round, num_clients=len(client_updates), total_samples=total_samples, aggregation_method='ffalora', custom_metrics=avg_metrics)
            lr_factor = kwargs.get('lr_factor', 1.0)
            self._method_logger.log_aggregate_analysis(round_num=self._current_round, global_model=global_model, client_updates=client_updates, lr_factor=lr_factor, effective_lrs={'lr': self.lr * lr_factor, 'lr_cls': self.lr_cls * lr_factor}, has_magnitude=False)
        return global_model

    def get_trainable_params(self, model):
        result = {}
        for (name, param) in model.named_parameters():
            if not param.requires_grad:
                continue
            is_lora_b = 'lora_B' in name
            is_trainable_classifier = ('classifier' in name or 'qa_outputs' in name) and 'modules_to_save' in name or name.endswith('.head.weight') or name.endswith('.head.bias')
            if is_lora_b or is_trainable_classifier:
                result[name] = param
        return result
