import logging
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model, TaskType
from typing import Any
logger = logging.getLogger('fedfl')

@dataclass
class ClientUpdate:
    client_id: Any
    state_dict: Any
    num_samples: Any
    loss: Any
    metrics: Any = field(default_factory=dict)

class FedExLoRA:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        from src.utils.config import get_method_params
        params = get_method_params(config)
        self.rank = params.get('rank', 8)
        self.alpha = params.get('alpha', params.get('lora_alpha', 16))
        self.target_modules = params.get('target_modules', ['query', 'value'])
        self.task_type_str = params.get('task_type', 'SEQ_CLS')
        self.lr_a = float(params.get('lr_a', 3e-05))
        self.lr_b = float(params.get('lr_b', 0.0003))
        self.lr_cls = float(params.get('lr_cls', 3e-05))
        self.local_epochs = params.get('local_epochs', 3)
        self.weight_decay = params.get('weight_decay', 0.01)
        self.grad_clip = params.get('grad_clip', 1.0)
        self.warmup = params.get('warmup', 'linear')
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.use_bf16 = params.get('use_bf16', True)
        if self.use_bf16 and (not torch.cuda.is_bf16_supported()):
            logging.getLogger('fedfl').warning('BF16 not supported, falling back to FP32')
            self.use_bf16 = False
        self.dirreg_lambda = float(config.get('training', {}).get('dirreg_lambda', 0.0))
        self.dirreg_detach_b = bool(config.get('training', {}).get('dirreg_detach_b', False))
        self.fedprox_mu = float(config.get('training', {}).get('fedprox_mu', 0.0))
        if self.dirreg_lambda > 0:
            mode = 'A-only (detach_b)' if self.dirreg_detach_b else 'A+B'
            logger.info(f'FEDEXLORA: DirReg enabled (lambda={self.dirreg_lambda}, mode={mode})')
        if self.fedprox_mu > 0:
            logger.info(f'FEDEXLORA: FedProx enabled (mu={self.fedprox_mu})')
        self._method_logger = None
        self._current_round = 0

    def set_method_logger(self, method_logger):
        self._method_logger = method_logger

    def setup_model(self, model):
        lora_config = LoraConfig(r=self.rank, lora_alpha=self.alpha, target_modules=self.target_modules, lora_dropout=0.1, bias='none', task_type=getattr(TaskType, self.task_type_str, TaskType.SEQ_CLS))
        peft_model = get_peft_model(model, lora_config)
        for (name, param) in peft_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
            elif ('classifier' in name or 'qa_outputs' in name) and 'modules_to_save' in name or name.endswith('.head.weight') or name.endswith('.head.bias'):
                param.requires_grad = True
        return peft_model

    def local_train(self, model, train_loader, client_id, **kwargs):
        model = model.to(self.device)
        model.train()
        lr_factor = kwargs.get('lr_factor', 1.0)
        _ref_a = {}
        _ref_b = {}
        _ref_all = {}
        if self.dirreg_lambda > 0 or self.fedprox_mu > 0:
            from src.utils.regularization import capture_lora_reference
            (_ref_a, _ref_b, _ref_all) = capture_lora_reference(model)
        trainable_params = self.get_trainable_params(model)
        params_A = []
        params_B = []
        params_classifier = []
        for (name, param) in trainable_params.items():
            if 'lora_A' in name:
                params_A.append(param)
            elif 'lora_B' in name:
                params_B.append(param)
            elif 'classifier' in name or 'qa_outputs' in name or name.endswith('.head.weight') or name.endswith('.head.bias'):
                params_classifier.append(param)
        param_groups = []
        if params_A:
            param_groups.append({'params': params_A, 'lr': self.lr_a * lr_factor})
        if params_B:
            param_groups.append({'params': params_B, 'lr': self.lr_b * lr_factor})
        if params_classifier:
            param_groups.append({'params': params_classifier, 'lr': self.lr_cls * lr_factor})
        optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)
        total_loss = 0.0
        num_batches = 0
        _reg_diag = {}
        for _ in range(self.local_epochs):
            for batch in train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for (k, v) in batch.items()}
                optimizer.zero_grad()
                if self.use_bf16:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(**batch)
                        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                    if self.dirreg_lambda > 0 and _ref_a:
                        from src.utils.regularization import compute_dirreg_loss
                        (dir_loss, dir_diag) = compute_dirreg_loss(model, _ref_a, _ref_b, detach_b=self.dirreg_detach_b)
                        loss = loss + self.dirreg_lambda * dir_loss
                        _reg_diag.update(dir_diag)
                    if self.fedprox_mu > 0 and _ref_all:
                        from src.utils.regularization import compute_fedprox_loss
                        (prox_loss, prox_diag) = compute_fedprox_loss(model, _ref_all)
                        loss = loss + self.fedprox_mu * prox_loss
                        _reg_diag.update(prox_diag)
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                else:
                    outputs = model(**batch)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                    if self.dirreg_lambda > 0 and _ref_a:
                        from src.utils.regularization import compute_dirreg_loss
                        (dir_loss, dir_diag) = compute_dirreg_loss(model, _ref_a, _ref_b, detach_b=self.dirreg_detach_b)
                        loss = loss + self.dirreg_lambda * dir_loss
                        _reg_diag.update(dir_diag)
                    if self.fedprox_mu > 0 and _ref_all:
                        from src.utils.regularization import compute_fedprox_loss
                        (prox_loss, prox_diag) = compute_fedprox_loss(model, _ref_all)
                        loss = loss + self.fedprox_mu * prox_loss
                        _reg_diag.update(prox_diag)
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                if num_batches == 0 and self._method_logger:
                    (_ga, _gb, _gc, _ac, _bc, _cc) = (0.0, 0.0, 0.0, 0, 0, 0)
                    for (name, param) in model.named_parameters():
                        if param.grad is not None:
                            g = param.grad.float().norm().item()
                            if 'lora_A' in name:
                                _ga += g
                                _ac += 1
                            elif 'lora_B' in name:
                                _gb += g
                                _bc += 1
                            elif 'classifier' in name or 'qa_outputs' in name or name.endswith('.head.weight') or name.endswith('.head.bias'):
                                _gc += g
                                _cc += 1
                    self._method_logger.log_gradient_stats(round_num=self._current_round, client_id=client_id, grad_a_norm=_ga / max(_ac, 1), grad_b_norm=_gb / max(_bc, 1), grad_m_norm=0.0, grad_cls_norm=_gc / max(_cc, 1))
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)
        lora_state_dict = {name: param.data.detach().clone().cpu() for (name, param) in trainable_params.items()}
        return ClientUpdate(client_id=client_id, state_dict=lora_state_dict, num_samples=len(train_loader.dataset), loss=avg_loss, metrics={'train_loss': avg_loss, **_reg_diag})

    def aggregate(self, global_model, client_updates, round_idx=-1, **kwargs):
        if not client_updates:
            raise ValueError('No client updates provided')
        if round_idx >= 0:
            self._current_round = round_idx
        total_samples = sum((update.num_samples for update in client_updates))
        if total_samples == 0:
            raise ValueError('Total samples is zero - all clients have empty data')
        num_clients = len(client_updates)
        global_state = global_model.state_dict()
        target_device = next(global_model.parameters()).device
        lora_pairs = self._identify_lora_pairs(client_updates[0].state_dict)
        for (layer_prefix, (a_name, b_name)) in lora_pairs.items():
            A_matrices = []
            B_matrices = []
            weights = []
            for update in client_updates:
                A_matrices.append(update.state_dict[a_name])
                B_matrices.append(update.state_dict[b_name])
                weights.append(update.num_samples / total_samples)
            A_stacked = torch.stack(A_matrices).to(target_device)
            B_stacked = torch.stack(B_matrices).to(target_device)
            weights_tensor = torch.tensor(weights, dtype=A_stacked.dtype, device=target_device)
            A_avg = torch.einsum('k,krn->rn', weights_tensor, A_stacked)
            B_avg = torch.einsum('k,kdr->dr', weights_tensor, B_stacked)
            M = torch.zeros(B_avg.shape[0], A_avg.shape[1], dtype=A_avg.dtype, device=target_device)
            for i in range(num_clients):
                M += weights[i] * (B_stacked[i] @ A_stacked[i])
            product_of_avg = B_avg @ A_avg
            residual = M - product_of_avg
            scaling_factor = self.alpha / self.rank
            scaled_residual = scaling_factor * residual
            base_weight_name = self._find_base_weight_name(layer_prefix, global_state)
            if base_weight_name is not None:
                base_weight = global_state[base_weight_name]
                global_state[base_weight_name] = base_weight + scaled_residual.to(base_weight.dtype)
            else:
                logging.getLogger('fedfl').warning(f'FedExLoRA: base weight not found for {layer_prefix}, residual not merged')
            global_state[a_name] = A_avg
            global_state[b_name] = B_avg
        classifier_params = self._get_classifier_params(client_updates[0].state_dict)
        for param_name in classifier_params:
            aggregated = torch.zeros_like(client_updates[0].state_dict[param_name], device=target_device)
            for update in client_updates:
                weight = update.num_samples / total_samples
                aggregated += weight * update.state_dict[param_name].to(target_device)
            global_state[param_name] = aggregated
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
            self._method_logger.log_aggregation(round_num=self._current_round, num_clients=len(client_updates), total_samples=total_samples, aggregation_method='fedexlora', custom_metrics=avg_metrics)
            lr_factor = kwargs.get('lr_factor', 1.0)
            self._method_logger.log_aggregate_analysis(round_num=self._current_round, global_model=global_model, client_updates=client_updates, lr_factor=lr_factor, effective_lrs={'lr_a': self.lr_a * lr_factor, 'lr_b': self.lr_b * lr_factor, 'lr_cls': self.lr_cls * lr_factor}, has_magnitude=False)
        return global_model

    def _identify_lora_pairs(self, state_dict):
        lora_pairs = {}
        for name in state_dict.keys():
            if 'lora_A' in name:
                prefix = name.split('lora_A')[0]
                b_name = name.replace('lora_A', 'lora_B')
                if b_name in state_dict:
                    lora_pairs[prefix] = (name, b_name)
        return lora_pairs

    def _find_base_weight_name(self, layer_prefix, state_dict):
        possible_names = [f'{layer_prefix}base_layer.weight', f'{layer_prefix}weight']
        for name in possible_names:
            if name in state_dict:
                return name
        clean_prefix = layer_prefix.rstrip('.')
        for name in state_dict.keys():
            if clean_prefix in name and 'weight' in name and ('lora' not in name):
                if not any((x in name for x in ['lora_A', 'lora_B', 'scaling'])):
                    return name
        return None

    def _get_classifier_params(self, state_dict):
        return [name for name in state_dict.keys() if 'classifier' in name or 'qa_outputs' in name or name.endswith('.head.weight') or name.endswith('.head.bias')]

    def get_trainable_params(self, model):
        result = {}
        for (name, param) in model.named_parameters():
            if not param.requires_grad:
                continue
            is_lora = 'lora' in name.lower()
            is_trainable_classifier = ('classifier' in name or 'qa_outputs' in name) and 'modules_to_save' in name or name.endswith('.head.weight') or name.endswith('.head.bias')
            if is_lora or is_trainable_classifier:
                result[name] = param
        return result
