import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from torch import nn
from torch import optim
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

class LoRAA2:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        from src.utils.config import get_method_params
        params = get_method_params(config)
        self.global_rank = params.get('global_rank', 8)
        self.client_rank_ratio = params.get('client_rank_ratio', 0.5)
        self.alpha = params.get('alpha', params.get('lora_alpha', 16))
        self.target_modules = params.get('target_modules', ['query', 'value'])
        self.task_type_str = params.get('task_type', 'SEQ_CLS')
        self.lr_b = float(params.get('lr_b', 0.0003))
        self.lr_a = float(params.get('lr_a', 3e-05))
        self.lr_cls = float(params.get('lr_cls', 3e-05))
        self.local_epochs = params.get('local_epochs', 3)
        self.weight_decay = params.get('weight_decay', 0.01)
        self.grad_clip = params.get('grad_clip', 1.0)
        self.warmup = params.get('warmup', 'linear')
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.use_bf16 = params.get('use_bf16', True)
        if self.use_bf16 and (not torch.cuda.is_bf16_supported()):
            logger.warning('BF16 not supported, falling back to FP32')
            self.use_bf16 = False
        self.current_round = 0
        self._method_logger = None

    def set_method_logger(self, method_logger):
        self._method_logger = method_logger

    def setup_model(self, model):
        lora_config = LoraConfig(r=self.global_rank, lora_alpha=self.alpha, target_modules=self.target_modules, lora_dropout=0.1, bias='none', task_type=getattr(TaskType, self.task_type_str, TaskType.SEQ_CLS))
        peft_model = get_peft_model(model, lora_config)
        for (name, param) in peft_model.named_parameters():
            if 'lora_A' in name:
                param.requires_grad = False
            elif 'lora_B' in name:
                param.requires_grad = True
            elif ('classifier' in name or 'qa_outputs' in name) and 'modules_to_save' in name or name.endswith('.head.weight') or name.endswith('.head.bias'):
                param.requires_grad = True
        return peft_model

    def _is_even_round(self):
        return self.current_round % 2 == 0

    def _freeze_for_round(self, model):
        is_even = self._is_even_round()
        train_B = is_even
        train_A = not is_even
        for (name, param) in model.named_parameters():
            if 'lora_A' in name:
                param.requires_grad = train_A
            elif 'lora_B' in name:
                param.requires_grad = train_B
            elif ('classifier' in name or 'qa_outputs' in name) and 'modules_to_save' in name or name.endswith('.head.weight') or name.endswith('.head.bias'):
                param.requires_grad = True

    def _compute_rank_scores(self, model, initial_state):
        scores = {}
        train_B = self._is_even_round()
        module_params = {}
        for (name, param) in model.named_parameters():
            if 'lora' in name.lower():
                parts = name.rsplit('.', 2)
                if len(parts) >= 2:
                    module_name = parts[0]
                    if module_name not in module_params:
                        module_params[module_name] = {}
                    if 'lora_A' in name:
                        module_params[module_name]['A'] = param.data
                        module_params[module_name]['A_init'] = initial_state.get(name, param.data)
                    elif 'lora_B' in name:
                        module_params[module_name]['B'] = param.data
                        module_params[module_name]['B_init'] = initial_state.get(name, param.data)
        for (module_name, params) in module_params.items():
            if 'A' not in params or 'B' not in params:
                continue
            A = params['A']
            B = params['B']
            rank = A.shape[0]
            module_scores = torch.zeros(rank, device=self.device)
            if train_B:
                delta_B = B - params['B_init']
                for i in range(rank):
                    contribution = torch.outer(delta_B[:, i], A[i, :])
                    module_scores[i] = torch.norm(contribution, p='fro')
            else:
                delta_A = A - params['A_init']
                for i in range(rank):
                    contribution = torch.outer(B[:, i], delta_A[i, :])
                    module_scores[i] = torch.norm(contribution, p='fro')
            scores[module_name] = module_scores
        return scores

    def _select_top_ranks(self, scores, num_select):
        all_scores = []
        module_info = []
        for (module_name, module_scores) in scores.items():
            for (i, score) in enumerate(module_scores):
                all_scores.append(score.item())
                module_info.append((module_name, i))
        if not all_scores:
            return {}
        all_scores_tensor = torch.tensor(all_scores, device=self.device)
        k = min(num_select, len(all_scores))
        (_, top_indices) = torch.topk(all_scores_tensor, k)
        masks = {name: torch.zeros(len(s), device=self.device, dtype=torch.bool) for (name, s) in scores.items()}
        for idx in top_indices:
            (module_name, rank_idx) = module_info[idx.item()]
            masks[module_name][rank_idx] = True
        return masks

    def _apply_rank_mask(self, updates, masks):
        masked_updates = {}
        for (name, update) in updates.items():
            mask = None
            for (module_name, module_mask) in masks.items():
                if module_name in name:
                    mask = module_mask
                    break
            if mask is not None:
                if 'lora_A' in name:
                    masked = update.clone()
                    masked[~mask, :] = 0
                    masked_updates[name] = masked
                elif 'lora_B' in name:
                    masked = update.clone()
                    masked[:, ~mask] = 0
                    masked_updates[name] = masked
            else:
                masked_updates[name] = update
        return masked_updates

    def local_train(self, model, train_loader, client_id, **kwargs):
        model = model.to(self.device)
        model.train()
        lr_factor = kwargs.get('lr_factor', 1.0)
        self._freeze_for_round(model)
        initial_state = {}
        for (name, param) in model.named_parameters():
            is_lora = 'lora' in name.lower()
            is_trainable_classifier = ('classifier' in name or 'qa_outputs' in name) and 'modules_to_save' in name or name.endswith('.head.weight') or name.endswith('.head.bias')
            if is_lora or is_trainable_classifier:
                initial_state[name] = param.clone().detach()
        trainable_params = []
        is_even = self._is_even_round()
        train_B = is_even
        for (name, param) in model.named_parameters():
            if param.requires_grad:
                if 'classifier' in name or 'qa_outputs' in name:
                    lr = self.lr_cls * lr_factor
                elif 'lora_B' in name or (train_B and 'lora' in name.lower()):
                    lr = self.lr_b * lr_factor
                else:
                    lr = self.lr_a * lr_factor
                trainable_params.append({'params': param, 'lr': lr})
        if not trainable_params:
            return ClientUpdate(client_id=client_id, state_dict={}, num_samples=len(train_loader.dataset), loss=0.0, metrics={'train_loss': 0.0, 'round': self.current_round})
        optimizer = optim.AdamW(trainable_params, weight_decay=self.weight_decay)
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
                            elif 'classifier' in name or 'qa_outputs' in name:
                                _gc += g
                                _cc += 1
                    self._method_logger.log_gradient_stats(round_num=self.current_round, client_id=client_id, grad_a_norm=_ga / max(_ac, 1), grad_b_norm=_gb / max(_bc, 1), grad_m_norm=0.0, grad_cls_norm=_gc / max(_cc, 1))
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)
        scores = self._compute_rank_scores(model, initial_state)
        num_modules = len(scores)
        total_ranks = self.global_rank * num_modules
        num_select = int(self.client_rank_ratio * total_ranks)
        masks = self._select_top_ranks(scores, num_select)
        updates = {}
        for (name, param) in model.named_parameters():
            if param.requires_grad and name in initial_state:
                delta = param.data - initial_state[name]
                updates[name] = delta.detach().clone()
        masked_updates = self._apply_rank_mask(updates, masks)
        cpu_updates = {name: tensor.detach().clone().cpu() for (name, tensor) in masked_updates.items()}
        return ClientUpdate(client_id=client_id, state_dict=cpu_updates, num_samples=len(train_loader.dataset), loss=avg_loss, metrics={'train_loss': avg_loss, 'round': self.current_round, 'train_B': train_B, 'train_A': not train_B, 'selected_ranks': num_select})

    def aggregate(self, global_model, client_updates, round_idx=-1, **kwargs):
        if round_idx >= 0:
            self.current_round = round_idx
        if not client_updates:
            raise ValueError('No client updates provided')
        total_samples = sum((update.num_samples for update in client_updates))
        if total_samples == 0:
            raise ValueError('Total samples is zero - all clients have empty data')
        global_state = global_model.state_dict()
        aggregated_updates = {}
        for update in client_updates:
            weight = update.num_samples / total_samples
            for (param_name, delta) in update.state_dict.items():
                if param_name not in aggregated_updates:
                    aggregated_updates[param_name] = torch.zeros_like(delta)
                aggregated_updates[param_name] += weight * delta
        for (param_name, aggregated_delta) in aggregated_updates.items():
            if param_name in global_state:
                current_param = global_state[param_name]
                aggregated_delta = aggregated_delta.to(current_param.device)
                global_state[param_name] = current_param + aggregated_delta
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
            self._method_logger.log_aggregation(round_num=self.current_round, num_clients=len(client_updates), total_samples=total_samples, aggregation_method='loraa2', custom_metrics=avg_metrics)
            lr_factor = kwargs.get('lr_factor', 1.0)
            self._method_logger.log_aggregate_analysis(round_num=self.current_round, global_model=global_model, client_updates=client_updates, lr_factor=lr_factor, effective_lrs={'lr_a': self.lr_a * lr_factor, 'lr_b': self.lr_b * lr_factor, 'lr_cls': self.lr_cls * lr_factor}, has_magnitude=False)
        self.current_round = round_idx + 1 if round_idx >= 0 else self.current_round + 1
        return global_model

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
