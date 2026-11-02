import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
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

class HetLoRA:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        from src.utils.config import get_method_params
        params = get_method_params(config)
        self.max_rank = params.get('max_rank', 12)
        self.rank_pool = params.get('rank_pool', [4, 8, 12])
        self.lora_alpha = params.get('alpha', params.get('lora_alpha', 16))
        self.target_modules = params.get('target_modules', ['query', 'value'])
        self.task_type_str = params.get('task_type', 'SEQ_CLS')
        for r in self.rank_pool:
            if r > self.max_rank:
                raise ValueError(f'rank_pool entry {r} exceeds max_rank {self.max_rank}')
        self.lr = float(params.get('lr', 0.0003))
        self.local_epochs = params.get('local_epochs', 1)
        self.weight_decay = float(params.get('weight_decay', 0.01))
        self.grad_clip = float(params.get('grad_clip', 1.0))
        self.warmup = params.get('warmup', 'linear')
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.use_bf16 = params.get('use_bf16', True)
        if self.use_bf16 and (not torch.cuda.is_bf16_supported()):
            logger.warning('BF16 not supported, falling back to FP32')
            self.use_bf16 = False
        if isinstance(config, dict):
            num_clients = config.get('dataset', {}).get('num_clients', 20)
        else:
            num_clients = getattr(getattr(config, 'dataset', None), 'num_clients', 20)
        self.client_ranks = {}
        for cid in range(num_clients):
            self.client_ranks[cid] = self.rank_pool[cid % len(self.rank_pool)]
        rank_counts = {}
        for r in self.client_ranks.values():
            rank_counts[r] = rank_counts.get(r, 0) + 1
        avg_rank = sum(self.client_ranks.values()) / len(self.client_ranks)
        self.dirreg_lambda = float(config.get('training', {}).get('dirreg_lambda', 0.0))
        self.dirreg_detach_b = bool(config.get('training', {}).get('dirreg_detach_b', False))
        self.fedprox_mu = float(config.get('training', {}).get('fedprox_mu', 0.0))
        if self.dirreg_lambda > 0:
            mode = 'A-only (detach_b)' if self.dirreg_detach_b else 'A+B'
            logger.info(f'HETLORA: DirReg enabled (lambda={self.dirreg_lambda}, mode={mode})')
        if self.fedprox_mu > 0:
            logger.info(f'HETLORA: FedProx enabled (mu={self.fedprox_mu})')
        self._method_logger = None
        self._current_round = 0
        logger.info(f'HetLoRA initialized: max_rank={self.max_rank}, rank_pool={self.rank_pool}, rank_distribution={rank_counts}, avg_rank={avg_rank:.1f}')

    def set_method_logger(self, method_logger):
        self._method_logger = method_logger

    def setup_model(self, model):
        lora_config = LoraConfig(r=self.max_rank, lora_alpha=self.lora_alpha, target_modules=self.target_modules, lora_dropout=0.1, bias='none', task_type=getattr(TaskType, self.task_type_str, TaskType.SEQ_CLS))
        peft_model = get_peft_model(model, lora_config)
        trainable_count = 0
        for (name, param) in peft_model.named_parameters():
            is_lora = 'lora_A' in name or 'lora_B' in name
            is_trainable_classifier = ('classifier' in name or 'qa_outputs' in name) and 'modules_to_save' in name or name.endswith('.head.weight') or name.endswith('.head.bias')
            if is_lora or is_trainable_classifier:
                param.requires_grad = True
                trainable_count += param.numel()
        logger.info(f'HetLoRA setup: max_rank={self.max_rank}, {trainable_count:,} trainable params')
        return peft_model

    def _get_client_rank(self, client_id):
        if client_id in self.client_ranks:
            return self.client_ranks[client_id]
        return self.rank_pool[client_id % len(self.rank_pool)]

    def _mask_to_rank(self, model, rank):
        with torch.no_grad():
            for (name, param) in model.named_parameters():
                if 'lora_A' in name and param.dim() == 2:
                    if rank < param.shape[0]:
                        param.data[rank:, :] = 0
                elif 'lora_B' in name and param.dim() == 2:
                    if rank < param.shape[1]:
                        param.data[:, rank:] = 0

    def _mask_gradients(self, model, rank):
        for (name, param) in model.named_parameters():
            if param.grad is None:
                continue
            if 'lora_A' in name and param.grad.dim() == 2:
                if rank < param.grad.shape[0]:
                    param.grad.data[rank:, :] = 0
            elif 'lora_B' in name and param.grad.dim() == 2:
                if rank < param.grad.shape[1]:
                    param.grad.data[:, rank:] = 0

    def local_train(self, model, train_loader, client_id, **kwargs):
        model = model.to(self.device)
        model.train()
        client_rank = self._get_client_rank(client_id)
        lr_factor = kwargs.get('lr_factor', 1.0)
        self._mask_to_rank(model, client_rank)
        _ref_a = {}
        _ref_b = {}
        _ref_all = {}
        if self.dirreg_lambda > 0 or self.fedprox_mu > 0:
            from src.utils.regularization import capture_lora_reference
            (_ref_a, _ref_b, _ref_all) = capture_lora_reference(model)
        trainable_params = self.get_trainable_params(model)
        optimizer = optim.AdamW(trainable_params.values(), lr=self.lr * lr_factor, weight_decay=self.weight_decay)
        total_loss = 0.0
        num_batches = 0
        _grad_norms = {}
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
                    self._mask_gradients(model, client_rank)
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    if num_batches == 0:
                        _grad_norms = self._log_grad_norms(model, client_id)
                    optimizer.step()
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
                    self._mask_gradients(model, client_rank)
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    if num_batches == 0:
                        _grad_norms = self._log_grad_norms(model, client_id)
                    optimizer.step()
                self._mask_to_rank(model, client_rank)
                total_loss += loss.item()
                num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)
        state_dict = {name: param.data.detach().clone().cpu() for (name, param) in trainable_params.items()}
        if self._method_logger:
            self._method_logger.log_client_train(client_id=client_id, round_num=self._current_round, loss=avg_loss, num_samples=len(train_loader.dataset), num_batches=num_batches, custom_metrics={'client_rank': float(client_rank), 'max_rank': float(self.max_rank)})
        return ClientUpdate(client_id=client_id, state_dict=state_dict, num_samples=len(train_loader.dataset), loss=avg_loss, metrics={'train_loss': avg_loss, 'client_rank': client_rank, **_grad_norms, **_reg_diag})

    def _log_grad_norms(self, model, client_id):
        (_ga, _gb, _gc) = (0.0, 0.0, 0.0)
        (_ac, _bc, _cc) = (0, 0, 0)
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
        result = {}
        if _ac > 0 and _bc > 0:
            _ga /= _ac
            _gb /= _bc
            result = {'grad_a_norm': _ga, 'grad_b_norm': _gb, 'grad_ratio_a_over_b': _ga / max(_gb, 1e-12)}
        avg_cls = _gc / _cc if _cc > 0 else 0.0
        if self._method_logger:
            self._method_logger.log_gradient_stats(round_num=self._current_round, client_id=client_id, grad_a_norm=_ga / max(_ac, 1), grad_b_norm=_gb / max(_bc, 1), grad_m_norm=0.0, grad_cls_norm=avg_cls)
        return result

    def aggregate(self, global_model, client_updates, round_idx=-1, **kwargs):
        if round_idx >= 0:
            self._current_round = round_idx
        if not client_updates:
            raise ValueError('No client updates provided')
        total_samples = sum((u.num_samples for u in client_updates))
        weights = [u.num_samples / total_samples for u in client_updates]
        client_ranks = []
        for u in client_updates:
            if u.metrics and 'client_rank' in u.metrics:
                client_ranks.append(int(u.metrics['client_rank']))
            else:
                client_ranks.append(self._get_client_rank(u.client_id))
        global_state = global_model.state_dict()
        for param_name in client_updates[0].state_dict.keys():
            params = [u.state_dict[param_name] for u in client_updates]
            is_lora_a = 'lora_A' in param_name and params[0].dim() == 2
            is_lora_b = 'lora_B' in param_name and params[0].dim() == 2
            if is_lora_a or is_lora_b:
                aggregated = torch.zeros_like(params[0])
                weight_sum = torch.zeros_like(params[0])
                for (p, w, rank) in zip(params, weights, client_ranks):
                    if is_lora_a:
                        mask = torch.zeros_like(p)
                        mask[:rank, :] = 1.0
                    else:
                        mask = torch.zeros_like(p)
                        mask[:, :rank] = 1.0
                    aggregated += w * p
                    weight_sum += w * mask
                nonzero = weight_sum > 0
                aggregated[nonzero] = aggregated[nonzero] / weight_sum[nonzero]
                global_state[param_name] = aggregated
            else:
                aggregated = torch.zeros_like(params[0])
                for (p, w) in zip(params, weights):
                    aggregated += w * p
                global_state[param_name] = aggregated
        global_model.load_state_dict(global_state, strict=False)
        similarity_metrics = self._compute_cross_client_similarity(client_updates)
        if self._method_logger:
            avg_metrics = {}
            all_keys = set()
            for u in client_updates:
                if u.metrics:
                    all_keys.update(u.metrics.keys())
            for key in all_keys:
                values = [u.metrics[key] for u in client_updates if u.metrics and key in u.metrics]
                if values:
                    avg_metrics[f'avg_{key}'] = sum(values) / len(values)
            avg_metrics.update(similarity_metrics)
            ranks = [float(u.metrics['client_rank']) for u in client_updates if u.metrics and 'client_rank' in u.metrics]
            if ranks:
                avg_metrics['round_avg_rank'] = sum(ranks) / len(ranks)
                avg_metrics['round_min_rank'] = min(ranks)
                avg_metrics['round_max_rank'] = max(ranks)
            self._method_logger.log_aggregation(round_num=self._current_round, num_clients=len(client_updates), total_samples=total_samples, aggregation_method='hetlora_weighted_avg', custom_metrics=avg_metrics)
            lr_factor = kwargs.get('lr_factor', 1.0)
            self._method_logger.log_aggregate_analysis(round_num=self._current_round, global_model=global_model, client_updates=client_updates, lr_factor=lr_factor, effective_lrs={'lr': self.lr * lr_factor}, has_magnitude=False)
        self._current_round = round_idx + 1 if round_idx >= 0 else self._current_round + 1
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

    def _compute_cross_client_similarity(self, client_updates):
        if len(client_updates) < 2:
            return {}
        client_a_params = []
        client_b_params = []
        for update in client_updates:
            a_dict = {}
            b_dict = {}
            for (name, param) in update.state_dict.items():
                if 'lora_A' in name:
                    a_dict[name] = param
                elif 'lora_B' in name:
                    b_dict[name] = param
            client_a_params.append(a_dict)
            client_b_params.append(b_dict)
        a_sims = []
        b_sims = []
        num_clients = len(client_updates)
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                for name in client_a_params[i]:
                    if name in client_a_params[j]:
                        a = client_a_params[i][name].flatten().float()
                        b = client_a_params[j][name].flatten().float()
                        if a.numel() > 0:
                            sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
                            a_sims.append(sim)
                for name in client_b_params[i]:
                    if name in client_b_params[j]:
                        a = client_b_params[i][name].flatten().float()
                        b = client_b_params[j][name].flatten().float()
                        if a.numel() > 0:
                            sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
                            b_sims.append(sim)
        result = {}
        if a_sims:
            result['a_similarity_mean'] = sum(a_sims) / len(a_sims)
        if b_sims:
            result['b_similarity_mean'] = sum(b_sims) / len(b_sims)
        result['num_client_pairs'] = num_clients * (num_clients - 1) // 2
        return result
