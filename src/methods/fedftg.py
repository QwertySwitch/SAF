import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings
from typing import Any
logger = logging.getLogger('fedfl')
try:
    from galore_torch import GaLoreAdamW
    GALORE_AVAILABLE = True
except ImportError:
    GALORE_AVAILABLE = False
    warnings.warn('galore-torch not available. Install with: pip install galore-torch', stacklevel=2)

@dataclass
class ClientUpdate:
    client_id: Any
    state_dict: Any
    num_samples: Any
    loss: Any
    metrics: Any = None

class FedFTG:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        from src.utils.config import get_method_params
        params = get_method_params(config)
        self.lr = float(params.get('lr', 3e-05))
        self.local_epochs = params.get('local_epochs', 2)
        self.weight_decay = float(params.get('weight_decay', 0.01))
        self.grad_clip = float(params.get('grad_clip', 1.0))
        self.beta1 = params.get('beta1', 0.9)
        self.beta2 = params.get('beta2', 0.98)
        self.warmup = params.get('warmup', 'linear')
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.galore_rank = params.get('galore_rank', 128)
        self.galore_update_proj_gap = params.get('galore_update_proj_gap', 200)
        self.galore_scale = params.get('galore_scale', 0.25)
        self.use_bf16 = params.get('use_bf16', True)
        if self.use_bf16 and (not torch.cuda.is_bf16_supported()):
            logger.warning('BF16 not supported, falling back to FP32')
            self.use_bf16 = False
        self._method_logger = None
        self._current_round = 0
        if not GALORE_AVAILABLE:
            raise ImportError('FedFTG requires galore-torch package. Install with: pip install galore-torch')

    def set_method_logger(self, method_logger):
        self._method_logger = method_logger

    def local_train(self, model, train_loader, client_id, **kwargs):
        model = model.to(self.device)
        model.train()
        lr_factor = kwargs.get('lr_factor', 1.0)
        effective_lr = self.lr * lr_factor
        galore_params = []
        regular_params = []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 2:
                galore_params.append(param)
            else:
                regular_params.append(param)
        param_groups = []
        if galore_params:
            param_groups.append({'params': galore_params, 'rank': self.galore_rank, 'update_proj_gap': self.galore_update_proj_gap, 'scale': self.galore_scale, 'proj_type': 'std'})
        if regular_params:
            param_groups.append({'params': regular_params})
        optimizer = GaLoreAdamW(param_groups, lr=effective_lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay, no_deprecation_warning=True)
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
                    optimizer.step()
                else:
                    outputs = model(**batch)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)
        state_dict = {}
        for (name, param) in model.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data.detach().clone().cpu()
        return ClientUpdate(client_id=client_id, state_dict=state_dict, num_samples=len(train_loader.dataset), loss=avg_loss, metrics={'train_loss': avg_loss})

    def aggregate(self, global_model, client_updates, round_idx=-1, **kwargs):
        if not client_updates:
            raise ValueError('No client updates provided')
        if round_idx >= 0:
            self._current_round = round_idx
        total_samples = sum((update.num_samples for update in client_updates))
        if total_samples == 0:
            raise ValueError('Total samples is zero - all clients have empty data')
        aggregated_state = {}
        param_names = client_updates[0].state_dict.keys()
        for param_name in param_names:
            aggregated_param = torch.zeros_like(client_updates[0].state_dict[param_name])
            for update in client_updates:
                weight = update.num_samples / total_samples
                aggregated_param += weight * update.state_dict[param_name]
            aggregated_state[param_name] = aggregated_param
        global_model.load_state_dict(aggregated_state, strict=False)
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
            self._method_logger.log_aggregation(round_num=self._current_round, num_clients=len(client_updates), total_samples=total_samples, aggregation_method='fedftg', custom_metrics=avg_metrics)
            lr_factor = kwargs.get('lr_factor', 1.0)
            self._method_logger.log_aggregate_analysis(round_num=self._current_round, global_model=global_model, client_updates=client_updates, lr_factor=lr_factor, effective_lrs={'lr': self.lr * lr_factor}, has_magnitude=False)
        return global_model

    def get_trainable_params(self, model):
        return {name: param for (name, param) in model.named_parameters() if param.requires_grad}

    def get_effective_trainable_params(self, model):
        effective_params = 0
        rank = self.galore_rank
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 2:
                (m, n) = (param.shape[0], param.shape[1])
                effective_rank = min(rank, m, n)
                effective_params += effective_rank * (m + n)
            else:
                effective_params += param.numel()
        return effective_params

    def setup_model(self, model):
        if hasattr(model, 'model'):
            hf_model = model.model
        else:
            hf_model = model
        if hasattr(hf_model, 'roberta'):
            for param in hf_model.roberta.embeddings.parameters():
                param.requires_grad = False
        elif hasattr(hf_model, 'bert'):
            for param in hf_model.bert.embeddings.parameters():
                param.requires_grad = False
        return model
