import logging
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional
from dataclasses import dataclass
from typing import Any
logger = logging.getLogger('fedfl')

@dataclass
class ClientUpdate:
    client_id: Any
    state_dict: Any
    num_samples: Any
    loss: Any
    metrics: Any = None

class FedAvg:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        from src.utils.config import get_method_params
        params = get_method_params(config)
        self.lr = float(params.get('lr', 3e-05))
        self.local_epochs = params.get('local_epochs', 3)
        self.weight_decay = params.get('weight_decay', 0.01)
        self.warmup = params.get('warmup', 'linear')
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.grad_clip = float(params.get('grad_clip', 1.0))
        self.use_bf16 = params.get('use_bf16', True)
        if self.use_bf16 and (not torch.cuda.is_bf16_supported()):
            logger.warning('BF16 not supported, falling back to FP32')
            self.use_bf16 = False
        self._method_logger = None
        self._current_round = 0

    def set_method_logger(self, method_logger):
        self._method_logger = method_logger

    def local_train(self, model, train_loader, client_id, **kwargs):
        model = model.to(self.device)
        model.train()
        lr_factor = kwargs.get('lr_factor', 1.0)
        effective_lr = self.lr * lr_factor
        optimizer = optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=self.weight_decay)
        total_loss = 0.0
        num_batches = 0
        for _ in range(self.local_epochs):
            epoch_loss = 0.0
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
                epoch_loss += loss.item()
                num_batches += 1
            total_loss += epoch_loss
        avg_loss = total_loss / max(num_batches, 1)
        state_dict_cpu = {k: v.detach().clone().cpu() for (k, v) in model.state_dict().items()}
        return ClientUpdate(client_id=client_id, state_dict=state_dict_cpu, num_samples=len(train_loader.dataset), loss=avg_loss, metrics={'train_loss': avg_loss})

    def aggregate(self, global_model, client_updates, **kwargs):
        from src.utils.aggregation import weighted_average_state_dicts, load_aggregated_state
        if not client_updates:
            raise ValueError('No client updates provided')
        aggregated_state = weighted_average_state_dicts(client_updates)
        result = load_aggregated_state(global_model, aggregated_state, strict=False)
        round_idx = kwargs.get('round_idx', -1)
        if round_idx >= 0:
            self._current_round = round_idx
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
            total_samples = sum((u.num_samples for u in client_updates))
            self._method_logger.log_aggregation(round_num=self._current_round, num_clients=len(client_updates), total_samples=total_samples, aggregation_method='fedavg', custom_metrics=avg_metrics)
            lr_factor = kwargs.get('lr_factor', 1.0)
            self._method_logger.log_aggregate_analysis(round_num=self._current_round, global_model=result, client_updates=client_updates, lr_factor=lr_factor, effective_lrs={'lr': self.lr * lr_factor}, has_magnitude=False)
        return result

    def get_trainable_params(self, model):
        return {name: param for (name, param) in model.named_parameters() if param.requires_grad}

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
