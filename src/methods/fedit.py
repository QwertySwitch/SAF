import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model, TaskType
import logging
from typing import Any
logger = logging.getLogger('fedfl')

@dataclass
class ClientUpdate:
    client_id: Any
    state_dict: Any
    num_samples: Any
    loss: Any
    metrics: Any = None

@dataclass
class LoRACrossClientSimilarity:
    a_similarity_mean: Any = 0.0
    a_similarity_std: Any = 0.0
    b_similarity_mean: Any = 0.0
    b_similarity_std: Any = 0.0
    pseudo_m_similarity_mean: Any = 0.0
    pseudo_m_similarity_std: Any = 0.0
    a_similarity_per_layer: Any = field(default_factory=dict)
    b_similarity_per_layer: Any = field(default_factory=dict)
    pseudo_m_similarity_per_layer: Any = field(default_factory=dict)
    a_variance_mean: Any = 0.0
    b_variance_mean: Any = 0.0
    pseudo_m_variance_mean: Any = 0.0
    num_client_pairs: Any = 0

    def to_dict(self):
        result = {'m_similarity_mean': self.pseudo_m_similarity_mean, 'm_similarity_std': self.pseudo_m_similarity_std, 'a_similarity_mean': self.a_similarity_mean, 'a_similarity_std': self.a_similarity_std, 'b_similarity_mean': self.b_similarity_mean, 'b_similarity_std': self.b_similarity_std, 'm_variance_mean': self.pseudo_m_variance_mean, 'a_variance_mean': self.a_variance_mean, 'b_variance_mean': self.b_variance_mean, 'num_client_pairs': float(self.num_client_pairs)}
        for (layer, sim) in self.a_similarity_per_layer.items():
            result[f'a_sim_{layer}'] = sim
        for (layer, sim) in self.b_similarity_per_layer.items():
            result[f'b_sim_{layer}'] = sim
        for (layer, sim) in self.pseudo_m_similarity_per_layer.items():
            result[f'm_sim_{layer}'] = sim
        return result

    def get_ordering(self):
        values = [('A', self.a_similarity_mean), ('B', self.b_similarity_mean)]
        if self.pseudo_m_similarity_mean > 0:
            values.append(('m', self.pseudo_m_similarity_mean))
        values.sort(key=lambda x: x[1], reverse=True)
        return '>'.join((v[0] for v in values))

class FedIT:

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
        self.weight_decay = float(params.get('weight_decay', 0.01))
        self.grad_clip = float(params.get('grad_clip', 1.0))
        self.warmup = params.get('warmup', 'linear')
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.use_bf16 = params.get('use_bf16', True)
        if self.use_bf16 and (not torch.cuda.is_bf16_supported()):
            logger.warning('BF16 not supported on this GPU, falling back to FP32')
            self.use_bf16 = False
        self._last_similarity = None
        self._method_logger = None
        self._current_round = 0
        self.dirreg_lambda = float(config.get('training', {}).get('dirreg_lambda', 0.0))
        self.dirreg_detach_b = bool(config.get('training', {}).get('dirreg_detach_b', False))
        self.fedprox_mu = float(config.get('training', {}).get('fedprox_mu', 0.0))
        if self.dirreg_lambda > 0:
            mode = 'A-only (detach_b)' if self.dirreg_detach_b else 'A+B'
            logger.info(f'FEDIT: DirReg enabled (lambda={self.dirreg_lambda}, mode={mode})')
        if self.fedprox_mu > 0:
            logger.info(f'FEDIT: FedProx enabled (mu={self.fedprox_mu})')
        self.track_pseudo_m = params.get('track_pseudo_m', False)
        if self.track_pseudo_m:
            logger.info('FEDIT: Tracking pseudo-magnitude (||BA||_col) for analysis')
        self.use_dna = params.get('use_dna', False)
        if self.use_dna:
            logger.info('FEDIT+DNA: Direction-Normalized Aggregation enabled for B matrices. This is a specificity test — DNA should NOT help LoRA.')

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
        _grad_norms = {}
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
                    if num_batches == 0:
                        _grad_a_norm = 0.0
                        _grad_b_norm = 0.0
                        _grad_cls_norm = 0.0
                        _grad_a_count = 0
                        _grad_b_count = 0
                        _grad_cls_count = 0
                        for (name, param) in model.named_parameters():
                            if param.grad is not None:
                                g = param.grad.float().norm().item()
                                if 'lora_A' in name:
                                    _grad_a_norm += g
                                    _grad_a_count += 1
                                elif 'lora_B' in name:
                                    _grad_b_norm += g
                                    _grad_b_count += 1
                                elif 'classifier' in name or 'qa_outputs' in name or name.endswith('.head.weight') or name.endswith('.head.bias'):
                                    _grad_cls_norm += g
                                    _grad_cls_count += 1
                        avg_a = _grad_a_norm / _grad_a_count if _grad_a_count > 0 else 0.0
                        avg_b = _grad_b_norm / _grad_b_count if _grad_b_count > 0 else 0.0
                        avg_cls = _grad_cls_norm / _grad_cls_count if _grad_cls_count > 0 else 0.0
                        if _grad_a_count > 0 and _grad_b_count > 0:
                            _grad_norms = {'grad_a_norm': avg_a, 'grad_b_norm': avg_b, 'grad_ratio_a_over_b': avg_a / max(avg_b, 1e-12)}
                        if self._method_logger:
                            self._method_logger.log_gradient_stats(round_num=self._current_round, client_id=client_id, grad_a_norm=avg_a, grad_b_norm=avg_b, grad_m_norm=0.0, grad_cls_norm=avg_cls)
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
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
                    if num_batches == 0:
                        _grad_a_norm = 0.0
                        _grad_b_norm = 0.0
                        _grad_cls_norm = 0.0
                        _grad_a_count = 0
                        _grad_b_count = 0
                        _grad_cls_count = 0
                        for (name, param) in model.named_parameters():
                            if param.grad is not None:
                                g = param.grad.float().norm().item()
                                if 'lora_A' in name:
                                    _grad_a_norm += g
                                    _grad_a_count += 1
                                elif 'lora_B' in name:
                                    _grad_b_norm += g
                                    _grad_b_count += 1
                                elif 'classifier' in name or 'qa_outputs' in name or name.endswith('.head.weight') or name.endswith('.head.bias'):
                                    _grad_cls_norm += g
                                    _grad_cls_count += 1
                        avg_a = _grad_a_norm / _grad_a_count if _grad_a_count > 0 else 0.0
                        avg_b = _grad_b_norm / _grad_b_count if _grad_b_count > 0 else 0.0
                        avg_cls = _grad_cls_norm / _grad_cls_count if _grad_cls_count > 0 else 0.0
                        if _grad_a_count > 0 and _grad_b_count > 0:
                            _grad_norms = {'grad_a_norm': avg_a, 'grad_b_norm': avg_b, 'grad_ratio_a_over_b': avg_a / max(avg_b, 1e-12)}
                        if self._method_logger:
                            self._method_logger.log_gradient_stats(round_num=self._current_round, client_id=client_id, grad_a_norm=avg_a, grad_b_norm=avg_b, grad_m_norm=0.0, grad_cls_norm=avg_cls)
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)
        lora_state_dict = {name: param.data.detach().clone().cpu() for (name, param) in trainable_params.items()}
        return ClientUpdate(client_id=client_id, state_dict=lora_state_dict, num_samples=len(train_loader.dataset), loss=avg_loss, metrics={'train_loss': avg_loss, **_grad_norms, **_reg_diag})

    def aggregate(self, global_model, client_updates, round_idx=-1, **kwargs):
        if not client_updates:
            raise ValueError('No client updates provided')
        if round_idx >= 0:
            self._current_round = round_idx
        total_samples = sum((update.num_samples for update in client_updates))
        if total_samples == 0:
            raise ValueError('Total samples is zero - all clients have empty data')
        similarity_metrics = self.compute_cross_client_similarity(client_updates)
        self._last_similarity = similarity_metrics
        if similarity_metrics.num_client_pairs > 0:
            ordering = similarity_metrics.get_ordering()
            if self.track_pseudo_m and similarity_metrics.pseudo_m_similarity_mean > 0:
                logger.info(f'[R{self._current_round}] [FedIT Similarity] A={similarity_metrics.a_similarity_mean:.4f}, B={similarity_metrics.b_similarity_mean:.4f}, pseudo_m={similarity_metrics.pseudo_m_similarity_mean:.4f}, ({ordering})')
            else:
                logger.info(f'[R{self._current_round}] [FedIT Similarity] A={similarity_metrics.a_similarity_mean:.4f}, B={similarity_metrics.b_similarity_mean:.4f}, ({ordering}), Δ(A-B)={similarity_metrics.a_similarity_mean - similarity_metrics.b_similarity_mean:.4f}')
            b_norms_per_client = []
            for update in client_updates:
                client_b_norm = 0.0
                b_count = 0
                for (name, param) in update.state_dict.items():
                    if 'lora_B' in name or 'lora_b' in name:
                        client_b_norm += torch.norm(param.float(), p='fro').item()
                        b_count += 1
                if b_count > 0:
                    b_norms_per_client.append(client_b_norm / b_count)
            if b_norms_per_client:
                b_mean = sum(b_norms_per_client) / len(b_norms_per_client)
                b_std = (sum(((x - b_mean) ** 2 for x in b_norms_per_client)) / len(b_norms_per_client)) ** 0.5
                b_cv = b_std / (b_mean + 1e-08)
                logger.info(f'  → B-norm stats: mean={b_mean:.4f}, std={b_std:.4f}, CV={b_cv:.4f}')
                if self._method_logger:
                    self._method_logger.log_custom('b_norm_stats', round_num=self._current_round, b_norm_mean=b_mean, b_norm_std=b_std, b_norm_cv=b_cv, b_norm_min=min(b_norms_per_client), b_norm_max=max(b_norms_per_client), num_clients=len(b_norms_per_client))
        weights = [u.num_samples / total_samples for u in client_updates]
        global_state = global_model.state_dict()
        param_names = client_updates[0].state_dict.keys()
        for param_name in param_names:
            params = [u.state_dict[param_name] for u in client_updates]
            if self.use_dna and ('lora_B' in param_name or 'lora_b' in param_name):
                aggregated_param = self._aggregate_b_dna(params, weights, param_name)
            else:
                aggregated_param = torch.zeros_like(params[0])
                for (param, weight) in zip(params, weights):
                    aggregated_param += weight * param
            global_state[param_name] = aggregated_param
        global_model.load_state_dict(global_state, strict=False)
        if hasattr(self, '_method_logger') and self._method_logger is not None:
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
            avg_metrics.update(similarity_metrics.to_dict())
            agg_method = 'fedavg+dna' if self.use_dna else 'fedavg'
            self._method_logger.log_aggregation(round_num=self._current_round, num_clients=len(client_updates), total_samples=total_samples, aggregation_method=agg_method, custom_metrics=avg_metrics)
            lr_factor = kwargs.get('lr_factor', 1.0)
            self._method_logger.log_aggregate_analysis(round_num=self._current_round, global_model=global_model, client_updates=client_updates, lr_factor=lr_factor, effective_lrs={'lr_a': self.lr_a * lr_factor, 'lr_b': self.lr_b * lr_factor, 'lr_cls': self.lr_cls * lr_factor}, similarity_metrics=similarity_metrics, has_magnitude=False)
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

    def _aggregate_b_dna(self, params, weights, param_name=''):
        eps = 1e-08
        directions = []
        magnitudes = []
        for param in params:
            norm = torch.norm(param.float(), p='fro')
            magnitudes.append(norm)
            directions.append(param.float() / (norm + eps))
        dir_agg = torch.zeros_like(params[0], dtype=torch.float32)
        for (direction, weight) in zip(directions, weights):
            dir_agg += weight * direction
        dir_norm = torch.norm(dir_agg, p='fro')
        dir_agg = dir_agg / (dir_norm + eps)
        mag_agg = torch.tensor(0.0, device=params[0].device)
        for (weight, mag) in zip(weights, magnitudes):
            mag_agg = mag_agg + weight * mag
        aggregated = mag_agg * dir_agg
        if self._current_round % 5 == 0 or self._current_round <= 5:
            client_norms = [m.item() for m in magnitudes]
            norm_mean = sum(client_norms) / len(client_norms)
            norm_std = (sum(((x - norm_mean) ** 2 for x in client_norms)) / len(client_norms)) ** 0.5
            norm_cv = norm_std / (norm_mean + eps)
            std_agg = torch.zeros_like(params[0], dtype=torch.float32)
            for (param, weight) in zip(params, weights):
                std_agg += weight * param.float()
            std_norm = torch.norm(std_agg, p='fro').item()
            dna_norm = torch.norm(aggregated, p='fro').item()
            short_name = param_name.split('.')[-3] if len(param_name.split('.')) > 3 else param_name[-30:]
            logger.info(f'[R{self._current_round}] [FEDIT+DNA] {short_name}: B_norms=[{norm_mean:.4f}±{norm_std:.4f}] CV={norm_cv:.4f}, agg_norm(std={std_norm:.4f}, dna={dna_norm:.4f})')
            if self._method_logger:
                self._method_logger.log_custom('fedit_dna_diagnostics', round_num=self._current_round, param_name=short_name, b_norm_mean=norm_mean, b_norm_std=norm_std, b_norm_cv=norm_cv, agg_norm_std=std_norm, agg_norm_dna=dna_norm)
        return aggregated.to(params[0].dtype)

    def compute_cross_client_similarity(self, client_updates):
        if len(client_updates) < 2:
            return LoRACrossClientSimilarity()
        client_a_params = []
        client_b_params = []
        client_pseudo_m = []
        for update in client_updates:
            a_dict = {}
            b_dict = {}
            for (name, param) in update.state_dict.items():
                if 'lora_A' in name or 'lora_a' in name:
                    a_dict[name] = param
                elif 'lora_B' in name or 'lora_b' in name:
                    b_dict[name] = param
            client_a_params.append(a_dict)
            client_b_params.append(b_dict)
            if self.track_pseudo_m:
                pseudo_m_dict = self._compute_pseudo_m(a_dict, b_dict)
                client_pseudo_m.append(pseudo_m_dict)
        a_similarities = []
        b_similarities = []
        a_layer_sims = {}
        b_layer_sims = {}
        num_clients = len(client_updates)
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                a_sims_pair = self._compute_param_similarity(client_a_params[i], client_a_params[j])
                for (layer, sim) in a_sims_pair.items():
                    a_similarities.append(sim)
                    if layer not in a_layer_sims:
                        a_layer_sims[layer] = []
                    a_layer_sims[layer].append(sim)
                b_sims_pair = self._compute_param_similarity(client_b_params[i], client_b_params[j])
                for (layer, sim) in b_sims_pair.items():
                    b_similarities.append(sim)
                    if layer not in b_layer_sims:
                        b_layer_sims[layer] = []
                    b_layer_sims[layer].append(sim)
        pseudo_m_similarities = []
        pseudo_m_layer_sims = {}
        pseudo_m_variances = []
        if self.track_pseudo_m and client_pseudo_m:
            for i in range(num_clients):
                for j in range(i + 1, num_clients):
                    m_sims_pair = self._compute_param_similarity(client_pseudo_m[i], client_pseudo_m[j])
                    for (layer, sim) in m_sims_pair.items():
                        pseudo_m_similarities.append(sim)
                        if layer not in pseudo_m_layer_sims:
                            pseudo_m_layer_sims[layer] = []
                        pseudo_m_layer_sims[layer].append(sim)
            pseudo_m_variances = self._compute_cross_client_variance(client_pseudo_m)
        a_variances = self._compute_cross_client_variance(client_a_params)
        b_variances = self._compute_cross_client_variance(client_b_params)
        result = LoRACrossClientSimilarity(a_similarity_mean=sum(a_similarities) / len(a_similarities) if a_similarities else 0.0, a_similarity_std=self._std(a_similarities) if a_similarities else 0.0, b_similarity_mean=sum(b_similarities) / len(b_similarities) if b_similarities else 0.0, b_similarity_std=self._std(b_similarities) if b_similarities else 0.0, pseudo_m_similarity_mean=sum(pseudo_m_similarities) / len(pseudo_m_similarities) if pseudo_m_similarities else 0.0, pseudo_m_similarity_std=self._std(pseudo_m_similarities) if pseudo_m_similarities else 0.0, a_similarity_per_layer={k: sum(v) / len(v) for (k, v) in a_layer_sims.items()}, b_similarity_per_layer={k: sum(v) / len(v) for (k, v) in b_layer_sims.items()}, pseudo_m_similarity_per_layer={k: sum(v) / len(v) for (k, v) in pseudo_m_layer_sims.items()}, a_variance_mean=sum(a_variances) / len(a_variances) if a_variances else 0.0, b_variance_mean=sum(b_variances) / len(b_variances) if b_variances else 0.0, pseudo_m_variance_mean=sum(pseudo_m_variances) / len(pseudo_m_variances) if pseudo_m_variances else 0.0, num_client_pairs=num_clients * (num_clients - 1) // 2)
        return result

    def _compute_param_similarity(self, params_a, params_b):
        similarities = {}
        for name in params_a:
            if name not in params_b:
                continue
            a = params_a[name].flatten().float()
            b = params_b[name].flatten().float()
            if a.numel() == 0 or b.numel() == 0:
                continue
            cos_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            layer_name = self._extract_layer_name(name)
            similarities[layer_name] = cos_sim
        return similarities

    def _extract_layer_name(self, param_name):
        parts = param_name.split('.')
        layer_num = None
        module_type = None
        for (i, part) in enumerate(parts):
            if part == 'layer' and i + 1 < len(parts):
                layer_num = parts[i + 1]
            if part in ['query', 'key', 'value', 'dense', 'output']:
                module_type = part
        if layer_num is not None and module_type is not None:
            return f'layer_{layer_num}_{module_type}'
        return param_name.split('.')[-3] if len(parts) > 3 else param_name

    def _compute_cross_client_variance(self, client_params):
        if len(client_params) < 2:
            return []
        variances = []
        param_names = list(client_params[0].keys())
        for name in param_names:
            params_list = []
            for client_param in client_params:
                if name in client_param:
                    params_list.append(client_param[name].flatten().float())
            if len(params_list) < 2:
                continue
            stacked = torch.stack(params_list)
            var = stacked.var(dim=0).mean().item()
            variances.append(var)
        return variances

    def _std(self, values):
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum(((x - mean) ** 2 for x in values)) / len(values)
        return variance ** 0.5

    def _compute_pseudo_m(self, a_dict, b_dict):
        pseudo_m = {}
        for (a_name, a_param) in a_dict.items():
            b_name = a_name.replace('lora_A', 'lora_B').replace('lora_a', 'lora_b')
            if b_name not in b_dict:
                continue
            b_param = b_dict[b_name]
            try:
                a = a_param.float()
                b = b_param.float()
                if a.dim() > 2:
                    a = a.squeeze()
                if b.dim() > 2:
                    b = b.squeeze()
                if a.dim() != 2 or b.dim() != 2:
                    continue
                ba = torch.mm(b, a)
                col_norms = torch.norm(ba, dim=0)
                layer_name = self._extract_layer_name(a_name)
                pseudo_m[layer_name] = col_norms
            except Exception as e:
                logger.debug(f'Could not compute pseudo_m for {a_name}: {e}')
                continue
        return pseudo_m

    def get_similarity_dict(self):
        if hasattr(self, '_last_similarity') and self._last_similarity is not None:
            return self._last_similarity.to_dict()
        return {}

    def set_method_logger(self, method_logger):
        self._method_logger = method_logger
