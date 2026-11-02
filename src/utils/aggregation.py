from typing import Dict, List, Optional, Protocol, Tuple
import logging
import statistics as _statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
logger = logging.getLogger('fedfl')

class HasNumSamples(Protocol):
    num_samples: Any
    state_dict: Any

def weighted_average_state_dicts(client_updates, param_names=None):
    if not client_updates:
        raise ValueError('No client updates provided')
    total_samples = sum((update.num_samples for update in client_updates))
    if total_samples == 0:
        raise ValueError('Total samples is zero - all clients have empty data')
    if param_names is None:
        param_names = list(client_updates[0].state_dict.keys())
    aggregated_state = {}
    for param_name in param_names:
        ref_param = client_updates[0].state_dict[param_name]
        aggregated_param = torch.zeros_like(ref_param, device=ref_param.device)
        for update in client_updates:
            weight = update.num_samples / total_samples
            aggregated_param += weight * update.state_dict[param_name]
        aggregated_state[param_name] = aggregated_param
    return aggregated_state

def load_aggregated_state(model, aggregated_state, strict=False):
    model.load_state_dict(aggregated_state, strict=strict)
    return model

def aggregate_into_model(model, client_updates, strict=False):
    aggregated_state = weighted_average_state_dicts(client_updates)
    return load_aggregated_state(model, aggregated_state, strict=strict)

def geometric_median(tensors, weights=None, max_iter=100, tol=1e-06):
    if len(tensors) == 1:
        return tensors[0].clone()
    n = len(tensors)
    if weights is None:
        weights = [1.0 / n] * n
    flat = [t.float().flatten() for t in tensors]
    estimate = torch.zeros_like(flat[0])
    for (t, w) in zip(flat, weights):
        estimate += w * t
    for _ in range(max_iter):
        distances = [torch.norm(t - estimate) for t in flat]
        inv_dists = []
        for (d, w) in zip(distances, weights):
            if d.item() < 1e-10:
                inv_dists.append(torch.tensor(0.0, device=estimate.device))
            else:
                inv_dists.append(w / d)
        total_inv = sum(inv_dists)
        if total_inv.item() < 1e-10:
            break
        new_estimate = torch.zeros_like(estimate)
        for (t, inv_d) in zip(flat, inv_dists):
            new_estimate += inv_d * t
        new_estimate /= total_inv
        shift = torch.norm(new_estimate - estimate) / (torch.norm(estimate) + 1e-10)
        estimate = new_estimate
        if shift.item() < tol:
            break
    return estimate.reshape(tensors[0].shape).to(tensors[0].dtype)

def trimmed_mean(tensors, weights=None, trim_ratio=0.1):
    if len(tensors) <= 2:
        result = torch.zeros_like(tensors[0], dtype=torch.float32)
        n = len(tensors)
        w = [1.0 / n] * n if weights is None else weights
        for (t, wi) in zip(tensors, w):
            result += wi * t.float()
        return result.to(tensors[0].dtype)
    n = len(tensors)
    if weights is None:
        weights = [1.0 / n] * n
    k = max(1, int(n * trim_ratio))
    if 2 * k >= n:
        k = max(1, (n - 1) // 2)
    stacked = torch.stack([t.float().flatten() for t in tensors], dim=0)
    weight_tensor = torch.tensor(weights, device=stacked.device, dtype=torch.float32)
    (sorted_vals, sorted_idx) = torch.sort(stacked, dim=0)
    trimmed_vals = sorted_vals[k:n - k]
    trimmed_idx = sorted_idx[k:n - k]
    trimmed_weights = weight_tensor[trimmed_idx]
    weight_sums = trimmed_weights.sum(dim=0, keepdim=True).clamp(min=1e-10)
    trimmed_weights = trimmed_weights / weight_sums
    result = (trimmed_vals * trimmed_weights).sum(dim=0)
    return result.reshape(tensors[0].shape).to(tensors[0].dtype)

def compute_aggregation_weights(client_updates):
    if not client_updates:
        return []
    total_samples = sum((update.num_samples for update in client_updates))
    if total_samples == 0:
        n = len(client_updates)
        return [1.0 / n] * n
    return [update.num_samples / total_samples for update in client_updates]

def direction_aware_aggregate(client_updates, global_state, daa_floor=0.3, param_names=None, round_idx=-1, log_every=5):
    if not client_updates:
        raise ValueError('No client updates provided')
    total_samples = sum((u.num_samples for u in client_updates))
    if total_samples == 0:
        raise ValueError('Total samples is zero')
    num_clients = len(client_updates)
    base_weights = [u.num_samples / total_samples for u in client_updates]
    if param_names is None:
        param_names = list(client_updates[0].state_dict.keys())
    aggregated_state = {}
    daa_metrics = {}
    all_disagreements = []
    all_activations = []
    all_weight_stds = []
    for param_name in param_names:
        params = [u.state_dict[param_name] for u in client_updates]
        if param_name not in global_state:
            aggregated = torch.zeros_like(params[0])
            for (p, w) in zip(params, base_weights):
                aggregated += w * p
            aggregated_state[param_name] = aggregated
            continue
        global_param = global_state[param_name]
        client_device = params[0].device
        if global_param.device != client_device:
            global_param = global_param.to(client_device)
        deltas = [p.float() - global_param.float() for p in params]
        flat_deltas = [d.flatten() for d in deltas]
        delta_norms = [fd.norm().item() for fd in flat_deltas]
        if max(delta_norms) < 1e-10:
            aggregated = torch.zeros_like(params[0])
            for (p, w) in zip(params, base_weights):
                aggregated += w * p
            aggregated_state[param_name] = aggregated
            continue
        pairwise_sims = []
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                if delta_norms[i] > 1e-10 and delta_norms[j] > 1e-10:
                    cos = F.cosine_similarity(flat_deltas[i].unsqueeze(0), flat_deltas[j].unsqueeze(0)).item()
                    pairwise_sims.append(cos)
        if not pairwise_sims:
            aggregated = torch.zeros_like(params[0])
            for (p, w) in zip(params, base_weights):
                aggregated += w * p
            aggregated_state[param_name] = aggregated
            continue
        mean_pairwise_cos = sum(pairwise_sims) / len(pairwise_sims)
        disagreement = max(0.0, min(1.0, 1.0 - mean_pairwise_cos))
        delta_mean = torch.zeros_like(flat_deltas[0])
        for fd in flat_deltas:
            delta_mean += fd / num_clients
        delta_mean_norm = delta_mean.norm().item()
        adjusted_weights = []
        for k in range(num_clients):
            if delta_mean_norm > 1e-10 and delta_norms[k] > 1e-10:
                s_k = F.cosine_similarity(flat_deltas[k].unsqueeze(0), delta_mean.unsqueeze(0)).item()
                alignment_k = daa_floor + (1.0 - daa_floor) * max(0.0, s_k)
            else:
                alignment_k = 1.0
            w_k = base_weights[k] * (1.0 - disagreement + disagreement * alignment_k)
            adjusted_weights.append(w_k)
        w_sum = sum(adjusted_weights)
        if w_sum > 1e-10:
            adjusted_weights = [w / w_sum for w in adjusted_weights]
        else:
            adjusted_weights = list(base_weights)
        aggregated = torch.zeros_like(params[0])
        for (p, w) in zip(params, adjusted_weights):
            aggregated += w * p
        aggregated_state[param_name] = aggregated
        all_disagreements.append(disagreement)
        all_activations.append(disagreement)
        if len(adjusted_weights) > 1:
            all_weight_stds.append(_statistics.stdev(adjusted_weights))
    if all_disagreements:
        daa_metrics['daa_disagreement_mean'] = sum(all_disagreements) / len(all_disagreements)
        daa_metrics['daa_disagreement_max'] = max(all_disagreements)
        daa_metrics['daa_activation_mean'] = sum(all_activations) / len(all_activations)
    if all_weight_stds:
        daa_metrics['daa_weight_std_mean'] = sum(all_weight_stds) / len(all_weight_stds)
    return (aggregated_state, daa_metrics)
