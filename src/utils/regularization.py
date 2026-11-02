import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import logging
logger = logging.getLogger('fedfl')
_dirreg_zero_warned = [False]

def extract_lora_ab_pairs(model):
    a_params = {}
    b_params = {}
    for (name, param) in model.named_parameters():
        if 'lora_A' in name and param.requires_grad:
            key = name.replace('lora_A', 'LORA').replace('lora_B', 'LORA')
            a_params[key] = (name, param)
        elif 'lora_B' in name and param.requires_grad:
            key = name.replace('lora_A', 'LORA').replace('lora_B', 'LORA')
            b_params[key] = (name, param)
    pairs = []
    for key in a_params:
        if key in b_params:
            (_, a_param) = a_params[key]
            (_, b_param) = b_params[key]
            layer_name = key.replace('.LORA.default.weight', '')
            pairs.append((layer_name, a_param, b_param))
    return pairs

def compute_dirreg_loss(model, ref_a, ref_b, eps=1e-08, detach_b=False):
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    cos_sims = []
    layer_cos_sims = {}
    n_layers = 0
    for (name, param) in model.named_parameters():
        if 'lora_A' not in name or not param.requires_grad:
            continue
        b_name = name.replace('lora_A', 'lora_B')
        if name not in ref_a or b_name not in ref_b:
            continue
        a_local = param
        a_ref = ref_a[name]
        b_local = None
        b_ref = ref_b[b_name]
        for (n2, p2) in model.named_parameters():
            if n2 == b_name:
                b_local = p2
                break
        if b_local is None:
            continue
        if detach_b:
            b_local = b_local.detach()
        BtB_ref = b_local.float().t() @ b_ref.float()
        AAt_ref = a_local.float() @ a_ref.float().t()
        inner = (BtB_ref * AAt_ref).sum()
        BtB_local = b_local.float().t() @ b_local.float()
        AAt_local = a_local.float() @ a_local.float().t()
        norm_local_sq = (BtB_local * AAt_local).sum()
        BtB_r = b_ref.float().t() @ b_ref.float()
        AAt_r = a_ref.float() @ a_ref.float().t()
        norm_ref_sq = (BtB_r * AAt_r).sum()
        norm_local = norm_local_sq.clamp(min=eps * eps).sqrt()
        norm_ref = norm_ref_sq.clamp(min=eps * eps).sqrt()
        if norm_ref.item() < eps or norm_local.item() < eps:
            continue
        cos_sim = inner / (norm_local * norm_ref + eps)
        cos_sim = cos_sim.clamp(-1.0, 1.0)
        layer_loss = 1.0 - cos_sim
        total_loss = total_loss + layer_loss
        cs_val = cos_sim.item()
        cos_sims.append(cs_val)
        n_layers += 1
        short = name.replace('base_model.model.roberta.encoder.', '').replace('.lora_A.default.weight', '')
        layer_cos_sims[f'dirreg_cs_{short}'] = cs_val
    if n_layers == 0:
        if not _dirreg_zero_warned[0]:
            logger.warning('DirReg: all layers skipped (near-zero norms). Loss = 0. (Suppressing further repeats.)')
            _dirreg_zero_warned[0] = True
    diagnostics = {'dirreg_n_layers': n_layers, 'dirreg_avg_cos_sim': sum(cos_sims) / max(len(cos_sims), 1), 'dirreg_min_cos_sim': min(cos_sims) if cos_sims else 0.0, 'dirreg_max_cos_sim': max(cos_sims) if cos_sims else 0.0, 'dirreg_loss': total_loss.item()}
    diagnostics.update(layer_cos_sims)
    return (total_loss, diagnostics)

def compute_fedprox_loss(model, ref_params):
    prox_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n_params = 0
    total_dist_sq = 0.0
    for (name, param) in model.named_parameters():
        if name in ref_params and param.requires_grad:
            diff = param.float() - ref_params[name].float()
            layer_loss = (diff * diff).sum()
            prox_loss = prox_loss + layer_loss
            total_dist_sq += layer_loss.item()
            n_params += 1
    prox_loss = 0.5 * prox_loss
    diagnostics = {'fedprox_n_params': n_params, 'fedprox_dist_sq': total_dist_sq, 'fedprox_loss': prox_loss.item()}
    return (prox_loss, diagnostics)

def capture_lora_reference(model):
    ref_a = {}
    ref_b = {}
    ref_all = {}
    for (name, param) in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_A' in name:
            ref_a[name] = param.clone().detach()
            ref_all[name] = ref_a[name]
        elif 'lora_B' in name:
            ref_b[name] = param.clone().detach()
            ref_all[name] = ref_b[name]
        elif 'lora_magnitude_vector' in name:
            ref_all[name] = param.clone().detach()
        elif 'classifier' in name:
            ref_all[name] = param.clone().detach()
    return (ref_a, ref_b, ref_all)
