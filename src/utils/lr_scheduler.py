import math
from typing import Optional

class FederatedLRScheduler:

    def __init__(self, total_rounds, warmup_ratio=0.1, lr_min_factor=0.1, schedule_type='cosine'):
        self.total_rounds = total_rounds
        self.warmup_rounds = int(total_rounds * warmup_ratio)
        self.lr_min_factor = lr_min_factor
        self.schedule_type = schedule_type
        self.decay_rounds = total_rounds - self.warmup_rounds

    def get_lr_factor(self, round_idx):
        if self.schedule_type == 'constant':
            return 1.0
        if round_idx < self.warmup_rounds:
            if self.warmup_rounds > 0:
                return (round_idx + 1) / self.warmup_rounds
            return 1.0
        if self.decay_rounds <= 0:
            return 1.0
        decay_progress = (round_idx - self.warmup_rounds) / self.decay_rounds
        decay_progress = min(decay_progress, 1.0)
        if self.schedule_type == 'cosine':
            cosine_factor = 0.5 * (1 + math.cos(math.pi * decay_progress))
            return self.lr_min_factor + (1.0 - self.lr_min_factor) * cosine_factor
        elif self.schedule_type == 'linear':
            return 1.0 - (1.0 - self.lr_min_factor) * decay_progress
        else:
            return 1.0

    def get_schedule_info(self):
        return {'schedule_type': self.schedule_type, 'total_rounds': self.total_rounds, 'warmup_rounds': self.warmup_rounds, 'decay_rounds': self.decay_rounds, 'lr_min_factor': self.lr_min_factor}

def create_lr_scheduler(config, total_rounds=None):
    opt_config = config.get('optimization', {})
    schedule_type = opt_config.get('lr_schedule', 'constant')
    lr_min_factor = opt_config.get('lr_min_factor', 0.1)
    method_params = config.get('method', {}).get('params', {})
    warmup_ratio = method_params.get('warmup_ratio', 0.1)
    if total_rounds is None:
        total_rounds = int(config.get('training', {}).get('global_rounds', 100))
    return FederatedLRScheduler(total_rounds=total_rounds, warmup_ratio=warmup_ratio, lr_min_factor=lr_min_factor, schedule_type=schedule_type)
