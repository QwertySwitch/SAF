import logging
import random
import numpy as np
import torch
logger = logging.getLogger(__name__)

def setup_device(gpu_id=0):
    if gpu_id == -1:
        logger.info('Using CPU')
        return torch.device('cpu')
    if not torch.cuda.is_available():
        logger.warning('CUDA not available, falling back to CPU')
        return torch.device('cpu')
    num_gpus = torch.cuda.device_count()
    if gpu_id >= num_gpus:
        raise ValueError(f'GPU {gpu_id} not available. Available GPUs: 0-{num_gpus - 1}')
    torch.cuda.set_device(gpu_id)
    device_name = torch.cuda.get_device_name(gpu_id)
    logger.info(f'Using GPU {gpu_id}: {device_name}')
    return torch.device(f'cuda:{gpu_id}')

def get_gpu_memory(device):
    if device.type == 'cpu':
        return 'N/A (CPU)'
    if not torch.cuda.is_available():
        return 'N/A (CUDA not available)'
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    return f'Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB'

def get_gpu_memory_stats(device):
    if device.type == 'cpu' or not torch.cuda.is_available():
        return {'allocated_gb': 0.0, 'reserved_gb': 0.0, 'free_gb': 0.0, 'total_gb': 0.0}
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    free = total - allocated
    return {'allocated_gb': allocated, 'reserved_gb': reserved, 'free_gb': free, 'total_gb': total}

def clear_gpu_memory(device):
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f'Cleared GPU memory cache for {device}')

def list_available_gpus():
    if not torch.cuda.is_available():
        logger.info('No GPUs available')
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        device_props = torch.cuda.get_device_properties(i)
        gpus.append({'id': i, 'name': device_props.name, 'total_memory_gb': device_props.total_memory / 1024 ** 3, 'multi_processor_count': device_props.multi_processor_count, 'compute_capability': f'{device_props.major}.{device_props.minor}'})
    return gpus

def print_gpu_info():
    if not torch.cuda.is_available():
        logger.info('CUDA is not available')
        return
    logger.info(f'CUDA Version: {torch.version.cuda}')
    logger.info(f'PyTorch Version: {torch.__version__}')
    logger.info(f'Available GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f'GPU {i}: {props.name}')
        logger.info(f'  Memory: {props.total_memory / 1024 ** 3:.2f}GB')
        logger.info(f'  Compute Capability: {props.major}.{props.minor}')
        logger.info(f'  Multi-processors: {props.multi_processor_count}')

def get_optimal_gpu():
    if not torch.cuda.is_available():
        return -1
    max_free_memory = 0
    optimal_gpu = 0
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        stats = get_gpu_memory_stats(device)
        free_memory = stats['free_gb']
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            optimal_gpu = i
    logger.info(f'Optimal GPU: {optimal_gpu} (Free: {max_free_memory:.2f}GB)')
    return optimal_gpu

def set_random_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f'Random seed set to {seed}')

def enable_tf32(enable=True):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.allow_tf32 = enable
        logger.info(f"TF32 precision: {('enabled' if enable else 'disabled')}")

def get_device_info(device):
    info = {'device_type': device.type, 'device_index': device.index if device.type == 'cuda' else None}
    if device.type == 'cuda' and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        info.update({'device_name': props.name, 'compute_capability': f'{props.major}.{props.minor}', 'total_memory_gb': props.total_memory / 1024 ** 3, 'multi_processor_count': props.multi_processor_count, 'cuda_version': torch.version.cuda})
        mem_stats = get_gpu_memory_stats(device)
        info.update(mem_stats)
    return info

def get_memory_snapshot(device):
    if device.type != 'cuda' or not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device) / 1024 ** 3

class GPUMemoryTracker:

    def __init__(self, device, name='Operation'):
        self.device = device
        self.name = name
        self.memory_before = 0.0
        self.memory_after = 0.0

    def __enter__(self):
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            self.memory_before = torch.cuda.memory_allocated(self.device) / 1024 ** 3
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            self.memory_after = torch.cuda.memory_allocated(self.device) / 1024 ** 3
            memory_used = self.memory_after - self.memory_before
            logger.info(f'{self.name} | GPU Memory: {memory_used:+.2f}GB (Total: {self.memory_after:.2f}GB)')

    def get_memory_used(self):
        return self.memory_after - self.memory_before
