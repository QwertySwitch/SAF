import importlib
import logging
from pathlib import Path
from typing import Dict, Type
logger = logging.getLogger('fedfl')
from .fedavg import FedAvg
from .fedit import FedIT
from .ffalora import FFALoRA
from .loraa2 import LoRAA2
from .fedftg import FedFTG
from .fedexlora import FedExLoRA
from .hetlora import HetLoRA
from .fedsalora import FedSALoRA
_METHOD_CLASSES = {}

def _discover_methods():
    methods_dir = Path(__file__).parent
    for py_file in methods_dir.glob('*.py'):
        if py_file.name.startswith('_'):
            continue
        module_name = py_file.stem
        full_module_name = f'src.methods.{module_name}'
        try:
            module = importlib.import_module(f'.{module_name}', package='src.methods')
            candidates = []
            for attr_name in dir(module):
                if attr_name.startswith('_'):
                    continue
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and hasattr(attr, 'local_train') and hasattr(attr, 'aggregate'):
                    is_native = getattr(attr, '__module__', '') == full_module_name
                    candidates.append((attr_name, attr, is_native))
            if candidates:
                candidates.sort(key=lambda x: (not x[2], x[0]))
                _METHOD_CLASSES[module_name] = candidates[0][1]
        except Exception as e:
            logger.debug(f'Could not load method from {py_file.name}: {e}')
_discover_methods()

def get_method(method_name, config, device):
    method_name = method_name.lower()
    _DIRREG_SUPPORTED = {'fedit', 'fedexlora', 'hetlora'}
    _training = config.get('training', {}) if isinstance(config, dict) else {}
    _dl = float(_training.get('dirreg_lambda', 0.0)) if isinstance(_training, dict) else 0.0
    _mu = float(_training.get('fedprox_mu', 0.0)) if isinstance(_training, dict) else 0.0
    _base = method_name.split('_')[0] if '_' in method_name else method_name
    if _base not in _DIRREG_SUPPORTED and (_dl > 0 or _mu > 0):
        logger.warning(f"Method '{method_name}' does not support DirReg/FedProx. Parameters dirreg_lambda={_dl}, fedprox_mu={_mu} will be ignored. Supported: {sorted(_DIRREG_SUPPORTED)}")
    if method_name in _METHOD_CLASSES:
        return _METHOD_CLASSES[method_name](config, device)
    if method_name == 'hetlora':
        return HetLoRA(config, device)
    available = sorted(_METHOD_CLASSES.keys())
    raise ValueError(f"\n{'=' * 60}\nERROR: Unknown method '{method_name}'\n{'=' * 60}\n\nAvailable methods: {', '.join(available)}\n\nTo add a new method:\n  1. Create src/methods/yourmethod.py\n  2. Define a class with local_train() and aggregate() methods\n  3. It will be auto-discovered.\n{'=' * 60}")

def method_exists(method_name):
    method_name = method_name.lower()
    if method_name in _METHOD_CLASSES:
        return True
    if method_name == 'hetlora':
        return True
    return False

def list_methods():
    return sorted(_METHOD_CLASSES.keys())
__all__ = ['FedAvg', 'FedIT', 'FFALoRA', 'LoRAA2', 'FedExLoRA', 'HetLoRA', 'FedSALoRA', 'FedFTG', 'get_method', 'method_exists', 'list_methods']
