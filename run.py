import argparse
import json
import logging
import math
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import yaml
from tqdm import tqdm
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from src.utils.config import load_config, validate_config, format_alpha_dir
from src.utils.logger import setup_logger
from src.utils.gpu import setup_device, get_gpu_memory
from src.utils.viz import plot_training_curves, plot_method_comparison, plot_client_distribution
from src.data.loader import get_data_loader
from src.data.splitter import DirichletSplitter
from src.data.transforms import GLUETransforms, CIFAR100Transforms
from src.models.nlp import get_nlp_model
from src.models.vision import get_vision_model
from src.methods import get_method, method_exists
from src.server.coordinator import FederatedCoordinator
from src.metrics.evaluator import get_evaluator
from src.metrics.tracker import MetricsTracker
from src.utils.method_logger import MethodLogger

class DictDataset(torch.utils.data.Dataset):

    def __init__(self, data_dict):
        self.data = data_dict
        self.length = len(next(iter(data_dict.values())))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {k: v[idx] for (k, v) in self.data.items()}

class VisionDictDataset(torch.utils.data.Dataset):

    def __init__(self, data_dict, transform=None):
        self.images = data_dict['images']
        self.labels = data_dict['labels']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return {'images': image, 'labels': label}

def parse_args():
    parser = argparse.ArgumentParser(description='FedDoRA: Federated Learning Framework', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--method', type=str, help='FL method (optional, defaults to config method.name)')
    parser.add_argument('--methods', type=str, help='Comma-separated methods for comparison (e.g., "fedavg,fedit,fedexlora")')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID (-1 for CPU, overrides config value)')
    parser.add_argument('--seed', type=int, help='Random seed (overrides config)')
    parser.add_argument('--resume', type=str, help='Checkpoint path to resume training')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation mode only (no training)')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint for evaluation (required with --eval_only)')
    parser.add_argument('--output', type=str, help='Custom output directory (default: results/{benchmark}/{dataset}/{method}/{alpha}/{exp_name})')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--save', action='store_true', help='Enable model checkpoint saving (disabled by default to save disk space)')
    parser.add_argument('--alpha', type=float, help='Dirichlet alpha for data splitting (overrides config). Use pre-split data if available.')
    parser.add_argument('--no_presplit', action='store_true', default=False, help='Disable pre-split data loading (default: use pre-split if available)')
    args = parser.parse_args()
    if args.method and args.methods:
        parser.error('Cannot specify both --method and --methods')
    if args.eval_only and (not args.checkpoint):
        parser.error('--checkpoint required with --eval_only')
    return args

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
GLUE_DATASETS = {'cola', 'sst2', 'mnli', 'qqp', 'qnli', 'rte', 'mrpc', 'stsb', 'wnli'}
SQUAD_DATASETS = {'squad_v2', 'squad', 'squad_v2_topic'}
CAUSAL_LM_DATASETS = {'e2e_nlg'}

def get_benchmark_type(dataset_name):
    if dataset_name in GLUE_DATASETS:
        return 'glue'
    elif dataset_name in SQUAD_DATASETS:
        return 'squad'
    elif dataset_name in ('cifar100', 'cifar100_pp', 'cifar10', 'pathmnist', 'octmnist'):
        return 'cifar100'
    elif dataset_name == 'domainnet':
        return 'domainnet'
    elif dataset_name == 'banking77':
        return 'banking77'
    elif dataset_name in CAUSAL_LM_DATASETS:
        return 'causal_lm'
    return dataset_name

def _sanitize_metric(value):
    import math
    if math.isnan(value):
        logging.getLogger('fedfl').warning(f'NaN metric detected, replacing with 0.0')
        return 0.0
    return value

def get_primary_metric(metrics, dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cola':
        value = metrics.get('matthews_correlation', metrics.get('accuracy', 0.0))
    elif dataset_name == 'stsb':
        value = metrics.get('pearson', metrics.get('spearman', 0.0))
    elif dataset_name in ('domainnet', 'cifar100', 'cifar100_pp', 'cifar10', 'pathmnist', 'octmnist'):
        value = metrics.get('top1_acc', metrics.get('accuracy', 0.0))
    elif dataset_name in SQUAD_DATASETS:
        value = metrics.get('f1', 0.0)
    elif dataset_name in CAUSAL_LM_DATASETS:
        value = -metrics.get('perplexity', float('inf'))
    else:
        value = metrics.get('accuracy', 0.0)
    return _sanitize_metric(value)

def _detect_ema_mode(config):
    training = config.get('training', {})
    params = config.get('method', {}).get('params', {})
    method_name = config.get('method', {}).get('name', '').lower()
    has_ema = training.get('ema_decay') is not None and training.get('ema_decay', 0) > 0
    ema_feedback = training.get('ema_feedback')
    if not has_ema:
        lr_values = set()
        for key in ('lr', 'lr_a', 'lr_b', 'lr_cls'):
            val = params.get(key)
            if val is not None:
                lr_values.add(float(val))
        all_eq_2e4 = len(lr_values) >= 1 and all((abs(v - 0.0002) < 1e-06 for v in lr_values))
        return 'fedavg' if all_eq_2e4 else 'asym_lr'
    if ema_feedback == 'a_only':
        return 'saf09'
    elif ema_feedback == 'b_only':
        return 'ema_b09'
    elif ema_feedback == 'full':
        return 'ema_full09'
    elif ema_feedback in (None, 'skip'):
        return 'ema_eval09'
    else:
        return 'ema_eval09'
_detect_lr_category = _detect_ema_mode

def setup_output_directory(args, config):
    if args.output:
        output_dir = Path(args.output)
    elif config.get('experiment', {}).get('output_dir'):
        base_output = Path(config['experiment']['output_dir'])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = config.get('experiment', {}).get('name', 'experiment')
        output_dir = base_output / f'{exp_name}_{timestamp}'
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = config['dataset']['name']
        method_name = args.method or config['method']['name']
        alpha_str = format_alpha_dir(config['dataset']['alpha'])
        exp_base_name = config.get('experiment', {}).get('name')
        if exp_base_name:
            exp_name = f'{exp_base_name}_{timestamp}'
        else:
            exp_name = f'{dataset_name}_{method_name}_{alpha_str}_{timestamp}'
        exp_type = config.get('experiment', {}).get('type', 'main').lower()
        ema_mode = _detect_ema_mode(config)
        if exp_type in ('paper', 'analysis', 'main'):
            output_dir = Path('results') / 'paper' / dataset_name / method_name / alpha_str / ema_mode / exp_name
        elif exp_type == 'tuning':
            output_dir = Path('results') / 'ablations' / dataset_name / method_name / alpha_str / 'hp_search' / exp_name
        elif exp_type == 'ablation':
            abl_category = config.get('experiment', {}).get('category', 'other')
            output_dir = Path('results') / 'ablations' / dataset_name / method_name / alpha_str / abl_category / exp_name
        else:
            output_dir = Path('results') / 'paper' / dataset_name / method_name / alpha_str / ema_mode / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    (output_dir / 'metrics').mkdir(exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)
    return output_dir

def try_load_presplit_data(config, logger):
    _answer_info = None
    dataset_name = config['dataset']['name']
    alpha = config['dataset']['alpha']
    script_dir = Path(__file__).parent.resolve()
    benchmark_type = get_benchmark_type(dataset_name)
    if benchmark_type == 'glue':
        splits_dir = script_dir / 'data' / 'glue' / 'splits' / dataset_name
    elif benchmark_type == 'cifar100':
        splits_dir = script_dir / 'data' / dataset_name / 'splits'
    elif benchmark_type == 'squad':
        splits_dir = script_dir / 'data' / dataset_name / 'splits'
    elif benchmark_type == 'banking77':
        splits_dir = script_dir / 'data' / 'banking77' / 'splits'
    elif benchmark_type == 'causal_lm':
        splits_dir = script_dir / 'data' / dataset_name / 'splits'
    else:
        splits_dir = script_dir / 'data' / benchmark_type / 'splits' / dataset_name
    split_seed = config['dataset'].get('split_seed', 42)
    alpha_dir_name = format_alpha_dir(alpha)
    if split_seed != 42:
        alpha_dir_name = f'{alpha_dir_name}_s{split_seed}'
    alpha_dir = splits_dir / alpha_dir_name
    train_path = splits_dir / 'train.pt'
    indices_path = alpha_dir / 'client_indices.pt'
    if dataset_name == 'mnli':
        val_matched_path = splits_dir / 'validation_matched.pt'
        val_mismatched_path = splits_dir / 'validation_mismatched.pt'
        required_paths = [train_path, val_matched_path, val_mismatched_path, indices_path]
        if not all((p.exists() for p in required_paths)):
            return None
        logger.info(f'Loading pre-split data from {splits_dir}')
        try:
            train_data = torch.load(train_path, weights_only=True)
            val_data = {'matched': torch.load(val_matched_path, weights_only=True), 'mismatched': torch.load(val_mismatched_path, weights_only=True)}
        except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
            logger.warning(f'Failed to load pre-split data: {e}. Falling back to fresh loading.')
            return None
        test_data = val_data
        logger.info('Loaded pre-split data:')
        logger.info('  Train samples: %d', len(train_data['labels']))
        logger.info('  Val matched: %d', len(val_data['matched']['labels']))
        logger.info('  Val mismatched: %d', len(val_data['mismatched']['labels']))
    elif benchmark_type == 'cifar100':
        test_path = splits_dir / 'test.pt'
        required_paths = [train_path, test_path, indices_path]
        if not all((p.exists() for p in required_paths)):
            return None
        logger.info(f'Loading pre-split CIFAR-100 data from {splits_dir}')
        try:
            train_data = torch.load(train_path, weights_only=True)
            test_data = torch.load(test_path, weights_only=True)
        except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
            logger.warning(f'Failed to load pre-split data: {e}. Falling back to fresh loading.')
            return None
        val_data = test_data
        logger.info('Loaded pre-split data:')
        logger.info('  Train samples: %d', len(train_data['labels']))
        logger.info('  Test samples: %d', len(test_data['labels']))
    elif benchmark_type == 'banking77':
        test_path = splits_dir / 'test.pt'
        required_paths = [train_path, test_path, indices_path]
        if not all((p.exists() for p in required_paths)):
            return None
        logger.info(f'Loading pre-split Banking77 data from {splits_dir}')
        try:
            train_data = torch.load(train_path, weights_only=True)
            test_data = torch.load(test_path, weights_only=True)
        except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
            logger.warning(f'Failed to load pre-split data: {e}. Falling back to fresh loading.')
            return None
        val_data = test_data
        logger.info('Loaded pre-split data:')
        logger.info('  Train samples: %d', len(train_data['labels']))
        logger.info('  Test samples: %d', len(test_data['labels']))
    elif benchmark_type == 'squad':
        val_path = splits_dir / 'validation.pt'
        required_paths = [train_path, val_path, indices_path]
        if not all((p.exists() for p in required_paths)):
            return None
        logger.info(f'Loading pre-split SQuAD data from {splits_dir}')
        try:
            train_data = torch.load(train_path, weights_only=True)
            val_data = torch.load(val_path, weights_only=True)
        except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
            logger.warning(f'Failed to load pre-split data: {e}. Falling back to fresh loading.')
            return None
        test_data = val_data
        answer_info_path = splits_dir / 'validation_answers.json'
        if answer_info_path.exists():
            import json as _json
            with open(answer_info_path, 'r') as f:
                _answer_info = _json.load(f)
            logger.info(f'  Answer info: {len(_answer_info)} gold answers loaded')
        logger.info('Loaded pre-split SQuAD data:')
        logger.info('  Train samples: %d', len(train_data['start_positions']))
        logger.info('  Val samples: %d', len(val_data['start_positions']))
    else:
        val_path = splits_dir / 'validation.pt'
        required_paths = [train_path, val_path, indices_path]
        if not all((p.exists() for p in required_paths)):
            return None
        logger.info(f'Loading pre-split data from {splits_dir}')
        try:
            train_data = torch.load(train_path, weights_only=True)
            val_data = torch.load(val_path, weights_only=True)
        except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
            logger.warning(f'Failed to load pre-split data: {e}. Falling back to fresh loading.')
            return None
        test_data = val_data
        logger.info('Loaded pre-split data:')
        logger.info('  Train samples: %d', len(train_data['labels']))
        logger.info('  Val samples: %d', len(val_data['labels']))
    try:
        client_indices_raw = torch.load(indices_path, weights_only=True)
    except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
        logger.warning(f'Failed to load client indices: {e}. Falling back to fresh loading.')
        return None
    client_indices = {int(k): np.array(v, dtype=np.int64) for (k, v) in client_indices_raw.items()}
    logger.info(f'  Clients: {len(client_indices)}')
    logger.info(f'  Alpha: {alpha}')
    return (train_data, val_data, test_data, client_indices, _answer_info)

def load_data(config, logger, use_presplit=True):
    logger.info(f"Loading dataset: {config['dataset']['name']}")
    if use_presplit:
        presplit_data = try_load_presplit_data(config, logger)
        if presplit_data is not None:
            (train_data, val_data, test_data, client_indices, answer_info) = presplit_data
            client_sizes = [len(indices) for indices in client_indices.values()]
            logger.info('Client data distribution (pre-split):')
            logger.info(f'  Mean samples: {np.mean(client_sizes):.1f}')
            logger.info(f'  Std samples: {np.std(client_sizes):.1f}')
            logger.info(f'  Min samples: {np.min(client_sizes)}')
            logger.info(f'  Max samples: {np.max(client_sizes)}')
            return (train_data, val_data, test_data, client_indices, answer_info)
    logger.info('Pre-split data not found, loading from HuggingFace...')
    data_loader = get_data_loader(config['dataset']['name'])
    data_loader.load()
    train_dataset = data_loader.get_train_data()
    val_dataset = data_loader.get_validation_data()
    test_dataset = data_loader.get_test_data()
    logger.info(f'Train samples: {len(train_dataset)}')
    logger.info(f'Val samples: {len(val_dataset)}')
    logger.info(f'Test samples: {len(test_dataset)}')
    dataset_name = config['dataset']['name']
    if dataset_name in GLUE_DATASETS:
        max_length = config['dataset'].get('max_length', 128)
        model_checkpoint = config['model'].get('checkpoint', 'FacebookAI/roberta-base')
        logger.info(f'Tokenizing {dataset_name} on-the-fly (max_length={max_length})...')
        transform = GLUETransforms(task_name=dataset_name, tokenizer_name=model_checkpoint, max_length=max_length)

        def _tokenize_to_dict(dataset_split):
            column_names = dataset_split.column_names
            tokenized = dataset_split.map(transform, batched=True, remove_columns=column_names, desc=f'Tokenizing {dataset_name}')
            return {'input_ids': torch.tensor(tokenized['input_ids']), 'attention_mask': torch.tensor(tokenized['attention_mask']), 'labels': torch.tensor(tokenized['labels'])}
        train_dataset = _tokenize_to_dict(train_dataset)
        if isinstance(test_dataset, dict):
            val_dataset = {k: _tokenize_to_dict(v) for (k, v) in test_dataset.items()}
        else:
            val_dataset = _tokenize_to_dict(val_dataset)
        test_dataset = val_dataset
        logger.info(f"Tokenization complete. Train: {len(train_dataset['labels'])} samples")
    elif dataset_name in SQUAD_DATASETS:
        max_length = config['dataset'].get('max_length', 384)
        model_checkpoint = config['model'].get('checkpoint', 'FacebookAI/roberta-base')
        logger.info(f'Tokenizing SQuAD on-the-fly (max_length={max_length})...')
        from src.data.squad_transforms import SQuADTransforms
        squad_transform = SQuADTransforms(tokenizer_name=model_checkpoint, max_length=max_length)
        tokenized_train = train_dataset.map(squad_transform, batched=True, remove_columns=train_dataset.column_names, desc='Tokenizing SQuAD train')
        train_dataset = {'input_ids': torch.tensor(tokenized_train['input_ids']), 'attention_mask': torch.tensor(tokenized_train['attention_mask']), 'start_positions': torch.tensor(tokenized_train['start_positions']), 'end_positions': torch.tensor(tokenized_train['end_positions'])}
        tokenized_val = val_dataset.map(squad_transform, batched=True, remove_columns=val_dataset.column_names, desc='Tokenizing SQuAD validation')
        val_dataset = {'input_ids': torch.tensor(tokenized_val['input_ids']), 'attention_mask': torch.tensor(tokenized_val['attention_mask']), 'start_positions': torch.tensor(tokenized_val['start_positions']), 'end_positions': torch.tensor(tokenized_val['end_positions'])}
        test_dataset = val_dataset
        logger.info(f"Tokenization complete. Train: {len(train_dataset['start_positions'])} samples")
    elif dataset_name in ('cifar100', 'cifar100_pp', 'cifar10', 'pathmnist', 'octmnist'):
        import torchvision.transforms as T_cv
        to_tensor = T_cv.ToTensor()

        def _cifar_to_dict(ds):
            images = torch.stack([to_tensor(ds[i][0]) for i in range(len(ds))])
            labels = torch.tensor([ds[i][1] for i in range(len(ds))])
            return {'images': images, 'labels': labels}
        logger.info('Converting CIFAR-100 to tensor dict format...')
        train_dataset = _cifar_to_dict(train_dataset)
        val_dataset = _cifar_to_dict(test_dataset)
        test_dataset = val_dataset
        logger.info(f"Conversion complete. Train: {len(train_dataset['labels'])} samples")
    logger.info(f"Splitting data with Dirichlet(α={config['dataset']['alpha']})")
    splitter = DirichletSplitter(seed=config['experiment']['seed'])
    if isinstance(train_dataset, dict) and 'labels' in train_dataset:
        labels_data = train_dataset['labels']
        labels = np.array(labels_data) if not isinstance(labels_data, torch.Tensor) else labels_data.numpy()
    elif isinstance(train_dataset, dict) and 'start_positions' in train_dataset:
        start_pos = train_dataset['start_positions']
        start_pos_np = np.array(start_pos) if not isinstance(start_pos, torch.Tensor) else start_pos.numpy()
        labels = (start_pos_np > 0).astype(np.int64)
    elif hasattr(train_dataset, 'features'):
        labels = np.array(train_dataset['label'])
    else:
        labels = np.array([ex['label'] for ex in train_dataset])
    client_data_raw = splitter.split(labels=labels, num_clients=config['dataset']['num_clients'], alpha=config['dataset']['alpha'])
    client_data = {int(k): np.array(v, dtype=np.int64) for (k, v) in client_data_raw.items()}
    client_sizes = [len(indices) for indices in client_data.values()]
    logger.info('Client data distribution:')
    logger.info(f'  Mean samples: {np.mean(client_sizes):.1f}')
    logger.info(f'  Std samples: {np.std(client_sizes):.1f}')
    logger.info(f'  Min samples: {np.min(client_sizes)}')
    logger.info(f'  Max samples: {np.max(client_sizes)}')
    return (train_dataset, val_dataset, test_dataset, client_data, None)

def get_model_max_seq_length(model_name):
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_name)
    max_pos = getattr(model_config, 'max_position_embeddings', 514)
    return max_pos - 2

def load_model(config, device, logger):
    logger.info(f"Loading model: {config['model']['name']}")
    dataset_type = config['dataset']['name']
    if dataset_type in GLUE_DATASETS or dataset_type == 'banking77':
        num_labels_map = {'cola': 2, 'sst2': 2, 'mnli': 3, 'qqp': 2, 'qnli': 2, 'rte': 2, 'mrpc': 2, 'stsb': 1, 'wnli': 2, 'banking77': 77}
        num_labels = num_labels_map.get(dataset_type, 2)
        model = get_nlp_model(task=dataset_type, num_labels=num_labels, model_name=config['model']['checkpoint'], pretrained=config['model']['pretrained'], gradient_checkpointing=config['model'].get('gradient_checkpointing', False))
        model = model.to(device)
        model_max_length = get_model_max_seq_length(config['model']['checkpoint'])
        config_max_length = config['dataset'].get('max_length')
        if config_max_length:
            max_seq_length = min(config_max_length, model_max_length)
            logger.info(f'Using config max_length={max_seq_length} (model max={model_max_length})')
        else:
            max_seq_length = model_max_length
            logger.info(f'Using model max sequence length: {max_seq_length}')
    elif dataset_type in SQUAD_DATASETS:
        from src.models.qa import get_qa_model
        model = get_qa_model(model_name=config['model']['checkpoint'], pretrained=config['model']['pretrained'], gradient_checkpointing=config['model'].get('gradient_checkpointing', False))
        model = model.to(device)
        model_max_length = get_model_max_seq_length(config['model']['checkpoint'])
        config_max_length = config['dataset'].get('max_length', 384)
        max_seq_length = min(config_max_length, model_max_length)
        logger.info(f'QA model max_length={max_seq_length} (model max={model_max_length})')
    elif dataset_type in CAUSAL_LM_DATASETS:
        from src.models.causal_lm import get_causal_lm_model
        model = get_causal_lm_model(model_name=config['model']['checkpoint'], pretrained=config['model']['pretrained'], gradient_checkpointing=config['model'].get('gradient_checkpointing', False))
        model = model.to(device)
        config_max_length = config['dataset'].get('max_length', 256)
        max_seq_length = config_max_length
        logger.info(f'Causal LM model max_length={max_seq_length}')
    elif dataset_type in ('domainnet', 'cifar100', 'cifar100_pp', 'cifar10', 'pathmnist', 'octmnist'):
        num_classes = config['dataset'].get('categories', 100)
        model = get_vision_model(task=dataset_type, num_classes=num_classes, model_name=config['model']['checkpoint'], pretrained=config['model']['pretrained'], image_size=config['model'].get('image_size', 224), gradient_checkpointing=config['model'].get('gradient_checkpointing', False))
        model = model.to(device)
        max_seq_length = None
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}')
    return (model, max_seq_length)

def truncate_sequences(dataset, max_length=128):
    if not isinstance(dataset, dict) or 'input_ids' not in dataset:
        return dataset
    truncated = {}
    for (key, data) in dataset.items():
        if key in ['input_ids', 'attention_mask', 'labels'] and hasattr(data, 'shape') and (len(data.shape) == 2):
            if data.shape[1] > max_length:
                truncated[key] = data[:, :max_length]
            else:
                truncated[key] = data
        else:
            truncated[key] = data
    return truncated

def compute_client_label_distribution(train_dataset, client_data, is_regression=False):
    if is_regression:
        return {client_id: {} for client_id in client_data.keys()}
    if isinstance(train_dataset, dict) and 'start_positions' in train_dataset and ('labels' not in train_dataset):
        start_pos = np.array(list(train_dataset['start_positions']))
        all_labels = (start_pos > 0).astype(np.int64)
    elif isinstance(train_dataset, dict) and 'labels' in train_dataset:
        labels_data = train_dataset['labels']
        if isinstance(labels_data, torch.Tensor) and labels_data.dim() == 2:
            return {client_id: {} for client_id in client_data.keys()}
        all_labels = np.array(list(labels_data))
    elif hasattr(train_dataset, 'features'):
        all_labels = np.array(list(train_dataset['label']))
    else:
        all_labels = np.array([ex['label'] for ex in train_dataset])
    if all_labels.dtype in (np.float32, np.float64, float):
        return {client_id: {} for client_id in client_data.keys()}
    client_label_dist = {}
    for (client_id, indices) in client_data.items():
        client_labels = all_labels[indices]
        label_counts = {}
        for label in client_labels:
            label_key = int(label)
            label_counts[label_key] = label_counts.get(label_key, 0) + 1
        client_label_dist[client_id] = label_counts
    return client_label_dist

def create_dataloader(dataset, batch_size, shuffle=False, max_seq_length=128, transform=None, num_workers=0, pin_memory=False):
    dl_kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    if isinstance(dataset, dict) and 'input_ids' in dataset:
        dataset = truncate_sequences(dataset, max_length=max_seq_length)
        tensors = []
        keys = []
        for key in ['input_ids', 'attention_mask', 'labels', 'start_positions', 'end_positions']:
            if key in dataset:
                data = dataset[key]
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data)
                tensors.append(data)
                keys.append(key)
        tensor_dict = dict(zip(keys, tensors))
        ds = DictDataset(tensor_dict)
        return torch.utils.data.DataLoader(ds, **dl_kwargs)
    elif isinstance(dataset, dict) and 'images' in dataset:
        ds = VisionDictDataset(dataset, transform=transform)
        return torch.utils.data.DataLoader(ds, **dl_kwargs)
    else:
        return torch.utils.data.DataLoader(dataset, **dl_kwargs)

def run_personalized_eval(model, fl_method, client_test_loaders, evaluator, device, dataset_name, logger, round_idx):
    if not hasattr(fl_method, 'get_client_state') or not hasattr(fl_method, 'get_participating_clients'):
        return {}
    participating = fl_method.get_participating_clients()
    if not participating:
        return {}
    eval_clients = [c for c in participating if c in client_test_loaders]
    if not eval_clients:
        return {}
    global_state = model.state_dict()
    # state_dict() tensors alias the live parameters, so loading a client's
    # params in-place would mutate global_state too and the final restore
    # would be a no-op. Clone every key we overwrite so the restore is real.
    overwritten = {}
    client_metrics = []
    model.eval()
    for client_id in eval_clients:
        client_state = fl_method.get_client_state(client_id)
        if client_state is None:
            continue
        for name in client_state:
            if name in global_state and name not in overwritten:
                overwritten[name] = global_state[name].detach().clone()
        personalized_state = {**global_state}
        for (name, param) in client_state.items():
            if name in personalized_state:
                personalized_state[name] = param.to(device)
        model.load_state_dict(personalized_state, strict=False)
        metrics = evaluator.evaluate(model, client_test_loaders[client_id])
        acc = get_primary_metric(metrics, dataset_name)
        client_metrics.append(acc)
    if overwritten:
        model.load_state_dict(overwritten, strict=False)
    if not client_metrics:
        return {}
    avg_acc = sum(client_metrics) / len(client_metrics)
    result = {'personalized_accuracy': avg_acc, 'personalized_num_clients': len(client_metrics), 'personalized_min': min(client_metrics), 'personalized_max': max(client_metrics)}
    logger.info(f'[R{round_idx + 1}] Personalized eval: avg_acc={avg_acc:.4f} over {len(client_metrics)} clients (min={min(client_metrics):.4f}, max={max(client_metrics):.4f})')
    return result

def _extract_ema_layer_name(param_name):
    parts = param_name.split('.')
    layer_parts = []
    for (i, part) in enumerate(parts):
        if part == 'layer' and i + 1 < len(parts):
            layer_parts.append(f'L{parts[i + 1]}')
        elif part in ('query', 'key', 'value', 'output'):
            layer_parts.append(part[:1].upper())
    return '.'.join(layer_parts) if layer_parts else param_name[-30:]

def compute_ema_beta(config, round_idx):
    training_cfg = config.get('training', {})
    schedule = training_cfg.get('ema_beta_schedule', 'constant')
    if schedule == 'constant':
        return training_cfg.get('ema_decay', 0.0)
    beta_min = training_cfg.get('ema_beta_min', 0.5)
    beta_max = training_cfg.get('ema_beta_max', 0.9)
    power = training_cfg.get('ema_beta_power', 1.0)
    total_rounds = training_cfg.get('global_rounds', 100)
    if total_rounds <= 1:
        return beta_max
    t_normalized = round_idx / (total_rounds - 1)
    t_normalized = max(0.0, min(1.0, t_normalized))
    beta = beta_min + (beta_max - beta_min) * t_normalized ** power
    return beta

def compute_ema_diagnostics(ema_state, model_state):
    comp_norms = {'a': {}, 'b': {}, 'm': {}}
    comp_drifts = {'a': {}, 'b': {}, 'm': {}}
    for (k, ema_v) in ema_state.items():
        if 'lora_A' in k:
            comp = 'a'
        elif 'lora_B' in k:
            comp = 'b'
        elif 'lora_magnitude_vector' in k:
            comp = 'm'
        else:
            continue
        if k not in model_state:
            continue
        layer = _extract_ema_layer_name(k)
        cur_v = model_state[k].cpu()
        if ema_v.dim() >= 2:
            ema_norm = torch.norm(ema_v, p='fro').item()
            drift = torch.norm(ema_v - cur_v, p='fro').item()
        else:
            ema_norm = torch.norm(ema_v, p=2).item()
            drift = torch.norm(ema_v - cur_v, p=2).item()
        comp_norms[comp][layer] = ema_norm
        comp_drifts[comp][layer] = drift
    result = {}
    for comp in ('a', 'b', 'm'):
        norms = comp_norms[comp]
        drifts = comp_drifts[comp]
        if not norms:
            continue
        for (layer, val) in norms.items():
            result[f'ema_{comp}_norm_{layer}'] = val
        for (layer, val) in drifts.items():
            result[f'ema_{comp}_drift_{layer}'] = val
        norm_vals = list(norms.values())
        drift_vals = list(drifts.values())
        norm_mean = sum(norm_vals) / len(norm_vals)
        drift_mean = sum(drift_vals) / len(drift_vals)
        result[f'ema_{comp}_norm_mean'] = norm_mean
        result[f'ema_{comp}_drift_mean'] = drift_mean
        if len(norm_vals) >= 2:
            norm_var = sum(((x - norm_mean) ** 2 for x in norm_vals)) / len(norm_vals)
            drift_var = sum(((x - drift_mean) ** 2 for x in drift_vals)) / len(drift_vals)
            result[f'ema_{comp}_norm_std'] = math.sqrt(norm_var)
            result[f'ema_{comp}_drift_std'] = math.sqrt(drift_var)
        else:
            result[f'ema_{comp}_norm_std'] = 0.0
            result[f'ema_{comp}_drift_std'] = 0.0
    return result

def train_federated(config, method_name, model, train_dataset, client_indices, val_dataset, device, output_dir, logger, resume_path=None, max_seq_length=128, answer_info=None):
    logger.info(f'Initializing FL method: {method_name}')
    fl_method = get_method(method_name, config, device)
    method_logger = None
    if hasattr(fl_method, 'set_method_logger'):
        method_logger = MethodLogger(method_name=method_name, output_dir=output_dir, log_level='DEBUG', enable_json=True, enable_traces=True)
        fl_method.set_method_logger(method_logger)
        logger.info(f'MethodLogger initialized for {method_name}')
    if hasattr(fl_method, 'setup_model'):
        model = fl_method.setup_model(model)
        model = model.to(device)
    total_params = sum((p.numel() for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    trainable_ratio = trainable_params / total_params
    logger.info('Parameters: %s trainable / %s total (%.2f%%)', f'{trainable_params:,}', f'{total_params:,}', trainable_ratio * 100)
    lora_param_count = sum((p.numel() for (n, p) in model.named_parameters() if p.requires_grad and ('lora_A' in n or 'lora_B' in n)))
    if hasattr(fl_method, 'target_modules') and lora_param_count == 0:
        target_mods = getattr(fl_method, 'target_modules', [])
        raise RuntimeError(f"FATAL: No LoRA adapters created! target_modules={target_mods} matched zero modules in the model. For ViT models, use target_modules: ['qkv'] instead of ['query', 'value']. For RoBERTa models, use target_modules: ['query', 'value'].")
    effective_trainable_params = None
    if hasattr(fl_method, 'get_effective_trainable_params'):
        effective_trainable_params = fl_method.get_effective_trainable_params(model)
        effective_ratio = effective_trainable_params / total_params
        logger.info('Effective Parameters: %s (%.2f%% of total) - GaLore rank=%d', f'{effective_trainable_params:,}', effective_ratio * 100, getattr(fl_method, 'galore_rank', 'N/A'))
    model_params_metrics = {'total_parameters': total_params, 'trainable_parameters': trainable_params, 'trainable_ratio': trainable_ratio}
    if effective_trainable_params is not None:
        model_params_metrics['effective_trainable_parameters'] = effective_trainable_params
        model_params_metrics['effective_trainable_ratio'] = effective_trainable_params / total_params
    samples_list = [len(indices) for indices in client_indices.values()]
    client_data_stats = {'num_clients': len(client_indices), 'total_samples': sum(samples_list), 'samples_per_client': {'mean': float(np.mean(samples_list)), 'std': float(np.std(samples_list)), 'min': int(np.min(samples_list)), 'max': int(np.max(samples_list))}}
    batch_size = config['method']['params'].get('batch_size', 256)
    is_vision = isinstance(train_dataset, dict) and 'images' in train_dataset
    vision_dl_kwargs = dict(num_workers=4, pin_memory=True) if is_vision else {}
    dataset_name = config['dataset']['name']
    train_transform = None
    test_transform = None
    if isinstance(train_dataset, dict) and 'images' in train_dataset:
        image_size = config['model'].get('image_size', 224)
        if dataset_name in ('cifar100', 'cifar100_pp', 'cifar10', 'pathmnist', 'octmnist'):
            train_transform = CIFAR100Transforms(train=True, image_size=image_size)
            test_transform = CIFAR100Transforms(train=False, image_size=image_size)
        else:
            from src.data.transforms import DomainNetTransforms
            train_transform = DomainNetTransforms(train=True, image_size=image_size)
            test_transform = DomainNetTransforms(train=False, image_size=image_size)
    client_loaders = {}
    for (client_id, indices) in client_indices.items():
        if len(indices) == 0:
            continue
        if isinstance(train_dataset, dict) and 'input_ids' in train_dataset:
            client_data_dict = {}
            for key in ['input_ids', 'attention_mask', 'labels', 'start_positions', 'end_positions']:
                if key in train_dataset:
                    data = train_dataset[key]
                    if isinstance(data, torch.Tensor):
                        client_data_dict[key] = data[indices]
                    else:
                        client_data_dict[key] = torch.tensor([data[i] for i in indices])
            client_loaders[client_id] = create_dataloader(client_data_dict, batch_size, shuffle=True, max_seq_length=max_seq_length or 128)
        elif isinstance(train_dataset, dict) and 'images' in train_dataset:
            client_data_dict = {'images': train_dataset['images'][indices], 'labels': train_dataset['labels'][indices]}
            client_loaders[client_id] = create_dataloader(client_data_dict, batch_size, shuffle=True, transform=train_transform, **vision_dl_kwargs)
        else:
            client_subset = train_dataset.select(indices)
            client_loaders[client_id] = create_dataloader(client_subset, batch_size, shuffle=True, max_seq_length=max_seq_length or 128)
    personalized_eval = config.get('evaluation', {}).get('personalized_eval', False)
    client_test_loaders = {}
    if personalized_eval and hasattr(fl_method, 'get_client_state'):
        logger.info('Personalized eval enabled: splitting client data 80/20 for per-client test')
        rng = np.random.RandomState(config['experiment']['seed'] + 1)
        for (client_id, indices) in client_indices.items():
            if len(indices) < 5:
                continue
            n = len(indices)
            n_train = int(n * 0.8)
            perm = rng.permutation(n)
            train_idx = indices[perm[:n_train]]
            test_idx = indices[perm[n_train:]]
            if isinstance(train_dataset, dict) and 'input_ids' in train_dataset:
                test_data_dict = {}
                for key in ['input_ids', 'attention_mask', 'labels', 'start_positions', 'end_positions']:
                    if key in train_dataset:
                        data = train_dataset[key]
                        if isinstance(data, torch.Tensor):
                            test_data_dict[key] = data[test_idx]
                        else:
                            test_data_dict[key] = torch.tensor([data[i] for i in test_idx])
                client_test_loaders[client_id] = create_dataloader(test_data_dict, batch_size, shuffle=False, max_seq_length=max_seq_length or 128)
                train_data_dict = {}
                for key in ['input_ids', 'attention_mask', 'labels', 'start_positions', 'end_positions']:
                    if key in train_dataset:
                        data = train_dataset[key]
                        if isinstance(data, torch.Tensor):
                            train_data_dict[key] = data[train_idx]
                        else:
                            train_data_dict[key] = torch.tensor([data[i] for i in train_idx])
                client_loaders[client_id] = create_dataloader(train_data_dict, batch_size, shuffle=True, max_seq_length=max_seq_length or 128)
            elif isinstance(train_dataset, dict) and 'images' in train_dataset:
                client_test_loaders[client_id] = create_dataloader({'images': train_dataset['images'][test_idx], 'labels': train_dataset['labels'][test_idx]}, batch_size, shuffle=False, transform=test_transform, **vision_dl_kwargs)
                client_loaders[client_id] = create_dataloader({'images': train_dataset['images'][train_idx], 'labels': train_dataset['labels'][train_idx]}, batch_size, shuffle=True, transform=train_transform, **vision_dl_kwargs)
            else:
                client_test_loaders[client_id] = create_dataloader(train_dataset.select(test_idx.tolist()), batch_size, shuffle=False, max_seq_length=max_seq_length or 128)
                client_loaders[client_id] = create_dataloader(train_dataset.select(train_idx.tolist()), batch_size, shuffle=True, max_seq_length=max_seq_length or 128)
        logger.info(f'  Per-client test loaders created for {len(client_test_loaders)} clients')
    evaluator = get_evaluator(config['dataset']['name'], device)
    if answer_info is not None and hasattr(evaluator, 'set_answer_info'):
        evaluator.set_answer_info(answer_info)
        logger.info(f'  SQuAD evaluator: loaded {len(answer_info)} gold answers')
    test_batch_size = config['evaluation'].get('test_batch_size', 64)
    dataset_name = config['dataset']['name']
    if dataset_name == 'mnli' and isinstance(val_dataset, dict) and ('matched' in val_dataset):
        val_loader = {'matched': create_dataloader(val_dataset['matched'], test_batch_size, max_seq_length=max_seq_length or 128), 'mismatched': create_dataloader(val_dataset['mismatched'], test_batch_size, max_seq_length=max_seq_length or 128)}
    else:
        val_loader = create_dataloader(val_dataset, test_batch_size, max_seq_length=max_seq_length or 128, transform=test_transform, **vision_dl_kwargs)
    tracker = MetricsTracker(save_dir=output_dir / 'metrics', experiment_name=method_name)
    coordinator = FederatedCoordinator(global_model=model, fl_method=fl_method, client_data=client_loaders, device=device, config=config)
    start_round = 0
    if resume_path:
        logger.info(f'Resuming from checkpoint: {resume_path}')
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_round = checkpoint['round'] + 1
        logger.info(f'Resuming from round {start_round}')
    global_rounds = config['training']['global_rounds']
    eval_every = config['training']['eval_every']
    round_times = []
    training_start_time = time.time()
    best_metric_value = -float('inf')
    best_model_state = None
    best_round = -1
    ema_decay = config.get('training', {}).get('ema_decay', 0.0)
    ema_decay_m = config.get('training', {}).get('ema_decay_m', None)
    ema_decay_a = config.get('training', {}).get('ema_decay_a', None)
    ema_decay_b = config.get('training', {}).get('ema_decay_b', None)
    ema_component_aware = any((x is not None for x in [ema_decay_m, ema_decay_a, ema_decay_b]))
    ema_feedback_mode = config.get('training', {}).get('ema_feedback', None)
    ema_feedback_threshold = config.get('training', {}).get('ema_feedback_threshold', 0.05)
    ema_feedback_var_floor = config.get('training', {}).get('ema_feedback_var_floor', 1.5)
    ema_feedback_var_ceil = config.get('training', {}).get('ema_feedback_var_ceil', 3.0)
    ema_feedback_var_window = config.get('training', {}).get('ema_feedback_var_window', 5)
    ema_feedback_cv_floor = config.get('training', {}).get('ema_feedback_cv_floor', 0.15)
    ema_feedback_cv_ceil = config.get('training', {}).get('ema_feedback_cv_ceil', 0.5)
    ema_feedback_drift_scale = config.get('training', {}).get('ema_feedback_drift_scale', 0)
    ema_feedback_gamma_max = config.get('training', {}).get('ema_feedback_gamma_max', 0.3)
    ema_feedback_gamma_b_ratio = config.get('training', {}).get('ema_feedback_gamma_b_ratio', 0.0)
    ema_feedback_gamma_b = config.get('training', {}).get('ema_feedback_gamma_b', 0.0)
    ema_state = None
    ema_best_metric = -float('inf')
    ema_best_round = -1
    recent_losses = []
    prev_a_state = {}
    prev_b_state = {}
    drift_history = []
    prev_fedavg_a = {}
    prev_ema_a = {}
    a_smooth_state = {}
    a_smooth_decay = config.get('training', {}).get('a_smooth_decay', 0.9)
    if ema_feedback_mode == 'a_smooth':
        a_smooth_state = {k: v.clone() for (k, v) in model.state_dict().items() if 'lora_A' in k}
        logger.info(f'SAF-standalone enabled: A-smoothing with decay={a_smooth_decay}, {len(a_smooth_state)} lora_A params')
        logger.info('No global EMA model — FedAvg model is the only evaluated model')
    if ema_decay > 0 or ema_component_aware:
        if ema_component_aware:
            logger.info(f'Component-aware EMA enabled: decay_m={ema_decay_m}, decay_a={ema_decay_a}, decay_b={ema_decay_b}')
        else:
            logger.info(f'EMA enabled with decay={ema_decay}')
        if ema_feedback_mode is not None:
            logger.info(f"EMA feedback mode='{ema_feedback_mode}': feeding back EMA params to training model each round")
            if ema_feedback_mode == 'adaptive_a':
                logger.info(f'  Adaptive SAF: drift_threshold={ema_feedback_threshold}, var_floor={ema_feedback_var_floor}, var_ceil={ema_feedback_var_ceil}, var_window={ema_feedback_var_window}')
            elif ema_feedback_mode in ('adaptive_v3_a', 'dasf_a'):
                logger.info(f'  SAF {ema_feedback_mode}: cv_floor={ema_feedback_cv_floor}, cv_ceil={ema_feedback_cv_ceil}, drift_scale={ema_feedback_drift_scale}, gamma_max={ema_feedback_gamma_max}, window={ema_feedback_var_window}')
            if ema_feedback_gamma_b > 0:
                logger.info(f'  SABF B-feedback enabled: gamma_b={ema_feedback_gamma_b} (constant, independent of gamma_a)')
            elif ema_feedback_gamma_b_ratio > 0:
                logger.info(f'  SABF B-feedback enabled: gamma_b_ratio={ema_feedback_gamma_b_ratio} (gamma_b = gamma_a * ratio)')
        ema_state = {k: v.clone().cpu() for (k, v) in model.state_dict().items()}
    logger.info(f'Starting federated training for {global_rounds} rounds')
    if hasattr(fl_method, 'get_method_info'):
        method_info = fl_method.get_method_info()
        logger.info(f'Method configuration: {method_info}')
    for round_idx in tqdm(range(start_round, global_rounds), desc=f'Training {method_name}'):
        round_start_time = time.time()
        round_metrics = coordinator.train_round(round_idx)
        if ema_state is not None:
            with torch.no_grad():
                for (k, v) in model.state_dict().items():
                    if ema_component_aware:
                        if 'lora_magnitude_vector' in k:
                            beta = ema_decay_m if ema_decay_m is not None else 0.0
                        elif 'lora_A' in k:
                            beta = ema_decay_a if ema_decay_a is not None else 0.0
                        elif 'lora_B' in k:
                            beta = ema_decay_b if ema_decay_b is not None else 0.0
                        else:
                            beta = 0.0
                    else:
                        beta = compute_ema_beta(config, round_idx)
                    v_cpu = v.cpu()
                    if beta > 0:
                        ema_state[k].mul_(beta).add_(v_cpu, alpha=1.0 - beta)
                    else:
                        ema_state[k].copy_(v_cpu)
            ema_diag = compute_ema_diagnostics(ema_state, model.state_dict())
            round_metrics.update(ema_diag)
            if 'ema_b_norm_mean' in ema_diag:
                b_info = f"‖B‖={ema_diag['ema_b_norm_mean']:.2f}±{ema_diag.get('ema_b_norm_std', 0):.2f}"
                b_drift = f"drift={ema_diag.get('ema_b_drift_mean', 0):.4f}±{ema_diag.get('ema_b_drift_std', 0):.4f}"
                n_layers = sum((1 for k in ema_diag if k.startswith('ema_b_norm_L')))
                m_info = ''
                if 'ema_m_norm_mean' in ema_diag:
                    m_info = f", ‖m‖={ema_diag.get('ema_m_norm_mean', 0):.4f}±{ema_diag.get('ema_m_norm_std', 0):.4f}"
                logger.info(f'  → EMA norms: {b_info}, {b_drift}{m_info} ({n_layers} layers)')
            if config.get('training', {}).get('ema_beta_schedule', 'constant') != 'constant':
                current_beta = compute_ema_beta(config, round_idx)
                round_metrics['ema_beta'] = current_beta
                logger.info(f'  → EMA β(t)={current_beta:.4f}')
        with torch.no_grad():
            current_state = model.state_dict()
            fedavg_a_drifts = []
            ema_a_drifts = []
            for (k, v) in current_state.items():
                if 'lora_A' not in k:
                    continue
                v_cpu = v.cpu()
                if k in prev_fedavg_a:
                    drift = torch.norm(v_cpu - prev_fedavg_a[k], p='fro').item()
                    fedavg_a_drifts.append(drift)
                prev_fedavg_a[k] = v_cpu.clone()
                if ema_state is not None and k in ema_state:
                    if k in prev_ema_a:
                        drift = torch.norm(ema_state[k] - prev_ema_a[k], p='fro').item()
                        ema_a_drifts.append(drift)
                    prev_ema_a[k] = ema_state[k].clone()
            if fedavg_a_drifts:
                round_metrics['a_drift_fedavg_mean'] = float(np.mean(fedavg_a_drifts))
                round_metrics['a_drift_fedavg_max'] = float(np.max(fedavg_a_drifts))
            if ema_a_drifts:
                round_metrics['a_drift_ema_mean'] = float(np.mean(ema_a_drifts))
                round_metrics['a_drift_ema_max'] = float(np.max(ema_a_drifts))
            if ema_state is not None:
                ema_fedavg_gaps = []
                for (k, v) in current_state.items():
                    if 'lora_A' not in k or k not in ema_state:
                        continue
                    gap = torch.norm(ema_state[k] - v.cpu(), p='fro').item()
                    ema_fedavg_gaps.append(gap)
                if ema_fedavg_gaps:
                    round_metrics['epsilon_a_mean'] = float(np.mean(ema_fedavg_gaps))
                    round_metrics['epsilon_a_max'] = float(np.max(ema_fedavg_gaps))
                    round_metrics['epsilon_a_std'] = float(np.std(ema_fedavg_gaps))
        if a_smooth_state:
            with torch.no_grad():
                state = model.state_dict()
                n_smoothed = 0
                for k in a_smooth_state:
                    if k in state:
                        a_smooth_state[k].mul_(a_smooth_decay).add_(state[k], alpha=1.0 - a_smooth_decay)
                        state[k].copy_(a_smooth_state[k])
                        n_smoothed += 1
                model.load_state_dict(state, strict=False)
                if round_idx == start_round:
                    logger.info(f'SAF-standalone: smoothed and fed back {n_smoothed} lora_A params (decay={a_smooth_decay})')
        tracker.log_train_metrics(round_idx, round_metrics)
        display_round = round_idx + 1
        avg_loss = round_metrics.get('avg_loss', 0.0)
        num_clients = round_metrics.get('num_clients', 0)
        recent_losses.append(avg_loss)
        logger.info(f'Round {display_round}: Train Loss = {avg_loss:.4f} | Clients = {num_clients}')
        if 'avg_dirreg_avg_cos_sim' in round_metrics:
            cos_sim = round_metrics['avg_dirreg_avg_cos_sim']
            min_cs = round_metrics.get('worst_dirreg_min_cos_sim', cos_sim)
            max_cs = round_metrics.get('avg_dirreg_max_cos_sim', cos_sim)
            dr_loss = round_metrics.get('avg_dirreg_loss', 0.0)
            logger.info(f'  → DirReg: cos_sim={cos_sim:.4f} [min={min_cs:.4f}, max={max_cs:.4f}], loss={dr_loss:.4f}')
            if display_round % eval_every == 0 or display_round == 1:
                layer_keys = sorted((k for k in round_metrics if k.startswith('avg_dirreg_cs_')))
                if layer_keys:
                    parts = [f"{k.replace('avg_dirreg_cs_', '')}={round_metrics[k]:.3f}" for k in layer_keys]
                    logger.info(f"    layers: {', '.join(parts)}")
        if 'avg_fedprox_loss' in round_metrics:
            prox_loss = round_metrics['avg_fedprox_loss']
            prox_dist = round_metrics.get('avg_fedprox_dist_sq', 0.0)
            logger.info(f'  → FedProx: loss={prox_loss:.4f}, dist_sq={prox_dist:.4f}')
        if hasattr(fl_method, 'get_adaptive_state'):
            adaptive_state = fl_method.get_adaptive_state()
            tracker.log_method_state(round_idx, adaptive_state)
        is_first_round = round_idx == start_round
        is_last_round = round_idx == global_rounds - 1
        is_eval_interval = display_round % eval_every == 0
        should_eval = is_first_round or is_eval_interval or (is_last_round and (not is_eval_interval))
        if should_eval:
            logger.info(f'Evaluating at round {display_round}')
            if dataset_name == 'mnli' and isinstance(val_loader, dict):
                val_matched_metrics = evaluator.evaluate(model, val_loader['matched'], split='matched')
                val_mismatched_metrics = evaluator.evaluate(model, val_loader['mismatched'], split='mismatched')
                matched_acc = val_matched_metrics.get('accuracy', 0)
                mismatched_acc = val_mismatched_metrics.get('accuracy', 0)
                avg_val_loss = (val_matched_metrics.get('loss', 0) + val_mismatched_metrics.get('loss', 0)) / 2
                val_metrics = {'matched_acc': matched_acc, 'mismatched_acc': mismatched_acc, 'accuracy': (matched_acc + mismatched_acc) / 2, 'loss': avg_val_loss}
            else:
                val_metrics = evaluator.evaluate(model, val_loader)
            tracker.log_test_metrics(round_idx, {'split': 'val', **val_metrics})
            if method_logger is not None:
                dataset_name_for_metric = config['dataset']['name'].lower()
                val_acc = get_primary_metric(val_metrics, dataset_name_for_metric)
                val_loss = val_metrics.get('loss')
                val_loss = float(val_loss) if val_loss is not None else avg_loss
                prev_acc = method_logger._last_val_acc
                delta_acc = val_acc - prev_acc if prev_acc is not None else None
                method_logger.log_convergence_metrics(round_num=round_idx, accuracy=val_acc, loss=val_loss, delta_accuracy=delta_acc)
                method_logger._last_val_acc = val_acc
                for rs in method_logger.round_summaries:
                    if rs.round_num == round_idx and rs.val_accuracy is None:
                        rs.val_accuracy = val_acc
            logger.info(f'Round {display_round} | Val: {val_metrics}')
            if personalized_eval and client_test_loaders:
                pers_metrics = run_personalized_eval(model, fl_method, client_test_loaders, evaluator, device, dataset_name, logger, round_idx)
                if pers_metrics:
                    tracker.log_test_metrics(round_idx, {'split': 'personalized', **pers_metrics})
            current_metric = get_primary_metric(val_metrics, dataset_name)
            if current_metric > best_metric_value:
                best_metric_value = current_metric
                best_round = display_round
                best_model_state = {k: v.clone().cpu() for (k, v) in model.state_dict().items()}
                logger.info(f'New best model at round {display_round} with metric={current_metric:.4f}')
            if ema_state is not None:
                original_state = {k: v.clone().cpu() for (k, v) in model.state_dict().items()}
                model.load_state_dict(ema_state, strict=False)
                if dataset_name == 'mnli' and isinstance(val_loader, dict):
                    ema_matched = evaluator.evaluate(model, val_loader['matched'], split='matched')
                    ema_mismatched = evaluator.evaluate(model, val_loader['mismatched'], split='mismatched')
                    ema_m_acc = ema_matched.get('accuracy', 0)
                    ema_mm_acc = ema_mismatched.get('accuracy', 0)
                    ema_avg_loss = (ema_matched.get('loss', 0) + ema_mismatched.get('loss', 0)) / 2
                    ema_metrics = {'matched_acc': ema_m_acc, 'mismatched_acc': ema_mm_acc, 'accuracy': (ema_m_acc + ema_mm_acc) / 2, 'loss': ema_avg_loss}
                else:
                    ema_metrics = evaluator.evaluate(model, val_loader)
                tracker.log_test_metrics(round_idx, {'split': 'ema', **ema_metrics})
                ema_metric = get_primary_metric(ema_metrics, dataset_name)
                logger.info(f'Round {display_round} | EMA: {ema_metrics}')
                if ema_metric > ema_best_metric:
                    ema_best_metric = ema_metric
                    ema_best_round = display_round
                    logger.info(f'New EMA best at round {display_round} with metric={ema_metric:.4f}')
                model.load_state_dict(original_state, strict=False)
        if ema_state is not None and ema_feedback_mode is not None:
            with torch.no_grad():
                state = model.state_dict()
                n_fed = 0
                if ema_feedback_mode == 'adaptive_a':
                    import torch.nn.functional as F
                    w = ema_feedback_var_window
                    if len(recent_losses) >= w:
                        loss_window = recent_losses[-w:]
                        loss_mean = sum(loss_window) / w
                        loss_var = sum(((x - loss_mean) ** 2 for x in loss_window)) / w
                    else:
                        loss_var = 0.0
                    var_range = ema_feedback_var_ceil - ema_feedback_var_floor
                    if var_range > 0:
                        var_gate = max(min((loss_var - ema_feedback_var_floor) / var_range, 1.0), 0.0)
                    else:
                        var_gate = 1.0
                    drift_sum = 0.0
                    gamma_sum = 0.0
                    gamma_b_sum = 0.0
                    n_layers = 0
                    n_fed_b = 0
                    for k in state:
                        if k not in ema_state or 'lora_A' not in k:
                            continue
                        a_flat = state[k].cpu().reshape(1, -1)
                        ema_flat = ema_state[k].reshape(1, -1)
                        cos_sim = F.cosine_similarity(a_flat, ema_flat).item()
                        drift = max(1.0 - cos_sim, 0.0)
                        drift_gamma = min(drift / ema_feedback_threshold, 1.0) if ema_feedback_threshold > 0 else 1.0
                        gamma = drift_gamma * var_gate
                        ema_k_dev = ema_state[k].to(state[k].device)
                        state[k] = (1.0 - gamma) * state[k] + gamma * ema_k_dev
                        n_fed += 1
                        drift_sum += drift
                        gamma_sum += gamma
                        n_layers += 1
                        if ema_feedback_gamma_b > 0 or ema_feedback_gamma_b_ratio > 0:
                            b_key = k.replace('lora_A', 'lora_B')
                            if b_key in state and b_key in ema_state:
                                gamma_b = ema_feedback_gamma_b if ema_feedback_gamma_b > 0 else gamma * ema_feedback_gamma_b_ratio
                                ema_b_dev = ema_state[b_key].to(state[b_key].device)
                                state[b_key] = (1.0 - gamma_b) * state[b_key] + gamma_b * ema_b_dev
                                gamma_b_sum += gamma_b
                                n_fed_b += 1
                    model.load_state_dict(state, strict=False)
                    if n_layers > 0:
                        avg_drift = drift_sum / n_layers
                        avg_gamma = gamma_sum / n_layers
                        avg_gamma_b = gamma_b_sum / n_fed_b if n_fed_b > 0 else 0.0
                        sabf_str = f', avg_gamma_b={avg_gamma_b:.4f} ({n_fed_b} B params)' if n_fed_b > 0 else ''
                        if round_idx == start_round:
                            logger.info(f'Adaptive SAF: {n_fed} lora_A params, drift_threshold={ema_feedback_threshold}, var_floor={ema_feedback_var_floor}, var_ceil={ema_feedback_var_ceil}')
                            if n_fed_b > 0:
                                b_mode = f'constant={ema_feedback_gamma_b}' if ema_feedback_gamma_b > 0 else f'ratio={ema_feedback_gamma_b_ratio}'
                                logger.info(f'  SABF: {n_fed_b} lora_B params with gamma_b {b_mode}')
                        logger.debug(f'  Adaptive SAF R{display_round}: avg_drift={avg_drift:.6f}, var_gate={var_gate:.4f}, avg_gamma={avg_gamma:.4f}, loss_var={loss_var:.4f}{sabf_str}')
                        if tracker.train_metrics:
                            tracker.train_metrics[-1]['saf_avg_drift'] = avg_drift
                            tracker.train_metrics[-1]['saf_var_gate'] = var_gate
                            tracker.train_metrics[-1]['saf_avg_gamma'] = avg_gamma
                            tracker.train_metrics[-1]['saf_loss_var'] = loss_var
                            if n_fed_b > 0:
                                tracker.train_metrics[-1]['saf_avg_gamma_b'] = avg_gamma_b
                elif ema_feedback_mode == 'adaptive_v3_a':
                    import torch.nn.functional as F
                    w = ema_feedback_var_window
                    if len(recent_losses) >= w:
                        loss_window = recent_losses[-w:]
                        loss_mean = sum(loss_window) / w
                        loss_std = math.sqrt(sum(((x - loss_mean) ** 2 for x in loss_window)) / w)
                        loss_cv = loss_std / loss_mean if loss_mean > 1e-08 else 0.0
                    else:
                        loss_cv = 0.0
                    cv_range = ema_feedback_cv_ceil - ema_feedback_cv_floor
                    if cv_range > 0:
                        cv_gate = max(min((loss_cv - ema_feedback_cv_floor) / cv_range, 1.0), 0.0)
                    else:
                        cv_gate = 1.0
                    drift_sum = 0.0
                    gamma_sum = 0.0
                    n_layers = 0
                    for k in list(state.keys()):
                        if k not in ema_state or 'lora_A' not in k:
                            continue
                        a_flat = state[k].cpu().reshape(1, -1)
                        if k in prev_a_state:
                            prev_flat = prev_a_state[k].reshape(1, -1)
                            drift = max(1.0 - F.cosine_similarity(a_flat, prev_flat).item(), 0.0)
                        else:
                            drift = 0.0
                        drift_sum += drift
                        n_layers += 1
                    avg_drift_this_round = drift_sum / n_layers if n_layers > 0 else 0.0
                    if avg_drift_this_round > 0:
                        drift_history.append(avg_drift_this_round)
                    if ema_feedback_drift_scale > 0:
                        drift_ref = ema_feedback_drift_scale
                    elif len(drift_history) >= ema_feedback_var_window:
                        drift_ref = max(sum(drift_history) / len(drift_history) * 3.0, 1e-10)
                    else:
                        drift_ref = float('inf')
                    gamma_sum = 0.0
                    gamma_b_sum = 0.0
                    n_fed_local = 0
                    n_fed_b = 0
                    for k in list(state.keys()):
                        if k not in ema_state or 'lora_A' not in k:
                            continue
                        a_flat = state[k].cpu().reshape(1, -1)
                        if k in prev_a_state:
                            prev_flat = prev_a_state[k].reshape(1, -1)
                            drift = max(1.0 - F.cosine_similarity(a_flat, prev_flat).item(), 0.0)
                        else:
                            drift = 0.0
                        drift_signal = min(drift / drift_ref, 1.0)
                        gamma = drift_signal * cv_gate * ema_feedback_gamma_max
                        ema_k_dev = ema_state[k].to(state[k].device)
                        state[k] = (1.0 - gamma) * state[k] + gamma * ema_k_dev
                        n_fed += 1
                        n_fed_local += 1
                        gamma_sum += gamma
                        if ema_feedback_gamma_b > 0 or ema_feedback_gamma_b_ratio > 0:
                            b_key = k.replace('lora_A', 'lora_B')
                            if b_key in state and b_key in ema_state:
                                gamma_b = ema_feedback_gamma_b if ema_feedback_gamma_b > 0 else gamma * ema_feedback_gamma_b_ratio
                                ema_b_dev = ema_state[b_key].to(state[b_key].device)
                                state[b_key] = (1.0 - gamma_b) * state[b_key] + gamma_b * ema_b_dev
                                gamma_b_sum += gamma_b
                                n_fed_b += 1
                    for k in state:
                        if 'lora_A' in k:
                            prev_a_state[k] = state[k].clone().cpu()
                    model.load_state_dict(state, strict=False)
                    if n_layers > 0:
                        avg_gamma = gamma_sum / n_fed_local if n_fed_local > 0 else 0.0
                        avg_gamma_b = gamma_b_sum / n_fed_b if n_fed_b > 0 else 0.0
                        sabf_str = f', avg_gamma_b={avg_gamma_b:.4f}' if n_fed_b > 0 else ''
                        if round_idx == start_round:
                            mode_str = 'auto-cal' if ema_feedback_drift_scale <= 0 else f'fixed={ema_feedback_drift_scale}'
                            logger.info(f'SAF v3: {n_fed_local} lora_A params, cv=[{ema_feedback_cv_floor},{ema_feedback_cv_ceil}], drift={mode_str}, gamma_max={ema_feedback_gamma_max}')
                            if n_fed_b > 0:
                                b_mode = f'constant={ema_feedback_gamma_b}' if ema_feedback_gamma_b > 0 else f'ratio={ema_feedback_gamma_b_ratio}'
                                logger.info(f'  SABF: {n_fed_b} lora_B params with gamma_b {b_mode}')
                        logger.debug(f'  SAF v3 R{display_round}: avg_drift={avg_drift_this_round:.6f}, drift_ref={drift_ref:.6f}, cv_gate={cv_gate:.4f}, cv={loss_cv:.4f}, avg_gamma={avg_gamma:.4f}{sabf_str}')
                        if tracker.train_metrics:
                            tracker.train_metrics[-1]['saf_avg_drift'] = avg_drift_this_round
                            tracker.train_metrics[-1]['saf_drift_ref'] = drift_ref if drift_ref != float('inf') else 0.0
                            tracker.train_metrics[-1]['saf_cv_gate'] = cv_gate
                            tracker.train_metrics[-1]['saf_loss_cv'] = loss_cv
                            tracker.train_metrics[-1]['saf_avg_gamma'] = avg_gamma
                            if n_fed_b > 0:
                                tracker.train_metrics[-1]['saf_avg_gamma_b'] = avg_gamma_b
                elif ema_feedback_mode == 'dasf_a':
                    import torch.nn.functional as F
                    w = ema_feedback_var_window
                    if len(recent_losses) >= w:
                        loss_window = recent_losses[-w:]
                        loss_mean = sum(loss_window) / w
                        loss_std = math.sqrt(sum(((x - loss_mean) ** 2 for x in loss_window)) / w)
                        loss_cv = loss_std / loss_mean if loss_mean > 1e-08 else 0.0
                    else:
                        loss_cv = 0.0
                    cv_range = ema_feedback_cv_ceil - ema_feedback_cv_floor
                    if cv_range > 0:
                        cv_gate = max(min((loss_cv - ema_feedback_cv_floor) / cv_range, 1.0), 0.0)
                    else:
                        cv_gate = 1.0
                    method_params = config.get('method', {}).get('params', {})
                    lora_scaling = method_params.get('alpha', 16) / method_params.get('rank', 8)
                    w0_map = {}
                    for sk in state:
                        if 'base_layer.weight' in sk:
                            prefix = sk.split('base_layer.weight')[0]
                            w0_map[prefix] = state[sk]
                    drift_sum = 0.0
                    n_layers = 0
                    for k in list(state.keys()):
                        if k not in ema_state or 'lora_A' not in k:
                            continue
                        b_key = k.replace('lora_A', 'lora_B')
                        prefix = k.split('lora_A')[0]
                        if b_key not in state or prefix not in w0_map:
                            continue
                        W0 = w0_map[prefix].cpu().float()
                        A_t = state[k].cpu().float()
                        B_t = state[b_key].cpu().float()
                        V_t = W0 + lora_scaling * (B_t @ A_t)
                        V_t_dir = V_t / (V_t.norm(dim=1, keepdim=True) + 1e-08)
                        if k in prev_a_state and b_key in prev_b_state:
                            A_prev = prev_a_state[k].float()
                            B_prev = prev_b_state[b_key].float()
                            V_prev = W0 + lora_scaling * (B_prev @ A_prev)
                            V_prev_dir = V_prev / (V_prev.norm(dim=1, keepdim=True) + 1e-08)
                            vt_flat = V_t_dir.reshape(1, -1)
                            vp_flat = V_prev_dir.reshape(1, -1)
                            drift = max(1.0 - F.cosine_similarity(vt_flat, vp_flat).item(), 0.0)
                        else:
                            drift = 0.0
                        drift_sum += drift
                        n_layers += 1
                    avg_drift_this_round = drift_sum / n_layers if n_layers > 0 else 0.0
                    if avg_drift_this_round > 0:
                        drift_history.append(avg_drift_this_round)
                    if ema_feedback_drift_scale > 0:
                        drift_ref = ema_feedback_drift_scale
                    elif len(drift_history) >= ema_feedback_var_window:
                        drift_ref = max(sum(drift_history) / len(drift_history) * 3.0, 1e-10)
                    else:
                        drift_ref = float('inf')
                    gamma_sum = 0.0
                    gamma_b_sum = 0.0
                    n_fed_local = 0
                    n_fed_b = 0
                    for k in list(state.keys()):
                        if k not in ema_state or 'lora_A' not in k:
                            continue
                        b_key = k.replace('lora_A', 'lora_B')
                        prefix = k.split('lora_A')[0]
                        if b_key not in state or prefix not in w0_map:
                            continue
                        W0 = w0_map[prefix].cpu().float()
                        A_t = state[k].cpu().float()
                        B_t = state[b_key].cpu().float()
                        V_t = W0 + lora_scaling * (B_t @ A_t)
                        V_t_dir = V_t / (V_t.norm(dim=1, keepdim=True) + 1e-08)
                        if k in prev_a_state and b_key in prev_b_state:
                            A_prev = prev_a_state[k].float()
                            B_prev = prev_b_state[b_key].float()
                            V_prev = W0 + lora_scaling * (B_prev @ A_prev)
                            V_prev_dir = V_prev / (V_prev.norm(dim=1, keepdim=True) + 1e-08)
                            vt_flat = V_t_dir.reshape(1, -1)
                            vp_flat = V_prev_dir.reshape(1, -1)
                            drift = max(1.0 - F.cosine_similarity(vt_flat, vp_flat).item(), 0.0)
                        else:
                            drift = 0.0
                        drift_signal = min(drift / drift_ref, 1.0)
                        gamma = drift_signal * cv_gate * ema_feedback_gamma_max
                        ema_k_dev = ema_state[k].to(state[k].device)
                        state[k] = (1.0 - gamma) * state[k] + gamma * ema_k_dev
                        n_fed += 1
                        n_fed_local += 1
                        gamma_sum += gamma
                        if (ema_feedback_gamma_b > 0 or ema_feedback_gamma_b_ratio > 0) and b_key in ema_state:
                            gamma_b = ema_feedback_gamma_b if ema_feedback_gamma_b > 0 else gamma * ema_feedback_gamma_b_ratio
                            ema_b_dev = ema_state[b_key].to(state[b_key].device)
                            state[b_key] = (1.0 - gamma_b) * state[b_key] + gamma_b * ema_b_dev
                            gamma_b_sum += gamma_b
                            n_fed_b += 1
                    for k in state:
                        if 'lora_A' in k:
                            prev_a_state[k] = state[k].clone().cpu()
                        elif 'lora_B' in k:
                            prev_b_state[k] = state[k].clone().cpu()
                    model.load_state_dict(state, strict=False)
                    if n_layers > 0:
                        avg_gamma = gamma_sum / n_fed_local if n_fed_local > 0 else 0.0
                        avg_gamma_b = gamma_b_sum / n_fed_b if n_fed_b > 0 else 0.0
                        sabf_str = f', avg_gamma_b={avg_gamma_b:.4f}' if n_fed_b > 0 else ''
                        if round_idx == start_round:
                            mode_str = 'auto-cal' if ema_feedback_drift_scale <= 0 else f'fixed={ema_feedback_drift_scale}'
                            logger.info(f'DA-SF: {n_fed_local} lora_A params, cv=[{ema_feedback_cv_floor},{ema_feedback_cv_ceil}], drift={mode_str}, gamma_max={ema_feedback_gamma_max}, lora_scaling={lora_scaling}')
                            if n_fed_b > 0:
                                b_mode = f'constant={ema_feedback_gamma_b}' if ema_feedback_gamma_b > 0 else f'ratio={ema_feedback_gamma_b_ratio}'
                                logger.info(f'  SABF: {n_fed_b} lora_B params with gamma_b {b_mode}')
                        logger.debug(f'  DA-SF R{display_round}: dir_drift={avg_drift_this_round:.6f}, drift_ref={drift_ref:.6f}, cv_gate={cv_gate:.4f}, cv={loss_cv:.4f}, avg_gamma={avg_gamma:.4f}{sabf_str}')
                        if tracker.train_metrics:
                            tracker.train_metrics[-1]['saf_avg_drift'] = avg_drift_this_round
                            tracker.train_metrics[-1]['saf_drift_ref'] = drift_ref if drift_ref != float('inf') else 0.0
                            tracker.train_metrics[-1]['saf_cv_gate'] = cv_gate
                            tracker.train_metrics[-1]['saf_loss_cv'] = loss_cv
                            tracker.train_metrics[-1]['saf_avg_gamma'] = avg_gamma
                            if n_fed_b > 0:
                                tracker.train_metrics[-1]['saf_avg_gamma_b'] = avg_gamma_b
                else:
                    n_fed_b = 0
                    for k in state:
                        if k not in ema_state:
                            continue
                        feed = ema_feedback_mode == 'full' or (ema_feedback_mode == 'a_only' and 'lora_A' in k) or (ema_feedback_mode == 'b_only' and 'lora_B' in k)
                        if feed:
                            state[k] = ema_state[k].clone().to(state[k].device)
                            n_fed += 1
                        elif ema_feedback_mode == 'a_only' and 'lora_B' in k and (ema_feedback_gamma_b > 0 or ema_feedback_gamma_b_ratio > 0):
                            gamma_b = ema_feedback_gamma_b if ema_feedback_gamma_b > 0 else ema_feedback_gamma_b_ratio
                            ema_k_dev = ema_state[k].to(state[k].device)
                            state[k] = (1.0 - gamma_b) * state[k] + gamma_b * ema_k_dev
                            n_fed_b += 1
                    model.load_state_dict(state, strict=False)
                    if round_idx == start_round:
                        logger.info(f"EMA feedback='{ema_feedback_mode}': {n_fed} params replaced in training model each round")
                        if n_fed_b > 0:
                            gb_val = ema_feedback_gamma_b if ema_feedback_gamma_b > 0 else ema_feedback_gamma_b_ratio
                            logger.info(f'  SABF: {n_fed_b} lora_B params blended with gamma_b={gb_val}')
        round_time = time.time() - round_start_time
        round_times.append(round_time)
        gpu_mem = get_gpu_memory(device)
        logger.debug(f'Round {display_round} in {round_time:.2f}s | {gpu_mem}')
    total_training_time = time.time() - training_start_time
    avg_round_time = sum(round_times) / len(round_times) if round_times else 0
    min_round_time = min(round_times) if round_times else 0
    max_round_time = max(round_times) if round_times else 0
    if a_smooth_state:
        logger.info('=' * 60)
        logger.info('SAF-STANDALONE SUMMARY')
        logger.info(f'A-smooth decay: {a_smooth_decay}')
        logger.info(f'A params:       {len(a_smooth_state)}')
        logger.info(f'Best metric:    {best_metric_value:.4f} at round {best_round}')
        logger.info(f'(FedAvg model with A-smoothing feedback, no global EMA)')
        logger.info('=' * 60)
    if ema_state is not None:
        logger.info('=' * 60)
        logger.info('EMA SUMMARY')
        if ema_component_aware:
            logger.info(f'EMA component-aware: m={ema_decay_m}, a={ema_decay_a}, b={ema_decay_b}')
        else:
            logger.info(f'EMA decay:      {ema_decay}')
        if ema_feedback_mode is not None:
            logger.info(f"EMA feedback:   '{ema_feedback_mode}'")
            if ema_feedback_mode == 'adaptive_a':
                logger.info(f'  drift_thresh: {ema_feedback_threshold}')
                logger.info(f'  var_floor:    {ema_feedback_var_floor}')
                logger.info(f'  var_ceil:     {ema_feedback_var_ceil}')
                logger.info(f'  var_window:   {ema_feedback_var_window}')
            elif ema_feedback_mode in ('adaptive_v3_a', 'dasf_a'):
                logger.info(f'  cv_floor:     {ema_feedback_cv_floor}')
                logger.info(f'  cv_ceil:      {ema_feedback_cv_ceil}')
                logger.info(f'  drift_scale:  {ema_feedback_drift_scale}')
                logger.info(f'  gamma_max:    {ema_feedback_gamma_max}')
                logger.info(f'  window:       {ema_feedback_var_window}')
            if ema_feedback_gamma_b > 0:
                logger.info(f'  SABF gamma_b: {ema_feedback_gamma_b} (constant)')
            elif ema_feedback_gamma_b_ratio > 0:
                logger.info(f'  SABF ratio:   {ema_feedback_gamma_b_ratio} (proportional)')
        logger.info(f'EMA best:       {ema_best_metric:.4f} at round {ema_best_round}')
        logger.info(f'Raw best:       {best_metric_value:.4f} at round {best_round}')
        logger.info(f'EMA vs Raw:     {ema_best_metric - best_metric_value:+.4f}')
        logger.info('=' * 60)
    logger.info('=' * 60)
    logger.info('TIMING SUMMARY')
    logger.info('=' * 60)
    total_mins = total_training_time / 60
    logger.info(f'Total training time: {total_training_time:.2f}s ({total_mins:.2f} min)')
    logger.info(f'Average round time:  {avg_round_time:.2f}s')
    logger.info(f'Min round time:      {min_round_time:.2f}s')
    logger.info(f'Max round time:      {max_round_time:.2f}s')
    logger.info(f'Total rounds:        {len(round_times)}')
    logger.info('=' * 60)
    if not config.get('no_save', True):
        if best_model_state is not None:
            best_path = output_dir / 'models' / f'{method_name}_best.pth'
            torch.save({'round': best_round - 1, 'display_round': best_round, 'model_state_dict': best_model_state, 'config': config, 'method': method_name, 'best_metric': best_metric_value}, best_path)
            logger.info(f'Best model saved: {best_path} (round {best_round}, metric={best_metric_value:.4f})')
        final_path = output_dir / 'models' / f'{method_name}_final.pth'
        torch.save({'round': global_rounds - 1, 'display_round': global_rounds, 'model_state_dict': model.state_dict(), 'config': config, 'method': method_name}, final_path)
        logger.info(f'Final model saved: {final_path}')
    else:
        logger.info('Model saving disabled (use --save to enable)')
    final_metrics = tracker.get_final_metrics()
    avg_test = tracker.get_average_metrics(exclude_first_round=True)
    if avg_test:
        final_metrics['avg_test'] = avg_test
    local_epochs = config['method']['params'].get('local_epochs', 1)
    client_fraction = config['training']['client_fraction']
    clients_per_round = int(len(client_indices) * client_fraction)
    avg_samples_per_client = client_data_stats['samples_per_client']['mean']
    flops_per_sample = 6 * trainable_params
    flops_per_client_per_round = flops_per_sample * avg_samples_per_client * local_epochs
    flops_per_round = flops_per_client_per_round * clients_per_round
    per_round_flops = [flops_per_round] * len(round_times)
    total_flops = flops_per_round * len(round_times)
    effective_flops_per_sample = None
    effective_total_flops = None
    if effective_trainable_params is not None:
        effective_flops_per_sample = 4 * trainable_params + 2 * effective_trainable_params
        effective_flops_per_client_per_round = effective_flops_per_sample * avg_samples_per_client * local_epochs
        effective_total_flops = effective_flops_per_client_per_round * clients_per_round * len(round_times)
    avg_client_time_per_round = avg_round_time / max(clients_per_round, 1)
    final_metrics['timing'] = {'total_training_time_seconds': total_training_time, 'total_training_time_minutes': total_training_time / 60, 'avg_round_time_seconds': avg_round_time, 'total_rounds': len(round_times)}
    final_metrics['model'] = model_params_metrics
    final_metrics['data'] = client_data_stats
    final_metrics['best_model'] = {'best_round': best_round, 'best_metric_value': best_metric_value, 'was_saved': best_model_state is not None}
    final_metrics['computation'] = {'flops_per_sample': flops_per_sample, 'avg_flops_per_client_per_round': flops_per_client_per_round, 'flops_per_round': flops_per_round, 'total_flops': total_flops, 'avg_time_per_client_per_round_seconds': avg_client_time_per_round, 'clients_per_round': clients_per_round, 'local_epochs': local_epochs}
    if effective_flops_per_sample is not None:
        final_metrics['computation']['effective_flops_per_sample'] = effective_flops_per_sample
        final_metrics['computation']['effective_total_flops'] = effective_total_flops
    if a_smooth_state:
        final_metrics['a_smooth_summary'] = {'mode': 'a_smooth', 'decay': a_smooth_decay, 'n_params': len(a_smooth_state), 'note': 'standalone A-smoothing, no global EMA'}
    if ema_feedback_mode in ('adaptive_a', 'adaptive_v3_a', 'dasf_a'):
        saf_gammas = [m.get('saf_avg_gamma', 0.0) for m in tracker.train_metrics if 'saf_avg_gamma' in m]
        saf_drifts = [m.get('saf_avg_drift', 0.0) for m in tracker.train_metrics if 'saf_avg_drift' in m]
        saf_gates = [m.get('saf_var_gate', m.get('saf_cv_gate', 0.0)) for m in tracker.train_metrics if 'saf_avg_gamma' in m]
        if saf_gammas:
            final_metrics['saf_summary'] = {'mode': ema_feedback_mode, 'avg_gamma': sum(saf_gammas) / len(saf_gammas), 'max_gamma': max(saf_gammas), 'avg_gate': sum(saf_gates) / len(saf_gates), 'avg_drift': sum(saf_drifts) / len(saf_drifts) if saf_drifts else 0.0, 'active_rounds': sum((1 for g in saf_gammas if g > 0.001)), 'total_rounds': len(saf_gammas)}
            if ema_feedback_mode in ('adaptive_v3_a', 'dasf_a'):
                saf_cvs = [m.get('saf_loss_cv', 0.0) for m in tracker.train_metrics if 'saf_loss_cv' in m]
                saf_refs = [m.get('saf_drift_ref', 0.0) for m in tracker.train_metrics if m.get('saf_drift_ref', 0) > 0]
                if saf_cvs:
                    final_metrics['saf_summary']['avg_loss_cv'] = sum(saf_cvs) / len(saf_cvs)
                    final_metrics['saf_summary']['max_loss_cv'] = max(saf_cvs)
                if saf_refs:
                    final_metrics['saf_summary']['final_drift_ref'] = saf_refs[-1] if saf_refs else 0.0
            saf_gamma_bs = [m.get('saf_avg_gamma_b', 0.0) for m in tracker.train_metrics if 'saf_avg_gamma_b' in m]
            if saf_gamma_bs:
                final_metrics['saf_summary']['gamma_b_constant'] = ema_feedback_gamma_b
                final_metrics['saf_summary']['gamma_b_ratio'] = ema_feedback_gamma_b_ratio
                final_metrics['saf_summary']['avg_gamma_b'] = sum(saf_gamma_bs) / len(saf_gamma_bs)
                final_metrics['saf_summary']['max_gamma_b'] = max(saf_gamma_bs)
    dirreg_lambda = config.get('training', {}).get('dirreg_lambda', 0.0)
    if dirreg_lambda > 0 and tracker.train_metrics:
        dr_cos_sims = [m.get('avg_dirreg_avg_cos_sim') for m in tracker.train_metrics if m.get('avg_dirreg_avg_cos_sim') is not None]
        dr_losses = [m.get('avg_dirreg_loss') for m in tracker.train_metrics if m.get('avg_dirreg_loss') is not None]
        dr_min_cs = [m.get('worst_dirreg_min_cos_sim', m.get('avg_dirreg_min_cos_sim')) for m in tracker.train_metrics if m.get('avg_dirreg_avg_cos_sim') is not None]
        dr_min_cs = [v for v in dr_min_cs if v is not None]
        if dr_cos_sims:
            n_active = len(dr_cos_sims)
            final_metrics['dirreg_summary'] = {'lambda': dirreg_lambda, 'detach_b': config.get('training', {}).get('dirreg_detach_b', False), 'n_active_rounds': n_active, 'avg_cos_sim': sum(dr_cos_sims) / n_active, 'min_cos_sim': min(dr_cos_sims), 'max_cos_sim': max(dr_cos_sims), 'first_cos_sim': dr_cos_sims[0], 'last_cos_sim': dr_cos_sims[-1], 'cos_sim_trend': dr_cos_sims[-1] - dr_cos_sims[0], 'avg_loss': sum(dr_losses) / len(dr_losses) if dr_losses else 0.0, 'max_loss': max(dr_losses) if dr_losses else 0.0, 'worst_min_cos_sim': min(dr_min_cs) if dr_min_cs else 0.0}
            layer_keys = [k for k in tracker.train_metrics[-1].keys() if k.startswith('avg_dirreg_cs_')]
            if layer_keys:
                per_layer = {}
                for lk in sorted(layer_keys):
                    vals = [m.get(lk) for m in tracker.train_metrics if m.get(lk) is not None]
                    if vals:
                        short = lk.replace('avg_dirreg_cs_', '')
                        per_layer[short] = {'avg': sum(vals) / len(vals), 'first': vals[0], 'last': vals[-1], 'min': min(vals)}
                final_metrics['dirreg_summary']['per_layer'] = per_layer
            logger.info(f"DirReg summary: avg_cs={final_metrics['dirreg_summary']['avg_cos_sim']:.4f}, trend={final_metrics['dirreg_summary']['cos_sim_trend']:+.4f}, worst_min={final_metrics['dirreg_summary']['worst_min_cos_sim']:.4f}")
    fedprox_mu = config.get('training', {}).get('fedprox_mu', 0.0)
    if fedprox_mu > 0 and tracker.train_metrics:
        fp_losses = [m.get('avg_fedprox_loss') for m in tracker.train_metrics if m.get('avg_fedprox_loss') is not None]
        fp_dists = [m.get('avg_fedprox_dist_sq') for m in tracker.train_metrics if m.get('avg_fedprox_dist_sq') is not None]
        if fp_losses:
            n_active = len(fp_losses)
            final_metrics['fedprox_summary'] = {'mu': fedprox_mu, 'n_active_rounds': n_active, 'avg_loss': sum(fp_losses) / n_active, 'max_loss': max(fp_losses), 'first_loss': fp_losses[0], 'last_loss': fp_losses[-1], 'avg_dist_sq': sum(fp_dists) / len(fp_dists) if fp_dists else 0.0, 'max_dist_sq': max(fp_dists) if fp_dists else 0.0, 'first_dist_sq': fp_dists[0] if fp_dists else 0.0, 'last_dist_sq': fp_dists[-1] if fp_dists else 0.0}
            logger.info(f"FedProx summary: avg_loss={final_metrics['fedprox_summary']['avg_loss']:.4f}, avg_dist_sq={final_metrics['fedprox_summary']['avg_dist_sq']:.4f}")
    detailed_timing = {'total_training_time_seconds': total_training_time, 'avg_round_time_seconds': avg_round_time, 'min_round_time_seconds': min_round_time, 'max_round_time_seconds': max_round_time, 'total_rounds': len(round_times), 'round_times': round_times}
    timing_path = output_dir / 'metrics' / f'{method_name}_timing.json'
    with open(timing_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_timing, f, indent=2)
    logger.info(f'Timing metrics saved: {timing_path}')
    detailed_flops = {'flops_per_sample': flops_per_sample, 'avg_flops_per_client_per_round': flops_per_client_per_round, 'flops_per_round': flops_per_round, 'total_flops': total_flops, 'clients_per_round': clients_per_round, 'local_epochs': local_epochs, 'per_round_flops': per_round_flops}
    flops_path = output_dir / 'metrics' / f'{method_name}_flops.json'
    with open(flops_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_flops, f, indent=2)
    logger.info(f'FLOPs metrics saved: {flops_path}')
    if tracker.train_metrics:
        metrics_for_plot = {}
        for key in tracker.train_metrics[0].keys():
            if key not in ['round', 'timestamp', 'num_clients']:
                values = [m.get(key) for m in tracker.train_metrics if m.get(key) is not None]
                if values and isinstance(values[0], (int, float)):
                    metrics_for_plot[key] = values
        if metrics_for_plot:
            plot_training_curves(metrics_for_plot, output_dir / 'plots' / f'{method_name}_training_curves.png')
    if tracker.test_metrics:
        skip_fields = ['round', 'timestamp', 'split', 'num_clients']
        metric_groups = {'accuracy': {}, 'f1': {}, 'mcc': {}, 'correlation': {}, 'loss': {}}
        for key in tracker.test_metrics[0].keys():
            if key in skip_fields:
                continue
            values = [m.get(key) for m in tracker.test_metrics if m.get(key) is not None]
            if not values or not isinstance(values[0], (int, float)):
                continue
            key_lower = key.lower()
            if 'loss' in key_lower:
                metric_groups['loss'][key] = values
            elif 'f1' in key_lower:
                metric_groups['f1'][key] = values
            elif 'mcc' in key_lower:
                metric_groups['mcc'][key] = values
            elif 'pearson' in key_lower or 'spearman' in key_lower:
                metric_groups['correlation'][key] = values
            elif 'acc' in key_lower or 'matched' in key_lower:
                metric_groups['accuracy'][key] = values
        group_titles = {'accuracy': 'Test Accuracy', 'f1': 'Test F1 Score', 'mcc': 'Test MCC', 'correlation': 'Test Correlation', 'loss': 'Test Loss'}
        for (group_name, metrics) in metric_groups.items():
            if metrics:
                plot_training_curves(metrics, output_dir / 'plots' / f'{method_name}_test_{group_name}.png', title=group_titles[group_name])
    tracker.save_metrics(prefix=f'{method_name}_')
    if method_logger is not None:
        method_logger.save(prefix=f'{method_name}_')
        logger.info(f"MethodLogger saved to {output_dir / 'logs'}")
        similarity_summary = method_logger.get_similarity_summary()
        if similarity_summary:
            summary_path = output_dir / 'metrics' / f'{method_name}_similarity_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(similarity_summary, f, indent=2)
            logger.info(f"Similarity summary: {similarity_summary.get('ordering', 'N/A')}")
            logger.info(f"FedSA-LoRA pattern (A>B): {similarity_summary.get('fedsa_pattern_confirmed', 'N/A')}")
            logger.info(f'Similarity summary saved to {summary_path}')
        experiment_report = method_logger.get_experiment_report()
        if experiment_report:
            report_path = output_dir / 'metrics' / f'{method_name}_experiment_report.json'
            with open(report_path, 'w') as f:
                json.dump(experiment_report, f, indent=2)
            logger.info(f'Experiment report saved to {report_path}')
            health = experiment_report.get('health', {})
            logger.info(f"Experiment health: {health.get('overall', 'unknown')}")
    logger.info(f'Training completed for {method_name}')
    logger.info(f'Final metrics: {final_metrics}')
    return final_metrics

def evaluate_model(config, method_name, model, test_dataset, device, checkpoint_path, logger, answer_info=None):
    logger.info(f'Loading checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f'Evaluating {method_name}')
    evaluator = get_evaluator(config['dataset']['name'], device)
    if answer_info is not None and hasattr(evaluator, 'set_answer_info'):
        evaluator.set_answer_info(answer_info)
    if isinstance(test_dataset, torch.utils.data.DataLoader):
        test_loader = test_dataset
    else:
        batch_size = config['training'].get('batch_size', 32)
        max_seq_length = config['dataset'].get('max_length', 128)
        test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False, max_seq_length=max_seq_length)
    metrics = evaluator.evaluate(model, test_loader)
    logger.info(f'Evaluation metrics: {metrics}')
    return metrics

def run_single_method(args, config, output_dir, logger):
    method_name = config['method']['name']
    set_random_seed(config['experiment']['seed'])
    device = setup_device(config['experiment']['gpu'])
    use_presplit = not getattr(args, 'no_presplit', False)
    (train_dataset, val_dataset, test_dataset, client_data, answer_info) = load_data(config, logger, use_presplit=use_presplit)
    dataset_name = config['dataset']['name'].lower()
    is_regression = dataset_name == 'stsb'
    client_label_dist = compute_client_label_distribution(train_dataset, client_data, is_regression=is_regression)
    plot_client_distribution(client_label_dist, output_dir / 'plots' / 'client_distribution.png')
    (model, max_seq_length) = load_model(config, device, logger)
    if args.eval_only:
        metrics = evaluate_model(config, method_name, model, test_dataset, device, args.checkpoint, logger, answer_info=answer_info)
        with open(output_dir / 'metrics' / f'{method_name}_eval.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
    else:
        final_metrics = train_federated(config, method_name, model, train_dataset, client_data, val_dataset, device, output_dir, logger, args.resume, max_seq_length=max_seq_length, answer_info=answer_info)
        with open(output_dir / 'metrics' / f'{method_name}_final.json', 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=2)

def run_method_comparison(args, config, output_dir, logger):
    methods = [m.strip() for m in args.methods.split(',')]
    logger.info(f'Running comparison: {methods}')
    set_random_seed(config['experiment']['seed'])
    device = setup_device(config['experiment']['gpu'])
    use_presplit = not getattr(args, 'no_presplit', False)
    (train_dataset, val_dataset, _test_dataset, client_data, answer_info) = load_data(config, logger, use_presplit=use_presplit)
    dataset_name = config['dataset']['name'].lower()
    is_regression = dataset_name == 'stsb'
    client_label_dist = compute_client_label_distribution(train_dataset, client_data, is_regression=is_regression)
    plot_client_distribution(client_label_dist, output_dir / 'plots' / 'client_distribution.png')
    all_results = {}
    for method_name in methods:
        logger.info(f"\n{'=' * 80}")
        logger.info(f'Training method: {method_name}')
        logger.info(f"{'=' * 80}\n")
        (model, max_seq_length) = load_model(config, device, logger)
        final_metrics = train_federated(config, method_name, model, train_dataset, client_data, val_dataset, device, output_dir, logger, max_seq_length=max_seq_length, answer_info=answer_info)
        all_results[method_name] = final_metrics
    with open(output_dir / 'metrics' / 'comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    plot_method_comparison(all_results, output_dir / 'plots' / 'method_comparison.png')
    logger.info(f"\n{'=' * 80}")
    logger.info('Comparison Results:')
    logger.info(f"{'=' * 80}")
    for (method, metrics) in all_results.items():
        logger.info(f'{method}: {metrics}')

def main():
    args = parse_args()
    config_obj = load_config(args.config)
    if args.gpu is not None:
        config_obj.experiment.gpu = args.gpu
    if args.seed is not None:
        config_obj.experiment.seed = args.seed
    config_save_model = getattr(config_obj.experiment, 'save_model', False)
    config_obj.no_save = not (args.save or config_save_model)
    if args.alpha is not None:
        config_obj.dataset.alpha = args.alpha
    if args.method is not None:
        config_obj.method.name = args.method
    validate_config(config_obj)
    config = config_obj.to_dict()
    method_name = config['method']['name']
    if args.methods:
        methods_to_validate = [m.strip() for m in args.methods.split(',')]
    else:
        methods_to_validate = [method_name]
    for method_to_check in methods_to_validate:
        if not method_exists(method_to_check):
            try:
                get_method(method_to_check, config, torch.device('cpu'))
            except ValueError as e:
                print(f'ERROR: {e}', file=sys.stderr)
                sys.exit(1)
    output_dir = setup_output_directory(args, config)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(name='fedfl', log_dir=output_dir / 'logs', log_file='experiment.log', level=log_level)
    with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    method_name = config['method']['name']
    dataset_name = config['dataset']['name']
    alpha_value = config['dataset']['alpha']
    experiment_title = f'{method_name} | {dataset_name} | α={alpha_value}'
    sep = '=' * 80
    logger.info(sep)
    logger.info(experiment_title)
    logger.info(sep)
    logger.info('Config: %s', args.config)
    logger.info('Output: %s', output_dir)
    logger.info('GPU: %s', config['experiment']['gpu'])
    logger.info('Seed: %s', config['experiment']['seed'])
    logger.info('%s\n', sep)
    try:
        if args.methods:
            run_method_comparison(args, config, output_dir, logger)
        else:
            run_single_method(args, config, output_dir, logger)
        logger.info('\n%s', sep)
        logger.info('%s completed successfully!', experiment_title)
        logger.info('Results saved to: %s', output_dir)
        logger.info('%s\n', sep)
    except Exception as e:
        logger.error('%s failed: %s', experiment_title, e, exc_info=True)
        sys.exit(1)
if __name__ == '__main__':
    main()
