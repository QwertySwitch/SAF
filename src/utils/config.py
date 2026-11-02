import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict

@dataclass
class ExperimentConfig:
    name: Any
    seed: Any = 42
    gpu: Any = 0
    output_dir: Any = None
    type: Any = 'main'
    save_model: Any = False

@dataclass
class DatasetConfig:
    name: Any
    split_method: Any = 'dirichlet'
    alpha: Any = 0.5
    num_clients: Any = 100
    max_length: Any = None
    categories: Any = None
    domains: Any = None
    split_seed: Any = 42

@dataclass
class ModelConfig:
    name: Any
    pretrained: Any = True
    checkpoint: Any = ''
    total_params: Any = None
    image_size: Any = None
    patch_size: Any = None
    gradient_checkpointing: Any = False

@dataclass
class MethodConfig:
    name: Any
    params: Any = field(default_factory=dict)

@dataclass
class TrainingConfig:
    global_rounds: Any = 50
    client_fraction: Any = 0.1
    eval_every: Any = 5
    save_every: Any = 10
    ema_decay: Any = 0.0
    ema_decay_m: Any = None
    ema_decay_a: Any = None
    ema_decay_b: Any = None
    ema_beta_schedule: Any = 'constant'
    ema_beta_min: Any = 0.5
    ema_beta_max: Any = 0.9
    ema_beta_power: Any = 1.0
    ema_feedback: Any = None
    ema_feedback_threshold: Any = 0.05
    ema_feedback_var_floor: Any = 1.5
    ema_feedback_var_ceil: Any = 3.0
    ema_feedback_var_window: Any = 5
    ema_feedback_cv_floor: Any = 0.15
    ema_feedback_cv_ceil: Any = 0.5
    ema_feedback_drift_scale: Any = 0
    ema_feedback_gamma_max: Any = 0.3
    ema_feedback_gamma_b: Any = 0.0
    ema_feedback_gamma_b_ratio: Any = 0.0
    a_smooth_decay: Any = 0.9
    dirreg_lambda: Any = 0.0
    dirreg_detach_b: Any = False
    fedprox_mu: Any = 0.0

@dataclass
class OptimizationConfig:
    optimizer: Any = 'adamw'
    lr_schedule: Any = 'constant'
    lr_min_factor: Any = 0.1
    weight_decay: Any = 0.01
    momentum: Any = None

@dataclass
class EvaluationConfig:
    metrics: Any = field(default_factory=list)
    test_batch_size: Any = 64
    personalized_eval: Any = False

@dataclass
class Config:
    experiment: Any
    dataset: Any
    model: Any
    method: Any
    training: Any
    optimization: Any
    evaluation: Any
    no_save: Any = False

    def to_dict(self):

        def filter_none(d):
            return {k: v for (k, v) in d.items() if v is not None}
        result = {'experiment': filter_none(asdict(self.experiment)), 'dataset': filter_none(asdict(self.dataset)), 'model': filter_none(asdict(self.model)), 'method': asdict(self.method), 'training': asdict(self.training), 'optimization': filter_none(asdict(self.optimization)), 'evaluation': asdict(self.evaluation)}
        result['no_save'] = self.no_save
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    if not raw_config:
        raise ValueError(f'Empty config file: {config_path}')
    try:
        experiment = ExperimentConfig(**raw_config.get('experiment', {}))
        dataset = DatasetConfig(**raw_config.get('dataset', {}))
        model = ModelConfig(**raw_config.get('model', {}))
        method = MethodConfig(**raw_config.get('method', {}))
        training = TrainingConfig(**raw_config.get('training', {}))
        optimization = OptimizationConfig(**raw_config.get('optimization', {}))
        evaluation = EvaluationConfig(**raw_config.get('evaluation', {}))
        config = Config(experiment=experiment, dataset=dataset, model=model, method=method, training=training, optimization=optimization, evaluation=evaluation, no_save=raw_config.get('no_save', False))
        validate_config(config)
        return config
    except TypeError as e:
        raise ValueError(f'Invalid config format: {e}') from e

def validate_config(config):
    if config.dataset.split_method not in ['dirichlet', 'iid']:
        raise ValueError(f'Invalid split_method: {config.dataset.split_method}')
    if config.dataset.alpha <= 0:
        raise ValueError(f'Alpha must be positive: {config.dataset.alpha}')
    if config.dataset.num_clients <= 0:
        raise ValueError(f'Number of clients must be positive: {config.dataset.num_clients}')
    valid_datasets = ['cola', 'sst2', 'mnli', 'qqp', 'qnli', 'rte', 'mrpc', 'stsb', 'wnli', 'domainnet', 'cifar100', 'cifar100_pp', 'cifar10', 'pathmnist', 'octmnist', 'banking77', '20newsgroups', 'squad_v2', 'squad_v2_topic', 'e2e_nlg']
    if config.dataset.name not in valid_datasets:
        raise ValueError(f'Invalid dataset: {config.dataset.name}. Must be one of {valid_datasets}')
    if config.training.global_rounds <= 0:
        raise ValueError(f'Global rounds must be positive: {config.training.global_rounds}')
    if not 0 < config.training.client_fraction <= 1:
        raise ValueError(f'Client fraction must be in (0, 1]: {config.training.client_fraction}')
    valid_optimizers = ['adamw', 'sgd', 'galore']
    if config.optimization.optimizer not in valid_optimizers:
        raise ValueError(f'Invalid optimizer: {config.optimization.optimizer}. Must be one of {valid_optimizers}')
    valid_schedules = ['constant', 'cosine', 'linear']
    if config.optimization.lr_schedule not in valid_schedules:
        raise ValueError(f'Invalid lr_schedule: {config.optimization.lr_schedule}. Must be one of {valid_schedules}')
    valid_methods = ['fedit', 'ffalora', 'loraa2', 'fedexlora', 'hetlora', 'fedsalora', 'fedftg', 'fedavg']
    if config.method.name not in valid_methods:
        raise ValueError(f'Invalid method: {config.method.name}. Must be one of {valid_methods}')
    valid_model_names = {'roberta-base', 'roberta-small', 'roberta-large', 'vit-small', 'gpt2'}
    valid_checkpoints = {'FacebookAI/roberta-base', 'roberta-base', 'FacebookAI/roberta-small', 'roberta-small', 'FacebookAI/roberta-large', 'roberta-large', 'google/vit-small-patch16-224', 'vit-small', 'gpt2', 'openai-community/gpt2'}
    is_valid_name = config.model.name in valid_model_names
    is_valid_checkpoint = config.model.name in valid_checkpoints or config.model.checkpoint in valid_checkpoints
    if not is_valid_name and (not is_valid_checkpoint):
        if not config.model.checkpoint:
            raise ValueError(f'Invalid model: {config.model.name}. Must be one of {sorted(valid_model_names)} or specify a valid checkpoint.')
    valid_experiment_types = ['main', 'analysis', 'ablation', 'tuning']
    if config.experiment.type not in valid_experiment_types:
        raise ValueError(f"Invalid experiment type: '{config.experiment.type}'. Must be one of {valid_experiment_types}")
    if config.experiment.gpu < -1:
        raise ValueError(f'GPU ID must be >= -1 (CPU): {config.experiment.gpu}')
    if config.training.dirreg_lambda < 0:
        raise ValueError(f'dirreg_lambda must be non-negative: {config.training.dirreg_lambda}')
    if config.training.fedprox_mu < 0:
        raise ValueError(f'fedprox_mu must be non-negative: {config.training.fedprox_mu}')

def merge_configs(base_config, override_dict):
    config_dict = base_config.to_dict()
    for (key, value) in override_dict.items():
        if '.' in key:
            parts = key.split('.')
            if len(parts) == 2:
                (section, field) = parts
                if section in config_dict:
                    if isinstance(config_dict[section], dict):
                        config_dict[section][field] = value
        elif key in config_dict:
            config_dict[key] = value
    experiment = ExperimentConfig(**config_dict['experiment'])
    dataset = DatasetConfig(**config_dict['dataset'])
    model = ModelConfig(**config_dict['model'])
    method = MethodConfig(**config_dict['method'])
    training = TrainingConfig(**config_dict['training'])
    optimization = OptimizationConfig(**config_dict['optimization'])
    evaluation = EvaluationConfig(**config_dict['evaluation'])
    merged_config = Config(experiment=experiment, dataset=dataset, model=model, method=method, training=training, optimization=optimization, evaluation=evaluation, no_save=config_dict.get('no_save', base_config.no_save))
    validate_config(merged_config)
    return merged_config

def get_method_params(config):
    if isinstance(config, dict):
        return config.get('method', {}).get('params', {}) or {}
    elif hasattr(config, 'method') and hasattr(config.method, 'params'):
        return config.method.params or {}
    return {}

def format_alpha_dir(alpha):
    return f'a{int(round(alpha * 100)):03d}'
