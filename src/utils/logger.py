import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    COLORS = {'DEBUG': '\x1b[36m', 'INFO': '\x1b[32m', 'WARNING': '\x1b[33m', 'ERROR': '\x1b[31m', 'CRITICAL': '\x1b[35m'}
    RESET = '\x1b[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f'{log_color}{record.levelname}{self.RESET}'
        return super().format(record)

def setup_logger(name='fedfl', log_dir=None, log_file=None, level=logging.INFO, console=True, file_logging=True, colored=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    console_fmt = '%(asctime)s | %(levelname)s | %(message)s'
    file_fmt = '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        if colored:
            console_formatter = ColoredFormatter(console_fmt, datefmt=date_fmt)
        else:
            console_formatter = logging.Formatter(console_fmt, datefmt=date_fmt)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    if file_logging:
        if log_dir is None:
            log_dir = Path('results/logs')
        else:
            log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'{name}_{timestamp}.log'
        log_path = log_dir / log_file
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(file_fmt, datefmt=date_fmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f'\nLogging to file: {log_path}')
    return logger

def get_logger(name='fedfl'):
    return logging.getLogger(name)

class ExperimentLogger:

    def __init__(self, experiment_name, log_dir, level=logging.INFO):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.train_logger = setup_logger(name=f'{experiment_name}_train', log_dir=log_dir, log_file='train.log', level=level)
        self.eval_logger = setup_logger(name=f'{experiment_name}_eval', log_dir=log_dir, log_file='eval.log', level=level)
        self.main_logger = setup_logger(name=experiment_name, log_dir=log_dir, log_file='main.log', level=level)

    def log_train(self, message, level=logging.INFO):
        self.train_logger.log(level, message)

    def log_eval(self, message, level=logging.INFO):
        self.eval_logger.log(level, message)

    def log_main(self, message, level=logging.INFO):
        self.main_logger.log(level, message)

    def info(self, message):
        self.main_logger.info(message)

    def debug(self, message):
        self.main_logger.debug(message)

    def warning(self, message):
        self.main_logger.warning(message)

    def error(self, message):
        self.main_logger.error(message)

    def critical(self, message):
        self.main_logger.critical(message)

def log_experiment_config(logger, config):
    logger.info('=' * 80)
    logger.info('EXPERIMENT CONFIGURATION')
    logger.info('=' * 80)
    for (section, values) in config.items():
        logger.info(f'\n{section.upper()}:')
        if isinstance(values, dict):
            for (key, value) in values.items():
                logger.info(f'  {key}: {value}')
        else:
            logger.info(f'  {values}')
    logger.info('=' * 80)

def log_metrics(logger, metrics, round_num=None, prefix=''):
    if round_num is not None:
        msg = f'{prefix}Round {round_num} | '
    else:
        msg = f'{prefix}'
    metric_strs = []
    for (name, value) in metrics.items():
        if isinstance(value, float):
            metric_strs.append(f'{name}: {value:.4f}')
        else:
            metric_strs.append(f'{name}: {value}')
    msg += ' | '.join(metric_strs)
    logger.info(msg)

def set_log_level(logger, level):
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
