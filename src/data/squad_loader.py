import logging
import os
from typing import Dict, List, Optional, Union
from datasets import load_dataset, DatasetDict, Dataset
logger = logging.getLogger(__name__)

class SQuADLoader:
    HF_DATASET_NAME = 'rajpurkar/squad_v2'
    NUM_LABELS = 2
    TRAIN_SIZE = 130319
    VAL_SIZE = 11873
    RECOMMENDED_NUM_CLIENTS = 100
    RECOMMENDED_PARTICIPATING_CLIENTS = 10
    RECOMMENDED_MAX_LENGTH = 384
    RECOMMENDED_BATCH_SIZE = 16

    def __init__(self, data_dir=None):
        base_dir = data_dir or 'data'
        self.base_dir = os.path.join(base_dir, 'squad')
        self.raw_dir = os.path.join(self.base_dir, 'raw')
        self.splits_dir = os.path.join(self.base_dir, 'splits')
        self.dataset = None
        logger.info(f'Initialized SQuADLoader\n  Raw data: {self.raw_dir}\n  Splits data: {self.splits_dir}')

    def load(self, split=None):
        try:
            logger.info('Loading SQuAD v2 dataset from HuggingFace (rajpurkar/squad_v2)...')
            self.dataset = load_dataset(self.HF_DATASET_NAME, cache_dir=self.raw_dir)
            if split is None:
                for split_name in self.dataset.keys():
                    logger.info(f'SQuAD v2 {split_name}: {len(self.dataset[split_name])} examples')
            else:
                logger.info(f'SQuAD v2 {split}: {len(self.dataset[split])} examples')
            return self.dataset if split is None else self.dataset[split]
        except Exception as e:
            logger.error(f'Failed to load SQuAD v2 dataset: {e}')
            raise RuntimeError(f'Dataset loading failed: {e}') from e

    def get_train_data(self):
        if self.dataset is None:
            self.load()
        assert self.dataset is not None
        return self.dataset['train']

    def get_validation_data(self):
        if self.dataset is None:
            self.load()
        assert self.dataset is not None
        return self.dataset['validation']

    def get_test_data(self):
        if self.dataset is None:
            self.load()
        assert self.dataset is not None
        return self.dataset['validation']

    def get_num_labels(self):
        return self.NUM_LABELS

    def get_task_info(self):
        return {'hf_name': self.HF_DATASET_NAME, 'num_labels': self.NUM_LABELS, 'train_size': self.TRAIN_SIZE, 'val_size': self.VAL_SIZE, 'metrics': ['f1', 'exact_match'], 'recommended_num_clients': self.RECOMMENDED_NUM_CLIENTS, 'recommended_participating_clients': self.RECOMMENDED_PARTICIPATING_CLIENTS, 'recommended_max_length': self.RECOMMENDED_MAX_LENGTH, 'recommended_batch_size': self.RECOMMENDED_BATCH_SIZE}
