import logging
from typing import Dict, Union
import torch
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
from transformers import AutoTokenizer
from PIL import Image
logger = logging.getLogger(__name__)

class GLUETransforms:
    TASK_INPUT_KEYS = {'cola': ['sentence'], 'sst2': ['sentence'], 'mnli': ['premise', 'hypothesis'], 'qqp': ['question1', 'question2'], 'qnli': ['question', 'sentence'], 'rte': ['sentence1', 'sentence2'], 'mrpc': ['sentence1', 'sentence2'], 'stsb': ['sentence1', 'sentence2'], 'wnli': ['sentence1', 'sentence2']}

    def __init__(self, task_name, tokenizer_name='FacebookAI/roberta-base', max_length=512, padding='max_length', truncation=True):
        if task_name.lower() not in self.TASK_INPUT_KEYS:
            raise ValueError(f'Unsupported task: {task_name}. Supported: {list(self.TASK_INPUT_KEYS.keys())}')
        self.task_name = task_name.lower()
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f'Loaded tokenizer {tokenizer_name} for task {task_name}')
        except Exception as e:
            logger.error(f'Failed to load tokenizer: {e}')
            raise

    def __call__(self, examples):
        input_keys = self.TASK_INPUT_KEYS[self.task_name]
        if len(input_keys) == 1:
            texts = examples[input_keys[0]]
        else:
            texts = list(zip(examples[input_keys[0]], examples[input_keys[1]]))
        tokenized = self.tokenizer(texts, max_length=self.max_length, padding=self.padding, truncation=self.truncation, return_tensors=None)
        if 'label' in examples:
            tokenized['labels'] = examples['label']
        return tokenized

class CIFAR100Transforms:

    def __init__(self, train=True, image_size=224, mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761), augment=True):
        self.train = train
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment and train
        self.transform = self._build_transform()
        logger.info(f'Initialized CIFAR-100 transforms (train={train}, augment={self.augment}, size={image_size})')

    def _build_transform(self):
        transforms = []
        if self.train and self.augment:
            transforms.extend([T2.RandomResizedCrop(self.image_size, scale=(0.8, 1.0), antialias=True), T2.RandomHorizontalFlip(p=0.5), T2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)])
        else:
            transforms.extend([T2.Resize(int(self.image_size * 1.14), antialias=True), T2.CenterCrop(self.image_size)])
        transforms.append(T2.Normalize(mean=self.mean, std=self.std))
        return T2.Compose(transforms)

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = T.ToTensor()(image)
        return self.transform(image)

class DomainNetTransforms:

    def __init__(self, train=True, image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), augment=True):
        self.train = train
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment and train
        self.transform = self._build_transform()
        logger.info(f'Initialized DomainNet transforms (train={train}, augment={self.augment}, size={image_size})')

    def _build_transform(self):
        transforms = []
        if self.train and self.augment:
            transforms.extend([T.RandomResizedCrop(self.image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(p=0.5), T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), T.RandomRotation(degrees=15)])
        else:
            transforms.extend([T.Resize(int(self.image_size * 1.14)), T.CenterCrop(self.image_size)])
        transforms.extend([T.ToTensor(), T.Normalize(mean=self.mean, std=self.std)])
        return T.Compose(transforms)

    def __call__(self, image):
        return self.transform(image)
