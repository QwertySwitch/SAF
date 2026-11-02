from typing import Any, Callable, Dict
import torch

def is_mnli(dataset_name):
    return dataset_name.lower() == 'mnli'

def has_matched_mismatched(val_dataset):
    return isinstance(val_dataset, dict) and 'matched' in val_dataset and ('mismatched' in val_dataset)

def create_mnli_val_loaders(val_dataset, create_dataloader_fn, batch_size, **dataloader_kwargs):
    return {'matched': create_dataloader_fn(val_dataset['matched'], batch_size, **dataloader_kwargs), 'mismatched': create_dataloader_fn(val_dataset['mismatched'], batch_size, **dataloader_kwargs)}

def evaluate_mnli(model, val_loaders, evaluator):
    matched_metrics = evaluator.evaluate(model, val_loaders['matched'], split='matched')
    mismatched_metrics = evaluator.evaluate(model, val_loaders['mismatched'], split='mismatched')
    matched_acc = matched_metrics.get('accuracy', 0)
    mismatched_acc = mismatched_metrics.get('accuracy', 0)
    return {'matched_acc': matched_acc, 'mismatched_acc': mismatched_acc, 'accuracy': (matched_acc + mismatched_acc) / 2}

def combine_mnli_metrics(matched_metrics, mismatched_metrics):
    matched_acc = matched_metrics.get('accuracy', 0)
    mismatched_acc = mismatched_metrics.get('accuracy', 0)
    return {'matched_acc': matched_acc, 'mismatched_acc': mismatched_acc, 'accuracy': (matched_acc + mismatched_acc) / 2}

def get_mnli_primary_metric(metrics):
    if 'accuracy' in metrics:
        return metrics['accuracy']
    matched = metrics.get('matched_acc', 0)
    mismatched = metrics.get('mismatched_acc', 0)
    return (matched + mismatched) / 2
