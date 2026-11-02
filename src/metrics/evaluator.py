import torch
import torch.nn as nn
from typing import Dict, Optional
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import numpy as np
import logging
logger = logging.getLogger('fedfl.evaluator')
GLUE_TASKS = ['cola', 'sst2', 'mnli', 'qqp', 'qnli', 'rte', 'mrpc', 'stsb', 'wnli']

class GLUEEvaluator:

    def __init__(self, task_name, device):
        self.task_name = task_name.lower()
        self.device = device
        if self.task_name not in GLUE_TASKS:
            raise ValueError(f'Unknown GLUE task: {task_name}. Supported: {GLUE_TASKS}')

    def evaluate(self, model, dataloader, split=None):
        if self.task_name == 'mnli':
            return self.evaluate_mnli(model, dataloader, split)
        elif self.task_name == 'sst2':
            return self.evaluate_accuracy_only(model, dataloader)
        elif self.task_name == 'cola':
            return self.evaluate_cola(model, dataloader)
        elif self.task_name in ['qqp', 'mrpc']:
            return self.evaluate_accuracy_f1(model, dataloader)
        elif self.task_name in ['qnli', 'rte', 'wnli']:
            return self.evaluate_accuracy_only(model, dataloader)
        elif self.task_name == 'stsb':
            return self.evaluate_stsb(model, dataloader)
        else:
            return self.evaluate_accuracy_only(model, dataloader)

    def evaluate_mnli(self, model, dataloader, split=None):
        model = model.to(self.device)
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for (k, v) in batch.items()}
                outputs = model(**batch)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                loss = outputs.get('loss') if isinstance(outputs, dict) else getattr(outputs, 'loss', None)
                if loss is None:
                    loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        accuracy = accuracy_score(all_labels, all_predictions)
        metrics = {'accuracy': accuracy, 'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0}
        if split == 'matched':
            metrics['matched_acc'] = accuracy
        elif split == 'mismatched':
            metrics['mismatched_acc'] = accuracy
        return metrics

    def evaluate_accuracy_only(self, model, dataloader):
        model = model.to(self.device)
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for (k, v) in batch.items()}
                outputs = model(**batch)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                loss = outputs.get('loss') if isinstance(outputs, dict) else getattr(outputs, 'loss', None)
                if loss is None:
                    loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        accuracy = accuracy_score(all_labels, all_predictions)
        return {'accuracy': accuracy, 'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0}

    def evaluate_accuracy_f1(self, model, dataloader):
        model = model.to(self.device)
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for (k, v) in batch.items()}
                outputs = model(**batch)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                loss = outputs.get('loss') if isinstance(outputs, dict) else getattr(outputs, 'loss', None)
                if loss is None:
                    loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='binary')
        return {'accuracy': accuracy, 'f1': f1, 'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0}

    def evaluate_stsb(self, model, dataloader):
        model = model.to(self.device)
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for (k, v) in batch.items()}
                outputs = model(**batch)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                loss = outputs.get('loss') if isinstance(outputs, dict) else getattr(outputs, 'loss', None)
                if loss is None:
                    loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                predictions = logits.squeeze(-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        (pearson_corr, _) = pearsonr(all_labels, all_predictions)
        (spearman_corr, _) = spearmanr(all_labels, all_predictions)
        return {'pearson': pearson_corr, 'spearman': spearman_corr, 'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0}

    def evaluate_cola(self, model, dataloader):
        model = model.to(self.device)
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for (k, v) in batch.items()}
                outputs = model(**batch)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                loss = outputs.get('loss') if isinstance(outputs, dict) else getattr(outputs, 'loss', None)
                if loss is None:
                    loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        accuracy = accuracy_score(all_labels, all_predictions)
        mcc = matthews_corrcoef(all_labels, all_predictions)
        return {'matthews_correlation': mcc, 'accuracy': accuracy, 'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0}

class CIFAR100Evaluator:

    def __init__(self, device):
        self.device = device

    def evaluate(self, model, dataloader, split=None):
        model = model.to(self.device)
        model.eval()
        all_predictions = []
        all_labels = []
        all_top5_predictions = []
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = model(images=images, labels=labels)
                logits = outputs['logits']
                loss = outputs.get('loss') if isinstance(outputs, dict) else getattr(outputs, 'loss', None)
                if loss is None:
                    loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                top5_pred = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1)[1]
                all_top5_predictions.extend(top5_pred.cpu().numpy())
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        top1_acc = accuracy_score(all_labels, all_predictions)
        correct = sum((1 for (label, top5) in zip(all_labels, all_top5_predictions) if label in top5))
        top5_acc = correct / len(all_labels) if all_labels else 0.0
        return {'top1_acc': top1_acc, 'top5_acc': top5_acc, 'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0}

def get_evaluator(task_name, device):
    task_name = task_name.lower()
    if task_name in GLUE_TASKS:
        return GLUEEvaluator(task_name, device)
    elif task_name in ('cifar100', 'cifar100_pp', 'cifar10', 'pathmnist', 'octmnist'):
        return CIFAR100Evaluator(device)
    elif task_name == 'domainnet':
        return DomainNetEvaluator(device)
    elif task_name in ('squad_v2', 'squad', 'squad_v2_topic'):
        from src.metrics.squad_evaluator import SQuADEvaluator
        return SQuADEvaluator(device)
    elif task_name == 'banking77':
        return Banking77Evaluator(device)
    elif task_name == 'e2e_nlg':
        return E2ENLGEvaluator(device)
    else:
        raise ValueError(f"Unknown task: {task_name}. Available: {GLUE_TASKS + ['cifar100', 'domainnet', 'squad_v2', 'banking77', 'e2e_nlg']}")

class Banking77Evaluator:

    def __init__(self, device):
        self.device = device

    def evaluate(self, model, dataloader, split=None):
        model = model.to(self.device)
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for (k, v) in batch.items()}
                outputs = model(**batch)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                loss = outputs.get('loss') if isinstance(outputs, dict) else getattr(outputs, 'loss', None)
                if loss is None:
                    loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        accuracy = accuracy_score(all_labels, all_predictions)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        return {'accuracy': accuracy, 'macro_f1': macro_f1, 'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0}

class DomainNetEvaluator:

    def __init__(self, device):
        self.device = device
        self.domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

    def evaluate(self, model, dataloader, domain=None):
        model = model.to(self.device)
        model.eval()
        all_predictions = []
        all_labels = []
        all_top5_predictions = []
        domain_predictions = {d: [] for d in self.domains}
        domain_labels = {d: [] for d in self.domains}
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device) if 'image' in batch else batch['images'].to(self.device)
                labels = batch['label'].to(self.device) if 'label' in batch else batch['labels'].to(self.device)
                outputs = model(images=images, labels=labels)
                logits = outputs['logits']
                loss = outputs.get('loss') if isinstance(outputs, dict) else getattr(outputs, 'loss', None)
                if loss is None:
                    loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                top5_pred = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1)[1]
                all_top5_predictions.extend(top5_pred.cpu().numpy())
                if 'domain' in batch:
                    batch_domains = batch['domain']
                    for (i, dom) in enumerate(batch_domains):
                        if dom in self.domains:
                            domain_predictions[dom].append(predictions[i].cpu().item())
                            domain_labels[dom].append(labels[i].cpu().item())
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        top1_acc = accuracy_score(all_labels, all_predictions)
        top5_acc = self._compute_top5_accuracy(np.array(all_labels), np.array(all_top5_predictions))
        metrics = {'top1_acc': top1_acc, 'top5_acc': top5_acc, 'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0}
        per_domain_acc = {}
        for dom in self.domains:
            if domain_labels[dom]:
                dom_acc = accuracy_score(domain_labels[dom], domain_predictions[dom])
                per_domain_acc[f'{dom}_acc'] = dom_acc
        if per_domain_acc:
            metrics['per_domain_acc'] = per_domain_acc
            metrics.update(per_domain_acc)
        return metrics

    def _compute_top5_accuracy(self, labels, top5_predictions):
        correct = 0
        for (label, top5) in zip(labels, top5_predictions):
            if label in top5:
                correct += 1
        return correct / len(labels) if len(labels) > 0 else 0.0

    def evaluate_per_domain(self, model, domain_dataloaders):
        results = {}
        for (domain, dataloader) in domain_dataloaders.items():
            logger.info(f'Evaluating on domain: {domain}')
            metrics = self.evaluate(model, dataloader, domain=domain)
            results[domain] = metrics
        return results

class E2ENLGEvaluator:

    def __init__(self, device):
        self.device = device

    def evaluate(self, model, dataloader, split=None):
        model = model.to(self.device)
        model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for (k, v) in batch.items()}
                outputs = model(**batch)
                loss = outputs.get('loss') if isinstance(outputs, dict) else getattr(outputs, 'loss', None)
                if loss is not None:
                    total_loss += loss.item()
                    num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        import math
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float('inf')
        perplexity = min(perplexity, 10000.0)
        return {'perplexity': perplexity, 'loss': avg_loss}
