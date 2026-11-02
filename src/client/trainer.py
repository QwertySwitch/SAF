import torch
import torch.nn as nn
from typing import Optional, Any
import copy
import logging
logger = logging.getLogger('fedfl.trainer')

class ClientTrainer:

    def __init__(self, fl_method, device):
        self.fl_method = fl_method
        self.device = device

    def train(self, model, train_loader, client_id, **kwargs):
        local_model = self._create_local_model(model)
        try:
            update = self.fl_method.local_train(model=local_model, train_loader=train_loader, client_id=client_id, **kwargs)
            logger.debug(f'Client {client_id}: Trained on {update.num_samples} samples, loss = {update.loss:.4f}')
            return update
        except Exception as e:
            logger.error(f'Client {client_id}: Training failed with error: {e}', exc_info=True)
            raise

    def _create_local_model(self, global_model):
        local_model = copy.deepcopy(global_model)
        local_model = local_model.to(self.device)
        local_model.train()
        return local_model

    def evaluate(self, model, test_loader, client_id=None):
        model = model.to(self.device)
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for (k, v) in batch.items()}
                outputs = model(**batch)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                logits = outputs.get('logits') if isinstance(outputs, dict) else getattr(outputs, 'logits', None)
                batch_size = len(batch.get('labels', batch.get('input_ids')))
                if loss is not None:
                    total_loss += loss.item() * batch_size
                if 'labels' in batch and logits is not None:
                    predictions = torch.argmax(logits, dim=-1)
                    total_correct += (predictions == batch['labels']).sum().item()
                    total_samples += len(batch['labels'])
                elif 'start_positions' in batch:
                    total_samples += len(batch['start_positions'])
        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1) if total_samples > 0 else 0.0
        metrics = {'loss': avg_loss, 'accuracy': accuracy, 'num_samples': total_samples}
        if client_id is not None:
            logger.debug(f'Client {client_id}: Eval loss = {avg_loss:.4f}, acc = {accuracy:.4f}')
        return metrics
