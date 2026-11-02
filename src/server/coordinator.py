import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import random
import numpy as np
from tqdm import tqdm
import logging
from ..client.trainer import ClientTrainer
from ..methods.fedavg import ClientUpdate
from ..utils.lr_scheduler import create_lr_scheduler
from .aggregator import Aggregator, FedAvgAggregator
logger = logging.getLogger('fedfl.coordinator')

class FederatedCoordinator:

    def __init__(self, global_model, fl_method, client_data, device, config, aggregator=None):
        self.global_model = global_model.to(device)
        self.fl_method = fl_method
        self.client_data = client_data
        self.device = device
        self.config = config
        self.aggregator = aggregator or FedAvgAggregator(device)
        self.client_trainer = ClientTrainer(fl_method, device)
        self.num_clients = len(client_data)
        if isinstance(config, dict):
            training_config = config.get('training', {})
            self.global_rounds = training_config.get('global_rounds', 50)
            self.client_fraction = training_config.get('client_fraction', 0.1)
            self.eval_every = training_config.get('eval_every', 5)
        else:
            training = getattr(config, 'training', None)
            if training is not None:
                self.global_rounds = getattr(training, 'global_rounds', 50)
                self.client_fraction = getattr(training, 'client_fraction', 0.1)
                self.eval_every = getattr(training, 'eval_every', 5)
            else:
                self.global_rounds = getattr(config, 'global_rounds', 50)
                self.client_fraction = getattr(config, 'client_fraction', 0.1)
                self.eval_every = getattr(config, 'eval_every', 5)
        if isinstance(config, dict):
            self.lr_scheduler = create_lr_scheduler(config, self.global_rounds)
        else:
            config_dict = config.to_dict() if hasattr(config, 'to_dict') else {}
            self.lr_scheduler = create_lr_scheduler(config_dict, self.global_rounds)
        if hasattr(self.lr_scheduler, 'get_schedule_info'):
            logger.info(f'LR Scheduler: {self.lr_scheduler.get_schedule_info()}')
        else:
            logger.info(f'LR Scheduler: {type(self.lr_scheduler).__name__}')
        logger.info(f'Client fraction: {self.client_fraction} ({round(self.num_clients * self.client_fraction)}/{self.num_clients} clients per round)')
        self.current_round = 0
        self.train_history = []

    def train(self, evaluator=None, test_loader=None):
        logger.info(f'Starting federated training for {self.global_rounds} rounds')
        logger.info(f'Total clients: {self.num_clients}')
        logger.info(f'Clients per round: {round(self.num_clients * self.client_fraction)}')
        for round_num in tqdm(range(self.global_rounds), desc='FL Training'):
            self.current_round = round_num
            selected_clients = self._select_clients(round_num)
            display_round = round_num + 1
            logger.debug(f'Round {display_round}: Selected {len(selected_clients)} clients')
            client_updates = self._train_clients(selected_clients)
            lr_factor = self.lr_scheduler.get_lr_factor(round_num)
            if hasattr(self.fl_method, 'aggregate'):
                self.global_model = self.fl_method.aggregate(self.global_model, client_updates, round_idx=round_num, lr_factor=lr_factor)
            else:
                self.global_model = self.aggregator.aggregate(self.global_model, client_updates)
            avg_loss = np.mean([u.loss for u in client_updates])
            logger.info(f'Round {display_round}: Avg client loss = {avg_loss:.4f}')
            round_entry = {'round': display_round, 'avg_loss': avg_loss, 'num_clients': len(client_updates)}
            reg_keys = ['dirreg_avg_cos_sim', 'dirreg_min_cos_sim', 'dirreg_loss', 'dirreg_n_layers', 'fedprox_dist_sq', 'fedprox_loss', 'fedprox_n_params']
            for key in reg_keys:
                values = [u.metrics[key] for u in client_updates if hasattr(u, 'metrics') and u.metrics and (key in u.metrics)]
                if values:
                    round_entry[f'avg_{key}'] = float(np.mean(values))
                    if 'min_cos_sim' in key:
                        round_entry[f'worst_{key}'] = float(min(values))
            self.train_history.append(round_entry)
            should_eval = display_round == 1 or display_round % self.eval_every == 0
            if evaluator and test_loader and should_eval:
                eval_metrics = self._evaluate(evaluator, test_loader)
                logger.info(f'Round {display_round}: Eval metrics = {eval_metrics}')
                self.train_history[-1].update(eval_metrics)
        logger.info('Federated training completed')
        return self._get_history()

    def _select_clients(self, round_num):
        num_selected = max(1, min(self.num_clients, round(self.num_clients * self.client_fraction)))
        if isinstance(self.config, dict):
            seed = self.config.get('experiment', {}).get('seed', 42)
        else:
            experiment = getattr(self.config, 'experiment', None)
            if experiment is not None:
                seed = getattr(experiment, 'seed', 42)
            else:
                seed = getattr(self.config, 'seed', 42)
        random.seed(round_num + seed)
        client_ids = list(self.client_data.keys())
        selected = random.sample(client_ids, num_selected)
        return selected

    def _train_clients(self, selected_clients):
        client_updates = []
        train_kwargs = self._get_client_train_kwargs()
        display_round = self.current_round + 1
        for client_id in tqdm(selected_clients, desc=f'Round {display_round}', leave=False):
            client_loader = self.client_data[client_id]
            update = self.client_trainer.train(model=self.global_model, train_loader=client_loader, client_id=client_id, **train_kwargs)
            client_updates.append(update)
        return client_updates

    def _get_client_train_kwargs(self):
        kwargs = {}
        lr_factor = self.lr_scheduler.get_lr_factor(self.current_round)
        kwargs['lr_factor'] = lr_factor
        kwargs['round_idx'] = self.current_round
        if hasattr(self.fl_method, 'get_shared_gradient'):
            shared_gradient = self.fl_method.get_shared_gradient()
            if shared_gradient is not None:
                kwargs['shared_gradient'] = shared_gradient
        return kwargs

    def _evaluate(self, evaluator, test_loader):
        self.global_model.eval()
        try:
            with torch.no_grad():
                if hasattr(evaluator, 'evaluate'):
                    metrics = evaluator.evaluate(self.global_model, test_loader)
                else:
                    metrics = evaluator(self.global_model, test_loader)
            return metrics
        finally:
            self.global_model.train()

    def save_checkpoint(self, filepath):
        checkpoint = {'round': self.current_round, 'model_state_dict': self.global_model.state_dict(), 'config': self.config, 'train_history': self.train_history}
        torch.save(checkpoint, filepath)
        logger.info(f'Checkpoint saved to {filepath}')

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.current_round = checkpoint['round']
        self.train_history = checkpoint['train_history']
        logger.info(f'Checkpoint loaded from {filepath}')
        logger.info(f'Resuming from round {self.current_round}')

    def _get_history(self):
        if not self.train_history:
            return {}
        all_keys = set()
        for entry in self.train_history:
            all_keys.update(entry.keys())
        history = {}
        for key in all_keys:
            history[key] = [entry.get(key) for entry in self.train_history]
        return history

    def train_round(self, round_num):
        self.current_round = round_num
        display_round = round_num + 1
        selected_clients = self._select_clients(round_num)
        logger.debug(f'Round {display_round}: Selected {len(selected_clients)} clients')
        client_updates = self._train_clients(selected_clients)
        lr_factor = self.lr_scheduler.get_lr_factor(round_num)
        if hasattr(self.fl_method, 'aggregate'):
            self.global_model = self.fl_method.aggregate(self.global_model, client_updates, round_idx=round_num, lr_factor=lr_factor)
        else:
            self.global_model = self.aggregator.aggregate(self.global_model, client_updates)
        avg_loss = np.mean([u.loss for u in client_updates])
        round_metrics = {'round': display_round, 'avg_loss': avg_loss, 'num_clients': len(client_updates), 'lr_factor': lr_factor}
        reg_keys = ['dirreg_avg_cos_sim', 'dirreg_min_cos_sim', 'dirreg_max_cos_sim', 'dirreg_loss', 'dirreg_n_layers', 'fedprox_dist_sq', 'fedprox_loss', 'fedprox_n_params']
        for u in client_updates:
            if hasattr(u, 'metrics') and u.metrics:
                for key in u.metrics:
                    if key.startswith('dirreg_cs_') and key not in reg_keys:
                        reg_keys.append(key)
                break
        client_reg_metrics = {}
        for key in reg_keys:
            values = [u.metrics[key] for u in client_updates if hasattr(u, 'metrics') and u.metrics and (key in u.metrics)]
            if values:
                client_reg_metrics[f'avg_{key}'] = float(np.mean(values))
                if 'min_cos_sim' in key:
                    client_reg_metrics[f'worst_{key}'] = float(min(values))
        if client_reg_metrics:
            round_metrics.update(client_reg_metrics)
        if len(client_updates) >= 2:
            a_divs = []
            b_divs = []
            ref_state = client_updates[0].state_dict
            for u in client_updates[1:]:
                for k in ref_state:
                    if k not in u.state_dict:
                        continue
                    diff = torch.norm(ref_state[k].float() - u.state_dict[k].float(), p='fro').item()
                    if 'lora_A' in k:
                        a_divs.append(diff)
                    elif 'lora_B' in k:
                        b_divs.append(diff)
            if a_divs:
                round_metrics['client_div_a_mean'] = float(np.mean(a_divs))
            if b_divs:
                round_metrics['client_div_b_mean'] = float(np.mean(b_divs))
        if hasattr(self.fl_method, 'get_similarity_dict'):
            similarity_metrics = self.fl_method.get_similarity_dict()
            if similarity_metrics:
                round_metrics.update(similarity_metrics)
        self.train_history.append(round_metrics)
        return round_metrics

    def get_global_model(self):
        return self.global_model

    def set_global_model(self, model):
        self.global_model = model.to(self.device)
