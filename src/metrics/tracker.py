import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
logger = logging.getLogger(__name__)

class MetricsTracker:

    def __init__(self, save_dir, experiment_name=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.train_metrics = []
        self.test_metrics = []
        self.comm_metrics = []
        self.round_summaries = []
        self.method_states = []
        logger.info(f'MetricsTracker initialized: {self.save_dir}')

    def log_train_metrics(self, round_num, metrics):
        display_round = round_num + 1
        entry = {'round': display_round, 'timestamp': datetime.now().isoformat(), **metrics}
        self.train_metrics.append(entry)
        logger.debug(f'Round {display_round} train metrics logged: {metrics}')

    def log_test_metrics(self, round_num, metrics):
        display_round = round_num + 1
        entry = {'round': display_round, 'timestamp': datetime.now().isoformat(), **metrics}
        self.test_metrics.append(entry)
        logger.debug(f'Round {display_round} test metrics logged: {metrics}')

    def log_comm_metrics(self, round_num, bytes_sent, bytes_received, params_exchanged=None):
        display_round = round_num + 1
        entry = {'round': display_round, 'bytes_sent': bytes_sent, 'bytes_received': bytes_received, 'total_bytes': bytes_sent + bytes_received, 'params_exchanged': params_exchanged, 'timestamp': datetime.now().isoformat()}
        self.comm_metrics.append(entry)

    def log_round_summary(self, round_num, num_clients, avg_loss, test_metrics=None):
        display_round = round_num + 1
        summary = {'round': display_round, 'num_clients': num_clients, 'avg_loss': avg_loss, 'timestamp': datetime.now().isoformat()}
        if test_metrics:
            summary['test_metrics'] = test_metrics
        self.round_summaries.append(summary)

    def log_method_state(self, round_num, method_state):
        display_round = round_num + 1
        entry = {'round': display_round, 'timestamp': datetime.now().isoformat(), **method_state}
        self.method_states.append(entry)
        logger.debug(f'Round {display_round} method state logged: {method_state}')

    def log_dora_metrics(self, round_num, magnitude_stats, direction_stats, drift_metrics=None):
        display_round = round_num + 1
        entry = {'round': display_round, 'timestamp': datetime.now().isoformat(), 'type': 'dora_metrics', **magnitude_stats, **direction_stats}
        if drift_metrics:
            entry.update(drift_metrics)
        self.method_states.append(entry)
        logger.debug(f'Round {display_round} DoRA metrics logged')

    def save_metrics(self, prefix=''):
        if self.train_metrics:
            train_path = self.save_dir / f'{prefix}train_metrics.json'
            self._save_json(self.train_metrics, train_path)
            logger.info(f'Train metrics saved to {train_path}')
        if self.test_metrics:
            test_path = self.save_dir / f'{prefix}test_metrics.json'
            self._save_json(self.test_metrics, test_path)
            logger.info(f'Test metrics saved to {test_path}')
        if self.comm_metrics:
            comm_path = self.save_dir / f'{prefix}comm_metrics.json'
            self._save_json(self.comm_metrics, comm_path)
            logger.info(f'Communication metrics saved to {comm_path}')
        if self.round_summaries:
            summary_path = self.save_dir / f'{prefix}round_summaries.json'
            self._save_json(self.round_summaries, summary_path)
            logger.info(f'Round summaries saved to {summary_path}')
        if self.method_states:
            method_path = self.save_dir / f'{prefix}method_states.json'
            self._save_json(self.method_states, method_path)
            logger.info(f'Method states saved to {method_path}')

    def load_metrics(self, prefix=''):
        train_path = self.save_dir / f'{prefix}train_metrics.json'
        if train_path.exists():
            self.train_metrics = self._load_json(train_path)
        test_path = self.save_dir / f'{prefix}test_metrics.json'
        if test_path.exists():
            self.test_metrics = self._load_json(test_path)
        comm_path = self.save_dir / f'{prefix}comm_metrics.json'
        if comm_path.exists():
            self.comm_metrics = self._load_json(comm_path)
        summary_path = self.save_dir / f'{prefix}round_summaries.json'
        if summary_path.exists():
            self.round_summaries = self._load_json(summary_path)
        method_path = self.save_dir / f'{prefix}method_states.json'
        if method_path.exists():
            self.method_states = self._load_json(method_path)
        logger.info(f'Metrics loaded from {self.save_dir}')

    def get_metric_history(self, metric_name, metric_type='test'):
        if metric_type == 'train':
            metrics = self.train_metrics
        elif metric_type == 'test':
            metrics = self.test_metrics
        elif metric_type == 'comm':
            metrics = self.comm_metrics
        else:
            raise ValueError(f'Unknown metric type: {metric_type}')
        return [m.get(metric_name) for m in metrics if metric_name in m]

    def get_best_metric(self, metric_name, metric_type='test', mode='max'):
        if metric_type == 'train':
            metrics = self.train_metrics
        elif metric_type == 'test':
            metrics = self.test_metrics
        elif metric_type == 'comm':
            metrics = self.comm_metrics
        else:
            raise ValueError(f'Unknown metric type: {metric_type}')
        values_with_idx = [(m.get(metric_name), i) for (i, m) in enumerate(metrics) if metric_name in m and m.get(metric_name) is not None]
        if not values_with_idx:
            return {}
        if mode == 'max':
            (best_val, best_idx) = max(values_with_idx, key=lambda x: x[0])
        else:
            (best_val, best_idx) = min(values_with_idx, key=lambda x: x[0])
        return {'round': metrics[best_idx].get('round'), 'value': best_val, 'metrics': metrics[best_idx]}

    def get_final_metrics(self):
        final = {}
        if self.train_metrics:
            final['train'] = self.train_metrics[-1]
        if self.comm_metrics:
            final['comm'] = self.comm_metrics[-1]
        has_ema = any((e.get('split') == 'ema' for e in self.test_metrics)) if self.test_metrics else False
        if has_ema:
            ema_entries = [e for e in self.test_metrics if e.get('split') == 'ema']
            if ema_entries:
                final['test'] = ema_entries[-1]
            best_ema = self.get_best_test_metrics(split_filter='ema')
            if best_ema:
                final['best_test'] = best_ema
            best_raw = self.get_best_test_metrics(split_filter='val')
            if best_raw:
                final['best_raw'] = best_raw
        else:
            if self.test_metrics:
                val_entries = [e for e in self.test_metrics if e.get('split') == 'val']
                if val_entries:
                    final['test'] = val_entries[-1]
                else:
                    final['test'] = self.test_metrics[-1]
            best_test = self.get_best_test_metrics(split_filter='val')
            if not best_test:
                best_test = self.get_best_test_metrics()
            if best_test:
                final['best_test'] = best_test
        return final

    def get_best_test_metrics(self, split_filter=None):
        if not self.test_metrics:
            return {}
        entries = self.test_metrics
        if split_filter is not None:
            entries = [e for e in entries if e.get('split') == split_filter]
            if not entries:
                return {}
        best_results = {}
        skip_fields = {'round', 'timestamp', 'split'}
        all_keys = set()
        for entry in entries:
            all_keys.update(entry.keys())
        metric_keys = all_keys - skip_fields
        lower_is_better = {'loss', 'error', 'perplexity'}
        for key in metric_keys:
            values_with_rounds = [(entry.get(key), entry.get('round')) for entry in entries if entry.get(key) is not None and isinstance(entry.get(key), (int, float))]
            if not values_with_rounds:
                continue
            is_lower_better = any((term in key.lower() for term in lower_is_better))
            if is_lower_better:
                (best_val, best_round) = min(values_with_rounds, key=lambda x: x[0])
            else:
                (best_val, best_round) = max(values_with_rounds, key=lambda x: x[0])
            best_results[f'best_{key}'] = {'value': best_val, 'round': best_round}
        return best_results

    def get_average_metrics(self, exclude_first_round=True):
        if not self.test_metrics:
            return {}
        average_results = {}
        skip_fields = {'round', 'timestamp', 'split'}
        all_keys = set()
        for entry in self.test_metrics:
            all_keys.update(entry.keys())
        metric_keys = all_keys - skip_fields
        excluded_rounds = []
        if exclude_first_round:
            filtered_entries = [entry for entry in self.test_metrics if entry.get('round', 0) > 1]
            excluded_rounds = [1]
        else:
            filtered_entries = self.test_metrics
        if not filtered_entries:
            return {}
        for key in metric_keys:
            values = [entry.get(key) for entry in filtered_entries if entry.get(key) is not None and isinstance(entry.get(key), (int, float))]
            if values:
                avg_value = sum(values) / len(values)
                average_results[f'avg_{key}'] = avg_value
        average_results['num_rounds_averaged'] = len(filtered_entries)
        average_results['excluded_rounds'] = excluded_rounds
        return average_results

    def _save_json(self, data, filepath):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_json(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    def reset(self):
        self.train_metrics = []
        self.test_metrics = []
        self.comm_metrics = []
        self.round_summaries = []
        self.method_states = []
        logger.info('Metrics tracker reset')

    def __repr__(self):
        return f'MetricsTracker(experiment={self.experiment_name}, train_rounds={len(self.train_metrics)}, test_rounds={len(self.test_metrics)})'
