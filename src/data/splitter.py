import logging
from typing import Dict, Optional
import numpy as np
from collections import defaultdict
logger = logging.getLogger(__name__)

class DirichletSplitter:

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        logger.info(f'Initialized DirichletSplitter with seed={seed}')

    def split(self, labels, num_clients, alpha, min_samples_per_client=10):
        if num_clients <= 0:
            raise ValueError(f'num_clients must be positive, got {num_clients}')
        if alpha <= 0:
            raise ValueError(f'alpha must be positive, got {alpha}')
        if min_samples_per_client < 1:
            raise ValueError(f'min_samples_per_client must be >= 1, got {min_samples_per_client}')
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        num_samples = len(labels)
        num_labels = len(np.unique(labels))
        min_required = num_clients * min_samples_per_client
        if num_samples < min_required:
            raise ValueError(f'Cannot satisfy min_samples_per_client={min_samples_per_client} with {num_samples} samples and {num_clients} clients. Need at least {min_required} samples.')
        logger.info(f'Splitting {num_samples} samples across {num_clients} clients using Dirichlet(alpha={alpha}), min_samples={min_samples_per_client}')
        logger.info(f'Dataset has {num_labels} unique labels')
        label_to_indices = defaultdict(list)
        for (idx, label) in enumerate(labels):
            label_to_indices[label].append(idx)
        label_to_indices = {label: np.array(indices) for (label, indices) in label_to_indices.items()}
        client_indices = {client_id: [] for client_id in range(num_clients)}
        for (_label, indices) in label_to_indices.items():
            self.rng.shuffle(indices)
            proportions = self.rng.dirichlet(alpha=np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()
            counts = (proportions * len(indices)).astype(int)
            remaining = len(indices) - counts.sum()
            if remaining > 0:
                idx_to_add = np.argsort(proportions)[-remaining:]
                counts[idx_to_add] += 1
            start_idx = 0
            for (client_id, count) in enumerate(counts):
                if count > 0:
                    end_idx = start_idx + count
                    client_indices[client_id].extend(indices[start_idx:end_idx])
                    start_idx = end_idx
        client_indices = {client_id: np.array(indices) for (client_id, indices) in client_indices.items()}
        client_indices = self._enforce_min_samples(client_indices, labels, min_samples_per_client)
        self._log_distribution_stats(client_indices, labels)
        return client_indices

    def _enforce_min_samples(self, client_indices, labels, min_samples):
        client_lists = {cid: list(indices) for (cid, indices) in client_indices.items()}
        deficit_clients = []
        surplus_clients = []
        for (cid, indices) in client_lists.items():
            count = len(indices)
            if count < min_samples:
                deficit_clients.append((cid, min_samples - count))
            elif count > min_samples:
                surplus_clients.append((cid, count - min_samples))
        if not deficit_clients:
            return client_indices
        total_deficit = sum((d for (_, d) in deficit_clients))
        total_surplus = sum((s for (_, s) in surplus_clients))
        if total_surplus < total_deficit:
            logger.warning(f'Insufficient surplus ({total_surplus}) to cover deficit ({total_deficit}). Some clients may have fewer than {min_samples} samples.')
        logger.info(f'Redistributing samples: {len(deficit_clients)} clients below minimum, transferring from {len(surplus_clients)} surplus clients')
        surplus_clients.sort(key=lambda x: x[1], reverse=True)
        surplus_idx = 0
        for (deficit_cid, deficit_amount) in deficit_clients:
            needed = deficit_amount
            while needed > 0 and surplus_idx < len(surplus_clients):
                (surplus_cid, available) = surplus_clients[surplus_idx]
                current_surplus_count = len(client_lists[surplus_cid])
                can_take = min(needed, current_surplus_count - min_samples)
                if can_take > 0:
                    samples_to_transfer = client_lists[surplus_cid][-can_take:]
                    client_lists[surplus_cid] = client_lists[surplus_cid][:-can_take]
                    client_lists[deficit_cid].extend(samples_to_transfer)
                    needed -= can_take
                    new_surplus = len(client_lists[surplus_cid]) - min_samples
                    if new_surplus <= 0:
                        surplus_idx += 1
                    else:
                        surplus_clients[surplus_idx] = (surplus_cid, new_surplus)
                else:
                    surplus_idx += 1
        result = {cid: np.array(indices) for (cid, indices) in client_lists.items()}
        final_counts = {cid: len(indices) for (cid, indices) in result.items()}
        clients_below_min = [cid for (cid, count) in final_counts.items() if count < min_samples]
        if clients_below_min:
            logger.warning(f'After redistribution, {len(clients_below_min)} clients still below minimum: {clients_below_min}')
        else:
            logger.info(f'Redistribution complete. All {len(result)} clients have >= {min_samples} samples.')
        return result

    def _log_distribution_stats(self, client_indices, labels):
        samples_per_client = [len(indices) for indices in client_indices.values()]
        logger.info(f'Samples per client - min: {min(samples_per_client)}, max: {max(samples_per_client)}, mean: {np.mean(samples_per_client):.1f}, std: {np.std(samples_per_client):.1f}')
        label_distributions = []
        for indices in client_indices.values():
            if len(indices) > 0:
                client_labels = labels[indices]
                (unique, counts) = np.unique(client_labels, return_counts=True)
                label_distributions.append(len(unique))
        logger.info(f'Unique labels per client - min: {min(label_distributions)}, max: {max(label_distributions)}, mean: {np.mean(label_distributions):.1f}')

    def compute_heterogeneity_metric(self, client_indices, labels):
        from scipy.stats import entropy
        entropies = []
        for indices in client_indices.values():
            if len(indices) > 0:
                client_labels = labels[indices]
                (unique, counts) = np.unique(client_labels, return_counts=True)
                probs = counts / counts.sum()
                entropies.append(entropy(probs))
        entropies = np.array(entropies)
        cv = np.std(entropies) / (np.mean(entropies) + 1e-10)
        return float(cv)
