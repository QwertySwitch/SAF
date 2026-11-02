import torch
import torch.nn as nn
from typing import List, Any
from abc import ABC, abstractmethod
from src.utils.aggregation import weighted_average_state_dicts, load_aggregated_state

class Aggregator(ABC):

    @abstractmethod
    def aggregate(self, global_model, client_updates):
        raise NotImplementedError

class FedAvgAggregator(Aggregator):

    def __init__(self, device):
        self.device = device

    def aggregate(self, global_model, client_updates):
        if not client_updates:
            return global_model
        aggregated_state = weighted_average_state_dicts(client_updates)
        return load_aggregated_state(global_model, aggregated_state, strict=False)
