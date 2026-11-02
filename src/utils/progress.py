from typing import Optional, Iterable
from tqdm import tqdm
from contextlib import contextmanager

class ProgressTracker:

    def __init__(self):
        self.pbars = {}

    @contextmanager
    def round_progress(self, total, desc='FL Training', position=0):
        pbar = tqdm(total=total, desc=desc, position=position, leave=True, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        self.pbars['round'] = pbar
        try:
            yield pbar
        finally:
            pbar.close()
            self.pbars.pop('round', None)

    @contextmanager
    def client_progress(self, total, desc='Clients', round_num=None, position=1):
        if round_num is not None:
            display_round = round_num + 1
            desc = f'Round {display_round} | {desc}'
        pbar = tqdm(total=total, desc=desc, position=position, leave=False, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')
        self.pbars['client'] = pbar
        try:
            yield pbar
        finally:
            pbar.close()
            self.pbars.pop('client', None)

    @contextmanager
    def epoch_progress(self, total, desc='Epochs', client_id=None, position=2):
        if client_id is not None:
            desc = f'Client {client_id} | {desc}'
        pbar = tqdm(total=total, desc=desc, position=position, leave=False, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        self.pbars['epoch'] = pbar
        try:
            yield pbar
        finally:
            pbar.close()
            self.pbars.pop('epoch', None)

    @contextmanager
    def batch_progress(self, iterable, desc='Batches', position=3, show_metrics=True):
        pbar = tqdm(iterable, desc=desc, position=position, leave=False, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}' if show_metrics else '{l_bar}{bar}| {n_fmt}/{total_fmt}')
        self.pbars['batch'] = pbar
        try:
            yield pbar
        finally:
            pbar.close()
            self.pbars.pop('batch', None)

    def update_round(self, n=1, postfix=None):
        if 'round' in self.pbars:
            self.pbars['round'].update(n)
            if postfix:
                self.pbars['round'].set_postfix(postfix)

    def update_client(self, n=1, postfix=None):
        if 'client' in self.pbars:
            self.pbars['client'].update(n)
            if postfix:
                self.pbars['client'].set_postfix(postfix)

    def close_all(self):
        for pbar in self.pbars.values():
            pbar.close()
        self.pbars.clear()

def create_round_pbar(total, desc='FL Training'):
    return tqdm(total=total, desc=desc, position=0, leave=True, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

def create_client_pbar(total, round_num, desc='Clients'):
    display_round = round_num + 1
    return tqdm(total=total, desc=f'Round {display_round} | {desc}', position=1, leave=False, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')

def wrap_dataloader(dataloader, desc='Batches', leave=False, position=3):
    return tqdm(dataloader, desc=desc, leave=leave, position=position, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')

def format_metrics(metrics):
    formatted = {}
    for (key, value) in metrics.items():
        if isinstance(value, float):
            formatted[key] = f'{value:.4f}'
        elif isinstance(value, int):
            formatted[key] = str(value)
        else:
            formatted[key] = str(value)
    return formatted
