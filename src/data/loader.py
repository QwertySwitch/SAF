import logging
import os
from typing import Dict, List, Optional, Union
from datasets import load_dataset, DatasetDict, Dataset
from torchvision import datasets
logger = logging.getLogger(__name__)

class GLUELoader:
    SUPPORTED_TASKS = {'cola': {'hf_name': 'glue', 'hf_subset': 'cola', 'num_labels': 2, 'train_size': 8551, 'val_size': 1043, 'test_size': 1063, 'input_keys': ['sentence'], 'metrics': ['matthews_correlation', 'accuracy'], 'recommended_num_clients': 20, 'recommended_participating_clients': 10}, 'sst2': {'hf_name': 'glue', 'hf_subset': 'sst2', 'num_labels': 2, 'train_size': 67349, 'val_size': 872, 'test_size': 1821, 'input_keys': ['sentence'], 'metrics': ['accuracy'], 'recommended_num_clients': 100, 'recommended_participating_clients': 10}, 'mnli': {'hf_name': 'glue', 'hf_subset': 'mnli', 'num_labels': 3, 'train_size': 392702, 'val_size': 9815, 'test_size': 9832, 'input_keys': ['premise', 'hypothesis'], 'metrics': ['accuracy', 'matched_acc', 'mismatched_acc'], 'recommended_num_clients': 100, 'recommended_participating_clients': 10}, 'qqp': {'hf_name': 'glue', 'hf_subset': 'qqp', 'num_labels': 2, 'train_size': 363846, 'val_size': 40430, 'test_size': 390965, 'input_keys': ['question1', 'question2'], 'metrics': ['accuracy', 'f1'], 'recommended_num_clients': 100, 'recommended_participating_clients': 10}, 'qnli': {'hf_name': 'glue', 'hf_subset': 'qnli', 'num_labels': 2, 'train_size': 104743, 'val_size': 5463, 'test_size': 5463, 'input_keys': ['question', 'sentence'], 'metrics': ['accuracy'], 'recommended_num_clients': 100, 'recommended_participating_clients': 10}, 'rte': {'hf_name': 'glue', 'hf_subset': 'rte', 'num_labels': 2, 'train_size': 2490, 'val_size': 277, 'test_size': 3000, 'input_keys': ['sentence1', 'sentence2'], 'metrics': ['accuracy'], 'recommended_num_clients': 10, 'recommended_participating_clients': 10}, 'mrpc': {'hf_name': 'glue', 'hf_subset': 'mrpc', 'num_labels': 2, 'train_size': 3668, 'val_size': 408, 'test_size': 1725, 'input_keys': ['sentence1', 'sentence2'], 'metrics': ['accuracy', 'f1'], 'recommended_num_clients': 10, 'recommended_participating_clients': 10}, 'stsb': {'hf_name': 'glue', 'hf_subset': 'stsb', 'num_labels': 1, 'train_size': 5749, 'val_size': 1500, 'test_size': 1379, 'input_keys': ['sentence1', 'sentence2'], 'metrics': ['pearson', 'spearman'], 'is_regression': True, 'recommended_num_clients': 10, 'recommended_participating_clients': 10}, 'wnli': {'hf_name': 'glue', 'hf_subset': 'wnli', 'num_labels': 2, 'train_size': 635, 'val_size': 71, 'test_size': 146, 'input_keys': ['sentence1', 'sentence2'], 'metrics': ['accuracy'], 'recommended_num_clients': 5, 'recommended_participating_clients': 5}}

    def __init__(self, task_name, data_dir=None):
        if task_name.lower() not in self.SUPPORTED_TASKS:
            raise ValueError(f'Unsupported task: {task_name}. Supported tasks: {list(self.SUPPORTED_TASKS.keys())}')
        self.task_name = task_name.lower()
        base_dir = data_dir or 'data'
        self.base_dir = os.path.join(base_dir, 'glue')
        self.raw_dir = os.path.join(self.base_dir, 'raw')
        self.task_dir = os.path.join(self.base_dir, self.task_name)
        self.splits_dir = os.path.join(self.task_dir, 'splits')
        self.task_config = self.SUPPORTED_TASKS[self.task_name]
        self.dataset = None
        logger.info(f'Initialized GLUELoader for task: {self.task_name}\n  Shared raw data: {self.raw_dir}\n  Task splits: {self.splits_dir}')

    def load(self, split=None):
        try:
            logger.info(f'Loading {self.task_name} dataset from HuggingFace...')
            self.dataset = load_dataset(self.task_config['hf_name'], self.task_config['hf_subset'], cache_dir=self.raw_dir)
            if split is None:
                for split_name in self.dataset.keys():
                    logger.info(f'{self.task_name} {split_name}: {len(self.dataset[split_name])} examples')
            else:
                logger.info(f'{self.task_name} {split}: {len(self.dataset[split])} examples')
            return self.dataset if split is None else self.dataset[split]
        except Exception as e:
            logger.error(f'Failed to load {self.task_name} dataset: {e}')
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
        if self.task_name == 'mnli':
            return self.dataset['validation_matched']
        return self.dataset['validation']

    def get_test_data(self):
        if self.dataset is None:
            self.load()
        assert self.dataset is not None
        if self.task_name == 'mnli':
            return {'matched': self.dataset['validation_matched'], 'mismatched': self.dataset['validation_mismatched']}
        else:
            return self.dataset['validation']

    def get_num_labels(self):
        return self.task_config['num_labels']

    def get_label_names(self):
        if self.dataset is None:
            self.load()
        assert self.dataset is not None
        features = self.dataset['train'].features
        if 'label' in features:
            return features['label'].names
        return []

    def get_task_info(self):
        return self.task_config.copy()

class DomainNetLoader:
    DOMAINS = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    DOMAIN_TO_ID = {'clipart': 0, 'infograph': 1, 'painting': 2, 'quickdraw': 3, 'real': 4, 'sketch': 5}
    TOTAL_CATEGORIES = 345
    DEFAULT_CATEGORIES = 100
    HF_DATASET_NAME = 'wltjr1007/DomainNet'

    def __init__(self, data_dir=None, num_categories=DEFAULT_CATEGORIES, domains=None):
        self.num_categories = num_categories
        self.domains = domains or self.DOMAINS
        if not 1 <= num_categories <= self.TOTAL_CATEGORIES:
            raise ValueError(f'num_categories must be between 1 and {self.TOTAL_CATEGORIES}, got {num_categories}')
        invalid_domains = set(self.domains) - set(self.DOMAINS)
        if invalid_domains:
            raise ValueError(f'Invalid domains: {invalid_domains}. Valid domains: {self.DOMAINS}')
        base_dir = data_dir or 'data'
        self.base_dir = os.path.join(base_dir, 'domainnet')
        self.domain_dirs = {}
        self.raw_dirs = {}
        self.splits_dirs = {}
        for domain in self.domains:
            domain_dir = os.path.join(self.base_dir, domain)
            self.domain_dirs[domain] = domain_dir
            self.raw_dirs[domain] = os.path.join(domain_dir, 'raw')
            self.splits_dirs[domain] = os.path.join(domain_dir, 'splits')
        self.dataset = None
        logger.info(f'Initialized DomainNetLoader with {num_categories} categories and domains: {self.domains}\n  Base directory: {self.base_dir}')

    def load(self, domain=None, split=None):
        if domain is not None and domain not in self.DOMAINS:
            raise ValueError(f'Invalid domain: {domain}. Valid: {self.DOMAINS}')
        try:
            if self.dataset is None:
                logger.info('Loading DomainNet from Hugging Face...')
                shared_raw_dir = os.path.join(self.base_dir, 'raw')
                self.dataset = load_dataset(self.HF_DATASET_NAME, cache_dir=shared_raw_dir)
                logger.info('DomainNet loaded successfully')
            if domain is None:
                if split is None:
                    return self.dataset
                else:
                    return {dom: self._filter_by_domain_and_categories(self.dataset[split], dom) for dom in self.domains}
            elif split is None:
                return DatasetDict({'train': self._filter_by_domain_and_categories(self.dataset['train'], domain), 'test': self._filter_by_domain_and_categories(self.dataset['test'], domain)})
            else:
                return self._filter_by_domain_and_categories(self.dataset[split], domain)
        except Exception as e:
            logger.error(f'Failed to load DomainNet: {e}')
            raise RuntimeError(f'Dataset loading failed: {e}') from e

    def _filter_by_domain_and_categories(self, dataset, domain):
        domain_id = self.DOMAIN_TO_ID[domain]
        if self.num_categories >= self.TOTAL_CATEGORIES:
            filtered = dataset.filter(lambda ex: ex['domain'] == domain_id)
        else:
            filtered = dataset.filter(lambda ex: ex['domain'] == domain_id and ex['label'] < self.num_categories)
        logger.info(f'Filtered {domain}: {len(filtered)} images ({self.num_categories} categories)')
        return filtered

    def get_train_data(self, domain=None):
        if self.dataset is None:
            self.load()
        assert self.dataset is not None
        if domain is None:
            domain = self.domains[0]
        return self._filter_by_domain_and_categories(self.dataset['train'], domain)

    def get_validation_data(self, domain=None):
        if self.dataset is None:
            self.load()
        assert self.dataset is not None
        if domain is None:
            domain = self.domains[0]
        return self._filter_by_domain_and_categories(self.dataset['test'], domain)

    def get_test_data(self, domain=None):
        if self.dataset is None:
            self.load()
        assert self.dataset is not None
        if domain is None:
            domain = self.domains[0]
        return self._filter_by_domain_and_categories(self.dataset['test'], domain)

    def get_num_labels(self):
        return self.num_categories

    def get_domains(self):
        return self.domains

class CIFAR100Loader:

    def __init__(self, data_dir=None):
        base_dir = data_dir or 'data'
        self.base_dir = os.path.join(base_dir, 'cifar100')
        self.raw_dir = os.path.join(self.base_dir, 'raw')
        self.splits_dir = os.path.join(self.base_dir, 'splits')
        self.dataset_train = None
        self.dataset_test = None
        logger.info(f'Initialized CIFAR100Loader\n  Raw data: {self.raw_dir}\n  Splits data: {self.splits_dir}')

    def load(self, split=None, download=True, transform=None):
        if split is not None and split not in ['train', 'test']:
            raise ValueError(f"Invalid split: {split}. Valid: ['train', 'test']")
        try:
            if split is None or split == 'train':
                logger.info('Loading CIFAR-100 training set from TorchVision...')
                self.dataset_train = datasets.CIFAR100(root=self.raw_dir, train=True, download=download, transform=transform)
                logger.info(f'Loaded {len(self.dataset_train)} training examples')
            if split is None or split == 'test':
                logger.info('Loading CIFAR-100 test set from TorchVision...')
                self.dataset_test = datasets.CIFAR100(root=self.raw_dir, train=False, download=download, transform=transform)
                logger.info(f'Loaded {len(self.dataset_test)} test examples')
            if split is None:
                return {'train': self.dataset_train, 'test': self.dataset_test}
            elif split == 'train':
                return self.dataset_train
            else:
                return self.dataset_test
        except Exception as e:
            logger.error(f'Failed to load CIFAR-100: {e}')
            raise RuntimeError(f'Dataset loading failed: {e}') from e

    def get_train_data(self, transform=None):
        if self.dataset_train is None:
            self.load(split='train', transform=transform)
        assert self.dataset_train is not None
        return self.dataset_train

    def get_validation_data(self, transform=None):
        if self.dataset_test is None:
            self.load(split='test', transform=transform)
        assert self.dataset_test is not None
        return self.dataset_test

    def get_test_data(self, transform=None):
        if self.dataset_test is None:
            self.load(split='test', transform=transform)
        assert self.dataset_test is not None
        return self.dataset_test

    def get_num_labels(self):
        return 100

    def get_class_names(self):
        if self.dataset_train is None:
            self.load(split='train')
        assert self.dataset_train is not None
        return self.dataset_train.classes

class Banking77Loader:
    HF_DATASET_NAME = 'PolyAI/banking77'
    HF_REVISION = 'refs/convert/parquet'
    NUM_LABELS = 77
    TRAIN_SIZE = 10003
    TEST_SIZE = 3080
    RECOMMENDED_NUM_CLIENTS = 30
    RECOMMENDED_PARTICIPATING_CLIENTS = 10
    RECOMMENDED_MAX_LENGTH = 64
    RECOMMENDED_BATCH_SIZE = 1024
    INTENT_LABELS = ['activate_my_card', 'age_limit', 'apple_pay_or_google_pay', 'atm_support', 'automatic_top_up', 'balance_not_updated_after_bank_transfer', 'balance_not_updated_after_cheque_or_cash_deposit', 'beneficiary_not_allowed', 'cancel_transfer', 'card_about_to_expire', 'card_acceptance', 'card_arrival', 'card_delivery_estimate', 'card_linking', 'card_not_working', 'card_payment_fee_charged', 'card_payment_not_recognised', 'card_payment_wrong_exchange_rate', 'card_swallowed', 'cash_withdrawal_charge', 'cash_withdrawal_not_recognised', 'change_pin', 'compromised_card', 'contactless_not_working', 'country_support', 'declined_card_payment', 'declined_cash_withdrawal', 'declined_transfer', 'direct_debit_payment_not_recognised', 'disposable_card_limits', 'edit_personal_details', 'exchange_charge', 'exchange_rate', 'exchange_via_app', 'extra_charge_on_statement', 'failed_transfer', 'fiat_currency_support', 'get_disposable_virtual_card', 'get_physical_card', 'getting_spare_card', 'getting_virtual_card', 'lost_or_stolen_card', 'lost_or_stolen_phone', 'order_physical_card', 'passcode_forgotten', 'pending_card_payment', 'pending_cash_withdrawal', 'pending_top_up', 'pending_transfer', 'pin_blocked', 'receiving_money', 'Refund_not_showing_up', 'request_refund', 'reverted_card_payment?', 'supported_cards_and_currencies', 'terminate_account', 'top_up_by_bank_transfer_charge', 'top_up_by_card_charge', 'top_up_by_cash_or_cheque', 'top_up_failed', 'top_up_limits', 'top_up_reverted', 'topping_up_by_card', 'transaction_charged_twice', 'transfer_fee_charged', 'transfer_into_account', 'transfer_not_received_by_recipient', 'transfer_timing', 'unable_to_verify_identity', 'verify_my_identity', 'verify_source_of_funds', 'verify_top_up', 'virtual_card_not_working', 'visa_or_mastercard', 'why_verify_identity', 'wrong_amount_of_cash_received', 'wrong_exchange_rate_for_cash_withdrawal']

    def __init__(self, data_dir=None):
        base_dir = data_dir or 'data'
        self.base_dir = os.path.join(base_dir, 'banking77')
        self.raw_dir = os.path.join(self.base_dir, 'raw')
        self.splits_dir = os.path.join(self.base_dir, 'splits')
        self.dataset = None
        logger.info(f'Initialized Banking77Loader\n  Raw data: {self.raw_dir}\n  Splits data: {self.splits_dir}')

    def load(self, split=None):
        try:
            logger.info('Loading Banking77 dataset from HuggingFace (PolyAI/banking77)...')
            self.dataset = load_dataset(self.HF_DATASET_NAME, revision=self.HF_REVISION, cache_dir=self.raw_dir)
            if split is None:
                for split_name in self.dataset.keys():
                    logger.info(f'Banking77 {split_name}: {len(self.dataset[split_name])} examples')
            else:
                logger.info(f'Banking77 {split}: {len(self.dataset[split])} examples')
            return self.dataset if split is None else self.dataset[split]
        except Exception as e:
            logger.error(f'Failed to load Banking77 dataset: {e}')
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
        return self.dataset['test']

    def get_test_data(self):
        if self.dataset is None:
            self.load()
        assert self.dataset is not None
        return self.dataset['test']

    def get_num_labels(self):
        return self.NUM_LABELS

    def get_label_names(self):
        return self.INTENT_LABELS.copy()

    def get_task_info(self):
        return {'hf_name': self.HF_DATASET_NAME, 'hf_revision': self.HF_REVISION, 'num_labels': self.NUM_LABELS, 'train_size': self.TRAIN_SIZE, 'test_size': self.TEST_SIZE, 'input_keys': ['text'], 'text_column': 'text', 'label_column': 'label', 'metrics': ['accuracy', 'f1_macro'], 'recommended_num_clients': self.RECOMMENDED_NUM_CLIENTS, 'recommended_participating_clients': self.RECOMMENDED_PARTICIPATING_CLIENTS, 'recommended_client_fraction': self.RECOMMENDED_PARTICIPATING_CLIENTS / self.RECOMMENDED_NUM_CLIENTS, 'recommended_max_length': self.RECOMMENDED_MAX_LENGTH, 'recommended_batch_size': self.RECOMMENDED_BATCH_SIZE}

class TwentyNewsgroupsLoader:
    HF_DATASET_NAME = 'SetFit/20_newsgroups'
    NUM_LABELS = 20
    TRAIN_SIZE = 11314
    TEST_SIZE = 7532
    RECOMMENDED_NUM_CLIENTS = 30
    RECOMMENDED_PARTICIPATING_CLIENTS = 10
    RECOMMENDED_MAX_LENGTH = 512
    RECOMMENDED_BATCH_SIZE = 128
    CATEGORY_LABELS = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

    def __init__(self, data_dir=None):
        base_dir = data_dir or 'data'
        self.base_dir = os.path.join(base_dir, '20newsgroups')
        self.raw_dir = os.path.join(self.base_dir, 'raw')
        self.splits_dir = os.path.join(self.base_dir, 'splits')
        self.dataset = None
        logger.info(f'Initialized TwentyNewsgroupsLoader\n  Raw data: {self.raw_dir}\n  Splits data: {self.splits_dir}')

    def load(self, split=None):
        try:
            logger.info('Loading 20 Newsgroups dataset from HuggingFace...')
            self.dataset = load_dataset(self.HF_DATASET_NAME, cache_dir=self.raw_dir)
            if split is None:
                for split_name in self.dataset.keys():
                    logger.info(f'20 Newsgroups {split_name}: {len(self.dataset[split_name])} examples')
            else:
                logger.info(f'20 Newsgroups {split}: {len(self.dataset[split])} examples')
            return self.dataset if split is None else self.dataset[split]
        except Exception as e:
            logger.error(f'Failed to load 20 Newsgroups dataset: {e}')
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
        return self.dataset['test']

    def get_test_data(self):
        if self.dataset is None:
            self.load()
        assert self.dataset is not None
        return self.dataset['test']

    def get_num_labels(self):
        return self.NUM_LABELS

    def get_label_names(self):
        return self.CATEGORY_LABELS.copy()

    def get_task_info(self):
        return {'hf_name': self.HF_DATASET_NAME, 'num_labels': self.NUM_LABELS, 'train_size': self.TRAIN_SIZE, 'test_size': self.TEST_SIZE, 'input_keys': ['text'], 'text_column': 'text', 'label_column': 'label', 'label_text_column': 'label_text', 'metrics': ['accuracy', 'f1_macro'], 'recommended_num_clients': self.RECOMMENDED_NUM_CLIENTS, 'recommended_participating_clients': self.RECOMMENDED_PARTICIPATING_CLIENTS, 'recommended_client_fraction': self.RECOMMENDED_PARTICIPATING_CLIENTS / self.RECOMMENDED_NUM_CLIENTS, 'recommended_max_length': self.RECOMMENDED_MAX_LENGTH, 'recommended_batch_size': self.RECOMMENDED_BATCH_SIZE}

def get_data_loader(dataset_name, data_dir=None):
    dataset_name = dataset_name.lower()
    if dataset_name in GLUELoader.SUPPORTED_TASKS:
        return GLUELoader(dataset_name, data_dir=data_dir)
    elif dataset_name == 'domainnet':
        return DomainNetLoader(data_dir=data_dir)
    elif dataset_name in ('cifar100', 'cifar100_pp', 'cifar10', 'pathmnist', 'octmnist'):
        return CIFAR100Loader(data_dir=data_dir)
    elif dataset_name == 'banking77':
        return Banking77Loader(data_dir=data_dir)
    elif dataset_name in ('20newsgroups', '20_newsgroups', 'newsgroups'):
        return TwentyNewsgroupsLoader(data_dir=data_dir)
    elif dataset_name in ('squad_v2', 'squad', 'squad_v2_topic'):
        from src.data.squad_loader import SQuADLoader
        return SQuADLoader(data_dir=data_dir)
    else:
        available = list(GLUELoader.SUPPORTED_TASKS.keys()) + ['domainnet', 'cifar100', 'banking77', '20newsgroups', 'squad_v2']
        raise ValueError(f'Unknown dataset: {dataset_name}. Available: {available}')
