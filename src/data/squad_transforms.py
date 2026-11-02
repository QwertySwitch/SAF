import logging
from typing import Dict, List
import torch
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)

class SQuADTransforms:

    def __init__(self, tokenizer_name='FacebookAI/roberta-base', max_length=384, doc_stride=128, padding='max_length'):
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.padding = padding
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f'Loaded tokenizer {tokenizer_name} for SQuAD (max_length={max_length}, doc_stride={doc_stride})')
        except Exception as e:
            logger.error(f'Failed to load tokenizer: {e}')
            raise

    def __call__(self, examples):
        questions = [q[:1000].strip() for q in examples['question']]
        contexts = examples['context']
        tokenized = self.tokenizer(questions, contexts, max_length=self.max_length, truncation='only_second', stride=self.doc_stride, return_overflowing_tokens=True, return_offsets_mapping=True, padding=self.padding, return_tensors=None)
        sample_mapping = tokenized.pop('overflow_to_sample_mapping')
        offset_mapping = tokenized.pop('offset_mapping')
        start_positions = []
        end_positions = []
        for (i, offsets) in enumerate(offset_mapping):
            original_idx = sample_mapping[i]
            answers = examples['answers'][original_idx]
            if len(answers['answer_start']) == 0 or len(answers['text']) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue
            answer_start_char = answers['answer_start'][0]
            answer_text = answers['text'][0]
            answer_end_char = answer_start_char + len(answer_text)
            sequence_ids = tokenized.sequence_ids(i)
            context_start = None
            context_end = None
            for (idx, seq_id) in enumerate(sequence_ids):
                if seq_id == 1:
                    if context_start is None:
                        context_start = idx
                    context_end = idx
            if context_start is None or context_end is None:
                start_positions.append(0)
                end_positions.append(0)
                continue
            if offsets[context_start][0] > answer_start_char or offsets[context_end][1] < answer_end_char:
                start_positions.append(0)
                end_positions.append(0)
                continue
            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= answer_start_char:
                token_start += 1
            token_start -= 1
            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= answer_end_char:
                token_end -= 1
            token_end += 1
            start_positions.append(token_start)
            end_positions.append(token_end)
        tokenized['start_positions'] = start_positions
        tokenized['end_positions'] = end_positions
        return tokenized

    def tokenize_for_eval(self, examples):
        questions = [q[:1000].strip() for q in examples['question']]
        contexts = examples['context']
        tokenized = self.tokenizer(questions, contexts, max_length=self.max_length, truncation='only_second', stride=self.doc_stride, return_overflowing_tokens=True, return_offsets_mapping=True, padding=self.padding, return_tensors=None)
        sample_mapping = tokenized['overflow_to_sample_mapping']
        tokenized['example_id'] = [examples['id'][sample_mapping[i]] for i in range(len(sample_mapping))]
        return tokenized
