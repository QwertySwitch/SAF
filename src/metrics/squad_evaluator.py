import collections
import json
import re
import string
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
logger = logging.getLogger('fedfl.evaluator')

def normalize_answer(s):

    def remove_articles(text):
        return re.sub('\\b(a|an|the)\\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join((ch for ch in text if ch not in string.punctuation))

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_exact(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

class SQuADEvaluator:

    def __init__(self, device, tokenizer_name='FacebookAI/roberta-base', answer_info_path=None):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.answer_info = None
        if answer_info_path and Path(answer_info_path).exists():
            with open(answer_info_path, 'r') as f:
                self.answer_info = json.load(f)
            logger.info(f'Loaded {len(self.answer_info)} answer entries from {answer_info_path}')

    def set_answer_info(self, answer_info):
        self.answer_info = answer_info

    def evaluate(self, model, dataloader, split=None):
        model = model.to(self.device)
        model.eval()
        total_loss = 0.0
        num_batches = 0
        all_start_preds = []
        all_end_preds = []
        all_start_positions = []
        all_end_positions = []
        all_input_ids = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for (k, v) in batch.items()}
                outputs = model(**batch)
                loss = outputs.get('loss') if isinstance(outputs, dict) else getattr(outputs, 'loss', None)
                if loss is not None:
                    total_loss += loss.item()
                    num_batches += 1
                start_logits = outputs['start_logits'] if isinstance(outputs, dict) else outputs.start_logits
                end_logits = outputs['end_logits'] if isinstance(outputs, dict) else outputs.end_logits
                all_start_preds.extend(torch.argmax(start_logits, dim=-1).cpu().numpy())
                all_end_preds.extend(torch.argmax(end_logits, dim=-1).cpu().numpy())
                if 'start_positions' in batch:
                    all_start_positions.extend(batch['start_positions'].cpu().numpy())
                if 'end_positions' in batch:
                    all_end_positions.extend(batch['end_positions'].cpu().numpy())
                all_input_ids.extend(batch['input_ids'].cpu().numpy())
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        if self.answer_info is not None and len(self.answer_info) == len(all_start_preds):
            return self._compute_metrics_with_answers(all_start_preds, all_end_preds, all_input_ids, avg_loss)
        if self.answer_info is not None and len(self.answer_info) != len(all_start_preds):
            logger.warning(f'SQuAD answer_info length mismatch: {len(self.answer_info)} answers vs {len(all_start_preds)} predictions. Using position-based metrics. Re-run data prep to fix: python scripts/analysis/00_prepare_data.py --benchmarks squad_v2')
        if all_start_positions:
            return self._compute_metrics_from_positions(all_start_preds, all_end_preds, all_start_positions, all_end_positions, all_input_ids, avg_loss)
        return {'loss': avg_loss, 'f1': 0.0, 'exact_match': 0.0}

    def _compute_metrics_with_answers(self, start_preds, end_preds, input_ids, avg_loss):
        f1_scores = []
        em_scores = []
        for (i, info) in enumerate(self.answer_info):
            pred_start = start_preds[i]
            pred_end = end_preds[i]
            if pred_start == 0 and pred_end == 0:
                pred_text = ''
            else:
                if pred_end < pred_start:
                    pred_end = pred_start
                tokens = input_ids[i][pred_start:pred_end + 1]
                pred_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            is_impossible = info.get('is_impossible', False)
            gold_answers = info.get('answers', [])
            if is_impossible:
                pred_is_noanswer = pred_start == 0 and pred_end == 0
                f1_scores.append(1.0 if pred_is_noanswer else 0.0)
                em_scores.append(1.0 if pred_is_noanswer else 0.0)
            elif not gold_answers:
                f1_scores.append(0.0)
                em_scores.append(0.0)
            else:
                max_f1 = max((compute_f1(pred_text, ans) for ans in gold_answers))
                max_em = max((compute_exact(pred_text, ans) for ans in gold_answers))
                f1_scores.append(max_f1)
                em_scores.append(max_em)
        return {'f1': float(np.mean(f1_scores)) if f1_scores else 0.0, 'exact_match': float(np.mean(em_scores)) if em_scores else 0.0, 'loss': avg_loss}

    def _compute_metrics_from_positions(self, start_preds, end_preds, start_golds, end_golds, input_ids, avg_loss):
        f1_scores = []
        em_scores = []
        for i in range(len(start_preds)):
            pred_start = start_preds[i]
            pred_end = end_preds[i]
            gold_start = start_golds[i]
            gold_end = end_golds[i]
            pred_is_noanswer = pred_start == 0 and pred_end == 0
            gold_is_noanswer = gold_start == 0 and gold_end == 0
            if gold_is_noanswer and pred_is_noanswer:
                f1_scores.append(1.0)
                em_scores.append(1.0)
                continue
            elif gold_is_noanswer or pred_is_noanswer:
                f1_scores.append(0.0)
                em_scores.append(0.0)
                continue
            if pred_end < pred_start:
                pred_end = pred_start
            pred_tokens = input_ids[i][pred_start:pred_end + 1]
            gold_tokens = input_ids[i][gold_start:gold_end + 1]
            pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
            gold_text = self.tokenizer.decode(gold_tokens, skip_special_tokens=True)
            f1_scores.append(compute_f1(pred_text, gold_text))
            em_scores.append(compute_exact(pred_text, gold_text))
        return {'f1': float(np.mean(f1_scores)) if f1_scores else 0.0, 'exact_match': float(np.mean(em_scores)) if em_scores else 0.0, 'loss': avg_loss}
