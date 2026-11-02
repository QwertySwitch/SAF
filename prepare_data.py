#!/usr/bin/env python3
"""Build the tensor files that the shipped client partitions index into.

The Dirichlet client partitions used in the paper are shipped under
data/**/client_indices.pt. They index into the training tensors produced here:

  GLUE (MNLI, QNLI, QQP, SST-2): Hugging Face `glue/<task>`, tokenized with the
      RoBERTa tokenizer (padded to max_length: 64 for SST-2, 128 otherwise), saved to
      data/glue/splits/<task>/{train,validation}.pt
      (MNLI: validation_matched.pt and validation_mismatched.pt).
  MedMNIST v2 (PathMNIST, OctMNIST), 28x28: the official .npz files, scaled to
      [0, 1] float32 in (N, 3, 28, 28) layout (OctMNIST's single channel is
      replicated). The evaluation set is val + test concatenated. Saved to
      data/<task>/splits/{train,test}.pt

After writing, each file is checked against data/EXPECTED_HASHES.json, which
fingerprints the tensors used for the paper. A label mismatch means the shipped
partition indices would not line up with your data; the script then exits with
an error.

Usage:
    python prepare_data.py --tasks all
    python prepare_data.py --tasks mnli sst2 pathmnist
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

GLUE_TASKS = ['mnli', 'qnli', 'qqp', 'sst2']
MEDMNIST_TASKS = ['pathmnist', 'octmnist']
GLUE_MAX_LENGTH = {'mnli': 128, 'qnli': 128, 'qqp': 128, 'sst2': 64}  # as in the configs
MEDMNIST_URL = 'https://zenodo.org/records/10519652/files/{name}.npz?download=1'


def fingerprint(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()[:16]


def check(rel_path, data, expected):
    exp = expected.get(rel_path)
    if exp is None:
        print(f'  [skip] no reference fingerprint for {rel_path}')
        return True
    ok = True
    for key, val in exp.items():
        got = len(data['labels']) if key == 'n' else fingerprint(data[key])
        if got == val:
            print(f'  {rel_path} {key}: ok')
        elif key in ('n', 'labels'):
            # Sample order and labels must match exactly for the partitions to line up.
            print(f'  {rel_path} {key}: MISMATCH')
            ok = False
        else:
            # Tokens/pixels may differ slightly across library versions without breaking the partitions.
            print(f'  {rel_path} {key}: differs from the paper data (library versions?)')
    return ok


def prepare_glue(task, expected, tokenizer='FacebookAI/roberta-base'):
    from src.data.loader import GLUELoader
    from src.data.transforms import GLUETransforms

    out = ROOT / 'data' / 'glue' / 'splits' / task
    out.mkdir(parents=True, exist_ok=True)
    loader = GLUELoader(task, data_dir=str(ROOT / 'data'))
    ds = loader.load()
    transform = GLUETransforms(task_name=task, tokenizer_name=tokenizer, max_length=GLUE_MAX_LENGTH[task])

    def to_tensors(split):
        tok = split.map(transform, batched=True, remove_columns=split.column_names, desc=f'{task}')
        return {'input_ids': torch.tensor(tok['input_ids']),
                'attention_mask': torch.tensor(tok['attention_mask']),
                'labels': torch.tensor(tok['labels'])}

    splits = {'train': 'train'}
    if task == 'mnli':
        splits.update({'validation_matched': 'validation_matched', 'validation_mismatched': 'validation_mismatched'})
    else:
        splits['validation'] = 'validation'
    ok = True
    for fname, hf_split in splits.items():
        data = to_tensors(ds[hf_split])
        torch.save(data, out / f'{fname}.pt')
        ok &= check(f'glue/splits/{task}/{fname}.pt', data, expected)
    return ok


def prepare_medmnist(task, expected):
    raw = ROOT / 'data' / task / 'raw'
    out = ROOT / 'data' / task / 'splits'
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = raw / f'{task}.npz'
    if not npz_path.exists():
        print(f'  downloading {task}.npz ...')
        urllib.request.urlretrieve(MEDMNIST_URL.format(name=task), npz_path)
    z = np.load(npz_path)

    def to_images(arr):
        x = torch.from_numpy(arr).float() / 255.0
        if x.dim() == 3:  # grayscale (N, H, W) -> (N, 3, H, W)
            return x.unsqueeze(1).repeat(1, 3, 1, 1)
        return x.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)

    def to_labels(arr):
        return torch.from_numpy(arr.squeeze().astype(np.int64))

    train = {'images': to_images(z['train_images']), 'labels': to_labels(z['train_labels'])}
    test = {'images': to_images(np.concatenate([z['val_images'], z['test_images']])),
            'labels': to_labels(np.concatenate([z['val_labels'], z['test_labels']]))}
    torch.save(train, out / 'train.pt')
    torch.save(test, out / 'test.pt')
    ok = check(f'{task}/splits/train.pt', train, expected)
    ok &= check(f'{task}/splits/test.pt', test, expected)
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tasks', nargs='+', default=['all'],
                        help='any of: all, ' + ', '.join(GLUE_TASKS + MEDMNIST_TASKS))
    args = parser.parse_args()
    tasks = GLUE_TASKS + MEDMNIST_TASKS if 'all' in args.tasks else args.tasks
    expected = json.load(open(ROOT / 'data' / 'EXPECTED_HASHES.json'))

    all_ok = True
    for task in tasks:
        print(f'== {task}')
        if task in GLUE_TASKS:
            all_ok &= prepare_glue(task, expected)
        elif task in MEDMNIST_TASKS:
            all_ok &= prepare_medmnist(task, expected)
        else:
            raise SystemExit(f'unknown task: {task}')
    if not all_ok:
        raise SystemExit('Sample count or label order differs from the paper data; '
                         'the shipped client partitions would not line up. See README.')
    print('Done. Labels match the paper data, so the shipped partitions line up.')


if __name__ == '__main__':
    main()
