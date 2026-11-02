#!/usr/bin/env python3
"""Compare reproduced runs under results/ with the accuracies recorded in configs/MANIFEST.csv.

Usage: python scripts/summarize.py [--results results]
The deployed-model accuracy at the final round is `test.accuracy` (GLUE; MNLI is the
mean of matched and mismatched) or `test.top1_acc` (MedMNIST) in metrics/*_final.json.
For FedSA-LoRA, the personalized accuracy is also read from metrics/*_test_metrics.json.
"""
import argparse, csv, glob, json, os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(ROOT / 'results'))
    args = ap.parse_args()
    manifest = {Path(r['config']).stem: r for r in csv.DictReader(open(ROOT / 'configs' / 'MANIFEST.csv'))}
    print(f"{'config':58s} {'reported':>9s} {'reproduced':>11s} {'diff':>7s}")
    for cfg_path in sorted(glob.glob(os.path.join(args.results, '**', 'config.yaml'), recursive=True)):
        run_dir = Path(cfg_path).parent
        name = yaml.safe_load(open(cfg_path)).get('experiment', {}).get('name')
        final = glob.glob(str(run_dir / 'metrics' / '*_final.json'))
        if name not in manifest or not final:
            continue
        test = json.load(open(final[0])).get('test', {})
        acc = test.get('accuracy', test.get('top1_acc'))
        if acc is None:
            continue
        rep = float(manifest[name]['final_acc'])
        line = f"{name:58s} {rep:9.2f} {100 * acc:11.2f} {100 * acc - rep:+7.2f}"
        if manifest[name].get('final_personalized_acc'):
            tm = glob.glob(str(run_dir / 'metrics' / '*_test_metrics.json'))
            pers = [e for e in json.load(open(tm[0])) if e.get('split') == 'personalized'] if tm else []
            if pers:
                line += f"   personalized: reported {float(manifest[name]['final_personalized_acc']):.2f}, " \
                        f"reproduced {100 * pers[-1]['personalized_accuracy']:.2f}"
        print(line)


if __name__ == '__main__':
    main()
