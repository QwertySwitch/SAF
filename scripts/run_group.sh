#!/usr/bin/env bash
# Run every config in one group sequentially on one GPU.
#   bash scripts/run_group.sh <group> [gpu] [filter]
# Example: bash scripts/run_group.sh main 0 "mnli_fedit_saf09"
set -euo pipefail
GROUP=${1:?usage: run_group.sh <group> [gpu] [filter]}
GPU=${2:-0}
FILTER=${3:-}
cd "$(dirname "$0")/.."
for cfg in configs/"$GROUP"/*.yaml; do
  [[ -n "$FILTER" && "$cfg" != *"$FILTER"* ]] && continue
  echo "=== $cfg (GPU $GPU)"
  python run.py --config "$cfg" --gpu "$GPU"
done
