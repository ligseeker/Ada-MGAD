#!/usr/bin/env bash
# Two-fold day-level evaluation for the Nezha OnlineBoutique dataset.
# Fold 1: train on 2022-08-22, test on 2022-08-23; fold 2: the reverse.
# Each run gets its own result directory via the runtime hash. Preprocess first:
#   python util/Nezha/pre_Nezha.py

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

for k in 1 0; do
  echo "=== Nezha OnlineBoutique fold test_experiment=${k} ==="
  python main.py --dataset nezha --test_experiment "${k}"
done
