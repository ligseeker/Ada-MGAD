#!/usr/bin/env bash
# Leave-one-fault-experiment-out (LOEO) evaluation for the Eadro TT dataset.
# Holds out each of the 9 fault-injection experiments in turn; every run gets
# its own result directory via the runtime hash. Preprocess first:
#   python util/TT/pre_TT.py

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

for k in 0 1 2 3 4 5 6 7 8; do
  echo "=== TT LOEO fold ${k} ==="
  python main.py --dataset tt --test_experiment "${k}"
done
