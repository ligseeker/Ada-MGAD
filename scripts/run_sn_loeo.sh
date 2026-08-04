#!/usr/bin/env bash
# Leave-one-fault-experiment-out (LOEO) evaluation for the Eadro SN dataset.
# Holds out each of the 4 fault-injection experiments in turn; every run gets
# its own result directory via the runtime hash. Preprocess first:
#   python util/SN/pre_SN.py

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

for k in 0 1 2 3; do
  echo "=== SN LOEO fold ${k} ==="
  python main.py --dataset sn --test_experiment "${k}"
done
