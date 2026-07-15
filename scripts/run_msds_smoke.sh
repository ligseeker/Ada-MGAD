#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python main.py \
  --dataset msds \
  --gpu false \
  --epochs 1 \
  --batch_size 8 \
  --num_workers 0 \
  --data_path ./data/examples/msds_tiny/pre \
  --dataset_path ./data/examples/msds_tiny/save \
  --result_dir ./result/msds_smoke
