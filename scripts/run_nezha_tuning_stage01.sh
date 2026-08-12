#!/bin/bash
# Stage 0 + Stage 1 of the OnlineBoutique tuning plan (see docs/Nezha_tuning_log.md).
# Serial execution; each run logs to logs/tuning/<name>.log.
cd /home/zhangll24/RCA_project/Ada-MGAD-exp || exit 1
mkdir -p logs/tuning
# If this driver dies, take the training child down with it (no orphans on the GPU).
trap 'kill 0' EXIT INT TERM

run() {
  name=$1; shift
  echo "=== $name start $(date '+%F %T') args: $*"
  python3 -u main.py --dataset nezha "$@" > "logs/tuning/$name.log" 2>&1
  code=$?
  echo "=== $name exit=$code $(date '+%F %T')"
  latest=$(ls -td result/Ada-MGAD-Nezha-save-* 2>/dev/null | head -1)
  if [ -n "$latest" ]; then
    echo "--- summary ($latest):"
    tail -3 "$latest/result_summary.log" 2>/dev/null
  fi
}

run s0_baseline
run s1_window15 --window 15 --dataset_path ./data/Nezha-save-w15s1
run s1_step2 --step 2 --dataset_path ./data/Nezha-save-w10s2
run s1_aw100 --abnormal_weight 100
run s1_lw1e-1 --label_weight 1e-1
run s1_cw0 --contrast_weight 0
run s1_lr5e-4 --learning_rate 5e-4

echo "=== ALL DONE $(date '+%F %T')"
