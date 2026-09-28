# P6-C1 G3 manual formal run handoff

Status on 2026-09-29: code, bounded smoke and full read-only input preflight pass; **no formal C1 stage has run**. The commands below use the frozen G2-v1.1 run ID `c1-prefix-oos-v1-seed42`. Run them in order with the DAG interpreter. Each full raw processing, detector training, RCA training, Test scoring and evaluation command is for the user to execute manually. Do not substitute P5 or C0 `all` commands.

## 1. Reserve the new run after full read-only preflight

`init` verifies all G2 bound files, rehashes the roughly 31.58 GB raw catalog, verifies the raw RCA index arrays, checks the tracked 1/24-worker smoke record, then creates the one-time run root. If the target run root already exists, it stops; never delete or reuse a partial root.

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c1.py init
```

## 2. Three prefix-fitted detector folds

Each `fold-input` reads the full bound raw catalog and fits Metric/Log/Trace only on that fold's Fit prefix, using 24 spawn workers. Each `fold-detector` performs one detector fit (at most 30 epochs), Selection-only checkpoint/threshold choice, and label-free Generation scoring. The current DAG environment reports no CUDA device, so these commands reserve CPU and use eight Torch threads. If you run on a CUDA-enabled environment instead, add `--gpu` to `init` and to **all three** `fold-detector` commands; the run lock rejects mixed device requests. Run the next command only if the previous one exits successfully.

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
bash -e <<'SH'
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c1.py fold-input --fold 1
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c1.py fold-detector --fold 1
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c1.py fold-input --fold 2
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c1.py fold-detector --fold 2
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c1.py fold-input --fold 3
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c1.py fold-detector --fold 3
SH
```

If a real prefix cannot supply frozen 45 Metric slots, 17 stable Log templates, or a Trace graph, the fold stops with `PREFIX_SCHEMA_NO_GO`. Preserve its `INCOMPLETE.json` and `failure.json`; do not fill slots from future data or rerun in the same directory.

## 3. Freeze the common Train cohort

This stage verifies all three sealed Generation outputs before loading GT. It performs causal OOS matching, extracts paired GT/detected W300-B15 68D features, writes the full matching and exclusion ledgers, and checks the fixed per-fold floors of 596/765/767. `NO_GO` is a completed coverage finding that blocks later stages.

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c1.py cohort
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('experiments/p6/c1_detector_aligned/c1-prefix-oos-v1-seed42/train_cohort/completion_manifest.json')
m = json.loads(p.read_text(encoding='utf-8'))
print(m['status'], m['details']['common_case_coverage'])
if m['status'] != 'COMPLETE':
    raise SystemExit('STOP: C1 common Train floor NO_GO; preserve the run')
PY
```

## 4. Shared-scaler RCA, Test prediction lock, and post-lock evaluation

`fit-rca` trains B/C on exactly the common Train cases with one GT-fitted scaler. `lock-test` reads only the frozen C0 Test episode universe, hashes the C0 Test prediction file without parsing its labels, and seals every Test episode's scope/features/B/C rankings before any Test GT join. `evaluate` is the first stage allowed to read C0 Test matching and raw GT; it reports C1 paired results, A oracle context diagnostics, and C2 failure semantics. Do not run `evaluate` if `lock-test` did not seal a `COMPLETE` manifest.

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
bash -e <<'SH'
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c1.py fit-rca
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c1.py lock-test
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c1.py evaluate
SH
```

The outputs are under `experiments/p6/c1_detector_aligned/c1-prefix-oos-v1-seed42/`. The run is a reused-Test study. A zero, negative, or uncertain B→C change remains a valid outcome; no fold, threshold, or case may be changed after seeing it.
