# P6-C0 long-onset loss weighting: controlled development plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-01
- Verification Status: PLANNED; no detector result implied
- Version Label: c0_long_onset_weight_v1

## Question

Does the C0 detector miss isolated long or memory onsets partly because their
recent-onset positive bins supply too little gradient? In Detector-Fit, 362
complete events last longer than 300 s, but only 718 of 12,897 positive
training bins lie within 60 s of such an onset. The first, separate experiment
that appended one global Metric drift scalar failed its Validation gates:
clean supported memory remained 0/30, and overall event F1 fell from 0.8432
to 0.8165. This plan tests a different **single training variable**, without
using that failed candidate as a baseline.

## Intervention and control

Create two independent runs from the same committed code, seed 42, and V100:

1. **GPU control**: exact C0 architecture, labels, loss, optimizer, sampler,
   checkpoint/threshold selection, and episode decoder, with loss weights 1.
   This isolates the device difference from the historical CPU-trained C0.
2. **Long-onset weighting**: identical to the GPU control except for a fixed
   per-bin multiplier in the existing masked BCE loss. Define `L` as
   Detector-Fit positive windows whose prediction time is within 0–60 s after
   any complete Detector-Fit event lasting >300 s. Define `S` as the other
   positive windows. Let `r=sqrt(|S|/|L|)` and
   `c=(|S|+|L|)/(|S|+r|L|)`. A long-positive bin receives `c*r`; another
   positive bin receives `c`; negative bins receive `1`; ignored bins retain
   loss weight `0`. Thus total positive weight equals the original positive
   count. These values are computed solely from Detector-Fit events/windows,
   recorded in the selection manifest, and never chosen on Validation/Test.

The model still sees only the original Metric/Log/Trace windows and one
system-level recent-onset label. Fault type, root service and duration are not
model inputs. Duration is used only for the Detector-Fit loss multiplier.
The 30 s grid, 300 s history, exact 60 s causal matching, 50/20/30 split,
preprocessed arrays, ignore semantics, original C0 architecture, batch 32,
AdaBelief settings, epoch 30/patience 8, seed, and Validation event-F1
selection remain fixed. No Test checkpoint, threshold, weight or offset sweep.

## Inputs, outputs and execution

- Worktree: `/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2-c0weight`, branch
  `experiment/p6-c0-long-weight`.
- New run directories: `experiments/p6/c0_long_weight/gpu-control-v1-seed42/`
  and `experiments/p6/c0_long_weight/weighted-v1-seed42/`; neither may exist
  before its run.
- Frozen data/artifacts/registry are referenced read-only from the original
  `Ada-MGAD-e2e-v2` checkout by absolute path. No preprocessing is rerun.
- The `develop` action does split audit, Fit training, Validation scoring and
  exact threshold selection. It never runs actual Test model inference. Each
  run stores source/input SHA bindings, selected checkpoint, full Validation
  predictions/episodes/matching, and a completion manifest.
- Run the GPU control first, then the weighted candidate, using the two
  committed configs. Execute the two commands separately and sequentially:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 python scripts/p6/run_c0_trigger.py develop \
    --config configs/e2e/gaia_p6_c0_long_weight_control_v1.json \
    --data-root /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/data/p5/v3_preprocessing_v2/ad \
    --artifact-root /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/artifacts/p5/v3_preprocessing_v2/ad \
    --registry /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/artifacts/p5/v3/protocol/gt_event_registry.csv \
    --gpu true --threads 8 --threshold-workers 8

  PYTHONDONTWRITEBYTECODE=1 python scripts/p6/run_c0_trigger.py develop \
    --config configs/e2e/gaia_p6_c0_long_weight_weighted_v1.json \
    --data-root /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/data/p5/v3_preprocessing_v2/ad \
    --artifact-root /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/artifacts/p5/v3_preprocessing_v2/ad \
    --registry /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/artifacts/p5/v3/protocol/gt_event_registry.csv \
    --gpu true --threads 8 --threshold-workers 8
  ```

No code or configuration is changed between their runs. Monitor
the live process, epoch log, and GPU. Never treat a partial directory as a
completed run.

## Fixed Validation GO / NO-GO

Before considering one exploratory Test run, require all of the following on
the selected checkpoints and the same 2,901-case Validation cohort:

1. The weighted candidate gains at least five of the 30 raw-audit-supported
   clean memory onsets versus the GPU control, losing none of its clean-memory
   matches.
2. Weighted TP exceeds GPU-control TP by at least 10, weighted event F1 is
   at least GPU-control F1, and weighted precision is at least 0.98.
3. Weighted TP is at least 2,134, and F1 is at least the exact historical C0
   value `4248/5038 = 0.8431917427550615`.
4. Both runs pass source/input identity, Fit-only weight, exact cohort,
   checkpoint replay and artifact-hash checks.

These are engineering development gates, not significance tests. Report
paired gained/lost IDs, memory/long/login/single/multi onset strata and UTC
day blocks even on NO-GO. Do not alter the formula or choose another epoch or
threshold after seeing Validation. If any gate fails, do not run actual Test.

If GO, commit and lock the selected detector source/checkpoint/threshold and
create a **label-free** Test scores/episodes lock; do the same for the fixed
68D+XGBRanker and 25.621 s Train-derived RCA backdating on the new episode
cohort. Only after both locks may Test GT be joined to report Stage-1 and full
Diagnosis P/R/F1@1/3/5. Existing Test has been reused, so any such result is
exploratory, not independent confirmation.
