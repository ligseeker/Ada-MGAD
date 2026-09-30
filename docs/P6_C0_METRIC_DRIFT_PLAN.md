# P6-C0 Metric temporal drift detector experiment

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-01
- Verification Status: PLANNED; no result is implied by this document
- Version Label: c0_metric_drift_plan_v1

## Question and evidence

Can one causal, service-agnostic Metric change statistic help the C0 detector
recognize isolated long or `memory_anomalies` onsets without sacrificing event
precision? The frozen C0 Validation detector finds 0/31 clean memory onsets at
its selected threshold. A read-only Fit/Validation raw-input audit found a
descriptive Metric-change AUC of 0.7984/0.6795 for clean memory versus sampled
quiet pseudo-onsets. This is weak, post hoc motivation, not an expected gain.
The C0R2 repeated-trigger rule did not improve clean-context recall relative to
equal-alert controls, so it is not part of this experiment.

## Single model variable

Keep the C0 50/20/30 chronology, frozen preprocessed arrays, 30 s grid, recent
onset labels, ignore mask, 60 s causal matching, encoder, optimizer, loss,
checkpoint/threshold selection, episode decoder, seed, and all training
hyperparameters. Add **one scalar residual input** to the existing system
logit. For each 10-bin Metric window `x` with shape `[10,10,48]`:

1. `b = median(x[0:8], time)` for each service and feature;
2. `d_j = mean(abs(x[j] - b), service, feature)` for `j = 8,9`;
3. `d = max(d_8,d_9)`;
4. `z = (d - Fit median(d)) / Fit IQR(d)`; reject zero/nonfinite IQR;
5. `new_logit = original_logit + alpha*z`, where `alpha` is one trainable
   scalar initialized at zero.

The Fit median/IQR use **all Detector-Fit windows only**, with no Validation or
Test information. No service is selected. The zero-initialized residual adds
no random draw and preserves the baseline model's initial logits. It tests
whether an explicit recent change signal can be learned; it does not claim that
the six-statistic raw audit identified an optimal feature.

## Development run and comparison

- Isolated branch/worktree: `experiment/p6-c0-metric-drift` in
  `/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2-c0metric`.
- Run directory: `experiments/p6/c0_metric_drift/c0-metric-drift-v1-seed42/`.
- Input arrays and registry are read-only references to the original
  `Ada-MGAD-e2e-v2` checkout. Never write into its P5 or C0 runs.
- Run one candidate with the original C0 Validation event-F1 checkpoint and
  exact-score threshold rule. No architecture, scale, threshold, epoch,
  sampler, or loss search after seeing Validation outcomes.
- After the implementation/tests are committed, execute exactly one real
  development run from this worktree (the `develop` action audits and trains;
  it does not call actual Test inference):

  ```bash
  PYTHONDONTWRITEBYTECODE=1 python scripts/p6/run_c0_trigger.py develop \
    --config configs/e2e/gaia_p6_c0_metric_drift_v1.json \
    --data-root /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/data/p5/v3_preprocessing_v2/ad \
    --artifact-root /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/artifacts/p5/v3_preprocessing_v2/ad \
    --registry /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/artifacts/p5/v3/protocol/gt_event_registry.csv \
    --gpu true --threads 8 --threshold-workers 8
  ```
- Replay the original Validation matching from the sealed C0 result as the
  comparison denominator. Report event TP/FP/FN/P/R/F1; clean memory, all
  memory, long, login, single-/multi-onset recall; paired gained/lost case IDs;
  and selected threshold, epoch, and Fit drift median/IQR.
- Confirm that a zero residual reproduces C0 initial model logits for a fixed
  seeded batch and that all drift calculations use only past/current bins.

## GO / NO-GO before any new Test inference

GO to a single exploratory Test evaluation only if the fixed selected model
meets **all** of these on Validation:

1. At least 5 additional correctly matched clean memory onsets among the 30
   raw-audit-supported cases, with no clean-memory case lost relative to C0;
2. overall event TP exceeds C0's 2,124 by at least 10 and event F1 is no lower
   than C0's 0.8432;
3. event precision is at least 0.98, and the split/identity/label and metric
   normalization checks all pass.

These are practical development gates, not significance tests. If any fail,
report NO-GO and do not run the new detector on Test. Do not change the rule or
try another feature, alpha initialization, seed, or threshold on this same
Validation result.

If GO, lock the source, model, Fit statistic, checkpoint, and selected
Validation threshold before Test inference. Test output uses a new directory.
To assess the complete two-stage pipeline, apply the already frozen 68D +
XGBRanker RCA model with its Train-derived 25.621 s anchor backdating to every
new legal detected episode, preserving all misses, false alarms and invalid
rankings in the diagnosis denominator. This is an exploratory reused-Test
comparison; it is not independent confirmation. Report both Stage-1 and full
Diagnosis P/R/F1@1/3/5 against the prior `0.7469` F1@1 pipeline.

## Monitoring and outputs

Use the V100 if training starts. Watch the live PID, training log, GPU usage,
and `validation_selection.json`; no orphan training process or partial run is
treated as complete. The new run directory must not pre-exist. Save source and
input hashes, config, Fit statistic, checkpoint, full Validation predictions,
paired case ledger, and a completion report. Keep large tensors/checkpoints
local and archive structured evidence in Git as appropriate.
