# P6-C0 long-onset weighting: controlled Validation NO-GO

**Status: `COMPLETE_DEVELOPMENT_ONLY`, `NO_GO_DEVELOPMENT`.** No new actual
Test inference, detector threshold change, RCA run, or full E2E evaluation was
performed. The execution source and plan were committed as
`c443aeea35b4d4f2517572a58675da7b748c35ec` before either training run.
The isolated worktree is `experiment/p6-c0-long-weight`.

## Hypothesis and single intervention

The C0 detector has weak recall on isolated long and memory onsets. The
predeclared hypothesis was that their recent-onset positive windows contribute
too little gradient. The same V100 trained two runs sequentially from the
same source, seed 42, frozen data, architecture, optimizer, batch order,
checkpoint rule, and Validation event matching:

- `experiments/p6/c0_long_weight/gpu-control-v1-seed42/`: original masked BCE.
- `experiments/p6/c0_long_weight/weighted-v1-seed42/`: only the Fit loss
  multiplier changed. Of 12,897 Fit positive windows, 718 lie within 60 s of
  one of 362 complete Fit events longer than 300 s. Those bins received weight
  `3.5092790910`; the other 12,179 positive bins received `0.8520681183`.
  Negative weight stayed 1 and ignored bins stayed masked. The measured
  positive weight mass was `12,897.00004`, equal to the unweighted mass within
  float32 tolerance. Fault type, duration, and root service were not inputs to
  detector inference.

Both runs completed their fixed early-stopping rule. The GPU control selected
epoch 1 of 10; the weighted run selected epoch 3 of 12. There was no rerun or
post-result threshold, weight, or checkpoint selection.

## Validation result

All three columns use the same 2,901 complete Validation GT cases and the
same exact causal 60 s event matching. The historical C0 was CPU-trained and
is shown as an operational reference; the GPU control is the causal comparator.

| Detector | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Historical C0 | 2,124 | 13 | 777 | 0.993917 | 0.732161 | 0.843192 |
| Same-device GPU control | 2,103 | 3 | 798 | 0.998575 | 0.724922 | 0.840024 |
| Long-onset weighting | 2,105 | 8 | 796 | 0.996214 | 0.725612 | 0.839649 |

The weighted run gained 39 control-missed cases and lost 37 control-matched
cases, a net gain of 2 TP; its five extra false alarms made F1 slightly lower.
The effect was not stable across the seven UTC onset days: the weighted-minus-
control TP differences were `+2, +2, -4, -6, +17, -8, -1` in date order.

| Validation subgroup | n | Historical TP | GPU control TP | Weighted TP | Weighted gained/lost vs control |
|---|---:|---:|---:|---:|---:|
| Raw-audit-supported clean memory | 30 | 0 | 0 | 0 | 0 / 0 |
| All memory | 110 | 15 | 15 | 14 | 0 / 1 |
| Duration >300 s | 125 | 16 | 18 | 22 | 5 / 1 |
| Login failure | 2,776 | 2,108 | 2,085 | 2,083 | 34 / 36 |
| Single-onset 30 s bin | 2,521 | 1,968 | 1,949 | 1,955 | 38 / 32 |
| Multi-onset 30 s bin | 380 | 156 | 154 | 150 | 1 / 5 |

The predeclared GO gate required clean-memory gain at least 5 without losses,
TP gain at least 10 over the GPU control, weighted F1 at least the GPU
control, precision at least 0.98, absolute TP at least 2,134, F1 at least the
historical `4248/5038`, and provenance/cohort replay. Only the no-clean-
memory-loss and precision conditions passed among the seven performance
conditions. Therefore this candidate is `NO_GO_DEVELOPMENT`; it must not
proceed to Test or E2E evaluation.

## Integrity and interpretation

Both development manifests report `COMPLETE_DEVELOPMENT_ONLY` and
`test_inference_run=false`, with execution commit `c443aee...`. All eight
recorded output hashes per run were recomputed and matched, as did both config
hashes and the twelve common input artifact hashes. The input records were
identical across runs. The selected checkpoint TP/FP/FN matched the saved
Validation matching files. Exact `case_id`, fault type, service, and start/end
metadata matched across all three cohorts. The complete gained/lost IDs,
subgroups, UTC days, selected checkpoint hashes, and input hashes are in
`experiments/p6/c0_long_weight/development_comparison_v1.json`.
Before the real runs, 31 focused and existing tests plus both synthetic smoke
checks passed. The large checkpoints remain in their local run directories;
the structured selections, predictions, matching files, and hashes are
archived with this branch. Exact model-forward replay requires the local
checkpoints or retraining from the frozen execution commit.

This one-seed, reused-Validation comparison shows a small gain on long events
but no recovery of supported clean memory and no aggregate F1 improvement.
It does not prove the loss weighting is harmful in every run. The historical
CPU-to-GPU difference is not attributable to this intervention. The inherited
split audit reads frozen Test population and GT summary metadata to verify
cohort identity, while both training and checkpoint/threshold selection use
Fit and Validation only; **no actual Test detector inference or result-based
selection occurred**. Any later Test analysis would be exploratory because
this Test has informed earlier P6 research.

The earlier single-scalar Metric-drift experiment also failed its Validation
gate (clean memory `0/30`, TP `2,071`, FP `101`, F1 `0.8165`) and never ran
Test. A limited read-only sweep of historical C0 Validation scores found that
lowering the threshold from `0.99982649` to `0.9998` increased TP only from
2,124 to 2,130 while FP rose from 13 to 23; its differently defined clean
memory group rose only from `0/57` to `1/57`. The old inference source for
that score copy remains `UNVERIFIED`; this sweep was diagnostic, not a new
threshold selection.

## Decision

Stop this long-onset-weight candidate and preserve the prior detector and
two-stage results. Small threshold, global Metric-scalar, and long-onset loss
changes have not repaired the independent memory-onset gap. The existing RCA
68D extractor uses 300 s on both sides of an anchor and would read future
telemetry if applied directly at a Stage-1 prediction time. A causal 68D
trigger would be a new model requiring a frozen past-only feature window,
Fit-only training and a new independent evaluation population. For the
current thesis pipeline, Scientific Freeze is more defensible than further
sequential tuning on this reused Validation/Test setting.
