# P6-C1-v2 supervision-OOS detector-aligned RCA protocol

## Material Passport

- Protocol ID: `P6-C1-SUPERVISION-OOS-v1`
- Run ID: `c1-supervision-oos-v1-seed42`
- Machine-readable lock: [`configs/e2e/gaia_p6_c1_v2_supervision_oos.json`](../configs/e2e/gaia_p6_c1_v2_supervision_oos.json)
  (14,784 bytes, SHA-256 `4e4529db1049fa5c04e658193b6b4e91eef0d9ad9e5d0efcd785dc776c475874`).
  Every execution stage must verify this hash before writing outputs.
- Freeze date: 2026-09-29. Status: `DESIGN_LOCKED_G1_STATIC_AUDIT_REQUIRED`.
- This document and the JSON above are frozen **before** any fold detector,
  Generation episode, cohort or Test result of this run exists.

## 1. Research question

With the RCA representation, the RCA model, the ten canonical candidate
services and the Train case cohort held fixed, does replacing the RCA Train
anchor from the GT onset to the detector-produced OOS detected timestamp change
detected-anchor root-service ranking?

```text
Arm B: Train = GT anchor       Test = detected t_hat
Arm C: Train = OOS detected    Test = detected t_hat
Primary estimand: paired AC@1(C) - AC@1(B)
```

Arm A (GT Train -> GT Test) is a post-lock oracle diagnostic only; it is not a
deployment path. No performance gate is imposed: `C > B`, `C ≈ B` and `C < B`
are all accepted validity outcomes and must be reported as such.

## 2. Evidence grade and information boundary

This round reuses the already frozen shared preprocessing produced by the
original P5/P6 V2 pipeline (Metric schema/statistics, Log Drain3
vocabulary/scales, Trace graph/scales, 30 s grid). Therefore this protocol may
claim only:

```text
supervision-OOS
detector-parameter-OOS
```

It must **not** claim strict prefix-preprocessing-OOS or full-pipeline
forward-OOS. The shared preprocessing was fitted once on the original 70 % Train
period with label-free decisions and may have observed label-free Train
telemetry later than a fold's Generation interval. It never used Generation
labels, root labels, fault labels or Test telemetry. This is a pre-registered
evidence boundary, not a post-hoc explanation.

The failed strict prefix run `c1-prefix-oos-v1-seed42`
(`PREFIX_SCHEMA_NO_GO`, fold 1 Metric slots 27/45) is preserved, is not resumed,
is not repaired and its run ID is not reused. Its `run_lock.json` is bound in
this protocol as non-reuse evidence.

## 3. Frozen shared inputs

All inputs are read-only and bound by path, byte count and SHA-256 in
`bindings` of the JSON lock. The essential ones:

| Binding | Role |
|---|---|
| `configs/e2e/gaia_p5_v3_preprocessing_v2.json` | shared preprocessing and timeline |
| `configs/e2e/gaia_p6_c0_system_trigger.json` | frozen detector architecture/training/selection rules |
| `data/p5/v3_preprocessing_v2/ad/train/{timestamps,metric,log,trace}.npy` | shared detector input arrays |
| `data/p5/v3_preprocessing_v2/ad/graph.npy` | detector static graph |
| `artifacts/p5/v3_preprocessing_v2/ad/ad_data_manifest.json`, `artifacts/p5/v3_preprocessing_v2/schema/frozen_preprocessing_schema.json` | shared array/schema provenance |
| `artifacts/p5/v3/protocol/gt_event_registry.csv` | trigger supervision + Train matching |
| `data/p5/v3/rca_raw_index/index_manifest.json` | label-free RCA feature source |
| `experiments/p6/system_event_trigger/{test_episodes.csv,test_matching.csv,test_metrics.json,test_predictions.csv}` | frozen C0 Test Stage-1 population and evaluation-only material |

Recorded frozen array geometry:

```text
Train metric [60984, 10, 48], log [60984, 10, 32], trace [60984, 10, 10, 8]
Train timestamps 1625133600000 .. 1626963090000 (30 s grid, 60,984 bins)
Test  timestamps 1626963120000 .. 1627747170000 (26,136 bins)
```

These inputs must not be modified. The `label*.npy` node-anomaly arrays are not
bound and are never read by this protocol.

## 4. Fold boundaries (unchanged from G2 `fit_only_3fold`)

All intervals are UTC, half-open, aligned to the 30 s grid and strictly
chronological. Generation intervals are pairwise disjoint, so a case belongs to
at most one Generation fold.

| Fold | Fit | Selection | Generation | Static GT/context upper bound | Cohort floor |
|---|---|---|---:|---:|---:|
| 1 | Jul 1 10:00 – Jul 4 10:36 | Jul 4 10:36 – Jul 7 11:12 | Jul 7 11:12 – Jul 10 11:48 | 1,192 | 596 |
| 2 | Jul 1 10:00 – Jul 7 11:12 | Jul 7 11:12 – Jul 10 11:48 | Jul 10 11:48 – Jul 13 12:24 | 1,529 | 765 |
| 3 | Jul 1 10:00 – Jul 10 11:48 | Jul 10 11:48 – Jul 13 12:24 | Jul 13 12:24 – Jul 16 13:00 | 1,533 | 767 |

Floors are `ceil(upper bound / 2)` and were fixed before execution. They are a
performance-blind feasibility gate, not a recall prediction. The static upper
bounds are G1 static counts, not observed anchors.

## 5. Shared-array fold slicing

The fold adapter only:

```text
read frozen shared arrays
slice by the locked timestamps
create legal 300 s windows
```

It must not re-fit the Metric schema, the scaler/statistics, Drain3 or the
graph. Each fold materialization binds the frozen source array hashes plus the
per-segment slice hashes and asserts identity across folds: same schema hash,
same graph, same canonical service order, same dimensions (48/32/8), and 30 s
grid monotonicity. Segment ownership: a window belongs to the segment owning its
prediction-available time and its 300 s history must lie wholly inside that
segment. Legal windows are sample indices `[0, time_bins - window_bins)`; the
window whose prediction time equals the segment end is scored in neither
segment (frozen G3 convention).

## 6. Detector: copied frozen P6-C0 model and training rules

Each fold builds a fresh, randomly initialised `SystemEventTrigger`; no P6-C0,
Ada-MGAD or other-fold checkpoint is ever loaded. Architecture: reused
multimodal encoder + dynamic graph learner + permutation-invariant service
pooling (`mean + max + std`) + `48 -> 32 -> 1` system head. Forbidden: node
anomaly head, service ranking head, root label, fault label, label-conditioned
reconstruction, node-score fusion.

Training values are read from the bound C0 config (source of truth), not guessed
here: max epochs 30, patience 8, batch 32, 2 loader workers, AdaBelief
(lr 1e-3, weight decay 5e-4), StepLR (step 10, gamma 0.5), grad clip 10, masked
BCE with `pos_weight = fit_negatives / fit_positives` over Fit only, plus the
label-free dynamic-graph sparsity regularizer (`graph_sparse_weight = 1e-3`).
Seed 42, Torch threads 8. No focal loss, no fault/duration/service weighting, no
hyper-parameter search.

Trigger labels keep the frozen P6-C0 definition on the prediction-time grid:

```text
POSITIVE: 0 <= prediction_available_time - gt_start_ms <= 60 s
IGNORE:   ongoing event older than 60 s (and not POSITIVE)
NEGATIVE: neither
```

No ERROR/unsupported/unknown class is reintroduced.

## 7. Selection rule

Every epoch is evaluated on that fold's Selection segment only: system scores ->
exact unique-score threshold sweep -> episode construction -> 60 s causal
matching -> Event P/R/F1. Checkpoint choice: highest Event F1, ties by higher
Recall, then higher threshold. Generation and Test metrics never select detector
state.

## 8. Generation label firewall and seal

The Generation loader has no label parameter and no label access path. It never
receives `trigger_label`, `root_service`, `fault_type` or GT event identity. It
produces only: `episode_id`, `t_det` (prediction-available time), episode start,
episode end, system score, fold id and source sample ids. After scoring, the
fold writes a completion manifest with source and output hashes and seals the
episode table. Train-side GT matching may read it only afterwards.

## 9. OOS Train matching and cohort gate

Sealed Generation episodes are joined to the Generation GT registry with the
frozen causal one-to-one matching rule `0 <= t_det - GT onset <= 60 s`
(max-cardinality, then minimum total delay). Matched events, unmatched GT,
false alarms, boundary exclusions and illegal contexts are all retained.

A legal paired case must have a unique GT `case_id`, a unique matched
prediction, both W300 contexts (`[GT onset ± 300 s)`, `[t_det ± 300 s)`) inside
the same Generation segment and the raw timeline, and finite GT/DET feature
tensors with the ten canonical candidates in order.

If any fold's legal common cohort is below its floor, the run is
`P6-C1-v2 FEASIBILITY = NO-GO`: stop immediately, do not fit RCA, do not touch
Test, do not change thresholds, folds or floors, and preserve every artifact and
the failure reason. Exclusion accounting categories: unmatched, boundary,
GT context illegal, DET context illegal, feature failure, duplicate, candidate
mismatch. The cohort is never selected by model correctness or Test labels. The
final common Train case IDs are identical for arms A, B and C.

## 10. Frozen RCA representation, scaler and arms

GAIA E2E adapter representation only: W300-B15, 40 bins, channels
metric/log/trace-error/trace-latency, Z2, 10 candidates x 68D. The canonical
Ada-RCA W600-B15 form is explicitly **not** used this round, and no 30 s RCA bin
or feature redesign is allowed.

One `StandardScaler` is fitted on the GT-anchor candidate rows of the common
Train cohort, frozen, and applied identically to GT Train, DET Train, GT Test
and DET Test. Arms B and C fit the frozen Conditional Logit
(`l2_lambda = 1.0`, max_iter 1000, gtol 1e-8, unit case weights) on the same
case IDs, roots, candidates, candidate order, scaler, optimizer, tolerance,
initialisation and weights. No hyper-parameter search.

Arm A uses B's weights on GT-anchor Test features and is generated only after
the B/C prediction lock.

## 11. Test input and prediction lock

Test input is the frozen P6-C0 `test_episodes.csv`; the C0 detector is never
re-run and its threshold is never reselected. Frozen Stage-1 Test result quoted
as-is: TP 4198, FP 16, FN 1589, P 0.9962, R 0.7254, F1 0.8395.

Lock procedure:

1. Read only the label-free columns of `test_episodes.csv`
   (`prediction_id, split, t_hat, episode_end_time, positive_bins, system_score,
   threshold`). `test_predictions.csv` (contains `trigger_label`) is hash-only
   provenance and is never parsed; `test_matching.csv` is evaluator-only.
2. Rank **every** legal predicted Test episode with B and C, including episodes
   that will later be false alarms. Feature-extraction or ranking failures are
   kept as failure rows, never dropped.
3. Before reading any Test GT, write the ordered episode universe, B/C
   rankings, feature availability, failure flags, model hashes, scaler hash and
   source hashes, then `prediction_lock.json` + `completion_manifest.json` with
   SHA-256 for every file. The prediction directory is immutable after this
   point.
4. Only after a successful lock may Test GT, matching, root labels and fault
   labels be read for evaluation.

Missing or invalid rankings contribute zero correctness for that arm and stay in
the denominator.

## 12. C1 evaluation

Primary: paired `AC@1(C) - AC@1(B)` on the same locked legal matched cohort,
with the 2x2 paired transition counts (both correct, B only, C only, both
wrong). Secondary: AC@3, AC@5, MRR, per-root and per-fault groups with `n`,
weighted overall and prespecified macro (`n = 0` undefined and listed,
`1 <= n < 20` descriptive only, `n >= 20` in the macro; the minimum-n rule is
never changed to improve a macro). Uncertainty: paired 95 % percentile interval
from 10,000 UTC onset-day cluster bootstrap replicates, seed 42 — a descriptive
summary on the reused Test, not an independent confirmatory test. Raw
numerators/denominators are reported without intermediate rounding.

## 13. C2 (runs regardless of the C1 sign)

Layer 1 quotes the frozen P6-C0 Stage-1 event metrics (TP/FP/FN, P/R/F1,
mean/median/P95 delay) without re-optimisation. Layer 2 reports A/B/C on the
same locked matched cohort. Layer 3 reports full-diagnosis Precision/Recall/F1
at 1/3/5 for B and C, counting Stage-1 misses, false alarms, invalid RCA
contexts, missing rankings and wrong rankings, with raw numerators and
denominators. Every raw GT event and predicted episode lands in exactly one
ledger category: `EVENT_MISSED`, `EVENT_FALSE_ALARM`, `RCA_CONTEXT_INVALID`,
`RCA_RANKING_MISSING`, `ROOT_OUTSIDE_TOP1`, `ROOT_OUTSIDE_TOP3`,
`ROOT_OUTSIDE_TOP5`, `SUCCESS_TOP1`. Stratification by duration, onset
multiplicity, fault and root service is reported with `n` and is used for
interpretation only.

Latency definitions are kept separate: detection latency `t_det - GT onset`;
RCA data-ready latency `(t_det + 300 s) - GT onset`; final diagnosis compute
latency `UNMEASURED` (no wall-clock instrumentation exists). The system is
described as online event triggering plus delayed root-cause diagnosis, not as
real-time RCA.

## 14. Explicitly forbidden in this round

```text
C0R2 / Stage-1 retraining / threshold rescue / duration or fault weighting
strict-prefix C1 rescue, Metric 45-slot changes, correlation threshold changes,
future padding, reusing c1-prefix-oos-v1-seed42
W300 -> W600, 15 s -> 30 s, 68D feature redesign
anchor offsets (t_det-15s / t_det-30s), offset sweeps, multi-anchor ensembles
GNN/Transformer/tree rankers, new feature fusion, joint AD-RCA training
Test hyper-parameter search
```

## 15. Provenance and STOP conditions

Every stage records git HEAD, source digest, config SHA, input hashes, output
hashes, seed, Torch thread count, DataLoader workers, environment, Python and
package versions, timestamps, fold bounds, checkpoint/threshold/model/scaler/
cohort/prediction-lock hashes. Bitwise reproducibility is not claimed unless
verified.

Automatic STOP (preserve the failed directory, write `failure.json` +
`INCOMPLETE.json`, commit the failure evidence, do not rescue):
input/hash mismatch, split violation, Generation label leakage, Test access
before lock, fold cohort below floor, non-finite features, case identity
mismatch, candidate order mismatch, shared scaler mismatch, RCA training
failure, prediction lock failure.

After a valid C1 and C2 the project defaults to Scientific Freeze: no further
model experiments unless the user explicitly opens a new research question.
