# P6-C0 Root-Agnostic System Event Trigger — Protocol

Status: **frozen before the formal Test evaluation**. This document defines the
task, labels, split, purge, architecture, loss, selection rules, event matching
and the acceptance gate. It is written and committed before any Test number was
produced. No value in this document was chosen by looking at Test.

Experiment root: `experiments/p6/system_event_trigger/`
Driver: `scripts/p6/run_c0_trigger.py`
Config: `configs/e2e/gaia_p6_c0_system_trigger.json`
Helpers: `src/e2e/system_trigger.py`, `src/e2e/system_trigger_data.py`,
`src/e2e/system_trigger_model.py`

---

## 1. Motivation and scope

P6-B0 established two facts about the frozen Ada-MGAD detector:

1. the node-level supervision target *is* the RCA-labelled root service
   (`labelled_service_positive_on_all_bins_ratio = 1.0000`), and
2. the reconstruction branch is label-conditioned by construction
   (`src/model.py:164-176`), so re-weighting classification vs reconstruction
   scores cannot decouple AD from RCA.

P6-C0 therefore builds **one new Stage-1 detector whose supervision never
mentions a service**. Ada-MGAD (the original standalone method) is untouched: no
preprocessing, no checkpoint, no loss, no fusion, no calibration and no formal
P5 artifact is modified.

### 1.1 Task definition — System-level Event Trigger

```text
Telemetry (metrics / logs / traces, 300 s history, 30 s grid)
        ↓
multimodal + graph encoder
        ↓
system trigger score            (one scalar per prediction timestamp)
        ↓
predicted event episode         (consecutive positive timestamps)
        ↓
detected t_hat                  (first positive prediction_available_time)
```

The detector **must not** output: a root service, a service ranking, or any
ten-way service classification result. Root localisation is Stage 2 (Ada-RCA)
and is explicitly out of scope this round.

---

## 2. Frozen inputs and reused protocol

Reused without modification:

| item | frozen value |
|---|---|
| preprocessing | `data/p5/v3_preprocessing_v2/ad/{train,test}` arrays + `graph.npy` (never re-generated) |
| GT registry | `artifacts/p5/v3/protocol/gt_event_registry.csv` (SHA-checked by `load_registry`) |
| grid | 30 s |
| input window | 10 bins = 300 s history |
| prediction time | `t_hat = prediction_available_time = target_bin_end` |
| matching tolerance | 60 s, causal (`0 <= t_hat - gt_start_ms <= 60000`) |
| matching semantics | causal max-cardinality + minimum-delay (`match_events`) |
| episode rule | consecutive positive prediction-available timestamps (`construct_predicted_episodes`) |
| fault taxonomy | the frozen `SUPPORTED_FAULT_TYPES`; `normal`, `ERROR`, traceback continuations and unsupported rows stay excluded |
| Test block | `[boundary_ms, absolute_end_ms)` — bit-identical to the frozen P5 Test block |

The Test window population is asserted to be **identical** to the frozen one
(same count and same sample indices `0..N-1` of the frozen Test array) by the
audit's `test_population_matches_frozen_p5` and
`test_window_identity_matches_frozen_p5` checks.

Not reused / not touched: Ada-MGAD checkpoints, the reconstruction calibration,
the fused score, the conditional logit, the 68D RCA representation, Ada-RCA, and
the P6-B0 outputs.

---

## 3. Trigger label — recent-onset target

For a prediction time `t` and a legal GT event with onset `t_start` and end
`t_end`:

```text
POSITIVE : exists an event with      0 <= t - t_start <= 60 s
IGNORE   : t_start + 60 s < t < t_end for some event, and no POSITIVE condition
           holds at t                (an ongoing fault older than 60 s)
NEGATIVE : neither of the above
```

Precedence is `POSITIVE > IGNORE > NEGATIVE`, so overlapping events can only
produce one system-level POSITIVE. The label is a single scalar per timestamp:
there is no service axis, and an 11 s event and a 3600 s event contribute the
same bounded number of positive supervision bins (verified by unit test).

**Evaluation grid.** `t` is the *prediction time*
`prediction_available_time = target_bin_end = target_bin_start + 30 s`, so the
label is rasterized on `timestamps + 30 s` (the `prediction_time_grid`), not on
the bin-start grid. Indexing that array with a window's target bin index yields
the state at that window's prediction time, which is the anchor `t_hat` the
episode/matching code uses. Rasterizing on the bin-start grid instead would lag
the supervision target by one bin relative to the evaluation anchor; the driver
fails closed on that (`assert_prediction_time_grid`) and the contract is pinned
by unit tests (`LabelGridAlignmentTests`, `LabelPredictionTimeContractTests`).

### 3.1 Why not "duration overlap"

The frozen AD node label marks every 30 s bin overlapping an injection. That
makes the supervision weight of an event proportional to its duration
(11 s → 1 bin, 3600 s → 120 bins) and makes a trigger trivially learnable from
the ignore region. The recent-onset target instead supervises *onset
responsiveness* only, which is exactly what event triggering needs.

### 3.2 Label source population

Labels are built from **every GT event inside the frozen metric detector
timeline** (`detector_domain == True`, 16,132 rows), including the single event
that crosses the frozen 70/30 boundary. Reason: the label must describe the true
fault state at time `t`; dropping a crossing event would inject false negatives
during an ongoing fault. The *metric* populations (Section 4.2) only use
complete in-block events.

---

## 4. Split — chronological 50 / 20 / 30

### 4.1 Blocks

Same deterministic construction as the frozen split, applied at 50 % and 70 %
(`K = (numerator * N) // denominator` on the 30 s metric grid):

```text
Detector-Fit        first 50 %   gradient updates + Fit-only statistics
Detector-Validation next 20 %    checkpoint selection + threshold selection
Final Test          last  30 %   observation only, frozen checkpoint/threshold
```

The Test block starts at the frozen `boundary_ms`, so the final Test window
population is unchanged. No random split is used anywhere.

### 4.2 Window and event purge

```text
window : kept only if its whole [t - 300 s, t) input lies inside the block
         owning t; otherwise purged (deterministic, >= 300 s purge)
event  : a metric case only if the complete injection lies inside one block
         (frozen assign_event_blocks rule); an event that crosses a split
         boundary is purged from every metric population
```

The purge is recorded as: purged windows (per source array, per boundary, per
reason), purged window label states (positive/ignore/negative), purged events
with their case IDs, and the label-legal event population.

### 4.3 Split feasibility audit — STOP condition

Running `audit` writes `split_audit.json` and `split_manifest.json`. The
following checks are pre-registered; **any failure stops the round** and no
training is started:

```text
fit_has_gt_events / validation_has_gt_events / test_has_gt_events
fit_has_positive_labels / validation_has_positive_labels / test_has_positive_labels
dominant_fault_type_<x>_present_in_fit      for every fault type with >= 10 % of Test GT
dominant_service_<x>_present_in_fit         for every root service with >= 10 % of Test GT
test_population_matches_frozen_p5
test_window_identity_matches_frozen_p5
```

The audit also reports window counts, Positive/Ignore/Negative bins, GT event
counts, fault-type and root-service distributions (audit only — never a model
input), duration distributions in the strata `<=15s / 15-30s / 30-60s /
60-300s / >300s`, and single/multi-onset bin statistics, for all three splits.

---

## 5. Architecture

```text
Metrics / Logs / Traces  (frozen 300 s windows)
        ↓  Embed (reused from src/model_util)
        ↓  DynamicGraphLearner (reused, label-free)
        ↓  Encoder (reused: spatial + temporal attention + FFN, 2 layers)
H       ten service representations at the target bin        shape (B, 10, 16)
        ↓  permutation-invariant pooling over the service axis
        ↓  concat[mean_service(H), max_service(H), std_service(H)]   shape (B, 48)
        ↓  MLP 48 -> 32 -> 1  (LeakyReLU)
        ↓
one scalar system trigger logit
```

Reused hyperparameters (from the frozen `ad_model` block): `feature_node=16`,
`feature_edge=4`, `feature_log=8`, heads `4/4/4/4/2`, `num_layer=2`,
`dropout=0.2`, `graph_hidden=16`, `graph_sparse_weight=0.001`.

New parts: the pooling and the scalar head.

* **Random initialisation.** No Ada-MGAD checkpoint is loaded; the driver has no
  code path that can load one.
* **Permutation invariance.** `pool_service_representations` is invariant to any
  permutation of the ten service representations; permuting `H` leaves the
  trigger logit numerically unchanged (unit tested). Scope note: the graph
  encoder upstream still consumes the canonical GAIA service order because the
  static/fused trace graph is defined by it. The invariance is guaranteed at the
  pooling/head interface, which is where "no fixed service position is exposed
  to the head" must hold.
* **No label paths.** The detector consumes no node anomaly label, no
  label-conditioned reconstruction objective, no node-score fusion and no
  root-service identity.

---

## 6. Loss

```text
loss = masked BCEWithLogitsLoss(system_logit, POSITIVE)
     + graph_sparse_weight * dynamic_graph_regularizer
mask: IGNORE bins contribute 0; NEGATIVE and POSITIVE bins contribute 1
pos_weight = (# Fit NEGATIVE bins) / (# Fit POSITIVE bins), computed on Detector-Fit only
optimizer  = AdaBelief(lr = 1e-3, weight_decay = 5e-4)
scheduler  = StepLR(step_size = 10, gamma = 0.5)
grad clip  = 10.0 (L2 norm)
batch size = 32 (frozen padded-sampler geometry), num_workers = 2, seed = 42
max_epochs = 30, early stopping patience = 8 (Validation event F1)
```

The only auxiliary term is the dynamic-graph sparsity regularizer. It depends
only on the encoder inputs `(x_node, x_log)` and the static service graph — it
never sees a label of any kind — and it is kept at the frozen weight `0.001`.
No focal loss, no fault-type/service/duration weighting, no Test-tuned class
weights.

---

## 7. Checkpoint and threshold selection

Every epoch, after the Fit pass:

```text
1. deterministic Validation inference (eval mode, no gradients)
2. enumerate the exact unique Validation system scores as candidate thresholds
3. for each candidate: episodes -> frozen 60 s causal matching -> Validation Event F1
4. threshold = argmax Validation Event F1; equal F1 -> highest threshold
5. record Validation P/R/F1 and mean/median/P95 delay
```

Checkpoint selection across epochs:

```text
maximise Validation Event F1
tie 1: higher Validation Event Recall
tie 2: higher Validation threshold
```

After selection the checkpoint and the threshold are frozen; `evaluate` refuses
to run without `validation_selection.json` and refuses to proceed if the
checkpoint SHA-256 changed. It also replays the Validation metrics at the frozen
threshold and aborts if they do not reproduce.

**Note on the tie-break.** The pre-registered P5 rule "highest threshold among
equal Train event F1" is preserved *inside* the threshold enumeration (step 4).
The extra epoch-level rule (F1 → Recall → threshold) is a new rule for a new
selection dimension (which checkpoint), pre-registered here; it does not modify
the frozen threshold rule and does not conflict with it.

Test is evaluated **once**, with the frozen checkpoint and the frozen threshold.
No Test-based threshold, epoch, architecture, `pos_weight` or gate adjustment is
performed.

---

## 8. Event episodes and matching

Unchanged from the frozen protocol:

```text
score >= frozen threshold -> positive timestamps
consecutive positive prediction-available timestamps merge into one episode
t_hat = first positive prediction_available_time
matching: 0 <= t_hat - gt_start_ms <= 60 s, causal max-cardinality + minimum-delay
```

Threshold selection reuses the frozen fast causal counter
(`_event_detection._event_confusion_counts`), and the reported Test metrics are
recomputed through the frozen `construct_predicted_episodes` / `match_events` /
`event_metrics` functions.

---

## 9. Test outputs

Overall: GT events, predicted episodes, TP/FP/FN, Precision, Recall, F1, mean /
median / P95 delay.

Bin diagnostic (auxiliary only, never a headline result): AUROC, AP, BCE, and
Positive/Ignore/Negative bin counts, with IGNORE bins masked out.

Stratifications: duration strata (`<=15s / 15-30s / 30-60s / 60-300s / >300s`)
with n / TP / FN / Recall / delays; fault type (n / recall / small-n flag);
root service (audit only); onset density with single-onset vs multi-onset
recall, the onset-count histogram, and same-bin multi-event / multi-root counts.

---

## 10. Pre-registered acceptance gate

```text
Precision >= 0.90
Recall    >= 0.60
F1        >= 0.72
```

These are **pre-registered engineering acceptance margins**, not statistical
significance thresholds. The historical node-supervised fused Ada-MGAD trigger
(`P 0.9827 / R 0.6399 / F1 0.7751`) is quoted as a historical reference only: it
has a different supervision target, label definition and split, so this is not a
controlled equal-protocol comparison.

Decision rule (pre-registered):

```text
GO          : all three margins hold AND no stratification failure
BORDERLINE  : not GO, but F1 >= 0.60 and Recall >= 0.40,
              OR the overall gate holds while a stratification failure exists
NO-GO       : F1 < 0.60 or Recall < 0.40

stratification failure
  duration : a duration stratum with n >= 30 and Recall < 0.25
  onset    : multi-onset Recall < 0.5 x single-onset Recall when both n >= 30
  fault    : a fault type with n >= 30 and Recall < 0.25
```

On BORDERLINE only a failure audit is produced: no Test tuning, no threshold
rescue, no architecture search, no RCA, no node-level supervision fallback.

---

## 11. Prohibited in this round

No modification of the original Ada-MGAD method, Ada-RCA, the 68D features or
the conditional logit; no Conditional-Logit training; no P6-C1 anchor alignment;
no final E2E diagnosis; no Test threshold tuning; no alpha sweep; no anchor
offset correction; no reintroduction of `ERROR`; no change to the GT taxonomy,
the 30 s grid or the 300 s history; no specialisation for `login_failure` or
`mobservic*`; no root-service identity as a feature or target; no Ada-MGAD
checkpoint as initialisation; no re-preprocessing of raw telemetry.

---

## 12. Provenance recorded

`manifest.json` + `split_manifest.json` + `validation_selection.json` +
`test_metrics.json` record: trigger config and its SHA-256, frozen base config
and its SHA-256, source artifact SHA-256s (all frozen Train/Test arrays, graph,
`ad_data_manifest.json`, GT event registry), random seed, torch thread count,
git commit, checkpoint path + SHA-256, selected epoch, selected Validation
threshold, Fit/Validation/Test block bounds, purged windows/events and label
statistics, parameter count, model args, training hyperparameters, and the
policy flags (`no_retraining_of_ada_mgad`, `ada_mgad_checkpoint_loaded`,
`node_anomaly_labels_read`, `root_service_label_used`, `rca_retrained`,
`conditional_logit_touched`, `test_used_for_checkpoint_selection`,
`test_used_for_threshold_selection`, `reintroduced_error_class`).

Execution note: the process thread count (`--threads`, recorded as
`torch_threads`) is a runtime cost control, not a scientific parameter. It is
fixed for the whole formal run and recorded, because CPU float32 reduction order
depends on it (the same effect documented in the P6-B0 audit); it is not used to
select anything.

Reproduce:

```bash
python scripts/p6/run_c0_trigger.py audit
python scripts/p6/run_c0_trigger.py train --threshold-workers 8
python scripts/p6/run_c0_trigger.py evaluate
```
