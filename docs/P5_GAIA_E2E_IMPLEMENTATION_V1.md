# P5-I1 GAIA Two-Stage End-to-End Implementation V1

## 1. Goal

P5-I1 implements an engineering integration, not a third algorithmic contribution:

```text
continuous GAIA telemetry
  -> Ada-MGAD-G node anomaly probabilities p_i(t)
  -> max-node trigger and detected onset t_hat
  -> Ada-RCA-G W300-B15 representation
  -> labelled injected/fault-service ranking
```

Ada-MGAD and Ada-RCA retain their frozen architectures, losses, ranking model,
regularization, and deterministic ordering.  The new code supplies the minimum
GAIA adapters, leakage-safe chronological protocol, trigger/matching logic,
baselines, metrics, artifacts, and reproducible command path.

## 2. Repository State

- Working repository: `/home/zhangll24/RCA_project/Ada-MGAD-e2e`
- Branch: `e2e`
- P5-G0R2 starting commit: `cbe57512b6bd1ce772ab42d9b8ec0b292c3cb91b`
- Canonical Ada-RCA reference: `main@a2c620922e7c0ab3615d34654d4a3690d1b22c8e`
- Frozen Ada-RCA sources: `docs/REPRESENTATION_FREEZE.md`,
  `src/rca/features.py`, `src/rca/final_method.py`, and `src/rca/p4.py`

The reference repository was read only.  Minimal copied logic records its source
commit in `src/e2e/rca_features.py` and `src/e2e/rca_model.py`.

## 3. Frozen Protocol

`configs/e2e/gaia_p5_v1.yaml` is the single protocol registry.  It freezes:

- ten candidate services in one canonical order;
- the 30-second detector grid and ten-bin detector window;
- duration-based chronological 60/20/20 blocks defined before preprocessing;
- validation-only detector checkpoint and event-threshold selection;
- one-to-one onset matching with a 60-second tolerance;
- W300-B15 RCA context, 40 bins, 20 pre and 20 post bins;
- Trace Error as `status_code != 200` and Trace Latency in seconds;
- Conditional Logit with L2=1, maximum 1,000 iterations and gradient tolerance
  `1e-8`;
- random seed 42.

Neither Test metrics nor Test labels select a checkpoint, threshold, temporal
configuration, representation, or model.

## 4. Event Registry

The event source is `artifacts/p5/g0r2/event_registry.csv`, SHA-256
`b95556a0ce7e274c16d52bb55aa5d8dc202f8878305744fc786befafc23ba4b5`.
It contains 16,200 supported injections established here from explicit injection
semantics.  It is not claimed to be GAIA's unique official canonical failure
taxonomy.

Overall fault counts are:

| Fault type | Events |
|---|---:|
| login failure | 15,478 |
| memory anomalies | 652 |
| file moving program | 43 |
| access permission denied exception | 15 |
| cpu anomalies | 12 |

Normal records, 38 ERROR records, 17 normal-memory-freed labels, and 36
unsupported/unknown rows do not enter P5-I1 fault GT.  Ada-MGAD-G node labels
are rasterized only from this bound registry.

The CPU parser was changed only from integer duration matching to floating-point
duration matching.  Regression evidence is 0/12 parsed before and 12/12 after;
the other four supported event semantics have dedicated tests.

## 5. Ada-MGAD-G Changes

`src/e2e/ad_preprocess.py` and `src/e2e/ad_data.py` provide split-local,
timestamp-preserving arrays.  Each inference row can be restored to:

- `window_start_time`;
- `window_end_time`;
- `prediction_timestamp`;
- canonical service names;
- node labels and both-class model scores.

The output CSV retains the anomaly probability for every service and timestamp;
binary predictions are auxiliary.  Metric/log normalization and schema-quality
decisions use Train only.  E2E metric shard reads fail closed instead of silently
accepting partial I/O.  The model receives the same tensor families and keeps its
architecture, dynamic edge weighting, gated multimodal fusion, consistency term,
loss, and frozen hyperparameters.

`util/train.py` now accepts `train_loader`, `val_loader`, and `test_loader` on the
E2E path.  Best-F1 checkpoint selection reads Validation only.  The Test loader is
used only after the selected checkpoint is loaded.  Existing historical result
files are not overwritten.

## 6. Event Trigger

For each 30-second prediction timestamp:

```text
S(t) = max_i p_i(t)
system positive iff S(t) >= theta
```

`theta` is selected by exact enumeration of Validation system scores to maximize
Validation Event F1.  Equal-F1 thresholds choose the highest threshold.  Test is
not passed to the selection function.

Consecutive positive timestamps exactly 30 seconds apart form one episode.  The
first positive timestamp is `t_hat`.  No learned eventizer, hysteresis, detector
latent feature, score fusion, or detector-based candidate filtering is used.

## 7. Event Matching

Predictions are traversed by `(t_hat, prediction_id)`.  Each prediction chooses
the nearest still-unmatched raw-injection onset within 60 seconds.  Ties use
earlier `gt_start`, then `source_index`, then `case_id`.  An unmatched prediction
is a false alarm and an unmatched GT injection is a miss.  Raw injections are not
merged, and their duration does not widen the onset-matching interval.

Signed Detection Delay is `t_hat - gt_start`.  Event outputs include Precision,
Recall, F1, and delay mean, median, P90, and P95.

## 8. Ada-RCA-G Raw Adapter

`src/e2e/gaia_rca_adapter.py` reads GAIA raw Metrics, Logs, and Traces independently
of Ada-MGAD processed tensors.  Feature construction accepts only an anchor,
telemetry schema, and the canonical service registry; labelled service and fault
type remain in a separate case registry.

- Metrics: filename/schema-based, label-free service-indicator mapping.
- Logs: high-resolution timestamps from the message prefix, converted to numeric
  per-level counts plus total count.  Raw text is not model input.
- Trace Error: count of `status_code != 200`, including 300, 400, and 500.
- Trace Latency: `end_time - start_time` in seconds; non-finite or negative rows
  are deterministically filtered and counted.

The four items above are feature channels, not four raw modalities.  The reusable
raw index stores array hashes; detected-anchor evaluation verifies every array
checksum before use.

## 9. W300-B15 Representation

`TemporalSpec` freezes `window_seconds=300`, `bin_seconds=15`, `pre_bins=20`,
`post_bins=20`, `n_bins=40`, onset sentinel 300 seconds, and post-position
denominator 19.  Context is `[t0-300s,t0+300s)`.

Each channel retains canonical Z2:

```text
8 base features + 9 morphology features = 17D
4 channels x 17D = 68D per candidate service
```

Pre-event median/MAD with IQR fallback, Q90 service aggregation, onset threshold,
two-consecutive-observed-bin persistence, coverage, masks, and morphology
normalization are unchanged.  Denominators and shapes are derived from
`TemporalSpec`; no new representation feature is added.

## 10. Train, Validation, and Test

Absolute duration boundaries are frozen in epoch milliseconds:

| Split | Start (UTC) | End (UTC) | GT events | W300 cases | AD windows purged | RCA cases purged |
|---|---|---|---:|---:|---:|---:|
| Train | 2021-07-01 03:06:52.249 | 2021-07-19 10:50:40.287 | 8,787 | 8,787 | 10 | 0 |
| Validation | 2021-07-19 10:50:40.287 | 2021-07-25 13:25:16.300 | 3,513 | 3,512 | 10 | 1 |
| Test | 2021-07-25 13:25:16.300 | 2021-07-31 15:59:52.313 | 3,900 | 3,899 | 10 | 1 |

No raw injection crosses a split boundary.  Two W300 cases are purged:
`gaia-5fb9af715fb19ade` at the Validation boundary and
`gaia-553873e756b76fd2` at the Test boundary.  The AD audit counts 30 split-owned
incomplete/crossing windows plus one grid timestamp outside the absolute range.
Actual detector datasets are constructed inside each block, so no model window
can cross a boundary.

The chronological distribution is intentionally not rebalanced.  In particular,
all 12 CPU events occur in Test; there is no CPU event in Train or Validation.

## 11. Ada-RCA-G Model and Baselines

The ranker is the canonical absolute-Z2 Conditional Logit.  `StandardScaler` is
fit on the 8,787 Train events' candidate rows only.  Validation/Test only call
transform and score.  Ties follow descending score and stable service-name order.

Two required baselines are implemented:

1. Root Frequency counts labelled services in Train only and emits one fixed
   ranking for Validation/Test.
2. Detector-only reads the ten Ada-MGAD node anomaly scores exactly at each
   successfully matched `t_hat`, ranks descending, and resolves ties by canonical
   registry order.

Detector scores never enter Ada-RCA-G features or candidate selection.

## 12. Metrics

Node-level AD retains Precision, Recall, F1, AUC, and AP.  Event-level AD reports
Precision, Recall, F1 and signed delay statistics.  RCA reports AC@1, AC@3, AC@5,
Avg@5, and MRR overall, root-service macro, and fault-type macro.

For `k in {1,3,5}` full-pipeline metrics are:

- TP@k: a prediction is matched and the labelled fault service is in Ada-RCA-G
  Top-k;
- FP@k: an unmatched prediction, or a matched prediction ranked outside Top-k;
- FN@k: an unmatched GT event, or a matched GT event ranked outside Top-k.

A matched ranking failure is therefore both FP and FN.  `Diagnosis F1@1` is the
primary full-pipeline summary.  Diagnosis uses the W300-eligible Test population;
GT-anchor and detected-anchor contexts crossing the Test boundary are explicitly
purged and counted in the artifact.  Event-detection metrics remain on all 3,900
complete Test injections.  Oracle-anchor versus detected-anchor deltas use
`detected - oracle` on the same matched cases.  Signed/absolute delay correlations
and early/near/late delay bands are also reported.

## 13. Results and Current Execution Status

The 30 GB GAIA source was not fully preprocessed during this implementation run,
following the explicit instruction to validate on small data and provide manual
commands for the long operation.  Consequently, no formal checkpoint or
GAIA-full node/event/RCA/Diagnosis metric is claimed in this document.

Current evidence is implementation-only:

| Check | Result | Scientific status |
|---|---|---|
| Ada-MGAD-G smoke | 39 windows/split, finite `39 x 10 x 2` scores | synthetic, non-formal |
| Event trigger/matching smoke | PASS | synthetic, non-formal |
| RCA adapter/representation smoke | finite `10 x 68` | synthetic, non-formal |
| Conditional Logit smoke | finite `20 x 10 x 68`, converged | synthetic, non-formal |
| Full E2E metric smoke | PASS | synthetic, non-formal |
| Automated test suite | recorded in `run_manifest.json` | implementation validation |

Formal files remain explicitly `PENDING_MANUAL_FULL_GAIA_RUN` in
`artifacts/p5/i1/run_manifest.json`.  Running the commands below fills those
paths; smoke scripts do not create or overwrite formal prediction/metric files.

## 14. Limitations

- Severe temporal distribution shift: CPU exists only in Test.
- Dense and overlapping raw injections remain independent GT, which a 30-second
  detector and simple episode merger may not resolve individually.
- Labels are coupled to injection messages; a labelled service is not independent
  proof of causal root service.
- Login failures and mobile-service labels dominate the registry; rare-fault and
  macro estimates may have high variance.
- Telemetry channels can be sparse or missing.  Feature health is a sanity check,
  not permission to redesign Z2.
- The log adapter uses numeric level/volume indicators and does not learn from raw
  text.  This is a documented minimal adapter compromise.
- The shared GAIA filesystem showed long-latency reads.  Raw indexing is fail
  closed; use a byte-layout-identical local mirror if needed.
- No topology, causal graph, propagation chain, detector-score fusion, or joint
  end-to-end learning claim is made.

## 15. Reproduction Commands

From a clean checkout on branch `e2e`:

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e
git checkout e2e
git status --short
python --version
pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

The validated environment uses Python 3.8 and PyTorch 1.12.0/CUDA 11.3.  Install
the matching PyTorch build for the host separately if it is not already present.

Quick validation, without GAIA full preprocessing:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q
python scripts/p5/run_i1_pipeline.py smoke --gpu false
```

The full run is intentionally separated into commands.  Raw preprocessing now
uses deterministic process-level parallelism documented in
`docs/P5_GAIA_PARALLEL_PREPROCESSING_V1.md`:

```bash
python scripts/p5/build_i1_protocol.py

# /usr/bin/time -v is optional.  If it is absent, omit the wrapper (or use the
# zsh/bash shell builtin `time python ...`); preprocessing itself does not depend
# on GNU time.

python scripts/p5/run_i1_ad.py preprocess \
  --raw-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS \
  --workers 8 --chunk-rows 150000 --start-method spawn
# Redirected training automatically disables batch-level tqdm updates.  The
# environment variable is an explicit, reproducible override for a quiet log.
P5_AD_BATCH_PROGRESS=0 python -u scripts/p5/run_i1_ad.py train --gpu true
python scripts/p5/run_i1_events.py evaluate \
  --workers 24 --start-method spawn

python scripts/p5/run_i1_rca_features.py index \
  --raw-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS \
  --workers 8 --chunk-rows 150000 --start-method spawn
python scripts/p5/run_i1_rca_features.py materialize \
  --workers 24 --case-chunk-size 128 --start-method spawn
python scripts/p5/run_i1_rca.py train-oracle

python scripts/p5/run_i1_e2e.py evaluate
PYTHONDONTWRITEBYTECODE=1 pytest -q
python scripts/p5/finalize_i1_manifest.py --pytest-result 'FULL_SUITE_PASS'
```

To display the same concise epoch-level output while saving stdout and stderr
to a file in zsh, use:

```bash
mkdir -p logs/p5/i1
LOG="logs/p5/i1/train-evaluate_$(date +%Y%m%d_%H%M%S).log"
P5_AD_BATCH_PROGRESS=0 python -u scripts/p5/run_i1_pipeline.py train-evaluate \
  --gpu true --event-workers 24 --start-method spawn 2>&1 | tee "$LOG"
TRAIN_RC=${pipestatus[1]}
echo "exit_code=${TRAIN_RC}" | tee -a "$LOG"
exit "$TRAIN_RC"
```

`P5_AD_BATCH_PROGRESS=1` restores the interactive batch progress bar.  The
setting only changes logging; it does not change model computation or
checkpoint selection.

The event workers parallelize the exact all-unique-score Validation threshold
sweep.  They do not subsample thresholds or change the highest-threshold
tie-break.  The selected threshold is re-evaluated through the full episode and
matching path before artifacts are written.

Mutating pipeline actions acquire `data/p5/i1/.pipeline.lock`.  A second
container pointed at the same checkout therefore fails fast instead of
overwriting the shared checkpoint and artifact paths.  Parallel raw/feature
workers within one pipeline remain supported; two independent training
pipelines must not share output paths.

If the canonical filesystem stalls, copy `MicroSS` to responsive local storage
without changing names or byte sizes and pass that path with `--raw-root`; both AD
and RCA indexers verify the P5-G0R2 relative-path/byte-size layout before reading.
For an uninterrupted run, the equivalent entry is:

```bash
python scripts/p5/run_i1_pipeline.py full --gpu true \
  --raw-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS \
  --raw-workers 8 --feature-workers 24 \
  --chunk-rows 150000 --case-chunk-size 128 --start-method spawn
```

For separate CPU/GPU allocations, run `run_i1_pipeline.py preprocess --gpu false`
in the CPU container, then run `run_i1_pipeline.py train-evaluate --gpu true` in
the GPU environment against the same `data/p5/i1/` outputs.

## 16. Artifact Manifest

Committed protocol/smoke evidence lives in `artifacts/p5/i1/`.  Large processed
arrays and checkpoints live under `data/p5/i1/` and are excluded from Git.
`run_manifest.json` records config/source hashes, the code commit used to generate
the manifest, smoke hashes, completed formal artifacts, and each pending formal
artifact.  Formal commands add:

```text
ad_data_manifest.json
ad_training_summary.json
ad_validation_predictions.csv
ad_test_predictions.csv
ad_event_predictions.csv
event_matching.csv
event_detection_metrics.json
rca_feature_health.json
rca_feature_manifest.json
rca_train_manifest.json
rca_oracle_predictions.csv
rca_detected_predictions.csv
detector_only_predictions.csv
root_frequency_predictions.csv
rca_metrics.json
e2e_diagnosis_metrics.json
```

Every formal stage records the config hash, source identities, git commit, seed,
row/case counts, and relevant input/output hashes.  Checkpoint hashes are stored in
the AD training summary and RCA training manifest.

## 17. Final Decision

**IMPLEMENTATION READY; FORMAL FULL-GAIA EXECUTION PENDING.**

P5-I1 now has a runnable, tested, provenance-bound two-stage code path.  The
appropriate thesis wording is continuous node-level anomaly detection followed by
event-triggered labelled fault-service ranking.  Publication of numerical Chapter
5 results must wait for the documented full-data commands and must use the
resulting formal artifacts without post-hoc protocol changes.
