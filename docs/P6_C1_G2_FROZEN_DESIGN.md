# P6-C1 G2 detector-aligned RCA design lock

## Material Passport

- Origin Skill: `academic-research-suite/experiment-agent`
- Origin Mode: `plan`
- Origin Date: 2026-09-24
- Version Label: `P6-C1-G2-v1`
- Verification Status: design and input integrity verified; fold OOS and C1
  outcome unverified

## Status and evidence grade

- Design version: `P6-C1-G2-v1`; seed: `42`; intended run ID:
  `c1-prefix-oos-v1-seed42` under `experiments/p6/c1_detector_aligned/`.
- **G2 design is fixed before fold or C1 Test results. G3 implementation and
  smoke are pending. Formal C1 execution is NO-GO until they pass.** This
  document does not claim that OOS anchors or the common Train cohort exist.
- The machine-readable lock is
  [`configs/e2e/gaia_p6_c1_g2_v1.json`](../configs/e2e/gaia_p6_c1_g2_v1.json).
  Its bound inputs include the [G1 static ledger](../experiments/p6/c1_feasibility/c1-g1-20260923T105441Z/feasibility_ledger.json),
  frozen C0 files, and a content-hash inventory of the raw modality CSVs and
  run table. Every execution must verify these bytes before writing outputs.
- Claim level: **prefix-fitted detector OOS anchors conditional on a frozen,
  label-free raw filename catalog; RCA uses the existing fixed 68D feature
  definition and full-corpus filename-derived raw-index indicator catalog.**
  Per-case RCA values remain exact-window local. This is not an unrestricted
  full-pipeline forward-OOS claim because the structural filename catalog is
  transductive. It is stronger than reusing the existing 70%-Train-fitted AD
  arrays. Any study requiring a prefix-bound RCA catalog needs a separately
  frozen protocol, because changing the RCA indicator set would also change
  the representation under comparison.
- Existing C0 `BORDERLINE` and C0F historical Fit/Validation source
  `UNVERIFIED` remain unchanged. The existing Test has been examined; C1/C2
  will be a reused-Test study, not independent confirmation.

| Evidence class | Current statement |
|---|---|
| FACT | G1 records 4,254 Fit-only static GT/context candidates; current shared AD arrays were fitted across the original 70% Train. |
| PROTOCOL CHOICE | Three rolling folds, prefix-fitted detector inputs, fixed transductive raw filename catalogs, half-upper-bound cohort floors and one GT-fitted RCA scaler. |
| PENDING | Actual fold schema feasibility, OOS episode/match counts, legal common cohort, C1 rankings and any effect estimate. |

## 1. Fixed detector folds and input boundary

Use the three `fit_only_3fold` intervals in the G1 ledger. All are UTC,
half-open and aligned to the frozen 30-second grid. No fold uses original C0
Validation or Test to generate Train anchors.

| Fold | Fit | Select checkpoint/threshold | Generate OOS anchors | GT/context upper bound | Minimum actual common cases |
|---|---|---|---|---:|---:|
| 1 | Jul 1 10:00–Jul 4 10:36 | Jul 4 10:36–Jul 7 11:12 | Jul 7 11:12–Jul 10 11:48 | 1,192 | 596 |
| 2 | Jul 1 10:00–Jul 7 11:12 | Jul 7 11:12–Jul 10 11:48 | Jul 10 11:48–Jul 13 12:24 | 1,529 | 765 |
| 3 | Jul 1 10:00–Jul 10 11:48 | Jul 10 11:48–Jul 13 12:24 | Jul 13 12:24–Jul 16 13:00 | 1,533 | 767 |

The minima are the ceiling of one half of each G1 static upper bound. They
are a performance-blind feasibility floor, not predicted recall or a power
claim. If any fold misses its floor, stop the formal C1 comparison and report
the observed OOS coverage. Do not add folds, alter thresholds, or replace
cases after seeing the results. The 4,254 static candidates are not the
observed denominator.

For every fold, fit Metric slot selection, quality/correlation filtering and
scalers, Drain3 state/templates/scales, and Trace graph/scales only on that
fold's Fit prefix. Selection and generation only transform the frozen fold
schema. Freeze service order and target dimensions as 10 services, Metric 48D,
Log 32D and Trace edge 8D. If an early prefix cannot satisfy these dimensions,
record `PREFIX_SCHEMA_NO_GO`; do not pad with future-selected slots or silently
change the model. Raw file *names* are an externally locked instrument catalog;
all numeric/statistical fitting is prefix-only. Bind full raw file contents,
policy, schema, fitted state, graph, arrays and source bytes to each fold.

Use a fresh, fold-specific schema, AD data root and artifact root. Preserve the
current Train/Test shared preprocessing and all P5/C0 outputs. Fold transform
partitions must not share mutable Drain3/scaler/graph state. Trace parent spans
are resolved inside each transformed interval; crossing parents remain
unmatched, as in the current parser. Report their counts. If a change to that
semantics becomes necessary, stop and version this design before any result.

The detector architecture, optimizer, loss, trigger labels, 30 epochs maximum,
patience 8, seed 42 and event construction/matching are copied from frozen
P6-C0. Each fold trains once on Fit and selects checkpoint plus exact unique
score threshold using only its subsequent Selection segment. Use the C0
event-F1, then recall, then threshold tie rule. No Test score or C1 RCA score
selects detector state. Generation is label-free model scoring followed by
thresholded episode construction; GT is joined afterward only for Train
one-to-one causal matching. Do not use old in-sample C0 Fit scores or its
historical Validation selection as OOS anchors.

For every target bin, the detector's 300-second input history must be wholly
inside its own Fit, Selection or Generation segment. Match a generation episode
to a complete generation GT event only under the frozen C0 causal
max-cardinality/minimum-delay rule with `0 <= t_hat - onset <= 60s`. Retain
unmatched GT and unmatched episodes in a separate ledger. Train RCA contexts
must satisfy both `[GT onset-300s, GT onset+300s)` and
`[detected t_hat-300s, detected t_hat+300s)` inside the same Generation segment
and raw timeline. Boundary events and invalid contexts are counted, not moved
between folds. Each `case_id` may occur in at most one generation fold.

## 2. Same-case RCA intervention

Build the common Train cohort only after all fold OOS episodes and matching
are locked. Require a unique GT `case_id`, a unique matched prediction, a
legal pair of GT/detected 68D tensors, finite values, identical ten service
candidates in canonical order, and both context windows legal. Do not select
the cohort by model correctness or Test labels. The complete cohort identity
and exclusion ledger are immutable inputs to both arms.

Use the existing label-free RCA raw index and frozen W300/B15 68D Z2
representation for both anchors and Test. Its filename-derived metric catalog
is a declared transductive structural input. Train labels are separate from
feature extraction. Fault/root labels never enter the detector input, episode
threshold, Test feature builder or prediction scope.

Fit one `StandardScaler` on all ten candidate rows of **GT-anchor features
from the common Train cases**. Apply exactly that scaler to both GT and
detected feature tensors. Fit two Conditional Logit weight vectors with the
same frozen `lambda=1`, optimizer, tolerance, candidate order, case IDs,
labels and unit case weights: B on GT-anchor Train features, C on OOS-detected
Train features. The current RCA trainer fits a scaler inside each call; G3
must add a shared-scaler interface or equivalent verified adapter. A uses B's
weights on GT-anchor Test features and is oracle context only. B and C both
rank the same detected-anchor Test contexts.

Before any Test GT join, create feature inputs and a complete ordered ranking
for **every legal frozen C0 Test detected episode**, including episodes that
will later be false alarms. Use only the label-free columns of
`test_episodes.csv` to define this universe. The bound `test_predictions.csv`
contains a `trigger_label` column and is **hash-only provenance** for C1;
the scorer must not parse it. `test_matching.csv` is evaluator-only.
Freeze the universe, legal-context exclusions, feature inputs, model/scaler
files, rankings, source digest and per-file hashes in a prediction/scope lock.
Missing or invalid rankings stay as failures; no label-derived fallback or
post-lock repair. Oracle arm A's GT-anchor Test features are generated only
after the B/C prediction lock; they cannot alter the B/C scope. C0 Test
inference is never rerun by C1.

## 3. Evaluation and fixed decisions

The primary C1 estimand is paired Test matched-case `AC@1(C)-AC@1(B)` on the
prelocked common legal episode scope, with the four paired correct/incorrect
transition counts. A missing or invalid ranking contributes zero correctness
for that arm and stays in the denominator. A is a diagnostic. Secondary
metrics are AC@3/5, MRR,
root/fault macro and per-group `n`. For macro summaries, a group with `n=0`
is undefined and listed; `1 <= n < 20` is reported descriptively and omitted
from the prespecified macro aggregate. Preserve all such rows in the detailed
table. Report both weighted overall and macro, with no claim of balanced
fault coverage; the Fit-only static candidate has no CPU cases.

Report a paired 95% percentile interval using 10,000 resamples of UTC onset
day clusters with replacement, seed 42; retain the B/C pair within each
resampled day. This describes uncertainty on the reused Test and is not an
independent significance claim. Report raw numerators and denominators
without intermediate rounding. No positive-gain threshold is imposed after
the fact: zero, negative or inconclusive differences remain valid outcomes.

C2 runs only after the C1 prediction lock and reports all raw GT events,
detected episodes, misses, false alarms, illegal contexts and ranking failures
under frozen C0 matching. Keep onset-to-detector, RCA-data-ready and final
diagnosis latency separate. Test matching, root/fault labels and final metrics
cannot affect C1 preprocessing, training, scoring or scope.

## 4. Resource and execution gates

The fixed compute budget is three fold preprocessing fits and three detector
fits, at most 30 epochs per fold, one GT-arm and one detected-arm RCA fit, and
one label-free Test ranking pass per fitted arm. Synthetic smoke and source-boundary checks
may be repeated during G3; a failed *formal* fold run is preserved and any
repair requires a new protocol version and new run directory. Do not run a
hyperparameter search or a rescue fold. Formal preprocessing uses 24 workers
with `spawn` (within the existing budget of 30); detector training uses 2
data-loader workers and 8 Torch threads. G3 must demonstrate workers=1 versus
workers=24 equivalence on a fixed fixture before parallel formal use.

The future run root is created once with exclusive-create semantics. Reserve
`folds/fold_01..03/{ad_data,ad_artifacts,detector}` for per-fold outputs,
`train_cohort/` for identities, matches and two feature bundles, `rca/` for
the common scaler and two models, `predictions/` for label-free Test scores,
and `evaluation/` for post-lock C1/C2 reports. Each stage writes a completion
manifest with source, input and output hashes before the next stage reads it.
No stage may adopt a partial directory as complete or overwrite an earlier
stage. A stopped run is retained and a new run ID is required for a rerun.

G3 must supply and validate: (1) fold-aware prefix materializer, schema and
manifest; (2) segment-aware detector dataset/trainer and generation-only
inference; (3) Train OOS matching and common-cohort feature builder; (4)
shared-scaler RCA arm fitting; (5) label-free all-episode Test scorer and
prediction lock; (6) evaluator/finalizer that consumes only the lock. Run
read-only source/input preflight and isolated synthetic smoke before any full
preprocessing/training. There is no current C1 full-run CLI, and the old P5
and C0 `all` commands do not implement this design. Formal execution remains
`NO_GO` until these interfaces, tests and manifests exist and pass.

## 5. Commands available now

These commands are read-only integrity checks, executable before G3:

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
PYTHONDONTWRITEBYTECODE=1 python scripts/p6/check_c1_g2_protocol.py --require-new-run
PYTHONDONTWRITEBYTECODE=1 python scripts/p6/bind_c1_raw_inputs.py check \
  --manifest experiments/p6/c1_protocol/c1-g2-v1-20260924/raw_content_manifest.json
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python - <<'PY'
from pathlib import Path
from src.e2e.gaia_rca_adapter import validate_raw_index_manifest
print(validate_raw_index_manifest(Path('data/p5/v3/rca_raw_index/index_manifest.json')))
PY
```

The raw content check reads about 31.58 GB, and the RCA index check reads its
array files. Neither modifies the frozen inputs. There is deliberately no
full C1 execution command at G2: its missing G3 interfaces are listed above.
