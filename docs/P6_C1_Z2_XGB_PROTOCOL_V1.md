# P6-C1-Z2-XGB-v1: frozen GAIA scorer transport

## Question and evidence grade

On the **same** C1-v2 detected-anchor Train cohort and detected-anchor Test
episodes, does replacing Arm C's Conditional Logit scorer with the thesis
68D event-group XGBRanker improve root-service ranking? This is a single
scorer replacement. Detector, event matching, candidate order, 68D GAIA
features, anchor times, cases, and evaluation formulas stay fixed.

The reused C0/C1 Test has been observed in prior P5/P6 work. The result is
**exploratory and descriptive**, not independent confirmation. No Test result
may select a hyperparameter, preprocessing rule, threshold, offset, model,
cohort, or follow-up variant. A negative or null result is a valid outcome.

## Source method and adaptation

Reference: `Ada-RCA-z2-xgb-closure`, formal execution commit
`318128f780835eefd849de9e2bcf14c0b1b7e1d6`, current closure HEAD
`684e26eebd24bc6bc61e8d9b7fa46349781bf9e8`. The reference's Z2
XGBRanker uses `rank:pairwise`, 200 trees, depth 3, learning rate 0.05,
subsample and column sample 1.0, lambda 1.0, seed 20260826, one worker and
`hist`. It uses **no scaler**, feature selection, class weighting, validation
selection, early stopping or search. Pinned runtime is Python 3.8.20,
XGBoost 2.1.4 and the versions in the JSON design lock.

The source experiment's W600-B15/80-bin RE2 features and learned model cannot
be transplanted into GAIA. Here the already frozen GAIA W300-B15/40-bin 68D
features are used. Each of the 3,225 C1-v2 common Train cases provides ten
contiguous candidate rows, one relevance-1 root and nine relevance-0 rows;
all Train cases have unit weight. XGB is fitted once on the **detected** Train
anchor. No scaler is applied, matching the reference XGB method. Existing
Arm C Conditional Logit used a GT-Train-fitted scaler, which is intrinsic to
that old scorer. The comparison therefore isolates the scorer *procedure*,
including its prescribed normalization, rather than a common scaled matrix.

## Fixed cohort and sequence

The sealed source run is `c1-supervision-oos-v1-seed42` in the original
`Ada-MGAD-e2e-v2` worktree. The new output is a unique
`experiments/p6/c1_z2_xgb/c1-z2-xgb-v1-seed20260826` directory in the new
`experiment/p6-z2-xgb` worktree. The design lock binds every reused source
file by SHA-256. The source run and its old outputs are read-only.

1. `preflight`: verify source code, fixed environment, source method hashes,
   old run provenance, 3,225 Train cases and fold floors. Open no Test data.
2. `init`: reserve a new run only after code, tests and design lock are
   committed. Bind that source commit and source file hashes.
3. `train`: fit one XGBRanker on 935/1,168/1,122 detected-anchor Train cases.
   Require unique cases and exact root index/service alignment. Save model.
4. `lock-test`: verify the old label-free prediction lock; read only the
   sealed feature tensors, validity mask, old Arm C rankings and frozen C0
   episode metadata. Rank all 4,213 legal episodes; preserve the one invalid
   episode. Seal the new ranking CSV and model/input hashes **before** opening
   Test matching or GT. No fallback or dropped episode.
5. `evaluate`: verify the new lock, then open evaluator-only C0 matching and
   GT registry. Reuse the existing C1 paired evaluator and C2 full-diagnosis
   evaluator. Internally map old Arm C to evaluator slot `b`, new XGB to slot
   `c`. Replay old Arm C AC@1 and E2E @1/3/5 exactly as a validity gate.

Each stage creates a directory once and seals file hashes. A failed or
incomplete stage is retained; any repair or rerun requires a new protocol
version and run ID. The detector is not retrained and C0 Test inference is
not rerun.

## Metrics and predeclared interpretation

The primary comparison is paired `AC@1(XGB) - AC@1(old C)` on the 4,197
legal matched Test cases, with all four paired transitions and a 10,000
replicate UTC-onset-day descriptive bootstrap interval (fixed seed 42).
Secondary metrics are AC@3/5, MRR, root/fault strata, and full E2E diagnosis
P/R/F1@1/3/5 on the frozen 5,787 GT events and 4,214 detected episodes.
Misses, false alarms, invalid context and wrong ranking stay in the ledger.

There is **no score-driven GO threshold**. Report the sign, magnitude,
trade-offs and failure decomposition. If XGB improves some metrics, this
supports only a reused-Test scorer-sensitivity observation; it does not
replace the prior C1-v2 conclusion or independently establish a better final
method. If it fails or regresses, retain the original C1-v2 results and close
this scorer transport without tuning.
