# P6-C1 G3 implementation status

Updated: 2026-09-29. Status: **FORMAL FOLD 1 `PREFIX_SCHEMA_NO_GO`; FROZEN C1 RUN STOPPED**. See the [fold 1 disposition](P6_C1_G3_FOLD1_PREFIX_SCHEMA_NO_GO_20260929.md).

The frozen [G2 design](P6_C1_G2_FROZEN_DESIGN.md) and [v1.1 source binding](../configs/e2e/gaia_p6_c1_g2_v1_1.json) are unchanged. The G3 runner is [`scripts/p6/run_c1.py`](../scripts/p6/run_c1.py). It uses the exact G2 run root and exclusive stage directories. `init` repeats the full read-only input preflight before reserving that root. Every subsequent stage checks the run lock and source snapshot; a failed stage retains `INCOMPLETE.json` and cannot be overwritten.

## Implemented sequence

1. [`c1_fold_preprocessing.py`](../src/e2e/c1_fold_preprocessing.py) fits Metric, Log and Trace on each locked fold Fit prefix and separately transforms Fit, Selection and Generation. Its sealed arrays have 48D/32D/8D dimensions and legal segment-local 300-second windows. [`c1_fold_detector_data.py`](../src/e2e/c1_fold_detector_data.py) prevents Generation labels from entering the dataset.
2. [`c1_fold_supervision.py`](../src/e2e/c1_fold_supervision.py) builds Fit/Selection trigger labels and complete Selection metric GT only. [`c1_fold_detector.py`](../src/e2e/c1_fold_detector.py) uses frozen C0 architecture, Fit-only class weight, AdaBelief, masked BCE plus graph regularization, 30 epochs maximum, patience 8, and Selection event-F1/recall/threshold choice. [`c1_fold_generation.py`](../src/e2e/c1_fold_generation.py) scores Generation without labels and seals fold-unique episode IDs before Train GT matching.
3. [`c1_oos_matching.py`](../src/e2e/c1_oos_matching.py) performs frozen causal one-to-one matching only after Generation seals, retaining misses, false alarms, boundary events and illegal dual contexts. [`c1_common_cohort.py`](../src/e2e/c1_common_cohort.py) extracts paired GT/detected W300-B15 10×68 Train features and reports the fixed 596/765/767 per-fold floor. A failed floor seals `NO_GO` coverage and blocks RCA fitting.
4. [`c1_shared_rca.py`](../src/e2e/c1_shared_rca.py) fits one scaler on common Train GT-anchor candidate rows, then B/C Conditional Logit on identical case IDs and root labels. The optional fixed-scaler interface in [`rca_model.py`](../src/e2e/rca_model.py) retains its original default P5 behavior.
5. [`c1_test_scoring.py`](../src/e2e/c1_test_scoring.py) accepts only the frozen label-free C0 `test_episodes.csv` columns. It emits one B/C scope row per episode plus feature tensors and an availability mask. The scorer hashes but does not parse `test_predictions.csv`, and does not read Test matching or GT. A legal-context ranking failure stays in the later C1 denominator with zero correctness for the missing arm. The runner seals these files under `predictions/` before any Test GT read.
6. [`c1_evaluation.py`](../src/e2e/c1_evaluation.py) consumes the lock, then Test matching and raw GT. It reports paired AC@1 transitions, AC@3/5/MRR, group sizes and n≥20 macro summaries, 10,000 UTC-onset-day cluster bootstrap replicates, post-lock A oracle diagnostic, and C2 raw-event failure counts. Final diagnosis wall time remains `UNMEASURED` rather than estimated.

## Evidence and limits

| Check | Current evidence |
|---|---|
| Bounded real-adapter smoke | PASS: [machine record](P6_C1_G3_RAW_SMOKE_20260928.json), 104 synthetic 30-second bins, 45 qualified Metric slots, 17 stable Log templates, one Trace edge; actual fit/transform calls produced identical fitted state, graph, diagnostics and Fit/Selection/Generation arrays with spawn workers 1 and 24. |
| Full read-only input preflight | PASS on 2026-09-29: `run_c1.py preflight` verified G2 source bindings, the smoke/source digest, all bound raw catalog bytes and the RCA raw-index arrays. It created no formal run root. |
| C1 interface tests | PASS: 22 isolated/synthetic C1 tests, including detector stage sealing, causal OOS matching, paired feature construction, shared scaler parity, full Test episode scope and denominator-preserving failure, exclusive stage seals and post-lock C1/C2/A evaluation. |
| Existing adapter/model regression | PASS: eight raw-adapter tests and five P5 RCA-model tests. |
| Frozen C0 detector regression | PASS: 30 model, label, loader and driver contract tests. |
| Real GAIA prefix schema and dimensions | NO-GO: formal fold 1 Fit retained 27 qualified real Metric slots against the frozen 45-slot budget. Its `INCOMPLETE.json` and `failure.json` are preserved; no fold was completed. |
| Formal OOS anchors/common Train cohort | PENDING: G1's 4,254 is a static GT/context upper bound, not observed anchors. The 596/765/767 floors have not been measured. |
| Test rankings/C1/C2 outcomes | PENDING: no formal detector, RCA fit, Test feature pass, prediction lock or evaluation has run. The existing Test is reused, not independent confirmation. |

The bounded smoke establishes worker equivalence on its synthetic raw catalog. It does not establish that real early GAIA prefixes meet the 45-slot/17-template/Trace requirements. Stage unit tests do not replace a successful formal execution or imply positive C1 effect.

The old `check_c1_g2_protocol.py` PASS line still says “pending G3” because that checker certifies G2 inputs only. This document and `run_c1.py preflight` carry the subsequent G3 gate; the frozen G2 checker and design record were not rewritten.

## Manual formal sequence

The [original manual command handoff](P6_C1_G3_MANUAL_RUN.md) records the planned sequence. `init` and fold 1 `fold-input` were attempted; fold 1 failed its frozen Metric schema gate. No further command from that handoff is valid for this run ID. Preserve the existing run and use the [fold 1 disposition](P6_C1_G3_FOLD1_PREFIX_SCHEMA_NO_GO_20260929.md) for the next research decision.
