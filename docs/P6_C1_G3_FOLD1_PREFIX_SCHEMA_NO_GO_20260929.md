# P6-C1 fold 1 prefix schema failure

Date: 2026-09-29. Disposition: **`PREFIX_SCHEMA_NO_GO` for the frozen G2-v1.1 C1 run**. This is a failed feasibility gate, not a C1/C2 effect estimate.

The [machine-readable disposition](P6_C1_G3_FOLD1_PREFIX_SCHEMA_NO_GO_20260929.json) binds the original failure and lock file hashes.

## Observed evidence

| Item | Record |
|---|---|
| Run | `experiments/p6/c1_detector_aligned/c1-prefix-oos-v1-seed42/` |
| Run lock | Git HEAD `313c2d8a0db283e2fde1e87cbf65fd89309b1a79`; `run_lock.json` SHA-256 `0c5bffea3ae934540fb26674290cc1f2d6dac4da65bf6eb80fff08673b1b9bc9` |
| Fold 1 failure | `folds/fold_01/failure.json` SHA-256 `7df74cb096c0387ac40563a89c692dda8b0134e230a93483bac81df778a47d68`; message: `qualified real Metric slots 27 below required budget 45; refusing padding` |
| Incomplete marker | `folds/fold_01/INCOMPLETE.json` SHA-256 `b3fc5bb7ce86392fc5037cb3a9c698f00d83ac98cea035bca5ead4b9b530b4d2` |
| Frozen policy | Metric `base_slots=45`, dimension 48 including three observability channels; Pearson and Spearman redundancy thresholds 0.995; global/host scope quotas 30/15. |
| Locked Fit interval | `[1625133600000, 1625394960000)` ms, 8,712 bins at 30 seconds. |

The run lock and current source/binding hashes were checked before this report was committed. The failure directory contains no completed fold manifest or detector output. No OOS cohort, RCA fit, Test prediction lock, or C1/C2 evaluation was produced.

A separate filename-only index check found 474 recognized Metric source groups across the bound full-corpus catalog (5,724 grouped files of 6,640 Metric CSV files). That catalog check reads no time-series values and cannot establish how many groups pass the fold 1 Fit-prefix quality rules.

## Interpretation and limits

`fit_metric()` applies Fit-prefix quality checks and correlation reduction before the required-slot gate. The exception's 27 is the number of remaining real candidates at that gate. Scope quotas cannot account for the deficit: the selector draws from still-qualified candidates outside the preferred quotas until the 45-slot budget is met or candidates are exhausted. An isolated call with 27 candidates reproduced the exact error; 50 qualified host candidates passed the same gate with 45 selected.

The frozen [G2 design](P6_C1_G2_FROZEN_DESIGN.md) explicitly says that an early prefix unable to provide 48D Metric input is `PREFIX_SCHEMA_NO_GO`, with no future-selected padding or silent model change. The original `failure.json` uses the generic `INCOMPLETE` status and does not itself spell out that design-level classification. This report makes the classification without editing the original run. The failed fit did not persist a per-feature rejection ledger, so the record does not establish which individual quality or correlation rule removed each candidate.

**Decision:** stop this run and preserve its exact bytes. Do not run folds 2/3, `cohort`, `fit-rca`, `lock-test`, or `evaluate` under this run ID. The synthetic 45-slot smoke and read-only raw-content preflight remain valid for their narrower claims; neither established real fold 1 schema feasibility. Any revised dimension, feature policy, fold geometry, or rescue method requires a separately frozen protocol and new run ID before data processing. The currently reused Test cannot serve as independent confirmation.

The next bounded study is the [independent Fit-prefix candidate audit](P6_C1_FOLD1_METRIC_PREFIX_AUDIT_PLAN_20260929.md). Its full raw scan remains a manual command and writes to a new directory outside this failed run.
