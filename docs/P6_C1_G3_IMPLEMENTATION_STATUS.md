# P6-C1 G3 implementation status

Date: 2026-09-28. Status: **PARTIAL_G3_IMPLEMENTATION; FORMAL_C1_EXECUTION_NO_GO**.

The frozen [G2 design](P6_C1_G2_FROZEN_DESIGN.md) and [v1.1 source binding](../configs/e2e/gaia_p6_c1_g2_v1_1.json) remain unchanged. This implementation adds a fold-local Ada-MGAD input materializer in [`c1_fold_preprocessing.py`](../src/e2e/c1_fold_preprocessing.py). It fits Metric, Log and Trace only on the locked fold Fit prefix, then transforms Fit, Selection and Generation separately using the fitted objects. The output has an exclusive new `folds/fold_XX` directory, frozen 48D/32D/8D schema, source and raw-content hashes, Drain3 state, graph, segment arrays, legal prediction-window indices and an output completion manifest. Trace diagnostics remain separate by segment. The source uses the historical raw manifest as an input; it does not modify it. [`c1_fold_detector_data.py`](../src/e2e/c1_fold_detector_data.py) reads only these sealed windows; Generation samples have no label field or label accessor.

The raw-content verifier checks all listed bytes before creating a fold directory and again before publishing its completion manifest. Each segment's legal window indices exclude predictions whose availability time equals that half-open segment's right boundary. An interrupted or failed fold retains `INCOMPLETE.json`; the writer refuses to reuse that directory. `validate_c1_fold` checks the completed file hashes, source identities, schema, graph and segment geometry.

## Evidence and limits

| Item | Status |
|---|---|
| Synthetic fold orchestration | PASS: Fit-only calls, separate segment arrays, 48/32/8 shape, legal-window boundary, label-free outputs, hash validation, tamper detection and incomplete-directory retention |
| Detector input firewall | PASS: 2 isolated tests for explicit Fit labels, Generation without labels, boundary windows and invalid index rejection |
| Existing V2 preprocessing tests | PASS: 17 tests |
| Existing C0 trigger protocol tests | PASS: 33 tests |
| Real GAIA prefix fit and target dimensions | PENDING; the synthetic test replaces expensive raw fit/transform primitives and cannot establish data feasibility |
| Workers=1 versus workers=24 numerical equivalence | PENDING; no formal parallel use is authorized by these tests |
| Fold detector training/selection/inference, OOS episode/matching, common Train RCA cohort, shared scaler, label-free Test scorer, prediction lock and evaluator | PENDING |

Only the G3 fold-input contract and synthetic orchestration have been checked. No real C1 fold data, OOS anchors, common Train cohort, rankings, Test metrics or full run directory were created. The G1 count 4,254 remains a static upper bound.

## Next implementation gate

Implement a bounded raw-data smoke that actually exercises the three modality fit/transform functions on a new isolated fixture, then verify workers=1 and workers=24 produce identical schema, graph and arrays. If an early Fit prefix cannot produce the frozen dimensions, record `PREFIX_SCHEMA_NO_GO` without padding or changing the fold. After that, implement fold-specific detector Fit → Selection → label-free Generation using the legal window indices; Generation labels may be joined only after episode production for Train matching. The later RCA, prediction-lock and evaluator interfaces remain separate G3 work.

The current read-only synthetic check is:

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python -m unittest discover -s tests -p 'test_p6_c1_fold*.py' -v
```

There is no formal C1 execution command. Do not invoke the fold materializer over the complete raw corpus until the remaining G3 gates pass and a new formal run is explicitly requested.
