# P6-C1 fold 1 Metric prefix audit plan

## Material Passport

- Origin Skill: academic-research-suite experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-29
- Verification Status: SCRIPT_AND_SYNTHETIC_CHECKS_ONLY; REAL_PREFIX_AUDIT_PENDING
- Version Label: c1_fold1_metric_prefix_audit_v1

## Question and boundary

The frozen C1 run stopped because fold 1 produced 27 qualified real Metric slots against the required 45. This independent diagnostic asks how many candidates were rejected by filename/scope eligibility, Fit-prefix quality checks, and Pearson/Spearman redundancy filtering. It tests whether the candidate path reproduces the recorded 27. It does **not** select a replacement feature policy, train a detector or RCA model, or parse Selection/Generation/Test telemetry or GT labels. Bound raw files outside Fit are hashed for identity only.

The input interval is the locked fold 1 detector Fit prefix `[1625133600000, 1625394960000)` ms. The script calls the same raw Metric value, quality and correlation functions as the formal adapter, with the same candidate order, 30-second bins, 0.20 minimum coverage, two unique values, `1e-4` dynamic ratio and 0.995 Pearson/Spearman thresholds. It checks the raw reader, Metric helper, fold interval code and parallel executor against the failed run's source lock; it also hashes the bound raw catalog before and after the scan. A successful diagnostic must reproduce **27** after correlation; any other count is a replay mismatch, not permission to continue C1.

## Manual execution

The `preflight` command is bounded and does not read time-series values. It has passed locally. The `run` command performs full raw-content verification and scans fold 1 Metric values; execute it manually only. The new output ID is exclusive and must not be reused.

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/audit_c1_fold1_metric.py preflight --run-id c1-fold1-metric-audit-v1-20260929
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/audit_c1_fold1_metric.py run --run-id c1-fold1-metric-audit-v1-20260929 --workers 24
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/audit_c1_fold1_metric.py check --run-id c1-fold1-metric-audit-v1-20260929
```

The run writes only `experiments/p6/c1_prefix_metric_audit/c1-fold1-metric-audit-v1-20260929/`. It never writes inside `experiments/p6/c1_detector_aligned/c1-prefix-oos-v1-seed42/`. It uses an `INCOMPLETE.json` marker until source, raw-content, original failure evidence, count and output hashes have been checked. A failed attempt retains its directory and must not be overwritten.

The first raw-content verification happens before the output directory is created; the terminal prints its stage. The `check` command verifies the sealed output hashes and ledger accounting after `run` succeeds. If `run` fails, preserve its `INCOMPLETE.json` and `failure.json` and do not invoke `check` as if the audit were complete.

## Outputs and interpretation

| Output | Purpose |
|---|---|
| `candidate_ledger.csv` | One row per eligible Metric candidate task: source filenames, scope, target, first Fit-prefix quality failure or score, quality rank, and first correlation partner/coefficients if rejected. |
| `summary.json` | Catalog and candidate counts, quality rejection reasons, per-scope counts, correlation rejection count, final real-slot count and surviving names. |
| `completion_manifest.json` | Source Git HEAD and hashes, raw-content binding, original failure binding, worker count and output checksums. |

Interpretation remains conditional on the audit result. If quality failures dominate, inspect their observed coverage/variation and source parsing before considering any new Train-only schema. If correlation reduction dominates, inspect the rejected pairs before proposing a different representation or redundancy rule. If the count does not reproduce 27, audit the parser, input/source binding and execution equivalence first. Any change to dimensions, fold geometry, feature policy or model requires a separately frozen protocol and new run ID; the stopped G2-v1.1 run stays intact. No Test result may be used to choose that revision.

The replay uses the same Metric value parser as the stopped run. Reproducing 27 verifies its selection path, but cannot by itself prove that raw timestamp/value parsing is correct. Review a small, predetermined set of rejected candidates against their bound raw CSV rows before attributing the deficit solely to data scarcity or changing policy.
