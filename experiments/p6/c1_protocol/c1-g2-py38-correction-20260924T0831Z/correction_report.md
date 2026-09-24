# P6-C1 G2 Python 3.8 compatibility correction

Date: 2026-09-24 (UTC). Status: **G2 correction validated; G3 implementation pending; formal C1 execution NO-GO**.

## Cause and scope

The original G2 checker and raw-content binder used `Path.is_relative_to`, which is unavailable in the project's DAG Python 3.8.20. Both read-only commands stopped at their path-containment checks before validating the bound inputs. The correction uses `Path.relative_to` with `ValueError` handling at all three call sites. It rejects sibling path prefixes, retains the original containment rule, and changes no preprocessing, training, split, model, metric, or Test-label decision.

The original [G2 v1 design-lock record](../c1-g2-v1-20260924/design_lock_report.md), [v1 config](../../../../configs/e2e/gaia_p6_c1_g2_v1.json), raw inventory, and frozen [design document](../../../../docs/P6_C1_G2_FROZEN_DESIGN.md) are unchanged. The [v1.1 config](../../../../configs/e2e/gaia_p6_c1_g2_v1_1.json) binds the corrected verifier source. Its `raw_manifest_creation_source` separately identifies the original binder at commit `00cc3e3f717a944ebf7a3d2c1df0dba761ca8ca6`; the corrected binder verified that historical manifest, and did not recreate it. After normalizing only the correction metadata, protocol ID, and verifier source binding, v1.1 equals v1 as structured JSON.

## Verification

| Gate | Result |
|---|---|
| Red reproduction in DAG Python 3.8.20 | Both original commands raised `AttributeError: 'PosixPath' object has no attribute 'is_relative_to'` |
| Regression tests | 3 passed, including path-prefix rejection and both Python 3.8 call sites |
| `check_c1_g2_protocol.py --require-new-run` | `PASS G2 design and bound inputs; C1 execution remains blocked pending G3` |
| Full `bind_c1_raw_inputs.py check` | `PASS 6661 31583130281 ee1ce9d50bf4002636223b31045e86c3d7000736700201ff655bb05a74776052` |
| G2 v1 versus v1.1 design comparison | PASS after normalizing correction metadata and source identity |

The [machine-readable correction manifest](correction_manifest.json) records SHA-256 identities for the old and new sources, config, tests, and raw input manifest. The pre-existing RCA raw-index validation remains in the original G2 record; it was not needed to diagnose this Python compatibility fault.

## Read-only commands available now

Run from the repository root in the DAG environment:

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python -m unittest discover -s tests -p 'test_p6_c1_g2_python38.py' -v
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/check_c1_g2_protocol.py --require-new-run
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/bind_c1_raw_inputs.py check --manifest experiments/p6/c1_protocol/c1-g2-v1-20260924/raw_content_manifest.json
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python - <<'PY'
from pathlib import Path
from src.e2e.gaia_rca_adapter import validate_raw_index_manifest
print(validate_raw_index_manifest(Path('data/p5/v3/rca_raw_index/index_manifest.json')))
PY
```

The raw-content check reads and hashes 31.58 GB; the index check reads the frozen RCA arrays. These commands create no new experiment run. G3 still needs fold-specific implementation, source snapshot, and isolated smoke before any formal C1 preprocessing, training, inference, or evaluation command can be published or executed.
