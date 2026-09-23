# P6-C1 G2 design-lock completion record

Date: 2026-09-24 (Asia/Shanghai)

Status: **DESIGN_LOCK_COMPLETE; G3 IMPLEMENTATION PENDING; C1 EXECUTION NO-GO**

The [protocol](../../../../docs/P6_C1_G2_FROZEN_DESIGN.md) fixes the
three Fit-only folds, detector prefix fitting, the fixed transductive
filename-catalog limitation, same-case RCA arms and shared GT-fitted scaler,
common-cohort floors, prediction firewall, evaluation and compute budget.
The [machine config](../../../../configs/e2e/gaia_p6_c1_g2_v1.json) binds
G1 and C0 source files, the P5 RCA raw-index manifest, and this run's raw
content inventory. The intended formal run ID is
`c1-prefix-oos-v1-seed42`; its directory has not been created.

| Bound item | SHA-256 |
|---|---|
| G2 machine config | `58bf044fef89674ba5bf472c57b70199ca39e41922801217dd860ca5f6d87b92` |
| G2 protocol document | `4df98633934bc18e1954187d82d2e358573035bb4c56218f5e6f7706bf95007a` |
| Raw content manifest | `03bdefc1ad817802a9f015a5cc07c71f5e8d625c66c5d5e9fbff31d0d5671718` |
| Raw binder source | `67a793896312c5d2cac8df79e3269676ffb341159d157bd6dae529c0191c5225` |
| G2 checker source | `0f8952dac99a8edd641051b2458ed062960631d95875f9afd4fd0ac2deec5d85` |
| G1 static ledger | `80718e06a801f4a4cfac085eec999285a162b4e1ebf5afff082f17925a99b3da` |

Read-only validation completed:

- Raw-content create and independent check agreed on **6,661 files**,
  **31,583,130,281 bytes**, aggregate records SHA-256
  `ee1ce9d50bf4002636223b31045e86c3d7000736700201ff655bb05a74776052`.
- The G2 checker passed source bytes, all 14 G1 direct inputs, fold geometry,
  C0 checkpoint/Test bindings, RCA model identity, resource choices and the
  Test label firewall; `--require-new-run` also passed.
- The existing RCA index validator verified **3,788 array files** against
  the frozen index manifest.
- Syntax parsing and staged whitespace checks passed. No preprocessing,
  detector/RCA training, C1 inference, Test evaluation or label-based
  selection was run.

The current executable commands are in protocol section 5. Old P5/C0 `all`
commands are not C1 commands. G3 must implement and smoke the required
interfaces before a full C1 command can be published or executed.
