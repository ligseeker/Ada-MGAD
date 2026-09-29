# P6-C1 fold 1 Metric prefix audit result

Date: 2026-09-29. Status: **sealed diagnostic complete; frozen C1 G2-v1.1 run remains `PREFIX_SCHEMA_NO_GO`**. This is a Fit-prefix schema feasibility result, not a detector, RCA, or E2E result.

## Bound evidence

The independent audit followed the [preregistered diagnostic plan](P6_C1_FOLD1_METRIC_PREFIX_AUDIT_PLAN_20260929.md). Its output is `experiments/p6/c1_prefix_metric_audit/c1-fold1-metric-audit-v1-20260929/`:

| File | SHA-256 |
|---|---|
| `summary.json` | `f951ebf8419d6499dfe3864e04b4c7f3ab04f258eb3c3d40a3f2ecc43623ae1a` |
| `candidate_ledger.csv` | `0b25bf30626055af55d6b35ab3fe9f8ff3a375b9023b9ade6bcc8f592923edf5` |
| `completion_manifest.json` | `322e3bf69bd9aaa16b4b0b667a3fd98802175bdd91ec655585f7c97c765ef847` |

The `check` command passed after the user's full audit. The completion manifest binds source HEAD `f6d87f98f5969a8090854262d9066bb5c73c8744`, the G2-v1.1 protocol hash, the original failure disposition, and the bound 6,661-file raw-content inventory. The audit verified the raw catalog before and after its scan. It parsed fold 1 Fit Metric values only; it did not parse GT labels or Selection, Generation, or Test values. Its own source and the original failed adapter's replay dependencies are hash-bound. The audit reused the formal Metric parser, so count reproduction alone is not an independent proof of all parsing semantics.

## Observations

The locked Fit interval was `[1625133600000, 1625394960000)` ms (2021-07-01 10:00 to 2021-07-04 10:36 UTC). The ledger contains 477 eligible candidate tasks from 474 source groups. Scope-ineligible source groups: 0.

| Stage | Count | Detail |
|---|---:|---|
| Host candidate tasks | 384 | All rejected at the first checked target, `dbservice1`, for zero Fit-prefix coverage. This is a fail-fast record; the ledger does not give every other target's coverage. |
| Global candidate tasks | 93 | 48 rejected: 28 for fewer than two unique values, 20 for zero 5th-to-95th percentile dynamic range. |
| Quality-pass candidates | 45 | All global; exactly equals the required real-slot budget before redundancy filtering. |
| Correlation rejects | 18 | Every recorded reject has absolute Pearson correlation at least 0.995 with an earlier kept candidate; 13 also cross the 0.995 Spearman threshold. |
| Post-correlation real slots | **27** | Reproduces the formal failure; **18 short** of the frozen 45 real slots. The three observability channels cannot replace real slots. |

The 18 rejected pairs fall within broad metric families: 11 memory, 4 CPU, 2 network, and 1 disk I/O. One pair (`docker_memory_stats_total_inactive_anon` and `docker_memory_stats_inactive_anon`) has recorded Pearson and Spearman coefficients of exactly 1.0. These counts describe the frozen greedy, score-ordered redundancy rule; they do not justify changing its threshold after seeing this result.

A small, predetermined raw-row check was made for the lexicographically first rejection in each quality category. The first data row of the selected host `system_core_id` file for `dbservice1` (`system_0.0.0.4_system_core_id_2021-07-01_2021-07-15.csv`) has timestamp `1625469689000` (2021-07-05 07:21:29 UTC), after Fit ends. The first data rows of the other two sampled host IP files for this metric are also after Fit (`1625467037000` and `1625470171000`). The first data rows of the sampled global `docker_cpu_system_norm_pct` and `docker_diskio_read_bytes` files for `dbservice1` are `1625133601000` and `1625133610000`, inside Fit. This row check supports the early host-coverage finding and timestamp scale for those files; it is not an exhaustive independent raw parser audit or proof that each CSV is time-sorted. The ledger, rather than this sample, supplies the aggregate counts.

## Decision and next study

**Decision:** retain the [original failed-run disposition](P6_C1_G3_FOLD1_PREFIX_SCHEMA_NO_GO_20260929.md) and its immutable `INCOMPLETE` directory. Do not resume its fold 1, run folds 2/3, train RCA, lock Test, or evaluate C1/C2 under `c1-prefix-oos-v1-seed42`. There are no OOS anchors, common Train RCA cohort, locked rankings, or new E2E results from this attempt.

The next main-study proposal is a separately versioned **supervision-OOS C1**: use the already frozen, label-free preprocessing fitted on the original Train period; fit each fold detector and select its threshold only on its chronological Fit and Selection intervals; generate anchors on its later Generation interval. This tests the same-anchor-domain RCA comparison while explicitly admitting that preprocessing saw later Train telemetry relative to each fold. It therefore supports a supervision-OOS claim, not G2's prefix-fitted input claim. A bounded read-only check found existing Train timestamps from `1625133600000` through `1626963090000` on the 30-second grid, with Metric `[60984,10,48]`, Log `[60984,10,32]`, and Trace `[60984,10,10,8]`; the three G2 fold intervals lie inside this Train span. The frozen schema declares Train-only decisions and no GT label use for schema selection. Array hashes, per-fold slice legality and execution binding still need their own gates. Before freezing this new study, verify those bindings, keep the G2 cohort/label-firewall/failure rules or enumerate each change, fix a fresh protocol and run ID, then pass synthetic and read-only identity gates. Any full fold training or evaluation remains a separate manual execution after those gates.

If prefix-fitted inputs are essential to the research question, predefine a new fold geometry using Train-only chronology, recompute the static GT/context upper bounds and cohort floors without model outcomes, and run a new, separately sealed Fit-prefix feasibility audit before training. A later Fit boundary might include host telemetry; this sample does not establish that any revised prefix passes the 45-slot gate. Do not retroactively relax the 0.995 rule, lower the dimension, fill slots with copies, or select a configuration from Test performance.

The existing Test has been examined in prior P6 work. Any future C1/C2 result on it is a reused-Test study with that limitation.
