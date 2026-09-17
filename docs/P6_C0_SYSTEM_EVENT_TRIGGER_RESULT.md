# P6-C0 Root-Agnostic System Event Trigger — Result

Protocol (frozen before the Test run): `docs/P6_C0_SYSTEM_EVENT_TRIGGER_PROTOCOL.md`
Artifacts: `experiments/p6/system_event_trigger/`
Preceding correction: `docs/P6_B0_SCORE_DECOMPOSITION_AUDIT.md` (P6-B0R)

---

## 1. Executive summary

P6-C0 asked whether a Stage-1 detector whose supervision never mentions a service
can carry GAIA system-level event triggering. The answer from this round is
**yes at the aggregate level, with two clearly localised failure modes.**

1. **The detector works.** Test `Precision = 0.9962`, `Recall = 0.7254`,
   `F1 = 0.8395` (`TP 4198 / FP 16 / FN 1589`, `4214` predicted episodes over
   `5787` GT events), mean delay `23.77 s`, median `23.76 s`, P95 `39.15 s`.
2. **All three pre-registered acceptance margins are met**
   (`P >= 0.90`, `R >= 0.60`, `F1 >= 0.72`). No Test-based tuning was performed;
   the checkpoint and the threshold come from Detector-Validation only.
3. **The verdict is nonetheless BORDERLINE**, because the pre-registered rule
   requires the gate *and* no stratification failure, and two pre-registered
   stratification failures are present: long-duration events (`>300 s`, `n=229`,
   recall `0.1397`) and `memory_anomalies` (`n=204`, recall `0.1471`). These are
   **the same 229 events** — in this registry, duration and fault type are
   perfectly confounded (every `login_failure` is 11 s; every event longer than
   300 s is `memory_anomalies`, `file_moving`, `access_permission_denied` or
   `normal_memory_freed`).
4. **The dominant population is handled well.** `login_failure` is `95.8 %` of
   Test GT (`n=5546`) and is detected at recall `0.7510`; the overall result is
   essentially that population, so the aggregate number is not carried by a
   minority stratum. Multi-onset bins (several injections sharing a 30 s bin)
   lose about half the recall (`0.4141` vs `0.7769` single-onset).
5. **A 30 s label-alignment defect was found by the failure audit of the first
   run and corrected before this result was produced** (Section 8). The defect
   degraded the first run to `F1 0.7021`; the corrected run reaches `0.8395`.
   Both runs are recorded; the corrected run is the formal P6-C0 result.
6. **Root-agnostic supervision costs nothing at the event layer.** The corrected
   trigger exceeds the historical node-supervised fused Ada-MGAD trigger on all
   three event metrics (`P +0.0135`, `R +0.0855`, `F1 +0.0644`), while never
   consuming a node anomaly label or a root-service identity. This is a
   historical reference, not a controlled equal-protocol comparison (Section 6).
7. The detector emits one scalar system trigger score. It produces no root
   service, no service ranking and no ten-way service output; Stage 2 (RCA) was
   not run in this round.

| quantity | value |
|---|---:|
| parameters | 2,903,354 |
| selected Validation epoch | 8 |
| frozen Validation threshold | 0.9998264908790588 |
| Validation P / R / F1 | 0.9939 / 0.7322 / 0.8432 |
| Test P / R / F1 | 0.9962 / 0.7254 / 0.8395 |
| Test TP / FP / FN | 4198 / 16 / 1589 |
| Test mean / median / P95 delay | 23.77 s / 23.76 s / 39.15 s |
| Test bin AUROC / AP | 0.9547 / 0.9558 |
| pre-registered gate | **PASSED** |
| verdict | **BORDERLINE** |

---

## 2. Provenance

| item | value |
|---|---|
| git commit | `cedc4a2` (`fix: rasterize P6-C0 trigger labels on the prediction-time grid`) |
| trigger config | `configs/e2e/gaia_p6_c0_system_trigger.json` — `a2c42beb19f5…` |
| frozen base config | `configs/e2e/gaia_p5_v3_preprocessing_v2.json` — `25abf4e8008e…` |
| frozen AD data manifest | `3b228061fd66…` (unchanged) |
| frozen GT registry | `7bc1b8b9c991…` (unchanged) |
| checkpoint | `best_validation_event_f1.pt` — `6a4317d64147…` |
| random seed / threads | 42 / 8 |
| policy flags | `no_retraining_of_ada_mgad`, `ada_mgad_checkpoint_loaded=false`, `node_anomaly_labels_read=false`, `root_service_label_used=false`, `rca_retrained=false`, `conditional_logit_touched=false`, `reintroduced_error_class=false`, `test_used_for_checkpoint_selection=false`, `test_used_for_threshold_selection=false`, `gate_values_changed_after_test=false`, `writes_into_frozen_artifact_tree=false` |

Validation replay: the frozen checkpoint and threshold reproduce the recorded
Validation metrics exactly (`P 0.9939 / R 0.7322 / F1 0.8432`, `TP 2124 / FP 13 /
FN 777`), i.e. the cross-process replay is exact under the same execution
configuration. The frozen Ada-MGAD checkpoint, calibration, prediction,
event-detection and RCA artifacts were re-hashed after the run and are
byte-identical to their recorded values.

---

## 3. Split and label audit (pre-training)

50 / 20 / 30 chronological split, 300 s history purge, deterministic:

| split | block | windows | GT events | positive / ignore / negative bins |
|---|---|---:|---:|---|---|
| Detector-Fit | `[1625133600000, 1626440400000)` | 43,550 | 7,443 | 12,897 / 4,682 / 25,970 |
| Detector-Validation | `[1626440400000, 1626963120000)` | 17,414 | 2,901 | 4,887 / 1,676 / 10,851 |
| Final Test | `[1626963120000, 1627747200000)` | 26,127 | 5,787 | 9,682 / 2,673 / 13,772 |

Test windows are **identical** to the frozen P5 Test population (same count, same
sample indices; asserted by `test_population_matches_frozen_p5` and
`test_window_identity_matches_frozen_p5`).

Purge: 11 windows (10 at the Fit/Validation boundary, 1 at the Validation/Test
boundary), whose labels were 2 positive / 6 ignore / 3 negative; 1 GT event
(`gaia-v3-10508`) crosses the frozen 70/30 boundary and is purged from every
metric population. All 11 pre-registered feasibility checks pass:
`fit/validation/test_has_gt_events`, `..._has_positive_labels`,
`dominant_fault_type_login_failure_present_in_fit`,
`dominant_service_mobservice{1,2}_present_in_fit`, the two Test-identity checks.

Fault and duration structure (audit only, never a model input): `login_failure`
is `95.8 %` of Test GT and is the only `<= 15 s` fault type; `mobserved1/2`
together are `96.5 %` of Test GT. Duration strata `15-30 s`, `30-60 s` and
`60-300 s` are **empty** in this registry, so the duration analysis is a
two-point comparison (`<= 15 s` vs `> 300 s`), not a resolution sweep.

---

## 4. Training and Validation selection

Fit-only gradients (no node labels, no root labels), masked BCE with
`pos_weight = 25,970 / 12,897 = 2.0135` computed on Fit only, AdaBelief
`lr 1e-3` / `wd 5e-4`, StepLR(10, 0.5), batch 32, seed 42, `max_epochs=30`,
patience 8. Training early-stopped after epoch 16 (`validation_event_f1_patience`).

| epoch | Val P | Val R | Val F1 |
|---:|---:|---:|---:|
| 0 | 0.9101 | 0.5932 | 0.7183 |
| 1 | 0.9947 | 0.7108 | 0.8291 |
| 5 | 0.9682 | 0.7242 | 0.8286 |
| 7 | 0.9887 | 0.7215 | 0.8342 |
| **8** | **0.9939** | **0.7322** | **0.8432** |
| 13 | 0.9796 | 0.7266 | 0.8344 |
| 16 | 0.9748 | 0.7215 | 0.8292 |

Selected: epoch 8, threshold `0.9998264908790588` (exact unique-validation-score
sweep maximising Validation event F1, highest threshold on ties).

---

## 5. Test event detection (single frozen evaluation)

### 5.1 Overall

| quantity | value |
|---|---:|
| GT events | 5,787 |
| predicted episodes | 4,214 |
| TP / FP / FN | 4,198 / 16 / 1,589 |
| precision | 0.99620 |
| recall | 0.72542 |
| F1 | 0.83952 |
| mean delay | 23.774 s |
| median delay | 23.760 s |
| P90 / P95 delay | 37.199 s / 39.148 s |

Validation → Test consistency is tight (F1 `0.8432 → 0.8395`, R `0.7322 →
0.7254`, P `0.9939 → 0.9962`), so the Test result is not an optimistic draw
relative to the selection split.

### 5.2 Bin diagnostic (auxiliary only)

AUROC `0.9547`, AP `0.9558`, BCE `0.2448` over `23,454` scored bins
(`9,682` positive, `13,772` negative, `2,673` IGNORE masked). The score is highly
discriminative at bin level, so the recall loss is not a ranking failure; it is an
operating-point and episode-alignment effect under the Validation-selected
threshold.

### 5.3 Duration stratification

| stratum | n | TP | FN | recall | mean delay | median delay | P95 delay |
|---|---:|---:|---:|---:|---:|---:|---:|
| `<= 15 s` | 5,558 | 4,166 | 1,392 | 0.7496 | 23.82 s | 23.84 s | 39.14 s |
| `15-30 s` | 0 | 0 | 0 | — | — | — | — |
| `30-60 s` | 0 | 0 | 0 | — | — | — | — |
| `60-300 s` | 0 | 0 | 0 | — | — | — | — |
| `> 300 s` | 229 | 32 | 197 | **0.1397** | 17.99 s | 14.10 s | 41.76 s |

### 5.4 Fault-type stratification

| fault type | n | recall | small-n |
|---|---:|---:|:---:|
| login_failure | 5,546 | 0.7510 | no |
| memory_anomalies | 204 | **0.1471** | no |
| file_moving | 16 | 0.0625 | yes |
| cpu_anomalies | 12 | 0.0833 | yes |
| access_permission_denied | 5 | 0.2000 | yes |
| normal_memory_freed | 4 | 0.0000 | yes |

### 5.5 Root-service stratification (audit only)

| root service | n | recall | small-n |
|---|---:|---:|:---:|
| mobservice1 | 2,799 | 0.7381 | no |
| mobservice2 | 2,787 | 0.7557 | no |
| dbservice1 | 33 | 0.1515 | no |
| dbservice2 | 32 | 0.1875 | no |
| webservice1 | 30 | 0.0667 | no |
| webservice2 | 30 | 0.0667 | no |
| redisservice1 / redisservice2 | 20 / 18 | 0.2000 / 0.2778 | yes |
| logservice2 / logservice1 | 20 / 18 | 0.1000 / 0.0000 | yes |

The low non-mob rows are **not** evidence of a service-specific bias: the root
service is perfectly confounded with the fault type and the duration in this
registry (every non-mob, non-login-failure injection is a 600 s / 3600 s event).
The detector never consumes the service identity in any form, so this table
measures which *fault processes* are hard to trigger, not which services are.

### 5.6 Onset-density stratification

| stratum | n | TP | recall | mean delay | median delay |
|---|---:|---:|---:|---:|---:|
| single onset in its 30 s bin | 4,966 | 3,858 | 0.7769 | 24.84 s | 24.63 s |
| multiple onsets in its 30 s bin | 821 | 340 | 0.4141 | 11.70 s | 9.02 s |

Onset-count histogram `1: 4,966`, `2: 782`, `3: 39`. Test bins with more than one
onset: `996`; bins with more than one root service: `531`; events sharing a bin
with another event: `821` (14.2 % of Test GT); events sharing a bin with a
different root service: `427`. The ratio `0.4141 / 0.7769 = 0.533` stays just
above the pre-registered `0.5x` failure boundary, so this is reported as a
degradation, not as a triggered failure.

---

## 6. Comparison with the historical node-supervised trigger

| trigger | supervision | Test P | Test R | Test F1 |
|---|---|---:|---:|---:|
| P5 formal fused Ada-MGAD (historical) | node anomaly label (= root identity) | 0.9827 | 0.6399 | 0.7751 |
| P6-C0 root-agnostic (this round, corrected) | system recent-onset only | **0.9962** | **0.7254** | **0.8395** |
| P6-C0 first run (label-lag defect, Section 8) | system recent-onset (shifted) | 0.9647 | 0.5519 | 0.7021 |

The three rows are **not** an equal-protocol comparison: the historical trigger is
node-supervised, uses the frozen 70/30 split without a validation block, and was
selected by a Train-only threshold; P6-C0 uses 50/20/30, a recent-onset label and
a Validation-selected threshold. It is quoted as a historical reference only.

---

## 7. Pre-registered decision

| margin | required | observed | met |
|---|---:|---:|:---:|
| Precision | ≥ 0.90 | 0.9962 | yes |
| Recall | ≥ 0.60 | 0.7254 | yes |
| F1 | ≥ 0.72 | 0.8395 | yes |

```text
gate = PASSED
stratification failures = [duration >300 s: n=229, recall=0.1397]
                          [fault type memory_anomalies: n=204, recall=0.1471]
verdict = BORDERLINE
```

Per the pre-registered rule, `GO` requires the gate *and* no stratification
failure; because two pre-registered failure conditions hold, the verdict is
**BORDERLINE**. Per the same rule, a BORDERLINE verdict permits a failure audit
only: **no Test tuning, no threshold rescue, no architecture search, no RCA, no
node-level supervision fallback was performed.**

---

## 8. Implementation defect found and corrected (P6-C0R)

The first run produced `P 0.9647 / R 0.5519 / F1 0.7021` (BORDERLINE). Its
failure audit surfaced the cause.

**Defect.** The pre-registered label is defined at the prediction time
`t = prediction_available_time = target_bin_end`, but the label array was
rasterized on the bin-start grid and indexed with a window's target bin index.
The supervision target therefore lagged the specified definition — and the
evaluation anchor `t_hat` — by exactly one 30 s bin.

**Evidence in the first run.** `576 / 3,194` matched Test episodes carried a
`NEGATIVE` label at their own `t_hat` while satisfying
`0 <= t_hat - gt_start <= 60 s`; `2,011 / 3,194` matches had delay `>= 30 s`
(mean delay `33.77 s` vs `23.77 s` after the fix). The model had learned to fire
one bin late, exactly as the shifted target implies.

**Correction** (`cedc4a2`). Rasterize on
`prediction_time_grid(timestamps) = timestamps + grid`, fail closed if a
bin-start grid is passed (`assert_prediction_time_grid`), plus regression tests
(`LabelGridAlignmentTests`, `LabelPredictionTimeContractTests`) that fail under
the previous behaviour. The label rule, split, purge, model, loss, selection
rules and gate are unchanged; only the grid the rule is evaluated on was
corrected. The split audit changed by one label
(Fit positive `12,898 -> 12,897`); windows, purge and GT counts are identical.

**Effect.** Validation F1 `0.6882 -> 0.8432`; Test F1 `0.7021 -> 0.8395`,
recall `0.5519 -> 0.7254`, delay `33.77 s -> 23.77 s`. The first run's artifacts
are kept locally at `experiments/p6/system_event_trigger_v1_label_lag/` and are
not versioned; the corrected run is the formal P6-C0 result.

---

## 9. Failure audit of the corrected run (why BORDERLINE)

**F1. Long-duration events are essentially missed.** `> 300 s` events: recall
`0.1397` (`32/229`), and `memory_anomalies` (600 s) `0.1471`, `file_moving`
(600 s) `0.0625`, `normal_memory_freed` (600 s) `0.0`,
`access_permission_denied` (3600 s) `0.2000`. Duration and fault type are
perfectly confounded in this registry, so these are one failure, not two. Note
that the delay on the few long events that *are* detected is small
(median `14.10 s`): the detector does not mis-locate them, it simply does not
fire. A 600 s / 3600 s injection presents a nearly identical system state for
its whole duration, while the recent-onset label rewards firing only in the 60 s
after the onset; the detector therefore converges to a state-change-sensitive
policy and under-fires on sustained faults. Removing the IGNORE band is the
labelled remedy candidate (see below), not a Test-side change.

**F2. The dominant 11 s population sets the ceiling.** `login_failure` is
`95.8 %` of Test GT at recall `0.7510`; the aggregate recall is that number. The
`FN` count is `1,589`, of which `1,381` are `login_failure`. Since bin-level
AUROC is `0.9547`, the residual is not a ranking failure: the episodes that are
produced are almost all correct (`FP = 16`), but `1,589` onsets produce no
matching episode within the 60 s causal window at the Validation-selected
operating point.

**F3. Onset collisions cost about half the recall.** Multi-onset bins
`0.4141` vs single-onset `0.7769`. With several injections starting inside the
same 30 s bin, the system-level score has to cross the threshold once per onset;
the merged positive region gives no additional supervision for the second and
third onset, and only one of them can be matched per episode.

**F4. Nothing indicates a threshold or calibration pathology.** Precision is
`0.9962`, the Validation replay is exact, and Validation/Test agree closely, so
the operating point is stable and honest; it is simply a high-precision point
whose recall is not enough for the strict "no stratification failure" GO rule.

---

## 10. Limits, anomalies and open questions

1. **Duration coverage is degenerate.** Only `<= 15 s` (2.9–11 s) and `> 300 s`
   (600 s / 3600 s) injections exist in the frozen registry; the three middle
   strata are empty. A duration-resolution claim cannot be made from this data.
2. **Fault-type / duration / service are perfectly confounded** for the failing
   population, so P6-C0 cannot separate "does not fire on sustained faults" from
   "does not fire on `memory_anomalies`". A controlled stratification would need
   injections that do not exist in the frozen registry.
3. **The `IGNORE` band is a hypothesis, not a finding.** The masked band removes
   any supervision inside sustained faults after 60 s; whether it causes F1 is
   untested here. Testing it requires a new pre-registered round, not a change to
   this one.
4. **Threshold semantics are score-scale bound.** The frozen threshold
   `0.9998` sits very close to 1 on a sigmoid score produced by a randomly
   initialised model; the exact value is not interpretable across models and must
   always be carried with the checkpoint.
5. **`cpu_anomalies` never appears in Fit or Validation** (Test `n=12`), so its
   `0.0833` recall is an out-of-training-distribution observation, not a measured
   capability. It is below the 10 % dominance threshold and therefore not a
   pre-registered feasibility failure.
6. **Stage 2 is untouched.** No RCA, no anchor alignment, no 68D features and no
   conditional-logit work was performed; whether these `t_hat` values are good
   anchors for Ada-RCA is the open question for P6-C1.
7. **Single execution configuration.** The run used 8 CPU threads (recorded in
   the manifest). The Validation replay is exact within that configuration; a
   different thread count is not guaranteed to reproduce the same threshold
   comparison at the last bit (the effect documented in P6-B0).

---

## 11. Artifacts and reproduction

```text
experiments/p6/system_event_trigger/
├── split_audit.json              pre-training split/label/purge audit
├── split_manifest.json           frozen split + purge record
├── validation_selection.json     epoch/threshold selection + full epoch history
├── training_log.json             parameter count, window counts, history
├── test_metrics.json             frozen Test metrics, stratifications, decision
├── manifest.json                 provenance, policy flags, hashes
├── checkpoint/best_validation_event_f1.pt         local (not versioned)
├── test_predictions.csv         local (row-level window scores)
├── test_episodes.csv            local
└── test_matching.csv            local
```

```bash
python scripts/p6/run_c0_trigger.py audit
python scripts/p6/run_c0_trigger.py train --threads 8 --threshold-workers 8
python scripts/p6/run_c0_trigger.py evaluate --threads 8
```

Tests: `pytest tests/test_p6_c0_trigger_protocol.py tests/test_p6_c0_trigger_model.py`
(63 tests) plus the full suite (200 tests) pass.
