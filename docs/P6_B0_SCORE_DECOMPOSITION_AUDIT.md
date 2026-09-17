# P6-B0 Score Decomposition / Label-Coupling Audit

## 0. Scope and evidence grading

This report is a **read-only audit** of the completed P5 formal run. It does not
retrain, re-preprocess, re-select any Test-dependent parameter, modify event
matching, modify the frozen Ada-RCA representation, or write into the formal run
directory. The new driver `scripts/p6/run_score_decomposition.py` and helper
`src/e2e/score_decomposition.py` only load the frozen checkpoint and calibration
and perform a single pure-inference pass.

Evidence grades used below:

| Grade | Meaning |
|---|---|
| **Confirmed** | Recomputed by this audit and cross-checked against frozen artifacts / identity constraints. |
| **Observed** | A relation seen in the fixed run's strata or in the controlled in-memory comparison; not a causal proof. |
| **Hypothesis** | Needs a new protocol or run to verify. |
| **Open question** | Not answerable from the current artifacts. |

Audit target:

```text
run:      experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440
config:   configs/e2e/gaia_p5_v3_preprocessing_v2.json
commit:   138a9c84385e7e636797f014577571b032cd6a04
outputs:  experiments/p6/score_decomposition/
```

### 0.1 Revision note (P6-B0R, 2026-09-17)

This is the corrected version of the P6-B0 audit. Three scientific
interpretations were clarified; **no measured number, artifact, gate input or
decision changed**. The corrections are:

1. The pre-registered raw replay condition `max_abs_diff <= 1e-6` is stated as
   **FAIL** and is never presented as passed. What is claimed instead is
   decision-level exact reproduction under small float32 numerical noise
   (Sections 3.2 and 3.4).
2. The high reconstruction-only localisation is attributed to the
   **label-conditioned reconstruction objective itself**, not only to the shared
   classification-trained backbone, with the code path cited (Section 2).
3. `exact_identity_ratio = 0.7045` is explained as a **same-bin multi-event
   overlap effect**; it must not be read as partial AD/RCA label disagreement.
   The identity quantities that matter are
   `labelled_service_positive_on_all_bins_ratio = 1.0000` and
   `rasterization_mismatch = 0` (Section 5.1).

---

## 1. Executive summary

The question this round must answer is:

> Can a reconstruction-derived system trigger carry GAIA system-level event
> detection while reducing the AD ↔ RCA label coupling?

The answer from this audit is **no, not with the current frozen Ada-MGAD.**

1. **The strict numerical replay gate is FAIL; the fused score reproduces the
   formal baseline exactly at the decision level.** The pre-registered condition
   was `max_abs_diff <= 1e-6`. The observed raw deviations are `3.7849e-06`
   (Train) and `1.6093e-06` (Test), so
   **`strict numerical replay gate = FAIL`** and this report does not claim
   otherwise. Everything the downstream decisions depend on is nevertheless
   exact: row identity (`split, sample_index, service,
   prediction_available_time, node_label`) matches the formal prediction
   artifacts with **zero** mismatches across all 871,020 rows, and the fused
   track reproduces the formal event detection **exactly**: same Train
   threshold, `TP=3703`, `FP=65`, `FN=2084`, `P=0.9827`, `R=0.6399`,
   `F1=0.7751`, mean delay `24.106 s`. Its downstream Ada-RCA `AC@1=0.4657` and
   Diagnosis `F1@1=0.3611` are identical to the formal values. The residual is
   multi-threaded CPU float32 reduction-order noise, not a protocol mismatch
   (Section 3). The correct characterisation is therefore **decision-level exact
   reproduction under small float32 numerical noise** — not "the `1e-6` gate
   passed".
2. **Reconstruction-only detection is degraded but not collapsed.** Test
   `P=0.9132`, `R=0.3999`, `F1=0.5562` (72% of fused F1, 62% of fused recall),
   `FP=220`, `FN=3473`. It is a real but clearly weaker trigger.
3. **The label coupling is not reduced.** The reconstruction score's own
   anomaly-score localization `AC@1` is `0.9084`, versus `0.9398` for the
   classification score — a gap of only `0.0300`. On the anchor-controlled
   common-case population (`n=2214`, identical GT cases and one fixed anchor
   set) the gap is only `0.019–0.025`. This is expected from the training
   objective, not incidental: in `src/model.py:164-176` the reconstruction loss
   is itself **label-conditioned** — for nodes whose node anomaly label is
   `abnormal` it optimises the *inverse* reconstruction energy
   (`node_wrong = node_rec ** -1`), i.e. it is trained to separate
   labelled-abnormal nodes from normal ones. The reconstruction branch of the
   shared backbone is therefore not a label-free evidence channel.
4. **The root margin is where the two branches actually differ.** Median
   root-vs-best-non-root margin is `0.9983` (classification), `0.7358` (fused),
   `0.1821` (reconstruction). Reconstruction still ranks the root first in
   `91%` of cases, but with much weaker separation.
5. **Trigger choice changes which events are detected, not the quality of RCA on
   detected events.** On the identical common-case population all three triggers
   give the same Ada-RCA `AC@1` (`0.4734 / 0.4729 / 0.4697`).

**Decision: Outcome C.** Score decomposition alone does not break the
AD ↔ RCA task coupling. The reconstruction trigger is partially usable for
event detection, but it does not deliver the decoupling this round was looking
for.

| summary quantity | classification | fused | reconstruction |
|---|---:|---:|---:|
| Train threshold | 0.994492 | 0.869351 | 0.661951 |
| Test event F1 | 0.7776 | 0.7751 | 0.5562 |
| Test event recall | 0.6397 | 0.6399 | 0.3999 |
| Anomaly-score localization AC@1 | 0.9398 | 0.9384 | 0.9084 |
| median root margin | 0.9983 | 0.7358 | 0.1821 |
| RCA native AC@1 (n) | 0.4647 (3701) | 0.4657 (3702) | 0.4715 (2314) |
| RCA common-case AC@1 (n=2214) | 0.4734 | 0.4729 | 0.4697 |
| Diagnosis F1@1 | 0.3615 | 0.3611 | 0.2624 |

---

## 2. Protocol

Fixed, not re-selected:

```text
No retraining                     : Ada-MGAD forward only, model.eval(), torch.no_grad()
No calibration re-fit             : reconstruction_calibration.json loaded verbatim
No conditional-logit re-fit       : conditional_logit.npz loaded verbatim
Same checkpoint                   : best_train_f1.pt (SHA-256 eb9ae9a8…f65263)
Same preprocessing / Train-Test   : frozen V2 arrays and chronological 70/30 boundary
Same graph / same window          : 30 s grid, window_bins=10, t_hat = target_bin_end
Same event matching               : causal max-cardinality + minimum-delay, 60 s tolerance
Same RCA 68D W300-B15 representation and window
Same RNG seed (42)
No alpha sweep was performed
```

Fusion is unchanged and literal:

```text
fused_score = 0.7 * classification_score + 0.3 * reconstruction_score
classification_score = softmax(show(rec))[:, 1]           # node classification P(anomaly)
reconstruction_score = sigmoid((E_rec - 2741…)/(1.4826*MAD))  # frozen Train calibration
```

Three fixed score tracks, each with its **own** Train-only threshold selected by
the frozen rule (exact unique Train-score enumeration → maximize Train event F1
→ on equal F1 take the highest threshold). Test only ever receives the frozen
Train threshold:

| track | node score | system score |
|---|---|---|
| classification | `classification_score` | `max_i classification_score(t,i)` |
| fused | `0.7*cls + 0.3*rec` | `max_i fused_score(t,i)` |
| reconstruction | `reconstruction_score` | `max_i reconstruction_score(t,i)` |

> **Reconstruction-only is not unsupervised — two separate label paths survive
> the decomposition.**
>
> 1. *Shared representation.* In the original Ada-MGAD training the shared
>    backbone is driven by the node-classification loss, so even a
>    "reconstruction-only" trigger consumes a representation shaped by the node
>    anomaly label.
> 2. *Label-conditioned reconstruction objective.* The reconstruction loss
>    itself also applies **different targets to normal and abnormal nodes
>    according to the node anomaly label**; it is not a label-free
>    reconstruction error. `src/model.py:164-176`:
>
>    ```python
>    label_pod = torch.argmax(x['groundtruth_cls'], dim=-1)
>    node_rec  = torch.sum(rec, dim=-1)
>    node_right  = torch.where(label_pod == 0, node_rec, 0)          # normal   -> minimise energy
>    node_wrong  = torch.where(label_pod == 1, node_rec ** -1, 0)    # abnormal -> maximise energy
>    node_unkown = torch.where(label_pod == 2, self.label_weight * node_rec, 0)
>    rec_loss = [node_right, node_wrong, node_unkown]
>    ```
>
>    `groundtruth_cls` is the rasterized per-service node anomaly label (the
>    `label_mask` produced by `build_semisupervised_mask`). The `abnormal` term
>    is the *inverse* of the reconstruction energy, so the reconstruction score
>    is explicitly trained to separate labelled-abnormal nodes from normal
>    nodes. Its high root-localisation `AC@1` (Section 5.2) is the expected
>    consequence, not an unexplained surprise.
>
> Consequence for the interpretation: this experiment only reduces the
> *inference trigger's* direct dependence on the node-classification score. It
> removes neither the classification objective from the representation nor the
> label conditioning from the reconstruction objective. A genuinely
> root-agnostic Stage-1 signal cannot be obtained by re-weighting these two
> scores; it requires an objective that never consumes the node anomaly label.

`prediction_available_time` and the 30 s episode rule are identical to P5;
"anomaly-score localization" below is a first-stage score diagnostic and is
**not** reported as RCA performance.

Provenance (`experiments/p6/score_decomposition/manifest.json`):

| item | SHA-256 |
|---|---|
| checkpoint `best_train_f1.pt` | `eb9ae9a83ef4…65263` |
| `reconstruction_calibration.json` | `7c58b3e20cf4…6536f` |
| `ad_data_manifest.json` | `3b228061fd66…34c2` |
| config `gaia_p5_v3_preprocessing_v2.json` | `25abf4e8008e…e35` |
| formal `ad_train_predictions.csv` | `7b897725bff1…99fe` |
| formal `ad_test_predictions.csv` | `6b015eeaed8a…098b` |
| `conditional_logit.npz` | `1714137d1664…02eb9` |
| `rca_raw_index/index_manifest.json` | `cd343dfad123…88c19` |

Policy flags recorded: `no_retraining=true`,
`test_used_for_threshold_selection=false`, `rca_retrained=false`,
`reconstruction_calibration_refit=false`, `alpha_sweep_performed=false`,
`writes_into_formal_run=false`.

---

## 3. Score replay

### 3.1 Row identity — **Confirmed**

`split`, `sample_index`, `service`, `prediction_available_time` and `node_label`
align exactly with the formal prediction artifacts for both splits.

| split | rows | unmatched | time mismatches | label mismatches | identity |
|---|---:|---:|---:|---:|---|
| Train | 609,750 | 0 | 0 | 0 | exact |
| Test | 261,270 | 0 | 0 | 0 | exact |

### 3.2 Score-level comparison

| split | max_abs_diff | mean_abs_diff | cells exactly equal | tolerance |
|---|---:|---:|---:|---:|
| Train | `3.7849e-06` | `1.949e-08` | 0 | `1e-6` |
| Test | `1.6093e-06` | `1.082e-08` | 0 | `1e-6` |

`score_level_passed = false`. The pre-registered condition is **FAIL**:

```text
strict numerical replay gate (max_abs_diff <= 1e-6) = FAIL
  Train max_abs_diff = 3.7849e-06   (3.8x the target)
  Test  max_abs_diff = 1.6093e-06   (1.6x the target)
  mean_abs_diff ~ 1e-08, exactly-equal cells = 0
```

This condition is not met and is not reinterpreted as met anywhere in this
report. It is kept as a recorded failure. What the audit *can* establish
independently is the decision-level reproduction of Section 3.4.

### 3.3 Root-cause check (required, not skipped)

The algebraic fusion identity was verified independently:

```text
max |fused_score - (0.7*cls + 0.3*rec)| = 7.15e-08
```

so the P6 pipeline's own fusion is internally consistent to float32 rounding.
Isolation of the residual:

1. **Calibration** — `reconstruction_calibration.json` was loaded verbatim and
   `MY._reconstruction_energy_to_prob` was reused unchanged → not the source.
2. **Softmax** — the classification probability comes from the same
   `softmax(self.show(rec))` call → not the source.
3. **dtype / fusion order** — the same `torch.float32` tensors and the same
   `MY._fuse_predict_with_reconstruction` call order are used → not the source.
4. **Determinism control** — two independent P6 inference processes produced
   **bitwise identical** classification, reconstruction, and fused arrays.
5. **Thread-count control** — re-running the same frozen inference with
   `torch.set_num_threads ∈ {1,2,4,6,8,12,16,32,48}` on a 3,072-window subset
   changed the raw reconstruction energy by up to `2.9e-06` and the fused score
   by up to `7.2e-07`, while same-configuration reruns stayed bitwise identical.
   **No** thread configuration reproduced the formal artifact exactly; the
   `4.8e-7–7.2e-7` subset floor is irreducible from this side.

Conclusion (**Observed**): the residual is multi-threaded CPU float32 reduction
ordering inside the shared frozen backbone forward pass. It is a numerical
precision limit of re-running the same network in a different process, not a
semantic, protocol, or configuration mismatch.

### 3.4 Decision-level reproduction (not a substitute for the failed gate)

The strict numerical gate is FAIL (Section 3.2) and stays FAIL. Independently of
it, the fused track reproduces every quantity the downstream decisions were made
on *exactly*. The P6-B0 results are therefore retained as

```text
decision-level exact reproduction under small float32 numerical noise
```

which is a weaker and different statement than "the `1e-6` gate passed". The
evidence for the weaker statement:

| quantity | formal | P6 fused | abs diff |
|---|---:|---:|---:|
| Train threshold | 0.8693510293960571 | 0.8693511486053467 | `1.19e-07` |
| Test TP | 3703 | 3703 | 0 |
| Test FP | 65 | 65 | 0 |
| Test FN | 2084 | 2084 | 0 |
| predicted episodes | 3768 | 3768 | 0 |
| Test Precision | 0.9827 | 0.9827 | 0 |
| Test Recall | 0.6399 | 0.6399 | 0 |
| Test F1 | 0.7751 | 0.7751 | 0 |
| mean delay (s) | 24.106421550094517 | 24.106421550094517 | 0 |
| median delay (s) | 24.109 | 24.109 | 0 |
| P95 delay (s) | 39.5143 | 39.5143 | 0 |

Additional cross-check: the fused track's downstream Ada-RCA `AC@1 = 0.4657` and
Diagnosis `F1@1 = 0.3611` equal the formal P5-A values exactly.

Verdict recorded in `score_replay.json`:

```text
IDENTITY_EXACT_EVENT_EXACT_FLOAT_NOISE
identity_alignment_exact   = true
algebraic_identity_passed  = true
score_level_passed         = false     # strict 1e-6 gate: FAIL (1.6e-6 / 3.8e-6)
event_level_reproduction   = true      # decision-level exact
event_level_passed         = true
```

**Gate policy, stated honestly.** The pre-registered `1e-6` score gate failed and
is recorded as failed. The audit proceeded on the weaker, independently verified
decision-level reproduction, and every conclusion below is a decision-level
conclusion (event counts, event P/R/F1, delays, RCA AC@k, diagnosis F1). No
conclusion in this report depends on the raw scores agreeing to `1e-6`. The
residual is recorded rather than waived; it is not silently absorbed by
redefining the gate as passed.

---

## 4. Event detection results

### 4.1 Table 1 — Event trigger (Test)

| Trigger | Train threshold | Test P | Test R | Test F1 | mean delay | P95 delay |
|---|---:|---:|---:|---:|---:|---:|
| Classification-only | 0.994492 | 0.9912 | 0.6397 | 0.7776 | 24.113 s | 39.508 s |
| Current fused | 0.869351 | 0.9827 | 0.6399 | 0.7751 | 24.106 s | 39.514 s |
| Reconstruction-only | 0.661951 | 0.9132 | 0.3999 | 0.5562 | 24.188 s | 39.788 s |

Counts (Test):

| Trigger | GT events | predicted episodes | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| Classification | 5787 | 3735 | 3702 | 33 | 2085 |
| Fused | 5787 | 3768 | 3703 | 65 | 2084 |
| Reconstruction | 5787 | 2534 | 2314 | 220 | 3473 |

Train diagnostics (Train-only threshold, Train metrics):

| Trigger | Train P | Train R | Train F1 |
|---|---:|---:|---:|
| Classification | 0.9692 | 0.6503 | 0.7784 |
| Fused | 0.9676 | 0.6228 | 0.7578 |
| Reconstruction | 0.7326 | 0.2232 | 0.3422 |

**Observed.** At its Train-frozen threshold `0.661951`, reconstruction-only keeps
high precision (`0.9132`) but loses `37.5%` of the fused recall. The threshold is
*not* lowered to recover recall — Test was never used for threshold selection.
The classification-only and fused triggers are nearly interchangeable at event
level (`F1 0.7776` vs `0.7751`); classification actually produces *fewer* false
alarms (`33` vs `65`) while matching one fewer GT event (`3702` vs `3703`).

### 4.2 Per-fault-type stratification (Test)

| fault_type | n | cls TP/FN/R | fused TP/FN/R | rec TP/FN/R | small-n |
|---|---:|---|---|---|---|
| login_failure | 5546 | 3608/1938/0.6506 | 3620/1926/0.6527 | 2262/3284/0.4079 | no |
| memory_anomalies | 204 | 82/122/0.4020 | 71/133/0.3480 | 40/164/0.1961 | no |
| file_moving | 16 | 6/10/0.3750 | 6/10/0.3750 | 6/10/0.3750 | **yes** |
| cpu_anomalies | 12 | 1/11/0.0833 | 2/10/0.1667 | 2/10/0.1667 | **yes** |
| access_permission_denied | 5 | 5/0/1.0000 | 4/1/0.8000 | 4/1/0.8000 | **yes** |
| normal_memory_freed | 4 | 0/4/0.0000 | 0/4/0.0000 | 0/4/0.0000 | **yes** |

The fault taxonomy is dominated by `login_failure` (`5546/5787 = 95.8%`); four of
six groups are `small_n` and must not be used for generalization claims.

### 4.3 Per-root-service stratification (Test)

| root service | n | cls R | fused R | rec R | small-n |
|---|---:|---:|---:|---:|---|
| mobservice1 | 2799 | 0.6370 | 0.6406 | 0.3044 | no |
| mobservice2 | 2787 | 0.6606 | 0.6616 | 0.5102 | no |
| dbservice1 | 33 | 0.3333 | 0.2121 | 0.1212 | no |
| dbservice2 | 32 | 0.3750 | 0.3438 | 0.2500 | no |
| webservice1 | 30 | 0.2667 | 0.2667 | 0.1333 | no |
| webservice2 | 30 | 0.5333 | 0.5333 | 0.4000 | no |
| logservice2 | 20 | 0.1500 | 0.2000 | 0.1000 | **yes** |
| redisservice1 | 20 | 0.4000 | 0.3500 | 0.1500 | **yes** |
| logservice1 | 18 | 0.4444 | 0.3889 | 0.2222 | **yes** |
| redisservice2 | 18 | 0.6667 | 0.3333 | 0.1667 | **yes** |

`mobservice1/2` account for `5586/5787 = 96.5%` of Test GT events. All other
services are `n ≤ 33`; results there are diagnostic only.

---

## 5. Label coupling evidence

### 5.1 AD label ↔ RCA root label identity — **Confirmed**

Both labels originate from the same frozen registry `service` column, so the
identity is structural; this audit quantifies it and measures bin-level overlap
ambiguity. Population: all 16,131 complete GT injections assigned to Train/Test
(events crossing the chronological boundary are purged by the frozen rule).

| statistic | value | meaning |
|---|---:|---|
| total GT events | 16,131 | complete injections assigned to Train/Test |
| `labelled_service_positive_on_all_bins_ratio` | **1.0000** (16,131 / 16,131) | For every GT event, the AD node label marks the RCA-labelled root service as positive on **every** 30 s bin the event overlaps. |
| `rasterization_mismatch_events` | **0** | No GT event has a labelled service that is not positive on an overlapped AD bin. |
| `exact_identity_ratio` | 0.7045 (11,364 / 16,131) | Stricter set equality: "the AD positive set at every overlapped bin equals exactly `{this event's service}`". **This is an overlap statistic, not a label-agreement rate.** |
| AD bins covered | 30,870 | |
| bins covered by exactly one event | 24,954 | |
| bins covered by multiple events | 5,916 | |
| bins with multiple distinct root services positive | 4,892 | |
| (service, bin) pairs covered by multiple events | 1,331 | |

**Confirmed — the three quantities above must not be confused.**

**There is no AD/RCA label disagreement in this dataset.** The fall from `1.0000`
to `0.7045` is produced entirely by the strict set-equality test: whenever a bin
is shared with another event, the AD positive set at that bin contains more than
one service, so `positive set == {event service}` is false even though the
event's own labelled service is correctly positive. `0.7045` is therefore a
**same-bin multi-event overlap statistic, not a label-agreement rate** and must
not be quoted as "only 70.45% of AD/RCA labels are consistent".

The overlap it measures is real and large: `29.6%` of GT events share at least
one 30 s bin with another event, `4,892` bins carry more than one root service,
and `1,331` `(service, bin)` pairs are covered by multiple events. The AD node
label is not an independent supervision signal — it is the rasterized
injected-service identity — so the two tasks are entangled by construction as
well as by representation.

### 5.2 Anomaly-score localization diagnostic

This measures how much root-service identity the **first-stage anomaly score
itself** contains. It is a score diagnostic, **not** RCA performance. Ties are
broken by descending score then canonical registry order. Each track is measured
at its own matched Test anchors (pre-RCA-purge).

| evaluated track | size | score source | AC@1 | AC@3 | AC@5 | MRR | median root margin |
|---|---:|---|---:|---:|---:|---:|---:|
| classification | 3702 | classification | **0.9398** | 0.9954 | 0.9973 | 0.9680 | 0.9983 |
| classification | 3702 | fused | 0.9392 | 0.9822 | 0.9938 | 0.9627 | 0.7357 |
| classification | 3702 | reconstruction | 0.7696 | 0.9506 | 0.9884 | 0.8656 | 0.1242 |
| fused | 3703 | classification | 0.9384 | 0.9951 | 0.9973 | 0.9673 | 0.9983 |
| fused | 3703 | fused | 0.9384 | 0.9814 | 0.9935 | 0.9621 | 0.7358 |
| fused | 3703 | reconstruction | 0.7726 | 0.9522 | 0.9897 | 0.8678 | 0.1245 |
| reconstruction | 2314 | classification | 0.9304 | 0.9965 | 0.9987 | 0.9635 | 0.9979 |
| reconstruction | 2314 | fused | 0.9283 | 0.9831 | 0.9931 | 0.9570 | 0.7537 |
| reconstruction | 2314 | reconstruction | **0.9084** | 0.9745 | 0.9888 | 0.9433 | 0.1821 |

The diagonal is the fair per-trigger comparison. Reconstruction-only localization
`AC@1 = 0.9084` is only `0.0300` below classification-only `0.9398`.

Because the three triggers select different anchor times, an anchor-controlled
comparison was also run on the common-case population (`n = 2214`, all three
scores evaluated at one fixed anchor set):

| fixed anchor set | classification score AC@1 | fused score AC@1 | reconstruction score AC@1 | gap |
|---|---:|---:|---:|---:|
| fused anchors (n=2214) | 0.9295 | 0.9313 | **0.9106** | 0.0189 |
| reconstruction anchors (n=2214) | 0.9363 | 0.9350 | **0.9160** | 0.0203 |

**Confirmed.** With identical cases and identical anchors, the reconstruction
score still localizes the labelled root service at `AC@1 ≈ 0.91`, within ~2
points of the classification score. The reconstruction representation of the
shared backbone strongly encodes the root identity. The mechanism is visible in
the training objective rather than only in the shared representation: the
reconstruction loss is label-conditioned (`src/model.py:164-176`, quoted in
Section 2) and is explicitly trained to drive the reconstruction energy *up* on
nodes whose node anomaly label is `abnormal`. A high reconstruction-score
localisation `AC@1` is therefore the intended behaviour of the original
objective, not a coincidental leakage of the classification head.

### 5.3 Root margin

`margin = score(root_service) - max(score(non_root_services))`, per matched Test
case, stratified by fault type and root service.

| evaluation track | score source | n | mean | median | q25 | q75 | fraction > 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| classification | classification | 3702 | 0.8675 | 0.9983 | 0.9935 | 0.9994 | 0.9392 |
| classification | fused | 3702 | 0.6384 | 0.7357 | 0.6922 | 0.7583 | 0.9392 |
| classification | reconstruction | 3702 | 0.0934 | 0.1242 | 0.0177 | 0.1977 | 0.7696 |
| fused | classification | 3703 | 0.8630 | 0.9983 | 0.9933 | 0.9994 | 0.9379 |
| fused | fused | 3703 | 0.6357 | 0.7358 | 0.6925 | 0.7584 | 0.9384 |
| fused | reconstruction | 3703 | 0.0944 | 0.1245 | 0.0211 | 0.1982 | 0.7726 |
| reconstruction | classification | 2314 | 0.8482 | 0.9979 | 0.9915 | 0.9991 | 0.9296 |
| reconstruction | fused | 2314 | 0.6430 | 0.7537 | 0.7349 | 0.7662 | 0.9283 |
| reconstruction | reconstruction | 2314 | 0.1531 | 0.1821 | 0.1244 | 0.2244 | 0.9084 |

**Observed.** This is the sharpest difference between the branches. Classification
separates the root by a near-saturated margin (median `0.9983`) because the
classification head is trained directly on the injected-service node label.
Reconstruction has a median margin of only `0.124–0.182` on the classification
and fused anchor populations, rising to `0.182` for the reconstruction-selected
population. The reconstruction score still *orders* the root first in `76.96%`
(classification anchors) to `90.84%` (its own anchors) of cases, but with a much
weaker decision margin.

`reconstruction margin > 0` stratified by fault on the classification anchor
population: `login_failure 0.7777` (n=3608), `memory_anomalies 0.4146` (n=82),
`file_moving 0.8333` (n=6), `access_permission_denied 0.8000` (n=5),
`cpu_anomalies 0.0000` (n=1). Non-`login_failure` strata are small-n diagnostics.

---

## 6. RCA downstream results

Forward pass only: GAIA raw index → 68D W300-B15 Z2 features at the matched
`t_hat` → frozen conditional logit → rankings. The model was loaded, never
re-fit. One materialization covered the union of all three tracks' anchors
(3,856 unique anchors).

### 6.1 Table 3 — RCA native population

Matched Test cases after the frozen W300 chronological boundary purge, each
trigger using its own matched anchors.

| Trigger | matched n | AC@1 | AC@3 | AC@5 | MRR | root-macro AC@1 | fault-macro AC@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classification | 3701 | 0.4647 | 0.9203 | 0.9830 | 0.6981 | 0.1681 | 0.3703 |
| Fused | 3702 | 0.4657 | 0.9214 | 0.9830 | 0.6989 | 0.1757 | 0.4946 |
| Reconstruction | 2314 | 0.4715 | 0.9296 | 0.9853 | 0.7046 | 0.1991 | 0.4685 |

**Observed.** Population shift is real but does not favour any trigger: the
reconstruction trigger's matched population is `37.5%` smaller yet its matched
case `AC@1` is marginally *higher*. Root-macro and fault-macro values remain far
below the weighted overall because `mobservice1/2` and `login_failure` dominate;
they are reported so that the overall is not mistaken for a general result.

### 6.2 Table 4 — RCA common cases

`common_case_ids = matched_classification ∩ matched_fused ∩ matched_reconstruction`
on the RCA-eligible populations. `common n = 2214` (pre-RCA-purge intersection is
also 2214; no case is lost to the boundary purge intersection).

| Trigger | common n | AC@1 | AC@3 | AC@5 | MRR |
|---|---:|---:|---:|---:|---:|
| Classification | 2214 | 0.4734 | 0.9300 | 0.9860 | 0.7063 |
| Fused | 2214 | 0.4729 | 0.9304 | 0.9860 | 0.7060 |
| Reconstruction | 2214 | 0.4697 | 0.9304 | 0.9851 | 0.7042 |

**Confirmed.** On identical cases the three triggers are indistinguishable for
Ada-RCA (`AC@1` spread `0.0037`). The trigger changes *which* events are
available to the second stage, not the ranking quality on the events it does
detect. Any native-population difference attributable to trigger choice is a
population effect, not a ranking-quality effect.

Common-case anomaly-score localization (same cases, each score at its own
trigger's anchors):

| score source | AC@1 | AC@3 | AC@5 | MRR |
|---|---:|---:|---:|---:|
| classification | 0.9291–0.9363 | 0.9964 | 0.9986 | 0.9630–0.9665 |
| fused | 0.9309–0.9350 | 0.9837–0.9851 | 0.9937–0.9941 | 0.9585–0.9609 |
| reconstruction | 0.9083–0.9160 | 0.9711–0.9756 | 0.9878–0.9892 | 0.9424–0.9477 |

This is the common-case version of Table 2 and reproduces the same conclusion:
reconstruction localization stays within ~2–3 points of classification.

### 6.3 Complementarity — does Ada-RCA add information beyond the anomaly score?

2×2 contingency of anomaly-score Top-1 localization versus Ada-RCA Top-1, on
each track's native matched population.

| Trigger / score source | both correct | score only | RCA only | both wrong | P(RCA correct \| score wrong) | P(RCA wrong \| score correct) |
|---|---:|---:|---:|---:|---:|---:|
| classification / classification | 1568 | 1910 | 152 | 71 | 0.6816 | 0.5492 |
| fused / fused | 1581 | 1893 | 143 | 85 | 0.6272 | 0.5449 |
| reconstruction / reconstruction | 966 | 1136 | 125 | 87 | **0.5896** | 0.5404 |

**Observed.** Ada-RCA is not redundant with the first-stage anomaly score: when
the reconstruction score's Top-1 is wrong, Ada-RCA's Top-1 is still correct in
`59%` of cases. But the two are only weakly complementary — when the anomaly
score's Top-1 is correct, Ada-RCA is still wrong in `54%` of cases. Ada-RCA
provides independent but modest incremental information, and it does not repair
the first-stage score's errors.

---

## 7. E2E results

### 7.1 Table 5 — Full diagnosis (includes event FN, event FP, and ranking failures)

| Trigger | P@1 | R@1 | F1@1 | P@3 | R@3 | F1@3 | P@5 | R@5 | F1@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Classification | 0.4606 | 0.2975 | **0.3615** | 0.9122 | 0.5892 | **0.7159** | 0.9743 | 0.6293 | **0.7647** |
| Fused | 0.4577 | 0.2982 | **0.3611** | 0.9055 | 0.5900 | **0.7145** | 0.9660 | 0.6295 | **0.7623** |
| Reconstruction | 0.4305 | 0.1887 | **0.2624** | 0.8489 | 0.3720 | **0.5173** | 0.8998 | 0.3943 | **0.5483** |

Failure decomposition (W300-purged Test populations):

| Trigger | matched | FP predictions | FN GT events | predicted episodes | GT events | Top-1 ranking failures | Top-3 ranking failures | Top-5 ranking failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classification | 3701 | 33 | 2080 | 3734 | 5781 | 1981 | 295 | 63 |
| Fused | 3702 | 65 | 2079 | 3767 | 5781 | 1978 | 291 | 63 |
| Reconstruction | 2314 | 220 | 3468 | 2534 | 5782 | 1223 | 163 | 34 |

**Confirmed.** Diagnosis `F1@1` is `0.3615 / 0.3611 / 0.2624`. Reconstruction's
ranking-failure counts are *lower in absolute terms* only because it matches
fewer events; its `F1@1` loss is dominated by `3,468` missed GT events (event
recall), not by Ada-RCA ranking. Even at `@5` reconstruction reaches only
`0.5483` versus `0.7623` for fused, i.e. the event-recall deficit propagates to
every diagnosis level and cannot be hidden by matched-case `AC@k`.

---

## 8. Decision

### 8.1 Pre-registered gates

Operationalized in advance, never tuned on Test:

```text
f1_ratio        = reconstruction Test event F1 / fused Test event F1
recall_ratio    = reconstruction Test event recall / fused Test event recall
localization_gap = classification AC@1 - reconstruction AC@1 (diagonal, own anchors)

usable            : f1_ratio >= 0.60
severe            : f1_ratio <  0.35
coupling_reduced  : localization_gap >= 0.20
coupling_persists : localization_gap <  0.05
```

Observed:

```text
f1_ratio                  = 0.5562 / 0.7751 = 0.7176     -> usable
recall_ratio              = 0.3999 / 0.6399 = 0.6249
localization_gap          = 0.9398 - 0.9084 = 0.0300     -> coupling persists
reconstruction RCA AC@1   = 0.4715 (native, n=2314)
fused RCA AC@1            = 0.4657 (native, n=3702)
```

### 8.2 Outcome

```text
Outcome C
```

**Rationale.** The reconstruction-only event trigger is not severely degraded
(`f1_ratio = 0.7176`, so Outcome B does not apply), but it does **not** reduce
the label coupling: its own anomaly-score localization `AC@1 = 0.9084` sits only
`0.0300` below the classification score's `0.9398`, and the anchor-controlled
common-case gap is `0.019–0.020`. The shared backbone's reconstruction branch
still strongly encodes the labelled fault-service identity. Score decomposition
is therefore insufficient to resolve the AD ↔ RCA task coupling.

This is consistent with §5.1: the two labels are the *same* registry identity,
rasterized in two places, so a representation that is good at reconstructing
fault telemetry will naturally rank the injected service highly.

### 8.3 What is genuinely mixed and must not be over-packaged

- Reconstruction-only detection is **partially usable** (`F1 0.5562`,
  `P 0.9132`) but loses `37.5%` of fused recall; it is not a drop-in trigger
  replacement at equal recall.
- Classification-only and fused are near-identical at event level (`F1 0.7776`
  vs `0.7751`), so the current `alpha = 0.7` fusion is not what is carrying
  event detection — the classification branch is.
- Reconstruction *does* reduce the raw margin dramatically (median `0.1821` vs
  `0.9983`), so the two branches are not identical; the coupling is weakened in
  confidence but not broken in rank.
- Ada-RCA adds real but modest independent information
  (`P(RCA correct | score wrong) = 0.59` for the reconstruction trigger).

### 8.4 Next step implied by Outcome C

Per the pre-registered gate, Outcome C means score decomposition alone is not the
answer. The next stage should discuss a **system-level label and a
redesigned/retrained AD objective**, e.g. a frozen-backbone plus system-level
anomaly head, or a system-level (non-root-identity) supervision target. No Test
threshold, alpha, or matching change should be used to rescue this result.

### 8.5 Open issues

1. **Strict numerical replay gate = FAIL.** `max_abs_diff` is `1.6e-6` (Test) /
   `3.8e-6` (Train) against a `1e-6` target, so the pre-registered gate did not
   pass. Root-caused to CPU float32 reduction order; the residual is irreducible
   without pinning the formal run's process/thread state. Only the decision-level
   (event / RCA / diagnosis) reproduction is exact, and every conclusion in this
   report is a decision-level conclusion.
2. **No per-case score replay.** Only the aggregate replay gate is recorded; a
   per-row `max_abs_diff` distribution is not persisted in the versioned record
   (the row-level decomposition CSVs stay local). **Open question** whether a
   different thread configuration would close the gap further.
3. **Label identity `exact_identity_ratio = 0.7045` is an overlap statistic, not
   a label-agreement rate.** The label-agreement quantities are
   `labelled_service_positive_on_all_bins_ratio = 1.0000` and
   `rasterization_mismatch = 0`. `4,892` bins carry multiple root services;
   `29.6%` of GT events share a bin with another event.
4. **Population skew is severe.** `login_failure` is `95.8%` of Test GT events
   and `mobservice1/2` is `96.5%`. All non-dominant strata are small-n and were
   marked as such.
5. **Anchors, not only scores, differ across triggers.** The anchor-controlled
   common-case comparison mitigates this, but a fully anchor-matched comparison
   is not available for cases only detected by one trigger.
6. **Reconstruction trigger threshold is Train-frozen.** It was not lowered to
   improve Test recall and must not be.

---

## 9. Artifacts

```text
experiments/p6/score_decomposition/
├── manifest.json                   provenance, gates, policy flags
├── score_replay.json               score-level + event-level replay gate
├── score_range.json                finite/range legality per track
├── prediction_artifacts.json       hashes of the local row-level CSVs
├── label_identity_audit.json       §5.1
├── comparison.json                 Tables 1–5, common-case, complementarity
├── common_cases.csv                per-case common-case rows (3 triggers)
├── {classification,fused,reconstruction}/
│   ├── ad_train_predictions.csv    local (not versioned)
│   ├── ad_test_predictions.csv     local (not versioned)
│   ├── threshold.json              Train-only threshold + provenance
│   ├── event_metrics.json          Train/Test + stratified event metrics
│   ├── event_matches.csv           local (not versioned)
│   ├── localization_metrics.json   §5.2 + §5.3
│   ├── rca_metrics.json            §6.1
│   ├── diagnosis_metrics.json      §7
│   └── rca_detected_predictions.csv  local (not versioned)
└── rca/
    ├── anchor_registry.csv         local
    ├── anchor_features/            local (68D feature bundle)
    ├── rca_feature_manifest.json
    └── rca_feature_health.json
```

Reproduce:

```bash
python scripts/p6/run_score_decomposition.py infer \
    --output-dir experiments/p6/score_decomposition
python scripts/p6/run_score_decomposition.py evaluate \
    --output-dir experiments/p6/score_decomposition \
    --threshold-workers 8 --feature-workers 16 --reuse-event-thresholds
```
