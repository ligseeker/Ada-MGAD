# P6-C1-v2 supervision-OOS detector-aligned RCA and P6-C2 full E2E results

## Material Passport

- Protocol: `P6-C1-SUPERVISION-OOS-v1` ([protocol](P6_C1_V2_SUPERVISION_OOS_PROTOCOL.md),
  [lock](../configs/e2e/gaia_p6_c1_v2_supervision_oos.json), SHA-256
  `4e4529db1049fa5c04e658193b6b4e91eef0d9ad9e5d0efcd785dc776c475874`).
- Run: `experiments/p6/c1_supervision_oos/c1-supervision-oos-v1-seed42/`
  (`run_lock.json` git head `ad133dd9e9248bf72ce7e3f3a464ec614049569d`).
- Outcome: **`ANCHOR_ALIGNMENT_SUPPORTED`** for the primary C1 estimand.
- Evidence grade: supervision-OOS and detector-parameter-OOS. The shared
  preprocessing is the frozen original-Train label-free fit; this is not a
  strict prefix-preprocessing-OOS or a full-pipeline forward-OOS result, and the
  Test population has been examined in earlier P5/P6 work.

## 1. What was run

| Stage | Status | Evidence |
|---|---|---|
| G0 protocol freeze | COMPLETE | config + protocol committed before any fold result |
| G1 static feasibility | PASS | `g1_static/`, upper bounds 1192/1529/1533 reproduced |
| G2 shared-array adapter | COMPLETE | `folds/fold_0*/completion_manifest.json` |
| G3 synthetic + bounded real smoke | PASS | `tests/test_p6_c1_v2_*.py`, `docs/P6_C1_V2_RAW_SMOKE_20260929.json` |
| G4 three fold detectors | COMPLETE | `folds/fold_0*/detector/` |
| G5 OOS matching + cohort floors | PASS | `train_cohort/` (935/1168/1122 vs 596/765/767) |
| G6 shared scaler + B/C arms | COMPLETE | `rca/` (3225 common cases) |
| G7 label-free Test ranking + lock | LOCKED | `predictions/` (4214 episodes, 4213 legal, 0 ranking failures) |
| G8 C1 evaluation | COMPLETE | `evaluation/c1_results.json` |
| G9 C2 full E2E | COMPLETE | `evaluation/c2_*.json`, `evaluation/c2_failure_ledger.csv` |

No STOP condition triggered. No frozen input, failed run or earlier experiment
directory was modified.

## 2. Fold detectors (frozen C0 architecture, per-fold random init)

| Fold | Fit windows | Selection windows | Generation windows | Selected epoch | Threshold | Selection P / R / F1 | Generation episodes |
|---|---:|---:|---:|---:|---:|---|---:|
| 1 | 8,702 | 8,702 | 8,702 | 9 | 0.9967268775 | 0.9829 / 0.7702 / 0.8636 | 953 |
| 2 | 17,414 | 8,702 | 8,702 | 2 | 0.9822322718 | 1.0000 / 0.7933 / 0.8847 | 1,172 |
| 3 | 26,126 | 8,702 | 8,702 | 2 | 0.9965513471 | 0.9991 / 0.7621 / 0.8646 | 1,124 |

Selection event counts: fold 1 TP 1089 / FP 19 / FN 325 of 1414 GT events;
fold 2 TP 948 / FP 0 / FN 247 of 1195; fold 3 TP 1169 / FP 1 / FN 365 of 1534.
Checkpoint and threshold were chosen on Selection only; Generation was scored
label-free and sealed before any GT join (`generation_labels_built: false`).

## 3. OOS matching and common Train cohort

| Fold | Generation GT | matched | false alarms | misses | context-illegal matched | common legal cases | floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,195 | 936 | 17 | 259 | 1 | 935 | 596 |
| 2 | 1,534 | 1,171 | 1 | 363 | 3 | 1,168 | 765 |
| 3 | 1,536 | 1,123 | 1 | 413 | 1 | 1,122 | 767 |

Two boundary GT rows (`gaia-v3-06443` crossing the fold 2 and fold 3 Generation
boundaries) are retained in the exclusion ledger. All three floors pass.
**Common Train cohort = 3,225 cases** (935 / 1,168 / 1,122; total matched 3,230
minus 5 context-illegal), with identical case IDs, roots, candidate order and
scaler for arms B and C.

RCA health: paired `[cases, 10, 68]` GT/DET tensors, all finite, no feature
failure; one StandardScaler fitted on GT-anchor candidate rows (mean/scale
recorded in `predictions/prediction_lock.json`); Conditional Logit
`lambda=1`, unit weights, B converged in 342 iterations (gradient inf-norm
`1.26e-10`), C in 353 (`6.84e-11`), identical train case indices.

## 4. C1 result (primary estimand)

Locked legal matched Test cohort: **n = 4,197** (4,214 predicted episodes:
4,198 matched, 16 false alarms, 1 illegal context; 4,213 legal rankings, 0
ranking failures; missing rankings would have stayed in the denominator).

| Arm | Anchor (Train → Test) | AC@1 | AC@3 | AC@5 | MRR |
|---|---|---:|---:|---:|---:|
| A (oracle diagnostic, B weights) | GT → GT | 0.9338 | 0.9967 | 0.9986 | 0.9652 |
| B (mismatch reference) | GT → detected | 0.4980 | 0.9714 | 0.9959 | 0.7358 |
| C (detector-aligned) | OOS detected → detected | 0.6945 | 0.9938 | 0.9962 | 0.8444 |

```text
Delta AC@1 = AC@1(C) - AC@1(B) = +0.19657
paired transitions: both correct 1,408 | B only 682 | C only 1,507 | both wrong 600
UTC onset-day cluster bootstrap (10,000 replicates, seed 42, 10 clusters):
95% descriptive percentile interval [0.17041, 0.22467]
```

Secondary: AC@3 `+0.02240`, AC@5 `+0.00024`, MRR `+0.10863`.

Prespecified macros (groups with `n >= 20` only; `n = 0` listed as undefined,
`1 <= n < 20` descriptive):

| Group | n | AC@1 B → C | delta |
|---|---:|---|---:|
| root macro (mobservice1, mobservice2) | 4,171 | 0.5010 → 0.6989 | +0.19794 |
| fault macro (login_failure, memory_anomalies) | 4,194 | 0.3337 → 0.4328 | +0.09906 |
| root mobservice1 | 2,066 | 0.4932 → 0.7072 | +0.21394 |
| root mobservice2 | 2,105 | 0.5088 → 0.6907 | +0.18195 |
| fault login_failure | 4,164 | 0.5007 → 0.6988 | +0.19813 |
| fault memory_anomalies | 30 | 0.1667 → 0.1667 | 0.00000 |

Small descriptive groups (never in the macro): dbservice1 (5), dbservice2 (6),
logservice2 (2), redisservice1 (4), redisservice2 (2), webservice1 (2),
webservice2 (2), cpu_anomalies (1), file_moving (1),
access_permission_denied (1); logservice1 and normal_memory_freed are `n = 0`.
Note that the fault macro drops at AC@3/AC@5 (`-0.12109` / `-0.08237`) because
the 30-case memory group regresses there while the dominant group improves.

**Outcome: `ANCHOR_ALIGNMENT_SUPPORTED`.** Replacing the RCA Train anchor with
the detector's own OOS detected timestamps improves detected-anchor root ranking
by +19.7 points of AC@1 on the same reused Test cohort, with the paired
transitions dominated by `C only correct` (1,507) over `B only correct` (682).
No performance gate was imposed; this is the observed sign, not a threshold
verdict. B is far below A (0.4980 vs 0.9338), so anchor-aligned training closes
about 44 % of the GT-to-detected anchor gap, not all of it.

## 5. C2 full E2E

### Layer 1 - Stage-1 event detection (quoted frozen P6-C0, not recomputed)

```text
GT 5,787 | TP 4,198 | FP 16 | FN 1,589
P = 0.996203 | R = 0.725419 | F1 = 0.839516
delay mean 23.774 s | median 23.760 s | P95 39.148 s
frozen verdict: BORDERLINE (gate passed; long-event and memory stratification failures)
```

### Layer 2 - matched legal RCA (A/B/C)

Identical to section 4 (`AC@1`: A 0.9338, B 0.4980, C 0.6945).

### Layer 3 - full diagnosis over all raw Test GT (denominator 5,787 GT events, 4,214 predicted episodes)

| Arm | P@1 | R@1 | F1@1 | P@3 | R@3 | F1@3 | P@5 | R@5 | F1@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 0.4960 | 0.3612 | 0.4180 | 0.9675 | 0.7045 | 0.8153 | 0.9919 | 0.7223 | 0.8359 |
| C | 0.6917 | 0.5037 | 0.5829 | 0.9898 | 0.7208 | 0.8341 | 0.9922 | 0.7225 | 0.8361 |

Raw numerators: B `TP@1 2090 / TP@3 4077 / TP@5 4180`; C `TP@1 2915 / TP@3 4171 /
TP@5 4181`; precision denominators 4,214 (predicted episodes), recall
denominators 5,787 (Stage-1 GT population) for both arms.

### Failure ledger (mutually exclusive; closes exactly)

| Category | B | C |
|---|---:|---:|
| SUCCESS_TOP1 | 2,090 | 2,915 |
| ROOT_OUTSIDE_TOP1 (rank 2-3) | 1,987 | 1,256 |
| ROOT_OUTSIDE_TOP3 (rank 4-5) | 103 | 10 |
| ROOT_OUTSIDE_TOP5 (rank > 5) | 17 | 16 |
| RCA_CONTEXT_INVALID | 1 | 1 |
| RCA_RANKING_MISSING | 0 | 0 |
| EVENT_MISSED | 1,589 | 1,589 |
| EVENT_FALSE_ALARM | 16 | 16 |

### Stratification (interpretation only, never post hoc selection)

| Stratum | GT n | matched | AC@1 B | AC@1 C |
|---|---:|---:|---:|---:|
| duration `le_15s` | 5,558 | 4,166 | 0.5005 | 0.6985 |
| duration `15_30s` / `30_60s` / `60_300s` | 0 | 0 | - | - |
| duration `gt_300s` | 229 | 32 | 0.1563 | 0.1563 |
| onset multiplicity 1 | 4,966 | 4,166 | 0.4819 | 0.7120 |
| onset multiplicity 2 | 782 | 782 | 0.6737 | 0.5015 |
| onset multiplicity 3 | 39 | 39 | 0.8889 | 0.2222 |

The middle duration strata are empty and kept as `n = 0` rows. Onset
multiplicity 2 and 3 favor B, a declared limitation rather than a tuned result.

### Latency definitions

```text
detection latency        t_det - GT onset                  mean 23.77 s / median 23.76 s / P95 39.15 s
RCA data-ready latency   (t_det + 300 s) - GT onset        mean 323.77 s
final diagnosis compute  wall-clock                        UNMEASURED
```

The system is therefore described as online event triggering plus delayed
root-cause diagnosis, never as real-time RCA.

## 6. Limitations (do not silently repair)

1. **Reused Test.** The frozen C0 Test has been examined in previous P5/P6 work;
   the bootstrap interval is a descriptive summary, not an independent
   confirmatory test.
2. **Supervision-OOS, not preprocessing-OOS.** The shared preprocessing was
   fitted once on the original 70 % Train period (label-free) and may have seen
   label-free Train telemetry later than a fold Generation interval.
3. **C0 remains BORDERLINE.** Stage-1 recall is 0.7254 and long/memory events are
   largely missed; C2 counts those misses in the full-diagnosis denominator.
4. **GAIA W300 adapter.** The E2E representation is W300-B15 / 40 bins; the
   canonical Ada-RCA form is W600-B15 / 80 bins and was not used.
5. **30 s Metric grid vs 15 s RCA bins.** The detector grid is 30 s, the RCA
   context bins are 15 s, so the anchor rounding is 30 s.
6. **Class and service imbalance.** login_failure 4,164/4,197 and
   mobservice1+mobservice2 4,171/4,197 of the matched cohort; the other groups
   are descriptive only.
7. **Multi-onset resolution.** On the 782 two-onset and 39 three-onset cases arm
   C is worse than B, so same-window multi-event resolution is unresolved.
8. **Comparability.** The P5 detected-anchor reference (AC@1 0.4657, diagnosis
   F1@1 0.3611) used a different Stage-1 detector and a 14,045-case Train
   cohort; the numbers here are not a controlled comparison against it.
9. **Boundary windows.** The detector window whose prediction time equals a
   segment end is scored in neither segment (frozen G3 convention), and GT
   events crossing a Generation boundary are excluded from matching.
10. **Single seed.** Everything runs with seed 42; no seed replication was
    performed, so no bitwise-reproducibility or robustness claim is made.

## 7. Artifact map

```text
experiments/p6/c1_supervision_oos/c1-supervision-oos-v1-seed42/
  run_lock.json                 git head, protocol, G1 and source hashes
  g1_static/                    input/fold manifests, static population, report
  folds/fold_0{1,2,3}/          shared-array slices, detector seal, episodes
  train_cohort/                 matching, cases, exclusions, paired features
  rca/                          shared scaler, arm_b, arm_c (+ metadata)
  predictions/                  lock, scope rankings, feature inputs
  evaluation/                   c1_results, oracle A, C2 layer 1/3 + ledger
```
