# P6-C0F Failure Mechanism Audit — Result

Run: `c0f-seed42-20260919T0700`  |  git commit: `b3a3ceb61bd22af594fdbdb3721380e9679f70f2`

Read-only audit of the frozen P6-C0 system trigger. No training, no threshold rescan,
no Test model inference, no RCA. The P6-C0 verdict stays **BORDERLINE**.

## 1. Completion and integrity

- `prepare`: COMPLETE
- `scores`: COMPLETE
- `analyze`: COMPLETE
- `capacity`: COMPLETE
- `oracle_status`: EXACT
- `input_integrity`: PASS
- `split_population`: PASS
- `validation_replay`: MATCH
- `ledger_invariants_hold`: True
- Validation replay: MATCH at the frozen threshold

## 2. Failure ledger (per split)

Detection funnel per split (identical populations, no event removed):

| split | complete GT | causal crossing in window | new episode start in window | matched (TP) |
|---|---:|---:|---:|---:|
| fit | 7443 | 7054 | 6369 | 5626 |
| validation | 2901 | 2722 | 2455 | 2124 |
| test | 5787 | 5390 | 4881 | 4198 |

The three losses are different objects: a missing crossing is a score/operating-point location, a missing new episode is an episode-construction location (the trigger was already running), and an unmatched candidate episode is a one-to-one matching location.

### fit

| failure location | n | share |
|---|---:|---:|
| MATCHED | 5626 | 0.7559 |
| NO_LEGAL_PREDICTION | 1 | 0.0001 |
| BELOW_THRESHOLD | 388 | 0.0521 |
| NO_NEW_EPISODE | 685 | 0.0920 |
| MATCHING_COMPETITION | 743 | 0.0998 |

| duration stratum | n | matched | recall |
|---|---:|---:|---:|
| le_15s | 7081 | 5585 | 0.7887 |
| 15_30s | 0 | 0 | n/a |
| 30_60s | 0 | 0 | n/a |
| 60_300s | 0 | 0 | n/a |
| gt_300s | 362 | 41 | 0.1133 |

Failure mix inside the two non-empty duration strata:

| stratum | n | MATCHED | NO_LEGAL_PREDICTION | BELOW_THRESHOLD | NO_NEW_EPISODE | MATCHING_COMPETITION |
|---|---:|---:|---:|---:|---:|---:|
| le_15s | 7081 | 5585 | 1 | 128 | 680 | 687 |
| gt_300s | 362 | 41 | 0 | 260 | 5 | 56 |

Failure mix by fault type (n >= 30 only; smaller groups stay in the JSON):

| fault type | n | MATCHED | BELOW_THRESHOLD | NO_NEW_EPISODE | MATCHING_COMPETITION | recall |
|---|---:|---:|---:|---:|---:|---:|
| login_failure | 7081 | 5585 | 128 | 680 | 687 | 0.7887 |
| memory_anomalies | 328 | 38 | 234 | 5 | 51 | 0.1159 |

Recall by onset-bin multiplicity:

| onsets in the bin | n | recall |
|---|---:|---:|
| 1 | 6625 | 0.7980 |
| 2 | 788 | 0.4213 |
| 3 | 30 | 0.2333 |

### validation

| failure location | n | share |
|---|---:|---:|
| MATCHED | 2124 | 0.7322 |
| NO_LEGAL_PREDICTION | 1 | 0.0003 |
| BELOW_THRESHOLD | 178 | 0.0614 |
| NO_NEW_EPISODE | 267 | 0.0920 |
| MATCHING_COMPETITION | 331 | 0.1141 |

| duration stratum | n | matched | recall |
|---|---:|---:|---:|
| le_15s | 2776 | 2108 | 0.7594 |
| 15_30s | 0 | 0 | n/a |
| 30_60s | 0 | 0 | n/a |
| 60_300s | 0 | 0 | n/a |
| gt_300s | 125 | 16 | 0.1280 |

Failure mix inside the two non-empty duration strata:

| stratum | n | MATCHED | NO_LEGAL_PREDICTION | BELOW_THRESHOLD | NO_NEW_EPISODE | MATCHING_COMPETITION |
|---|---:|---:|---:|---:|---:|---:|
| le_15s | 2776 | 2108 | 0 | 93 | 267 | 308 |
| gt_300s | 125 | 16 | 1 | 85 | 0 | 23 |

Failure mix by fault type (n >= 30 only; smaller groups stay in the JSON):

| fault type | n | MATCHED | BELOW_THRESHOLD | NO_NEW_EPISODE | MATCHING_COMPETITION | recall |
|---|---:|---:|---:|---:|---:|---:|
| login_failure | 2776 | 2108 | 93 | 267 | 308 | 0.7594 |
| memory_anomalies | 110 | 15 | 74 | 0 | 20 | 0.1364 |

Recall by onset-bin multiplicity:

| onsets in the bin | n | recall |
|---|---:|---:|
| 1 | 2521 | 0.7806 |
| 2 | 368 | 0.4158 |
| 3 | 12 | 0.2500 |

### test

| failure location | n | share |
|---|---:|---:|
| MATCHED | 4198 | 0.7254 |
| NO_LEGAL_PREDICTION | 3 | 0.0005 |
| BELOW_THRESHOLD | 394 | 0.0681 |
| NO_NEW_EPISODE | 509 | 0.0880 |
| MATCHING_COMPETITION | 683 | 0.1180 |

| duration stratum | n | matched | recall |
|---|---:|---:|---:|
| le_15s | 5558 | 4166 | 0.7496 |
| 15_30s | 0 | 0 | n/a |
| 30_60s | 0 | 0 | n/a |
| 60_300s | 0 | 0 | n/a |
| gt_300s | 229 | 32 | 0.1397 |

Failure mix inside the two non-empty duration strata:

| stratum | n | MATCHED | NO_LEGAL_PREDICTION | BELOW_THRESHOLD | NO_NEW_EPISODE | MATCHING_COMPETITION |
|---|---:|---:|---:|---:|---:|---:|
| le_15s | 5558 | 4166 | 3 | 246 | 504 | 639 |
| gt_300s | 229 | 32 | 0 | 148 | 5 | 44 |

Failure mix by fault type (n >= 30 only; smaller groups stay in the JSON):

| fault type | n | MATCHED | BELOW_THRESHOLD | NO_NEW_EPISODE | MATCHING_COMPETITION | recall |
|---|---:|---:|---:|---:|---:|---:|
| login_failure | 5546 | 4165 | 236 | 504 | 638 | 0.7510 |
| memory_anomalies | 204 | 30 | 130 | 5 | 39 | 0.1471 |

Recall by onset-bin multiplicity:

| onsets in the bin | n | recall |
|---|---:|---:|
| 1 | 4966 | 0.7769 |
| 2 | 782 | 0.4233 |
| 3 | 39 | 0.2308 |

## 3. Structural bounds

| split | observed TP | U_episode | U_grid | observed recall | episode bound recall |
|---|---:|---:|---:|---:|---:|
| fit | 5626 | 6841 | 7427 | 0.7559 | 0.9191 |
| validation | 2124 | 2607 | 2890 | 0.7322 | 0.8987 |
| test | 4198 | 5173 | 5766 | 0.7254 | 0.8939 |

Oracles: fit=EXACT, validation=EXACT, test=EXACT

## 4. Response bands

| split | <=60 s | 60-120 s | 120-300 s | >300 s | never | censored |
|---|---:|---:|---:|---:|---:|---:|
| fit | 7054 | 93 | 184 | 50 | 62 | 0 |
| validation | 2722 | 61 | 85 | 15 | 18 | 0 |
| test | 5390 | 128 | 195 | 17 | 57 | 0 |

## 5. Evidence ledger

| claim | grade | statement | limitations |
|---|---|---|---|
| C0F-INTEGRITY | CONFIRMED | frozen input SHAs and the C0 artifact snapshot match the plan binding | a mismatch would have stopped the run; the snapshot is the plan's documented value, not a re-derivation |
| C0F-SPLIT | CONFIRMED | the three split populations reproduce the plan table (windows, complete GT, >300 s GT) | counts are verification targets, not filters |
| C0F-REPLAY | CONFIRMED | the frozen checkpoint reproduces the recorded Validation metrics at the frozen threshold | row-level Validation scores were never persisted by C0, so only aggregate replay is verifiable |
| C0F-LEDGER | CONFIRMED | every complete GT event has exactly one failure location code and the ledger closes | codes are occurrence locations, not causal explanations |
| C0F-ORACLE | CONFIRMED | the label reference, the relaxed grid bound and the complete-protocol episode bound hold the ordering observed <= U_episode <= U_grid <= 1 | U_grid is a relaxation and usually not realisable; U_episode is exact only with its witness replay |
| C0F-MIX-FIT | CONFIRMED | failure-location mix for fit: {"BELOW_THRESHOLD": 0.05212951766760715, "MATCHED": 0.7558780061803037, "MATCHING_COMPETITION": 0.09982533924492812, "NO_LEGAL_PREDICTION": 0.00013435442697836894, "NO_NEW_EPISODE": 0.09203278248018272} | the mix describes where detection is lost, not why |
| C0F-NO-FIRST-POSITIVE-FIT | OBSERVED | 62 events reach no threshold crossing at all inside [onset, onset+60 s] | no crossing is an observation about the frozen operating point, not a diagnosis of the score |
| C0F-MIX-VALIDATION | CONFIRMED | failure-location mix for validation: {"BELOW_THRESHOLD": 0.06135815236125474, "MATCHED": 0.7321613236814891, "MATCHING_COMPETITION": 0.11409858669424336, "NO_LEGAL_PREDICTION": 0.0003447087211306446, "NO_NEW_EPISODE": 0.09203722854188211} | the mix describes where detection is lost, not why |
| C0F-NO-FIRST-POSITIVE-VALIDATION | OBSERVED | 18 events reach no threshold crossing at all inside [onset, onset+60 s] | no crossing is an observation about the frozen operating point, not a diagnosis of the score |
| C0F-MIX-TEST | CONFIRMED | failure-location mix for test: {"BELOW_THRESHOLD": 0.06808363573526871, "MATCHED": 0.7254190426818732, "MATCHING_COMPETITION": 0.11802315534819423, "NO_LEGAL_PREDICTION": 0.0005184033177812338, "NO_NEW_EPISODE": 0.08795576291688267} | the mix describes where detection is lost, not why |
| C0F-NO-FIRST-POSITIVE-TEST | OBSERVED | 56 events reach no threshold crossing at all inside [onset, onset+60 s] | no crossing is an observation about the frozen operating point, not a diagnosis of the score |

## 6. Unresolved

- **C0F-U1** concurrent onsets can carry a score crossing that is not attributable to the event under audit; the marker columns expose it but most long events live in busy regions
- **C0F-U2** the frozen registry has no injections between 15 s and 300 s and no cpu_anomalies in Fit/Validation, so duration and fault-type effects cannot be separated
- **C0F-U3** whether the IGNORE band, the recent-onset objective, the representation or telemetry observability causes the sustained-state behaviour is not identifiable from these artifacts; no causal mechanism is claimed

## 7. Next scope

Loss dominance rule: `structural = NO_NEW_EPISODE + MATCHING_COMPETITION; score-side = BELOW_THRESHOLD + NO_LEGAL_PREDICTION; structural_dominant iff structural >= 1.5 x score-side on every split, score_side_dominant iff the reverse holds on every split`

| split | structural loss share | score-side loss share | ratio |
|---|---:|---:|---:|
| fit | 0.1919 | 0.0523 | 3.67 |
| validation | 0.2061 | 0.0617 | 3.34 |
| test | 0.2060 | 0.0686 | 3.00 |

Verdict: **structural_dominant**

the loss is dominated by the episode/matching structure on every split; a separately frozen P6-C0R2 candidate may study episode/selection structure, keeping P6-C0 BORDERLINE and without changing the frozen checkpoint or threshold in this round

