# P6-C0 Metric temporal drift detector: Validation NO-GO

**Status: `COMPLETE_DEVELOPMENT_ONLY`, `NO_GO_DEVELOPMENT`. No new actual Test
model inference, threshold selection, or two-stage evaluation was run.**

The single-candidate source and plan were committed as `3891fdbdbd2ed7648fd4b45f30f3ca802af09338`
before training. The isolated worktree is `experiment/p6-c0-metric-drift`. The
run is `experiments/p6/c0_metric_drift/c0-metric-drift-v1-seed42/`; its
development manifest binds input hashes, source commit, checkpoint, and eight
result files. All eight recorded file hashes were rechecked after completion.
The V100 run completed ten epochs and early-stopped at patience 8, selecting
epoch index 1 under the fixed Validation event-F1 rule. There was no rerun,
candidate search, or post-result threshold change.

## Intended test

The only model change was a zero-initialized scalar residual on the C0 logit:
`alpha * z`, where `z` is the all-service, all-48-Metric mean absolute change
of the last two 30 s bins from the median of the first eight bins, then scaled
by Detector-Fit median/IQR. The Fit statistic was computed from all 43,550 Fit
windows only: median `0.01759361382573843`, IQR `0.006728033069521189`.
Selected alpha was `0.18561436235904694`; selected Validation threshold was
`0.9964967370033264`. A zero-residual initial model exactly replayed C0
initial logits in a fixed test; 32 focused/existing tests and synthetic smoke
passed before the real run.

## Validation event results

The same 2,901 complete Validation GT cases were joined by exact `case_id` and
checked for matching fault, service, start, and end metadata. The historical C0
matching replayed 2,124/13/777 TP/FP/FN. The candidate's matching integer
counts agreed with its selected-checkpoint record.

| Detector | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| C0 historical | 2,124 | 13 | 777 | 0.9939 | 0.7322 | 0.8432 |
| Metric drift | 2,071 | 101 | 830 | 0.9535 | 0.7139 | 0.8165 |

The candidate gained 35 and lost 88 C0-matched cases, a net loss of 53.
Every one of seven UTC onset days had a nonpositive net change. Login failure
fell from 2,108/2,776 to 2,051/2,776; multi-onset events fell from 156/380
to 145/380. The longer-event strata gained modestly, but this did not repair
the isolated target cases:

| Validation cohort | n | C0 TP | Candidate TP |
|---|---:|---:|---:|
| Clean supported `memory_anomalies` | 30 | 0 | 0 |
| All `memory_anomalies` | 110 | 15 | 16 |
| Events longer than 300 s | 125 | 16 | 20 |

All five gates from the plan were applied without adjustment. Only the
trivial no-clean-memory-loss gate passed; clean-memory gain ≥5, total TP gain
≥10, F1 ≥0.8432, and precision ≥0.98 all failed. Therefore this candidate
must not proceed to Test or full E2E; the C0 detector and prior full-diagnosis
numbers remain the current references.

## Interpretation and limits

The Fit/Validation raw Metric-change AUC was only descriptive, and this trial
does not support the hypothesis that **one global Metric drift scalar** repairs
the isolated memory-onset gap. It cannot establish that the 48 Metric channels
are uninformative; the global mean may dilute a specific channel or service.
The candidate used GPU, while the historical C0 training used CPU. That device
difference prevents clean causal attribution of the overall F1 decline to the
residual branch alone. It does **not** change the prespecified absolute NO-GO:
zero clean-memory gain, lower overall TP/F1, and higher FP versus the current
operational C0 reference. A same-device control would be required before any
claim about the residual's isolated effect.

The Test was previously seen in other P6 work, so any later Test outcome would
be exploratory even after a new Validation GO. The large checkpoint remains in
the local run directory; structured predictions, matching, selection, hashes,
and the exact paired ledger are archived with this branch. Full checkpoint
replay requires that local checkpoint or retraining from the frozen source.
