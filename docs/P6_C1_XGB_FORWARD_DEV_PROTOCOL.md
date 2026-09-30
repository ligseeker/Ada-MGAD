# P6-C1 RCA Train-only forward-fold robustness check

**Evidence grade: Train-only temporal development, post hoc to reused Test.**
The prior 68D+XGB Test comparison motivated this check, so neither a positive
nor a negative result can be called an independent confirmation. Do not open
Test features, Test rankings, Test matching or Test GT in this run.

## Question and fixed design

Does the fixed 68D+XGBRanker procedure improve detected-anchor AC@1 over
the existing detected-anchor Conditional Logit on later **Train** cases that
were absent from each RCA fit? Use exactly the sealed C1-v2 common Train
cohort of 3,225 cases, unchanged detected-anchor 68D tensors, root labels,
ten-service order and chronological fold IDs.

| Forward comparison | RCA fit cases | Evaluation cases |
|---|---:|---:|
| F1→F2 | fold 1 (935) | fold 2 (1,168) |
| F1+F2→F3 | folds 1–2 (2,103) | fold 3 (1,122) |

Do not evaluate fold 1 using later folds. For each comparison, fit both
scorers on identical earlier case IDs and roots. Conditional Logit retains
its frozen lambda/optimizer and a StandardScaler fitted solely on earlier
GT-anchor candidate rows, then applied to earlier detected-anchor Train and
later detected-anchor evaluation tensors. XGB uses the frozen 68D pairwise
ranker parameters and **no scaler**. Candidate order, event grouping,
one-positive-per-case relevance, seed and case weight stay fixed. No
hyperparameter, threshold, offset or representation search.

## Label boundary and outputs

`fit-predict` first verifies sealed Train input hashes and chronology; it
indexes only earlier root indices to fit, scores later detected-anchor features
without decoding or using their labels, saves complete ten-service rankings and seals
the ordered case universe/model hashes. The source label array is still read as
raw bytes for its integrity hash, so this is a code-enforced use boundary, not
separate physical storage. Only a separate `evaluate` action
then decodes later roots/faults and reports per-fold and pooled AC@1/3/5,
MRR, paired Top-1 transitions and group counts. Missing rankings are failures
and stay in the denominator. The unique output root is
`experiments/p6/c1_xgb_forward_dev/c1-xgb-forward-v1/` in the isolated XGB
worktree; no C1-v2 or earlier XGB artifact is overwritten.

## Decision rule

Report both folds regardless of sign. A scorer advantage worth considering
for future independent evaluation should have positive paired AC@1 in **both**
forward comparisons and no evidence that the gain is solely a few unstable
minority cases. If one fold regresses, classify the reused-Test XGB gain as
unstable and stop RCA model development. Either outcome is descriptive:
the shared original-Train preprocessing remains label-free transductive
relative to the internal folds, and the Test has already been seen.
