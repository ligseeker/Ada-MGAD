# P6-C0F audit correction protocol

Status: correction specification for the 2026-09-19 C0F audit. This protocol was
written after inspecting that audit and the reused Test data. It is an
explanatory correction, not a preregistered detector experiment or independent
confirmation. The P6-C0 Stage-1 verdict remains `BORDERLINE`.

## Frozen source and scope

- Historical run: `experiments/p6/c0f_failure_audit/c0f-seed42-20260919T0700/`.
- Historical `completion_manifest.json` SHA-256:
  `97e1fd3e34da06e7b64808b9f38372628c7007b9a2533c46582b50e5b7ace53a`.
- The historical Fit and Validation score files have SHA-256
  `140c58508263cd5c5fca280cac284c6d3e52798275eb40b36c8921f9a797687b`
  and `5b5e592636753ada3615ff4c373ef739660dcc5196908a78bceb9eb85892ce2f`.
  They must also match the historical completion manifest. The original run did
  not archive its executing source bytes; its Fit/Validation inference source
  remains `UNVERIFIED` even if these files are copied successfully.
- P6-C0 checkpoint, threshold, grid, GT and split bindings are those in
  [the C0F plan](P6_C0F_FAILURE_MECHANISM_AUDIT_PLAN.md), section 2. No new
  training, checkpoint choice, calibration, threshold search, preprocessing,
  Test model inference, RCA, or GT/episode/matching change is allowed.
- The driver must read the canonical data/artifact roots and config paths
  declared by the frozen trigger config. CLI root/config overrides that point
  elsewhere are rejected even if the canonical files still pass their hashes.
- A correction writes only to a new, unique C0F run directory. The historical
  run, its ignored detailed files, P6-C0 and all shared inputs stay immutable.

## What is being corrected

The historical result package remains as evidence of what was produced. These
issues require a separately labeled correction:

1. `decision.json:mechanism_findings.*.grid_recall_upper_bound` read the label
   decoding reference instead of `capacity_summary.json:splits.*.grid_bound`.
   For Test it reported `0.5930533955417314`; the actual relaxed grid bound
   in the historical capacity output is `5766/5787 = 0.9963711767755313`.
   The label reference and grid bound must be separate fields.
2. The old response and concurrency summaries did not consistently account for
   open/closed grid endpoints, score coverage/right censoring, long-event end
   exclusion, or events whose end times are out of onset order. Corrected
   trajectory markers use the complete label-legal event context, including
   purged cross-boundary events; no subset is independently rematched.
3. The old `1.5x` loss-dominance rule was added after seeing the ledger,
   including Test, and selected a follow-up route. The correction may display
   location shares but must label them `DESCRIPTIVE_ONLY`; it cannot choose
   C0R2 or C1, or change the C0 verdict.
4. The old completion record named a HEAD preceding the C0F implementation
   commit and did not archive the executing source. A new source snapshot
   binds the correction code and tests, but does not repair that history.

## Required gates and comparisons

1. Run the synthetic C0/C0F code checks against the exact source bytes to be
   used. `prepare` reruns the fixed test selection itself; an externally
   supplied validation record alone is not proof of a passing suite. Record
   command, exit status, logs and source digest from the run-owned checks. Archive all
   tracked and unignored Python source bytes, tests, both frozen configs and
   this protocol. A missing required source file is a STOP, not an omission
   from the snapshot. Any source change after the check requires a new check.
2. Refuse any `--reuse-run` other than the exact historical path and manifest
   SHA above. Verify every historical completion record and the two score
   hashes before copying. Copy Fit/Validation scores; validate unique sample
   identity, timestamps, labels, threshold decisions and Validation replay.
   Test consumes only the frozen `test_predictions.csv` and existing C0
   episode/matching records, with full identity and metric checks.
3. Recompute ledger, trajectories, concurrency, empty cross-cells and the
   label/grid/episode capacity diagnostics in the new directory. Keep original
   event populations and denominators. On each split require
   `observed TP <= U_episode <= U_grid <= complete GT`; `EXACT` additionally
   requires solver proof and witness replay through the frozen pipeline.
   The solver budget is 3600 seconds per split; a timeout cannot publish
   `EXACT` or `COMPLETE`.
4. Compare historical and corrected case IDs, split, onset/end, failure code,
   eligible and positive slot counts, candidate episode count, assigned
   prediction and official matched time. These should agree exactly. Compare
   the historical and corrected capacity ordering and label-reference event
   metrics. Corrected response bands, censoring, concurrency and presentation
   fields may differ; report their changed-row counts. Any failure of an
   unchanged check stops publication and requires investigation, not a
   relaxed equality rule.
5. Verify frozen inputs before and after; verify stage outputs before each
   dependent stage; publish `completion_manifest.json` only after all required
   outputs and checks pass. A completed or stopped directory is not reused.

The correction decision must record the original `BORDERLINE` verdict, the
historical inference-source limitation, an evidence ledger with
`CONFIRMED`/`OBSERVED`/`HYPOTHESIS`/`UNRESOLVED` grades, and no automatic
next-stage selection. The reused Test is not an independent holdout.

## Execution boundary

The executable entry point is `scripts/p6/audit_c0_failure.py`. Use its
`check-code` action with a unique validation directory, then `correct` with a
different unique correction directory, the passing `--validation-record`, and
the pinned `--reuse-run`. `correct` imports the historical Fit/Validation
score files and never calls model inference. It reads the frozen Test CSV and
does not write to the historical run. Inspect the new completion manifest and
the stable-field comparison before treating its report as the correction.

A future fresh Fit/Validation forward replay with a source-bound implementation
would be a separate run and claim. This correction does not make that claim.
