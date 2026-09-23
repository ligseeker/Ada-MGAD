# P6-C0F correction archive

Archive scope: the completed 2026-09-23 correction of the 2026-09-19 C0F
failure audit. This is a post hoc audit correction, not a new detector or RCA
experiment. The original C0 verdict remains `BORDERLINE`.

## Identity and custody

| Item | Frozen identity |
|---|---|
| Historical C0F run | `experiments/p6/c0f_failure_audit/c0f-seed42-20260919T0700/` |
| Historical completion manifest SHA-256 | `97e1fd3e34da06e7b64808b9f38372628c7007b9a2533c46582b50e5b7ace53a` |
| Correction run | `experiments/p6/c0f_failure_audit/c0f-correction-20260923T0755Z/` |
| Correction completion manifest SHA-256 | `faabd0a5e6486a89c6e6e3cab1af8f146af6cdab9fe5e728c027f5d18f80745a` |
| Correction execution base HEAD | `d518c17f4a76ddb30e9bfaeb33fecc982fad3f21` |
| Archived execution source digest | `afbcf349692e2746ab99e961b15bb3bf129b8205c3ca9d62ee6e3f3d7716b720` |
| Status | `COMPLETE_WITH_DECLARED_LIMITATIONS` |

The correction run's `source_snapshot.json` lists 109 source/config/test files.
Its `provenance/source_snapshot/` directory preserves their actual execution
bytes, including uncommitted changes, and `provenance/working_tree.diff`
preserves the then-current tracked diff. The base HEAD alone is **not** the
execution source identity. The archived source files and present working-tree
source passed the snapshot's SHA checks during this archive review.

The completion manifest binds 184 required outputs, including detailed CSVs,
solver witnesses, validation logs and aggregate records. All 184 present files
passed size and SHA-256 verification during archive review. The manifest and
aggregate JSON/Markdown records are versioned according to `.gitignore`;
row-level CSVs, witness files, logs and the byte-for-byte source snapshot remain
in the immutable local correction run directory. The historical run and its
outputs were not rewritten. Preserve the local run directory together with the
Git archive: the versioned aggregates alone do not contain every detailed
output.

## Validation and interpretation

- The correction run's own code check completed with 119 passing tests, with
  source digest bound in `test_results.json`; input integrity after execution is
  `PASS`, and Validation replay is `MATCH`.
- Stable identities and official detection/matching fields for all 16,131 GT
  events match the historical run. Corrected fields include two Test response
  bands (`never` to `censored`), 13 censor flags and five active concurrency
  counts.
- Test observed TP is 4,198/5,787. The exact episode-protocol diagnostic bound
  is 5,173/5,787, and the relaxed grid bound is 5,766/5,787. These bounds do
  not establish an achievable detector improvement.
- The prior `1.5x` loss-dominance route rule is `DESCRIPTIVE_ONLY`; neither
  C0R2 nor C1 is selected by this audit.
- The correction reused historical Fit/Validation score bytes. The historical
  inference source was not archived and remains `UNVERIFIED`; the correction's
  source binding does not retroactively verify it. Test had already been
  inspected during research design and is not an independent confirmation set.

Read the [correction protocol](P6_C0F_CORRECTION_PROTOCOL.md), the correction
run's `final_report.md`, `decision.json`, `evidence_ledger.json`,
`correction_summary.json`, `source_snapshot.json` and `completion_manifest.json`
for the full definitions and limitations. New analyses must use a unique run
directory and must not overwrite either completed C0F run.
