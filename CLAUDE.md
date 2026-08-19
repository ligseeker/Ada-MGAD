# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape: two decoupled code lines

This repo hosts **two independent research lines** that share only the checkout. Keeping them separate is a frozen design decision (`任务级解耦，事件级集成` — task-level decoupling, event-level integration), documented in `docs/PROJECT_CONTEXT.md`.

| | Ada-MGAD (anomaly detection) | Standalone RCA (root-cause ranking) |
|---|---|---|
| Entry | `main.py` → `util/runtime.py` | `scripts/*.py` |
| Code | `util/`, `src/model.py`, `src/model_util.py`, `src/train*`… | `src/data/`, `src/features/`, `src/models/`, `src/baselines/`, `src/evaluation/` |
| Unit | node × time window | fault case × candidate service |
| Deps | torch, torch_geometric (**not in `requirements.txt`** — install separately) | numpy, pandas, scikit-learn only |
| Tests | none | `tests/` (15 files, `unittest`) |

**Hard rule:** the RCA line must never consume Ada-MGAD latent features or checkpoints. Dependencies flow one way only: `scripts/` → `src/{data,features,models,baselines,evaluation}`. Nothing under those RCA packages may import `src.model`, `src.model_util`, or `util.*`.

## Commands

### RCA line — tests and the de-facto lint gate

There is no pytest, Makefile, tox, pyproject, or CI config. The project's own audit trail (`docs/EXPERIMENT_LOG.md`) treats these three commands as the gate that must pass before any experiment record is written:

```bash
python -m unittest discover -s tests -v     # full suite; must be 100% green
python -m compileall -q src scripts tests   # stands in for a linter
git diff --check                            # whitespace/conflict-marker check
```

Single test file / single test method / direct execution:

```bash
python -m unittest tests.test_split_integrity
python -m unittest tests.test_metrics.RankingMetricsTest.test_individual_metrics
python tests/test_metrics.py                # many test files have a __main__ block
```

### Ada-MGAD line

```bash
python util/MSDS/pre_MSDS.py                # preprocess (writes data/MSDS-pre)
python main.py --dataset msds               # train + evaluate
python util/GAIA/pre_GAIA.py
python main.py --dataset gaia
bash scripts/run_msds_smoke.sh              # 1-epoch CPU smoke on data/examples/msds_tiny
python main.py --dataset msds --evaluate true --model_path ./result/<run_dir>
```

Any key in `DATASET_PROFILES` (`util/runtime_config.py`) is overridable as `--<key>`, because `ARG_SPECS` builds every flag with `default=argparse.SUPPRESS` — CLI values are layered over the profile dict, so absent flags never clobber profile defaults. Expected data layout is in `README.md`; `data/` and `result/` are gitignored.

### P1 pipeline (benchmark layer — frozen, normally only re-run to verify)

P1 scripts are the **only** place raw dataset paths enter the system, so they require explicit paths:

- GAIA MicroSS: `/home/zhangll24/project_2/MultimodalAD/MSTGAD-GAIA/data/GAIA/MicroSS`
- RCAEval / RE2-OB: `/home/zhangll24/RCA_project/datasets/RCAEval` (`.../RE2-OB`, archive `.../RE2-OB.zip`)

Verbatim reproduction commands with all flags live in `docs/EXPERIMENT_LOG.md` — copy from there rather than reconstructing them. Order matters:

```
diagnose_p1_telemetry → prepare_p1_manifests (--gaia-context-seconds 300)
  → diagnose_p1_context_groups → prepare_p1_splits (--folds 5 --seed 20260819)
  → run_p1_sanity_baselines → run_p1_metric_change
  → finalize_p1_gaia_inclusion → pin_p1_rcaeval_source → audit_p1_gates
```

### P2 pipeline (representation + model ladder — the active work)

P2 scripts take **no required arguments**: they default to `artifacts/p1/{manifests,splits,inclusion/gaia,source_snapshots/re2ob,telemetry_diagnostics.json}` and hard-fail with `FileNotFoundError("required source bindings are missing: …")` if a P1 artifact is absent. Raw telemetry is reached through URIs recorded in the manifests, never through new path flags.

```bash
python scripts/extract_p2_metric_features_smoke.py   # always smoke before full extraction
python scripts/extract_p2_metric_features.py
python scripts/extract_p2_event_features_smoke.py
python scripts/extract_p2_event_features.py
python scripts/run_p2_c0_metric.py                   # then audit_p2_c0_metric.py
python scripts/run_p2_linear_ablations.py            # C0-L / C0-T / C1-I; then audit_p2_linear_ablations.py
python scripts/bootstrap_p2_c1_i.py
python scripts/run_p2_m1_stage.py                    # M1-S
```

Every `run_*` script has a paired `audit_*` script that independently recomputes metrics and re-checks fold isolation. **Run the audit; a run without its audit is not evidence.**

`scripts/run_p2_linear_ablations.py` and `scripts/run_p2_m1_stage.py` import private helpers (`_fit_and_rank`, `_case_rows`, `_root_macro_avg5`, `_sha256`, `_write_jsonl`, `RUN_SCHEMA_VERSION`…) from `scripts.run_p2_c0_metric`. That module is a de-facto shared library — changing those helpers changes three stages at once, and any change invalidates already-recorded SHA-256 digests.

## Architecture invariants (the parts that are easy to break)

### The artifact chain is the architecture

There are no service objects or DI wiring; stages communicate through content-addressed files under `artifacts/` (gitignored, present locally):

```
raw datasets → artifacts/p1/manifests/<ds>/{inputs,labels}.jsonl + manifest.json
             → artifacts/p1/splits/  → artifacts/p1/baselines/
             → artifacts/p1/inclusion/gaia/ + source_snapshots/re2ob/ + gate_audit.json
             → artifacts/p2/{features,event_features}/<ds>/  (values.npy + observed.npy + index.jsonl + manifest.json)
             → artifacts/p2/runs/<method>/<ds>/  → artifacts/p2/*_summary.json, *_audit.json
```

To understand any stage, read its `manifest.json` before its code. Schema versions gate compatibility: `p1_rca_manifest_v1`, `p1_split_manifest_v1`, `p2_feature_bundle_v2`.

Writes are deterministic by construction and must stay that way: canonical JSON (`sort_keys=True, separators=(",",":"), allow_nan=False, ensure_ascii=False`, newline-terminated), `<f4` value matrices with parallel bool `observed` masks, atomic `.tmp` + `replace()`, SHA-256 per file in the manifest. Re-running a stage on unchanged inputs must reproduce identical digests — that byte-identity is the reproducibility claim, so never introduce dict-ordering, timestamps, or unsorted iteration into a writer.

### Label Firewall

Labels are physically separated, not just conventionally:

- `src/data/schema.py` — `RCACaseInput` (prediction-visible) vs `RCACaseLabel` (train/eval only), plus `assert_label_free()` and `_FORBIDDEN_METADATA_KEYS` matched after `re.sub(r"[^a-z0-9]", "", key.lower())` normalization, so `Root_Service`/`rootService`/`ROOTSERVICE` are all caught.
- `src/features/schema.py` — `_FORBIDDEN_FEATURE_TOKENS` rejects label-ish substrings in *feature names*; masked cells must be exactly `0.0`; coverage must equal the input `(case_id, service)` pairs exactly, sorted and duplicate-free.
- `src/evaluation/evaluator.py` — `predict_rankings(inputs, predictor)` takes no labels at all; the signature is the enforcement. `validate_ranking` demands a duplicate-free permutation of exactly `case_input.services`.

When adding a feature or a modality, the failure mode to watch is not a crash but leakage: derive nothing from `labels`, fit every scaler/vocabulary/threshold inside the training fold only, and keep template/vocabulary-dependent features out of L0 (they belong in a train-fold-only L1 stage).

### Frozen protocol constants

Do not change these without an explicit user decision plus a documented version bump — every recorded number depends on them:

- Window `T_pre = T_post = 300 s`, half-open, milliseconds.
- Seed `20260819`; 5-fold OOF (GAIA grouped-stratified over 322 context groups; RE2-OB singleton-stratified).
- Nested OOF selection: inner 4-fold rotation inside each outer-train fold; L2 logistic regression, `C ∈ {0.01, 0.1, 1, 10}`; per-case weight root 0.5 / all non-root 0.5; tie-break = simpler model → stronger regularization → shorter coverage-supported onset → config lexicographic.
- Metric onset candidates restricted to **60/120 s** (30 s is an unsupported control only — GAIA 30 s onset is observable in ~0.09% of service rows).
- Feature widths: metric 10 comparison blocks × 17 = 170; L0 logs 10 × 5 = 50; T0 traces 10 × 8 = 80.
- Metrics: AC@1/3/5, `Avg@5 = mean(AC@1..AC@5)`, MRR — always reported at all three layers (overall / fault-type macro / root-service macro). Primary endpoint = **root-service macro Avg@5**; key secondary = root-macro AC@1.

### Ada-MGAD runtime pattern

`util/runtime.py` is a thin orchestrator: `prepare_args` either hashes a new `result/<hash>` dir and dumps `params.json`, or (evaluate mode) restores `params.json` and honors only `RUNTIME_OVERRIDE_KEYS = {model_path, evaluate, result_dir, data_path, dataset_path}`. The dataset module is loaded by `importlib` and called per `process_mode` (`dict` → `Process(args)`, `kwargs` → `Process(**args)`). The train/test split is a **70/30 chronological cut** (`int(len(dataset) * 0.7)`), not random. Evaluation loads *both* the `loss` and `f1` checkpoints and appends both to `result_summary.log`. `src/model.py` caches dynamic-graph edge weights every `graph_update_steps` and tracks hit/refresh counters — a stale cache silently changes results, so reset it when changing training loops.

## Documentation governance (read before writing docs or claiming results)

`docs/` (16 files, Chinese) is the governance layer and `docs/README.md` is the single entry point. Its rules are binding:

- **Information authority order:** (1) repo code/tests/manifests/reproducible artifacts → (2) decisions marked 已冻结/用户已确认 → (3) official papers/repos/data docs → (4) unverified conversation claims → (5) hypotheses. When docs and artifacts disagree, **artifacts win**.
- `docs/RESEARCH_STATUS.md` is the declared state owner and must be synced after every gate. `docs/EXPERIMENT_LOG.md` is **append-only** — never overwrite an old conclusion; append a new record using the template in its section 17.
- Every number reported anywhere must carry: data version, split, seed, command, commit hash, artifact path. Status tags in use: `仓库已验证`, `官方已验证`, `用户已确认`, `对话报告`, `待验证`; evidence grades: `reproduced-current`, `artifact-verified`, `conversation-reported`, `planned`.
- Track C (in-protocol controlled ladder) is the **sole** source of H1/H2/H3 claims. Track E (RCAEval official methods, pinned commit `4695aa69f4f1f57b9094ca04ff235908b73a8e24`) is external reference only and must be reported in a separate table. Oracle-trigger and detected-trigger results also go in separate tables.
- Go/No-Go thresholds: exploratory signal = positive point estimates on both datasets; claim-ready = 95% paired-bootstrap CI lower bound > 0 on both datasets **and** no key-secondary regression > 0.01.
- `docs/PROJECT_CONTEXT.md` lists explicit claim boundaries (e.g. cannot claim structure necessarily helps, cannot claim indicator-level RCA solved, cannot claim Oracle RCA is end-to-end). Respect them in any text you write.

## Current state

P1 gates G1–G8 are closed and the benchmark layer is frozen. In P2, the model ladder has completed C0-M, C0-L, C0-T and C1-I (`artifacts/p2/runs/{c0_m,c0_l,c0_t,c1_i}/`, `c1_i_bootstrap.json`, `linear_ablation_{summary,audit}.json` all exist); **M1-S is in progress** — no `artifacts/p2/runs/m1_s/` or `m1_s_summary.json` yet.

Note: `docs/RESEARCH_STATUS.md` (`research_state_v9`) lags this — it still says full L0/T0 extraction is pending, while `artifacts/p2/event_features/{gaia_main,re2ob}` exist and `docs/P2_EXPERIMENT_PLAN.md` V0.2 already records C0-M/L/T and C1-I as completed. Per the authority order above, trust the artifacts, and sync `RESEARCH_STATUS.md` at the next gate.

## Repo mechanics

This checkout is a **git worktree** (`.git` is a file pointing at `/home/zhangll24/RCA_project/Ada-MGAD/.git/worktrees/Ada-MGAD-rca-claudecode`) on branch `claudecode`; PRs target `main`. `artifacts/`, `data/*`, and `result/` are gitignored — the only tracked data files are the two `data/examples/msds_tiny/` pickles kept via `.gitignore` negations for the smoke test.
