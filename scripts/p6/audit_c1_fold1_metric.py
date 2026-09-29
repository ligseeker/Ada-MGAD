#!/usr/bin/env python3
"""Read-only, Fit-prefix-only Metric candidate audit for failed P6-C1 fold 1.

The full `run` scan is intentionally a manual command. It does not continue
the failed C1 run, change its frozen policy, or inspect labels/Test values.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.e2e.c1_fold_preprocessing import (
    G2_CONFIG, G2_CONFIG_SHA256, _bound_path, _fold_intervals, _sha256,
    _write_json_new, verify_raw_content,
)
from src.e2e.gaia_preprocessing.raw import (
    GAIA_SERVICES, MetricSlot, _candidate_correlation, _grid,
    _metric_quality, _metric_values, _scope_for_targets,
    index_metric_files, metric_semantic_kind,
)
from src.e2e.parallel import ordered_process_map


AUDIT_PARENT = ROOT / "experiments/p6/c1_prefix_metric_audit"
DISPOSITION = ROOT / "docs/P6_C1_G3_FOLD1_PREFIX_SCHEMA_NO_GO_20260929.json"
SOURCE_PATHS = (
    "scripts/p6/audit_c1_fold1_metric.py",
    "src/e2e/gaia_preprocessing/raw.py",
    "src/e2e/gaia_preprocessing/metric.py",
    "src/e2e/parallel.py",
    "src/e2e/c1_fold_preprocessing.py",
    "configs/e2e/gaia_p6_c1_g2_v1_1.json",
    "configs/e2e/gaia_p5_v3_preprocessing_v2.json",
    "configs/e2e/gaia_p5_v3_preprocessing_v2.policy.json",
    "docs/P6_C1_G3_FOLD1_PREFIX_SCHEMA_NO_GO_20260929.json",
)
LEDGER_COLUMNS = (
    "slot_name", "source_key", "scope", "statistic", "semantic_kind",
    "targets", "source_file_count", "source_files", "quality_status", "failed_target",
    "quality_failure_reason", "failed_target_coverage", "failed_target_unique",
    "failed_target_dynamic", "failed_target_dynamic_ratio", "quality_score",
    "quality_rank", "correlation_status", "redundant_with", "pearson",
    "spearman",
)


def _source_hashes():
    return {name: _sha256(ROOT / name) for name in SOURCE_PATHS}


def _require_committed_sources():
    for name in SOURCE_PATHS:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", name],
                                 cwd=str(ROOT), stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        if tracked.returncode:
            raise ValueError("Metric audit source is not committed: {}".format(name))
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *SOURCE_PATHS],
                           cwd=str(ROOT), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    if clean.returncode:
        raise ValueError("Metric audit source/config differs from Git HEAD")


def _failure_binding():
    disposition = json.loads(DISPOSITION.read_text(encoding="utf-8"))
    if (disposition.get("status") != "PREFIX_SCHEMA_NO_GO"
            or disposition.get("fold") != 1
            or disposition.get("metric_qualified_real_slots_after_quality_and_correlation") != 27
            or disposition.get("metric_required_real_slots") != 45):
        raise ValueError("C1 fold 1 disposition identity drift")
    for record in disposition["evidence"].values():
        path = (ROOT / record["path"]).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            raise ValueError("C1 failure evidence escaped repository")
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise ValueError("C1 failure evidence changed: {}".format(path))
    lock = json.loads((ROOT / disposition["evidence"]["run_lock"]["path"]).read_text(encoding="utf-8"))
    failure = json.loads((ROOT / disposition["evidence"]["fold_failure"]["path"]).read_text(encoding="utf-8"))
    replay_sources = (
        "src/e2e/gaia_preprocessing/raw.py",
        "src/e2e/gaia_preprocessing/metric.py",
        "src/e2e/c1_fold_preprocessing.py",
        "src/e2e/parallel.py",
    )
    if (lock.get("git_head") != disposition["run_git_head"]
            or any(lock.get("source_sha256", {}).get(name) != _sha256(ROOT / name)
                   for name in replay_sources)
            or failure.get("message")
            != "qualified real Metric slots 27 below required budget 45; refusing padding"):
        raise ValueError("C1 failure source or message drift")
    return disposition


def _inputs(run_id: str):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{4,79}", run_id):
        raise ValueError("audit run ID must be a simple unique slug")
    output = AUDIT_PARENT / run_id
    if output.exists():
        raise ValueError("Metric audit output already exists; no overwrite")
    _require_committed_sources()
    if _sha256(G2_CONFIG) != G2_CONFIG_SHA256:
        raise ValueError("G2 protocol bytes drift")
    protocol = json.loads(G2_CONFIG.read_text(encoding="utf-8"))
    if protocol["run_id"] != "c1-prefix-oos-v1-seed42":
        raise ValueError("C1 protocol run identity drift")
    _failure_binding()
    base_path = _bound_path(protocol["bindings"]["base_config"])
    policy_path = _bound_path(protocol["bindings"]["preprocessing_policy"])
    raw_manifest = _bound_path(protocol["bindings"]["raw_content_manifest"])
    if json.loads(raw_manifest.read_text(encoding="utf-8"))["config_sha256"] != _sha256(base_path):
        raise ValueError("C1 raw inventory base-config binding drift")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    metric_policy = policy["metric"]
    if (int(metric_policy["base_slots"]) != 45 or int(metric_policy["dimension"]) != 48
            or float(metric_policy["pearson_threshold"]) != 0.995
            or float(metric_policy["spearman_threshold"]) != 0.995):
        raise ValueError("C1 frozen Metric policy drift")
    grid_ms = int(base["ad"]["grid_seconds"]) * 1000
    window_bins = int(protocol["detector"]["history_seconds"]) * 1000 // grid_ms
    fit_interval = _fold_intervals(protocol, 1, grid_ms, window_bins)["fit"]
    raw_root = Path(base["gaia_raw_root"]).resolve()
    metric_dir = raw_root / "metric/metric_split/metric"
    if not metric_dir.is_dir():
        raise ValueError("bound raw Metric directory missing")
    return output, protocol, metric_policy, raw_root, raw_manifest, metric_dir, fit_interval, grid_ms


def _quality_details(values, score, *, min_coverage, min_unique, min_dynamic_ratio):
    finite = values[np.isfinite(values)]
    coverage = float(len(finite) / len(values)) if len(values) else 0.0
    unique = int(len(np.unique(finite)))
    dynamic = None
    ratio = None
    if coverage < min_coverage:
        reason = "coverage"
    elif unique < min_unique:
        reason = "unique"
    else:
        q05, q95 = np.quantile(finite, [0.05, 0.95])
        dynamic = float(q95 - q05)
        ratio = dynamic / (float(np.mean(np.abs(finite))) + 1e-6)
        if dynamic <= 0.0:
            reason = "dynamic_zero"
        elif ratio < min_dynamic_ratio:
            reason = "dynamic_ratio"
        else:
            reason = None
    if (score is None) != (reason is not None):
        raise ValueError("Metric diagnostic quality rule differs from adapter")
    return {"coverage": coverage, "unique": unique, "dynamic": dynamic,
            "dynamic_ratio": ratio, "reason": reason}


def _audit_worker(task):
    (key, records, services, start_ms, end_ms, grid_ms, logical, scope,
     targets, semantic_kind, statistic, min_coverage, min_unique,
     min_dynamic_ratio, temporary) = task
    grid = _grid(start_ms, end_ms, grid_ms)
    semantic = "rate" if semantic_kind == "counter" else "direct" if semantic_kind == "direct_rate" else "value"
    slot = MetricSlot("{}::{}::{}".format(scope, logical, statistic), scope,
                      logical, statistic, semantic_kind, tuple(targets), (key,))
    values_by_target = {}
    scores = []
    row = {"slot_name": slot.name, "source_key": key, "scope": scope,
           "statistic": statistic, "semantic_kind": semantic_kind,
           "targets": json.dumps(targets, separators=(",", ":")),
           "source_file_count": len(records),
           "source_files": json.dumps([Path(record.path).name for record in records], separators=(",", ":")),
           "quality_status": "pass",
           "failed_target": "", "quality_failure_reason": "",
           "failed_target_coverage": "", "failed_target_unique": "",
           "failed_target_dynamic": "", "failed_target_dynamic_ratio": "",
           "quality_score": "", "quality_rank": "", "correlation_status": "not_tested",
           "redundant_with": "", "pearson": "", "spearman": ""}
    for target in targets:
        values = _metric_values(records, target, grid, start_ms, end_ms, semantic, statistic)
        score = _metric_quality(values, min_coverage, min_unique, min_dynamic_ratio)
        details = _quality_details(values, score, min_coverage=min_coverage,
                                   min_unique=min_unique, min_dynamic_ratio=min_dynamic_ratio)
        if score is None:
            row.update({"quality_status": "reject", "failed_target": target,
                        "quality_failure_reason": details["reason"],
                        "failed_target_coverage": details["coverage"],
                        "failed_target_unique": details["unique"],
                        "failed_target_dynamic": details["dynamic"] if details["dynamic"] is not None else "",
                        "failed_target_dynamic_ratio": details["dynamic_ratio"] if details["dynamic_ratio"] is not None else ""})
            return row, slot, None
        values_by_target[target] = values
        scores.append(float(score))
    row["quality_score"] = float(np.mean(scores))
    with tempfile.NamedTemporaryFile(prefix="metric_audit_", suffix=".npz",
                                     dir=str(temporary), delete=False) as stream:
        np.savez(stream, **values_by_target)
        descriptor = stream.name
    return row, slot, descriptor


def _candidate_tasks(index, services, interval, grid_ms, temporary):
    tasks = []
    scope_ineligible = 0
    start_ms, end_ms = interval
    for key, records in index.items():
        source_scope = "host" if key.startswith("host::") else "service"
        logical = key.split("::", 1)[1]
        targets = tuple(sorted(set(target for record in records for target in record.targets),
                               key=lambda service: list(services).index(service)))
        scope = _scope_for_targets(targets, source_scope, services)
        if scope is None:
            scope_ineligible += 1
            continue
        applicable = tuple(service for service in services if service in targets)
        semantic_kind = metric_semantic_kind(logical)
        core_count = len(set(record.core_id for record in records if record.core_id is not None))
        statistics = ("mean", "max") if core_count > 1 else ("value",)
        for statistic in statistics:
            tasks.append((key, records, tuple(services), start_ms, end_ms,
                          grid_ms, logical, scope, applicable, semantic_kind,
                          statistic, 0.20, 2, 1e-4, str(temporary)))
    return tasks, scope_ineligible


def analyze_metric_candidates(metric_dir: Path, interval: tuple, *, grid_ms: int,
                              workers: int, pearson_threshold: float,
                              spearman_threshold: float, temporary: Path):
    """Replay the frozen candidate path and explain each Fit-prefix rejection."""
    services = GAIA_SERVICES
    index = index_metric_files(metric_dir, services)
    tasks, scope_ineligible = _candidate_tasks(index, services, interval, grid_ms, temporary)
    results, parallel = ordered_process_map(_audit_worker, tasks, workers=workers, start_method="spawn")
    rows = []
    candidates = []
    for row, slot, descriptor in results:
        rows.append(row)
        if descriptor is None:
            continue
        with np.load(descriptor, allow_pickle=False) as payload:
            values = {target: np.asarray(payload[target]) for target in slot.targets}
        Path(descriptor).unlink()
        candidates.append((slot, float(row["quality_score"]), values))
    candidates.sort(key=lambda item: (-item[1], item[0].name))
    row_by_name = {row["slot_name"]: row for row in rows}
    if len(row_by_name) != len(rows):
        raise ValueError("Metric audit candidate slot names are not unique")
    reduced = []
    for rank, candidate in enumerate(candidates, 1):
        row = row_by_name[candidate[0].name]
        row["quality_rank"] = rank
        for kept in reduced:
            pearson, spearman = _candidate_correlation(candidate[2], kept[2], services)
            pearson_hit = np.isfinite(pearson) and abs(pearson) >= pearson_threshold
            spearman_hit = np.isfinite(spearman) and abs(spearman) >= spearman_threshold
            if pearson_hit or spearman_hit:
                row.update({"correlation_status": "reject", "redundant_with": kept[0].name,
                            "pearson": pearson if np.isfinite(pearson) else "",
                            "spearman": spearman if np.isfinite(spearman) else ""})
                break
        else:
            row["correlation_status"] = "keep"
            reduced.append(candidate)
    if len(reduced) + sum(row["correlation_status"] == "reject" for row in rows) != len(candidates):
        raise ValueError("Metric audit correlation accounting differs from adapter")
    summary = {
        "schema_version": "p6_c1_fold1_metric_candidate_audit_v1",
        "fit_interval_ms": list(interval), "grid_ms": int(grid_ms),
        "catalog_metric_csv_files": sum(1 for _ in metric_dir.glob("*.csv")),
        "catalog_indexed_metric_files": sum(len(records) for records in index.values()),
        "catalog_source_groups": len(index), "scope_ineligible_source_groups": scope_ineligible,
        "candidate_tasks": len(tasks), "quality_pass": len(candidates),
        "quality_pass_by_scope": dict(sorted(Counter(row["scope"] for row in rows
                                                    if row["quality_status"] == "pass").items())),
        "quality_rejections": dict(sorted(Counter(row["quality_failure_reason"] for row in rows
                                                  if row["quality_status"] == "reject").items())),
        "correlation_reject": sum(row["correlation_status"] == "reject" for row in rows),
        "post_correlation_real_slots": len(reduced),
        "post_correlation_by_scope": dict(sorted(Counter(candidate[0].scope for candidate in reduced).items())),
        "surviving_slot_names": [candidate[0].name for candidate in reduced],
        "parallel": parallel,
        "gt_labels_parsed": False, "selection_or_test_values_parsed": False,
    }
    return rows, summary


def _run(run_id: str, workers: int):
    output, protocol, policy, raw_root, raw_manifest, metric_dir, interval, grid_ms = _inputs(run_id)
    budget = int(protocol["resource_budget"]["ad_preprocessing_workers"])
    if not 1 <= workers <= budget:
        raise ValueError("audit worker count outside frozen C1 budget")
    sources_before = _source_hashes()
    head_before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    print("VERIFY bound raw catalog before audit", flush=True)
    raw_before = verify_raw_content(raw_manifest, raw_root)
    if _source_hashes() != sources_before:
        raise ValueError("Metric audit source changed during raw preflight")
    output.mkdir(parents=True, exist_ok=False)
    _write_json_new(output / "INCOMPLETE.json", {"status": "INCOMPLETE", "stage": "metric_prefix_audit"})
    try:
        print("AUDIT fold 1 Fit-prefix Metric candidates with {} spawn workers".format(workers), flush=True)
        with tempfile.TemporaryDirectory(prefix="candidate_arrays_", dir=str(output)) as temporary:
            rows, summary = analyze_metric_candidates(
                metric_dir, interval, grid_ms=grid_ms, workers=workers,
                pearson_threshold=float(policy["pearson_threshold"]),
                spearman_threshold=float(policy["spearman_threshold"]),
                temporary=Path(temporary))
        ledger = output / "candidate_ledger.csv"
        with ledger.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=LEDGER_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        summary["required_real_slots"] = int(policy["base_slots"])
        summary["reproduces_original_27"] = summary["post_correlation_real_slots"] == 27
        _write_json_new(output / "summary.json", summary)
        if not summary["reproduces_original_27"]:
            raise ValueError("Metric audit does not reproduce original 27-slot failure")
        print("VERIFY bound raw catalog after audit", flush=True)
        if verify_raw_content(raw_manifest, raw_root) != raw_before:
            raise ValueError("bound raw content changed during Metric audit")
        if _source_hashes() != sources_before:
            raise ValueError("Metric audit source changed during run")
        _failure_binding()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
        if head != head_before:
            raise ValueError("Metric audit Git HEAD changed during run")
        files = {name: {"bytes": (output / name).stat().st_size,
                        "sha256": _sha256(output / name)}
                 for name in ("candidate_ledger.csv", "summary.json")}
        _write_json_new(output / "completion_manifest.json", {
            "schema_version": "p6_c1_fold1_metric_audit_completion_v1", "status": "COMPLETE",
            "run_id": run_id, "source_git_head": head,
            "source_sha256": sources_before, "g2_protocol_sha256": _sha256(G2_CONFIG),
            "raw_content": raw_before, "failed_run_disposition_sha256": _sha256(DISPOSITION),
            "workers": workers, "files": files,
        })
        (output / "INCOMPLETE.json").unlink()
        print("COMPLETE Fit-prefix Metric audit:", summary["post_correlation_real_slots"], "of 45 real slots")
        print("quality_pass", summary["quality_pass"],
              "quality_rejections", summary["quality_rejections"],
              "correlation_reject", summary["correlation_reject"])
        print(output / "summary.json")
    except Exception as exc:
        try:
            _write_json_new(output / "failure.json", {"status": "INCOMPLETE",
                                                       "error_type": type(exc).__name__,
                                                       "message": str(exc)})
        except OSError:
            pass
        raise


def _check(run_id: str):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{4,79}", run_id):
        raise ValueError("audit run ID must be a simple unique slug")
    output = AUDIT_PARENT / run_id
    if (output / "INCOMPLETE.json").exists():
        raise ValueError("Metric audit run is incomplete")
    manifest = json.loads((output / "completion_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "p6_c1_fold1_metric_audit_completion_v1"
            or manifest.get("status") != "COMPLETE" or manifest.get("run_id") != run_id
            or manifest.get("source_sha256") != _source_hashes()
            or manifest.get("g2_protocol_sha256") != _sha256(G2_CONFIG)
            or manifest.get("failed_run_disposition_sha256") != _sha256(DISPOSITION)
            or set(manifest.get("files", {})) != {"candidate_ledger.csv", "summary.json"}):
        raise ValueError("Metric audit completion/source binding drift")
    _failure_binding()
    for name, record in manifest["files"].items():
        path = output / name
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]):
            raise ValueError("Metric audit output drift: {}".format(name))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    with (output / "candidate_ledger.csv").open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != LEDGER_COLUMNS:
            raise ValueError("Metric audit candidate ledger schema drift")
        rows = list(reader)
    if (summary.get("reproduces_original_27") is not True
            or summary.get("post_correlation_real_slots") != 27
            or summary.get("required_real_slots") != 45
            or summary.get("candidate_tasks") != len(rows)
            or summary.get("quality_pass") != sum(row["quality_status"] == "pass" for row in rows)
            or summary.get("quality_rejections") != dict(sorted(Counter(
                row["quality_failure_reason"] for row in rows if row["quality_status"] == "reject").items()))
            or summary.get("candidate_tasks") != summary.get("quality_pass") + sum(
                row["quality_status"] == "reject" for row in rows)
            or summary.get("correlation_reject") != sum(row["correlation_status"] == "reject" for row in rows)
            or summary.get("post_correlation_real_slots") != sum(row["correlation_status"] == "keep" for row in rows)):
        raise ValueError("Metric audit ledger/summary accounting drift")
    print("PASS sealed Metric audit", output)
    print("quality_pass", summary["quality_pass"],
          "quality_rejections", summary["quality_rejections"],
          "correlation_reject", summary["correlation_reject"],
          "post_correlation_real_slots", summary["post_correlation_real_slots"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run", "check"))
    parser.add_argument("--run-id", default="c1-fold1-metric-audit-v1-20260929")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.action == "preflight":
        output, _, _, _, _, metric_dir, interval, _ = _inputs(args.run_id)
        print("PASS bound failure/source/policy; new output", output)
        print("Fit-only Metric directory", metric_dir, "interval_ms", interval)
    elif args.action == "run":
        _run(args.run_id, args.workers)
    else:
        _check(args.run_id)


if __name__ == "__main__":
    main()
