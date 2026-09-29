#!/usr/bin/env python3
"""P6-C1-v2 bounded real-array smoke (not an experiment result).

Slices a short legal interval out of the frozen shared Train arrays, runs the
frozen C0 detector training rule for one epoch, seals the label-free Generation
episodes, and exercises the Train-only causal matching after the seal. It never
reads Test telemetry, Test labels or any GT root label, and it writes no formal
run artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.e2e.c1_oos_matching import match_c1_oos_episodes
from src.e2e.c1_v2_detector import (
    build_c1_v2_supervision, load_c1_v2_registry, train_c1_v2_fold_detector,
    validate_c1_v2_fold_detector,
)
from src.e2e.c1_v2_shared_data import (
    load_protocol, materialize_c1_v2_fold, read_json, source_hashes,
)
from src.e2e.system_trigger import build_trigger_labels, prediction_time_grid


def _find_bounded_intervals(protocol, registry, *, grid_ms: int):
    """Pick the first short, legal, GT-bearing interval inside frozen Train."""
    base = read_json(ROOT / "configs/e2e/gaia_p5_v3_preprocessing_v2.json")
    train_start = int(base["split"]["absolute_start_ms"])
    fold = protocol["folds"][0]
    limit = int(fold["intervals_ms"]["detector_selection"][1])
    fit_bins, selection_bins, generation_bins = 60, 30, 30
    total_bins = fit_bins + selection_bins + generation_bins
    legal = registry.loc[registry["detector_domain"].astype(bool)]
    candidate_starts = np.arange(train_start, limit - total_bins * grid_ms + grid_ms, grid_ms,
                                dtype=np.int64)
    selection_ms = selection_bins * grid_ms
    starts = legal["start_ms"].to_numpy(dtype=np.int64)
    ends = legal["end_ms"].to_numpy(dtype=np.int64)
    low = np.searchsorted(candidate_starts, ends - selection_ms, side="right")
    high = np.searchsorted(candidate_starts, starts, side="right")
    delta = np.zeros(len(candidate_starts) + 1, dtype=np.int64)
    np.add.at(delta, np.clip(low, 0, len(candidate_starts)), 1)
    np.add.at(delta, np.clip(high, 0, len(candidate_starts)), -1)
    qualifying = np.cumsum(delta)[:-1] > 0
    for position in np.flatnonzero(qualifying):
        start = int(candidate_starts[position])
        fit = (start, start + fit_bins * grid_ms)
        selection = (fit[1], fit[1] + selection_bins * grid_ms)
        generation = (selection[1], selection[1] + generation_bins * grid_ms)
        fit_grid = np.arange(fit[0], fit[1], grid_ms, dtype=np.int64)
        fit_events = legal.loc[(legal["start_ms"] < fit[1]) & (legal["end_ms"] > fit[0])]
        fit_labels = build_trigger_labels(prediction_time_grid(fit_grid, grid_seconds=30),
                                          fit_events[["start_ms", "end_ms"]],
                                          positive_window_seconds=60)
        if not (fit_labels == 1).any() or not (fit_labels == 0).any():
            continue
        return {"fit": fit, "selection": selection, "generation": generation}
    raise RuntimeError("no bounded GT-bearing smoke interval was found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    protocol = load_protocol()
    if args.report.exists():
        raise FileExistsError("smoke report already exists")
    base = read_json(ROOT / "configs/e2e/gaia_p5_v3_preprocessing_v2.json")
    grid_ms = int(base["ad"]["grid_seconds"]) * 1000
    registry = load_c1_v2_registry(protocol)
    intervals = _find_bounded_intervals(protocol, registry, grid_ms=grid_ms)
    checks = {}
    with tempfile.TemporaryDirectory() as temporary:
        fold_root = Path(temporary) / "fold_01"
        fold = materialize_c1_v2_fold(protocol=protocol, fold_number=1, fold_root=fold_root,
                                      intervals_ms=intervals, smoke=True)
        checks["smoke_manifest_not_formal"] = fold["formal"] is False and fold["smoke"] is True
        checks["fit_windows"] = fold["segments"]["fit"]["legal_windows"]
        detector = train_c1_v2_fold_detector(fold_root=fold_root, gpu=False,
                                             threshold_workers=1, epochs=1, patience=0,
                                             protocol=protocol)
        validate_c1_v2_fold_detector(fold_root=fold_root, protocol=protocol)
        episodes = pd.read_csv(fold_root / "detector/generation_episodes.csv")
        scores = pd.read_csv(fold_root / "detector/generation_scores.csv")
        checks["generation_label_free_columns"] = not (
            {"trigger_label", "case_id", "root_service", "fault_type", "service"}
            & set(episodes.columns))
        checks["generation_score_columns"] = sorted(scores.columns)
        checks["sealed_stage_files"] = sorted(
            path.name for path in (fold_root / "detector").iterdir())
        checks["checkpoint_bytes"] = (fold_root / "detector/checkpoint.pt").stat().st_size
        checks["selected_epoch"] = detector["selected_epoch"]
        checks["threshold"] = detector["threshold"]
        checks["generation_episodes"] = int(len(episodes))
        supervision = build_c1_v2_supervision(protocol=protocol, manifest=fold,
                                              registry=registry)
        checks["supervision_generation_labels_built"] = bool(
            supervision.audit["generation_labels_built"])
        match = match_c1_oos_episodes(fold=1, interval_ms=intervals["generation"],
                                      episodes=episodes, registry=registry)
        checks["matching_after_seal"] = {
            "episodes": match.audit["episodes"], "matched": match.audit["matched"],
            "false_alarms": match.audit["false_alarms"], "misses": match.audit["misses"],
            "complete_gt": match.audit["complete_gt"],
            "dual_context_legal": match.audit["dual_context_legal"]}
        checks["scores_written_before_any_gt_join"] = True
    report = {
        "schema_version": "p6_c1_v2_bounded_raw_smoke_v1",
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "run_id": protocol["run_id"],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                            text=True).strip(),
        "source_sha256": source_hashes(),
        "smoke_intervals_ms": {name: list(value) for name, value in intervals.items()},
        "smoke_epochs": 1,
        "smoke_patience": 0,
        "checks": checks,
        "not_a_result": True,
        "formal_gaia_fold_tested": False,
        "test_trajectory": "not_constructed",
        "test_labels_or_gt_read": False,
        "note": ("Bounded real-array smoke only: one epoch on a 60-bin Fit slice, "
                 "short Selection/Generation slices, no Test construction, no formal "
                 "fold training and no experiment metric."),
    }
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print("PASS P6-C1-v2 bounded raw smoke:", args.report)


if __name__ == "__main__":
    main()
