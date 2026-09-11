#!/usr/bin/env python3
"""Run detector-triggered Ada-RCA-G, baselines, and Diagnosis@1/3/5."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.e2e_evaluation import (
    anchor_metric_delta,
    delay_ranking_relationship,
    detector_only_rankings,
    diagnosis_metrics,
    parse_ranking_row,
)
from src.e2e.gaia_rca_adapter import GaiaRcaRawIndex
from src.e2e.protocol import (
    GAIA_SERVICES, load_config, sha256_file, temporal_blocks, write_json,
)
from src.e2e.rca_features import TemporalSpec, extract_case_features_from_indicators, flatten_features
from src.e2e.rca_model import load_conditional_logit, predict_rankings, rca_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="evaluate", choices=("evaluate", "smoke"))
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v1.yaml")
    parser.add_argument("--artifact-root", default="artifacts/p5/i1")
    parser.add_argument("--index-manifest", default="data/p5/i1/rca_raw_index/index_manifest.json")
    parser.add_argument("--model-path", default="data/p5/i1/rca_model/conditional_logit.npz")
    parser.add_argument("--detected-feature-path", default="data/p5/i1/rca_detected_features.npy")
    parser.add_argument("--case-registry", default="artifacts/p5/i1/rca_case_registry.csv")
    parser.add_argument("--event-matching", default=None)
    parser.add_argument("--test-node-predictions", default=None)
    parser.add_argument("--oracle-predictions", default=None)
    parser.add_argument("--root-frequency-predictions", default=None)
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()


def temporal_spec(config: Mapping[str, object]) -> TemporalSpec:
    rca = config["rca"]
    return TemporalSpec(
        window_seconds=int(rca["window_seconds"]),
        bin_seconds=int(rca["bin_seconds"]),
        pre_bins=int(rca["pre_bins"]),
        post_bins=int(rca["post_bins"]),
        n_bins=int(rca["n_bins"]),
        onset_sentinel=float(rca["onset_sentinel_seconds"]),
        post_position_denominator=float(rca["post_position_denominator"]),
    )


def _validate_index_arrays(manifest_path: Path) -> Mapping[str, object]:
    """Fail closed if a persisted raw-index array no longer matches its hash."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent.resolve()
    pairs = []
    for record in manifest["metric_series"]:
        pairs.extend(((record["timestamps"], record["timestamps_sha256"]),
                      (record["values"], record["values_sha256"])))
    for record in manifest["logs"].values():
        pairs.extend(((record["timestamps"], record["timestamps_sha256"]),
                      (record["levels"], record["levels_sha256"])))
    for record in manifest["traces"]["parts"]:
        pairs.extend(((record["timestamps"], record["timestamps_sha256"]),
                      (record["errors"], record["errors_sha256"]),
                      (record["latencies"], record["latencies_sha256"])))
    for relative, expected in pairs:
        path = (root / str(relative)).resolve()
        # Raw-index generations are deliberately nested under
        # ``builds/<build_id>/``.  Keep the traversal guard, but do not
        # require every array to sit directly beside index_manifest.json.
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError("raw-index manifest references a path outside its root")
        if not path.is_file():
            raise ValueError("raw-index manifest references an invalid array path")
        if sha256_file(path) != str(expected):
            raise ValueError("raw-index array checksum mismatch: {}".format(path))
    return {"array_files": len(pairs), "arrays_checksum_verified": True}


def _empty_metrics() -> Mapping[str, object]:
    values = {metric: None for metric in ("AC@1", "AC@3", "AC@5", "Avg@5", "MRR")}
    return {
        "overall": {"case_count": 0, **values},
        "root_macro": {"by_group": {}, "group_count": 0, "macro": values.copy()},
        "fault_macro": {"by_group": {}, "group_count": 0, "macro": values.copy()},
        "case_metrics": [],
        "status": "NO_MATCHED_EVENTS",
    }


def _rankings_for_cases(path: Path, case_ids: Sequence[str]):
    frame = pd.read_csv(path)
    if "case_id" not in frame or frame["case_id"].duplicated().any():
        raise ValueError("ranking artifact lacks unique case IDs: {}".format(path))
    by_id = frame.set_index(frame["case_id"].astype(str), drop=False)
    missing = [case_id for case_id in case_ids if case_id not in by_id.index]
    if missing:
        raise ValueError("ranking artifact is missing matched case IDs: {}".format(missing[:3]))
    return tuple(parse_ranking_row(by_id.loc[case_id]) for case_id in case_ids)


def _prediction_frame(matched: pd.DataFrame, scores: np.ndarray,
                      rankings: Sequence[Sequence[str]], method: str) -> pd.DataFrame:
    columns = [
        "prediction_id", "case_id", "split", "anchor_type", "t_hat", "gt_start_ms",
        "detection_delay_seconds", "gt_service", "fault_type", "method", "labelled_rank",
        "ranking_json",
    ] + ["rank_{}".format(i) for i in range(1, 11)] + [
        "score_{}".format(service) for service in GAIA_SERVICES
    ]
    if len(matched) == 0:
        return pd.DataFrame(columns=columns)
    if scores.shape != (len(matched), len(GAIA_SERVICES)) or len(rankings) != len(matched):
        raise ValueError("matched prediction arrays are not aligned")
    rows = []
    for position, source in enumerate(matched.itertuples(index=False)):
        ranking = tuple(rankings[position])
        root = str(source.gt_service)
        row = {
            "prediction_id": str(source.prediction_id),
            "case_id": str(source.case_id),
            "split": "test",
            "anchor_type": "detected t_hat",
            "t_hat": int(source.t_hat),
            "gt_start_ms": int(source.gt_start_ms),
            "detection_delay_seconds": float(source.detection_delay_seconds),
            "gt_service": root,
            "fault_type": str(source.fault_type),
            "method": method,
            "labelled_rank": int(ranking.index(root) + 1),
            "ranking_json": json.dumps(ranking, separators=(",", ":")),
        }
        row.update({"rank_{}".format(i + 1): value for i, value in enumerate(ranking)})
        row.update({
            "score_{}".format(service): float(scores[position, service_index])
            for service_index, service in enumerate(GAIA_SERVICES)
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _metrics(rankings, matched):
    if not len(matched):
        return _empty_metrics()
    roots = matched["gt_service"].map({name: i for i, name in enumerate(GAIA_SERVICES)})
    if roots.isna().any():
        raise ValueError("matched event contains an unknown labelled service")
    return rca_metrics(
        rankings,
        roots.to_numpy(dtype=np.int64),
        GAIA_SERVICES,
        matched["fault_type"].astype(str).tolist(),
    )


def _purge_rca_ineligible_matching(
    matching: pd.DataFrame,
    case_registry: pd.DataFrame,
    config: Mapping[str, object],
):
    """Apply the frozen GT-anchor and detected-anchor W300 boundary purge."""

    eligible_ids = set(case_registry.loc[
        case_registry["split"].astype(str) == "test", "case_id"
    ].astype(str))
    status = matching["match_status"].astype(str)
    has_gt = status.isin(("matched", "miss"))
    gt_ineligible = has_gt & ~matching["case_id"].astype(str).isin(eligible_ids)
    retained = matching.loc[~gt_ineligible].copy()
    test_block = next(block for block in temporal_blocks(config) if block.name == "test")
    radius_ms = int(config["rca"]["window_seconds"]) * 1000
    detected_crossing = pd.Series(False, index=retained.index)
    for index, row in retained.loc[
        retained["match_status"].astype(str) == "matched"
    ].iterrows():
        anchor = int(row["t_hat"])
        detected_crossing.loc[index] = not test_block.contains_interval(
            anchor - radius_ms, anchor + radius_ms
        )
    detected_ids = retained.loc[detected_crossing, "prediction_id"].astype(str).tolist()
    retained = retained.loc[~detected_crossing].reset_index(drop=True)
    return retained, {
        "gt_w300_ineligible_rows": int(gt_ineligible.sum()),
        "gt_w300_ineligible_case_ids": sorted(
            matching.loc[gt_ineligible, "case_id"].astype(str).unique().tolist()
        ),
        "detected_anchor_w300_crossing_cases": int(len(detected_ids)),
        "detected_anchor_w300_crossing_prediction_ids": sorted(detected_ids),
        "rule": "exclude GT or detected-anchor RCA context crossing the chronological Test boundary",
    }


def evaluate(config: Mapping[str, object], artifact_root: Path, index_manifest: Path,
             model_path: Path, detected_feature_path: Path, matching_path: Path,
             node_path: Path, oracle_path: Path, frequency_path: Path,
             case_registry_path: Path):
    source_paths = (
        index_manifest, model_path, model_path.with_suffix(".json"), matching_path,
        node_path, oracle_path, frequency_path, case_registry_path,
        artifact_root / "rca_metrics.json",
    )
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("full E2E prerequisites are missing: {}".format(missing))
    index_check = _validate_index_arrays(index_manifest)
    matching_all = pd.read_csv(matching_path)
    if "split" not in matching_all:
        raise ValueError("event matching artifact lacks split")
    full_test_matching = matching_all.loc[
        matching_all["split"].astype(str) == "test"
    ].copy()
    statuses = set(full_test_matching["match_status"].astype(str))
    if not statuses.issubset({"matched", "false_alarm", "miss"}):
        raise ValueError("event matching artifact contains an unknown status")
    case_registry = pd.read_csv(case_registry_path)
    if not {"case_id", "split"}.issubset(case_registry.columns):
        raise ValueError("RCA case registry lacks case_id/split")
    matching, boundary_purge = _purge_rca_ineligible_matching(
        full_test_matching, case_registry, config
    )
    matched = matching.loc[matching["match_status"].astype(str) == "matched"].copy()
    matched = matched.sort_values(["t_hat", "prediction_id"], kind="stable").reset_index(drop=True)
    if matched["prediction_id"].duplicated().any() or matched["case_id"].duplicated().any():
        raise ValueError("one-to-one event matching identity constraint is violated")

    model = load_conditional_logit(model_path)
    spec = temporal_spec(config)
    detected_feature_path.parent.mkdir(parents=True, exist_ok=True)
    if len(matched):
        index = GaiaRcaRawIndex.from_manifest(index_manifest)
        features = np.lib.format.open_memmap(
            detected_feature_path, mode="w+", dtype=np.float32,
            shape=(len(matched), len(GAIA_SERVICES), int(config["rca"]["feature_dimension"])),
        )
        for position, row in enumerate(matched.itertuples(index=False)):
            # Only the detector anchor is passed to telemetry construction.
            # The labelled service/fault type remain outside this call.
            indicators = index.case_indicators(int(row.t_hat), spec)
            value = extract_case_features_from_indicators(
                str(row.prediction_id), GAIA_SERVICES, indicators, spec
            )
            z2 = flatten_features(value, "z2")
            if z2.shape != (10, 68) or not np.all(np.isfinite(z2)):
                raise ValueError("detected-anchor RCA feature row is invalid")
            features[position] = z2.astype(np.float32)
        features.flush()
        feature_values = np.asarray(features, dtype=np.float64)
        detected_scores = model.scores(feature_values)
        detected_rankings = predict_rankings(feature_values, model, GAIA_SERVICES)
        detected_feature_hash = sha256_file(detected_feature_path)
    else:
        np.save(detected_feature_path, np.empty((0, 10, 68), dtype=np.float32), allow_pickle=False)
        detected_scores = np.empty((0, 10), dtype=float)
        detected_rankings = tuple()
        detected_feature_hash = sha256_file(detected_feature_path)

    node_predictions = pd.read_csv(node_path)
    detector_records = detector_only_rankings(matching, node_predictions)
    detector_rankings = tuple(
        detector_records[str(row.prediction_id)]["ranking"]
        for row in matched.itertuples(index=False)
    )
    detector_scores = np.asarray([
        [detector_records[str(row.prediction_id)]["scores"][service] for service in GAIA_SERVICES]
        for row in matched.itertuples(index=False)
    ], dtype=float).reshape(len(matched), len(GAIA_SERVICES))

    case_ids = matched["case_id"].astype(str).tolist()
    oracle_rankings = _rankings_for_cases(oracle_path, case_ids) if case_ids else tuple()
    frequency_rankings = _rankings_for_cases(frequency_path, case_ids) if case_ids else tuple()
    detected_metrics = _metrics(detected_rankings, matched)
    detector_metrics = _metrics(detector_rankings, matched)
    oracle_metrics = _metrics(oracle_rankings, matched)
    frequency_metrics = _metrics(frequency_rankings, matched)

    detected_frame = _prediction_frame(
        matched, detected_scores, detected_rankings, "Ada-RCA-G"
    )
    detector_frame = _prediction_frame(
        matched, detector_scores, detector_rankings, "Detector-only"
    )
    detected_path = artifact_root / "rca_detected_predictions.csv"
    detector_path = artifact_root / "detector_only_predictions.csv"
    detected_frame.to_csv(detected_path, index=False)
    detector_frame.to_csv(detector_path, index=False)

    rankings_by_prediction = {
        str(row.prediction_id): detected_rankings[position]
        for position, row in enumerate(matched.itertuples(index=False))
    }
    diagnosis = diagnosis_metrics(matching, rankings_by_prediction)
    diagnosis.update({
        "schema_version": "p5_i1_e2e_diagnosis_metrics_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FORMAL_FULL_DATA",
        "git_commit": git_head(),
        "random_seed": int(config["random_seed"]),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml"),
        "evaluation_population": "W300-eligible Test GT and detected anchors after chronological boundary purge",
        "event_detection_population": {
            "matching_rows_before_rca_boundary_purge": int(len(full_test_matching)),
            "note": "event_detection_metrics.json remains evaluated on all complete Test injections",
        },
        "rca_boundary_purge": boundary_purge,
        "source_artifacts": {
            path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in source_paths[:-1]
        },
        "detected_feature_file": {
            "path": str(detected_feature_path.resolve()),
            "sha256": detected_feature_hash,
            "shape": [int(len(matched)), 10, 68],
        },
        "output_artifacts": {
            "rca_detected_predictions.csv": sha256_file(detected_path),
            "detector_only_predictions.csv": sha256_file(detector_path),
        },
    })
    write_json(artifact_root / "e2e_diagnosis_metrics.json", diagnosis)

    rca_summary_path = artifact_root / "rca_metrics.json"
    rca_summary = json.loads(rca_summary_path.read_text(encoding="utf-8"))
    rca_summary["detected_anchor_matched_test"] = detected_metrics
    rca_summary["oracle_anchor_matched_test"] = oracle_metrics
    rca_summary["detector_only_matched_test"] = detector_metrics
    rca_summary["root_frequency_matched_test"] = frequency_metrics
    rca_summary["anchor_degradation"] = (
        anchor_metric_delta(oracle_metrics, detected_metrics) if len(matched) else {
            "status": "NO_MATCHED_EVENTS", "delta_detected_minus_oracle": None
        }
    )
    rca_summary["detection_delay_relationship"] = delay_ranking_relationship(
        matching, rankings_by_prediction
    )
    rca_summary["detected_anchor_artifact"] = {
        "path": str(detected_path.resolve()), "sha256": sha256_file(detected_path)
    }
    rca_summary["detector_only_artifact"] = {
        "path": str(detector_path.resolve()), "sha256": sha256_file(detector_path)
    }
    write_json(rca_summary_path, rca_summary)
    return {"rca_metrics": rca_summary, "e2e_diagnosis_metrics": diagnosis,
            "raw_index_check": index_check}


def smoke(config: Mapping[str, object], artifact_root: Path):
    matching = pd.DataFrame([
        {"split": "test", "prediction_id": "p0", "t_hat": 30_000,
         "match_status": "matched", "case_id": "c0", "gt_service": "dbservice1",
         "fault_type": "login failure", "gt_start_ms": 20_000,
         "detection_delay_seconds": 10.0},
        {"split": "test", "prediction_id": "p1", "t_hat": 90_000,
         "match_status": "matched", "case_id": "c1", "gt_service": "webservice2",
         "fault_type": "memory_anomalies", "gt_start_ms": 100_000,
         "detection_delay_seconds": -10.0},
        {"split": "test", "prediction_id": "p2", "t_hat": 150_000,
         "match_status": "false_alarm", "case_id": None, "gt_service": None,
         "fault_type": None, "gt_start_ms": None, "detection_delay_seconds": None},
        {"split": "test", "prediction_id": None, "t_hat": None,
         "match_status": "miss", "case_id": "c2", "gt_service": "logservice1",
         "fault_type": "cpu_anomalies", "gt_start_ms": 210_000,
         "detection_delay_seconds": None},
    ])
    node_rows = []
    for anchor in (30_000, 90_000):
        for index, service in enumerate(GAIA_SERVICES):
            node_rows.append({
                "split": "test", "prediction_timestamp": anchor, "service": service,
                "anomaly_score": 1.0 if index == (0 if anchor == 30_000 else 9) else 0.0,
            })
    detector = detector_only_rankings(matching, pd.DataFrame(node_rows))
    rankings = {key: value["ranking"] for key, value in detector.items()}
    summary = {
        "schema_version": "p5_i1_e2e_smoke_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "formal_result": False,
        "fixture": "synthetic only",
        "git_commit": git_head(),
        "random_seed": int(config["random_seed"]),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml"),
        "detector_only_rankings": {key: list(value) for key, value in rankings.items()},
        "diagnosis": diagnosis_metrics(matching, rankings),
        "delay_relationship": delay_ranking_relationship(matching, rankings),
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "e2e_smoke_summary.json", summary)
    return summary


def main():
    args = parse_args()
    config = load_config((PROJECT_ROOT / args.config).resolve())
    artifact_root = (PROJECT_ROOT / args.artifact_root).resolve()
    if args.action == "smoke":
        result = smoke(config, artifact_root)
    else:
        result = evaluate(
            config,
            artifact_root,
            (PROJECT_ROOT / args.index_manifest).resolve(),
            (PROJECT_ROOT / args.model_path).resolve(),
            (PROJECT_ROOT / args.detected_feature_path).resolve(),
            Path(args.event_matching or artifact_root / "event_matching.csv").resolve(),
            Path(args.test_node_predictions or artifact_root / "ad_test_predictions.csv").resolve(),
            Path(args.oracle_predictions or artifact_root / "rca_oracle_predictions.csv").resolve(),
            Path(args.root_frequency_predictions or artifact_root / "root_frequency_predictions.csv").resolve(),
            (PROJECT_ROOT / args.case_registry).resolve(),
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
