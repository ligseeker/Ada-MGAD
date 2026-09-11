#!/usr/bin/env python3
"""Build the GAIA raw RCA index and materialize frozen W300-B15 Z2 features."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.e2e.gaia_rca_adapter import GaiaRcaRawIndex, IndexedSeries, build_raw_index
from src.e2e.protocol import GAIA_SERVICES, load_config, sha256_file, write_json
from src.e2e.rca_features import (
    TemporalSpec,
    extract_case_features_from_indicators,
    flatten_features,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("index", "materialize", "smoke", "all"))
    parser.add_argument("--config", default="configs/e2e/gaia_p5_v1.yaml")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--index-root", default="data/p5/i1/rca_raw_index")
    parser.add_argument("--feature-root", default="data/p5/i1/rca_features")
    parser.add_argument("--artifact-root", default="artifacts/p5/i1")
    parser.add_argument("--case-registry", default="artifacts/p5/i1/rca_case_registry.csv")
    parser.add_argument("--chunk-rows", default=500000, type=int)
    parser.add_argument("--limit-cases", default=0, type=int,
                        help="Smoke/debug limit only; zero is required for formal artifacts.")
    return parser.parse_args()


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()


def temporal_spec(config):
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


def _health_from_arrays(features, counters):
    rows = np.asarray(features).reshape(-1, features.shape[-1])
    finite = np.isfinite(rows)
    constant = np.zeros(rows.shape[1], dtype=bool)
    finite_columns = np.all(finite, axis=0)
    constant[finite_columns] = np.ptp(rows[:, finite_columns], axis=0) <= 1e-12
    signatures = {
        __import__("hashlib").sha256(np.asarray(row, dtype="<f4").tobytes()).digest()
        for row in rows
    }
    candidate_rows = int(len(rows))
    return {
        "purpose": "implementation sanity check only; no representation selection",
        "cases": int(features.shape[0]),
        "candidate_rows": candidate_rows,
        "feature_dimension": int(features.shape[-1]),
        "finite_ratio": float(finite.mean()),
        "channel_available_ratio": float(counters["available_sum"] / counters["available_size"]),
        "mask_coverage": float(counters["mask_sum"] / counters["mask_size"]),
        "morphology_active_ratio": float(counters["active_sum"] / counters["active_size"]),
        "all_zero_rows": int(np.all(np.isclose(rows, 0.0), axis=1).sum()),
        "all_zero_row_ratio": float(np.all(np.isclose(rows, 0.0), axis=1).mean()),
        "all_masked_rows": int(counters["all_masked"]),
        "all_masked_row_ratio": float(counters["all_masked"] / candidate_rows),
        "constant_features": int(constant.sum()),
        "constant_feature_ratio": float(constant.mean()),
        "unique_candidate_signatures": int(len(signatures)),
        "candidate_signature_duplication": int(candidate_rows - len(signatures)),
        "candidate_signature_duplication_ratio": float(
            (candidate_rows - len(signatures)) / candidate_rows
        ),
    }


def materialize(config, index_root, feature_root, artifact_root, case_path, limit_cases=0):
    import pandas as pd

    cases = pd.read_csv(case_path)
    required = {"case_id", "start_ms", "split"}
    if not required.issubset(cases.columns):
        raise ValueError("RCA case registry lacks prediction-visible columns")
    formal = int(limit_cases) == 0
    if limit_cases:
        cases = cases.iloc[:int(limit_cases)].copy()
    feature_root.mkdir(parents=True, exist_ok=True)
    spec = temporal_spec(config)
    index_manifest_path = index_root / "index_manifest.json"
    index = GaiaRcaRawIndex.from_manifest(index_manifest_path)
    feature_path = feature_root / "z2_features.npy"
    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32,
        shape=(len(cases), len(GAIA_SERVICES), int(config["rca"]["feature_dimension"])),
    )
    counters = {
        "available_sum": 0, "available_size": 0,
        "mask_sum": 0, "mask_size": 0,
        "active_sum": 0, "active_size": 0,
        "all_masked": 0,
    }
    for case_index, row in enumerate(cases.itertuples(index=False)):
        indicators = index.case_indicators(int(row.start_ms), spec)
        value = extract_case_features_from_indicators(
            str(row.case_id), GAIA_SERVICES, indicators, spec
        )
        z2 = flatten_features(value, "z2")
        if z2.shape != (len(GAIA_SERVICES), 68) or not np.isfinite(z2).all():
            raise ValueError("invalid W300-B15 Z2 feature shape or value")
        features[case_index] = z2.astype(np.float32)
        counters["available_sum"] += int(value.base[:, :, 7].sum())
        counters["available_size"] += int(value.base[:, :, 7].size)
        counters["mask_sum"] += int(value.q_mask.sum())
        counters["mask_size"] += int(value.q_mask.size)
        counters["active_sum"] += int(value.morphology_active.sum())
        counters["active_size"] += int(value.morphology_active.size)
        counters["all_masked"] += int((~np.any(value.q_mask.astype(bool), axis=(1, 2))).sum())
    features.flush()
    health = _health_from_arrays(features, counters)
    health.update({
        "schema_version": "p5_i1_rca_feature_health_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FORMAL" if formal else "NON_FORMAL_LIMITED_SMOKE",
        "git_commit": git_head(),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml"),
        "index_manifest_sha256": sha256_file(index_manifest_path),
        "case_registry_sha256": sha256_file(case_path),
    })
    health_name = "rca_feature_health.json" if formal else "rca_feature_smoke_health.json"
    write_json(artifact_root / health_name, health)
    case_ids_path = feature_root / "case_ids.npy"
    splits_path = feature_root / "splits.npy"
    anchors_path = feature_root / "anchors_ms.npy"
    np.save(case_ids_path, cases["case_id"].astype(str).to_numpy(dtype="U32"), allow_pickle=False)
    np.save(splits_path, cases["split"].astype(str).to_numpy(dtype="U10"), allow_pickle=False)
    np.save(anchors_path, cases["start_ms"].to_numpy(dtype=np.int64), allow_pickle=False)
    manifest = {
        "schema_version": "p5_i1_rca_feature_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FORMAL" if formal else "NON_FORMAL_LIMITED_SMOKE",
        "git_commit": git_head(),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml"),
        "source_commit": config["source"]["ada_rca_commit"],
        "source_files": config["source"]["ada_rca_files"],
        "difference_from_canonical": "temporal constants parameterized to W300-B15 only",
        "feature_variant": "z2",
        "shape": list(features.shape),
        "files": {
            "features": {"path": str(feature_path.resolve()), "sha256": sha256_file(feature_path)},
            "case_ids": {"path": str(case_ids_path.resolve()), "sha256": sha256_file(case_ids_path)},
            "splits": {"path": str(splits_path.resolve()), "sha256": sha256_file(splits_path)},
            "anchors_ms": {"path": str(anchors_path.resolve()), "sha256": sha256_file(anchors_path)},
        },
        "label_firewall": "feature construction received case_id/start_ms/split only; service and fault labels remain in the separate registry",
        "health_artifact": str((artifact_root / health_name).resolve()),
    }
    manifest_name = "rca_feature_manifest.json" if formal else "rca_feature_smoke_manifest.json"
    write_json(artifact_root / manifest_name, manifest)
    return manifest


def smoke(config, feature_root, artifact_root):
    spec = temporal_spec(config)
    anchor = 1_625_108_900_000
    starts = anchor - spec.window_seconds * 1000 + np.arange(spec.n_bins) * spec.bin_seconds * 1000
    metric = []
    logs = {}
    traces = {}
    for service_index, service in enumerate(GAIA_SERVICES):
        values = np.arange(spec.n_bins, dtype=float) + service_index
        values[spec.pre_bins:spec.pre_bins + 2] += 100.0
        metric.append(IndexedSeries(service, "smoke_metric", starts, values))
        logs[service] = (starts, np.mod(np.arange(spec.n_bins), 5).astype(np.uint8))
        traces[service] = [(starts, np.arange(spec.n_bins) % 7 == 0, values / 100.0)]
    index = GaiaRcaRawIndex(metric, logs, traces)
    feature_root.mkdir(parents=True, exist_ok=True)
    index_manifest = {
        "schema_version": "p5_i1_smoke_in_memory_index_v1",
        "metric_series": [], "logs": {}, "traces": {"parts": []},
    }
    write_json(feature_root / "index_manifest.json", index_manifest)
    indicators = index.case_indicators(anchor, spec)
    value = extract_case_features_from_indicators("smoke-case", GAIA_SERVICES, indicators, spec)
    z2 = flatten_features(value, "z2")
    health = {
        "schema_version": "p5_i1_rca_feature_smoke_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "git_commit": git_head(),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml"),
        "shape": list(z2.shape),
        "finite": bool(np.isfinite(z2).all()),
        "n_bins": spec.n_bins,
        "pre_bins": spec.pre_bins,
        "post_bins": spec.post_bins,
        "onset_sentinel": spec.onset_sentinel,
        "post_position_denominator": spec.post_position_denominator,
        "trace_error_rule": config["rca"]["trace_error_rule"],
    }
    if health["shape"] != [10, 68] or not health["finite"]:
        raise ValueError("RCA adapter smoke failed")
    write_json(artifact_root / "rca_feature_smoke_summary.json", health)
    return health


def main():
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)
    raw_root = Path(args.raw_root or config["gaia_raw_root"]).resolve()
    index_root = (PROJECT_ROOT / args.index_root).resolve()
    feature_root = (PROJECT_ROOT / args.feature_root).resolve()
    artifact_root = (PROJECT_ROOT / args.artifact_root).resolve()
    case_path = (PROJECT_ROOT / args.case_registry).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    result = None
    if args.action in ("index", "all"):
        timing = json.loads(
            (PROJECT_ROOT / "artifacts/p5/g0r2/raw_telemetry_timing.json").read_text()
        )
        result = build_raw_index(
            raw_root, index_root,
            timing["source_binding"]["current_inventory"],
            chunk_rows=args.chunk_rows,
            provenance={
                "git_commit": git_head(),
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
                "random_seed": int(config["random_seed"]),
            },
        )
    if args.action in ("materialize", "all"):
        result = materialize(
            config, index_root, feature_root, artifact_root, case_path,
            limit_cases=args.limit_cases,
        )
    if args.action == "smoke":
        result = smoke(config, feature_root, artifact_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
