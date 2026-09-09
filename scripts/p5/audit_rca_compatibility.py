#!/usr/bin/env python3
"""Audit GAIA compatibility with frozen Ada-RCA without fitting a ranker."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p5.common import sha256_file, write_json  # noqa: E402


SCHEMA_VERSION = "p5_g0r2_rca_compatibility_v1"
CHANNELS = ("metric", "log", "trace-error", "trace-latency")


def _read_registry(path: Path) -> Sequence[Mapping[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return tuple(
            {
                **row,
                "source_index": int(row["source_index"]),
                "start_ms": int(row["start_ms"]),
                "end_ms": int(row["end_ms"]),
            }
            for row in csv.DictReader(handle)
        )


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def _group_intervals(
    rows: Sequence[Mapping[str, object]], radius_seconds: int, include_injection: bool
) -> Mapping[str, object]:
    intervals = []
    radius_ms = radius_seconds * 1000
    for index, row in enumerate(rows):
        anchor = int(row["start_ms"])
        start = anchor - radius_ms
        end = anchor + radius_ms
        if include_injection:
            start = min(start, int(row["start_ms"]))
            end = max(end, int(row["end_ms"]))
        intervals.append((index, start, end))
    parent = list(range(len(rows)))
    degrees = [0] * len(rows)
    pair_count = same_root_pairs = different_root_pairs = 0

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    active = []
    for index, start, end in sorted(intervals, key=lambda value: (value[1], value[2], value[0])):
        active = [(other, other_end) for other, other_end in active if other_end > start]
        for other, _ in active:
            pair_count += 1
            degrees[index] += 1
            degrees[other] += 1
            union(index, other)
            if rows[index]["service"] == rows[other]["service"]:
                same_root_pairs += 1
            else:
                different_root_pairs += 1
        active.append((index, end))
    components = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    sizes = [len(values) for values in components.values()]
    size_distribution = Counter(sizes)
    multi_root_groups = [
        values
        for values in components.values()
        if len({str(rows[index]["service"]) for index in values}) > 1
    ]
    multi_fault_groups = [
        values
        for values in components.values()
        if len({str(rows[index]["fault_type"]) for index in values}) > 1
    ]
    return {
        "interval_kind": "injection union context" if include_injection else "context only",
        "interval_semantics": "half-open; boundary touching is not overlap",
        "groups": len(components),
        "non_singleton_groups": sum(size > 1 for size in sizes),
        "cases_in_non_singleton_groups": sum(size for size in sizes if size > 1),
        "largest_group": max(sizes, default=0),
        "group_size_distribution": {str(key): value for key, value in sorted(size_distribution.items())},
        "direct_overlap_pairs": pair_count,
        "same_root_direct_pairs": same_root_pairs,
        "different_root_direct_pairs": different_root_pairs,
        "different_root_direct_pair_ratio": different_root_pairs / pair_count if pair_count else None,
        "multi_root_groups": len(multi_root_groups),
        "cases_in_multi_root_groups": sum(len(values) for values in multi_root_groups),
        "multi_fault_groups": len(multi_fault_groups),
        "cases_in_multi_fault_groups": sum(len(values) for values in multi_fault_groups),
        "maximum_direct_overlap_degree": max(degrees, default=0),
    }


def _read_index(path: Path) -> list[tuple[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append((str(row["case_id"]), str(row["service"])))
    return rows


def _load_proxy_bundle(directory: Path):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("index.jsonl", "values.npy", "observed.npy"):
        if sha256_file(directory / name) != manifest["files"][name]["sha256"]:
            raise ValueError("historical feature bundle checksum mismatch: {}".format(directory / name))
    return {
        "directory": str(directory.resolve()),
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "index": _read_index(directory / "index.jsonl"),
        "values": np.load(directory / "values.npy", mmap_mode="r"),
        "observed": np.load(directory / "observed.npy", mmap_mode="r"),
    }


def _feature_stats(values: np.ndarray, observed: np.ndarray, names: Sequence[str]) -> Mapping[str, object]:
    rows = []
    for index, name in enumerate(names):
        mask = np.asarray(observed[:, index], dtype=bool)
        selected = np.asarray(values[:, index], dtype=float)[mask]
        finite = selected[np.isfinite(selected)]
        variance = float(np.var(finite)) if finite.size else None
        rows.append(
            {
                "feature": name,
                "observed_ratio": float(np.mean(mask)),
                "missing_masked_ratio": float(1.0 - np.mean(mask)),
                "finite_ratio_among_observed": float(finite.size / selected.size) if selected.size else None,
                "variance_among_observed": variance,
                "constant": variance is None or variance <= 1e-12,
                "near_constant": variance is None or variance <= 1e-8,
            }
        )
    return {
        "feature_count": len(rows),
        "constant_feature_ratio": sum(row["constant"] for row in rows) / len(rows) if rows else None,
        "near_constant_feature_ratio": sum(row["near_constant"] for row in rows) / len(rows) if rows else None,
        "features": rows,
    }


def _channel_health(values: np.ndarray, observed: np.ndarray, names: Sequence[str]) -> Mapping[str, object]:
    finite_values = np.isfinite(np.asarray(values, dtype=float))
    row_observed = np.any(observed, axis=1)
    row_all_masked = ~row_observed
    row_all_zero = np.all(np.where(observed, values, 0.0) == 0.0, axis=1)
    row_active_proxy = np.any(observed & finite_values & (np.abs(values) > 1e-12), axis=1)
    return {
        "channel_available_proxy_ratio": float(np.mean(row_observed)),
        "coverage_observed_cell_ratio": float(np.mean(observed)),
        "morphology_active_proxy_ratio": float(np.mean(row_active_proxy)),
        "valid_service_row_ratio": float(np.mean(row_observed & np.all(~observed | finite_values, axis=1))),
        "all_zero_ratio": float(np.mean(row_all_zero)),
        "all_masked_ratio": float(np.mean(row_all_masked)),
        "finite_ratio": float(np.mean(finite_values)),
        "feature_health": _feature_stats(values, observed, names),
    }


def _proxy_representation_health(history_repo: Path) -> Mapping[str, object]:
    directories = {
        "metric": history_repo / "artifacts/p2/features/gaia_main/p2_metric_summary_v1",
        "log": history_repo / "artifacts/p2/event_features/gaia_main/p2_log_l0_v1",
        "trace": history_repo / "artifacts/p2/event_features/gaia_main/p2_trace_t0_v1",
    }
    bundles = {key: _load_proxy_bundle(value) for key, value in directories.items()}
    indices = [bundle["index"] for bundle in bundles.values()]
    if any(index != indices[0] for index in indices[1:]):
        raise ValueError("historical proxy bundles do not share case/service row ordering")

    selected = {}
    for channel in CHANNELS:
        if channel == "metric":
            bundle = bundles["metric"]
            positions = [i for i, name in enumerate(bundle["manifest"]["feature_names"]) if name.startswith("whole.")]
        elif channel == "log":
            bundle = bundles["log"]
            positions = [i for i, name in enumerate(bundle["manifest"]["feature_names"]) if name.startswith("whole.")]
        elif channel == "trace-error":
            bundle = bundles["trace"]
            positions = [i for i, name in enumerate(bundle["manifest"]["feature_names"]) if name == "whole.error_fraction_shift"]
        else:
            bundle = bundles["trace"]
            positions = [i for i, name in enumerate(bundle["manifest"]["feature_names"]) if name.startswith("whole.duration_")]
        names = [bundle["manifest"]["feature_names"][i] for i in positions]
        values = np.asarray(bundle["values"][:, positions], dtype=float)
        observed = np.asarray(bundle["observed"][:, positions], dtype=bool)
        selected[channel] = {"values": values, "observed": observed, "names": names}

    combined = np.concatenate([selected[channel]["values"] for channel in CHANNELS], axis=1)
    combined_mask = np.concatenate([selected[channel]["observed"] for channel in CHANNELS], axis=1)
    case_ids = [case_id for case_id, _ in indices[0]]
    grouped_positions = defaultdict(list)
    for position, case_id in enumerate(case_ids):
        grouped_positions[case_id].append(position)
    all_identical_cases = 0
    any_identical_cases = 0
    pair_identical = 0
    pair_total = 0
    discrimination = []
    effective_candidates = []
    for positions in grouped_positions.values():
        signatures = []
        for position in positions:
            signature = (
                np.asarray(combined[position], dtype=np.float64).tobytes()
                + np.asarray(combined_mask[position], dtype=np.bool_).tobytes()
            )
            signatures.append(hashlib.sha256(signature).hexdigest())
        unique = len(set(signatures))
        discrimination.append(unique / len(positions))
        effective_candidates.append(
            int(sum(bool(np.any(combined_mask[position])) for position in positions))
        )
        all_identical_cases += int(unique == 1)
        case_has_pair = False
        for left, right in itertools.combinations(range(len(signatures)), 2):
            pair_total += 1
            if signatures[left] == signatures[right]:
                pair_identical += 1
                case_has_pair = True
        any_identical_cases += int(case_has_pair)

    return {
        "status": "HISTORICAL EVIDENCE REBOUND TO CURRENT RAW LAYOUT; NON-68D PROXY",
        "cohort": "historical 13,470-case anchor-unique-root GAIA main cohort",
        "window_seconds": 300,
        "bin_size": "stage summaries, not frozen 15s/30s Z2 bins",
        "channels": {
            channel: _channel_health(
                selected[channel]["values"],
                selected[channel]["observed"],
                selected[channel]["names"],
            )
            for channel in CHANNELS
        },
        "per_case": {
            "case_count": len(grouped_positions),
            "candidate_count": len(indices[0]) // len(grouped_positions),
            "effective_candidate_count": {
                "min": min(effective_candidates),
                "median": float(np.median(effective_candidates)),
                "mean": float(np.mean(effective_candidates)),
                "max": max(effective_candidates),
            },
            "all_candidates_identical_case_ratio": all_identical_cases / len(grouped_positions),
            "any_identical_candidate_pair_case_ratio": any_identical_cases / len(grouped_positions),
            "identical_candidate_pair_ratio": pair_identical / pair_total,
            "candidate_discrimination_unique_signature_ratio": {
                "min": float(np.min(discrimination)),
                "median": float(np.median(discrimination)),
                "mean": float(np.mean(discrimination)),
                "max": float(np.max(discrimination)),
            },
        },
        "source_bundles": {
            key: {
                "directory": bundle["directory"],
                "manifest_sha256": bundle["manifest_sha256"],
                "case_count": bundle["manifest"]["case_count"],
                "service_row_count": bundle["manifest"]["service_row_count"],
                "feature_count": bundle["manifest"]["feature_count"],
            }
            for key, bundle in bundles.items()
        },
        "limitations": [
            "This is not an exact Ada-RCA 68D Z2 extraction.",
            "Trace error and latency proxies select only whole.error_fraction_shift and whole.duration_* fields.",
            "The historical trace error proxy used status >=500 and is semantically incompatible with the revalidated official !=200 rule.",
            "No W600 proxy bundle exists; W600 feature health is NOT COMPUTED.",
        ],
    }


def build(
    event_rows: Sequence[Mapping[str, object]],
    raw_telemetry: Mapping[str, object],
    ada_rca_repo: Path,
    history_repo: Path,
) -> Mapping[str, object]:
    spec_path = ada_rca_repo / "docs/REPRESENTATION_FREEZE.md"
    features_path = ada_rca_repo / "src/rca/features.py"
    final_path = ada_rca_repo / "src/rca/final_method.py"
    contexts = {}
    for radius in (300, 600):
        contexts[str(radius)] = {
            "context_only": _group_intervals(event_rows, radius, False),
            "injection_union_context": _group_intervals(event_rows, radius, True),
            "telemetry_coverage": {
                modality: raw_telemetry["historical_full_scan_coverage"][modality][str(radius)]
                for modality in ("metrics", "logs", "traces")
            },
            "eligible_registry_coverage": {
                "cases": len(event_rows),
                "roots": len({row["service"] for row in event_rows}),
                "fault_types": len({row["fault_type"] for row in event_rows}),
            },
        }

    cadence_gap = raw_telemetry["metrics"]["inferred_sampling_gap_ratio"]
    configs = {}
    for window_seconds in (600, 300):
        for bin_seconds in (15, 30):
            key = "W{}_B{}".format(window_seconds, bin_seconds)
            configs[key] = {
                "window_seconds_each_side": window_seconds,
                "bin_seconds": bin_seconds,
                "bins_total": 2 * window_seconds // bin_seconds,
                "metric_native_cadence_seconds": 30,
                "metric_structural_occupancy_upper_bound_without_fill": min(1.0, bin_seconds / 30.0),
                "metric_cadence_gap_adjusted_occupancy_reference": min(1.0, bin_seconds / 30.0) * (1.0 - cadence_gap),
                "exact_68d_representation_built": False,
                "exact_health_status": "NOT COMPUTED: GAIA four-channel materializer is not frozen",
                "full_scan_activity_coverage_available": True,
                "w300_non_68d_proxy_available": window_seconds == 300,
                "selection_basis": "coverage, contamination, representation health, and split feasibility only",
            }

    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "REVALIDATED FACT plus explicitly labelled proxy/inference",
        "ada_rca_repository": {
            "path": str(ada_rca_repo.resolve()),
            "branch": _git(ada_rca_repo, "branch", "--show-current"),
            "head": _git(ada_rca_repo, "rev-parse", "HEAD"),
            "working_tree": _git(ada_rca_repo, "status", "--short") or "clean",
            "canonical_specification": str(spec_path.resolve()),
            "canonical_specification_sha256": sha256_file(spec_path),
            "feature_extractor": str(features_path.resolve()),
            "feature_extractor_sha256": sha256_file(features_path),
            "final_method": str(final_path.resolve()),
            "final_method_sha256": sha256_file(final_path),
        },
        "frozen_contract": {
            "raw_modalities": ["Metrics", "Logs", "Traces"],
            "feature_channels": ["Metric", "Log", "Trace Error", "Trace Latency"],
            "terminology": "three raw modalities, four derived feature channels; never four modalities",
            "window": "half-open [t0-600s, t0+600s)",
            "bin_seconds": 15,
            "bins": 80,
            "z2_dimension": 68,
            "normalization": "case-local indicator-local pre-event median/MAD with IQR fallback",
            "aggregation": "service-channel Q90 of finite absolute normalized deviations",
            "missingness": "mask-preserving; no interpolation/fill; numeric zero is observed zero",
        },
        "context_overlap": contexts,
        "configuration_health": configs,
        "representation_health_proxy": _proxy_representation_health(history_repo),
        "raw_vs_processed": {
            "option_a_ada_mgad_processed": {
                "verdict": "NOT RECOMMENDED",
                "information_loss": [
                    "30s aggregation removes native millisecond log and microsecond trace timing",
                    "raw log text/schema is replaced by Ada-MGAD-specific template/count representation",
                    "trace status and latency are aggregated for detector edges rather than frozen RCA scalar channels",
                    "parent/span identifiers and dynamic trace structure are not preserved as raw evidence",
                    "Ada-MGAD node tensors do not satisfy Ada-RCA's four derived CSV channel contract",
                ],
            },
            "option_b_independent_raw_event_relative": {
                "verdict": "RECOMMENDED",
                "minimum_adapter": [
                    "long metric shards -> label-free <candidate>_<indicator> scalar series",
                    "message-prefix logs -> numeric log indicator series",
                    "trace status != 200 -> Trace Error series",
                    "end_time-start_time -> Trace Latency series",
                    "consistent numeric time unit for channel files and event anchor",
                    "trusted source hashes/config/masks; labels remain outside prediction-visible input",
                ],
            },
        },
        "recommendations": {
            "W600": "CONDITIONAL: telemetry coverage is acceptable, but context grouping/contamination is severe",
            "W300": "CONDITIONAL ADVANTAGE: much smaller context components with similar activity coverage; exact 68D health still not computed",
            "B15": "CONDITIONAL: no all-missing structural failure is proven, but 30s metrics imply alternating-bin sparsity and about half cadence occupancy",
            "B30": "CONDITIONAL ADVANTAGE for GAIA metric cadence; changes frozen temporal resolution and remains NOT FROZEN",
            "trace_error": "use status != 200 for GAIA adapter; historical >=500 proxy is not semantically valid",
            "configuration_freeze": "NOT FROZEN",
        },
        "prohibited_selection_metrics": ["AC@1", "Avg@5", "MRR"],
        "limitations": [
            "No ranker was trained and no ranking metric was computed.",
            "Exact GAIA Z2 feature health for the four window/bin configurations requires a frozen audit-only four-channel materializer.",
            "Historical W300 P2 arrays are used only as checksum-verified proxy evidence and preserve their status>=500 limitation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-registry", required=True, type=Path)
    parser.add_argument("--raw-telemetry", required=True, type=Path)
    parser.add_argument("--ada-rca-repo", required=True, type=Path)
    parser.add_argument("--historical-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = json.loads(args.raw_telemetry.read_text(encoding="utf-8"))
    # Normalize the rebound full-scan activity table for the compatibility audit.
    historical = json.loads(
        (args.historical_repo / "artifacts/p1/telemetry_diagnostics.json").read_text(encoding="utf-8")
    )
    raw["historical_full_scan_coverage"] = {
        modality: historical["gaia"]["modalities"][modality]["case_window_activity"]
        for modality in ("metrics", "logs", "traces")
    }
    report = build(
        _read_registry(args.event_registry), raw, args.ada_rca_repo, args.historical_repo
    )
    report["source"] = {
        "event_registry": str(args.event_registry.resolve()),
        "event_registry_sha256": sha256_file(args.event_registry),
        "raw_telemetry_artifact": str(args.raw_telemetry.resolve()),
        "raw_telemetry_artifact_sha256": sha256_file(args.raw_telemetry),
        "historical_repository": str(args.historical_repo.resolve()),
        "historical_head": _git(args.historical_repo, "rev-parse", "HEAD"),
    }
    write_json(args.output, report)
    print(
        "W300_groups={}; W600_groups={}; exact_z2={}".format(
            report["context_overlap"]["300"]["injection_union_context"]["groups"],
            report["context_overlap"]["600"]["injection_union_context"]["groups"],
            report["configuration_health"]["W600_B15"]["exact_68d_representation_built"],
        )
    )


if __name__ == "__main__":
    main()
