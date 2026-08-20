#!/usr/bin/env python
"""E1: build the RE2-TT protocol-extension manifests and pin their raw sources.

This is a separate entry point on purpose. ``scripts/prepare_p1_manifests.py`` and
``scripts/pin_p1_rcaeval_source.py`` write the frozen GAIA/RE2-OB subtrees, so they
must not be re-run; this script only ever writes under ``artifacts/ext/re2tt/``.

Before writing anything it re-derives the frozen RE2-OB manifest with the newly
parameterized adapter and compares it byte-for-byte against
``artifacts/p1/manifests/re2ob/``. That is the proof that adding RE2-TT did not
perturb an already-recorded dataset.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    ConsumedSource,
    RE2OB_PROFILE,
    RE2TT_PROFILE,
    RCAEvalDatasetProfile,
    build_singleton_groups,
    load_rcaeval_cases,
    verify_manifest_bundle,
    verify_source_snapshot,
    write_manifest_bundle,
    write_source_snapshot,
)


DEFAULT_RAW_PATH = "/home/zhangll24/RCA_project/datasets/RCAEval/RE2/RE2-TT"
DEFAULT_ARCHIVE = "/home/zhangll24/RCA_project/datasets/RCAEval/RE2/RE2-TT.zip"
DEFAULT_OUTPUT_ROOT = "artifacts/ext/re2tt"
FROZEN_RE2OB_MANIFEST = "artifacts/p1/manifests/re2ob"
PROTECTED_ROOTS = ("artifacts/p1", "artifacts/p2")
REGRESSION_FILES = ("inputs.jsonl", "labels.jsonl", "sources.jsonl", "groups.jsonl")
ROLE_FIELDS = {
    "inject_time": "inject_time_path",
    "logs": "logs_path",
    "metrics": "metrics_path",
    "traces": "traces_path",
}


class ExtensionIsolationError(RuntimeError):
    """Raised when an extension stage would touch a frozen artifact subtree."""


def _assert_isolated(output_root: Path) -> None:
    resolved = output_root.resolve()
    for protected in PROTECTED_ROOTS:
        guard = (PROJECT_ROOT / protected).resolve()
        if resolved == guard or guard in resolved.parents:
            raise ExtensionIsolationError(
                "refusing to write inside frozen subtree {}: {}".format(guard, resolved)
            )


def _group_records(groups: Iterable[object]) -> Sequence[Mapping[str, object]]:
    return [
        {
            "case_id": group.case_id,
            "group_id": group.group_id,
            "group_size": group.group_size,
            "overlap_degree": group.overlap_degree,
        }
        for group in groups
    ]


def _group_summary(groups: Iterable[object]) -> Mapping[str, int]:
    values = tuple(groups)
    sizes_by_id: Dict[str, int] = {}
    for group in values:
        sizes_by_id[group.group_id] = group.group_size
    size_counts = Counter(sizes_by_id.values())
    return {
        "groups": len(sizes_by_id),
        "non_singleton_groups": sum(count for size, count in size_counts.items() if size > 1),
        "cases_in_non_singleton_groups": sum(
            size * count for size, count in size_counts.items() if size > 1
        ),
        "largest_group": max(sizes_by_id.values(), default=0),
    }


def _sidecars(result) -> Mapping[str, Sequence[Mapping[str, object]]]:
    groups = build_singleton_groups(case.case_id for case in result.inputs)
    return {
        "excluded.jsonl": [
            {"reason": row.reason, "relative_directory": row.relative_directory}
            for row in result.excluded
        ],
        "groups.jsonl": _group_records(groups),
        "sources.jsonl": [
            {
                "case_id": row.case_id,
                "inject_time_path": row.inject_time_path,
                "logs_path": row.logs_path,
                "metrics_path": row.metrics_path,
                "relative_directory": row.relative_directory,
                "traces_path": row.traces_path,
            }
            for row in result.sources
        ],
    }


def _write_bundle(
    raw_path: str,
    profile: RCAEvalDatasetProfile,
    output_directory: Path,
    status: str,
) -> Mapping[str, object]:
    result = load_rcaeval_cases(raw_path, profile)
    sidecars = _sidecars(result)
    group_summary = _group_summary(build_singleton_groups(case.case_id for case in result.inputs))
    write_manifest_bundle(
        str(output_directory),
        dataset=profile.dataset,
        inputs=result.inputs,
        labels=result.labels,
        trusted_sidecars=sidecars,
        metadata={
            "adapter": "src.data.rcaeval.load_rcaeval_cases",
            "adapter_profile": profile.key,
            "excluded_cases": len(result.excluded),
            "grouping": "one official failure-case directory per group",
            "group_summary": group_summary,
            "status": status,
        },
    )
    verify_manifest_bundle(str(output_directory))
    return {
        "cases": len(result.inputs),
        "excluded": len(result.excluded),
        "candidate_services": len(result.inputs[0].services) if result.inputs else 0,
        **group_summary,
    }


def _re2ob_raw_root(frozen_manifest: Path) -> Path:
    sources = frozen_manifest / "sources.jsonl"
    with sources.open("r", encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    # <raw root>/<condition>/<replicate>/metrics.csv
    return Path(first["metrics_path"]).parents[2]


def _regression_re2ob(frozen_manifest: Path) -> Mapping[str, object]:
    """Re-derive RE2-OB with the parameterized adapter; require byte identity."""

    if not frozen_manifest.is_dir():
        return {"status": "skipped", "reason": "frozen manifest absent"}
    raw_root = _re2ob_raw_root(frozen_manifest)
    if not raw_root.is_dir():
        return {"status": "skipped", "reason": "raw RE2-OB root absent: {}".format(raw_root)}

    scratch = Path(tempfile.mkdtemp(prefix="re2ob-regression-"))
    try:
        rebuilt = scratch / "manifests"
        result = load_rcaeval_cases(str(raw_root), RE2OB_PROFILE)
        write_manifest_bundle(
            str(rebuilt),
            dataset=RE2OB_PROFILE.dataset,
            inputs=result.inputs,
            labels=result.labels,
            trusted_sidecars=_sidecars(result),
            metadata={"regression": "byte-identity check only"},
        )
        compared = {}
        for name in REGRESSION_FILES:
            frozen_bytes = (frozen_manifest / name).read_bytes()
            rebuilt_bytes = (rebuilt / name).read_bytes()
            compared[name] = frozen_bytes == rebuilt_bytes
        mismatched = sorted(name for name, identical in compared.items() if not identical)
        if mismatched:
            raise ExtensionIsolationError(
                "adapter refactor changed frozen RE2-OB output: {}".format(
                    ",".join(mismatched)
                )
            )
        return {
            "status": "passed",
            "cases": len(result.inputs),
            "compared_files": sorted(compared),
            "raw_root": str(raw_root),
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _consumed_sources(manifest_directory: Path) -> Sequence[ConsumedSource]:
    consumed = []
    with (manifest_directory / "sources.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for role, field in sorted(ROLE_FIELDS.items()):
                consumed.append(
                    ConsumedSource(case_id=record["case_id"], role=role, path=record[field])
                )
    return tuple(consumed)


def _progress(prefix: str):
    def report(done: int, total: int, relative_path: str) -> None:
        if done == 1 or done == total or done % 40 == 0:
            print("{} {}/{} {}".format(prefix, done, total, relative_path), flush=True)

    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", default=DEFAULT_RAW_PATH)
    parser.add_argument(
        "--source-root",
        default=None,
        help=(
            "snapshot root that must contain both the case tree and the archive; "
            "defaults to the parent of --raw-path, mirroring the frozen RE2-OB "
            "snapshot whose root is RCAEval/ with RE2-OB.zip inside it"
        ),
    )
    parser.add_argument("--archive", action="append", default=None)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frozen-re2ob-manifest", default=FROZEN_RE2OB_MANIFEST)
    parser.add_argument(
        "--skip-source-snapshot",
        action="store_true",
        help="build manifests only; the snapshot hashes ~25 GB and can be pinned later",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    _assert_isolated(output_root)

    summary: Dict[str, object] = {}
    summary["re2ob_regression"] = _regression_re2ob(Path(args.frozen_re2ob_manifest))

    manifest_directory = output_root / "manifests"
    summary["manifests"] = _write_bundle(
        args.raw_path,
        RE2TT_PROFILE,
        manifest_directory,
        status="complete local case inventory; release identifier pinned by source_snapshot",
    )
    summary["manifest_directory"] = str(manifest_directory)

    if args.skip_source_snapshot:
        summary["source_snapshot"] = {"status": "skipped"}
    else:
        source_root = args.source_root or str(Path(args.raw_path).resolve().parent)
        archives = args.archive
        if archives is None:
            archives = [DEFAULT_ARCHIVE] if Path(DEFAULT_ARCHIVE).is_file() else []
        snapshot_directory = output_root / "source_snapshot"
        index = write_source_snapshot(
            output_directory=str(snapshot_directory),
            dataset=RE2TT_PROFILE.dataset,
            source_root=source_root,
            consumed_sources=_consumed_sources(manifest_directory),
            archives=archives,
            metadata={
                "case_manifest_sha256": _sha256(manifest_directory / "manifest.json"),
                "case_sources_sha256": _sha256(manifest_directory / "sources.jsonl"),
                "extension": "re2tt_protocol_extension",
                "scope": (
                    "one source archive plus the exact metrics/logs/traces/inject-time "
                    "files referenced by the 90-case RE2-TT extension manifest"
                ),
            },
            progress=_progress("hash"),
        )
        verify_source_snapshot(
            str(snapshot_directory), source_root, progress=_progress("verify")
        )
        summary["source_snapshot"] = {
            "status": "written",
            "archive_count": index["files"]["archives.jsonl"]["rows"],
            "case_count": index["case_count"],
            "consumed_bytes": index["files"]["consumed_files.jsonl"]["indexed_bytes"],
            "consumed_file_count": index["files"]["consumed_files.jsonl"]["rows"],
            "content_identity_sha256": index["content_identity_sha256"],
            "directory": str(snapshot_directory),
            "source_root": source_root,
        }

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
