#!/usr/bin/env python3
"""Create or verify a content hash inventory of C1 raw telemetry inputs.

This reads files only. It does not parse telemetry, fit preprocessing, or read
event labels. Creation refuses to replace an existing output file.
"""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/e2e/gaia_p5_v3_preprocessing_v2.json"
MODALITIES = (
    "metric/metric_split/metric",
    "business/business_split/business",
    "trace/trace_split/trace",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw_root = Path(config["gaia_raw_root"]).resolve()
    paths = []
    for relative in MODALITIES:
        directory = raw_root / relative
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files = sorted(directory.glob("*.csv"))
        if not files:
            raise ValueError("empty raw modality: {}".format(directory))
        paths.extend(files)
    run_table = Path(config["run_table"]["path"]).resolve()
    if not run_table.is_relative_to(raw_root) or not run_table.is_file():
        raise ValueError("run table is missing or outside raw root")
    paths.append(run_table)
    paths = sorted(paths, key=lambda path: str(path.relative_to(raw_root)))
    records = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("raw source is not a regular file: {}".format(path))
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("raw file changed during hashing: {}".format(path))
        records.append({"path": str(path.relative_to(raw_root)),
                        "bytes": before.st_size, "sha256": digest})
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"schema_version": "p6_c1_raw_content_inventory_v1",
            "raw_root": str(raw_root), "config_sha256": sha256(CONFIG),
            "file_count": len(records), "total_bytes": sum(row["bytes"] for row in records),
            "records_sha256": hashlib.sha256(canonical).hexdigest(), "files": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "check"))
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    if args.action == "create" and manifest.exists():
        raise FileExistsError(manifest)
    if args.action == "check" and not manifest.is_file():
        raise FileNotFoundError(manifest)
    current = inventory()
    if args.action == "create":
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(current, stream, sort_keys=True, indent=2)
            stream.write("\n")
    else:
        expected = json.loads(manifest.read_text(encoding="utf-8"))
        if current != expected:
            raise ValueError("C1 raw-content inventory drift")
    print("PASS", current["file_count"], current["total_bytes"], current["records_sha256"])


if __name__ == "__main__":
    main()
