"""Source/input snapshots and fail-closed publication for the C0F audit only."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def verify_records(records):
    failures = []
    for name, expected in records.items():
        path = Path(expected["path"])
        if not path.is_file():
            failures.append(name + ": missing")
            continue
        actual = file_record(path)
        if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
            failures.append(name + ": changed")
    if failures:
        raise ValueError("artifact identity failure: " + "; ".join(failures))


def source_records(project_root):
    root = Path(project_root)
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=str(root)
    ).decode().split("\0")
    required = {
        "docs/P6_C0F_CORRECTION_PROTOCOL.md",
        "configs/e2e/gaia_p5_v3_preprocessing_v2.json",
        "configs/e2e/gaia_p6_c0_system_trigger.json",
    }
    missing = required.difference(names)
    if missing:
        raise ValueError("required correction protocol or frozen config missing from source snapshot: "
                         + ", ".join(sorted(missing)))
    selected = sorted({name for name in names if name.endswith(".py") or name in (
        "pytest.ini", "docs/P6_C0F_CORRECTION_PROTOCOL.md",
        "configs/e2e/gaia_p5_v3_preprocessing_v2.json", "configs/e2e/gaia_p6_c0_system_trigger.json")})
    return {name: file_record(root / name) for name in selected}


def source_digest(records):
    stable = {name: {"sha256": record["sha256"], "bytes": record["bytes"]}
              for name, record in sorted(records.items())}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def snapshot_sources(project_root, output_dir, argv):
    root, output = Path(project_root), Path(output_dir)
    records = source_records(root)
    for name, record in records.items():
        target = output / "provenance" / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record["path"], target)
    (output / "provenance" / "working_tree.diff").write_bytes(subprocess.check_output(
        ["git", "diff", "HEAD", "--binary"], cwd=str(root)))
    snapshot = {
        "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=str(root), text=True),
        "source_digest": source_digest(records), "files": records,
        "argv": list(argv), "python_executable": sys.executable,
        "scope": "all tracked/unignored Python sources, tests, frozen configs and the correction protocol",
        "identity": "HEAD plus the archived working-tree source bytes; not a claim of a clean committed execution",
    }
    atomic_json(output / "source_snapshot.json", snapshot)
    return snapshot


def check_sources(project_root, output_dir):
    saved = json.loads((Path(output_dir) / "source_snapshot.json").read_text())
    current = source_records(project_root)
    if source_digest(current) != saved["source_digest"]:
        raise ValueError("audit source changed since freeze")
    verify_records(saved["files"])
    for name, record in saved["files"].items():
        archived = file_record(Path(output_dir) / "provenance" / "source_snapshot" / name)
        if archived["sha256"] != record["sha256"]:
            raise ValueError("source archive changed: " + name)
    return saved


def run_code_checks(project_root, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    sources = source_records(project_root)
    selected = sorted(str(path.relative_to(project_root)) for path in (Path(project_root) / "tests").glob("test_p6_c0*.py"))
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--junitxml=" + str(output / "pytest.xml")] + selected
    start = time.monotonic()
    with (output / "pytest.log").open("w") as log:
        completed = subprocess.run(command, cwd=str(project_root), stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    if source_digest(source_records(project_root)) != source_digest(sources):
        raise ValueError("source changed during tests")
    result = {"command": command, "exit_code": completed.returncode, "runtime_seconds": time.monotonic() - start,
              "source_digest": source_digest(sources),
              "files": {name: file_record(output / name) for name in ("pytest.log", "pytest.xml")}}
    atomic_json(output / "test_results.json", result)
    if completed.returncode:
        raise ValueError("C0F code checks failed; see " + str(output / "pytest.log"))
    return result


def bind_tests(project_root, output_dir, record_path):
    record = json.loads(Path(record_path).read_text())
    if record["exit_code"] != 0 or record["source_digest"] != source_digest(source_records(project_root)):
        raise ValueError("tests did not pass against these source bytes")
    verify_records(record["files"])
    target = Path(output_dir) / "validation"
    # A supplied JSON record is not a trusted attestation of what ran. Execute
    # the exact selected suite again and archive that run as the completion gate.
    own_result = run_code_checks(project_root, target)
    own_result["precheck_record"] = file_record(record_path)
    atomic_json(Path(output_dir) / "test_results.json", own_result)


def snapshot_inputs(paths, output_dir):
    records = {str(Path(path).resolve()): file_record(path) for path in sorted(set(paths))}
    atomic_json(Path(output_dir) / "input_snapshot.json", {"files": records})
    return records


def assert_stage_results(stage, checks):
    """Reject missing/false checks; do not turn a failed validation into COMPLETE."""
    if not checks or any(value is not True for value in checks.values()):
        raise ValueError("{} validation failed: {}".format(stage, checks))


def capacity_checks(results):
    checks = {}
    for split in ("fit", "validation", "test"):
        entry = results[split]
        bound = entry["episode_bound"]
        grid = entry["grid_bound"]
        ordering = entry["ordering"]
        checks[split + ":ordering"] = bool(
            ordering["observed_tp"] <= bound["upper_bound_tp"] <= grid["matched_events"] <= bound["ground_truth_events"])
        checks[split + ":witness"] = bool(
            bound["witness_replay_agrees"] and bound["witness_anchors_match"]
            and bound["witness_realised_tp"] == bound["upper_bound_tp"])
        checks[split + ":grid_cross_check"] = grid["cross_check_agrees"] is not False
        checks[split + ":exact_proof"] = bool(bound["status"] == "EXACT" and bound["proof"])
    return checks
