import json
from pathlib import Path
import tempfile
import unittest

from src.data.source_snapshot import (
    ConsumedSource,
    SourceSnapshotError,
    verify_source_snapshot,
    write_source_snapshot,
)


class SourceSnapshotTest(unittest.TestCase):
    def _write_snapshot(self, root: Path, output: Path):
        (root / "dataset" / "case-1").mkdir(parents=True)
        metric = root / "dataset" / "case-1" / "metrics.csv"
        metric.write_bytes(b"time,value\n1,2\n")
        archive = root / "dataset.zip"
        archive.write_bytes(b"immutable archive bytes")
        return write_source_snapshot(
            str(output),
            "toy",
            str(root),
            (ConsumedSource("case-1", "metrics", str(metric)),),
            (str(archive),),
            metadata={"purpose": "test"},
        )

    def test_snapshot_is_location_independent_and_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first) / "source"
            second_root = Path(second) / "source"
            first_root.mkdir()
            second_root.mkdir()
            first_output = Path(first) / "snapshot"
            second_output = Path(second) / "snapshot"
            first_index = self._write_snapshot(first_root, first_output)
            second_index = self._write_snapshot(second_root, second_output)

            self.assertEqual(first_index, second_index)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first_output.iterdir()},
                {path.name: path.read_bytes() for path in second_output.iterdir()},
            )
            record = json.loads(
                (first_output / "consumed_files.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(record["relative_path"], "dataset/case-1/metrics.csv")
            self.assertNotIn(first, json.dumps(record))
            verify_source_snapshot(str(first_output), str(first_root))

    def test_verification_detects_raw_source_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            output = Path(directory) / "snapshot"
            self._write_snapshot(root, output)
            (root / "dataset" / "case-1" / "metrics.csv").write_bytes(b"changed")
            with self.assertRaisesRegex(SourceSnapshotError, "source size mismatch"):
                verify_source_snapshot(str(output), str(root))

    def test_rejects_paths_outside_source_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            outside = Path(directory) / "outside.csv"
            outside.write_bytes(b"outside")
            with self.assertRaisesRegex(SourceSnapshotError, "outside"):
                write_source_snapshot(
                    str(Path(directory) / "snapshot"),
                    "toy",
                    str(root),
                    (ConsumedSource("case-1", "metrics", str(outside)),),
                )


if __name__ == "__main__":
    unittest.main()
