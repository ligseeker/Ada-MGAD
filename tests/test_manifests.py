import json
from pathlib import Path
import tempfile
import unittest

from src.data.manifest import (
    ManifestIntegrityError,
    verify_manifest_bundle,
    write_manifest_bundle,
)
from src.data.schema import RCACaseInput, RCACaseLabel, TelemetryRef


class ManifestTest(unittest.TestCase):
    def make_case(self):
        case_input = RCACaseInput(
            case_id="opaque-case-1",
            dataset="toy",
            anchor_time=1000,
            services=("svc-a", "svc-b"),
            metrics=TelemetryRef(
                uri="toy://opaque-case-1/metrics",
                format="csv",
                metadata={"timestamp_unit": "ms"},
            ),
            metadata={"timezone": "UTC"},
        )
        label = RCACaseLabel("opaque-case-1", "svc-a", "cpu")
        return case_input, label

    def write_bundle(self, directory: Path):
        case_input, label = self.make_case()
        return write_manifest_bundle(
            str(directory),
            dataset="toy",
            inputs=(case_input,),
            labels=(label,),
            trusted_sidecars={
                "sources.jsonl": [
                    {
                        "case_id": case_input.case_id,
                        "raw_path": "/trusted/svc-a_cpu/1/metrics.csv",
                    }
                ]
            },
            metadata={"grouping": "singleton"},
        )

    def test_inputs_labels_and_sources_are_physically_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self.write_bundle(root)
            input_record = json.loads(
                (root / "inputs.jsonl").read_text(encoding="utf-8")
            )
            label_record = json.loads(
                (root / "labels.jsonl").read_text(encoding="utf-8")
            )
            source_record = json.loads(
                (root / "sources.jsonl").read_text(encoding="utf-8")
            )

            self.assertNotIn("root_service", input_record)
            self.assertNotIn("fault_type", input_record)
            self.assertNotIn("/trusted/", json.dumps(input_record))
            self.assertEqual(label_record["root_service"], "svc-a")
            self.assertIn("svc-a_cpu", source_record["raw_path"])
            self.assertEqual(
                index["files"]["inputs.jsonl"]["role"], "prediction_input"
            )
            verify_manifest_bundle(str(root))

    def test_bundle_bytes_are_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            self.write_bundle(first_root)
            self.write_bundle(second_root)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first_root.iterdir()},
                {path.name: path.read_bytes() for path in second_root.iterdir()},
            )

    def test_checksum_verification_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            with (root / "inputs.jsonl").open("ab") as handle:
                handle.write(b"{}\n")
            with self.assertRaisesRegex(ManifestIntegrityError, "mismatch"):
                verify_manifest_bundle(str(root))

    def test_sidecar_cannot_shadow_mandatory_files(self):
        with tempfile.TemporaryDirectory() as directory:
            case_input, label = self.make_case()
            with self.assertRaisesRegex(ManifestIntegrityError, "sidecar names"):
                write_manifest_bundle(
                    directory,
                    "toy",
                    (case_input,),
                    (label,),
                    {"inputs.jsonl": []},
                )


if __name__ == "__main__":
    unittest.main()
