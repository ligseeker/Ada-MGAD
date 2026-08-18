import json
from pathlib import Path
import tempfile
import unittest

from src.data import RCACaseInput
from src.features import (
    FeatureRow,
    FeatureSchemaError,
    load_feature_matrices,
    read_feature_bundle,
    verify_feature_bundle,
    write_feature_bundle,
)


class FeatureSchemaTest(unittest.TestCase):
    def _inputs(self):
        return (
            RCACaseInput("case-b", "toy", 0, ("svc-a", "svc-b")),
            RCACaseInput("case-a", "toy", 0, ("svc-a", "svc-b")),
        )

    def _rows(self):
        return tuple(
            FeatureRow(
                case_id=case_id,
                service=service,
                extractor="toy-v1",
                feature_names=("metric.shift", "metric.coverage"),
                values=(1.5, 0.0),
                observed=(True, False),
            )
            for case_id in ("case-b", "case-a")
            for service in ("svc-a", "svc-b")
        )

    def test_bundle_is_complete_label_free_and_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_index = write_feature_bundle(
                first, "toy", self._inputs(), self._rows(), {"window": 10}, {"input": "abc"}
            )
            second_index = write_feature_bundle(
                second, "toy", self._inputs(), reversed(self._rows()), {"window": 10}, {"input": "abc"}
            )
            self.assertEqual(first_index, second_index)
            self.assertEqual(
                {path.name: path.read_bytes() for path in Path(first).iterdir()},
                {path.name: path.read_bytes() for path in Path(second).iterdir()},
            )
            manifest, rows = read_feature_bundle(first)
            self.assertEqual(manifest["case_count"], 2)
            self.assertEqual(len(rows), 4)
            matrix_manifest, index, values, observed = load_feature_matrices(first)
            self.assertEqual(matrix_manifest, manifest)
            self.assertEqual(len(index), 4)
            self.assertEqual(values.shape, observed.shape)
            text = (Path(first) / "index.jsonl").read_text(encoding="utf-8")
            text += (Path(first) / "manifest.json").read_text(encoding="utf-8")
            self.assertNotIn("root_service", text)
            self.assertNotIn("fault_type", text)
            verify_feature_bundle(first, self._inputs())

    def test_rejects_sensitive_feature_names_and_nonzero_masked_values(self):
        with self.assertRaisesRegex(FeatureSchemaError, "Label Firewall"):
            FeatureRow("case", "svc", "x", ("root_service_hint",), (0.0,), (True,))
        with self.assertRaisesRegex(FeatureSchemaError, "masked"):
            FeatureRow("case", "svc", "x", ("metric.x",), (1.0,), (False,))

    def test_rejects_incomplete_service_coverage_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FeatureSchemaError, "coverage"):
                write_feature_bundle(
                    directory,
                    "toy",
                    self._inputs(),
                    self._rows()[:-1],
                    {},
                    {},
                )
            write_feature_bundle(
                directory, "toy", self._inputs(), self._rows(), {}, {}
            )
            with (Path(directory) / "index.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"root_service": "svc-a"}) + "\n")
            with self.assertRaisesRegex(FeatureSchemaError, "checksum"):
                verify_feature_bundle(directory)


if __name__ == "__main__":
    unittest.main()
