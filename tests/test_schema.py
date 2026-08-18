import unittest

from src.data.schema import (
    RCACaseInput,
    RCACaseLabel,
    SchemaValidationError,
    TelemetryRef,
    validate_case_collection,
)


class RCACaseSchemaTest(unittest.TestCase):
    def make_input(self, **overrides):
        values = {
            "case_id": "toy/case-1",
            "dataset": "toy",
            "anchor_time": 1000,
            "services": ("svc-a", "svc-b", "svc-c"),
            "metrics": TelemetryRef(uri="/data/metrics.csv", format="csv"),
            "metadata": {"timezone": "UTC", "timestamp_unit": "ms"},
        }
        values.update(overrides)
        return RCACaseInput(**values)

    def test_valid_case_collection(self):
        case_input = self.make_input()
        label = RCACaseLabel(
            case_id=case_input.case_id,
            root_service="svc-b",
            fault_type="cpu",
        )
        validate_case_collection([case_input], [label])

    def test_services_must_be_nonempty_unique_tuple(self):
        with self.assertRaisesRegex(SchemaValidationError, "tuple"):
            self.make_input(services=["svc-a"])
        with self.assertRaisesRegex(SchemaValidationError, "empty"):
            self.make_input(services=())
        with self.assertRaisesRegex(SchemaValidationError, "duplicates"):
            self.make_input(services=("svc-a", "svc-a"))

    def test_case_ids_must_be_unique(self):
        with self.assertRaisesRegex(SchemaValidationError, "unique"):
            validate_case_collection([self.make_input(), self.make_input()])

    def test_root_must_be_a_candidate_service(self):
        case_input = self.make_input()
        label = RCACaseLabel(case_id=case_input.case_id, root_service="not-observed")
        with self.assertRaisesRegex(SchemaValidationError, "not in services"):
            validate_case_collection([case_input], [label])

    def test_labels_are_rejected_from_nested_prediction_metadata(self):
        with self.assertRaisesRegex(SchemaValidationError, "Label Firewall"):
            self.make_input(metadata={"adapter": {"root-service": "svc-a"}})
        with self.assertRaisesRegex(SchemaValidationError, "Label Firewall"):
            TelemetryRef(uri="/data/metrics.csv", metadata={"faultType": "cpu"})


if __name__ == "__main__":
    unittest.main()
