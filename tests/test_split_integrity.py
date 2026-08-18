from types import SimpleNamespace
import unittest

from src.data.split import (
    CaseGroup,
    SplitAssignment,
    SplitIntegrityError,
    build_overlap_groups,
    validate_split_integrity,
)


class SplitIntegrityTest(unittest.TestCase):
    def make_events(self):
        return [
            SimpleNamespace(case_id="a", overlapping_case_ids=("b",)),
            SimpleNamespace(case_id="b", overlapping_case_ids=("a", "c")),
            SimpleNamespace(case_id="c", overlapping_case_ids=("b",)),
            SimpleNamespace(case_id="d", overlapping_case_ids=()),
        ]

    def test_overlap_groups_are_transitive_and_deterministic(self):
        first = build_overlap_groups(self.make_events())
        second = build_overlap_groups(reversed(self.make_events()))
        first_by_case = {row.case_id: row for row in first}
        second_by_case = {row.case_id: row for row in second}

        self.assertEqual(first, second)
        self.assertEqual(first_by_case["a"].group_id, first_by_case["c"].group_id)
        self.assertEqual(first_by_case["a"].group_size, 3)
        self.assertNotEqual(first_by_case["a"].group_id, first_by_case["d"].group_id)
        self.assertEqual(second_by_case["b"].overlap_degree, 2)

    def test_unknown_overlap_reference_is_rejected(self):
        events = [SimpleNamespace(case_id="a", overlapping_case_ids=("missing",))]
        with self.assertRaisesRegex(SplitIntegrityError, "unknown"):
            build_overlap_groups(events)

    def test_valid_grouped_split_passes(self):
        groups = build_overlap_groups(self.make_events())
        assignments = [
            SplitAssignment("a", "train"),
            SplitAssignment("b", "train"),
            SplitAssignment("c", "train"),
            SplitAssignment("d", "test"),
        ]
        validate_split_integrity(assignments, groups)

    def test_overlap_component_cannot_cross_splits(self):
        groups = build_overlap_groups(self.make_events())
        assignments = [
            SplitAssignment("a", "train"),
            SplitAssignment("b", "test"),
            SplitAssignment("c", "train"),
            SplitAssignment("d", "test"),
        ]
        with self.assertRaisesRegex(SplitIntegrityError, "crosses splits"):
            validate_split_integrity(assignments, groups)

    def test_split_must_cover_every_case_exactly_once(self):
        groups = build_overlap_groups(self.make_events())
        with self.assertRaisesRegex(SplitIntegrityError, "mismatch"):
            validate_split_integrity(
                [SplitAssignment("a", "train")],
                groups,
            )
        with self.assertRaisesRegex(SplitIntegrityError, "duplicate"):
            validate_split_integrity(
                [
                    SplitAssignment("a", "train"),
                    SplitAssignment("a", "train"),
                    SplitAssignment("b", "train"),
                    SplitAssignment("c", "train"),
                    SplitAssignment("d", "test"),
                ],
                groups,
            )

    def test_declared_group_size_is_checked(self):
        groups = (
            CaseGroup("a", "shared", 3, 1),
            CaseGroup("b", "shared", 3, 1),
        )
        with self.assertRaisesRegex(SplitIntegrityError, "member count"):
            validate_split_integrity(
                [SplitAssignment("a", "train"), SplitAssignment("b", "train")],
                groups,
            )


if __name__ == "__main__":
    unittest.main()
