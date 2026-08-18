from types import SimpleNamespace
from collections import Counter
import unittest

from src.data.split import (
    CaseGroup,
    SplitAssignment,
    CaseInterval,
    SplitIntegrityError,
    assign_balanced_group_folds,
    assign_contiguous_group_folds,
    build_overlap_groups,
    validate_split_integrity,
    build_interval_overlap_groups,
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

    def test_interval_groups_use_half_open_boundaries(self):
        touching = (
            CaseInterval("a", 0, 10),
            CaseInterval("b", 10, 20),
        )
        groups = {
            row.case_id: row for row in build_interval_overlap_groups(touching)
        }
        self.assertNotEqual(groups["a"].group_id, groups["b"].group_id)
        self.assertEqual(groups["a"].overlap_degree, 0)

    def test_interval_groups_are_transitive_and_count_direct_degree(self):
        intervals = (
            CaseInterval("a", 0, 10),
            CaseInterval("b", 10, 20),
            CaseInterval("bridge", 9, 11),
            CaseInterval("d", 30, 40),
        )
        groups = {
            row.case_id: row for row in build_interval_overlap_groups(intervals)
        }
        self.assertEqual(groups["a"].group_id, groups["b"].group_id)
        self.assertEqual(groups["a"].group_size, 3)
        self.assertEqual(groups["a"].overlap_degree, 1)
        self.assertEqual(groups["bridge"].overlap_degree, 2)
        self.assertNotEqual(groups["a"].group_id, groups["d"].group_id)

    def test_interval_groups_reject_duplicates_and_empty_ranges(self):
        with self.assertRaisesRegex(SplitIntegrityError, "duplicate"):
            build_interval_overlap_groups(
                (CaseInterval("a", 0, 10), CaseInterval("a", 20, 30))
            )
        with self.assertRaisesRegex(SplitIntegrityError, "start_ms < end_ms"):
            build_interval_overlap_groups((CaseInterval("a", 10, 10),))


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

    def test_balanced_group_folds_are_deterministic_and_group_safe(self):
        groups = (
            CaseGroup("a1", "a", 2, 1),
            CaseGroup("a2", "a", 2, 1),
            CaseGroup("b1", "b", 1, 0),
            CaseGroup("c1", "c", 1, 0),
            CaseGroup("d1", "d", 1, 0),
            CaseGroup("e1", "e", 1, 0),
        )
        strata = {
            "a1": {"root": "r1", "fault": "f1"},
            "a2": {"root": "r2", "fault": "f2"},
            "b1": {"root": "r1", "fault": "f2"},
            "c1": {"root": "r2", "fault": "f1"},
            "d1": {"root": "r1", "fault": "f1"},
            "e1": {"root": "r2", "fault": "f2"},
        }
        first = assign_balanced_group_folds(groups, strata, n_folds=3, seed=17)
        second = assign_balanced_group_folds(
            tuple(reversed(groups)), strata, n_folds=3, seed=17
        )
        self.assertEqual(first, second)
        validate_split_integrity(first, groups)
        by_case = {row.case_id: row.split for row in first}
        self.assertEqual(by_case["a1"], by_case["a2"])
        self.assertEqual(set(by_case.values()), {"fold-0", "fold-1", "fold-2"})

    def test_contiguous_group_folds_preserve_time_order(self):
        groups = (
            CaseGroup("a1", "a", 2, 1),
            CaseGroup("a2", "a", 2, 1),
            CaseGroup("b", "b", 1, 0),
            CaseGroup("c", "c", 1, 0),
            CaseGroup("d1", "d", 2, 1),
            CaseGroup("d2", "d", 2, 1),
        )
        order = {"a1": 1, "a2": 2, "b": 3, "c": 4, "d1": 5, "d2": 6}
        assignments = assign_contiguous_group_folds(
            tuple(reversed(groups)), order, n_folds=3
        )
        validate_split_integrity(assignments, groups)
        fold_number = {
            row.case_id: int(row.split.split("-", 1)[1])
            for row in assignments
        }
        ordered_folds = [fold_number[case_id] for case_id in order]
        self.assertEqual(ordered_folds, sorted(ordered_folds))
        self.assertEqual(fold_number["a1"], fold_number["a2"])
        self.assertEqual(fold_number["d1"], fold_number["d2"])

    def test_fold_builders_reject_incomplete_metadata_and_interleaving(self):
        groups = (
            CaseGroup("a1", "a", 2, 1),
            CaseGroup("a2", "a", 2, 1),
            CaseGroup("b", "b", 1, 0),
        )
        with self.assertRaisesRegex(SplitIntegrityError, "strata/group"):
            assign_balanced_group_folds(
                groups,
                {"a1": {"root": "r"}, "a2": {"root": "r"}},
                n_folds=2,
            )
        with self.assertRaisesRegex(SplitIntegrityError, "interleave"):
            assign_contiguous_group_folds(
                groups,
                {"a1": 1, "a2": 4, "b": 2},
                n_folds=2,
            )

    def test_balanced_folds_handle_repeated_two_axis_grid(self):
        groups = []
        strata = {}
        for root_index in range(5):
            for fault_index in range(6):
                for repeat in range(3):
                    case_id = "r{}-f{}-{}".format(root_index, fault_index, repeat)
                    groups.append(CaseGroup(case_id, case_id, 1, 0))
                    strata[case_id] = {
                        "fault": "f{}".format(fault_index),
                        "joint": "r{}-f{}".format(root_index, fault_index),
                        "root": "r{}".format(root_index),
                    }
        assignments = assign_balanced_group_folds(
            groups,
            strata,
            n_folds=5,
            seed=20260819,
            axis_weights={"fault": 1.0, "joint": 0.25, "root": 1.0},
        )
        by_fold = {"fold-{}".format(index): [] for index in range(5)}
        for row in assignments:
            by_fold[row.split].append(row.case_id)
        self.assertEqual(sorted(map(len, by_fold.values())), [18] * 5)
        for case_ids in by_fold.values():
            roots = Counter(strata[case_id]["root"] for case_id in case_ids)
            faults = Counter(strata[case_id]["fault"] for case_id in case_ids)
            self.assertLessEqual(max(roots.values()), 4)
            self.assertGreaterEqual(min(roots.values()), 3)
            self.assertEqual(set(faults.values()), {3})


if __name__ == "__main__":
    unittest.main()
