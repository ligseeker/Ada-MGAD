import unittest

from src.data import (
    CaseGroup,
    LabeledEventInterval,
    SplitIntegrityError,
    build_event_purity_flags,
)


class EventPurityTest(unittest.TestCase):
    def test_half_open_concurrency_and_root_cardinality(self):
        events = (
            LabeledEventInterval("a", 0, 10, "r1", "f1"),
            LabeledEventInterval("b", 5, 15, "r2", "f2"),
            LabeledEventInterval("c", 10, 20, "r2", "f1"),
            LabeledEventInterval("d", 30, 40, "r3", "f3"),
        )
        groups = (
            CaseGroup("a", "g1", 3, 1),
            CaseGroup("b", "g1", 3, 2),
            CaseGroup("c", "g1", 3, 1),
            CaseGroup("d", "g2", 1, 0),
        )
        flags = {
            row.case_id: row
            for row in build_event_purity_flags(events, groups, context_radius_ms=5)
        }
        self.assertEqual(flags["a"].actual_overlap_degree, 1)
        self.assertEqual(flags["a"].anchor_concurrent_degree, 0)
        self.assertTrue(flags["b"].anchor_has_multiple_root_services)
        self.assertEqual(flags["c"].anchor_concurrent_degree, 1)
        self.assertFalse(flags["c"].anchor_has_multiple_root_services)
        self.assertEqual(flags["c"].context_group_root_service_count, 2)
        self.assertEqual(flags["d"].context_group_size, 1)

    def test_context_start_count_is_half_open(self):
        events = (
            LabeledEventInterval("a", 0, 1, "r1", "f1"),
            LabeledEventInterval("b", 5, 6, "r1", "f1"),
        )
        groups = (
            CaseGroup("a", "a", 1, 0),
            CaseGroup("b", "b", 1, 0),
        )
        flags = {
            row.case_id: row
            for row in build_event_purity_flags(events, groups, context_radius_ms=5)
        }
        self.assertEqual(flags["a"].other_event_starts_in_context, 0)
        self.assertEqual(flags["b"].other_event_starts_in_context, 1)

    def test_invalid_intervals_and_group_mismatch_are_rejected(self):
        with self.assertRaisesRegex(SplitIntegrityError, "invalid"):
            build_event_purity_flags(
                (LabeledEventInterval("a", 1, 1, "r", "f"),),
                (CaseGroup("a", "a", 1, 0),),
                5,
            )
        with self.assertRaisesRegex(SplitIntegrityError, "do not match"):
            build_event_purity_flags(
                (LabeledEventInterval("a", 0, 1, "r", "f"),),
                (CaseGroup("b", "b", 1, 0),),
                5,
            )


if __name__ == "__main__":
    unittest.main()
