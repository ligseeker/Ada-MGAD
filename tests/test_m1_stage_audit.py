import unittest

from scripts.audit_p2_m1_stage import (
    EXPECTED_C_VALUES,
    EXPECTED_INNER_FITS,
    EXPECTED_ONSETS,
    compact_metrics,
    compare_with_comparator,
    metric_delta,
    reconstruct_selection,
    verify_design_width,
    verify_fold_audit,
)
from scripts.run_p2_c0_metric import _id_digest


FROZEN_CONFIG = {
    "C_candidates": [0.01, 0.1, 1.0, 10.0],
    "log_stage_excluded": "RE2 content-complete 0.794900 below frozen 0.80 threshold",
    "onset_candidates_seconds": [60, 120],
    "random_seed": 20260819,
    "stage_feature_columns_before_masks": {
        "120": {"metric": 51, "trace": 24},
        "60": {"metric": 51, "trace": 24},
    },
    "whole_feature_columns_before_masks": {"log": 5, "metric": 17, "trace": 8},
}


def _fold_case_ids(fold_count=5, fold_size=4):
    fold_case_ids = {}
    counter = 0
    for index in range(fold_count):
        ids = []
        for _ in range(fold_size):
            ids.append("case-{:04d}".format(counter))
            counter += 1
        fold_case_ids["fold_{}".format(index)] = tuple(sorted(ids))
    return fold_case_ids


def _build_folds(fold_case_ids, score_fn=None):
    """Build a well-formed M1-S training audit trail for the given split."""

    if score_fn is None:
        score_fn = lambda outer, value, onset: 0.5
    names = tuple(sorted(fold_case_ids))
    cohort = tuple(
        sorted(case_id for ids in fold_case_ids.values() for case_id in ids)
    )
    folds = []
    for name in names:
        test_ids = fold_case_ids[name]
        train_ids = tuple(
            case_id for case_id in cohort if case_id not in set(test_ids)
        )
        candidates = []
        for value in EXPECTED_C_VALUES:
            for onset in EXPECTED_ONSETS:
                inner_folds = []
                for other in names:
                    if other == name:
                        continue
                    validation_ids = fold_case_ids[other]
                    fit_ids = tuple(
                        case_id
                        for case_id in train_ids
                        if case_id not in set(validation_ids)
                    )
                    inner_folds.append(
                        {
                            "fit_case_count": len(fit_ids),
                            "fit_case_ids_sha256": _id_digest(fit_ids),
                            "fit_validation_group_overlap": 0,
                            "fit_validation_overlap": 0,
                            "root_service_macro_Avg@5": score_fn(name, value, onset),
                            "validation_case_count": len(validation_ids),
                            "validation_case_ids_sha256": _id_digest(validation_ids),
                            "validation_fold": other,
                        }
                    )
                candidates.append(
                    {
                        "C": value,
                        "inner_folds": inner_folds,
                        "mean_root_service_macro_Avg@5": score_fn(name, value, onset),
                        "onset_seconds": onset,
                    }
                )
        selected = reconstruct_selection(candidates)
        folds.append(
            {
                "inner_candidates": candidates,
                "outer_fold": name,
                "selected_C": selected["C"],
                "selected_onset_seconds": selected["onset_seconds"],
                "test_case_count": len(test_ids),
                "test_case_ids_sha256": _id_digest(test_ids),
                "train_case_count": len(train_ids),
                "train_case_ids_sha256": _id_digest(train_ids),
                "train_test_group_overlap": 0,
                "train_test_overlap": 0,
            }
        )
    return folds


class StageAuditMetricsTest(unittest.TestCase):
    def test_compact_metrics_keeps_the_three_reported_layers(self):
        report = {
            "fault_type": {"by_category": {}, "macro": {"Avg@5": 0.4}},
            "overall": {"Avg@5": 0.5, "case_count": 3},
            "root_service": {"by_category": {}, "macro": {"Avg@5": 0.6}},
        }
        self.assertEqual(
            compact_metrics(report),
            {
                "fault_type_macro": {"Avg@5": 0.4},
                "overall": {"Avg@5": 0.5, "case_count": 3},
                "root_service_macro": {"Avg@5": 0.6},
            },
        )

    def test_metric_delta_is_signed_actual_minus_reference(self):
        template = {"AC@1": 0.0, "AC@3": 0.0, "AC@5": 0.0, "Avg@5": 0.0, "MRR": 0.0}
        actual = {
            "fault_type_macro": dict(template, **{"Avg@5": 0.7}),
            "overall": dict(template),
            "root_service_macro": dict(template, **{"AC@1": 0.2, "Avg@5": 0.6}),
        }
        reference = {
            "fault_type_macro": dict(template, **{"Avg@5": 0.5}),
            "overall": dict(template),
            "root_service_macro": dict(template, **{"AC@1": 0.3, "Avg@5": 0.5}),
        }
        delta = metric_delta(actual, reference)
        self.assertAlmostEqual(delta["fault_type_macro"]["Avg@5"], 0.2)
        self.assertAlmostEqual(delta["root_service_macro"]["Avg@5"], 0.1)
        self.assertAlmostEqual(delta["root_service_macro"]["AC@1"], -0.1)
        self.assertEqual(delta["overall"]["MRR"], 0.0)

    def test_comparison_flags_follow_the_frozen_endpoints(self):
        template = {"AC@1": 0.0, "AC@3": 0.0, "AC@5": 0.0, "Avg@5": 0.0, "MRR": 0.0}
        # The guardrail is compared without tolerance, matching the already frozen
        # bootstrap_p2_c1_i.py, so these fixtures stay clear of the exactly -0.01
        # boundary, which is not representable in binary floating point.
        actual = {
            "fault_type_macro": dict(template),
            "overall": dict(template),
            "root_service_macro": dict(template, **{"AC@1": 0.295, "Avg@5": 0.62}),
        }
        reference = {
            "fault_type_macro": dict(template),
            "overall": dict(template),
            "root_service_macro": dict(template, **{"AC@1": 0.30, "Avg@5": 0.60}),
        }
        comparison = compare_with_comparator(actual, reference)
        self.assertTrue(comparison["m1_s_primary_improved"])
        self.assertTrue(comparison["m1_s_secondary_within_guardrail"])
        regressed = dict(
            actual,
            root_service_macro=dict(template, **{"AC@1": 0.20, "Avg@5": 0.59}),
        )
        comparison = compare_with_comparator(regressed, reference)
        self.assertFalse(comparison["m1_s_primary_improved"])
        self.assertFalse(comparison["m1_s_secondary_within_guardrail"])

    def test_guardrail_reads_only_the_root_service_macro_layer(self):
        template = {"AC@1": 0.0, "AC@3": 0.0, "AC@5": 0.0, "Avg@5": 0.0, "MRR": 0.0}
        actual = {
            "fault_type_macro": dict(template, **{"AC@1": 0.10, "Avg@5": 0.10}),
            "overall": dict(template, **{"AC@1": 0.10, "Avg@5": 0.10}),
            "root_service_macro": dict(template, **{"AC@1": 0.40, "Avg@5": 0.62}),
        }
        reference = {
            "fault_type_macro": dict(template, **{"AC@1": 0.90, "Avg@5": 0.90}),
            "overall": dict(template, **{"AC@1": 0.90, "Avg@5": 0.90}),
            "root_service_macro": dict(template, **{"AC@1": 0.40, "Avg@5": 0.60}),
        }
        comparison = compare_with_comparator(actual, reference)
        self.assertTrue(comparison["m1_s_primary_improved"])
        self.assertTrue(comparison["m1_s_secondary_within_guardrail"])
        self.assertAlmostEqual(comparison["m1_s_delta"]["overall"]["AC@1"], -0.8)


class StageSelectionTest(unittest.TestCase):
    def test_ties_prefer_stronger_regularization_then_shorter_onset(self):
        candidates = [
            {"C": value, "mean_root_service_macro_Avg@5": 0.5, "onset_seconds": onset}
            for value in EXPECTED_C_VALUES
            for onset in EXPECTED_ONSETS
        ]
        selected = reconstruct_selection(candidates)
        self.assertEqual(selected["C"], 0.01)
        self.assertEqual(selected["onset_seconds"], 60)

    def test_higher_score_outranks_the_tie_break(self):
        candidates = [
            {
                "C": value,
                "mean_root_service_macro_Avg@5": 0.9 if value == 10.0 else 0.5,
                "onset_seconds": onset,
            }
            for value in EXPECTED_C_VALUES
            for onset in EXPECTED_ONSETS
        ]
        selected = reconstruct_selection(candidates)
        self.assertEqual(selected["C"], 10.0)
        self.assertEqual(selected["onset_seconds"], 60)

    def test_rejects_a_reduced_candidate_grid(self):
        candidates = [
            {"C": value, "mean_root_service_macro_Avg@5": 0.5, "onset_seconds": 60}
            for value in EXPECTED_C_VALUES
        ]
        with self.assertRaisesRegex(ValueError, "frozen four C by two onset"):
            reconstruct_selection(candidates)

    def test_rejects_an_empty_candidate_list(self):
        with self.assertRaisesRegex(ValueError, "no inner candidates"):
            reconstruct_selection([])


class StageDesignWidthTest(unittest.TestCase):
    def test_frozen_config_yields_the_210_column_design(self):
        design = verify_design_width(FROZEN_CONFIG)
        self.assertEqual(design["whole_value_columns"], 30)
        self.assertEqual(design["stage_value_columns"], 75)
        self.assertEqual(design["value_columns"], 105)
        self.assertEqual(design["design_column_count"], 210)

    def test_rejects_a_shrunken_onset_grid(self):
        config = dict(FROZEN_CONFIG, onset_candidates_seconds=[60])
        with self.assertRaisesRegex(ValueError, "frozen 60/120 s pair"):
            verify_design_width(config)

    def test_rejects_a_shrunken_c_grid(self):
        config = dict(FROZEN_CONFIG, C_candidates=[0.1, 1.0])
        with self.assertRaisesRegex(ValueError, "frozen grid"):
            verify_design_width(config)

    def test_rejects_staged_log_channels(self):
        config = dict(
            FROZEN_CONFIG,
            stage_feature_columns_before_masks={
                "120": {"log": 15, "metric": 51, "trace": 24},
                "60": {"log": 15, "metric": 51, "trace": 24},
            },
        )
        with self.assertRaisesRegex(ValueError, "frozen 51/24"):
            verify_design_width(config)

    def test_requires_the_log_exclusion_rationale(self):
        config = dict(FROZEN_CONFIG, log_stage_excluded="")
        with self.assertRaisesRegex(ValueError, "log staged channels are excluded"):
            verify_design_width(config)


class StageFoldAuditTest(unittest.TestCase):
    def test_well_formed_audit_reports_160_inner_fits(self):
        fold_case_ids = _fold_case_ids()
        report = verify_fold_audit(_build_folds(fold_case_ids), fold_case_ids)
        self.assertEqual(report["outer_fold_count"], 5)
        self.assertEqual(report["inner_fit_count"], EXPECTED_INNER_FITS)
        self.assertEqual(report["inner_fit_count"], 160)
        self.assertEqual(report["test_overlap_max"], 0)
        self.assertEqual(report["test_group_overlap_max"], 0)
        self.assertEqual(report["inner_fit_validation_overlap_max"], 0)
        self.assertEqual(report["inner_fit_validation_group_overlap_max"], 0)
        self.assertEqual(report["selected_C_by_outer_fold"], [0.01] * 5)
        self.assertEqual(report["selected_onset_seconds_by_outer_fold"], [60] * 5)

    def test_selection_is_reconstructed_per_outer_fold(self):
        fold_case_ids = _fold_case_ids()
        winners = {
            "fold_0": (10.0, 120),
            "fold_1": (1.0, 60),
            "fold_2": (0.1, 120),
            "fold_3": (10.0, 60),
            "fold_4": (0.01, 120),
        }
        folds = _build_folds(
            fold_case_ids,
            score_fn=lambda outer, value, onset: (
                0.9 if (value, onset) == winners[outer] else 0.5
            ),
        )
        report = verify_fold_audit(folds, fold_case_ids)
        self.assertEqual(
            report["selected_C_by_outer_fold"], [10.0, 1.0, 0.1, 10.0, 0.01]
        )
        self.assertEqual(
            report["selected_onset_seconds_by_outer_fold"], [120, 60, 120, 60, 120]
        )

    def test_rejects_a_missing_outer_fold(self):
        fold_case_ids = _fold_case_ids()
        folds = _build_folds(fold_case_ids)[:4]
        with self.assertRaisesRegex(ValueError, "five outer folds"):
            verify_fold_audit(folds, fold_case_ids)

    def test_rejects_outer_train_test_overlap(self):
        fold_case_ids = _fold_case_ids()
        folds = _build_folds(fold_case_ids)
        folds[2]["train_test_overlap"] = 1
        with self.assertRaisesRegex(ValueError, "outer fold leakage"):
            verify_fold_audit(folds, fold_case_ids)

    def test_rejects_outer_group_overlap(self):
        fold_case_ids = _fold_case_ids()
        folds = _build_folds(fold_case_ids)
        folds[0]["train_test_group_overlap"] = 2
        with self.assertRaisesRegex(ValueError, "outer fold leakage"):
            verify_fold_audit(folds, fold_case_ids)

    def test_rejects_inner_fit_validation_overlap(self):
        fold_case_ids = _fold_case_ids()
        folds = _build_folds(fold_case_ids)
        folds[1]["inner_candidates"][3]["inner_folds"][2][
            "fit_validation_overlap"
        ] = 1
        with self.assertRaisesRegex(ValueError, "inner fold leakage"):
            verify_fold_audit(folds, fold_case_ids)

    def test_rejects_a_test_digest_that_does_not_match_the_frozen_split(self):
        fold_case_ids = _fold_case_ids()
        folds = _build_folds(fold_case_ids)
        folds[4]["test_case_ids_sha256"] = _id_digest(("case-9999",))
        with self.assertRaisesRegex(ValueError, "outer test IDs"):
            verify_fold_audit(folds, fold_case_ids)

    def test_rejects_an_inner_fit_digest_that_leaks_a_validation_case(self):
        fold_case_ids = _fold_case_ids()
        folds = _build_folds(fold_case_ids)
        inner = folds[0]["inner_candidates"][0]["inner_folds"][0]
        leaked = tuple(
            sorted(
                set(fold_case_ids["fold_1"]) | set(fold_case_ids["fold_2"])
            )
        )
        inner["fit_case_ids_sha256"] = _id_digest(leaked)
        with self.assertRaisesRegex(ValueError, "inner fit IDs"):
            verify_fold_audit(folds, fold_case_ids)

    def test_rejects_a_recorded_selection_that_ignores_the_tie_break(self):
        fold_case_ids = _fold_case_ids()
        folds = _build_folds(fold_case_ids)
        folds[3]["selected_C"] = 10.0
        with self.assertRaisesRegex(ValueError, "selected C"):
            verify_fold_audit(folds, fold_case_ids)

    def test_rejects_a_recorded_onset_that_ignores_the_tie_break(self):
        fold_case_ids = _fold_case_ids()
        folds = _build_folds(fold_case_ids)
        folds[3]["selected_onset_seconds"] = 120
        with self.assertRaisesRegex(ValueError, "selected onset"):
            verify_fold_audit(folds, fold_case_ids)

    def test_rejects_a_truncated_inner_rotation(self):
        fold_case_ids = _fold_case_ids()
        folds = _build_folds(fold_case_ids)
        folds[0]["inner_candidates"][0]["inner_folds"].pop()
        with self.assertRaisesRegex(ValueError, "four inner folds"):
            verify_fold_audit(folds, fold_case_ids)

    def test_rejects_an_inner_rotation_that_uses_the_outer_test_fold(self):
        fold_case_ids = _fold_case_ids()
        folds = _build_folds(fold_case_ids)
        inner = folds[0]["inner_candidates"][0]["inner_folds"][0]
        inner["validation_fold"] = folds[0]["outer_fold"]
        with self.assertRaisesRegex(ValueError, "four outer-train folds"):
            verify_fold_audit(folds, fold_case_ids)

    def test_rejects_a_repeated_outer_fold(self):
        fold_case_ids = _fold_case_ids()
        folds = _build_folds(fold_case_ids)
        folds[4] = folds[3]
        with self.assertRaisesRegex(ValueError, "repeats an outer fold"):
            verify_fold_audit(folds, fold_case_ids)


if __name__ == "__main__":
    unittest.main()
