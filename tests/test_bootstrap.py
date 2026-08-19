import unittest

from src.evaluation import paired_root_macro_bootstrap


class PairedBootstrapTest(unittest.TestCase):
    def test_root_macro_point_and_distribution_are_deterministic(self):
        arguments = {
            "case_ids": ("c1", "c2", "c3", "c4"),
            "root_services": ("a", "a", "b", "b"),
            "resampling_units": ("g1", "g1", "g2", "g3"),
            "actual_metrics": {
                "AC@1": (1.0, 0.0, 1.0, 1.0),
                "Avg@5": (1.0, 0.4, 0.8, 1.0),
            },
            "reference_metrics": {
                "AC@1": (0.0, 0.0, 1.0, 0.0),
                "Avg@5": (0.2, 0.4, 0.8, 0.0),
            },
            "iterations": 500,
            "random_seed": 7,
            "batch_size": 31,
        }
        first = paired_root_macro_bootstrap(**arguments)
        second = paired_root_macro_bootstrap(**arguments)
        self.assertEqual(first, second)
        self.assertAlmostEqual(
            first["metric_results"]["AC@1"]["point_delta"], 0.5
        )
        self.assertAlmostEqual(
            first["metric_results"]["Avg@5"]["point_delta"], 0.45
        )
        self.assertEqual(first["resampling_unit_count"], 3)

    def test_rejects_misaligned_inputs(self):
        with self.assertRaises(ValueError):
            paired_root_macro_bootstrap(
                ("c1",),
                ("a",),
                ("g",),
                {"m": (1.0,)},
                {"other": (0.0,)},
            )


if __name__ == "__main__":
    unittest.main()
