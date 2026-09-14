import unittest
from types import SimpleNamespace

from src.e2e.parallel import ordered_process_map
from scripts.p5.run_i1_ad import modality_workers


def _label(value):
    return value * value


class OrderedProcessMapTests(unittest.TestCase):
    def test_parallel_order_and_replay_are_deterministic(self):
        tasks = tuple(range(12))
        serial, _ = ordered_process_map(_label, tasks, workers=1)
        first, metadata = ordered_process_map(_label, tasks, workers=2, max_in_flight=3)
        second, _ = ordered_process_map(_label, tasks, workers=2, max_in_flight=3)
        self.assertEqual(serial, first)
        self.assertEqual(first, second)
        self.assertEqual(metadata["max_in_flight"], 3)
        self.assertEqual(metadata["result_order"], "submitted task order")

    def test_modality_worker_bounds_reject_zero_and_budget_overflow(self):
        config = {"preprocessing": {
            "metric_workers": 2, "log_workers": 2, "trace_workers": 2,
            "cpu_budget": 4, "chunk_rows": 1,
            "multiprocessing_start_method": "spawn", "rca_case_chunk_size": 1,
        }}
        valid = SimpleNamespace(workers=None, metric_workers=3, log_workers=2, trace_workers=1)
        self.assertEqual(modality_workers(config, valid), {"metric": 3, "logs": 2, "traces": 1})
        for name in ("metric_workers", "log_workers", "trace_workers"):
            invalid = SimpleNamespace(workers=None, metric_workers=None, log_workers=None, trace_workers=None)
            setattr(invalid, name, 0)
            with self.assertRaises(ValueError):
                modality_workers(config, invalid)
            invalid = SimpleNamespace(workers=None, metric_workers=None, log_workers=None, trace_workers=None)
            setattr(invalid, name, 5)
            with self.assertRaises(ValueError):
                modality_workers(config, invalid)


if __name__ == "__main__":
    unittest.main()
