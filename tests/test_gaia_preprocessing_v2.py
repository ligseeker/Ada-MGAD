import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


TRACE_SLOTS = [
    "200_count_log",
    "200_mean_latency_log",
    "300_count_log",
    "300_mean_latency_log",
    "400_count_log",
    "400_mean_latency_log",
    "500_count_log",
    "500_mean_latency_log",
]


def _valid_schema():
    stable = ["stable_template_{}".format(index) for index in range(17)]
    levels = ["INFO", "WARNING", "ERROR", "DEBUG", "UNKNOWN"]
    return {
        "schema_version": "gaia_ad_preprocessing_v2",
        "status": "FROZEN",
        "fit_split": "train",
        "decision_inputs": ["train"],
        "gt_labels_used": False,
        "test_used_for_selection": False,
        "source_binding": {
            "config_sha256": "a" * 64,
            "policy_sha256": "b" * 64,
            "audit_sha256": "c" * 64,
        },
        "metric": {
            "ordered_slots": (
                ["metric_{:02d}".format(index) for index in range(45)]
                + [
                    "global_observed_fraction",
                    "host_applicable",
                    "host_observed_fraction",
                ]
            ),
        },
        "logs": {
            "stable_cluster_ids": list(range(17)),
            "ordered_slots": (
                stable
                + ["RARE_{}".format(level) for level in levels]
                + ["UNK_{}".format(level) for level in levels]
                + ["level_{}".format(level) for level in levels]
            ),
        },
        "traces": {
            "status_order": ["200", "300", "400", "500"],
            "statistic_order": ["count", "mean_latency"],
            "ordered_slots": TRACE_SLOTS,
            "directed_edges": [["dbservice1", "webservice1"]],
        },
    }


class FrozenSchemaFirewallTests(unittest.TestCase):
    def test_valid_schema_loads_with_frozen_dimensions_and_file_hash(self):
        import hashlib

        from src.e2e.gaia_preprocessing import load_frozen_preprocessing_schema

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "schema.json"
            raw = json.dumps(_valid_schema()).encode("utf-8")
            path.write_bytes(raw)

            schema = load_frozen_preprocessing_schema(path)

            self.assertEqual(schema.dimensions, {"raw_node": 48, "log_len": 32, "raw_edge": 8})
            self.assertEqual(schema.sha256, hashlib.sha256(raw).hexdigest())

    def test_schema_rejects_test_as_decision_input(self):
        from src.e2e.gaia_preprocessing import load_frozen_preprocessing_schema

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "schema.json"
            payload = _valid_schema()
            payload["decision_inputs"] = ["train", "test"]
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "decision_inputs"):
                load_frozen_preprocessing_schema(path)

    def test_schema_rejects_missing_metric_observability_slots(self):
        from src.e2e.gaia_preprocessing import load_frozen_preprocessing_schema

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "schema.json"
            payload = _valid_schema()
            payload["metric"]["ordered_slots"][-1] = "metric_replacement"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "observability"):
                load_frozen_preprocessing_schema(path)

    def test_bound_policy_drift_is_rejected(self):
        import hashlib

        from src.e2e.gaia_preprocessing import (
            load_frozen_preprocessing_schema,
            verify_frozen_schema_sources,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {}
            for name in ("config", "policy", "audit"):
                path = root / "{}.json".format(name)
                path.write_text('{"name":"%s"}' % name, encoding="utf-8")
                sources[name] = path
            payload = _valid_schema()
            for name, path in sources.items():
                payload["source_binding"]["{}_sha256".format(name)] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            schema_path = root / "schema.json"
            schema_path.write_text(json.dumps(payload), encoding="utf-8")
            schema = load_frozen_preprocessing_schema(schema_path)
            sources["policy"].write_text('{"name":"changed"}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "policy_sha256"):
                verify_frozen_schema_sources(
                    schema,
                    config_path=sources["config"],
                    policy_path=sources["policy"],
                    audit_path=sources["audit"],
                )


class MetricTransformTests(unittest.TestCase):
    def test_counter_rate_rejects_resets_and_long_gaps(self):
        from src.e2e.gaia_preprocessing.metric import derive_reset_aware_rate

        timestamps = np.asarray([0, 30_000, 60_000, 180_000, 210_000], dtype=np.int64)
        values = np.asarray([10.0, 40.0, 5.0, 125.0, 155.0], dtype=np.float64)

        result = derive_reset_aware_rate(timestamps, values, max_gap_ms=60_000)

        np.testing.assert_allclose(result[[1, 4]], np.asarray([1.0, 1.0]))
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[2]))
        self.assertTrue(np.isnan(result[3]))

    def test_forward_fill_is_age_bounded_and_split_local(self):
        from src.e2e.gaia_preprocessing.metric import forward_fill_by_age

        timestamps = np.arange(5, dtype=np.int64) * 30_000
        values = np.asarray([2.0, np.nan, np.nan, 4.0, np.nan])

        result = forward_fill_by_age(
            timestamps,
            values,
            max_age_ms=60_000,
            split_slices=(slice(0, 2), slice(2, 5)),
        )

        np.testing.assert_allclose(result[[0, 1, 3, 4]], [2.0, 2.0, 4.0, 4.0])
        self.assertTrue(np.isnan(result[2]))

    def test_train_scaler_clips_test_and_preserves_observed_mask(self):
        from src.e2e.gaia_preprocessing.metric import (
            apply_metric_scaler,
            fit_metric_scaler,
        )

        scaler = fit_metric_scaler(np.asarray([0.0, 1.0, 2.0, 3.0]))
        transformed, observed = apply_metric_scaler(
            np.asarray([0.0, np.nan, 100.0]), scaler
        )

        np.testing.assert_allclose(transformed, [0.0, 0.5, 1.0], atol=1e-6)
        np.testing.assert_array_equal(observed, [True, False, True])
        self.assertAlmostEqual(scaler.lower, 0.03)
        self.assertAlmostEqual(scaler.upper, 2.97)

    def test_multicore_aggregation_exposes_short_single_core_hotspot(self):
        from src.e2e.gaia_preprocessing.metric import aggregate_multicore

        cores = np.asarray([[10.0, 10.0, 10.0, 90.0]])
        aggregated = aggregate_multicore(cores)

        np.testing.assert_allclose(aggregated["mean"], [30.0])
        np.testing.assert_allclose(aggregated["max"], [90.0])
        self.assertEqual(set(aggregated), {"mean", "max"})

    def test_uniform_tensor_keeps_host_not_applicable_distinct_from_missing(self):
        from src.e2e.gaia_preprocessing.metric import assemble_metric_tensor

        global_values = np.asarray([[[0.0, 0.5], [0.2, 0.5]]], dtype=np.float32)
        global_observed = np.asarray([[[True, False], [True, True]]])
        host_values = np.asarray([[[0.7], [np.nan]]], dtype=np.float32)
        host_observed = np.asarray([[[True], [False]]])

        result = assemble_metric_tensor(
            global_values=global_values,
            global_observed=global_observed,
            host_values=host_values,
            host_observed=host_observed,
            host_applicable=np.asarray([True, False]),
            neutral_value=0.5,
        )

        self.assertEqual(result.shape, (1, 2, 6))
        np.testing.assert_allclose(result[0, 0, -3:], [0.5, 1.0, 1.0])
        np.testing.assert_allclose(result[0, 1, -3:], [1.0, 0.0, 0.0])
        self.assertEqual(result[0, 1, 2], 0.5)


class LogTransformTests(unittest.TestCase):
    def test_stable_rare_and_unseen_are_routed_without_log_total_slot(self):
        from src.e2e.gaia_preprocessing.logs import route_log_bin

        routed = route_log_bin(
            events=[(11, "ERROR"), (22, "ERROR"), (None, "ERROR")],
            stable_cluster_ids=[11, 33],
        )

        self.assertEqual(routed.counts[routed.slot_names.index("stable_template_11")], 1)
        self.assertEqual(routed.counts[routed.slot_names.index("RARE_ERROR")], 1)
        self.assertEqual(routed.counts[routed.slot_names.index("UNK_ERROR")], 1)
        self.assertEqual(routed.counts[routed.slot_names.index("level_ERROR")], 3)
        self.assertEqual(routed.audit_total, 3)
        self.assertNotIn("log_total", routed.slot_names)
        self.assertEqual(len(routed.slot_names), 17)

    def test_zero_support_unk_scalers_use_frozen_train_fallback_chain(self):
        from src.e2e.gaia_preprocessing.logs import fit_log_scalers, log_slot_names

        names = log_slot_names([11])
        counts = np.zeros((3, len(names)), dtype=np.float64)
        counts[:, names.index("stable_template_11")] = [1, 2, 1]
        counts[:, names.index("RARE_ERROR")] = [0, 4, 0]
        counts[:, names.index("level_ERROR")] = [5, 10, 2]
        counts[:, names.index("level_WARNING")] = [0, 3, 0]

        fitted = fit_log_scalers(counts, names)

        self.assertEqual(fitted.fallback_sources["UNK_ERROR"], "RARE_ERROR")
        self.assertEqual(fitted.fallback_sources["UNK_WARNING"], "level_WARNING")
        self.assertEqual(
            fitted.fallback_sources["UNK_DEBUG"], "__pooled_train_messages__"
        )
        self.assertGreater(fitted.scales["UNK_ERROR"], 0.0)


class TraceTransformTests(unittest.TestCase):
    def test_parent_resolution_uses_trace_id_and_outputs_count_plus_mean(self):
        from src.e2e.gaia_preprocessing.traces import aggregate_trace_split

        rows = [
            {
                "trace_id": "trace-a",
                "span_id": "same-span",
                "parent_id": "",
                "service": "dbservice1",
                "start_time_ms": 0,
                "end_time_ms": 100,
                "status_code": "200",
            },
            {
                "trace_id": "trace-b",
                "span_id": "same-span",
                "parent_id": "",
                "service": "redisservice1",
                "start_time_ms": 0,
                "end_time_ms": 100,
                "status_code": "200",
            },
            {
                "trace_id": "trace-a",
                "span_id": "child",
                "parent_id": "same-span",
                "service": "webservice1",
                "start_time_ms": 1_000,
                "end_time_ms": 1_100,
                "status_code": "500",
            },
        ]
        services = ["dbservice1", "redisservice1", "webservice1"]

        result = aggregate_trace_split(
            rows,
            services=services,
            grid_ms=np.asarray([0, 30_000], dtype=np.int64),
            split_start_ms=0,
            split_end_ms=60_000,
            frozen_directed_edges=[("dbservice1", "webservice1")],
        )

        source = services.index("dbservice1")
        destination = services.index("webservice1")
        self.assertEqual(result.features[0, source, destination, 6], 1.0)
        self.assertAlmostEqual(result.features[0, source, destination, 7], 0.1)
        self.assertEqual(result.diagnostics["matched_cross_service_rows"], 1)
        self.assertEqual(result.diagnostics["ambiguous_parent_rows"], 0)
        self.assertEqual(result.features.shape[-1], 8)

    def test_ambiguous_parent_is_rejected_and_unseen_edge_is_diagnostic_only(self):
        from src.e2e.gaia_preprocessing.traces import aggregate_trace_split

        rows = [
            {"trace_id": "a", "span_id": "p", "parent_id": "", "service": "db", "start_time_ms": 0, "end_time_ms": 10, "status_code": "200"},
            {"trace_id": "a", "span_id": "p", "parent_id": "", "service": "redis", "start_time_ms": 0, "end_time_ms": 10, "status_code": "200"},
            {"trace_id": "a", "span_id": "c", "parent_id": "p", "service": "web", "start_time_ms": 20, "end_time_ms": 30, "status_code": "500"},
            {"trace_id": "b", "span_id": "q", "parent_id": "", "service": "redis", "start_time_ms": 0, "end_time_ms": 10, "status_code": "200"},
            {"trace_id": "b", "span_id": "d", "parent_id": "q", "service": "web", "start_time_ms": 20, "end_time_ms": 30, "status_code": "500"},
        ]

        result = aggregate_trace_split(
            rows,
            services=["db", "redis", "web"],
            grid_ms=np.asarray([0, 30_000]),
            split_start_ms=0,
            split_end_ms=60_000,
            frozen_directed_edges=[("db", "web")],
        )

        self.assertEqual(result.diagnostics["ambiguous_parent_rows"], 1)
        self.assertEqual(result.diagnostics["unseen_directed_edge_rows"], 1)
        self.assertEqual(result.observed_directed_edges, (("redis", "web"),))

    def test_trace_scaler_falls_back_from_sparse_edge_to_train_status_pool(self):
        from src.e2e.gaia_preprocessing.traces import fit_trace_scalers

        services = ["db", "redis", "web"]
        features = np.zeros((3, 3, 3, 8), dtype=np.float32)
        features[:, 0, 2, 6] = [1.0, 2.0, 1.0]
        features[:, 0, 2, 7] = [0.1, 0.2, 0.15]

        fitted = fit_trace_scalers(
            features,
            services=services,
            frozen_directed_edges=[("db", "web"), ("redis", "web")],
            min_positive_bins=2,
        )

        sparse_key = "redis->web::500_count_log"
        self.assertEqual(
            fitted.fallback_sources[sparse_key], "__status_pool__::500_count_log"
        )
        self.assertGreater(fitted.scales[sparse_key], 0.0)


class MaterializationBoundaryTests(unittest.TestCase):
    def test_legacy_materializer_refuses_v2_schema_declaration(self):
        from src.e2e.ad_preprocess import build_ad_data

        with self.assertRaisesRegex(ValueError, "V2 materializer"):
            build_ad_data(
                {"ad_preprocessing": {"schema_version": "gaia_ad_preprocessing_v2"}},
                Path("."),
                Path("unused-data"),
                Path("unused-artifacts"),
            )

    def test_modalities_must_match_frozen_dimensions_before_publish(self):
        from src.e2e.gaia_preprocessing import (
            load_frozen_preprocessing_schema,
            validate_transformed_modalities,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "schema.json"
            path.write_text(json.dumps(_valid_schema()), encoding="utf-8")
            schema = load_frozen_preprocessing_schema(path)
            metric = np.zeros((2, 10, 48), dtype=np.float32)
            logs = np.zeros((2, 10, 32), dtype=np.float32)
            trace = np.zeros((2, 10, 10, 8), dtype=np.float32)

            summary = validate_transformed_modalities(
                schema=schema,
                metric=metric,
                logs=logs,
                trace=trace,
                service_count=10,
            )

            self.assertEqual(summary["dimensions"], schema.dimensions)
            with self.assertRaisesRegex(ValueError, "trace shape"):
                validate_transformed_modalities(
                    schema=schema,
                    metric=metric,
                    logs=logs,
                    trace=trace[..., :4],
                    service_count=10,
                )


if __name__ == "__main__":
    unittest.main()
