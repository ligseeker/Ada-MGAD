"""P6-C1-v2 C2 layer: ledger hierarchy, penalized P/R/F1, latency and strata."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.e2e import c1_v2_c2 as c2_module
from src.e2e.c1_v2_c2 import evaluate_c2_full_diagnosis, quote_frozen_stage1

START = 1626963120000


def _ranking_at(service: str, position: int) -> str:
    """A complete canonical ranking with ``service`` at ``position``."""
    others = [item for item in c2_module.GAIA_SERVICES if item != service]
    order = others[:position - 1] + [service] + others[position - 1:]
    return json.dumps(order, separators=(",", ":"))


def _frames():
    matching = pd.DataFrame([
        {"prediction_id": "p1", "match_status": "matched", "case_id": "c1",
         "gt_service": "mobservice1", "fault_type": "login_failure",
         "gt_start_ms": START, "gt_end_ms": START + 10000, "t_hat": START + 30000},
        {"prediction_id": "p2", "match_status": "matched", "case_id": "c2",
         "gt_service": "mobservice2", "fault_type": "memory_anomalies",
         "gt_start_ms": START, "gt_end_ms": START + 600000, "t_hat": START + 60000},
        {"prediction_id": "p3", "match_status": "matched", "case_id": "c3",
         "gt_service": "webservice1", "fault_type": "file_moving",
         "gt_start_ms": START + 600000, "gt_end_ms": START + 630000, "t_hat": START + 660000},
        {"prediction_id": "p4", "match_status": "matched", "case_id": "c4",
         "gt_service": "logservice1", "fault_type": "login_failure",
         "gt_start_ms": START + 900000, "gt_end_ms": START + 945000, "t_hat": START + 960000},
        {"prediction_id": None, "match_status": "miss", "case_id": "c5",
         "gt_service": "dbservice1", "fault_type": "login_failure",
         "gt_start_ms": START + 1200000, "gt_end_ms": START + 1220000, "t_hat": None},
        {"prediction_id": "p5", "match_status": "false_alarm", "case_id": None,
         "gt_service": None, "fault_type": None, "gt_start_ms": None, "gt_end_ms": None,
         "t_hat": START + 1500000},
    ])
    scope = pd.DataFrame([
        {"prediction_id": "p1", "scope_status": "legal", "ranking_status": "complete",
         "ranking_b": _ranking_at("mobservice1", 1), "ranking_c": _ranking_at("mobservice1", 2)},
        {"prediction_id": "p2", "scope_status": "legal", "ranking_status": "complete",
         "ranking_b": _ranking_at("mobservice2", 3), "ranking_c": _ranking_at("mobservice2", 1)},
        {"prediction_id": "p3", "scope_status": "illegal_context",
         "ranking_status": "not_applicable", "ranking_b": "", "ranking_c": ""},
        {"prediction_id": "p4", "scope_status": "legal", "ranking_status": "failed",
         "ranking_b": "", "ranking_c": _ranking_at("logservice1", 6)},
        {"prediction_id": "p5", "scope_status": "legal", "ranking_status": "complete",
         "ranking_b": _ranking_at("webservice2", 1), "ranking_c": _ranking_at("webservice2", 1)},
    ])
    registry = pd.DataFrame([
        {"case_id": "c1", "source_index": 1, "service": "mobservice1",
         "fault_type": "login_failure", "start_ms": START, "end_ms": START + 10000,
         "detector_domain": True},
        {"case_id": "c2", "source_index": 2, "service": "mobservice2",
         "fault_type": "memory_anomalies", "start_ms": START, "end_ms": START + 600000,
         "detector_domain": True},
        {"case_id": "c3", "source_index": 3, "service": "webservice1",
         "fault_type": "file_moving", "start_ms": START + 600000, "end_ms": START + 630000,
         "detector_domain": True},
        {"case_id": "c4", "source_index": 4, "service": "logservice1",
         "fault_type": "login_failure", "start_ms": START + 900000, "end_ms": START + 945000,
         "detector_domain": True},
        {"case_id": "c5", "source_index": 5, "service": "dbservice1",
         "fault_type": "login_failure", "start_ms": START + 1200000, "end_ms": START + 1220000,
         "detector_domain": True},
    ])
    return matching, scope, registry


def _evaluate():
    matching, scope, registry = _frames()
    return evaluate_c2_full_diagnosis(scope=scope, registry=registry, matching=matching,
                                      test_interval_ms=(START, START + 3600000))


class C1V2C2Test(unittest.TestCase):
    def test_ledger_closes_and_penalizes_failures(self):
        result = _evaluate()
        summary = result["summary"]
        arm_b = summary["arms"]["b"]
        arm_c = summary["arms"]["c"]
        self.assertEqual(summary["gt_population"], 5)
        self.assertEqual(summary["predicted_episodes"], 5)
        self.assertEqual(arm_b["ledger_categories"]["SUCCESS_TOP1"], 1)
        self.assertEqual(arm_b["ledger_categories"]["ROOT_OUTSIDE_TOP1"], 1)
        self.assertEqual(arm_b["ledger_categories"]["RCA_CONTEXT_INVALID"], 1)
        self.assertEqual(arm_b["ledger_categories"]["RCA_RANKING_MISSING"], 1)
        self.assertEqual(arm_b["ledger_categories"]["EVENT_MISSED"], 1)
        self.assertEqual(arm_b["ledger_categories"]["EVENT_FALSE_ALARM"], 1)
        self.assertEqual(arm_c["ledger_categories"]["SUCCESS_TOP1"], 1)
        self.assertEqual(arm_c["ledger_categories"]["ROOT_OUTSIDE_TOP1"], 1)
        self.assertEqual(arm_c["ledger_categories"]["ROOT_OUTSIDE_TOP5"], 1)
        self.assertTrue(arm_b["ledger_closure"] and arm_c["ledger_closure"])
        at1 = arm_b["metrics"]["@1"]
        self.assertEqual(at1["diagnosis_true_positive"], 1)
        self.assertEqual(at1["diagnosis_false_positive"], 4)
        self.assertEqual(at1["diagnosis_false_negative"], 4)
        self.assertAlmostEqual(at1["f1"], 0.2)
        at3 = arm_b["metrics"]["@3"]
        self.assertEqual(at3["diagnosis_true_positive"], 2)
        self.assertAlmostEqual(at3["recall"], 0.4)
        self.assertEqual(arm_c["metrics"]["@1"]["diagnosis_true_positive"], 1)
        ledger = result["ledger"]
        self.assertEqual(set(ledger["category"]),
                         set(c2_module.LEDGER_CATEGORIES) - {"ROOT_OUTSIDE_TOP3"})
        for arm in ("b", "c"):
            self.assertEqual(len(ledger.loc[ledger["arm"] == arm]), 6)

    def test_latency_definitions_and_strata_keep_empty_groups(self):
        summary = _evaluate()["summary"]
        latency = summary["latency"]
        self.assertEqual(latency["final_diagnosis_compute_latency"], "UNMEASURED")
        self.assertAlmostEqual(latency["rca_data_ready_latency_seconds"]["mean"],
                               latency["detection_latency_seconds"]["mean"] + 300.0)
        strata = summary["strata"]
        self.assertEqual(set(strata["duration"]),
                         {"le_15s", "15_30s", "30_60s", "60_300s", "gt_300s"})
        self.assertEqual(strata["duration"]["gt_300s"]["n"], 1)
        self.assertEqual(set(strata["onset_multiplicity"]),
                         {"1", "2", "3", "4_plus", "histogram"})
        self.assertEqual(sum(strata["onset_multiplicity"]["histogram"].values()), 5)
        self.assertIn("mobservice1", strata["root_service"])
        self.assertIn("login_failure", strata["fault"])

    def test_missing_ranking_stays_in_the_denominator(self):
        arm_b = _evaluate()["summary"]["arms"]["b"]
        self.assertEqual(arm_b["counters"]["ranking_failure"], 1)
        self.assertEqual(arm_b["metrics"]["@5"]["diagnosis_recall_denominator"], 5)
        self.assertEqual(arm_b["metrics"]["@5"]["diagnosis_precision_denominator"], 5)

    def test_quote_frozen_stage1_binds_the_frozen_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_metrics.json"
            path.write_text(json.dumps({
                "test_event_metrics": {"event_precision": 0.9962, "event_recall": 0.7254,
                                       "event_f1": 0.8395},
                "frozen_validation_threshold": 0.9998264908790588, "selected_epoch": 8,
                "pre_registered_gate": {"precision": 0.9}, "p6_c0_decision": "BORDERLINE"}),
                encoding="utf-8")
            protocol = {"bindings": {"c0_test_metrics": {"path": str(path)}}}
            with patch.object(c2_module, "bound_path", return_value=path):
                quoted = quote_frozen_stage1(protocol)
            self.assertFalse(quoted["recomputed"])
            self.assertEqual(quoted["frozen_verdict"], "BORDERLINE")
            self.assertEqual(quoted["test_event_metrics"]["event_f1"], 0.8395)


if __name__ == "__main__":
    unittest.main()
