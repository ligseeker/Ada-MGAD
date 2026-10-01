import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.p6.tcn_replication import check_replication_config
from scripts.p6.analyze_c0_tcn_replication import matching_identity, temporal_pairs


def test_replication_cannot_change_recipe_or_pick_an_extra_seed():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/e2e/gaia_p6_c0_tcn_replication_v1.json").read_text())
    reference = json.loads(Path(config["replication"]["reference_config"]).read_text())
    for seed in (17, 2026):
        assert check_replication_config(config, reference, seed) == seed
    for seed in (None, 42, 123):
        with pytest.raises(ValueError):
            check_replication_config(config, reference, seed)
    for key in ("trigger_label", "training", "model", "threshold"):
        altered = copy.deepcopy(config)
        altered[key]["unregistered_change"] = True
        with pytest.raises(ValueError):
            check_replication_config(altered, reference, 17)


def test_time_partition_uses_gt_start_and_global_fp_without_rematching():
    gt = pd.DataFrame({"case_id": ["crossing", "later"], "start_ms": [49, 80]})
    baseline = pd.DataFrame({"case_id": ["crossing", "later"],
                             "match_status": ["miss", "matched"], "t_hat": [None, 85]})
    # crossing is matched after the artificial half boundary, still belongs to
    # the first GT half. The unrelated FP belongs to the second prediction half.
    candidate = pd.DataFrame({"case_id": ["crossing", "later", None],
        "match_status": ["matched", "matched", "false_alarm"], "t_hat": [55, 85, 95]})
    result = temporal_pairs(gt, baseline, candidate, 0, 100)
    assert result["first_half"]["candidate_tp"] == 1
    assert result["first_half"]["delta_recall"] == 1
    assert result["second_half"]["delta_recall"] == 0
    assert result["first_half"]["candidate_global_fp"] == 0
    assert result["second_half"]["candidate_global_fp"] == 1


def test_matching_validator_keeps_misses_and_rejects_lost_gt_rows():
    gt = pd.DataFrame({"case_id": ["a", "b"], "source_index": [1, 2],
                       "service": ["s", "s"], "fault_type": ["f", "f"],
                       "start_ms": [10, 40], "end_ms": [20, 50]})
    matching = gt.rename(columns={"start_ms": "gt_start_ms", "end_ms": "gt_end_ms",
                                 "service": "gt_service"})
    matching["match_status"] = ["matched", "miss"]
    metrics = {"true_positive_events": 1, "false_negative_events": 1, "false_positive_events": 0}
    matching_identity(matching, gt, metrics)
    with pytest.raises(ValueError):
        matching_identity(matching.iloc[:1], gt, metrics)
