"""Read-only observer of the frozen training path; no model/optimizer updates."""

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from .bin_trigger_decoder import independent_bin_counts
from .protocol import sha256_file, write_json
from .system_trigger import to_builtin


def rng_snapshot():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            [value.clone() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else [])


def assert_same_rng(before, after):
    if (before[0] != after[0] or before[1][0] != after[1][0]
            or not np.array_equal(before[1][1], after[1][1])
            or before[1][2:] != after[1][2:]
            or not torch.equal(before[2], after[2])
            or len(before[3]) != len(after[3])
            or any(not torch.equal(a, b) for a, b in zip(before[3], after[3]))):
        raise ValueError("epoch observer changed an RNG state")


def padded_fit_counts(dataset, batch_size):
    """Count the exact padded labels consumed by the unchanged loader."""
    from .ad_data import PaddedSequentialSampler
    labels = dataset.labels_at(list(PaddedSequentialSampler(dataset, batch_size)))
    batches = labels.reshape(-1, batch_size)
    return {"unique_windows": len(dataset), "padded_windows": len(labels),
            "positive": int((labels == 1).sum()), "negative": int((labels == 0).sum()),
            "ignore": int((labels == 2).sum()), "effective_bins": int((labels != 2).sum()),
            "usable_batches": int((batches != 2).any(axis=1).sum()),
            "loss_aggregation": "unweighted mean of each usable batch's masked mean loss"}


def score_diagnostics(output, gt, threshold):
    times = output["prediction_available_time"]
    scores = output["system_score"]
    labels = output["trigger_label"]
    high = scores >= threshold
    tp, fp, fn = independent_bin_counts(times, scores, np.sort(gt.start_ms.to_numpy()), threshold)
    count = tp + fp
    metrics = {"true_positive_events": tp, "false_positive_events": fp,
               "false_negative_events": fn, "gt_events": len(gt), "predicted_events": count,
               "event_precision": tp / count if count else 0.0,
               "event_recall": tp / len(gt) if len(gt) else 0.0,
               "event_f1": 2 * tp / (len(gt) + count) if len(gt) + count else 0.0}
    edges = np.diff(np.r_[False, high, False].astype(np.int8))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return {"bin_metrics_at_merged_threshold": metrics,
            "high_score_by_target": {
                name: {"bins": int((labels == value).sum()),
                       "high_bins": int((high & (labels == value)).sum()),
                       "high_fraction": float(high[labels == value].mean())
                       if (labels == value).any() else None}
                for name, value in (("POS", 1), ("NEG", 0), ("IGNORE", 2))},
            "high_score_segments": {"count": len(lengths),
                                    "max_seconds": int(lengths.max() * 30) if len(lengths) else 0,
                                    "total_seconds": int(lengths.sum() * 30)}}


class EpochAudit:
    """Persist existing outputs and compare archived early training epochs.

    Counts, thresholds and selection must replay exactly. Floating-point loss
    and weights must be within fixed absolute 1e-6; exactness is also reported.
    This numerical tolerance is an integrity gate, not a performance gate.
    """

    def __init__(self, output_dir, old_run, gt, fit_counts, enforce_historical_replay=True):
        self.directory = Path(output_dir) / "epochs"
        self.directory.mkdir()
        self.old_run = Path(old_run)
        self.old_history = json.loads((self.old_run / "training_log.json").read_text())["history"]
        self.old_selection = json.loads((self.old_run / "validation_selection.json").read_text())
        self.old_weights = torch.load(self.old_selection["checkpoint"]["path"], map_location="cpu")
        self.gt = gt
        self.fit_counts = fit_counts
        self.records = []
        self.enforce_historical_replay = enforce_historical_replay

    def __call__(self, model, output, entry, learning_rates, next_learning_rates):
        before = rng_snapshot()
        epoch = int(entry["epoch"])
        directory = self.directory / "epoch-{:02d}".format(epoch)
        directory.mkdir()
        weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        torch.save(weights, directory / "model_state.pt")
        np.savez(directory / "validation_outputs.npz",
                 **{key: value for key, value in output.items() if key != "split"})
        replay = {"applicable": epoch < len(self.old_history), "passed": True}
        if replay["applicable"]:
            previous = self.old_history[epoch]
            errors = {key: abs(float(entry[key]) - float(previous[key]))
                      for key in ("train_loss", "train_bce", "train_graph_regularization")}
            replay.update({"loss_absolute_errors": errors,
                           "losses_exact": all(value == 0 for value in errors.values()),
                           "threshold_exact": entry["validation_threshold"] == previous["validation_threshold"],
                           "selection_key_exact": entry["selection_key"] == previous["selection_key"],
                           "merged_metrics_exact": entry["validation_metrics"] == previous["validation_metrics"]})
            replay["passed"] = (max(errors.values()) <= 1e-6 and replay["threshold_exact"]
                                and replay["selection_key_exact"] and replay["merged_metrics_exact"])
        if epoch == self.old_selection["selected_epoch"]:
            if weights.keys() != self.old_weights.keys():
                raise ValueError("archived checkpoint has different state keys")
            error = max(float((value - self.old_weights[name]).abs().max())
                        for name, value in weights.items())
            frozen = np.genfromtxt(self.old_run / "validation_predictions.csv", delimiter=",", names=True)
            replay.update({"selected_weight_max_abs_error": error,
                           "selected_weights_exact": error == 0,
                           "selected_logits_exact": np.array_equal(output["logits"],
                                                                    frozen["system_trigger_logit"])})
            replay["passed"] = replay["passed"] and error <= 1e-6 and replay["selected_logits_exact"]
        record = {**entry, "learning_rates_used": learning_rates,
                  "learning_rates_next_epoch": next_learning_rates,
                  "fit_consumed_labels": self.fit_counts,
                  ("early_trajectory_replay" if self.enforce_historical_replay
                   else "historical_trajectory_comparison"): replay,
                  "diagnostics": score_diagnostics(output, self.gt, entry["validation_threshold"]),
                  "checkpoint_sha256": sha256_file(directory / "model_state.pt"),
                  "validation_outputs_sha256": sha256_file(directory / "validation_outputs.npz")}
        assert_same_rng(before, rng_snapshot())
        record["observer_rng_unchanged"] = True
        write_json(directory / "epoch_audit.json", to_builtin(record))
        self.records.append(record)
        write_json(self.directory.parent / "epoch_progress.json", to_builtin({"history": self.records}))
        bin_metrics = record["diagnostics"]["bin_metrics_at_merged_threshold"]
        logging.info("budget audit epoch %d: lr=%s bin TP/FP/FN=%d/%d/%d historical_match=%s", epoch,
                     learning_rates, bin_metrics["true_positive_events"], bin_metrics["false_positive_events"],
                     bin_metrics["false_negative_events"], replay["passed"])
        if self.enforce_historical_replay and not replay["passed"]:
            raise ValueError("archived early trajectory did not replay at epoch {}: {}".format(epoch, replay))
