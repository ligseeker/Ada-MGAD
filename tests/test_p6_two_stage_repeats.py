"""Fail-closed family locks and denominator preservation, using synthetic data."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.p6 import repeat_protocol as protocol
from scripts.p6 import run_two_stage_repeats as runner
from scripts.p6.repeat_rca import evaluate_seed, summarize
from src.e2e.protocol import GAIA_SERVICES, TemporalBlock
from src.e2e.bin_trigger_decoder import construct_bin_candidates
from src.e2e.system_trigger import system_score_frame, to_builtin


def complete_family(folder):
    folder.mkdir()
    config = protocol.read(protocol.ROOT / 'configs/e2e/gaia_p6_two_stage_repeats_v1.json')
    training = protocol.read(protocol.path(config['detector_config']))['training']
    protocol.write(folder / 'scope_lock.json', {'config': config, 'test': 'synthetic only'})
    for seed in protocol.SEEDS:
        for stage in ('detector', 'rca'):
            sub = folder / ('seed-%02d' % seed) / stage
            sub.mkdir(parents=True)
            for name in protocol.REQUIRED[stage]:
                file = sub / name; file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text('synthetic bytes, no production input')
            if stage == 'detector':
                checkpoint = sub / 'checkpoint/best_validation_event_f1.pt'
                (sub / 'validation_selection.json').write_text(json.dumps(dict(effective_seed=seed,
                    model_args=dict(random_seed=seed), training=training, model_class='WindowCausalTCNTrigger',
                    selected_validation_threshold=.5,
                    checkpoint=dict(path=str(checkpoint), sha256=protocol.sha(checkpoint)))))
                record = dict(seed=seed, threshold=.5, hash_seed=str(seed))
            else:
                record = dict(seed=seed, hash_seed='20260826',
                    scorer_sha256=config['frozen_rca']['model_sha256'], scorer_retrained=False,
                    scaler=None, anchor_backdate_ms=25621)
            (sub / 'method_record.json').write_text(json.dumps(record))
            protocol.seal(sub, status='COMPLETE', seed=seed,
                scope_sha256=protocol.sha(folder / 'scope_lock.json'), test_gt_or_matching_read=False)


def test_barrier_requires_all_ten_and_every_required_file(tmp_path):
    folder = tmp_path / 'family'; complete_family(folder)
    assert protocol.family_bindings(folder)
    last = folder / 'seed-10/rca'
    (last / 'completion_manifest.json').unlink()
    with pytest.raises(FileNotFoundError):
        protocol.family_bindings(folder)
    (last / 'features.npy').unlink()
    protocol.seal(last, status='COMPLETE', seed=10,
        scope_sha256=protocol.sha(folder / 'scope_lock.json'), test_gt_or_matching_read=False)
    with pytest.raises(ValueError, match='required prediction'):
        protocol.family_bindings(folder)


def test_barrier_rejects_changed_prediction_bytes_or_another_seed(tmp_path):
    folder = tmp_path / 'family'; complete_family(folder)
    file = folder / 'seed-05/rca/rankings.csv'; file.write_text('changed')
    with pytest.raises(ValueError, match='sealed stage changed'):
        protocol.family_bindings(folder)
    manifest = file.parent / 'completion_manifest.json'; manifest.unlink()
    protocol.seal(file.parent, status='COMPLETE', seed=6,
        scope_sha256=protocol.sha(folder / 'scope_lock.json'), test_gt_or_matching_read=False)
    with pytest.raises(ValueError, match='seed/stage'):
        protocol.family_bindings(folder)


def test_evaluation_requires_exact_committed_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, 'ROOT', tmp_path)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    file = tmp_path / 'lock.json'; protocol.write(file, {'family': protocol.SEEDS})
    subprocess.run(['git', 'add', 'lock.json'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=Synthetic Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'synthetic lock'], cwd=tmp_path, check=True)
    protocol.require_committed(file)
    file.write_text('{}')
    with pytest.raises(ValueError, match='commit the exact'):
        protocol.require_committed(file)


def test_barrier_rejects_another_seed_inside_resealed_selection(tmp_path):
    folder = tmp_path / 'family'; complete_family(folder)
    sub = folder / 'seed-03/detector'
    selected = protocol.read(sub / 'validation_selection.json')
    selected['model_args']['random_seed'] = 2
    (sub / 'validation_selection.json').write_text(json.dumps(selected))
    (sub / 'completion_manifest.json').unlink()
    protocol.seal(sub, status='COMPLETE', seed=3,
        scope_sha256=protocol.sha(folder / 'scope_lock.json'), test_gt_or_matching_read=False)
    with pytest.raises(ValueError, match='detector seed/checkpoint/recipe'):
        protocol.family_bindings(folder)


def test_child_interpreters_receive_startup_hash_seed_before_import(tmp_path, monkeypatch):
    output = tmp_path / 'new-family'
    launches = []
    def prepare(*args):
        output.mkdir()
        protocol.write(output / 'scope_lock.json', {'synthetic': True})
    monkeypatch.setattr(runner, 'prepare', prepare)
    monkeypatch.setattr(runner, 'verify_scope', lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, 'family_bindings', lambda *args: {})
    monkeypatch.setattr(runner.subprocess, 'run', lambda command, **kwargs: launches.append((command, kwargs)))
    args = SimpleNamespace(resume=False)
    config = dict(detector_python='synthetic-detector', rca_python='synthetic-rca')
    runner.predict(config, tmp_path / 'config.json', output, args)
    assert len(launches) == 20
    for seed, (detector, rca) in zip(protocol.SEEDS, zip(launches[::2], launches[1::2])):
        assert detector[0][0] == 'synthetic-detector'
        assert detector[1]['env']['PYTHONHASHSEED'] == str(seed)
        assert rca[0][0] == 'synthetic-rca'
        assert rca[1]['env']['PYTHONHASHSEED'] == '20260826'


def synthetic_predictions(empty=False):
    starts = np.array([310000, 400000, 500000, 700000])
    gt = pd.DataFrame(dict(case_id=['a', 'b', 'c', 'd'], source_index=np.arange(4), split='test',
        service=['dbservice1', 'mobservice1', 'webservice1', 'dbservice2'],
        fault_type='memory_anomalies', start_ms=starts, end_ms=starts + 1000))
    times = np.arange(300000, 900001, 30000)
    probabilities = np.where(np.isin(times, [330000, 480000, 720000]) & (not empty), .9, .1)
    scores = pd.DataFrame(dict(prediction_available_time=times, system_score=probabilities))
    episodes = construct_bin_candidates(system_score_frame('test', times, probabilities), .5)
    ranking = episodes[['prediction_id', 't_hat']].copy()
    ranking['anchor_ms'] = ranking.t_hat - 25621
    ranking['legal'] = ranking.t_hat.ne(720000)
    ranking['ranking'] = [json.dumps(list(GAIA_SERVICES)) if valid else '' for valid in ranking.legal]
    blocks = [TemporalBlock('fit', -600000, -300000), TemporalBlock('validation', -300000, 0),
              TemporalBlock('test', 0, 900000)]
    return episodes, scores, ranking, gt, blocks


def test_miss_false_alarm_and_illegal_rca_remain_in_e2e_denominator():
    episodes, scores, ranking, gt, blocks = synthetic_predictions()
    result, _, _, ledger = evaluate_seed(episodes, scores, ranking, gt, gt, blocks, .5)
    assert tuple(result['detector'][key] for key in
        ('true_positive_events', 'false_positive_events', 'false_negative_events')) == (2, 1, 2)
    diagnosis = result['diagnosis']
    assert diagnosis['legal_matched'] == 1
    assert diagnosis['matched_rca']['AC@1'] == 1.
    assert diagnosis['e2e']['@1'] == dict(tp=1, fp=2, fn=3, precision=1/3, recall=1/4, f1=2/7)
    assert diagnosis['failure']['RCA_CONTEXT_INVALID'] == 1
    assert diagnosis['failure']['EVENT_FALSE_ALARM'] == 1
    assert diagnosis['failure']['EVENT_MISSED'] == 2
    assert len(ledger) == 5
    shifted = ranking.copy(); shifted.loc[0, 't_hat'] += 1
    with pytest.raises(ValueError, match='anchors differ'):
        evaluate_seed(episodes, scores, shifted, gt, gt, blocks, .5)


def test_zero_predictions_is_a_complete_failed_result_not_a_dropped_seed():
    episodes, scores, ranking, gt, blocks = synthetic_predictions(empty=True)
    result, _, _, _ = evaluate_seed(episodes, scores, ranking, gt, gt, blocks, .5)
    diagnosis = result['diagnosis']
    assert diagnosis['complete_gt'] == 4
    assert diagnosis['matched_rca']['n'] == 0
    assert diagnosis['matched_rca']['MRR'] is None
    assert diagnosis['e2e']['@5'] == dict(tp=0, fp=0, fn=4, precision=0., recall=0., f1=0.)


def test_summary_contains_all_seeds_and_sample_std(tmp_path):
    evaluation = tmp_path / 'evaluation'; evaluation.mkdir()
    episodes, scores, ranking, gt, blocks = synthetic_predictions()
    result, _, _, _ = evaluate_seed(episodes, scores, ranking, gt, gt, blocks, .5)
    for seed in protocol.SEEDS:
        folder = evaluation / ('seed-%02d' % seed); folder.mkdir()
        value = dict(result, seed=seed,
            selection=dict(selected_epoch=seed, epochs_completed=seed + 8, selected_validation_threshold=.5),
            validation_bin=result['detector'])
        protocol.write(folder / 'result.json', to_builtin(value))
    protocol.seal(evaluation, status='COMPLETE')
    summarize(tmp_path)
    table = pd.read_csv(tmp_path / 'summary/per_seed.csv')
    assert table.seed.tolist() == protocol.SEEDS
    stats = protocol.read(tmp_path / 'summary/statistics.json')
    assert stats['selected_best_seed'] is None
    assert stats['statistics']['selected_epoch']['sample_std'] == pytest.approx(np.std(protocol.SEEDS, ddof=1))
    assert stats['statistics']['e2e_fn@1']['defined_seeds'] == 10
