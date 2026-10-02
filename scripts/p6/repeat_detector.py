#!/usr/bin/env python3
"""One registered seed: Fit/Validation training -> label-free Test predictions."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch

from scripts.p6.repeat_protocol import SEEDS, read, sha, write, path, validate, seal, verify_scope
from scripts.p6.run_c0_trigger import build_model_args, run_training, setup_logging, infer_trigger_logits
from scripts.p6.run_c0_window_causal import real_window_causality_gate
from scripts.p6.run_frozen_test_review import InputOnlyWindows, probability
from src.e2e.onset_trigger import OnsetDevelopmentState
from src.e2e.system_trigger_tcn import WindowCausalTCNTrigger, TCN_SPEC
from src.e2e.system_trigger_data import build_trigger_loader
from src.e2e.system_trigger import system_score_frame, trigger_temporal_blocks, to_builtin
from src.e2e.bin_trigger_decoder import construct_bin_candidates, evaluate_bin_threshold


def replay(model, data_root, directory, device):
    reference = pd.read_csv(directory / 'validation_predictions.csv', float_precision='round_trip')
    reports = []
    for begin in sorted({0, (len(reference) // 64) * 32, ((len(reference) - 1) // 32) * 32}):
        block = reference.iloc[begin:begin + 32]
        dataset = InputOnlyWindows(data_root / 'train', block.sample_index.to_numpy())
        indices, logits = infer_trigger_logits(model, build_trigger_loader(dataset, batch_size=32, num_workers=0), device)
        if not np.array_equal(indices, block.sample_index.to_numpy()):
            raise ValueError('selected Validation checkpoint coverage mismatch')
        prob_error = float(np.max(np.abs(probability(logits).astype(float) - block.system_trigger_score.to_numpy())))
        logit_error = float(np.max(np.abs(logits.astype(float) - block.system_trigger_logit.to_numpy())))
        if prob_error > 1e-6 or logit_error > 1e-4:
            raise ValueError('selected checkpoint Validation replay failed')
        reports.append(dict(first_row=begin, n=len(block), probability_error=prob_error, logit_error=logit_error))
    return dict(passed=True, batches=reports, replay_labels_consumed=False)


def predict(output, seed):
    scope = verify_scope(output); config = scope['config']; runtime = scope['runtime']
    detector = validate(config)
    if detector['tcn_spec'] != TCN_SPEC:
        raise ValueError('TCN architecture changed')
    if seed not in SEEDS:
        raise ValueError('unregistered seed')
    if os.environ.get('PYTHONHASHSEED') != str(seed):
        raise ValueError('launch through the orchestrator with the registered hash seed')
    if runtime['gpu'] and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(runtime['threads'])
    directory = output / ('seed-%02d' % seed) / 'detector'
    directory.mkdir(parents=True, exist_ok=False)
    setup_logging(directory)
    data_root = path(config['data_root']); base = read(path(config['base_config']))
    state = OnsetDevelopmentState(base, data_root, path(config['registry']))
    cohort = {name: dict(events=len(state.gt_events(name)), windows=len(state.build_dataset(name)))
              for name in ('fit', 'validation')}
    if {name + '_events': cohort[name]['events'] for name in cohort} != detector['expected_cohort']:
        raise ValueError('Fit/Validation event population drift')
    manifest_path = path(config['artifact_root']) / 'ad_data_manifest.json'
    model_args = build_model_args(detector, base, read(manifest_path), seed, runtime['gpu'])
    selection = run_training(state, directory, model_args, detector['training'], manifest_path,
        sha(manifest_path), runtime['threshold_workers'], 'spawn', model_class=WindowCausalTCNTrigger)
    # The historical training helper marks its record formal. This new scope
    # explicitly grades the reused-Test frozen-scorer transfer before sealing.
    selection.update(formal_result=False, protocol_id=config['protocol_id'], effective_seed=seed,
                     evidence_grade=scope['evidence_grade'], model_class='WindowCausalTCNTrigger')
    from src.e2e.protocol import write_json
    write_json(directory / 'validation_selection.json', to_builtin(selection))
    device = torch.device('cuda' if runtime['gpu'] else 'cpu')
    weights = torch.load(selection['checkpoint']['path'], map_location=device)
    gate = real_window_causality_gate(state, model_args, weights, device, model_class=WindowCausalTCNTrigger)
    model = WindowCausalTCNTrigger(np.load(data_root / 'graph.npy'), **model_args).to(device).eval()
    model.load_state_dict(weights)
    write(directory / 'validation_replay.json', replay(model, data_root, directory, device))
    validation = pd.read_csv(directory / 'validation_predictions.csv', float_precision='round_trip')
    tau = float(selection['selected_validation_threshold'])
    vf = system_score_frame('validation', validation.prediction_available_time, validation.system_trigger_score)
    _, _, val_metrics = evaluate_bin_threshold(vf, state.gt_events('validation'), tau, tolerance_seconds=60)
    write(directory / 'validation_bin_metrics.json', to_builtin(val_metrics))
    # No ProtocolState('test'), Test labels, matching, or Test metric here.
    dataset = InputOnlyWindows(data_root / 'test')
    indices, logits = infer_trigger_logits(model, build_trigger_loader(dataset, batch_size=32,
        num_workers=detector['training']['num_workers'], pin_memory=runtime['gpu']), device)
    if not np.array_equal(indices, dataset.sample_indices) or not np.isfinite(logits).all():
        raise ValueError('complete Test input coverage or finite logits failed')
    times = np.asarray(dataset.timestamps[indices + 9], dtype=np.int64) + 30000
    blocks = trigger_temporal_blocks(base)
    if times[0] != blocks[2].start_ms + 300000 or times[-1] != blocks[2].end_ms:
        raise ValueError('frozen Test window/time bounds changed')
    scores = pd.DataFrame(dict(sample_index=indices, prediction_available_time=times,
                               system_score=probability(logits), logit=logits))
    scores.to_csv(directory / 'scores.csv', index=False)
    episodes = construct_bin_candidates(system_score_frame('test', times, scores.system_score), tau)
    episodes.to_csv(directory / 'episodes.csv', index=False)
    write(directory / 'method_record.json', dict(seed=seed, hash_seed=os.environ['PYTHONHASHSEED'],
        cohort=cohort, causality_gate=gate,
        cuda_runtime=dict(cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            torch_cuda=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
            gpu_device=torch.cuda.get_device_name(0) if runtime['gpu'] else None),
        threshold=tau, checkpoint_sha256=selection['checkpoint']['sha256'],
        checkpoint_selector='merged Validation event F1 / recall / threshold',
        decoder=config['decoder'], test_label_columns_consumed=[]))
    seal(directory, status='COMPLETE', seed=seed, scope_sha256=sha(output / 'scope_lock.json'),
         test_gt_or_matching_read=False)
    print('DETECTOR_COMPLETE', seed, 'windows', len(scores), 'episodes', len(episodes), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('predict',))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    predict(args.output_dir.resolve(), args.seed)
