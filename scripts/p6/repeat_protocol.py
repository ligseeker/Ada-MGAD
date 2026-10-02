"""Standard-library provenance and stage seals for the ten-seed transfer run."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SEEDS = list(range(1, 11))
PROTOCOL = 'P6-TWO-STAGE-FROZEN-XGB-SEEDS1-10-V1'
REQUIRED = {
    'detector': {'checkpoint/best_validation_event_f1.pt', 'validation_selection.json',
        'training_log.json', 'validation_predictions.csv', 'validation_episodes.csv',
        'validation_matching.csv', 'validation_bin_metrics.json', 'validation_replay.json',
        'scores.csv', 'episodes.csv', 'method_record.json'},
    'rca': {'rankings.csv', 'features.npy', 'valid.npy', 'method_record.json'},
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def git(*args):
    return subprocess.check_output(['git'] + list(args), cwd=str(ROOT), text=True).strip()


def path(value):
    return (ROOT / value).resolve()


def validate(config):
    if config['protocol_id'] != PROTOCOL or config['seeds'] != SEEDS:
        raise ValueError('the complete fixed seed family must be 1..10')
    detector = read(path(config['detector_config']))
    if sha(path(config['detector_config'])) != config['detector_config_sha256']:
        raise ValueError('frozen detector config changed')
    if sha(path(detector['source_config']['path'])) != detector['source_config']['sha256']:
        raise ValueError('original detector hyperparameter source changed')
    for name, expected in config['frozen_input_manifests_sha256'].items():
        if sha(path(name)) != expected:
            raise ValueError('frozen preprocessing/raw-index manifest changed: ' + name)
    if detector['training']['max_epochs'] != 30 or detector['training']['patience'] != 8:
        raise ValueError('normal training budget is max_epochs=30, patience=8')
    if (detector['model']['graph_batch_scope'] != 'window'
            or detector['model'].get('metric_drift_residual', False)
            or detector['training'].get('long_onset_weighting', False)
            or detector['trigger_label']['target'] != 'one_bin_onset_frozen_ignore'
            or config['decoder'] != 'independent_bins_at_merged_validation_selected_tau'):
        raise ValueError('fixed TCN/onset/decoder method changed')
    if config['frozen_rca']['anchor_backdate_ms'] != 25621 or config['expected_test_gt'] != 5787:
        raise ValueError('anchor or complete evaluation population changed')
    if sha(path(config['frozen_rca']['model'])) != config['frozen_rca']['model_sha256']:
        raise ValueError('frozen RCA scorer changed')
    trained = read(path(config['frozen_rca']['train_manifest']))
    for name in ('train_manifest', 'run_lock'):
        if sha(path(config['frozen_rca'][name])) != config['frozen_rca'][name + '_sha256']:
            raise ValueError('frozen RCA metadata bytes changed: ' + name)
    if (trained['status'] != 'COMPLETE' or trained['cases'] != 3225 or trained['test_read']
            or trained['model_sha256'] != config['frozen_rca']['model_sha256']
            or trained['model_parameters'] != config['frozen_rca']['model_parameters']
            or trained['run_lock_sha256'] != sha(path(config['frozen_rca']['run_lock']))):
        raise ValueError('frozen scorer training/provenance differs')
    for name, expected in config['feature_source_sha256'].items():
        if sha(path(name)) != expected:
            raise ValueError('frozen 68D feature source changed: ' + name)
    return detector


def seal(directory, name='completion_manifest.json', **extra):
    directory = Path(directory)
    files = {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name != name and p.suffix != '.pyc'}
    write(directory / name, dict(extra, files=files))


def verify_seal(directory, name='completion_manifest.json'):
    directory = Path(directory)
    manifest = read(directory / name)
    for relative, expected in manifest['files'].items():
        local = (directory / relative).resolve()
        local.relative_to(directory.resolve())
        if sha(local) != expected:
            raise ValueError('sealed stage changed: ' + str(local))
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*')
              if p.is_file() and p.name != name and p.suffix != '.pyc'}
    if actual != set(manifest['files']):
        raise ValueError('sealed stage file population changed')
    if manifest['status'] != 'COMPLETE':
        raise ValueError('stage is incomplete')
    return manifest


def verify_scope(output, deep=False):
    scope = read(Path(output) / 'scope_lock.json')
    if subprocess.run(['git', 'merge-base', '--is-ancestor', scope['execution_commit'], 'HEAD'], cwd=str(ROOT)).returncode:
        raise ValueError('execution commit is no longer an ancestor')
    for name, expected in scope['source_sha256'].items():
        if sha(Path(name)) != expected:
            raise ValueError('execution source/config drift: ' + name)
    for name, record in scope['inputs'].items():
        stat = Path(name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (record['bytes'], record['mtime_ns']):
            raise ValueError('input changed during experiment: ' + name)
        if deep and sha(name) != record['sha256']:
            raise ValueError('frozen input bytes changed: ' + name)
    validate(scope['config'])
    return scope


def family_bindings(output):
    """Require all ten complete label-free detector/RCA stages before GT joins."""
    output = Path(output)
    config = read(output / 'scope_lock.json')['config']
    expected_training = read(path(config['detector_config']))['training']
    bindings = {}
    for seed in SEEDS:
        for stage in ('detector', 'rca'):
            folder = output / ('seed-%02d' % seed) / stage
            manifest = verify_seal(folder)
            if not REQUIRED[stage].issubset(manifest['files']):
                raise ValueError('required prediction stage files are missing: ' + str(folder))
            if manifest['seed'] != seed or manifest['scope_sha256'] != sha(output / 'scope_lock.json'):
                raise ValueError('seed/stage belongs to another scope')
            if manifest.get('test_gt_or_matching_read', True):
                raise ValueError('Test label firewall violated')
            record = read(folder / 'method_record.json')
            if record['seed'] != seed:
                raise ValueError('inner method record seed mismatch')
            if stage == 'detector':
                selected = read(folder / 'validation_selection.json')
                checkpoint = folder / 'checkpoint/best_validation_event_f1.pt'
                if (selected['effective_seed'] != seed or selected['model_args']['random_seed'] != seed
                        or selected['training'] != expected_training
                        or selected['model_class'] != 'WindowCausalTCNTrigger'
                        or Path(selected['checkpoint']['path']).resolve() != checkpoint.resolve()
                        or selected['checkpoint']['sha256'] != manifest['files']['checkpoint/best_validation_event_f1.pt']
                        or record['threshold'] != selected['selected_validation_threshold']
                        or record['hash_seed'] != str(seed)):
                    raise ValueError('detector seed/checkpoint/recipe differs from scope')
            elif (record['scorer_sha256'] != config['frozen_rca']['model_sha256']
                    or record['scorer_retrained'] or record['scaler'] is not None
                    or record['anchor_backdate_ms'] != 25621 or record['hash_seed'] != '20260826'):
                raise ValueError('RCA scorer/anchor differs from scope')
            for relative, digest in manifest['files'].items():
                bindings[str((folder / relative).relative_to(output))] = digest
            bindings[str((folder / 'completion_manifest.json').relative_to(output))] = sha(folder / 'completion_manifest.json')
    return bindings


def require_committed(file):
    relative = str(Path(file).resolve().relative_to(ROOT))
    committed = subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=str(ROOT))
    if hashlib.sha256(committed).hexdigest() != sha(file):
        raise ValueError('commit the exact scope/prediction lock before GT evaluation')
