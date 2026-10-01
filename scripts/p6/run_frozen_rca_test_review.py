#!/usr/bin/env python3
"""Frozen RCA ablations and scorer transfer; predictions precede GT joins."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import xgboost as xgb
from src.e2e.protocol import sha256_file, write_json
from src.e2e.rca_model import GAIA_SERVICES, rank_candidates
from src.e2e.c1_common_cohort import extract_c1_anchor_features
from src.e2e.gaia_rca_adapter import GaiaRcaRawIndex, validate_raw_index_manifest

PARAMETERS = dict(objective='rank:pairwise', n_estimators=200, max_depth=3,
    learning_rate=.05, subsample=1., colsample_bytree=1., reg_lambda=1.,
    random_state=20260826, n_jobs=1, tree_method='hist')
INDEX = None


def read(path):
    return json.loads(Path(path).read_text())


def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def check(mapping):
    for path, digest in mapping.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('immutable input/source drift: ' + path)


def seal(directory, name, **extra):
    write_json(directory / name, dict(extra, files={str(p.relative_to(directory)): sha256_file(p)
        for p in directory.rglob('*') if p.is_file() and p.name != name}))


def masks():
    return {'no_metric': np.arange(17, 68), 'no_all_magnitude': np.asarray(
        [17 * c + j for c in range(4) for j in range(3, 17)])}


def verify(output):
    scope = read(output / 'scope_lock.json')
    if scope['execution_commit'] != head():
        raise ValueError('execution HEAD drift')
    check(scope['bindings'])
    return scope['config']


def prepare(config, config_path, output):
    if output.exists():
        raise FileExistsError(output)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip():
        raise ValueError('commit RCA source/config/receipts before execution')
    if xgb.__version__ != '2.1.4' or config['model_parameters'] != PARAMETERS:
        raise ValueError('frozen scorer environment/parameters differ')
    detector = Path(config['detector_run'])
    pred = read(detector / 'global_prediction_lock.json')
    if (sha256_file(detector / 'global_prediction_lock.json') != config['detector_prediction_lock_sha256']
            or pred['status'] != 'PREDICTION_LOCKED' or len(pred['arms']) != 19):
        raise ValueError('detector family prediction receipt mismatch')
    check(pred['bindings'])
    paths = [config_path, ROOT / 'docs/P6_FROZEN_RCA_TEST_REVIEW_PROTOCOL.md',
        ROOT / 'scripts/p6/run_frozen_rca_test_review.py',
        ROOT / 'scripts/p6/execute_frozen_rca_test_review.py',
        ROOT / 'scripts/p6/audit_frozen_detector_test_review.py', detector / 'scope_lock.json',
        detector / 'global_prediction_lock.json']
    paths.extend(p for p in (ROOT / 'src').rglob('*.py'))
    paths.extend(Path(p) for p in config['inputs'].values())
    paths.extend(Path(p) for p in config['models'].values())
    paths.extend(detector / 'models' / m['id'] / 'completion_manifest.json'
        for m in read(detector / 'scope_lock.json')['config']['models'])
    bindings = {str(p.resolve()): sha256_file(p) for p in paths}
    bindings.update(pred['bindings'])
    check(config['expected_input_sha256'])
    check(config['expected_feature_source_sha256'])
    for path, digest in config['detector_model_manifest_sha256'].items():
        if bindings.get(path) != digest:
            raise ValueError('committed detector model metadata receipt mismatch')
    ablation = read(config['inputs']['ablation_results'])
    if ablation['model_parameters'] != PARAMETERS:
        raise ValueError('ablation recipe drift')
    for name, keep in masks().items():
        if any(fold['variants'][name]['features'] != keep.tolist() for fold in ablation['folds'].values()):
            raise ValueError('ablation feature mask drift')
    raw = validate_raw_index_manifest(Path(config['inputs']['raw_manifest']))
    # Raw arrays stay read-only; manifest supplies all identities for a final recheck.
    output.mkdir(parents=True)
    write_json(output / 'scope_lock.json', dict(execution_commit=head(), config=config,
        bindings=bindings, raw_validation=raw, evidence_grade='frozen-scorer transfer / reused-Test',
        test_used_for_selection=False, test_gt_read=False))
    print('RCA_SCOPE_LOCKED', raw, flush=True)


def train(config, output):
    directory = output / 'train'; directory.mkdir(exist_ok=False)
    x = np.load(config['inputs']['train_features'], allow_pickle=False)
    roots = np.load(config['inputs']['train_roots'], allow_pickle=False)
    cases = pd.read_csv(config['inputs']['train_cases'], keep_default_na=False)
    if (x.shape != (3225, 10, 68) or roots.shape != (3225,) or len(cases) != 3225
            or not np.isfinite(x).all() or not cases.case_id.is_unique
            or [GAIA_SERVICES[int(r)] for r in roots] != cases.root_service.tolist()):
        raise ValueError('common Train cohort contract mismatch')
    target = np.zeros((3225, 10), dtype=np.float32)
    target[np.arange(3225), roots] = 1.
    for name, keep in masks().items():
        started = time.monotonic()
        model = xgb.XGBRanker(**PARAMETERS)
        values = np.ascontiguousarray(x[:, :, keep].reshape(-1, len(keep)), dtype=np.float32)
        model.fit(values, target.reshape(-1), group=np.full(3225, 10), verbose=False)
        model.save_model(str(directory / (name + '.ubj')))
        write_json(directory / (name + '.json'), dict(train_cases=3225, group_size=10,
            features=keep.tolist(), model_parameters=PARAMETERS, test_read=False,
            elapsed_seconds=time.monotonic() - started))
        print('FINAL_TRAIN_FIT', name, len(keep), flush=True)
    seal(directory, 'completion_manifest.json', status='COMPLETE', execution_commit=head())


def initialize_index(manifest):
    global INDEX
    INDEX = GaiaRcaRawIndex.from_manifest(Path(manifest))


def extract(anchor):
    return int(anchor), extract_c1_anchor_features(INDEX, case_id='anchor-' + str(anchor), anchor_ms=int(anchor))


def features(config, output, workers):
    directory = output / 'features'; directory.mkdir(exist_ok=False)
    detector = Path(config['detector_run']); pred = read(detector / 'global_prediction_lock.json')
    needed = set()
    start, end = config['test_interval_ms']
    for arm in pred['arms']:
        episodes = pd.read_csv(detector / 'arms' / arm['id'] / 'episodes.csv')
        for offset in (0, -25621):
            anchors = episodes.t_hat.to_numpy(dtype=np.int64) + offset
            needed.update(int(a) for a in anchors if a - 300000 >= start and a + 300000 <= end)
    anchors = np.asarray(sorted(needed), dtype=np.int64)
    np.save(directory / 'anchors.npy', anchors)
    positions = {int(a): i for i, a in enumerate(anchors)}
    values = np.lib.format.open_memmap(directory / 'values.npy', mode='w+', dtype=np.float32,
        shape=(len(anchors), 10, 68))
    ready = np.zeros(len(anchors), dtype=bool)
    old_scope = pd.read_csv(config['inputs']['native_scope'], keep_default_na=False)
    old_valid = np.load(config['inputs']['native_valid'], allow_pickle=False)
    cached_rows = 0
    initialize_index(config['inputs']['raw_manifest'])
    for key, offset in [('native_features', 0), ('backdate_features', -25621)]:
        cached = np.load(config['inputs'][key], mmap_mode='r', allow_pickle=False)
        if cached.shape != (len(old_scope), 10, 68) or len(old_valid) != len(old_scope):
            raise ValueError('exact-anchor cache shape mismatch')
        selected = np.flatnonzero(old_valid)
        for p in selected[[0, len(selected)//2, -1]]:
            anchor = int(old_scope.t_hat.iloc[p]) + offset
            _, replay = extract(anchor)
            if not np.array_equal(replay, cached[p]):
                raise ValueError('raw-index / feature transform cache replay mismatch')
        for p in selected:
            anchor = int(old_scope.t_hat.iloc[p]) + offset
            if anchor in positions:
                i = positions[anchor]
                if ready[i] and not np.array_equal(values[i], cached[p]):
                    raise ValueError('same anchor has inconsistent cache values')
                values[i] = cached[p]; ready[i] = True; cached_rows += 1
    pending = anchors[~ready]
    write_json(directory / 'extraction_plan.json', dict(unique_anchors=len(anchors),
        cached_rows=cached_rows, pending=len(pending), workers=workers,
        raw_manifest_sha256=sha256_file(Path(config['inputs']['raw_manifest'])),
        schema='canonical10 x four17D Z2 W300-B15; no scaler; exact anchor only'))
    print('FEATURE_PLAN', len(anchors), 'unique', len(pending), 'new', flush=True)
    started = time.monotonic()
    with mp.get_context('spawn').Pool(workers, initializer=initialize_index,
        initargs=(config['inputs']['raw_manifest'],)) as pool:
        for n, (anchor, row) in enumerate(pool.imap_unordered(extract, pending, chunksize=8), 1):
            i = positions[anchor]; values[i] = row; ready[i] = True
            if n % 256 == 0:
                values.flush(); np.save(directory / 'ready.npy', ready)
                print('FEATURE_PROGRESS', n, '/', len(pending), 'seconds', round(time.monotonic()-started), flush=True)
    if not ready.all() or not np.isfinite(values).all():
        raise ValueError('feature cache coverage failed')
    values.flush(); np.save(directory / 'ready.npy', ready)
    validate_raw_index_manifest(Path(config['inputs']['raw_manifest']))
    seal(directory, 'completion_manifest.json', status='COMPLETE', unique_anchors=len(anchors))
    print('FEATURES_COMPLETE', len(anchors), flush=True)


def score(model_path, values, valid, keep=None):
    model = xgb.XGBRanker(); model.load_model(str(model_path))
    selected = values[valid]
    if keep is not None:
        selected = selected[:, :, keep]
    if not np.isfinite(selected).all():
        raise ValueError('nonfinite features')
    rankings = [''] * len(values)
    if not len(selected):
        return rankings
    scores = model.predict(np.ascontiguousarray(selected.reshape(-1, selected.shape[-1]))).reshape(-1, 10)
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite scorer output')
    for p, row in zip(np.flatnonzero(valid), scores):
        rankings[int(p)] = json.dumps(rank_candidates(GAIA_SERVICES, row), separators=(',', ':'))
    return rankings


def rank(config, output):
    directory = output / 'predictions'; directory.mkdir(exist_ok=False)
    detector = Path(config['detector_run']); pred = read(detector / 'global_prediction_lock.json')
    for sub in ('train', 'features'):
        m = read(output / sub / 'completion_manifest.json')
        check({str(output / sub / p): d for p, d in m['files'].items()})
    anchors = np.load(output / 'features/anchors.npy', allow_pickle=False)
    cache = np.load(output / 'features/values.npy', mmap_mode='r', allow_pickle=False)
    positions = {int(a): p for p, a in enumerate(anchors)}
    start, end = config['test_interval_ms']; arms = []
    for arm in pred['arms']:
        episodes = pd.read_csv(detector / 'arms' / arm['id'] / 'episodes.csv')
        for name, offset in [('original', 0), ('backdate', -25621)]:
            times = episodes.t_hat.to_numpy(dtype=np.int64) + offset
            valid = (times - 300000 >= start) & (times + 300000 <= end)
            values = np.zeros((len(episodes), 10, 68), dtype=np.float32)
            for p in np.flatnonzero(valid):
                values[p] = cache[positions[int(times[p])]]
            frame = episodes[['prediction_id', 't_hat']].copy()
            frame['anchor_ms'] = times; frame['legal'] = valid
            frame['ranking'] = score(config['models'][name], values, valid)
            aid = arm['id'] + '__' + name; frame.to_csv(directory / (aid + '.csv'), index=False)
            arms.append(dict(id=aid, detector_arm=arm['id'], scorer=name, native=False))
    old = pd.read_csv(config['inputs']['native_scope'], keep_default_na=False)
    values = np.load(config['inputs']['native_features'], allow_pickle=False)
    valid = np.load(config['inputs']['native_valid'], allow_pickle=False)
    original = score(config['models']['original'], values, valid)
    if original != old.ranking_xgb.tolist():
        raise ValueError('old XGB complete ranking replay failed')
    for name, keep in masks().items():
        frame = old[['prediction_id', 't_hat']].copy()
        frame['anchor_ms'] = frame.t_hat; frame['legal'] = valid
        frame['ranking'] = score(output / 'train' / (name + '.ubj'), values, valid, keep)
        frame.to_csv(directory / ('native__' + name + '.csv'), index=False)
        arms.append(dict(id='native__' + name, detector_arm='legacy_native', scorer=name, native=True))
    baseline = old[['prediction_id', 't_hat']].copy()
    baseline['anchor_ms'] = baseline.t_hat; baseline['legal'] = valid
    baseline['ranking'] = original
    baseline.to_csv(directory / 'native__original.csv', index=False)
    if len(arms) != 40:
        raise ValueError('incomplete 38 transfer / 2 ablation family')
    seal(directory, 'global_prediction_lock.json', status='PREDICTION_LOCKED', arms=arms,
        scope_lock_sha256=sha256_file(output / 'scope_lock.json'), execution_commit=head(),
        test_gt_or_matching_read=False, test_used_for_selection=False)
    print('RCA_GLOBAL_PREDICTION_LOCKED', len(arms), flush=True)


def metrics_for(frame, matching, gt_n):
    if not frame.prediction_id.is_unique or matching.loc[matching.match_status.ne('miss'), 'prediction_id'].duplicated().any():
        raise ValueError('prediction identities are not unique')
    if set(frame.prediction_id) != set(matching.loc[matching.match_status.ne('miss'), 'prediction_id']):
        raise ValueError('prediction universe differs from matching')
    joined = matching.merge(frame, on='prediction_id', how='left', validate='many_to_one', suffixes=('', '_rank'))
    matched = joined.match_status.eq('matched')
    if joined.loc[joined.match_status.ne('miss'), 'legal'].isna().any():
        raise ValueError('prediction identity coverage mismatch')
    if 't_hat_rank' in joined and not np.array_equal(
            joined.loc[joined.match_status.ne('miss'), 't_hat'].to_numpy(dtype=np.int64),
            joined.loc[joined.match_status.ne('miss'), 't_hat_rank'].to_numpy(dtype=np.int64)):
        raise ValueError('ranking anchors differ from matched prediction anchors')
    ranks = []
    for row in joined.itertuples():
        if row.match_status == 'matched' and row.legal and isinstance(row.ranking, str) and row.ranking:
            ranking = json.loads(row.ranking)
            if len(ranking) != 10 or set(ranking) != set(GAIA_SERVICES):
                raise ValueError('incomplete candidate ranking')
            ranks.append(ranking.index(row.gt_service) + 1)
        else:
            ranks.append(11)
    joined['root_rank'] = ranks
    sufficient = matched & joined.legal.fillna(False).astype(bool)
    n = int(sufficient.sum()); predictions = len(frame)
    r = joined.loc[sufficient, 'root_rank'].to_numpy()
    result = dict(predictions=predictions, complete_gt=gt_n, matched=int(matched.sum()),
        legal_matched=n, illegal_matched=int((matched & ~sufficient).sum()),
        matched_rca={'n': n, 'MRR': float(np.mean(np.where(r <= 10, 1./r, 0))) if n else None}, e2e={})
    for k in (1, 3, 5):
        tp = int((matched & joined.root_rank.le(k)).sum())
        result['matched_rca']['AC@' + str(k)] = float(np.mean(r <= k)) if n else None
        result['e2e']['@' + str(k)] = dict(tp=tp, fp=predictions-tp, fn=gt_n-tp,
            precision=tp/predictions if predictions else 0., recall=tp/gt_n,
            f1=2.*tp/(predictions+gt_n))
    ledger = np.where(joined.match_status.eq('miss'), 'EVENT_MISSED',
        np.where(joined.match_status.eq('false_alarm'), 'EVENT_FALSE_ALARM',
        np.where(~joined.legal.fillna(False).astype(bool), 'RCA_CONTEXT_INVALID',
        np.where(joined.root_rank.eq(11), 'RCA_RANKING_MISSING',
        np.where(joined.root_rank.eq(1), 'SUCCESS_TOP1',
        np.where(joined.root_rank.le(3), 'ROOT_OUTSIDE_TOP1',
        np.where(joined.root_rank.le(5), 'ROOT_OUTSIDE_TOP3', 'ROOT_OUTSIDE_TOP5')))))))
    joined['failure_category'] = ledger
    result['failure'] = {str(k): int(v) for k, v in joined.failure_category.value_counts().items()}
    if len(joined) != gt_n + int(matching.match_status.eq('false_alarm').sum()):
        raise ValueError('full failure population does not close')
    result['strata'] = []
    for dimension in ('gt_service', 'fault_type'):
        for label, group in joined.loc[joined.match_status.ne('false_alarm')].groupby(dimension):
            result['strata'].append(dict(dimension=dimension, group=label, n=len(group),
                recall_at_1=float(group.root_rank.le(1).sum()/len(group))))
    return result, joined


def evaluate(config, output, expected_lock, expected_audit):
    path = output / 'predictions/global_prediction_lock.json'
    if not expected_lock or sha256_file(path) != expected_lock:
        raise ValueError('supply independently captured RCA prediction-lock receipt')
    lock = read(path)
    if lock['status'] != 'PREDICTION_LOCKED' or len(lock['arms']) != 40:
        raise ValueError('RCA prediction family incomplete')
    check({str(path.parent / p): d for p, d in lock['files'].items()})
    directory = output / 'evaluation'; directory.mkdir(exist_ok=False)
    detector = Path(config['detector_run']); results = {}; table = []; ledgers = {}
    audit_path = Path(config['detector_audit_dir']) / 'audit.json'
    if not expected_audit or sha256_file(audit_path) != expected_audit:
        raise ValueError('independent detector audit receipt mismatch')
    audit = read(audit_path)
    if (audit['status'] != 'PASS' or audit['prediction_lock_sha256'] != config['detector_prediction_lock_sha256']
            or audit['evaluation_manifest_sha256'] != sha256_file(detector / 'evaluation/completion_manifest.json')):
        raise ValueError('detector matching population is not bound to independent audit')
    detector_evaluation = read(detector / 'evaluation/completion_manifest.json')
    check({str(detector / 'evaluation' / p): d for p, d in detector_evaluation['files'].items()})
    gt = pd.read_csv(detector / 'evaluation/test_gt.csv')
    expected_cases = set(gt.case_id)
    if len(gt) != 5787 or len(expected_cases) != 5787:
        raise ValueError('Test GT identity population drift')
    native_matching = pd.read_csv(config['inputs']['native_matching'], float_precision='round_trip')
    native_frame = pd.read_csv(path.parent / 'native__original.csv', keep_default_na=False)
    native_result, native_ledger = metrics_for(native_frame, native_matching, 5787)
    if (native_result['legal_matched'] != 4197 or native_result['e2e']['@1']['tp'] != 3093
            or native_result['e2e']['@3']['tp'] != 4189 or native_result['e2e']['@5']['tp'] != 4194):
        raise ValueError('native original XGB count replay mismatch')
    for arm in lock['arms']:
        source = Path(config['inputs']['native_matching']) if arm['native'] else (
            detector / 'evaluation' / arm['detector_arm'] / 'matching.csv')
        matching = pd.read_csv(source, float_precision='round_trip')
        frame = pd.read_csv(path.parent / (arm['id'] + '.csv'), keep_default_na=False)
        if matching.match_status.isin(['matched', 'miss']).sum() != 5787:
            raise ValueError('complete Test GT population mismatch')
        if set(matching.loc[matching.match_status.isin(['matched','miss']), 'case_id']) != expected_cases:
            raise ValueError('GT case identities differ across detector/native cohorts')
        observed = matching.loc[matching.match_status.isin(['matched','miss'])].set_index('case_id')
        canonical = gt.set_index('case_id').loc[observed.index]
        for left, right in [('gt_service','service'),('fault_type','fault_type'),
                ('gt_start_ms','start_ms'),('gt_end_ms','end_ms')]:
            if not np.array_equal(observed[left].to_numpy(), canonical[right].to_numpy()):
                raise ValueError('GT label/timing metadata differ from registry: ' + left)
        result, ledger = metrics_for(frame, matching, 5787)
        results[arm['id']] = dict(arm=arm, **result); ledgers[arm['id']] = ledger
        if arm['native']:
            a = native_ledger.loc[native_ledger.match_status.eq('matched') & native_ledger.legal.eq(True),
                ['case_id', 'root_rank']]
            b = ledger.loc[ledger.match_status.eq('matched') & ledger.legal.eq(True), ['case_id', 'root_rank']]
            paired = a.merge(b, on='case_id', suffixes=('_original', '_candidate'), validate='one_to_one')
            correct_a=paired.root_rank_original.eq(1);correct_b=paired.root_rank_candidate.eq(1)
            results[arm['id']]['paired_vs_original'] = dict(n=len(paired),
                both=int((correct_a&correct_b).sum()), original_only=int((correct_a&~correct_b).sum()),
                candidate_only=int((~correct_a&correct_b).sum()), neither=int((~correct_a&~correct_b).sum()),
                delta_ac1=float((correct_b.sum()-correct_a.sum())/len(paired)))
        ledger.to_csv(directory / (arm['id'] + '_failure.csv'), index=False)
        table.append(dict(arm_id=arm['id'], detector_arm=arm['detector_arm'], scorer=arm['scorer'],
            matched=result['matched'], legal_matched=result['legal_matched'], predictions=result['predictions'],
            **result['matched_rca'], **{'E2E_'+key+'_'+m: v[m] for key,v in result['e2e'].items()
                for m in ('precision','recall','f1','tp','fp','fn')}))
    for arm in lock['arms']:
        if arm['scorer'] != 'backdate': continue
        left = ledgers[arm['detector_arm']+'__original']; right = ledgers[arm['id']]
        paired = left.loc[left.match_status.eq('matched') & left.legal.eq(True), ['case_id','root_rank']].merge(
            right.loc[right.match_status.eq('matched') & right.legal.eq(True), ['case_id','root_rank']],
            on='case_id', suffixes=('_original','_backdate'), validate='one_to_one')
        a=paired.root_rank_original.eq(1);b=paired.root_rank_backdate.eq(1)
        results[arm['id']]['paired_common_legal'] = dict(n=len(paired), both=int((a&b).sum()),
            original_only=int((a&~b).sum()), backdate_only=int((~a&b).sum()), neither=int((~a&~b).sum()),
            delta_ac1=float((b.sum()-a.sum())/len(paired)) if len(paired) else None)
    write_json(directory/'results.json', dict(status='COMPLETE', results=results,
        native_baseline_replay=native_result, reused_test=True,
        test_used_for_selection=False, prediction_lock_sha256=expected_lock))
    pd.DataFrame(table).to_csv(directory/'rca_e2e_results.csv',index=False)
    seal(directory,'completion_manifest.json',status='COMPLETE',execution_commit=head())
    print('RCA_E2E_EVALUATION_COMPLETE', len(results), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train','features','rank','evaluate'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--prediction-lock-sha256')
    parser.add_argument('--detector-audit-sha256')
    args=parser.parse_args(); config=read(args.config)
    if args.action=='prepare':prepare(config,args.config,args.output_dir);return
    if verify(args.output_dir)!=config:raise ValueError('configuration drift')
    if args.action=='train':train(config,args.output_dir)
    elif args.action=='features':features(config,args.output_dir,args.workers)
    elif args.action=='rank':rank(config,args.output_dir)
    elif args.action=='evaluate':evaluate(config,args.output_dir,args.prediction_lock_sha256,args.detector_audit_sha256)
    verify(args.output_dir)


if __name__=='__main__':main()
