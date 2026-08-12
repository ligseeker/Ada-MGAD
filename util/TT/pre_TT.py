import argparse
import logging
import os
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util.TT.constant import (
    TT_SERVICES, TT_SERVICE2NID, TT_NUM_NODES, TT_METRICS, TT_EDGES,
    STATS_TOP_K, DEFAULT_SRC, DEFAULT_RAW_DIR, DEFAULT_SAVE_PATH,
)

OFFSET_MIN_OVERLAP = 30
OFFSET_MIN_KEPT = 200
OFFSET_TAIL_MARGIN = 3600
CONSENSUS_DELTAS = (-120, -60, -30, -10, 0, 10, 30, 60, 120)


def robust_zscore(frame):
    medians = frame.median()
    mads = frame.sub(medians).abs().median()
    flat = mads < 1e-9
    normalized = frame.sub(medians).div(1.4826 * mads.where(~flat, 1.0))
    normalized.loc[:, flat] = 0.0
    return normalized.fillna(0.0)


def find_raw_exp_dir(raw_dir, exp_name):
    for sub in ('data', 'no fault'):
        candidate = os.path.join(raw_dir, sub, exp_name)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"experiment {exp_name} not found under {raw_dir}/data or {raw_dir}/no fault")


def load_metric_frame(raw_dir, exp_name, t0, T):
    exp_dir = find_raw_exp_dir(raw_dir, exp_name)
    frames = []
    for service in TT_SERVICES:
        csv_path = os.path.join(exp_dir, 'metrics', f'{service}.csv')
        df = pd.read_csv(csv_path)
        rename_map = {kpi: f'{service}_{kpi}' for kpi in TT_METRICS if kpi in df.columns}
        assert len(rename_map) == len(TT_METRICS), f"missing KPI columns in {csv_path}"
        frames.append(df.rename(columns=rename_map)[['timestamp'] + list(rename_map.values())].set_index('timestamp'))
        del df
    metric = pd.concat(frames, axis=1).sort_index()
    del frames
    assert int(metric.index.min()) == t0, \
        f"{exp_name}: raw metric start {int(metric.index.min())} != TT-pre axis start {t0}"
    metric = metric.reindex(range(t0, t0 + T)).ffill().fillna(0.0)
    return robust_zscore(metric)


def spearman_corr(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _score_offset_candidates(span_df, A, a_origin, B, t0, T, extra_offsets):
    """Score candidate trace-clock offsets for one experiment.

    Candidates: the strongest np.correlate lags, the dump-tail lag (Jaeger
    cumulative dumps end shortly after the experiment), and offsets around a
    cross-experiment consensus skew. A candidate is accepted only when the
    aligned segments correlate (Spearman, heavy-tailed durations crush
    Pearson), the experiment axis retains enough spans, and the dump does not
    end far past the experiment (which indicates another experiment's span
    mass locked the correlation). Returns (offset, spear) or (None, best).
    """
    ends = span_df['end_sec'].values
    max_end_floor = int(np.floor(ends.max()))
    corr = np.correlate(A, B, 'full')
    candidates = set()
    for peak in np.argsort(corr)[::-1][:300]:
        candidates.add(a_origin + int(peak) - (len(B) - 1) - t0)
    candidates.add(max_end_floor - t0 - (T - 1))
    for base in extra_offsets:
        for delta in CONSENSUS_DELTAS:
            candidates.add(int(base) + delta)

    best = None
    for offset in candidates:
        k = offset - a_origin + t0
        n_lo, n_hi = max(0, -k), min(len(B), len(A) - k)
        if n_hi - n_lo < OFFSET_MIN_OVERLAP:
            continue
        a_seg, b_seg = A[k + n_lo:k + n_hi], B[n_lo:n_hi]
        if a_seg.std() < 1e-12 or b_seg.std() < 1e-12:
            continue
        spear = spearman_corr(a_seg, b_seg)
        if spear <= 0.1:
            continue
        end_rel = np.floor(ends - offset - t0).astype(np.int64)
        kept_mask = (end_rel >= 0) & (end_rel < T)
        kept = int(kept_mask.sum())
        if kept < OFFSET_MIN_KEPT:
            continue
        tail_gap = max_end_floor - offset - t0 - (T - 1)
        if tail_gap > OFFSET_TAIL_MARGIN:
            continue
        end_kept = end_rel[kept_mask]
        coverage = min(1.0, float(end_kept.max() - end_kept.min() + 1) / T)
        score = spear + 0.2 * coverage
        if best is None or score > best[0]:
            best = (score, offset, spear)
    if best is None:
        return None, 0.0
    return best[1], best[2]


def estimate_trace_offset(span_df, label, log_secs, metric_values, t0, T, consensus):
    """Estimate offset = trace_clock - metric_clock for one experiment.

    Stage order: fault-label x duration mass (fault experiments), then span
    count x metric activity, then span count x log rate. All experiments share
    one Jaeger and one metric clock, so a consensus skew from the other
    experiments is searched as well.
    """
    starts = span_df['start_sec'].values
    a_origin = int(np.floor(starts.min()))
    bins = np.floor(starts).astype(np.int64) - a_origin
    span_len = int(np.floor(starts.max())) - a_origin + 1

    A_dur = np.zeros(span_len)
    np.add.at(A_dur, bins, span_df['duration'].values.astype(np.float64))
    A_cnt = np.zeros(span_len)
    np.add.at(A_cnt, bins, 1)

    stages = []
    if label.sum() > 0:
        stages.append(('corr(label)', A_dur, (label.sum(axis=1) > 0).astype(np.float64)))
        stages.append(('corr(metric)', A_cnt, (metric_values.astype(np.float64) ** 2).sum(axis=(1, 2))))
        if log_secs is not None and len(log_secs):
            L = np.zeros(T)
            np.add.at(L, log_secs, 1)
            stages.append(('corr(log)', A_cnt, L))
    else:
        if log_secs is not None and len(log_secs):
            L = np.zeros(T)
            np.add.at(L, log_secs, 1)
            stages.append(('corr(log)', A_cnt, L))
        stages.append(('corr(metric)', A_cnt, (metric_values.astype(np.float64) ** 2).sum(axis=(1, 2))))

    extra = [consensus] if consensus is not None else []
    for method, A, B in stages:
        offset, spear = _score_offset_candidates(span_df, A, a_origin, B, t0, T, extra)
        if offset is not None:
            return offset, spear, method
    return None, 0.0, 'failed'


def fault_windows(label):
    active = label.sum(axis=1) > 0
    windows, start = [], None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            windows.append((start, i))
            start = None
    if start is not None:
        windows.append((start, len(active)))
    return windows


def read_label(exp_dir, T):
    label = pickle.load(open(os.path.join(exp_dir, 'label.pkl'), 'rb'))
    if label.shape[0] == T + 1:
        label = label[:T]
    assert label.shape[0] == T and label.shape[1] == TT_NUM_NODES, \
        f"label shape {label.shape} inconsistent with metric axis ({T}, {TT_NUM_NODES})"
    return label


def quick_offset_scan(src, experiments):
    """Consensus skew: coarse per-fault-experiment estimate, then the median."""
    estimates = []
    for exp_name in experiments:
        exp_dir = os.path.join(src, exp_name)
        now = pd.read_csv(os.path.join(exp_dir, 'metric.csv'), usecols=['now'])['now']
        t0, T = int(now.min()), len(now)
        del now
        label = read_label(exp_dir, T)
        if label.sum() == 0:
            continue
        span_df = pd.read_csv(os.path.join(exp_dir, 'trace.csv'),
                              usecols=['start_sec', 'end_sec', 'duration']).drop_duplicates()
        starts = span_df['start_sec'].values
        a_origin = int(np.floor(starts.min()))
        bins = np.floor(starts).astype(np.int64) - a_origin
        A = np.zeros(int(np.floor(starts.max())) - a_origin + 1)
        np.add.at(A, bins, span_df['duration'].values.astype(np.float64))
        B = (label.sum(axis=1) > 0).astype(np.float64)
        offset, _ = _score_offset_candidates(span_df, A, a_origin, B, t0, T, [])
        if offset is not None:
            estimates.append(int(offset))
            logging.info("consensus scan %s: offset=%d", exp_name, offset)
        del span_df, label
    if not estimates:
        return None
    consensus = int(np.median(estimates))
    logging.info("consensus trace-metric skew from %d fault experiments: %d", len(estimates), consensus)
    return consensus


def process_experiment(exp_name, src, raw_dir, save_path, miner, vocab2idx, consensus):
    logging.info("processing experiment %s", exp_name)
    exp_dir = os.path.join(src, exp_name)

    now = pd.read_csv(os.path.join(exp_dir, 'metric.csv'), usecols=['now'])['now']
    t0, T = int(now.min()), len(now)
    del now

    metric = load_metric_frame(raw_dir, exp_name, t0, T)
    metric_values = metric.values.astype(np.float32).reshape(T, TT_NUM_NODES, len(TT_METRICS))
    del metric

    label = read_label(exp_dir, T).astype(np.int8)

    log_src = pd.read_csv(os.path.join(exp_dir, 'log.csv'))
    log_src = log_src.sort_values(by='@timestamp')
    log_events, log_secs, log_dropped = [], [], 0
    for ts, host, payload in zip(log_src['@timestamp'].values, log_src['Hostname'].values,
                                 log_src['Payload'].astype(str).values):
        rel = int(ts) - t0
        if rel < 0 or rel >= T:
            log_dropped += 1
            continue
        service_id = TT_SERVICE2NID.get(host)
        if service_id is None:
            log_dropped += 1
            continue
        cluster_id = miner.add_log_message(payload)['cluster_id']
        log_events.append((int(cluster_id), service_id, rel))
        log_secs.append(rel)
    del log_src

    span_df = pd.read_csv(os.path.join(exp_dir, 'trace.csv'))
    raw_spans = len(span_df)
    span_df = span_df.drop_duplicates()
    diag = {'name': exp_name, 'T': T, 't0': t0, 'offset': None, 'peak_corr': 0.0, 'method': 'none',
            'raw_spans': raw_spans, 'num_spans': 0, 'window_spans': []}
    trace_events = []
    if len(span_df):
        offset, peak_corr, method = estimate_trace_offset(
            span_df, label, np.asarray(log_secs, dtype=np.int64), metric_values, t0, T, consensus)
        diag['peak_corr'], diag['method'] = peak_corr, method
        if offset is None:
            logging.warning("!" * 100)
            logging.warning(
                "EXPERIMENT %s: TRACE OFFSET UNAVAILABLE (all correlation stages failed). "
                "TRACE FEATURES ZEROED FOR THIS EXPERIMENT.", exp_name)
            logging.warning("!" * 100)
        else:
            diag['offset'] = float(offset)
            end_rel = np.floor(span_df['end_sec'].values - offset - t0).astype(np.int64)
            keep = (end_rel >= 0) & (end_rel < T)
            kept = span_df[keep]
            end_kept = end_rel[keep]
            src_id = kept['cmbd_id'].map(TT_SERVICE2NID)
            dst_id = kept['fatherpod'].map(TT_SERVICE2NID)
            stats_idx = kept['stats'].map(vocab2idx)
            valid = src_id.notna() & dst_id.notna() & stats_idx.notna()
            valid &= src_id.ne(dst_id)
            valid = valid.values
            src_v = src_id.values[valid].astype(np.int64)
            dst_v = dst_id.values[valid].astype(np.int64)
            stats_v = stats_idx.values[valid].astype(np.int64)
            end_v = end_kept[valid]
            dur_v = kept['duration'].values[valid].astype(np.float64)
            trace_events = list(zip(src_v.tolist(), dst_v.tolist(), stats_v.tolist(),
                                    end_v.tolist(), dur_v.tolist()))
            diag['num_spans'] = len(trace_events)
            for s, e in fault_windows(label)[:3]:
                count = int(((end_v >= s) & (end_v < e)).sum())
                diag['window_spans'].append((s, e, count))
    del span_df

    exp_save = os.path.join(save_path, exp_name)
    os.makedirs(exp_save, exist_ok=True)
    np.save(os.path.join(exp_save, 'metric.npy'), metric_values)
    np.save(os.path.join(exp_save, 'label.npy'), label)
    pickle.dump(log_events, open(os.path.join(exp_save, 'log_events.pkl'), 'wb'))
    pickle.dump(trace_events, open(os.path.join(exp_save, 'trace_events.pkl'), 'wb'))
    diag['log_kept'] = len(log_events)
    diag['log_dropped'] = log_dropped
    diag['templates'] = len(miner.drain.clusters)
    diag['label_nonzero'] = int(label.sum())
    del metric_values, label, log_events, trace_events
    return diag


def scan_stats_vocab(src, experiments):
    counter = Counter()
    for exp_name in experiments:
        stats = pd.read_csv(os.path.join(src, exp_name, 'trace.csv'), usecols=['stats'])['stats']
        counter.update(stats.tolist())
        del stats
    vocab = [name for name, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:STATS_TOP_K]]
    logging.info("stats vocab: %d unique stats seen, keeping top-%d", len(counter), len(vocab))
    return vocab


def preprocess(src, raw_dir, save_path):
    os.makedirs(save_path, exist_ok=True)

    experiments = sorted(d for d in os.listdir(src)
                         if d.startswith('TT.') and os.path.isdir(os.path.join(src, d)))
    assert len(experiments) == 11, f"expected 11 TT experiments under {src}, found {len(experiments)}"
    logging.info("found %d experiments", len(experiments))

    vocab = scan_stats_vocab(src, experiments)
    vocab2idx = {name: idx for idx, name in enumerate(vocab)}
    consensus = quick_offset_scan(src, experiments)

    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    config = TemplateMinerConfig()
    config.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tt.ini'))
    config.profiling_enabled = False
    miner = TemplateMiner(config=config)

    bounds, diags = [], []
    concat_offset = 0
    for exp_name in experiments:
        diag = process_experiment(exp_name, src, raw_dir, save_path, miner, vocab2idx, consensus)
        T = diag['T']
        bounds.append({'name': exp_name, 'start': concat_offset, 'end': concat_offset + T,
                       'has_fault': diag['label_nonzero'] > 0, 't0': diag['t0']})
        diags.append(diag)
        concat_offset += T
        logging.info(
            "experiment %s done: T=%d | logs kept=%d dropped=%d templates=%d | "
            "trace offset=%s method=%s peak_corr=%.4f spans=%d | label nonzero=%d",
            exp_name, T, diag['log_kept'], diag['log_dropped'], diag['templates'],
            f"{diag['offset']:.0f}" if diag['offset'] is not None else 'n/a',
            diag['method'], diag['peak_corr'], diag['num_spans'], diag['label_nonzero'])

    adjacency = np.zeros((TT_NUM_NODES, TT_NUM_NODES))
    for src_name, dst_name in TT_EDGES:
        adjacency[TT_SERVICE2NID[src_name], TT_SERVICE2NID[dst_name]] = 1
        adjacency[TT_SERVICE2NID[dst_name], TT_SERVICE2NID[src_name]] = 1
    pickle.dump(adjacency, open(os.path.join(save_path, 'trace_path.pkl'), 'wb'))
    pickle.dump(bounds, open(os.path.join(save_path, 'bounds.pkl'), 'wb'))
    pickle.dump(vocab, open(os.path.join(save_path, 'stats_vocab.pkl'), 'wb'))

    logging.info("=" * 80)
    logging.info("per-experiment diagnostics (T_total=%d):", concat_offset)
    for diag in diags:
        logging.info("  %s | T=%d | t0=%d | offset=%s | method=%s | peak_corr=%.4f | raw_spans=%d -> kept=%d",
                     diag['name'], diag['T'], diag['t0'],
                     f"{diag['offset']:.0f}" if diag['offset'] is not None else 'n/a',
                     diag['method'], diag['peak_corr'], diag['raw_spans'], diag['num_spans'])
        for s, e, count in diag['window_spans']:
            logging.info("      fault window [%d, %d) spans_in_window=%d", s, e, count)
    logging.info("drain templates total: %d | stats vocab: %d", len(miner.drain.clusters), len(vocab))
    logging.info("saved bounds.pkl (%d experiments), stats_vocab.pkl, trace_path.pkl to %s",
                 len(bounds), save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description='Preprocess the Eadro TT dataset (full, 11 experiments).')
    parser.add_argument('--src', type=str, default=DEFAULT_SRC)
    parser.add_argument('--raw_dir', type=str, default=DEFAULT_RAW_DIR)
    parser.add_argument('--save_path', type=str, default=DEFAULT_SAVE_PATH)
    cli = parser.parse_args()
    preprocess(cli.src, cli.raw_dir, cli.save_path)
