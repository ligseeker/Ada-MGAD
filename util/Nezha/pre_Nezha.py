import argparse
import glob
import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util.Nezha.constant import (
    NEZHA_DAYS, NEZHA_SERVICES, NEZHA_SERVICE2NID, NEZHA_NUM_NODES,
    NEZHA_KPIS, FAULT_DURATION_SEC, BUCKET_SEC, GO_SERVICES, LOG_LEVELS,
    pod_to_service, normalize_level, level_from_text,
)

DEFAULT_RAW_DIR = "/home/zhangll24/RCA_project/Nezha"
DEFAULT_SAVE_PATH = "./data/Nezha-pre"


def robust_zscore(frame):
    medians = frame.median()
    mads = frame.sub(medians).abs().median()
    flat = mads < 1e-9
    normalized = frame.sub(medians).div(1.4826 * mads.where(~flat, 1.0))
    normalized.loc[:, flat] = 0.0
    return normalized.fillna(0.0)


def epoch_from_time_column(time_str):
    return pd.to_datetime(time_str.str[:19], format='%Y-%m-%d %H:%M:%S', utc=True).astype(np.int64) // 10**9


def utc_minute(epoch):
    """Epoch seconds -> 'YYYY-MM-DD HH:MM' UTC string (array or scalar)."""
    return pd.to_datetime(np.asarray(epoch, dtype=np.int64), unit='s', utc=True).strftime('%Y-%m-%d %H:%M')


def repair_timestamps(df, pod):
    time_epoch = epoch_from_time_column(df['Time']).values
    delta = df['TimeStamp'].values - time_epoch
    offset = int(np.median(delta))
    if abs(offset) <= 1:
        return df['TimeStamp'].values.astype(np.int64)
    repaired = df['TimeStamp'].values - offset
    residual = int(np.abs(repaired - time_epoch).max())
    if residual <= 30:
        logging.warning("pod %s: TimeStamp shifted by %d seconds, repaired with constant offset", pod, offset)
        return repaired.astype(np.int64)
    logging.warning("pod %s: TimeStamp corrupted (offset %d, residual %d s), rebuilt from Time column",
                    pod, offset, residual)
    return time_epoch.astype(np.int64)


def load_day_metric(raw_dir, day):
    metric_dir = os.path.join(raw_dir, 'rca_data', day, 'metric')
    frames = {}
    for pod in sorted(os.listdir(metric_dir)):
        if not pod.endswith('_metric.csv'):
            continue
        service = pod_to_service(pod[:-len('_metric.csv')])
        assert service in NEZHA_SERVICE2NID, f"unknown service {service} from {pod}"
        df = pd.read_csv(os.path.join(metric_dir, pod))
        missing = [kpi for kpi in NEZHA_KPIS if kpi not in df.columns]
        assert not missing, f"missing KPI columns {missing} in {pod}"
        ts = repair_timestamps(df, pod)
        frames[service] = pd.DataFrame(
            {f'{service}_{kpi}': df[kpi].values for kpi in NEZHA_KPIS}, index=ts)
    assert set(frames) == set(NEZHA_SERVICES), f"day {day}: pods {sorted(frames)} != expected services"

    t0 = int(min(frame.index.min() for frame in frames.values()))
    t_end = int(max(frame.index.max() for frame in frames.values()))
    T = (t_end - t0) // BUCKET_SEC + 1
    parts = []
    for service in NEZHA_SERVICES:
        frame = frames[service]
        buckets = (frame.index.values - t0) // BUCKET_SEC
        grouped = frame.groupby(buckets).mean()
        grouped = grouped.reindex(range(T)).ffill().fillna(0.0)
        parts.append(grouped)
    metric = pd.concat(parts, axis=1).sort_index()
    assert len(metric) == T, f"day {day}: metric rows {len(metric)} != {T}"
    logging.info("day %s metric: t0=%d t_end=%d rows=%d, per-pod rows=%s",
                 day, t0, t_end, T,
                 {svc: len(frames[svc]) for svc in NEZHA_SERVICES})
    return robust_zscore(metric), t0, T


def parse_log_record(service, raw):
    """Extract (message, level) from one raw log cell, per-service (Nezha log_parsing.py).

    Go services (adservice/cartservice) keep the raw container text as the message;
    adservice additionally carries a textual level token, cartservice carries none.
    All other services are double-JSON with an explicit 'message' and 'severity'.
    """
    try:
        outer = json.loads(raw)
        inner_raw = outer['log']
    except (json.JSONDecodeError, KeyError, TypeError):
        return raw, ''

    if service in GO_SERVICES:
        message = str(inner_raw)
        level = level_from_text(message) if service == 'adservice' else ''
        return message, level

    try:
        inner = json.loads(inner_raw)
        message = inner.get('message')
        level = normalize_level(inner.get('severity', ''))
        return (message if message else raw), level
    except (json.JSONDecodeError, TypeError, AttributeError):
        return raw, ''


def day_window_files(raw_dir, day, kind):
    files = sorted(glob.glob(os.path.join(raw_dir, 'rca_data', day, kind, '*.csv')))
    files += sorted(glob.glob(os.path.join(raw_dir, 'construct_data', day, kind, '*.csv')))
    return files


def load_day_logs(raw_dir, day, miner, t0, T, concat_offset):
    seen, rows, skipped = set(), [], 0
    files = day_window_files(raw_dir, day, 'log')
    for fi, path in enumerate(files):
        df = pd.read_csv(path, usecols=['TimeUnixNano', 'PodName', 'SpanID', 'Log'])
        for ts_ns, pod, span_id, raw in df.values.tolist():
            key = (ts_ns, pod, span_id)
            if key in seen:
                continue
            seen.add(key)
            bucket = int(ts_ns // 10**9 - t0) // BUCKET_SEC
            if bucket < 0 or bucket >= T:
                skipped += 1
                continue
            service = pod_to_service(pod)
            if service not in NEZHA_SERVICE2NID:
                skipped += 1
                continue
            message, level = parse_log_record(service, raw)
            cluster_id = miner.add_log_message(message)['cluster_id']
            rows.append((cluster_id, service, bucket + concat_offset, level, t0 + bucket * BUCKET_SEC))
        logging.info("day %s log file %d/%d %s: kept=%d clusters=%d",
                     day, fi + 1, len(files), os.path.basename(path), len(rows),
                     len(miner.drain.clusters))
    logging.info("day %s logs: %d rows kept (%d out-of-range/unknown skipped)", day, len(rows), skipped)
    return pd.DataFrame(rows, columns=['templateid', 'Hostname', '@timestamp', 'level', 'time_epoch'])


def load_day_traces(raw_dir, day, t0, T, concat_offset):
    seen, records, raw_spans = set(), [], 0
    for path in day_window_files(raw_dir, day, 'trace'):
        df = pd.read_csv(path)
        raw_spans += len(df)
        span_pod = dict(zip(df['SpanID'], df['PodName']))
        fresh = df[~df['SpanID'].isin(seen)]
        seen.update(df['SpanID'].values.tolist())
        fresh = fresh[fresh['ParentID'] != 'root']
        parent_pod = fresh['ParentID'].map(span_pod)
        fresh = fresh[parent_pod.notna()]
        src = fresh['PodName'].map(pod_to_service)
        dst = parent_pod.map(pod_to_service)
        mask = (src != dst) & src.isin(NEZHA_SERVICE2NID) & dst.isin(NEZHA_SERVICE2NID)
        fresh = fresh[mask]
        src, dst = src[mask], dst[mask]
        if not len(fresh):
            continue
        end_sec = fresh['EndTimeUnixNano'].values // 10**9
        end_bucket = (end_sec - t0) // BUCKET_SEC
        keep = (end_bucket >= 0) & (end_bucket < T)
        duration = (fresh['EndTimeUnixNano'].values - fresh['StartTimeUnixNano'].values) / 1e9
        records.append(pd.DataFrame({
            'cmbd_id': src.values[keep],
            'fatherpod': dst.values[keep],
            'stats': fresh['OperationName'].values[keep],
            'end_time': end_bucket[keep] + concat_offset,
            'end_time_epoch': end_sec[keep],
            'duration': duration[keep],
        }))
    columns = ['cmbd_id', 'fatherpod', 'stats', 'end_time', 'end_time_epoch', 'duration']
    trace_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=columns)
    logging.info("day %s traces: %d raw spans, %d unique SpanIDs, %d edge records kept",
                 day, raw_spans, len(seen), len(trace_df))
    return trace_df, raw_spans


def load_day_labels(raw_dir, day, t0, T, concat_offset):
    label = np.zeros((T, NEZHA_NUM_NODES))
    fault_records = []
    fault_path = os.path.join(raw_dir, 'rca_data', day, f'{day}-fault_list.json')
    fault_data = json.load(open(fault_path))
    faults = [fault for faults in fault_data.values() for fault in faults]
    for fault in sorted(faults, key=lambda item: int(item['inject_timestamp'])):
        service = pod_to_service(fault['inject_pod'])
        assert service in NEZHA_SERVICE2NID, f"unknown fault pod {fault['inject_pod']}"
        start = int(fault['inject_timestamp'])
        duration = int(fault.get('duration', FAULT_DURATION_SEC))
        lo = (start - t0) // BUCKET_SEC
        hi = (start + duration - 1 - t0) // BUCKET_SEC
        lo, hi = max(0, lo), min(T - 1, hi)
        label[lo:hi + 1, NEZHA_SERVICE2NID[service]] = 1
        nows = [concat_offset + i for i in range(lo, hi + 1)]
        fault_records.append({
            'day': day,
            'service': service,
            'inject_pod': fault['inject_pod'],
            'inject_type': fault['inject_type'],
            'inject_time': fault['inject_time'],
            'start_epoch': start,
            'duration_sec': duration,
            'end_epoch': start + duration,
            'lo_bucket': concat_offset + lo,
            'hi_bucket': concat_offset + hi,
            'num_buckets': hi - lo + 1,
        })
        logging.info("day %s fault %-19s %-14s %-38s +%ds -> buckets %s",
                     day, fault['inject_time'], fault['inject_type'], service,
                     (hi - lo + 1) * BUCKET_SEC, nows)
    return label, fault_records


def load_adjacency(raw_dir):
    adjacency = np.zeros((NEZHA_NUM_NODES, NEZHA_NUM_NODES))
    for day in NEZHA_DAYS:
        dep = pd.read_csv(os.path.join(raw_dir, 'rca_data', day, 'metric', 'dependency.csv'))
        for src, dst in dep[['Source', 'Target']].values.tolist():
            if src in NEZHA_SERVICE2NID and dst in NEZHA_SERVICE2NID:
                adjacency[NEZHA_SERVICE2NID[src], NEZHA_SERVICE2NID[dst]] = 1
                adjacency[NEZHA_SERVICE2NID[dst], NEZHA_SERVICE2NID[src]] = 1
    return adjacency


def save_log_templates(miner, save_path):
    rows = []
    for cluster in miner.drain.clusters:
        rows.append({
            'cluster_id': cluster.cluster_id,
            'size': cluster.size,
            'template': ' '.join(cluster.log_template_tokens),
        })
    templates = pd.DataFrame(rows, columns=['cluster_id', 'size', 'template']).sort_values('cluster_id')
    templates.to_csv(os.path.join(save_path, 'log_templates.csv'), index=False)
    logging.info("saved log_templates.csv: %d templates", len(templates))
    return len(templates)


def preprocess(raw_dir, save_path):
    os.makedirs(save_path, exist_ok=True)

    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    config = TemplateMinerConfig()
    config.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nezha.ini'))
    config.profiling_enabled = False
    # Mine in-memory; a FilePersistence handler would rewrite the state file on every
    # cluster change and stall on ~4M messages. State is snapshotted once at the end.
    miner = TemplateMiner(config=config)

    metric_parts, log_parts, trace_parts, label_parts, bounds = [], [], [], [], []
    day_epochs, fault_alignment, diag = [], [], []
    concat_offset = 0
    for day in NEZHA_DAYS:
        logging.info("processing day %s", day)
        metric, t0, T = load_day_metric(raw_dir, day)
        metric_parts.append(metric.values.astype(np.float32))
        day_epochs.append(t0 + np.arange(T) * BUCKET_SEC)
        log_parts.append(load_day_logs(raw_dir, day, miner, t0, T, concat_offset))
        trace_df, raw_spans = load_day_traces(raw_dir, day, t0, T, concat_offset)
        if len(trace_df):
            trace_parts.append(trace_df)
        label, fault_records = load_day_labels(raw_dir, day, t0, T, concat_offset)
        label_parts.append(label)
        fault_alignment.extend(fault_records)
        bounds.append({'name': day, 'start': concat_offset, 'end': concat_offset + T,
                       't0_epoch': t0, 'has_fault': len(fault_records) > 0})
        diag.append({'name': day, 'T': T, 'raw_spans': raw_spans, 'kept_spans': len(trace_df),
                     'num_faults': len(fault_records), 'label_nonzero': int(label.sum())})
        concat_offset += T
    T_total = concat_offset

    # ---- metric.csv (dual-track: 0-based 'now' + absolute time) ----
    columns = [f'{service}_{kpi}' for service in NEZHA_SERVICES for kpi in NEZHA_KPIS]
    metric_all = np.concatenate(metric_parts, axis=0)
    epoch_all = np.concatenate(day_epochs).astype(np.int64)
    assert len(epoch_all) == T_total
    metric_df = pd.DataFrame(metric_all, columns=columns)
    metric_df.insert(0, 'time', utc_minute(epoch_all))
    metric_df.insert(0, 'time_epoch', epoch_all)
    metric_df.insert(0, 'now', np.arange(T_total))
    metric_df.to_csv(os.path.join(save_path, 'metric.csv'), index=False)
    logging.info("saved metric.csv: %s", metric_df.shape)

    # ---- log.csv ----
    log_df = pd.concat(log_parts, ignore_index=True) if log_parts else pd.DataFrame(
        columns=['templateid', 'Hostname', '@timestamp', 'level', 'time_epoch'])
    log_df = log_df.sort_values(by='@timestamp')
    log_df.to_csv(os.path.join(save_path, 'log.csv'), index=False)
    logging.info("saved log.csv: %d rows", len(log_df))

    # ---- trace.csv ----
    trace_columns = ['cmbd_id', 'fatherpod', 'stats', 'end_time', 'end_time_epoch', 'duration']
    trace_df = pd.concat(trace_parts, ignore_index=True).sort_values(by='end_time') if trace_parts \
        else pd.DataFrame(columns=trace_columns)
    trace_df.to_csv(os.path.join(save_path, 'trace.csv'), index=False)
    logging.info("saved trace.csv: %d spans", len(trace_df))

    # ---- label.pkl + label.csv + fault_alignment.csv ----
    label_all = np.concatenate(label_parts, axis=0)
    pickle.dump(label_all, open(os.path.join(save_path, 'label.pkl'), 'wb'))
    logging.info("saved label.pkl: %s, nonzero=%d", label_all.shape, int(label_all.sum()))

    label_df = pd.DataFrame(label_all.astype(int), columns=NEZHA_SERVICES)
    label_df.insert(0, 'time', utc_minute(epoch_all))
    label_df.insert(0, 'time_epoch', epoch_all)
    label_df.insert(0, 'now', np.arange(T_total))
    label_df.to_csv(os.path.join(save_path, 'label.csv'), index=False)

    alignment_df = pd.DataFrame(fault_alignment)
    alignment_df['start_time'] = utc_minute(alignment_df['start_epoch'].values)
    alignment_df['end_time'] = utc_minute(alignment_df['end_epoch'].values)
    alignment_df.to_csv(os.path.join(save_path, 'fault_alignment.csv'), index=False)
    logging.info("saved fault_alignment.csv: %d faults, %d with >=1 bucket",
                 len(alignment_df), int((alignment_df['num_buckets'] > 0).sum()))
    assert int((alignment_df['num_buckets'] > 0).sum()) == len(alignment_df), \
        "some faults landed outside the metric grid"

    # ---- graph / bounds / stats vocab ----
    adjacency = load_adjacency(raw_dir)
    pickle.dump(adjacency, open(os.path.join(save_path, 'trace_path.pkl'), 'wb'))
    logging.info("saved trace_path.pkl: %dx%d adjacency, %d edges",
                 NEZHA_NUM_NODES, NEZHA_NUM_NODES, int(adjacency.sum() // 2))

    pickle.dump(bounds, open(os.path.join(save_path, 'bounds.pkl'), 'wb'))
    stats_vocab = sorted(trace_df['stats'].unique().tolist()) if len(trace_df) else []
    pickle.dump(stats_vocab, open(os.path.join(save_path, 'stats_vocab.pkl'), 'wb'))
    logging.info("saved bounds.pkl (%d days), stats_vocab.pkl (%d stats)", len(bounds), len(stats_vocab))

    # ---- log templates + miner state ----
    num_templates = save_log_templates(miner, save_path)
    from drain3.file_persistence import FilePersistence
    miner.persistence_handler = FilePersistence(os.path.join(save_path, 'miner_state.bin'))
    miner.save_state(snapshot_reason='preprocess_complete')
    logging.info("saved miner_state.bin (drain3 snapshot)")

    # ---- level-concentration diagnostic ----
    abnormal_bucket = label_all.sum(axis=1) > 0
    for level in LOG_LEVELS:
        rows = log_df[log_df['level'] == level]
        ts = rows['@timestamp'].values
        in_fault = int(((ts >= 0) & (ts < T_total) & abnormal_bucket[np.clip(ts, 0, T_total - 1)]).sum())
        logging.info("level %-8s: %6d rows, %6d (%.1f%%) inside fault buckets",
                     level, len(rows), in_fault, 100.0 * in_fault / max(1, len(rows)))

    logging.info("=" * 80)
    logging.info("per-day diagnostics (T_total=%d, templates=%d):", T_total, num_templates)
    for entry in diag:
        logging.info("  %s | T=%d | faults=%d | label_nonzero=%d | raw_spans=%d -> kept_spans=%d",
                     entry['name'], entry['T'], entry['num_faults'],
                     entry['label_nonzero'], entry['raw_spans'], entry['kept_spans'])
    logging.info("all data saved to %s", save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description='Preprocess the Nezha OnlineBoutique dataset.')
    parser.add_argument('--raw_dir', type=str, default=DEFAULT_RAW_DIR)
    parser.add_argument('--save_path', type=str, default=DEFAULT_SAVE_PATH)
    args = parser.parse_args()
    preprocess(args.raw_dir, args.save_path)
