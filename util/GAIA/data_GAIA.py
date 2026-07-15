import gc
import logging
import os
import pickle
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from util.GAIA.constant import *


def read_graph(data_dir):

    logging.info("read Graph edge data")
    graph_path = os.path.join(data_dir, 'trace_path.pkl')
    if not os.path.exists(graph_path):
        logging.info("read no graph data")
        return None
    data = pickle.load(open(graph_path, 'rb'))
    return data


class Process:
    def __init__(self, args):

        self.args = args
        self.window = args['window']
        self.step = args['step']
        self.dataset_path = args['dataset_path']
        self.rawdata_path = args['data_path']

        self.log_len = args['log_len']
        self.metric_len = args['raw_node']
        self.num_node = args['num_nodes']
        self.percent = args['label_percent']
        self.max_timesteps = args.get('max_timesteps', 0)
        self.set, self.dataset, self.trace_type = {}, [], []

        if self._dataset_cache_is_fresh():
            try:
                self.read_data()
                self.graph = read_graph(data_dir=self.rawdata_path)
            except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as exc:
                logging.warning(
                    "Failed to read cached GAIA dataset from %s: %s. Rebuilding from raw data.",
                    self.dataset_path,
                    exc,
                )
                self._rebuild_from_raw()
        else:
            self._rebuild_from_raw()

    def _rebuild_from_raw(self):
        self.load_raw()
        self.graph = read_graph(data_dir=self.rawdata_path)
        logging.info("Transform data into time windows")
        self.dataset = self._transform()
        self.save_data()

    def _dataset_cache_is_fresh(self):
        dataset_file = os.path.join(self.dataset_path, 'dataset.pkl')
        if not os.path.exists(dataset_file):
            return False

        raw_dependencies = [
            'metric.csv', 'label.csv', 'log.csv', 'trace.csv', 'trace_path.pkl'
        ]
        dataset_mtime = os.path.getmtime(dataset_file)
        for filename in raw_dependencies:
            path = os.path.join(self.rawdata_path, filename)
            if os.path.exists(path) and os.path.getmtime(path) > dataset_mtime:
                logging.info(f"Rebuilding dataset cache because {filename} is newer than dataset.pkl")
                return False
        return True

    def load_raw(self):
        if not os.path.exists(self.rawdata_path):
            logging.info("Find no data")
            return
        logging.info("LOADing GAIA data ...")

        interval_ms = GAIA_SAMPLE_INTERVAL * 1000

        label_df = pd.read_csv(os.path.join(self.rawdata_path, 'label.csv'))
        label_timestamps = set(label_df['timestamp'].values.tolist())
        logging.info(f"Label timesteps: {len(label_timestamps)}")

        logging.info("Loading metric data (chunked)...")
        metric_csv = os.path.join(self.rawdata_path, 'metric.csv')

        header_df = pd.read_csv(metric_csv, nrows=0)
        feature_cols = [c for c in header_df.columns if c != 'timestamp']

        service_feature_map = {}
        for svc in GAIA_SERVICES:
            svc_cols = sorted([c for c in feature_cols if c.startswith(svc + '_')])
            service_feature_map[svc] = svc_cols

        features_per_node = min(len(cols) for cols in service_feature_map.values())

        if self.metric_len == 0 or self.metric_len > features_per_node:
            self.metric_len = features_per_node
            self.args['raw_node'] = features_per_node
            logging.info(f"Auto-detected raw_node = {features_per_node}")
        actual_features = self.metric_len
        logging.info(f"Features per node: {features_per_node}, using: {actual_features}")

        needed_cols = ['timestamp']
        for svc in GAIA_SERVICES:
            needed_cols.extend(service_feature_map[svc][:actual_features])

        svc_col_groups = {}
        for svc in GAIA_SERVICES:
            svc_col_groups[svc] = service_feature_map[svc][:actual_features]

        aggregated_chunks = []
        chunk_size = 200000  
        total_rows = 0

        cache_file = os.path.join(self.rawdata_path, 'metric_aggregated.pkl')
        use_metric_cache = (
            os.path.exists(cache_file) and
            os.path.getmtime(cache_file) >= os.path.getmtime(metric_csv)
        )
        if use_metric_cache:
            logging.info(f"Loading aggregated metric data from cache: {cache_file}")
            with open(cache_file, 'rb') as fh:
                all_grouped = pickle.load(fh)
        else:
            if os.path.exists(cache_file):
                logging.info(f"Ignoring stale aggregated metric cache: {cache_file}")
            for chunk in tqdm(pd.read_csv(metric_csv, usecols=needed_cols, chunksize=chunk_size),
                              desc='Reading metric chunks'):
                total_rows += len(chunk)

                chunk['aligned_ts'] = (chunk['timestamp'].values // interval_ms) * interval_ms

                valid_mask = chunk['aligned_ts'].isin(label_timestamps)
                chunk_valid = chunk[valid_mask]

                if len(chunk_valid) == 0:
                    continue

                grouped = chunk_valid.drop(columns=['timestamp']).groupby('aligned_ts').mean()
                aggregated_chunks.append(grouped)

            logging.info(f"Read {total_rows} metric rows from {len(aggregated_chunks)} valid chunks")

            if aggregated_chunks:
                all_grouped = pd.concat(aggregated_chunks)
                all_grouped = all_grouped.groupby(level=0).mean()
                logging.info(f"Saving aggregated metric data to cache: {cache_file}")
                with open(cache_file, 'wb') as fh:
                    pickle.dump(all_grouped, fh)
            else:
                all_grouped = pd.DataFrame()

        del aggregated_chunks
        gc.collect()

        common_timestamps = np.array(sorted(all_grouped.index.values.astype(int)))
        logging.info(f"Found {len(common_timestamps)} aligned timestamps with valid metric data")

        if self.max_timesteps > 0 and len(common_timestamps) > self.max_timesteps:
            logging.info(f"Limiting to first {self.max_timesteps} timesteps (out of {len(common_timestamps)})")
            common_timestamps = common_timestamps[:self.max_timesteps]

        metric_data = np.zeros((len(common_timestamps), self.num_node, actual_features))
        for ni, svc in enumerate(GAIA_SERVICES):
            cols = svc_col_groups[svc]
            svc_vals = all_grouped.loc[common_timestamps, cols].values.astype(float)
            metric_data[:, ni, :] = np.nan_to_num(svc_vals)

        del all_grouped
        gc.collect()

        self.actual_metric_features = actual_features
        valid_timestamps = common_timestamps
        self.valid_timestamps = valid_timestamps
        logging.info(f"Metric data shape: {metric_data.shape}")

        if len(valid_timestamps) > 1:
            diffs = np.diff(valid_timestamps)
            gap_threshold = interval_ms * 3
            segments = []
            seg_start = 0
            for i, d in enumerate(diffs):
                if d > gap_threshold:
                    segments.append((seg_start, i + 1))
                    seg_start = i + 1
            segments.append((seg_start, len(valid_timestamps)))
            logging.info(f"Found {len(segments)} continuous segments:")
            for si, (s, e) in enumerate(segments):
                logging.info(f"  Segment {si}: {e - s} timesteps")
        else:
            segments = [(0, len(valid_timestamps))]
        self.segments = segments

        label_indexed = label_df.set_index('timestamp')
        label_valid = label_indexed.loc[valid_timestamps, GAIA_SERVICES].values.astype(int)
        label = np.eye(2)[label_valid]

        label_mask_valid = label_valid.copy()
        times = np.zeros((self.num_node, 2))
        for idx in range(label.shape[0]):
            if idx < self.window:
                continue
            times += label[idx]
            mask = times[label[idx] == 1] % 10 >= 10 * self.percent
            label_mask_valid[idx, mask] = 2
        label_mask = np.eye(3)[label_mask_valid]
        del label_df
        gc.collect()

        logging.info("Loading log data...")
        log_csv = os.path.join(self.rawdata_path, 'log.csv')
        log_df = pd.read_csv(log_csv)
        template_cols = sorted([c for c in log_df.columns if c.startswith('template_')])
        level_cols = [c for c in ['level_INFO', 'level_WARNING', 'level_ERROR', 'level_DEBUG', 'level_UNKNOWN']
                      if c in log_df.columns]
        extra_log_cols = [c for c in ['log_total'] if c in log_df.columns]
        log_feature_cols = template_cols + level_cols + extra_log_cols
        actual_log_len = len(log_feature_cols)

        if self.log_len == 0 or self.log_len > actual_log_len:
            self.log_len = actual_log_len
            self.args['log_len'] = actual_log_len
            logging.info(f"Auto-detected log_len = {actual_log_len}")
        else:
            actual_log_len = self.log_len
        logging.info(
            f"Log features: templates={len(template_cols)}, levels={len(level_cols)}, extras={len(extra_log_cols)}, "
            f"using {actual_log_len}"
        )

        logging.info("Building log feature matrix...")
        svc_to_idx = {svc: i for i, svc in enumerate(GAIA_SERVICES)}
        log_df['svc_idx'] = log_df['service'].map(svc_to_idx)
        log_df = log_df.dropna(subset=['svc_idx'])
        log_df['svc_idx'] = log_df['svc_idx'].astype(int)

        ts_set = set(valid_timestamps.tolist())
        log_df_valid = log_df[log_df['timestamp'].isin(ts_set)].copy()

        use_log_cols = log_feature_cols[:actual_log_len]

        log_values = log_df_valid[use_log_cols].values.astype(float)
        max_record = np.zeros(actual_log_len)
        min_record = np.ones(actual_log_len) * np.inf
        if len(log_values) > 0:
            max_record = np.maximum(max_record, log_values.max(axis=0))
            min_record = np.minimum(min_record, log_values.min(axis=0))

        log_record = {}
        for ts in valid_timestamps:
            log_record[ts] = np.zeros((self.num_node, actual_log_len))

        logging.info("Filling log matrix (vectorized)...")
        for svc_idx, group in log_df_valid.groupby('svc_idx'):
            for _, row in group.iterrows():
                ts = int(row['timestamp'])
                log_record[ts][int(svc_idx), :] = row[use_log_cols].values.astype(float)


        dis = max_record - min_record + 1e-6
        for ts in log_record:
            log_record[ts] = (log_record[ts] - min_record) / dis

        del log_df, log_df_valid
        gc.collect()

        logging.info("Loading trace data...")
        trace_csv = os.path.join(self.rawdata_path, 'trace.csv')
        trace_df = pd.read_csv(trace_csv)
        self.trace_type = sorted(trace_df['status_code'].astype(str).unique().tolist())
        num_trace_types = len(self.trace_type)

        if self.args.get('raw_edge', 0) == 0 or self.args.get('raw_edge', 0) > num_trace_types:
            self.args['raw_edge'] = num_trace_types
            logging.info(f"Auto-detected raw_edge = {num_trace_types}")
        logging.info(f"Trace types: {self.trace_type}")

        trace_data = np.zeros((len(valid_timestamps), self.num_node, self.num_node, num_trace_types))

        sc_to_idx = {sc: i for i, sc in enumerate(self.trace_type)}
        trace_df['sc_idx'] = trace_df['status_code'].astype(str).map(sc_to_idx)
        trace_df = trace_df.dropna(subset=['sc_idx'])
        trace_df['sc_idx'] = trace_df['sc_idx'].astype(int)

        ts_to_idx = {int(ts): i for i, ts in enumerate(valid_timestamps)}
        trace_valid = trace_df[trace_df['timestamp_aligned'].isin(ts_set)].copy()
        trace_valid['ts_idx'] = trace_valid['timestamp_aligned'].map(ts_to_idx)
        logging.info(f"Valid trace records: {len(trace_valid)} / {len(trace_df)}")

        has_directed_edges = {'src_service', 'dst_service'}.issubset(trace_valid.columns)
        if has_directed_edges:
            trace_valid['src_idx'] = trace_valid['src_service'].map(svc_to_idx)
            trace_valid['dst_idx'] = trace_valid['dst_service'].map(svc_to_idx)
            trace_valid = trace_valid.dropna(subset=['src_idx', 'dst_idx'])
            trace_valid['src_idx'] = trace_valid['src_idx'].astype(int)
            trace_valid['dst_idx'] = trace_valid['dst_idx'].astype(int)

            if len(trace_valid) > 0:
                agg = trace_valid.groupby(
                    ['ts_idx', 'src_idx', 'dst_idx', 'sc_idx']
                )['duration_sum'].sum().reset_index()
                ts_indices = agg['ts_idx'].values.astype(int)
                src_indices = agg['src_idx'].values.astype(int)
                dst_indices = agg['dst_idx'].values.astype(int)
                sc_indices = agg['sc_idx'].values.astype(int)
                duration_values = agg['duration_sum'].values
                trace_data[ts_indices, src_indices, dst_indices, sc_indices] = duration_values
            logging.info("Using directed trace edge features (src_service -> dst_service)")
        else:
            if 'service_name' not in trace_df.columns and 'service' in trace_df.columns:
                trace_df = trace_df.rename(columns={'service': 'service_name'})
                trace_valid = trace_df[trace_df['timestamp_aligned'].isin(ts_set)].copy()
                trace_valid['ts_idx'] = trace_valid['timestamp_aligned'].map(ts_to_idx)

            if 'service_name' not in trace_valid.columns:
                raise KeyError("src_service/dst_service or service_name")

            trace_valid['svc_idx'] = trace_valid['service_name'].map(svc_to_idx)
            trace_valid = trace_valid.dropna(subset=['svc_idx'])
            trace_valid['svc_idx'] = trace_valid['svc_idx'].astype(int)

            if len(trace_valid) > 0:
                agg = trace_valid.groupby(['ts_idx', 'svc_idx', 'sc_idx'])['duration_sum'].sum().reset_index()
                ts_indices = agg['ts_idx'].values.astype(int)
                svc_indices = agg['svc_idx'].values.astype(int)
                sc_indices = agg['sc_idx'].values.astype(int)
                duration_values = agg['duration_sum'].values
                trace_data[ts_indices, svc_indices, svc_indices, sc_indices] = duration_values
            logging.warning("Using legacy self-loop trace features; re-run util.GAIA.pre_GAIA for directed edges")

        trace_mean = trace_data.mean(axis=0)
        trace_data = trace_data / (trace_mean * 10 + 1e-6)
        logging.info(f"Trace data shape: {trace_data.shape}")

        del trace_df, trace_valid
        gc.collect()

        self.set['metric'] = metric_data
        self.set['log'] = log_record
        self.set['trace'] = trace_data
        self.set['label'] = label
        self.set['mask'] = label_mask

        logging.info(f"Data loading complete. Timesteps: {len(valid_timestamps)}, "
                     f"Nodes: {self.num_node}, "
                     f"Metric(raw_node): {actual_features}/node, "
                     f"Log(log_len): {actual_log_len}, "
                     f"Trace(raw_edge): {num_trace_types}")

    def read_data(self):
        logging.info("read Transform data")
        if not os.path.exists(self.dataset_path):
            logging.info("read no data")
            return None, None

        dataset_file = os.path.join(self.dataset_path, 'dataset.pkl')
        if os.path.exists(dataset_file):
            logging.info("Loading unified dataset.pkl...")
            with open(dataset_file, 'rb') as fh:
                self.dataset = pickle.load(fh)
            if self.max_timesteps > 0:
                logging.info(f"Limiting to first {self.max_timesteps} timesteps")
                self.dataset = self.dataset[:self.max_timesteps]
        else:
            logging.info("dataset.pkl not found, loading individual .pkl files...")
            dataset = os.listdir(self.dataset_path)
            dataset = [f for f in dataset if f.endswith('.pkl')]
            dataset.sort(key=lambda x: (int(re.split(r"[-_.]", x)[0])))
    
            if self.max_timesteps > 0:
                dataset = dataset[:self.max_timesteps]
    
            for file in tqdm(dataset, desc="Loading saved windows"):
                with open(os.path.join(self.dataset_path, file), 'rb') as fh:
                    data = pickle.load(fh)
                self.dataset.append(data)

        if self.dataset:
            sample = self.dataset[0]
            if 'data_node' in sample and self.args.get('raw_node', 0) == 0:
                self.args['raw_node'] = sample['data_node'].shape[-1]
                self.metric_len = self.args['raw_node']
                logging.info(f"Inferred raw_node = {self.args['raw_node']} from saved data")
            if 'data_log' in sample and self.args.get('log_len', 0) == 0:
                self.args['log_len'] = sample['data_log'].shape[-1]
                self.log_len = self.args['log_len']
                logging.info(f"Inferred log_len = {self.args['log_len']} from saved data")
            if 'data_edge' in sample and self.args.get('raw_edge', 0) == 0:
                self.args['raw_edge'] = sample['data_edge'].shape[-1]
                logging.info(f"Inferred raw_edge = {self.args['raw_edge']} from saved data")

    def save_data(self):
        logging.info("save Transform data")
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
            
        logging.info("Removing 'name' key and saving to a unified dataset.pkl...")
        for item in self.dataset:
            if 'name' in item:
                del item['name']
                
        dataset_file = os.path.join(self.dataset_path, 'dataset.pkl')
        with open(dataset_file, 'wb') as f:
            pickle.dump(self.dataset, f)
        logging.info(f"Successfully saved all data to {dataset_file}")

    def _transform(self):
        self.trace_type = list(set(self.trace_type))
        num = 0
        count1, count2, count3 = 0, 0, 0
        data_list = []

        metric = self.set['metric']
        log = self.set['log']
        trace = self.set['trace']
        label = self.set['label']
        label_mask = self.set['mask']

        valid_ts = self.valid_timestamps

        for seg_idx, (seg_start, seg_end) in enumerate(self.segments):
            seg_len = seg_end - seg_start
            if seg_len < self.window:
                logging.info(f"Segment {seg_idx} too short ({seg_len} < {self.window}), skipping")
                continue

            for i in range(seg_len - self.window + 1):
                global_start = seg_start + i
                global_end = seg_start + i + self.window

                record = {}
                if num % 500 == 0:
                    logging.info(f"deal ...{num}...trace:{count3}...error see:{count1}...error real:{count2}")

                select_metric = metric[global_start:global_end]
                record['data_node'] = select_metric[:, :, :self.metric_len]

                log_data = []
                for t_idx in range(global_start, global_end):
                    ts = valid_ts[t_idx]
                    if ts in log:
                        log_data.append(log[ts])
                    else:
                        log_data.append(np.zeros((self.num_node, self.log_len)))
                log_arr = np.stack(log_data, axis=0)
                if log_arr.shape[-1] < self.log_len:
                    padding = np.zeros((log_arr.shape[0], log_arr.shape[1],
                                       self.log_len - log_arr.shape[-1]))
                    log_arr = np.concatenate([log_arr, padding], axis=-1)
                elif log_arr.shape[-1] > self.log_len:
                    log_arr = log_arr[:, :, :self.log_len]
                record['data_log'] = np.nan_to_num(log_arr)

                select_label = label[global_end - 1, :]
                select_mask = label_mask[global_end - 1, :]
                record['groundtruth_cls'] = select_mask
                record['groundtruth_real'] = select_label
                count1 += 1 if record['groundtruth_cls'].sum(axis=0)[1] > 0 else 0
                count2 += 1 if record['groundtruth_real'].sum(axis=0)[1] > 0 else 0

                select_trace = trace[global_start:global_end]
                if select_trace.shape[-1] < len(self.trace_type):
                    padding = np.zeros((*select_trace.shape[:-1],
                                       len(self.trace_type) - select_trace.shape[-1]))
                    select_trace = np.concatenate([select_trace, padding], axis=-1)
                count3 += 1 if select_trace.sum() > 0 else 0
                record['data_edge'] = select_trace
                record['name'] = f'{num}'
                num += 1
                data_list.append(record)
                del record

        logging.info(f"deal ...{num}...error see:{count1}...error real:{count2}...")
        return data_list
