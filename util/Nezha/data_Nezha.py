import logging
import os
import pickle

import numpy as np
import pandas as pd

from util.Nezha.constant import (
    NEZHA_SERVICE2NID, NEZHA_NUM_NODES, LOG_LEVELS, NUM_LEVEL_CHANNELS,
)


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
        self.set, self.dataset, self.trace_type = {}, [], []

        bounds_path = os.path.join(self.rawdata_path, 'bounds.pkl')
        if not os.path.exists(bounds_path):
            raise FileNotFoundError(f"{bounds_path} not found; run util/Nezha/pre_Nezha.py first")
        self.bounds = pickle.load(open(bounds_path, 'rb'))

        # Log feature width = K template counts + [ERROR, WARNING, INFO, total].
        templates_path = os.path.join(self.rawdata_path, 'log_templates.csv')
        if not os.path.exists(templates_path):
            raise FileNotFoundError(f"{templates_path} not found; run util/Nezha/pre_Nezha.py first")
        self.num_templates = int(pd.read_csv(templates_path).shape[0])
        self.log_len = self.num_templates + NUM_LEVEL_CHANNELS
        if self.log_len != args['log_len']:
            logging.info("Nezha log_len auto-aligned: %d templates + %d level channels -> log_len=%d",
                         self.num_templates, NUM_LEVEL_CHANNELS, self.log_len)
        args['log_len'] = self.log_len

        dataset_file = os.path.join(self.dataset_path, 'dataset.pkl')
        if os.path.exists(dataset_file):
            try:
                self.read_data()
            except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as exc:
                logging.warning(
                    "Failed to read cached Nezha dataset from %s: %s. Rebuilding from raw data.",
                    self.dataset_path,
                    exc,
                )
                self._rebuild_from_raw()
        else:
            self._rebuild_from_raw()
        if not self.dataset:
            logging.warning("Cached Nezha dataset is empty, rebuilding from raw data.")
            self._rebuild_from_raw()

        self.graph = read_graph(data_dir=self.rawdata_path)
        self._autofill_dims()
        self._build_split()

    def _rebuild_from_raw(self):
        self.load_raw()
        logging.info("Transform data into time windows (within each day)")
        self.dataset = self._transform()
        self.save_data()

    def load_raw(self):
        if not os.path.exists(self.rawdata_path):
            raise FileNotFoundError(f"Find no Nezha preprocessed data at {self.rawdata_path}")
        logging.info("LOADing Nezha data ...")
        T_total = self.bounds[-1]['end']

        label_raw = pickle.load(open(os.path.join(self.rawdata_path, 'label.pkl'), 'rb'))
        label_int = label_raw.astype(int)
        self.set['label'] = np.eye(2)[label_int]
        # Nezha is fully labeled: the unknown class is always 0
        self.set['mask'] = np.eye(3)[label_int]

        metric = pd.read_csv(os.path.join(self.rawdata_path, 'metric.csv'), sep=',')
        metric_values = metric.drop(columns=['now', 'time_epoch', 'time']).values.astype(np.float32)
        assert len(metric_values) == T_total, f"metric rows {len(metric_values)} != {T_total}"
        metric_full = metric_values.reshape(T_total, self.num_node, -1)
        self.set['metric'] = metric_full

        self.set['log'] = self._load_logs(T_total)
        self.set['trace'] = self._load_traces(T_total)

    def _load_logs(self, T_total):
        K = self.num_templates
        width = self.log_len  # K + NUM_LEVEL_CHANNELS
        log_full = np.zeros((T_total, self.num_node, width), dtype=np.float32)
        log = pd.read_csv(os.path.join(self.rawdata_path, 'log.csv'), sep=',')
        if len(log):
            host_idx = log['Hostname'].map(NEZHA_SERVICE2NID)
            tid = log['templateid'].astype(int) - 1
            ts = log['@timestamp'].astype(int)
            base_valid = host_idx.notna() & (ts >= 0) & (ts < T_total)

            tmpl_valid = base_valid & (tid >= 0) & (tid < K)
            np.add.at(log_full, (ts[tmpl_valid].values,
                                 host_idx[tmpl_valid].values.astype(int),
                                 tid[tmpl_valid].values), 1)

            total_ch = K + len(LOG_LEVELS)
            np.add.at(log_full, (ts[base_valid].values,
                                 host_idx[base_valid].values.astype(int),
                                 np.full(int(base_valid.sum()), total_ch, dtype=int)), 1)

            levels = log['level'].fillna('').astype(str).values
            for i, level in enumerate(LOG_LEVELS):
                lvl_valid = base_valid & (levels == level)
                if lvl_valid.any():
                    np.add.at(log_full, (ts[lvl_valid].values,
                                         host_idx[lvl_valid].values.astype(int),
                                         np.full(int(lvl_valid.sum()), K + i, dtype=int)), 1)

        sec_max = log_full.max(axis=1)
        max_record = sec_max.max(axis=0)
        present = log_full.sum(axis=(1, 2)) > 0
        min_record = sec_max.min(axis=0) if present.all() else np.zeros(width)
        dis = max_record - min_record + 1e-6
        log_full = (log_full - min_record) / dis
        logging.info(f"Log data: {len(log)} rows, {K} templates + {NUM_LEVEL_CHANNELS} level channels, "
                     f"present minutes {int(present.sum())}/{T_total}")
        return np.nan_to_num(log_full).astype(np.float32)

    def _load_traces(self, T_total):
        self.trace_type = pickle.load(open(os.path.join(self.rawdata_path, 'stats_vocab.pkl'), 'rb'))
        stat2idx = {stat: idx for idx, stat in enumerate(self.trace_type)}
        num_stats = len(self.trace_type)
        trace_df = pd.read_csv(os.path.join(self.rawdata_path, 'trace.csv'), sep=',')
        trace_full = np.zeros((T_total, self.num_node, self.num_node, num_stats), dtype=np.float32)
        logging.info(f"Trace stats vocab: {num_stats}, rows: {len(trace_df)}")

        for bound in self.bounds:
            seg = trace_df[(trace_df['end_time'] >= bound['start']) & (trace_df['end_time'] < bound['end'])]
            if not len(seg):
                continue
            src = seg['cmbd_id'].map(NEZHA_SERVICE2NID)
            dst = seg['fatherpod'].map(NEZHA_SERVICE2NID)
            k = seg['stats'].map(stat2idx)
            valid = src.notna() & dst.notna() & k.notna()
            seg = seg[valid]
            T_day = bound['end'] - bound['start']
            trace_a = np.zeros((self.num_node, self.num_node, num_stats, T_day))
            np.add.at(
                trace_a,
                (src[valid].values.astype(int), dst[valid].values.astype(int), k[valid].values.astype(int),
                 (seg['end_time'] - bound['start']).values.astype(int)),
                seg['duration'].values,
            )
            trace_a = np.log1p(trace_a)
            trace_a = trace_a / (trace_a.mean() * 10 + 1e-6)
            trace_full[bound['start']:bound['end']] = trace_a.transpose(3, 0, 1, 2)
        return trace_full

    def read_data(self):
        logging.info("read Transform data")
        dataset_file = os.path.join(self.dataset_path, 'dataset.pkl')
        if os.path.exists(dataset_file):
            logging.info("Loading unified dataset.pkl...")
            with open(dataset_file, 'rb') as fh:
                self.dataset = pickle.load(fh)

    def save_data(self):
        logging.info("save Transform data")
        os.makedirs(self.dataset_path, exist_ok=True)
        dataset_file = os.path.join(self.dataset_path, 'dataset.pkl')
        with open(dataset_file, 'wb') as f:
            pickle.dump(self.dataset, f)
        logging.info(f"Successfully saved all data to {dataset_file}")

    def _transform(self):
        metric = self.set['metric']
        log = self.set['log']
        trace = self.set['trace']
        label = self.set['label']
        label_mask = self.set['mask']

        span = (self.window - 1) * self.step
        num, count1, count2 = 0, 0, 0
        data_list = []
        for day_id, bound in enumerate(self.bounds):
            if bound['end'] - bound['start'] < self.window:
                logging.info(f"day {bound['name']} too short, skipping")
                continue
            for t_start in range(bound['start'], bound['end'] - span, self.step):
                idx = np.arange(t_start, t_start + span + 1, self.step)
                last = idx[-1]
                record = {
                    'data_node': metric[idx][:, :, :self.metric_len],
                    'data_log': log[idx],
                    'data_edge': trace[idx],
                    'groundtruth_cls': label_mask[last],
                    'groundtruth_real': label[last],
                    'exp_id': day_id,
                }
                count1 += 1 if record['groundtruth_cls'].sum(axis=0)[1] > 0 else 0
                count2 += 1 if record['groundtruth_real'].sum(axis=0)[1] > 0 else 0
                num += 1
                data_list.append(record)
            logging.info(f"day {day_id} ({bound['name']}): windows so far {num}")
        logging.info(f"Total windows: {num}, abnormal windows cls:{count1} real:{count2}")
        return data_list

    def _autofill_dims(self):
        sample = self.dataset[0]
        num_stats = sample['data_edge'].shape[-1]
        if self.args.get('raw_edge', 0) == 0 or self.args['raw_edge'] > num_stats:
            self.args['raw_edge'] = num_stats
            logging.info(f"Auto-detected raw_edge = {num_stats}")

    def _build_split(self):
        fault_ids = [i for i, bound in enumerate(self.bounds) if bound['has_fault']]
        if not fault_ids:
            raise ValueError("no fault days found in bounds.pkl")
        test_experiment = int(self.args.get('test_experiment', -1))
        try:
            held_out = fault_ids[test_experiment]
        except IndexError:
            raise ValueError(
                f"test_experiment={test_experiment} out of range for {len(fault_ids)} fault days")

        exp_ids = np.array([sample['exp_id'] for sample in self.dataset])
        self.test_indices = np.flatnonzero(exp_ids == held_out).tolist()
        self.train_indices = np.flatnonzero(exp_ids != held_out).tolist()

        def abnormal_ratio(indices):
            if not indices:
                return 0.0
            hits = sum(1 for i in indices if self.dataset[i]['groundtruth_real'][:, 1].sum() > 0)
            return hits / len(indices)

        logging.info(
            "Nezha split: held-out day %d (%s) | train=%d (abnormal %.2f%%) test=%d (abnormal %.2f%%)",
            held_out, self.bounds[held_out]['name'],
            len(self.train_indices), abnormal_ratio(self.train_indices) * 100,
            len(self.test_indices), abnormal_ratio(self.test_indices) * 100,
        )
