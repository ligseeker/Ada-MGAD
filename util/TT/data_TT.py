import logging
import os
import pickle

import numpy as np

from util.TT.constant import TT_NUM_NODES


def read_graph(data_dir):
    logging.info("read Graph edge data")
    graph_path = os.path.join(data_dir, 'trace_path.pkl')
    if not os.path.exists(graph_path):
        logging.info("read no graph data")
        return None
    data = pickle.load(open(graph_path, 'rb'))
    return data


class WindowDataset:
    def __init__(self, base, index, window, step, metric_len):
        self.metric = base['metric']
        self.log = base['log']
        self.trace = base['trace']
        self.label = base['label']
        self.label_mask = base['mask']
        self.index = index
        self.window = window
        self.step = step
        self.metric_len = metric_len
        self.span = (window - 1) * step

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        exp_id, t0 = self.index[i]
        idx = np.arange(t0, t0 + self.span + 1, self.step)
        last = idx[-1]
        return {
            'data_node': np.ascontiguousarray(self.metric[idx][:, :, :self.metric_len], dtype=np.float32),
            'data_log': np.ascontiguousarray(self.log[idx], dtype=np.float32),
            'data_edge': np.ascontiguousarray(self.trace[idx], dtype=np.float32),
            'groundtruth_cls': self.label_mask[last].astype(np.float32),
            'groundtruth_real': self.label[last].astype(np.float32),
            'exp_id': int(exp_id),
        }


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
        self.max_timesteps = args.get('max_timesteps', 0)
        self.set, self.dataset, self.trace_type = {}, [], []

        bounds_path = os.path.join(self.rawdata_path, 'bounds.pkl')
        if not os.path.exists(bounds_path):
            raise FileNotFoundError(f"{bounds_path} not found; run util/TT/pre_TT.py first")
        self.bounds = pickle.load(open(bounds_path, 'rb'))
        self.trace_type = pickle.load(open(os.path.join(self.rawdata_path, 'stats_vocab.pkl'), 'rb'))

        self.set = self._load_or_build_base()
        self.graph = read_graph(data_dir=self.rawdata_path)
        self._build_dataset()
        self._autofill_dims()
        self._build_split()

    def _base_file(self):
        return os.path.join(self.dataset_path, 'base.pkl')

    def _load_or_build_base(self):
        base_file = self._base_file()
        T_total = self.bounds[-1]['end']
        if os.path.exists(base_file):
            try:
                logging.info("Loading cached TT base tensors from %s ...", base_file)
                with open(base_file, 'rb') as fh:
                    base = pickle.load(fh)
                assert base['metric'].ndim == 3 and base['metric'].shape[:2] == (T_total, self.num_node)
                assert base['log'].shape == (T_total, self.num_node, self.log_len), base['log'].shape
                assert base['trace'].shape == (T_total, self.num_node, self.num_node, len(self.trace_type))
                assert base['label'].shape == (T_total, self.num_node, 2)
                assert base['mask'].shape == (T_total, self.num_node, 3)
                return base
            except (pickle.UnpicklingError, EOFError, AttributeError, ValueError, AssertionError) as exc:
                logging.warning("Failed to load cached TT base from %s: %s. Rebuilding.", base_file, exc)
        base = self._build_base(T_total)
        os.makedirs(self.dataset_path, exist_ok=True)
        logging.info("Saving TT base tensors to %s ...", base_file)
        with open(base_file, 'wb') as fh:
            pickle.dump(base, fh)
        return base

    def _build_base(self, T_total):
        if not os.path.exists(self.rawdata_path):
            raise FileNotFoundError(f"Find no TT preprocessed data at {self.rawdata_path}")
        logging.info("LOADing TT data ...")
        num_stats = len(self.trace_type)

        first_metric = np.load(os.path.join(self.rawdata_path, self.bounds[0]['name'], 'metric.npy'))
        metric_width = first_metric.shape[2]
        del first_metric
        metric_full = np.zeros((T_total, self.num_node, metric_width), dtype=np.float32)
        log_full = np.zeros((T_total, self.num_node, self.log_len), dtype=np.float32)
        trace_full = np.zeros((T_total, self.num_node, self.num_node, num_stats), dtype=np.float32)
        label_flat = np.zeros((T_total, self.num_node), dtype=np.int64)

        log_rows, trace_rows = 0, 0
        for bound in self.bounds:
            exp_dir = os.path.join(self.rawdata_path, bound['name'])
            seg = slice(bound['start'], bound['end'])
            T_exp = bound['end'] - bound['start']

            metric = np.load(os.path.join(exp_dir, 'metric.npy'))
            assert metric.shape == (T_exp, self.num_node, metric_width), metric.shape
            metric_full[seg] = metric
            del metric

            label = np.load(os.path.join(exp_dir, 'label.npy')).astype(np.int64)
            assert label.shape == (T_exp, self.num_node), label.shape
            label_flat[seg] = label
            del label

            log_events = pickle.load(open(os.path.join(exp_dir, 'log_events.pkl'), 'rb'))
            if log_events:
                ev = np.asarray(log_events)
                tid = ev[:, 0].astype(np.int64) - 1
                svc = ev[:, 1].astype(np.int64)
                rel = ev[:, 2].astype(np.int64)
                valid = (tid >= 0) & (tid < self.log_len) & (rel >= 0) & (rel < T_exp)
                np.add.at(log_full, (rel[valid] + bound['start'], svc[valid], tid[valid]), 1.0)
                log_rows += int(valid.sum())
            del log_events

            trace_events = pickle.load(open(os.path.join(exp_dir, 'trace_events.pkl'), 'rb'))
            if trace_events:
                ev = np.asarray(trace_events)
                src = ev[:, 0].astype(np.int64)
                dst = ev[:, 1].astype(np.int64)
                k = ev[:, 2].astype(np.int64)
                rel = ev[:, 3].astype(np.int64)
                dur = ev[:, 4].astype(np.float64)
                valid = (k >= 0) & (k < num_stats) & (rel >= 0) & (rel < T_exp)
                trace_a = np.zeros((self.num_node, self.num_node, num_stats, T_exp), dtype=np.float64)
                np.add.at(trace_a, (src[valid], dst[valid], k[valid], rel[valid]), dur[valid])
                trace_a = np.log1p(trace_a)
                trace_a = trace_a / (trace_a.mean() * 10 + 1e-6)
                trace_full[seg] = trace_a.transpose(3, 0, 1, 2).astype(np.float32)
                trace_rows += int(valid.sum())
            del trace_events
            logging.info("loaded experiment %s (%d..%d)", bound['name'], bound['start'], bound['end'])

        sec_max = log_full.max(axis=1)
        max_record = sec_max.max(axis=0)
        present = log_full.sum(axis=(1, 2)) > 0
        min_record = sec_max.min(axis=0) if present.all() else np.zeros(self.log_len)
        dis = max_record - min_record + 1e-6
        log_full = np.nan_to_num((log_full - min_record) / dis).astype(np.float32)
        logging.info("Log data: %d events, present seconds %d/%d", log_rows, int(present.sum()), T_total)
        logging.info("Trace data: %d events, stats vocab %d", trace_rows, num_stats)

        base = {
            'metric': metric_full,
            'log': log_full,
            'trace': trace_full,
            'label': np.eye(2, dtype=np.float32)[label_flat],
            'mask': np.eye(3, dtype=np.float32)[label_flat],
        }
        nbytes = sum(tensor.nbytes for tensor in base.values())
        logging.info(
            "TT base tensors: metric %s, log %s, trace %s, label %s, mask %s; total %.2f GB",
            base['metric'].shape, base['log'].shape, base['trace'].shape,
            base['label'].shape, base['mask'].shape, nbytes / 2**30)
        return base

    def _build_dataset(self):
        span = (self.window - 1) * self.step
        index = []
        for exp_id, bound in enumerate(self.bounds):
            if bound['end'] - bound['start'] < self.window:
                logging.info(f"experiment {bound['name']} too short, skipping")
                continue
            for t0 in range(bound['start'], bound['end'] - span, self.step):
                index.append((exp_id, t0))
        if self.max_timesteps > 0 and len(index) > self.max_timesteps:
            logging.info(f"Limiting to first {self.max_timesteps} windows (out of {len(index)})")
            index = index[:self.max_timesteps]
        self.window_index = index
        self.dataset = WindowDataset(self.set, index, self.window, self.step, self.metric_len)

        abnormal_sec = self.set['label'][:, :, 1].sum(axis=1) > 0
        last_secs = np.array([t0 + span for _, t0 in index])
        per_exp = {}
        for exp_id, _ in index:
            per_exp[exp_id] = per_exp.get(exp_id, 0) + 1
        logging.info(
            "TT lazy windows: total=%d abnormal=%d (%.2f%%), per-experiment %s",
            len(index), int(abnormal_sec[last_secs].sum()),
            abnormal_sec[last_secs].mean() * 100, per_exp)

    def _autofill_dims(self):
        num_stats = len(self.trace_type)
        if self.args.get('raw_edge', 0) == 0 or self.args['raw_edge'] > num_stats:
            self.args['raw_edge'] = num_stats
            logging.info(f"Auto-detected raw_edge = {num_stats}")

    def _build_split(self):
        fault_ids = [i for i, bound in enumerate(self.bounds) if bound['has_fault']]
        if not fault_ids:
            raise ValueError("no fault experiments found in bounds.pkl")
        test_experiment = int(self.args.get('test_experiment', -1))
        try:
            held_out = fault_ids[test_experiment]
        except IndexError:
            raise ValueError(
                f"test_experiment={test_experiment} out of range for {len(fault_ids)} fault experiments")

        exp_ids = np.array([exp_id for exp_id, _ in self.window_index])
        self.test_indices = np.flatnonzero(exp_ids == held_out).tolist()
        self.train_indices = np.flatnonzero(exp_ids != held_out).tolist()

        abnormal_sec = self.set['label'][:, :, 1].sum(axis=1) > 0
        span = (self.window - 1) * self.step

        def abnormal_ratio(indices):
            if not indices:
                return 0.0
            hits = sum(1 for i in indices if abnormal_sec[self.window_index[i][1] + span])
            return hits / len(indices)

        logging.info(
            "TT split: held-out fault experiment %d (%s) | train=%d (abnormal %.2f%%) test=%d (abnormal %.2f%%)",
            held_out, self.bounds[held_out]['name'],
            len(self.train_indices), abnormal_ratio(self.train_indices) * 100,
            len(self.test_indices), abnormal_ratio(self.test_indices) * 100,
        )
