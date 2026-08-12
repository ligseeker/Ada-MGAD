import os
import pickle

import numpy as np

PRE = './data/TT-pre'
bounds = pickle.load(open(os.path.join(PRE, 'bounds.pkl'), 'rb'))
total_windows, empty_windows = 0, 0
for b in bounds:
    if not b['has_fault']:
        continue
    d = os.path.join(PRE, b['name'])
    label = np.load(os.path.join(d, 'label.npy'))
    ev = pickle.load(open(os.path.join(d, 'trace_events.pkl'), 'rb'))
    ends = np.asarray([e[3] for e in ev], dtype=np.int64) if ev else np.zeros(0, dtype=np.int64)
    active = label.sum(axis=1) > 0
    windows, start = [], None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            windows.append((start, i)); start = None
    if start is not None:
        windows.append((start, len(active)))
    counts = [int(((ends >= s) & (ends < e)).sum()) for s, e in windows]
    total_windows += len(windows)
    empty = [w for w, c in zip(windows, counts) if c == 0]
    empty_windows += len(empty)
    print('%s windows=%d min=%d max=%d empty=%s' %
          (b['name'], len(windows), min(counts), max(counts), empty if empty else 'none'))
print('TOTAL fault windows=%d empty=%d' % (total_windows, empty_windows))
