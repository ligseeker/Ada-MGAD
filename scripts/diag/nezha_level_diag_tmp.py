import pickle
import time

import numpy as np
import pandas as pd

t = time.time()
label = pickle.load(open('data/Nezha-pre/label.pkl', 'rb'))
T_total = label.shape[0]
abnormal = label.sum(axis=1) > 0
print(f"T_total={T_total}, abnormal minutes={int(abnormal.sum())}")

log = pd.read_csv('data/Nezha-pre/log.csv', usecols=['@timestamp', 'level'])
print(f"log rows={len(log)}  (read {time.time() - t:.1f}s)")
log['level'] = log['level'].fillna('').astype(str)
ts = log['@timestamp'].values
valid = (ts >= 0) & (ts < T_total)
for lvl in ['ERROR', 'WARNING', 'INFO', '']:
    m = (log['level'].values == lvl) & valid
    n = int(m.sum())
    in_fault = int(abnormal[np.clip(ts[m], 0, T_total - 1)].sum()) if n else 0
    tag = lvl if lvl else '<none>'
    print(f"level {tag:8s}: {n:7d} rows, {in_fault:6d} ({100 * in_fault / max(1, n):.1f}%) inside fault buckets")
print(f"total {time.time() - t:.1f}s")
