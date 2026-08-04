"""Trivial-detector upper-bound check for SN.

For each experiment, flag a (t, node) point abnormal if any of its 7 KPIs
deviates from the per-node robust baseline (median/MAD) by |z| > tau.
Point-level F1 vs the fault-JSON labels. No learning involved: this
quantifies feature-label alignment and signal strength.
"""
import json
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

RAW = "/home/zhangll24/RCA_project/Eadro/SN_Dataset/SN Dataset/data"
SERVICES = [
    "social-graph-service", "compose-post-service", "post-storage-service",
    "user-timeline-service", "url-shorten-service", "user-service",
    "media-service", "text-service", "unique-id-service",
    "user-mention-service", "home-timeline-service", "nginx-web-server",
]
KPI = ["cpu_usage_system", "cpu_usage_total", "cpu_usage_user",
       "memory_usage", "memory_working_set", "rx_bytes", "tx_bytes"]
CONTAINER_MAP = {f"socialnetwork-{s}-1": s for s in SERVICES}
TAUS = [3, 4, 6]


def run():
    y_all, p_all = {t: [] for t in TAUS}, {t: [] for t in TAUS}
    for ff in sorted(glob.glob(os.path.join(RAW, "SN.fault-*.json"))):
        exp_name = "SN." + os.path.basename(ff).replace("SN.fault-", "").replace(".json", "")
        exp_dir = os.path.join(RAW, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        fault = json.load(open(ff))
        frames = []
        for svc in SERVICES:
            df = pd.read_csv(os.path.join(exp_dir, "metrics", f"{svc}.csv"))
            df = df.rename(columns={c: f"{svc}_{c}" for c in df.columns if c != "timestamp"})
            frames.append(df.set_index("timestamp"))
        m = pd.concat(frames, axis=1).sort_index()
        T = len(m)
        t0 = int(m.index.min())
        label = np.zeros((T, len(SERVICES)), dtype=int)
        for f in fault["faults"]:
            svc = CONTAINER_MAP.get(f["name"])
            if svc is None:
                continue
            s = int(f["start"]) - t0
            label[max(0, s):min(T, s + int(f["duration"])), SERVICES.index(svc)] = 1

        vals = m[[f"{s}_{k}" for s in SERVICES for k in KPI]].values  # (T, N*K)
        vals = vals.reshape(T, len(SERVICES), len(KPI)).astype(float)
        med = np.median(vals, axis=0, keepdims=True)
        mad = np.median(np.abs(vals - med), axis=0, keepdims=True)
        mad = np.where(mad < 1e-9, np.nan, mad)
        z = np.abs((vals - med) / (1.4826 * mad))
        z = np.nan_to_num(z, nan=0.0)
        score = z.max(axis=-1)  # (T, N)

        print(f"\n{exp_name} T={T} abnormal_pts={label.sum()} ({label.mean():.3%})")
        for t in TAUS:
            pred = (score > t).astype(int)
            f1 = f1_score(label.ravel(), pred.ravel(), zero_division=0)
            pr = precision_score(label.ravel(), pred.ravel(), zero_division=0)
            rc = recall_score(label.ravel(), pred.ravel(), zero_division=0)
            print(f"  tau={t}: P={pr:.3f} R={rc:.3f} F1={f1:.3f}")
            y_all[t].append(label.ravel())
            p_all[t].append(pred.ravel())

    print("\n===== pooled =====")
    for t in TAUS:
        y = np.concatenate(y_all[t]); p = np.concatenate(p_all[t])
        print(f"tau={t}: P={precision_score(y, p, zero_division=0):.3f} "
              f"R={recall_score(y, p, zero_division=0):.3f} F1={f1_score(y, p, zero_division=0):.3f}")


if __name__ == "__main__":
    run()
