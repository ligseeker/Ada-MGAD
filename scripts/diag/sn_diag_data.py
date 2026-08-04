"""Data-level diagnostics for the SN dataset adaptation (no training).

Checks:
1. Label-feature sanity: does the faulted service's metric actually change
   inside each labeled fault window (e.g. CPU rise for cpu_load)?
2. Trace clock analysis: epoch base of spans.json vs metric epoch, and how
   the MSTGAD adapter's offset correction behaves.
3. Label density / class balance statistics.

Run from the MSTGAD repo root:
    python3 <workspace>/scripts/diag/sn_diag_data.py
"""
import json
import os
import glob
import numpy as np
import pandas as pd

RAW = "/home/zhangll24/RCA_project/Eadro/SN_Dataset/SN Dataset/data"
SERVICES = [
    "social-graph-service", "compose-post-service", "post-storage-service",
    "user-timeline-service", "url-shorten-service", "user-service",
    "media-service", "text-service", "unique-id-service",
    "user-mention-service", "home-timeline-service", "nginx-web-server",
]
CONTAINER_MAP = {
    "socialnetwork-social-graph-service-1": "social-graph-service",
    "socialnetwork-compose-post-service-1": "compose-post-service",
    "socialnetwork-post-storage-service-1": "post-storage-service",
    "socialnetwork-user-timeline-service-1": "user-timeline-service",
    "socialnetwork-url-shorten-service-1": "url-shorten-service",
    "socialnetwork-user-service-1": "user-service",
    "socialnetwork-media-service-1": "media-service",
    "socialnetwork-text-service-1": "text-service",
    "socialnetwork-unique-id-service-1": "unique-id-service",
    "socialnetwork-user-mention-service-1": "user-mention-service",
    "socialnetwork-home-timeline-service-1": "home-timeline-service",
    "socialnetwork-nginx-web-server-1": "nginx-web-server",
}
CPU_COLS = ["cpu_usage_system", "cpu_usage_total", "cpu_usage_user"]
NET_COLS = ["rx_bytes", "tx_bytes"]


def load_metric(exp_dir):
    frames = []
    for svc in SERVICES:
        df = pd.read_csv(os.path.join(exp_dir, "metrics", f"{svc}.csv"))
        df = df.rename(columns={c: f"{svc}_{c}" for c in df.columns if c != "timestamp"})
        frames.append(df.set_index("timestamp"))
    m = pd.concat(frames, axis=1).sort_index()
    return m


def main():
    fault_files = sorted(glob.glob(os.path.join(RAW, "SN.fault-*.json")))
    for ff in fault_files:
        exp_name = "SN." + os.path.basename(ff).replace("SN.fault-", "").replace(".json", "")
        exp_dir = os.path.join(RAW, exp_name)
        if not os.path.isdir(exp_dir):
            print(f"[skip] no dir for {exp_name}")
            continue
        fault = json.load(open(ff))
        m = load_metric(exp_dir)
        t0 = int(m.index.min())
        print(f"\n===== {exp_name} | T={len(m)}s =====")

        # --- 1. label-feature sanity per fault window ---
        for f in fault["faults"]:
            svc = CONTAINER_MAP.get(f["name"], f["name"])
            ft, start, dur = f["fault"], int(f["start"]) - t0, int(f["duration"])
            if svc not in SERVICES or start < 0 or start >= len(m):
                print(f"  [?] {f['name']} -> {svc} out of range")
                continue
            win = slice(start, start + dur)
            base = slice(max(0, start - 300), start)
            cols = CPU_COLS if ft == "cpu_load" else NET_COLS
            for c in cols:
                col = f"{svc}_{c}"
                inside = m[col].iloc[win].mean()
                before = m[col].iloc[base].mean()
                ratio = inside / before if abs(before) > 1e-9 else float("inf")
                print(f"  {svc[:22]:22s} {ft:13s} {c:18s} in={inside:12.1f} before={before:12.1f} ratio={ratio:6.2f}")

        # --- 2. trace clock analysis ---
        spans_path = os.path.join(exp_dir, "spans.json")
        try:
            traces = json.load(open(spans_path))
        except Exception as e:
            print(f"  [trace] unreadable: {e}")
            continue
        starts = []
        n_spans = 0
        for tr in traces:
            for sp in tr["spans"]:
                starts.append(sp["startTime"] / 1e6)
                n_spans += 1
        starts = np.array(starts)
        trace_min, trace_max = starts.min(), starts.max()
        print(f"  [trace] spans={n_spans} trace_epoch=[{trace_min:.0f},{trace_max:.0f}] "
              f"metric_epoch=[{m.index.min()},{m.index.max()}] offset={trace_min - m.index.min():.0f}s")
        # adapter correction: ts_offset = trace_start - fault_json['start']
        adapter_offset = trace_min - fault["start"]
        corrected = starts - adapter_offset - t0  # relative to metric timeline
        in_range = ((corrected >= 0) & (corrected < len(m))).mean()
        print(f"  [trace] adapter_offset={adapter_offset:.0f}s -> {in_range:.2%} spans inside metric timeline")
        # per-fault-window span coverage after correction
        for f in fault["faults"][:2]:
            s, d = int(f["start"]) - t0, int(f["duration"])
            cnt = ((corrected >= s) & (corrected < s + d)).sum()
            print(f"    spans inside fault window ({f['fault']}): {cnt}")


if __name__ == "__main__":
    main()
