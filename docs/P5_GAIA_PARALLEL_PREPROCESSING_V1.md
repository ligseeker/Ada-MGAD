# P5-I1 GAIA Parallel Preprocessing V1

## Scope and status

This change accelerates the existing P5-I1 raw-data adapters without changing
Ada-MGAD, Ada-RCA, Protocol T, event labels, temporal bins, feature definitions,
or model inputs.  It has been validated on deterministic small fixtures.  No
formal full-GAIA run or performance claim is made by this document.

The target execution allocation is at most 30 CPU cores and 120 GB RAM.  The
frozen default is deliberately lower than the CPU ceiling because the 30 GB raw
source is I/O-heavy:

| Stage | Default processes | Task boundary |
|---|---:|---|
| Ada-MGAD metric | 8 | service and logical metric feature |
| Ada-MGAD log | 8, capped by 10 tasks | service |
| Ada-MGAD span index | 8, capped by 10 tasks | service |
| Ada-MGAD trace | 8, capped by 10 tasks | child service |
| Ada-RCA raw metric | 8 | service and logical metric indicator |
| Ada-RCA raw log | 8, capped by 10 tasks | service |
| Ada-RCA raw trace | 8, capped by source-file count | source file |
| Ada-RCA 68D materialization | 24 | contiguous 128-case shard |

`spawn` is used instead of `fork` because the Ada-MGAD command imports PyTorch.
Each process is restricted to one native BLAS/OpenMP/NumExpr thread, preventing
process-by-library thread multiplication.  Pools run sequentially by modality;
the AD and RCA raw index jobs must not be launched concurrently against the same
shared filesystem.

## Determinism and failure behavior

- Task lists use canonical service, feature, source-file, and case order.
- Process completion order never determines manifest or tensor order.
- Workers never mutate a shared NumPy array or shared manifest.
- `.npy` and JSON files are written to same-filesystem temporary files and then
  atomically renamed.
- The RCA raw index writes into a generation-specific directory and publishes
  `index_manifest.json` only after every worker succeeds.
- RCA case workers publish independent checksum-bound shards.  The parent merges
  them by original case range and atomically replaces `z2_features.npy` only after
  all shards succeed.
- AD metric, AD span-index, and RCA feature shards are resumable.  AD log/trace
  scans and the RCA raw-index scan are deterministic but currently restart their
  respective modality if the command is rerun.
- Feature workers receive only `case_id` and `start_ms`; label-bearing columns are
  not read into the materialization frame.  The full case-registry hash remains
  provenance only, while shard identity uses prediction-visible columns.

Small-fixture regression tests cover serial-versus-parallel equality for AD
metrics, logs, traces, graph construction, the RCA raw index, and 68D case
features.  They also cover stable case order, cache resume, the 30-CPU bound,
worker failure without final raw-index publication, and failure without replacing
an existing final feature tensor.

## Recommended full preprocessing command

Run from a clean checkout on the 30-CPU, 120-GB CPU container.  Prefer a local
SSD/NVMe mirror that has exactly the audited relative paths and byte sizes.  Both
raw adapters verify that layout before parsing.

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e
git branch --show-current
git rev-parse HEAD
git status --short
ulimit -n

/usr/bin/time -v python scripts/p5/run_i1_pipeline.py preprocess \
  --raw-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS \
  --raw-workers 8 \
  --feature-workers 24 \
  --chunk-rows 150000 \
  --case-chunk-size 128 \
  --start-method spawn \
  --gpu false
```

If the source is on fast local NVMe and observed I/O utilization has headroom,
one controlled retry may use `--raw-workers 12`.  Do not start at 30 raw workers.
If memory pressure occurs, reduce `--chunk-rows` to `75000`; if storage latency or
throughput worsens, reduce `--raw-workers` to 4 or 6.  These are execution controls,
not scientific protocol choices.

After preprocessing, use the existing GPU environment for the remaining stages:

```bash
python scripts/p5/run_i1_pipeline.py train-evaluate --gpu true
PYTHONDONTWRITEBYTECODE=1 pytest -q
python scripts/p5/finalize_i1_manifest.py --pytest-result 'FULL_SUITE_PASS'
```

The preprocessing manifests record requested/effective workers, task counts,
chunk size, deterministic ordering policy, per-phase wall time, config hash, and
source bindings.  Preserve `/usr/bin/time -v` output as external execution evidence
if wall time and peak RSS are to be reported.

## Known risks and operational limits

The primary limit is shared-filesystem contention, not CPU availability.  More
workers can make the job slower.  A 150,000-row chunk bounds per-process Pandas
memory, but raw string columns can still expand substantially in RAM.  Do not run
two full preprocess commands into the same output root simultaneously.  Stale
generation directories and cache shards are intentionally retained for recovery;
they may be removed only after their active manifest bindings and checksums have
been inspected.

Parallelization does not merge injections, alter timestamps, relax the label
firewall, or use Test results.  Poor downstream performance must not be addressed
by changing these execution settings or the frozen scientific protocol.
