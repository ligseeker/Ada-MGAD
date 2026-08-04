# Ada-MGAD RCA Extension

> 开发进度与实验日志见 [RCA_DEV_PLAN.md](RCA_DEV_PLAN.md)（接手必读；每完成一步需同步更新该文档）。

Root cause analysis (RCA) research context built on top of the Ada-MGAD anomaly detection model, evaluated on the GAIA (MicroSS) dataset.

## Language

**Node (节点)**:
One of the 10 service-instance entities in the GAIA graph: dbservice1/2, logservice1/2, mobservice1/2, redisservice1/2, webservice1/2. The graph is NOT expanded beyond these 10.
_Avoid_: service, instance, pod (these conflate service-level and instance-level granularities)

**Observation window (观测窗口)**:
A length-10 sequence of 30s timesteps over the graph — the unit of model input, same as Ada-MGAD detection.

**Anomalous window (异常窗口)**:
An observation window confirmed to contain an anomaly; the unit at which RCA ranking is invoked.

**Root-cause score (根因分数)**:
A per-node scalar output by the model for a given anomalous window; the 10 scores induce a full candidate ranking.

**Root-cause node (根因节点)**:
The node on which the fault-injection event was injected, per `label_events.csv` / `run_table`. Ground truth for ranking.
_Avoid_: affected node, anomalous node (a node may be anomalous due to propagation without being the root cause)

**Single-root window / Multi-root window**:
An anomalous window whose time span overlaps exactly one / more than one fault-injection event. Main experiments use single-root windows only; multi-root windows are excluded or form a separate test set.

**RCA fault set (RCA 故障集)**:
The four fault-injection types used as RCA ground truth: login failure, memory_anomalies, file moving program, access permission denied exception. Excluded: cpu_anomalies (3s duration is below the 30s observability floor; only 12 events), and the non-fault labels normal / normal memory freed label / unknown / error_event.

**RCA sample (RCA 样本)**:
One evaluation sample = one observation window whose last timestep (aligned with the detection output `rec[:, -1]`) falls inside a fault-injection event's floor-aligned 30s windows; the label is the injected node. Pure window-level evaluation (no event-level aggregation table).

**Propagation-aware localization (传播感知定位)**:
The chosen RCA method route. A node's root-cause score fuses three evidence sources: (i) its own anomaly evidence (cls + reconstruction scores from the frozen detector), (ii) propagation evidence on the learned dynamic graph with directed trace prior (root cause is anomalous earlier/stronger than its downstream neighbors), (iii) temporal onset evidence (root cause fires first within the 10-step window). Each evidence source is an ablation row. Two-stage implementation: the detector is frozen, only the localization module is trained. Plain detection-score ranking (no new parameters) is the mandatory baseline row; a ranking-head-only variant (evidence i only) is the degenerate ablation.

**Code organization (代码组织)**:
All RCA work happens on the `rca` branch of the Ada-MGAD repo (`main` stays as-published). Baseline implementations live INSIDE the repo under `baselines/` (user's choice over external directories). `label_rca.csv` is generated into the external `GAIA-pre/` data dir alongside existing labels. Experiments run 5 seeds, reported as mean±std.

**Baseline set (Baseline 阵容)**:
Main-table comparison rows: Random, historical root-cause frequency prior, three single-modality heuristic rankers (metric deviation / log error count / trace anomaly degree), Ada-MGAD detection-score ranking (zero-cost), DiagFusion (already GAIA-adapted), Eadro (requires writing a GAIA data adapter). All evaluated under the same oracle, macro-per-type protocol.

**Evaluation scope (评估范围)**:
RCA is evaluated on GAIA ONLY. MSDS is excluded from RCA: its real fault injections (31 Rally container restarts) all target the same node, making node-level ranking degenerate, and the `groundtruth_*`/`fault_type_*` files in MSDS-pre are heuristic artifacts misaligned with the real injection records. MSDS remains a detection-only dataset.

**RCA label file (RCA 标签文件)**:
A NEW label file (`label_rca.csv`: timestep × root-cause node id × fault type × multi-root flag) built from `label_events.csv` with the 4-type fault set. The existing detection labels (`label.csv`) are FROZEN exactly as published (their ~1.4% label noise — normal-memory-freed / cpu / error_event / unknown windows — stays, documented as a limitation), so the frozen-detector two-stage story and the published detection numbers remain intact.

**Label budget (标签预算)**:
Localization training uses the same 50% semi-supervised labeled-window partition as Ada-MGAD detection; ranking loss is computed only on anomalous windows. Main table at 50%; appendix reports a label-efficiency curve over 10/25/50/100% budgets.

**Oracle evaluation (Oracle 评估)**:
RCA is evaluated on ALL windows that have a labeled root cause, regardless of detector output; detection quality never gates RCA evaluation.
_Avoid_: triggered / end-to-end evaluation (detector-gated) — at most a qualitative system-level discussion, never the main protocol

**RCA metrics**:
Per-fault-type HR@1/3/5 and MRR, macro-averaged across the four types as the headline number (neutralizes the 90% login-failure skew); NDCG@3/5 also reported. Fault-type classification is NOT part of the main task.
