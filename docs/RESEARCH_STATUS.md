# 研究状态与决策日志

> 状态版本：`research_state_v12`
> 最近更新：2026-08-20
> 状态所有者：本文件；完成 Gate 后必须同步更新

## 1. 当前状态摘要

| 项目 | 状态 | 证据 |
|---|---|---|
| Ada-MGAD 异常检测代码 | 已存在 | 当前 `src/`、`util/`、`README.md` |
| 异常检测论文/完整实验产物 | 当前工作区不可用 | 仅参考对话提及附件与结果 |
| 旧 Ada-MGAD-dependent RCA Pilot | 对话报告已完成，当前 clone 不可核验 | [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) |
| Standalone RCA 研究定位 | 已冻结 | [RCA_RESEARCH_DESIGN.md](RCA_RESEARCH_DESIGN.md) V0.2 |
| P1 Benchmark Protocol | 已完成并冻结 | [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) V0.5 |
| `rca-standalone` 分支 | 已建立；当前工作分支为 `claudecode` | Git 当前分支与远端 tracking branch |
| `RCACase` / adapters / evaluator | 已实现首版 | `src/data/`、`src/evaluation/`、`tests/` |
| Label-separated manifests | GAIA 16,200 + RE2-OB 90 已按修正后的 log 时间引用重生并校验 | `artifacts/p1/manifests/` |
| Telemetry diagnostics scanner | 约 38 GB raw telemetry 全量扫描完成 | `artifacts/p1/telemetry_diagnostics.json`、[TELEMETRY_DIAGNOSTICS.md](TELEMETRY_DIAGNOSTICS.md) |
| Actual 5-fold assignments | GAIA grouped-stratified + RE2-OB singleton stratified 已生成并校验 | `artifacts/p1/splits/`、[SPLIT_DIAGNOSTICS.md](SPLIT_DIAGNOSTICS.md) |
| P1 sanity baselines | B0/B1/B2 已保存完整 rankings、macro metrics 与 run manifests | `artifacts/p1/baselines/`、[BASELINE_RESULTS.md](BASELINE_RESULTS.md) |
| GAIA main cohort | 13,470 anchor-unique-root cases；2,730 multi-root sensitivity | `artifacts/p1/inclusion/gaia/`、[GAIA_INCLUSION_AUDIT.md](GAIA_INCLUSION_AUDIT.md) |
| RE2-OB 数据 | 90/90 可读；归档与 360 个实际输入文件已 content-pinned | `artifacts/p1/source_snapshots/re2ob/`、[P1_REPRODUCIBILITY_AUDIT.md](P1_REPRODUCIBILITY_AUDIT.md) |
| P1 G1–G8 | 8/8 completed；最终自动门禁通过 | `artifacts/p1/gate_audit.json`、见下方 Gate 表 |
| P2 Experiment Plan | V0.2；metric onset 由 coverage 收缩为 60/120 s | [P2_EXPERIMENT_PLAN.md](P2_EXPERIMENT_PLAN.md) |
| P2-G1 metric | completed；170 维全量 bundle 与独立 coverage audit | [P2_METRIC_FEATURES.md](P2_METRIC_FEATURES.md)、`artifacts/p2/features/` |
| P2-G2 C0-M | completed；nested OOF、泄漏审计、确定性复跑 | [P2_C0_M_RESULTS.md](P2_C0_M_RESULTS.md)、`artifacts/p2/runs/c0_m/` |
| P2-G1 logs/traces | completed；L0/T0 全量流式 extraction 与 coverage audit 已完成 | [P2_G3_MODALITY_RESULTS.md](P2_G3_MODALITY_RESULTS.md) §2、`artifacts/p2/event_features/`、`artifacts/p2/event_feature_{summary,audit}.json` |
| P2-G3 C0-L / C0-T / C1-I | completed；三个 run + 独立审计 + 配对 bootstrap | [P2_G3_MODALITY_RESULTS.md](P2_G3_MODALITY_RESULTS.md)、`artifacts/p2/runs/{c0_l,c0_t,c1_i}/`、`artifacts/p2/linear_ablation_{summary,audit}.json` |
| C1-I 门禁状态 | `exploratory_signal=true`；`claim_ready=false` | `artifacts/p2/c1_i_bootstrap.json`（`p2_c1_i_paired_bootstrap_v1`） |
| P2-G4 M1-S | completed；run + 独立审计 + 配对 bootstrap 均已完成 | [P2_G4_STAGE_RESULTS.md](P2_G4_STAGE_RESULTS.md)、`artifacts/p2/runs/m1_s/`、`artifacts/p2/m1_s_{summary,audit,bootstrap}.json` |
| M1-S 门禁状态 | `exploratory_signal=false`；`claim_ready=false`；**`p2_g4_decision="no-go"`** | `artifacts/p2/m1_s_bootstrap.json`（`p2_m1_s_paired_bootstrap_v1`） |
| H1（event-stage 表征优于 naive early fusion） | **未获支持**（冻结门禁下） | 同上；GAIA 双端点为正但 RE2-OB 主端点符号为负 |
| RE2-TT protocol extension | V0.2；E1–E5 audit-first 阶段全部执行完毕 | [RE2TT_EXTENSION_PROTOCOL.md](RE2TT_EXTENSION_PROTOCOL.md)、`artifacts/ext/re2tt/` |
| EXT-RE2TT E1 manifests / E2 telemetry / E3 splits / E5 audit | 4/4 通过；RE2-OB manifest 字节级回归 `passed` | `artifacts/ext/re2tt/{manifests,telemetry_diagnostics.json,splits,gate_audit.json}` |
| EXT-RE2TT E4 headroom gate | **`decision="fail"`，`failed_gates=["H-1"]`** | `artifacts/ext/re2tt/headroom_gate.json`（`ext_re2tt_headroom_gate_v1`）；B2 主端点 0.904444 > 0.90 上限 |
| EXT-RE2TT E6 / E7（feature extraction、H1 replication） | **`blocked_stages=["E6","E7"]`；未执行** | 预登记规则「E1–E5 全部通过前不进入 E6/E7」；见 [RE2TT_EXTENSION_PROTOCOL.md](RE2TT_EXTENSION_PROTOCOL.md) §9.6 |

**当前研究阶段：P2 — Multimodal RCA Representation and Attribution；并行的 RE2-TT protocol extension 已在 audit 阶段停下，等待用户决策。**

## 2. 仓库快照

```text
branch:       claudecode
worktree:     .git 指向 /home/zhangll24/RCA_project/Ada-MGAD/.git/worktrees/Ada-MGAD-rca-claudecode
commit:       c2af48f2be4066de7363d7e5f0871052e8564301
subject:      update P2-G4
PR target:    main
working tree: 不干净。已修改：docs/{EXPERIMENT_LOG,RESEARCH_STATUS}.md、
              src/data/{__init__,rcaeval,telemetry_diagnostics}.py、
              scripts/run_p1_metric_change.py（后四项为 RE2-TT extension 的
              参数化改造，已用字节级回归证明不改变 P1/RE2-OB 既有输出）；
              未跟踪新增：docs/RE2TT_EXTENSION_PROTOCOL.md、
              scripts/{prepare,diagnose,run,audit}_ext_re2tt_*.py、
              tests/test_ext_re2tt_extension.py、tests/test_rcaeval_profiles.py
```

**引用本文件中任何 EXT-RE2TT 数字时必须注意**：上述 extension driver 尚未提交，
所以 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) §23 记录的是逐文件 SHA-256 而不是
单一 commit hash；提交后应回填 commit。

当前事实：

- 当前检出为 git worktree，工作分支 `claudecode`，PR 目标为 `main`；
- `docs/` 现有 19 个文件，`docs/README.md` 为唯一入口；
- 旧 Ada-MGAD 的 GAIA/MSDS 异常检测实现保持不动；
- `src/data/schema.py` 已物理拆分 prediction input 与 label；
- `src/evaluation/` 已实现完整 ranking 校验与 AC@1/3/5、Avg@5、MRR；
- `src/data/gaia.py` 已解析 16,200 条候选注入并保留重叠审计；
- `src/data/rcaeval.py` 已在本地 RE2-OB 达到 90/90 可读；
- `src/data/manifest.py` 已输出分离的 input/label/source bundle 与文件 SHA-256；
- GAIA 原始注入区间有 4,037 个重叠 case；±300 s context-aware manifest 为 322 组、最大组 471；
- 原始资产体积约为 GAIA 30.3 GB、RE2-OB 8.1 GB，scanner 采用 chunked streaming；
- GAIA business-log 的真实毫秒时间位于 `message` 前缀，adapter 与 manifest 已修正；
- RE2-OB 的 `cluster_info.json` 是日志模板元数据；pod-node 表是部署映射，不是服务调用图；
- 全量 scanner 共读取 349,120,558 行，GAIA/RE2-OB 三模态均无未完成文件；
- G8 已完成；主窗口已冻结为 ±300 s，context-aware groups 已写入 manifest；
- GAIA 主 split 已冻结为 grouped-stratified 5-fold，RE2-OB 为 singleton
  stratified 5-fold，统一 seed `20260819`；
- Random/Frequency/Metric-change 已在两个数据集完成并字节级复现；
- P1 closeout 时 `tests/` 51/51 通过；P2 新增 metric/event feature、linear-ranker
  与 P2-G4 后处理 tests，再加 RE2-TT extension 的 12 个 + RCAEval profile tests 后
  当前全套 131/131 通过；
- GAIA 16,200 event inventory 已分为 13,470-case main 与 2,730-case
  multi-root sensitivity；
- RE2-OB 原始归档与 8,441,465,341 字节实际消费文件已逐字节固定，content
  identity 为 `cec0030da8b499914af7979283eb1a2cd4f2cad2430742c70130829cd28133c9`；
- 最终 `p1_gate_audit_v1` 已验证 9 组 baseline、全部 bindings、指标重算与
  Label Firewall，`raw_re2_source_bytes_verified=true`；
- G1–G8 与 reproducibility closeout 均 completed，P1 已关闭；
- P2 metric 全量表征为 GAIA 134,700 rows、RE2 990 rows、每行 170 features；
- GAIA 30 s metric onset block 仅约 0.09% service rows 可观察，故调参候选只保留
  60/120 s；
- C0-M GAIA/RE2 root-macro Avg@5 为 0.5820/0.9756；相对 P1 B2 为
  −0.1266/+0.0422，不形成统一优越性结论；
- P2-G2 的 predictions/metrics/training-audit 核心文件重跑 SHA-256 一致；
- L0/T0 全量 extraction 已完成：GAIA log/trace 各 134,700 rows × 50/80
  （扫描 87,974,871 / 28,681,438 raw rows），RE2 各 990 rows × 50/80
  （15,053,223 / 34,461,235）；
- entity-observed 比例为 GAIA log/trace 0.999777、RE2 log 0.911111、
  RE2 trace 0.636364；
- RE2 log 三档 onset 的 content-complete 比例同为 0.794900，低于冻结的 0.80
  绝对阈值，故 M1-S 的 staged 通道排除 log，只用 metric + trace；
- C0-L / C0-T / C1-I root-macro Avg@5 为 GAIA 0.3567/0.3549/0.6072、
  RE2 0.4667/0.8156/0.9956；两数据集 best single modality 均为 C0-M；
- C1-I 相对 C0-M 主端点为 GAIA +0.0253 / RE2 +0.0200，次端点 AC@1 为
  GAIA −0.0911 / RE2 +0.0667；
- C1-I 配对 bootstrap 判定 `exploratory_signal=true`、`claim_ready=false`
  （GAIA 主端点 CI 下界 −0.003557 ≤ 0，且 GAIA AC@1 −0.0911 < −0.01）；
- GAIA 上所有 learned 方法的 root-macro 主端点仍低于未学习的 P1 B2
  （C1-I 相对 B2 为 −0.1013，M1-S 为 −0.0767，差距缩小但未反转）；
- M1-S 已完成：design 为 105 值 + 105 masks = 210 列，两数据集各 160 inner
  + 5 outer fits，root-macro Avg@5 为 GAIA 0.6319 / RE2 0.9933；
- M1-S 相对 C1-I 的主端点为 GAIA +0.024646 / RE2 **−0.002222**，次端点 AC@1 为
  GAIA +0.039918 / RE2 **−0.011111**；
- M1-S 配对 bootstrap 判定 `exploratory_signal=false`、`claim_ready=false`、
  `p2_g4_decision="no-go"`（GAIA 双端点 CI 下界均 > 0，但 RE2 主端点 CI 下界
  −0.007500 ≤ 0 且 RE2 AC@1 −0.011111 < −0.01）；
- RE2-OB 的全部退化溯源为单个 case `re2ob-c9c8f348d3f5974b`（fold_1，root
  `recommendationservice`，rank 1 → 2）；89/90 个 case 只发生尾部重排；
- M1-S runtime 为 GAIA 12,326.37 s / RE2 24.01 s；audit 与 bootstrap 二次运行
  逐字节复现；
- RE2-TT extension 的全部产物只写在 `artifacts/ext/re2tt/`，四个 driver 均在
  `main()` 首行调用 `_assert_isolated()` 拒绝写入 `artifacts/p1`、`artifacts/p2`；
- RE2-TT 为 90 cases / 排除 0 / 68 个候选服务 / 90 个 singleton groups；
  roots 为 `ts-{auth,order,route,train,travel}-service` 各 18 个，faults 为
  `cpu/delay/disk/loss/mem/socket` 各 15 个，完全平衡；
- 因该平衡设计，RE2-TT 的 overall、fault-type macro 与 root-service macro
  **三层报告数值完全相同**，无法暴露“整体好但某一组塌陷”的失败模式；
- RE2-TT source snapshot 固定了 1 个归档（`RE2-TT.zip`，2,801,345,134 B）与
  360 个实际消费文件（22,752,639,186 B），content identity 为
  `ad1396d0713fe21343a9db9233a60e51aa043026f8f39cd30fa1ba0fd5f15fc0`；
- RE2-TT 三模态 label-free 候选 presence ratio（±300 s）为 metric 1.000000、
  log 0.654739、trace 0.383333；metric 采样为精确 1000 ms、无重复无缺口；
- RE2-TT 5-fold 为 18/18/18/18/18，`max_root_service_total_variation`
  0.066667 < 0.10，case/group overlap 均为 0，复跑逐字节一致；
- RE2-TT baselines 的 root-macro Avg@5 为 B0 0.037778 / B1 0.566667 /
  B2 **0.904444**，B2 AC@1 为 0.822222；
- 预登记 headroom 门禁 H-1（B2 主端点 ≤ 0.90）**不通过**（超出 0.004444，
  恰为 2 个 Avg@5 case 量子），H-2/H-3 通过，故 `blocked_stages=["E6","E7"]`；
- B1 在 RE2-TT 上与 RE2-OB **逐位相同**（Avg@5 0.566667、AC@5 1.000000），已
  排除接线错误：这是 5 root × 18 case 平衡设计的结构性后果，说明 **RE2-TT 并未
  修复 5-root frequency prior 的退化**；
- B2 从 RE2-OB 的 0.933333 只降到 0.904444（−0.028889），说明 RE2 系列天花板
  不来自候选空间大小，而来自单点注入 + metric 全覆盖的数据构造方式；
- E5 root-conditioned coverage：metrics 90/90、traces 90/90、**logs 72/90**，
  缺口 18 例全部落在 root=`ts-train-service`，与该 root 完全共变（confound）。

## 3. ARS 工作流位置

本轮从既有 research state 接续，使用 `academic-research-suite` 的 `experiment-agent` 路由实施 P1；没有启动完整论文十阶段 pipeline，也没有声称通过论文完整性 Gate。

当前实质工作属于研究设计到实验规划的过渡：

```text
研究问题与路线收敛       completed
Standalone RCA Design    completed / frozen V0.2
P1 Benchmark Protocol    completed / frozen V0.5
P1 implementation        G1-G8 + reproducibility closeout completed
P2 representation        metric + L0/T0 full extraction completed
P2 controlled experiments P2-G2 completed / P2-G3 completed / P2-G4 completed (no-go)
RE2-TT extension        E1-E5 completed (E4 gate failed) / E6-E7 blocked
paper full drafting      not started in this workspace
```

## 4. P1 Gate 状态

| Gate | 内容 | 状态 | 证据路径 |
|---|---|---|---|
| G1 | GAIA event → RCACase | completed | 16,200 inventory 全部映射；13,470 main + 2,730 multi-root sensitivity；`artifacts/p1/inclusion/gaia/`、[GAIA_INCLUSION_AUDIT.md](GAIA_INCLUSION_AUDIT.md) |
| G2 | RE2-OB 90 cases → RCACase | completed | `src/data/rcaeval.py`、`artifacts/p1/manifests/re2ob/`、`artifacts/p1/source_snapshots/re2ob/`；90/90、排除 0、实际输入逐字节固定 |
| G3 | 两数据集完整 service ranking | completed | `artifacts/p1/baselines/{gaia,re2ob}/*/predictions.jsonl`；三种 baseline 均通过完整排列校验 |
| G4 | AC/Avg/MRR toy tests | completed | `src/evaluation/`、`tests/test_metrics.py`；最终审计又从 9 组完整 rankings 重算 metrics |
| G5 | 三个 sanity baselines | completed | `artifacts/p1/baseline_summary.json`、[BASELINE_RESULTS.md](BASELINE_RESULTS.md)；B0/B1/B2 两数据集均完成 |
| G6 | event/case split 无泄漏 | completed | `artifacts/p1/splits/`、`artifacts/p1/split_diagnostics.json`、`src/data/split.py`、`tests/test_split_integrity.py`；实际 assignment 与 source checksum binding 已校验 |
| G7 | Label Firewall | completed | 分离 manifests、input-only predictors、B1 train-fold-only ID audit、B2 within-case fit scope、prediction 敏感字段零命中与测试 |
| G8 | dataset diagnostic report | completed | `artifacts/p1/telemetry_diagnostics.json`、[TELEMETRY_DIAGNOSTICS.md](TELEMETRY_DIAGNOSTICS.md)；全量 349,120,558 行、未完成文件 0 |

G1–G8 已全部 `completed` 且具备代码/测试/产物路径。RCAEval 原始数据
content snapshot 与全产物一致性 closeout 也已完成；最终审计 SHA-256 为
`a02625a0a37136f0d763c8c7fa83d39b003c15eeac033310f6990ab55ce1bf45`。

## 5. 已冻结决策

1. 毕设包含独立的异常检测与 RCA 两项研究，最终在事件层集成。
2. Standalone RCA 不使用 Ada-MGAD latent representation。
3. 旧 `rca` 只作为 Pilot，不作为新方法开发基线。
4. 新 RCA 任务是 event-level、service-level、single-root ranking 主协议。
5. 主数据为 GAIA + RE2-OB；RE2-TT 为扩展。2026-08-20 用户已确认以**独立
   protocol extension** 的形式引入 RE2-TT：不修改 P1/P2 冻结协议、不删除 RE2-OB、
   不放宽 Go/No-Go 阈值，产物物理隔离在 `artifacts/ext/re2tt/`，且必须
   audit-first（E1–E5 全部通过才允许做 H1 replication）。见
   [RE2TT_EXTENSION_PROTOCOL.md](RE2TT_EXTENSION_PROTOCOL.md)。
6. 主实验为 Oracle RCA；Trigger Robustness 与 Ada-MGAD-triggered RCA 后置。
7. 候选集合由测试时可得信息构造，主实验包含全部可观测服务。
8. 先 case/event split，再拟合任何学习型 preprocessing。
9. P1 实现 Label Firewall，并只做 Random/Frequency/Metric-change。
10. H1/H2/H3 是可检验假设，不是预设创新点。
11. P1 不把 Event-stage 三段切分固化到数据协议，也不实现新神经网络。
12. 主上下文冻结为 `T_pre=T_post=300 s`，采用毫秒半开区间且不合成边界填充值。
13. 只有 P1 G1–G8 与 reproducibility closeout 完成后才进入 P2；该条件现已满足。
14. GAIA 主 split 为 322 个 context groups 上的 grouped-stratified 5-fold；
    RE2-OB 为 singleton stratified 5-fold；seed 均为 `20260819`。时间块只作
    GAIA temporal-shift sensitivity。
15. GAIA 主结果使用 13,470 个锚点唯一 root-service cases；2,730 个锚点
    multi-root cases 只作 concurrency sensitivity；完整 inventory 不删除。

## 6. 尚未决定的问题

这些问题必须由 P2 实验回答，不能在新对话中凭经验直接填值：

- RE2-OB 指标候选已稳定为 11 个应用服务；trace 静态 alias 已冻结，但缺失候选的
  mask/coverage 对模型的影响仍待实验；
- topology 的统一表示与缺失/动态策略；
- ~~logs/traces 的 30/60/120 s stage coverage~~ 已由 P2-G1 全量 extraction 解决：
  GAIA log/trace 三档均 supported，RE2 trace 三档均 supported，RE2 log 三档均
  unsupported（0.794900 < 0.80）；metric 仍冻结为 60/120 s；
- H2 的比较模块与 H3 的结构机制；
- RE2-TT 的 E4 门禁失败如何处置（接受 E6/E7 永久 blocked，还是在新的 protocol
  version bump 中重新定义 headroom 准入口径）——不得就地放宽 0.90 上限；
- 最终论文题目是否加入“根因分析”。

## 7. 下一步执行队列

### P1.1：schema + evaluator（completed）

- 建立 `RCACaseInput` / `RCACaseLabel`；
- 实现 schema invariants；
- 实现 AC@1/3/5、Avg@5、MRR；
- 用手工 toy cases 覆盖正确首位、Top-k 命中、缺失根因、重复/不完整 ranking；
- 建立 Label Firewall 的最小接口测试。

### P1.2：GAIA adapter（completed）

- 已复用并审计 `parse_anomaly_event`，修复 RCA 路径的 CPU 浮点时长；
- 已生成 16,200 条独立 injection manifest 与 952 条排除记录；
- 已生成注入区间与 ±300 s 半开上下文并集的 context-aware group manifest；
- 流式 scanner 与全量 diagnostics 已完成；13,470-case 主 cohort 与
  2,730-case multi-root sensitivity 已冻结。

### P1.3：RE2-OB adapter（completed）

- 已完成 1-case smoke test 与 90/90 case 覆盖；
- 已输出 input/label/source/group/exclusion manifests；
- scanner 已记录 layout digest 与 inject_time 内容摘要；原始 `RE2-OB.zip` 和
  90 cases 实际消费的 360 个文件已有逐文件 SHA-256 content snapshot。

### P1.3.5：telemetry diagnostics（completed）

- 已实现 Metrics/Logs/Traces 的 chunked streaming scanner；
- 定时 metric 分开报告 value-cell missingness、采样时间缺口与重复 timestamp；
- event streams 分开报告必填字段/时间解析失败和 30 s activity coverage；
- 已识别 GAIA log timestamp 来源与 RE2 topology metadata 语义；
- 真实 smoke、manifest regeneration 与全量扫描均 exit 0；G8 报告已生成。

### P1.4：split + sanity baselines（completed）

- 322 组 context-aware manifest 与 split-integrity validator 已实现；
- grouped/time-aware 候选已比较，GAIA/RE2-OB 实际 5-fold assignment 已生成；
- train-only Root Frequency 与 within-case Random/Metric-change 已实现；
- per-case ranking、overall/fault-macro/root-macro 与 run manifest 已保存。

### P1.5：Gate audit（completed）

- G1–G8 状态与证据路径已审核为 completed；
- `p1_gate_audit_v1` 已交叉验证 9 组 baseline、split/cohort/source bindings、
  指标重算、Label Firewall、全量 diagnostics 与 RE2 原始字节；
- 51/51 tests、全量 compile 与 `git diff --check` 通过。

### P2.0：representation experiment design（completed）

- 在 P1 冻结边界内定义 metric/log/trace 的测试时可得表征；
- 固定 learned preprocessing 的 inner-validation 与 train-fold-only fit 协议；
- 将 H1/H2/H3 拆成可独立否证的最小实验与 ablation；
- 先实现单模态/简单融合对照，再决定是否进入结构传播模块。

执行协议见 [P2_EXPERIMENT_PLAN.md](P2_EXPERIMENT_PLAN.md) V0.2。外部
RCAEval 参考实现固定审计 commit 为
`4695aa69f4f1f57b9094ca04ff235908b73a8e24`；官方输入/窗口与 Track C 不等价，
必须分表。

### P2-G1：feature schema + modality extractors（completed）

- 通用 label-free feature bundle 与 metric extractor 已完成；
- GAIA main/RE2-OB metric 全量产物、coverage audit 与确定性 smoke 已完成；
- L0/T0 raw schema、`frontendservice→frontend` alias、纯函数 tests 与真实 smoke
  已完成；
- readers 已改为可恢复的全量流式实现（GAIA 各 10 checkpoints、RE2 各 90），
  双数据集 L0/T0 全量 extraction 与 coverage audit 已完成；
- coverage audit 由 extractor 在同一次运行内经 `verify_feature_bundle` 复验后
  写出，**不是**独立第二进程产物，引用时不得与 `run_*`/`audit_*` 的独立审计混称；
- 证据：`artifacts/p2/event_features/`、`artifacts/p2/event_feature_{summary,audit}.json`、
  [P2_G3_MODALITY_RESULTS.md](P2_G3_MODALITY_RESULTS.md) §2。

### P2-G2：C0-M nested OOF（completed）

- outer 5-fold、outer-train 内四折轮换、四个 C 候选全部运行；
- 两数据集各 80 inner fits 与 5 outer fits，case/group overlap 均为 0；
- 完整 rankings、service scores、metrics、fit-state hashes 与 source bindings 已保存；
- 独立审计精确重算 metrics，第二次全量运行核心文件 hashes 一致；
- GAIA 不优于 P1 B2 的 root-macro 主端点，负结果保留。

### P2-G3：C0-L / C0-T / C1-I 单模态与 naive 融合（completed）

- 三个方法共用 P2-G2 冻结装置，只改输入 design（C0-L 5/10、C0-T 8/16、
  C1-I 30/60），均只用 `whole.*` 通道；
- 每方法每数据集 5 outer fits + 80 inner fits，case/group overlap 与 inner
  fit/validation overlap 均为 0；
- 独立审计精确重算 metrics、重建 selected C、逐文件校验 SHA-256；
- 两数据集 best single modality 均为 C0-M；C1-I 主端点两侧为正
  （GAIA +0.0253、RE2 +0.0200），GAIA 次端点 AC@1 退化 −0.0911；
- 配对 bootstrap（10,000 次，seed `20260819`，GAIA 按 context group、RE2 按 case）
  判定 `exploratory_signal=true`、`claim_ready=false`；
- GAIA AC@1 退化只作为诊断结果保留，**不**作为修改 inner selection objective
  的依据；若要调整须单独做 protocol version bump + sensitivity experiment；
- 证据：`artifacts/p2/runs/{c0_l,c0_t,c1_i}/`、
  `artifacts/p2/linear_ablation_{summary,audit}.json`、`artifacts/p2/c1_i_bootstrap.json`、
  [P2_G3_MODALITY_RESULTS.md](P2_G3_MODALITY_RESULTS.md)。

### P2-G4：M1-S event-stage 单因素对照（completed，判定 `no-go`）

- design 为 C1-I 的 `whole.` M/L/T（30 值）加 selected-onset 的 metric+trace
  `stage{60|120}.`（75 值），共 105 值 + 105 masks = 210 列，audit 已核对；
- 网格 4 C × 2 onsets = 8 未缩减，每 outer fold 32 inner fits ⇒ 每数据集
  实测 160 inner + 5 outer fits，为 C0-M/C1-I 的两倍；
- 冻结项全程未动：inner 目标仍唯一为 root-service macro Avg@5，
  `C={0.01,0.1,1,10}`、`onset={60,120}` 与既定 tie-break 均保持原样；未引入 RE2-TT；
- selected 超参：GAIA 五折均为 C=10 / onset=60 s；RE2-OB 为 C 1/1/10/0.1/0.1、
  onset 120/60/120/120/120 s——两数据集落在不同 onset 上；
- root-macro 结果：GAIA AC@1 0.405516 / Avg@5 0.631871；RE2-OB 0.966667 / 0.993333；
- 相对 C1-I：GAIA AC@1 +0.039918 / Avg@5 +0.024646；RE2 AC@1 −0.011111 /
  Avg@5 −0.002222；
- 配对 bootstrap（10,000 次，seed `20260819`，GAIA 按 322 context group、RE2 按
  90 case）：GAIA 主端点 CI [+0.005371, +0.043954]、次端点 CI [+0.008128, +0.070546]，
  两者下界均 > 0；RE2 主端点 CI [−0.007500, 0.000000]、次端点 CI [−0.037500, 0.000000]；
- 判定 `exploratory_signal=false`、`claim_ready=false`（两项检查均 false）⇒
  `p2_g4_decision="no-go"`；**H1 在冻结门禁下未获支持**，不得宣称 event-stage
  切分带来一致收益；
- 独立审计（`p2_m1_stage_audit_v1`）两数据集全部通过：outer/inner fold 与冻结
  split 一致、case 与 group overlap 全为 0、metrics 逐位重算一致、selected
  C/onset 可由 inner scores 重建、Label Firewall 干净，且 C1-I 对照方一并复验；
- audit 与 bootstrap 二次运行（输出至 scratch 路径）得到逐字节相同 SHA-256；
- RE2-OB 退化的确定性溯源与 6 项 limitation 记于
  [P2_G4_STAGE_RESULTS.md](P2_G4_STAGE_RESULTS.md) §7 与 §9；
- 证据：`artifacts/p2/runs/m1_s/`、
  `artifacts/p2/m1_s_{summary,audit,bootstrap}.json`、
  [P2_G4_STAGE_RESULTS.md](P2_G4_STAGE_RESULTS.md)、
  [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) §22。

### EXT-RE2TT：第三 benchmark 的 audit-first 验收（E1–E5 completed；E4 门禁 fail；E6/E7 blocked）

用户在 P2-G4 `no-go` 之后选择了 §7 末尾的**路线②**，并给出边界条件：P2-G4 的
`no-go` 与 H1「冻结门禁下未获支持」保持不变；不修改 inner selection objective；
不放宽 gate；不删除 RE2-OB；先审计数据质量 / 候选空间 / 三模态 coverage /
性能 headroom，只有审计全部通过才做 H1 replication。执行协议为
[RE2TT_EXTENSION_PROTOCOL.md](RE2TT_EXTENSION_PROTOCOL.md) V0.2。

- **E1 manifests（通过）**：90 cases、排除 0、68 候选服务、90 singleton groups；
  5 roots × 18 + 6 fault types × 15 完全平衡；`roots_outside_candidate_space=[]`；
  90/90 通过 `assert_label_free`；与 RE2-OB 零 case-ID 冲突。同一次运行用参数化后的
  adapter 重建 RE2-OB manifest 并与 `artifacts/p1/manifests/re2ob/` 做
  `inputs/labels/sources/groups.jsonl` 四文件**字节级比对，结果 `passed`**——这是
  「新增第三数据集没有扰动已记录数据集」的证据。
- **E2 telemetry（通过）**：metrics 129,690 行 / 0 非法 / 精确 1000 ms 采样 /
  0 重复 0 缺口 / missing value ratio 0.003632；logs 21,291,651 行、traces
  67,345,051 行、必填字段 0 缺失（`parentSpanID` 的 563,677 缺失为 root span）。
  异常两项：1 个 case 的 log 起点在 t0**+123.289 s**；无显式静态服务图，
  90/90 需由 trace 推导动态图。
- **E3 splits（通过）**：`selection_eligible=true`，18/18/18/18/18，
  `max_fault_type_total_variation=0.0`、`max_root_service_total_variation=0.066667`，
  case 与 group overlap 均为 0，复跑逐字节一致。
- **E4 headroom（不通过）**：B2 metric-change 的 root-macro Avg@5 = **0.904444**，
  超出预登记上限 0.90，`decision="fail"`、`failed_gates=["H-1"]`。H-2（B2 AC@1
  0.822222 < 0.95）与 H-3（B2 相对 B1 主端点 +0.337778）通过。
- **E5 独立审计（通过）**：`integrity_gates_passed=true`，四个 section 全通过，
  三层 metrics 在 `1e-12` 内重算一致，全部预测为合法的 68 服务排列，独立复判
  headroom 判定同为 `fail` / `["H-1"]`（`matches_recorded_gate=true`）。
- **判定**：按预登记规则「E1–E5 全部通过前不进入 E6/E7」，
  `blocked_stages=["E6","E7"]`。用户明确要求「不放宽 gate」，因此不得为让 E7
  开工而调整 0.90 上限。
- **结论方向**：审计结果不是「换个数据集就能救 H1」，而是「第三个 benchmark
  同样饱和」，这**加强**而非推翻既有的 P2-G4 `no-go`。
- 证据：`artifacts/ext/re2tt/{manifests,source_snapshot,splits,baselines,
  telemetry_diagnostics.json,split_diagnostics.json,baseline_summary.json,
  headroom_gate.json,gate_audit.json}`（该目录下**不存在** `features/`、
  `event_features/`、`runs/`，与 E6/E7 未执行一致）、
  [RE2TT_EXTENSION_PROTOCOL.md](RE2TT_EXTENSION_PROTOCOL.md) §9、
  [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) §23。

### P2-G5 及之后：待用户决策，尚未开工

P2-G4 `no-go` 之后曾列出三条互斥方向，用户已选择其中的**路线②（重议 RE2-TT）**，
该路线现已在 audit 阶段收口（见上一节）。当前仍未开工、且都须用户明确决定后才立项
的选项（均不得回改已冻结协议或既有记录，详见
[P2_G4_STAGE_RESULTS.md](P2_G4_STAGE_RESULTS.md) §10 与
[RE2TT_EXTENSION_PROTOCOL.md](RE2TT_EXTENSION_PROTOCOL.md) §9.6）：

1. 维持现状，把 H1 记为“未获支持”，并把「三个 benchmark 都饱和」写成方法/
   limitation 叙述；
2. 就 RE2-TT 的 H-1 失败作出显式处置：或者接受 E6/E7 永久 blocked，或者在**新的
   protocol version bump**（不是就地放宽）中重新定义 headroom 准入口径；
3. 对 inner selection objective 做单独的 protocol version bump + sensitivity
   experiment。

是否保留 Event-stage、是否进入 H2，按用户设定的顺序需在看到 RE2-TT 结果**之后**
决定；结果现已产生，但由于 E7 未执行，RE2-TT 尚未提供 H1 replication 证据。
M2-R / M2-D / M3-G 在上述决策产生前不实现。H2 / H3 未检验，不得预判其结论。

## 8. 主要风险

| 风险 | 当前控制 |
|---|---|
| 把异常检测标签语义直接当 RCA 证据 | GAIA + RE2-OB 双数据集；Frequency baseline；宏平均 |
| 同一/重叠事件或共享上下文跨 split | ±300 s context-aware groups + 实际 assignment + split-integrity/checksum test |
| test label 进入特征/候选/normalization | 物理拆分 Input/Label + Label Firewall test |
| 旧 Pilot 数字被误当新方法结果 | 在实验日志标记 `conversation-reported` |
| 先设计模型后发现任务不统一 | P1 benchmark 与最终门禁已完成；P2 必须复用冻结协议 |
| 结构创新被预设 | H3 Go/No-Go |
| RCAEval 主分支持续变化 | 本地归档与实际消费文件已 content-pinned；上游 provenance 缺失需披露 |
| 官方 service-level 口径与自定义 evaluator 漂移 | toy tests + 对照官方 evaluator |
| 为参数化 RE2-TT 而改动的 adapter 悄悄改变已记录的 RE2-OB 数字 | E1 在写任何 extension 产物前重建 RE2-OB manifest 并做四文件字节级比对（`passed`）；四个 extension driver 的 `main()` 首行拒绝写入 `artifacts/p1`、`artifacts/p2` |
| RE2-TT 的 log 缺失与 root 身份共变，使「log 通道被 mask」成为近确定性的 root 指示器 | 缺口 18/18 全在 root=`ts-train-service`、其余 72 例全覆盖，已记为 confound：C1-I 的 log 整通道需报 coverage 切片；M1-S staged 通道只用 metric+trace，H1 单因素不吃这条捷径；E7 若出现该 root 的组异常必须先排除此 confound |
| RE2-TT 三层报告数值相同，掩盖分组塌陷型失败 | 已在 §2 与协议 §7 显式记录：RE2-TT 不能替代 GAIA 承担 macro 分层诊断职能 |

## 9. 决策日志

| 日期 | 旧理解 | 新决策 | 原因/证据 |
|---|---|---|---|
| 2026-08-18 | RCA 可作为 Ada-MGAD 下游 head | RCA 应独立成立，只在事件层集成 | 用户希望形成两项独立研究；AD 表征目标与 root attribution 不同 |
| 2026-08-18 | 继续旧 `rca` 分支 | 旧分支冻结为 Pilot；新分支从 Ada-MGAD 基线重新开发 | 避免继承 A/P/T 和检测器强耦合设计 |
| 2026-08-18 | GAIA 窗口可作为 RCA 样本 | 一次 fault injection / failure case 为一个样本 | 避免 pseudo-replication 与 event leakage |
| 2026-08-18 | 重叠注入可逐条随机分配 | 直接或传递重叠的 injection 构成不可拆分 group | 防止相关事件及其 telemetry 上下文跨 split；不预设 fold 类型 |
| 2026-08-18 | RCA 粒度可能含 indicator | 主任务固定为 service-level | GAIA 与 RCAEval 的公共可比较粒度 |
| 2026-08-19 | G8 只有 scanner 实现、原始分组只看注入区间 | G8 以全量产物完成；主窗口冻结为 ±300 s，并以注入区间与上下文并集生成 split 原子组 | 全量扫描 349,120,558 行；±300 s 为 322 组/最大 471，±600 s 仅 23 组/最大 3,206 |
| 2026-08-19 | GAIA grouped fold 与 purged time block 尚未选择，RE2-OB 无实际 fold | 主协议冻结为 seed `20260819` 的 grouped-stratified 5-fold；GAIA 时间块只作敏感性分析 | GAIA 时间块把 12 条 CPU fault 全放入 fold-4；grouped folds 大小 3239/3239/3239/3241/3242 且全类别覆盖；RE2 每折 18 cases |
| 2026-08-19 | P1 只有 evaluator，没有实际 sanity rankings | B0 Random、B1 train-fold Root Frequency、B2 within-case Metric Change 全部运行并保存 | GAIA Frequency 暴露 overall shortcut；B2 GAIA/RE2 AC@1 为 0.1414/0.8556；预测与 run manifests 字节级复现 |
| 2026-08-19 | GAIA 16,200 cases 的 single-root 纳入语义未冻结 | 主结果排除锚点时存在其他 root service 的 2,730 cases；13,470 cases 进入 main，同-root 并发保留 | 规则直接来自 service-level label cardinality，不使用模型表现；主 folds 仍完整覆盖 root/fault |
| 2026-08-19 | RE2-OB 只有 90/90 可读和目录布局摘要，P1 仍待收口 | 对原始归档与 360 个实际消费文件做 content-addressed snapshot；最终自动门禁通过后关闭 P1、进入 P2 | 本地无上游 version 元数据；内容身份可逐字节固定实际实验输入，gate audit 同时验证 9 组 baseline 与全部绑定 |
| 2026-08-19 | metric onset 候选为 30/60/120 s | 统一 metric 调参只允许 60/120 s，30 s 保留 unsupported control | GAIA 30 s onset blocks 仅约 0.09% service rows 可观察；决策发生在 stage 模型运行前 |
| 2026-08-19 | C0-M 只是计划项 | C0-M nested OOF 完成并作为 learned metric reference；不宣称统一优于 B2 | GAIA/RE2 root-macro Avg@5 为 0.5820/0.9756，相对 B2 为 −0.1266/+0.0422；完整泄漏与确定性审计通过 |
| 2026-08-19 | log/trace entity 与派生字段边界未定 | L0 只用 raw 无词表统计；T0 固定 `frontendservice→frontend`，缺失 trace entities 保留 mask | GAIA 无 raw template ID；RE2 template 字段已派生且 trace 仅直接覆盖 7 类 entity |
| 2026-08-19 | L0/T0 staged 通道是否含 log 未定 | M1-S 的 staged 通道排除 log，只用 metric + trace 的 `stage{60,120}` | RE2 log 三档 onset 的 content-complete 比例同为 0.794900，低于冻结的 0.80 绝对阈值；决策发生在 M1-S 运行前 |
| 2026-08-19 | C0-L/C0-T/C1-I 只是计划项 | P2-G3 完成；C1-I 记为 `exploratory_signal=true` / `claim_ready=false`，不得用于任何论文级 claim | 两数据集主端点点估计为正（+0.0253/+0.0200），但 GAIA CI 下界 −0.003557 ≤ 0 且 GAIA AC@1 −0.0911 < −0.01 |
| 2026-08-19 | 是否因 C1-I 的 GAIA AC@1 退化调整 inner selection objective | 不调整；保持 inner 唯一目标为 root-service macro Avg@5，AC@1 退化只作 P2-G4 诊断 | 用户已确认：中途改选择目标会破坏 H1 的单因素可解释性；如需调整须单独 protocol version bump + sensitivity experiment |
| 2026-08-19 | 是否为绕开 RE2 天花板效应引入 RE2-TT | 暂不引入；先在冻结的 GAIA + RE2-OB 协议上完成 P2-G4，RE2 天花板记为 limitation | 用户已确认：是否扩展待 H1 结果出来后再决定，当前不重新打开 P1/P2 数据协议 |
| 2026-08-20（**取代上一行**） | RE2-TT 暂不引入 | 引入 RE2-TT，但只作为**独立 protocol extension**：P2-G4 的 `no-go` 与 H1「未获支持」不变、不改 inner selection objective、不放宽 gate、不删 RE2-OB，且必须 audit-first（先验数据质量/候选空间/三模态 coverage/headroom，全部通过才做 H1 replication） | 用户已确认（路线②）；产物物理隔离在 `artifacts/ext/re2tt/`，四个 driver 的 `main()` 首行 `_assert_isolated()` 拒写 `artifacts/p1`、`artifacts/p2` |
| 2026-08-20 | RE2-TT 可能提供未饱和的第三 benchmark，从而给 H1 一次干净的 replication | E1/E2/E3/E5 通过，但 **E4 预登记门禁 H-1 不通过**（B2 root-macro Avg@5 = 0.904444 > 0.90 上限，超出 0.004444 = 2 个 case 量子）⇒ `blocked_stages=["E6","E7"]`，H1 replication 未执行 | `artifacts/ext/re2tt/headroom_gate.json` 与独立复判的 `gate_audit.json` 判定一致（`matches_recorded_gate=true`）；门禁在看到任何 C1-I/M1-S 数字之前已预登记 |
| 2026-08-20 | RE2 天花板可能由 RE2-OB 的 11 个候选服务太少造成，换 68 候选的 RE2-TT 即可缓解 | 天花板**不是**候选空间问题：候选从 11 → 68 使 B0 从 0.255556 崩到 0.037778，但 B2 只从 0.933333 降到 0.904444（−0.028889） | 归因于 RCAEval RE2 系列共有的单点注入 + metric 全覆盖构造（RE2-TT metric presence ratio 在 30/60/120/300 s 全为 1.000000）；因此再换一个 RE2 release 也不会解决 |
| 2026-08-20 | 更大的候选空间会顺带修复 5-root frequency prior 的退化 | 未修复：B1 在 RE2-TT 与 RE2-OB 上**逐位相同**（Avg@5 0.566667、AC@5 1.000000） | 已排除接线错误（run manifest 绑定 `artifacts/ext/re2tt/{manifests,splits}` 且预测排满 68 个 RE2-TT 服务）；这是 5 root × 18 case 平衡设计 + B1 只依赖 train-fold root prior 的结构性后果 |
| 2026-08-20 | E-G2 的 root-conditioned coverage 缺口应落在 `telemetry_diagnostics.json` | 改落在 E5 的 `gate_audit.json → root_conditioned_coverage`；E5 是本扩展中唯一被允许读标签的阶段（纯审计、不产出任何进入模型的特征） | 该缺口是 root-conditioned 事实，必须读 `labels.jsonl` 才能算；`telemetry_diagnostics.json` 与 P1 同口径、刻意不接触标签。落点更正发生在看到 E4/E7 任何数值之前 |
| 2026-08-20 | M1-S 只是计划项，H1 待检验 | P2-G4 完成并判定 `no-go`；H1 记为“在冻结门禁下未获支持”，不得宣称 event-stage 切分带来一致收益 | GAIA 双端点为正且 CI 下界 > 0（+0.024646 / +0.039918），但 RE2 主端点点估计 −0.002222 为负、CI 下界 −0.007500 ≤ 0，次端点 −0.011111 < −0.01；三项检查全 false |
| 2026-08-20 | RE2-OB 退化可能提示 stage 通道有系统性害处 | RE2-OB 退化溯源为单个 case 的 rank 1 → 2，只作诊断记录，**不**改变 `no-go` 判定，也**不**作为放宽阈值或更换数据集的依据 | 逐 case 比对显示 89/90 只有尾部重排、仅 1/90 的 root 排名变化；5 root × 18 cases 下该 case 恰好解释 −0.011111 与 −0.002222 |
| 2026-08-20 | `no-go` 后可直接推进 M2-R/M2-D/M3-G | 在用户就三条互斥路线（维持现状 / 重议 RE2-TT / selection objective 独立 bump）明确决定前不立项、不实现 | 用户已确认停止点为 `M1-S run → audit → bootstrap → P2-G4 go/no-go`；越过 gate 推进会使 H2/H3 失去可解释的对照基线 |
| 2026-08-18 | HR@k 为主 | 对齐 AC@1/3/5 + Avg@5，MRR 补充 | 对齐 RCAEval 官方 evaluator |
| 2026-08-18 | 直接进入方法开发 | 先完成 P1 Benchmark Layer | 当前任务协议、候选集合、split 和泄漏风险尚未落地 |
| 2026-08-18 | P1 可能预切 event stages | P1 只提供完整锚点上下文 | 阶段划分属于 H1，不应固化成数据协议 |
| 2026-08-18 | 参考对话即项目状态 | 仓库证据与对话报告分级 | 当前 clone 缺少旧 Pilot 和 RCA 实现 |

## 10. 新对话恢复检查

继续工作前只需确认：

1. 当前分支与提交是否仍与本文件一致；
2. 是否已有未登记的 RCA 文件、数据或实验产物；
3. P1 冻结产物的哈希是否仍通过 `scripts/audit_p1_gates.py`；
4. P2-G4 已收口为 `no-go`：`artifacts/p2/runs/m1_s/` 与
   `artifacts/p2/m1_s_{summary,audit,bootstrap}.json` 均应存在，且
   `m1_s_bootstrap.json` 的 `p2_g4_decision` 应为 `"no-go"`；
5. RE2-TT extension 已收口在 audit 阶段：`artifacts/ext/re2tt/gate_audit.json`
   的 `integrity_gates_passed` 应为 `true`，而
   `artifacts/ext/re2tt/headroom_gate.json` 的 `decision` 应为 `"fail"`、
   `failed_gates` 应为 `["H-1"]`；不应存在 `artifacts/ext/re2tt/features/`、
   `event_features/` 或 `runs/`（E6/E7 被阻塞，未执行）；
6. 下一项**不是**继续跑实验，而是等用户在 §7 末尾三条互斥路线中做出选择；
   在此之前不得改动冻结协议、不得放宽 RE2-TT 的 0.90 headroom 上限、
   不得修改 inner selection objective、不得删除 RE2-OB；
7. M2-R / M2-D / M3-G 在上述决策产生前不实现；H2 / H3 未检验，不得预判结论；
   「是否保留 Event-stage、是否进入 H2」按用户设定顺序须在 RE2-TT 结果之后决定，
   而 E7 未执行意味着 RE2-TT 尚未提供 H1 replication 证据。

如仓库事实变化，以代码/测试/产物为准更新本文件，不沿用过期状态。
