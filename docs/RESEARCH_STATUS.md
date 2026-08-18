# 研究状态与决策日志

> 状态版本：`research_state_v1`  
> 最近更新：2026-08-18  
> 状态所有者：本文件；完成 Gate 后必须同步更新

## 1. 当前状态摘要

| 项目 | 状态 | 证据 |
|---|---|---|
| Ada-MGAD 异常检测代码 | 已存在 | 当前 `src/`、`util/`、`README.md` |
| 异常检测论文/完整实验产物 | 当前工作区不可用 | 仅参考对话提及附件与结果 |
| 旧 Ada-MGAD-dependent RCA Pilot | 对话报告已完成，当前 clone 不可核验 | [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) |
| Standalone RCA 研究定位 | 已冻结 | [RCA_RESEARCH_DESIGN.md](RCA_RESEARCH_DESIGN.md) V0.2 |
| P1 Benchmark Protocol | 已确认 | [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) V0.1 |
| `rca-standalone` 分支 | 已建立 | Git 当前分支与远端 tracking branch |
| `RCACase` / adapters / evaluator / baselines | 未实现 | 全仓库搜索无 RCA 代码 |
| RE2-OB 数据 | 未下载/未登记 | 当前 `data/` 只有 `msds_tiny` smoke data |
| P1 G1–G8 | 0/8 有实现证据 | 见下方 Gate 表 |

**当前研究阶段：P1 — Unified RCA Benchmark Layer。**

## 2. 仓库快照

```text
branch:       rca-standalone
remote:       origin/rca-standalone
commit:       ab31282055b625838f41a1a3c91e51098175f254
subject:      first commit
commit date:  2026-07-15T22:27:48+08:00
working tree: 文档变更前 clean
```

当前事实：

- Git 仅显示一个本地分支与对应远端分支；
- `docs/` 在本轮之前为空；
- 代码仍是 Ada-MGAD 的 GAIA/MSDS 异常检测实现；
- `util/GAIA/pre_GAIA.py` 已有事件解析与 `label_events.csv` 输出，可作为 GAIA RCA adapter 的起点；
- 现有代码同时把事件映射到 30 s 节点异常标签，这部分属于旧异常检测协议；
- 没有 RE2-OB adapter、统一 schema、RCA evaluator、baseline 和 RCA tests。

## 3. ARS 工作流位置

本轮使用 `academic-research-suite` 的 `academic-pipeline` 路由与 `state_tracker` 规范建立跨对话研究状态，但**没有启动完整论文十阶段 pipeline**，也没有声称通过论文完整性 Gate。

当前实质工作属于研究设计到实验规划的过渡：

```text
研究问题与路线收敛       completed
Standalone RCA Design    completed / frozen V0.2
P1 Benchmark Protocol    completed / approved V0.1
P1 implementation        pending
P2 representation        blocked by P1 G1-G8
paper full drafting      not started in this workspace
```

## 4. P1 Gate 状态

| Gate | 内容 | 状态 | 证据路径 |
|---|---|---|---|
| G1 | GAIA event → RCACase | pending | — |
| G2 | RE2-OB 90 cases → RCACase | pending | — |
| G3 | 两数据集完整 service ranking | pending | — |
| G4 | AC/Avg/MRR toy tests | pending | — |
| G5 | 三个 sanity baselines | pending | — |
| G6 | event/case split 无泄漏 | pending | — |
| G7 | Label Firewall | pending | — |
| G8 | dataset diagnostic report | pending | — |

P2 的进入条件为 G1–G8 全部 `completed` 且具备代码/测试/产物路径。

## 5. 已冻结决策

1. 毕设包含独立的异常检测与 RCA 两项研究，最终在事件层集成。
2. Standalone RCA 不使用 Ada-MGAD latent representation。
3. 旧 `rca` 只作为 Pilot，不作为新方法开发基线。
4. 新 RCA 任务是 event-level、service-level、single-root ranking 主协议。
5. 主数据为 GAIA + RE2-OB；RE2-TT 为扩展。
6. 主实验为 Oracle RCA；Trigger Robustness 与 Ada-MGAD-triggered RCA 后置。
7. 候选集合由测试时可得信息构造，主实验包含全部可观测服务。
8. 先 case/event split，再拟合任何学习型 preprocessing。
9. P1 实现 Label Firewall，并只做 Random/Frequency/Metric-change。
10. H1/H2/H3 是可检验假设，不是预设创新点。
11. P1 不把 Event-stage 三段切分固化到数据协议，也不实现新神经网络。
12. 只有 P1 G1–G8 完成后才进入 P2。

## 6. 尚未决定的问题

这些问题必须由 P1 诊断或实验回答，不能在新对话中凭经验直接填值：

- GAIA 有效独立 injection 的数量与清洗规则；
- GAIA 单根因、多根因、重叠事件的最终处理；
- GAIA 使用 5-fold、grouped fold 还是时间分块；
- 两数据集的 `T_pre/T_post`；
- RE2-OB 服务名、指标名、trace/log entity 的统一映射规则；
- topology 的统一表示与缺失/动态策略；
- learned method 的 fold stratification 与 validation 划分；
- Metric Change Score 的距离函数和跨指标聚合；
- H1 的阶段边界、H2 的比较模块、H3 的结构机制；
- 最终论文题目是否加入“根因分析”。

## 7. 下一步执行队列

### P1.1：schema + evaluator（下一项）

- 建立 `RCACaseInput` / `RCACaseLabel`；
- 实现 schema invariants；
- 实现 AC@1/3/5、Avg@5、MRR；
- 用手工 toy cases 覆盖正确首位、Top-k 命中、缺失根因、重复/不完整 ranking；
- 建立 Label Firewall 的最小接口测试。

### P1.2：GAIA adapter

- 复用并审计 `parse_anomaly_event` / `label_events.csv`；
- 生成独立 injection manifest；
- 不使用 30 s 窗口作为 case id；
- 输出第一版 GAIA diagnostics。

### P1.3：RE2-OB adapter

- 固定 RCAEval 代码/数据版本；
- 先用 1–3 cases 做 schema smoke test，再覆盖 90 cases；
- 输出 RE2-OB diagnostics 与无效 case 清单。

### P1.4：split + sanity baselines

- 实现 case manifest 与 train-only preprocessing；
- Random / Frequency / Metric-change；
- 保存 per-case ranking 与聚合指标。

### P1.5：Gate audit

- 审核 G1–G8；
- 只有全部通过后更新本文件并进入 P2。

## 8. 主要风险

| 风险 | 当前控制 |
|---|---|
| 把异常检测标签语义直接当 RCA 证据 | GAIA + RE2-OB 双数据集；Frequency baseline；宏平均 |
| 同一事件窗口跨 split | case manifest + split integrity test |
| test label 进入特征/候选/normalization | 物理拆分 Input/Label + Label Firewall test |
| 旧 Pilot 数字被误当新方法结果 | 在实验日志标记 `conversation-reported` |
| 先设计模型后发现任务不统一 | P1 G1–G8 阻塞 P2 |
| 结构创新被预设 | H3 Go/No-Go |
| RCAEval 主分支持续变化 | 固定 commit/tag 和数据 manifest |
| 官方 service-level 口径与自定义 evaluator 漂移 | toy tests + 对照官方 evaluator |

## 9. 决策日志

| 日期 | 旧理解 | 新决策 | 原因/证据 |
|---|---|---|---|
| 2026-08-18 | RCA 可作为 Ada-MGAD 下游 head | RCA 应独立成立，只在事件层集成 | 用户希望形成两项独立研究；AD 表征目标与 root attribution 不同 |
| 2026-08-18 | 继续旧 `rca` 分支 | 旧分支冻结为 Pilot；新分支从 Ada-MGAD 基线重新开发 | 避免继承 A/P/T 和检测器强耦合设计 |
| 2026-08-18 | GAIA 窗口可作为 RCA 样本 | 一次 fault injection / failure case 为一个样本 | 避免 pseudo-replication 与 event leakage |
| 2026-08-18 | RCA 粒度可能含 indicator | 主任务固定为 service-level | GAIA 与 RCAEval 的公共可比较粒度 |
| 2026-08-18 | HR@k 为主 | 对齐 AC@1/3/5 + Avg@5，MRR 补充 | 对齐 RCAEval 官方 evaluator |
| 2026-08-18 | 直接进入方法开发 | 先完成 P1 Benchmark Layer | 当前任务协议、候选集合、split 和泄漏风险尚未落地 |
| 2026-08-18 | P1 可能预切 event stages | P1 只提供完整锚点上下文 | 阶段划分属于 H1，不应固化成数据协议 |
| 2026-08-18 | 参考对话即项目状态 | 仓库证据与对话报告分级 | 当前 clone 缺少旧 Pilot 和 RCA 实现 |

## 10. 新对话恢复检查

继续工作前只需确认：

1. 当前分支与提交是否仍与本文件一致；
2. 是否已有未登记的 RCA 文件、数据或实验产物；
3. P1 Gate 表是否需要更新；
4. 下一项未完成任务是否仍是 P1.1。

如仓库事实变化，以代码/测试/产物为准更新本文件，不沿用过期状态。

