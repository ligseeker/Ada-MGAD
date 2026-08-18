# 研究连续性文档入口

> 状态版本：`research_context_v1`  
> 最近更新：2026-08-18  
> 当前分支：`rca-standalone`  
> 基准提交：`ab31282055b625838f41a1a3c91e51098175f254`

本目录是后续对话继续 Ada-MGAD / Standalone RCA 研究时的唯一入口。它把参考对话中的已确认决策、当前仓库中的可验证事实、外部官方资料和仍待验证的研究假设分开记录，避免在新对话中重复讨论或把计划误写成已完成结果。

## 建议阅读顺序

| 顺序 | 文档 | 用途 |
|---|---|---|
| 1 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | 理解毕设背景、两项研究的关系、术语和证据边界 |
| 2 | [RESEARCH_STATUS.md](RESEARCH_STATUS.md) | 查看当前真实进度、已冻结决策、阻塞项和下一步 |
| 3 | [RCA_RESEARCH_DESIGN.md](RCA_RESEARCH_DESIGN.md) | 查看 Standalone RCA Research Design V0.2 |
| 4 | [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) | 执行 P1 统一 Event-level / Service-level RCA 协议 |
| 5 | [DATASET_AUDIT.md](DATASET_AUDIT.md) | 查看 GAIA / RE2-OB 的真实本地布局、初步计数和未决风险 |
| 6 | [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | 查看旧 Pilot 证据及后续实验记录规范 |

## 当前一句话状态

当前正式阶段仍是 **P1：统一 GAIA 与 RCAEval RE2-OB 的事件级、服务级 RCA Benchmark Layer**。统一 schema、评价器、两个 adapter、确定性 manifest bundle 与 overlap split-integrity 首版已实现；G2、G4 完成，G1、G6、G7 partial。下一步是完整模态覆盖/missingness 诊断、冻结 split，再实现 sanity baselines；不进入新神经网络设计。

## 新对话启动方式

在新对话中可直接使用以下提示：

```text
请先完整阅读 docs/README.md、docs/PROJECT_CONTEXT.md、
docs/RESEARCH_STATUS.md、docs/RCA_RESEARCH_DESIGN.md、
docs/BENCHMARK_PROTOCOL.md、docs/DATASET_AUDIT.md 和 docs/EXPERIMENT_LOG.md。
以 docs/RESEARCH_STATUS.md 的当前状态为准，从 P1 的下一项未完成任务继续；
不要把 conversation-reported 的旧 Pilot 结果当作当前分支已复现结果，
不要在 P1 阶段加入 Ada-MGAD 表征或新的复杂神经网络。
```

## 信息权威顺序

发生冲突时按以下顺序处理：

1. 当前仓库中的代码、测试、数据清单和可复现实验产物；
2. 本目录中标记为“已冻结/用户已确认”的决策；
3. 官方论文、官方仓库和官方数据说明；
4. 参考对话中尚未被本地材料验证的陈述；
5. 研究建议或待检验假设。

## 状态标签

- **仓库已验证**：可由当前提交中的代码或 Git 状态直接核验。
- **官方已验证**：可由外部项目的官方论文、仓库或数据说明核验。
- **用户已确认**：在参考对话中由用户明确接受的研究决策。
- **对话报告**：参考对话声称已完成，但当前工作区没有对应代码或原始产物。
- **待验证**：尚未通过实验或数据审计支持的假设、数量或方案。

## 更新规则

- 不覆盖旧实验结论；新结果追加到 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。
- 每完成一个 P1 Gate，同时更新 [RESEARCH_STATUS.md](RESEARCH_STATUS.md) 的状态与证据路径。
- 研究设计发生实质变化时递增版本号，并在决策日志说明原因。
- 任何数值结果必须附数据版本、split、seed、命令、提交哈希和产物路径。
- 外部资料注明核验日期；无法核验的内容明确标为“待验证”。

