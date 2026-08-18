# 研究连续性文档入口

> 状态版本：`research_context_v5`
> 最近更新：2026-08-19
> 当前分支：`rca-standalone`
> 基准提交：`ab31282055b625838f41a1a3c91e51098175f254`

本目录是后续对话继续 Ada-MGAD / Standalone RCA 研究时的唯一入口。它把参考对话中的已确认决策、当前仓库中的可验证事实、外部官方资料和仍待验证的研究假设分开记录，避免在新对话中重复讨论或把计划误写成已完成结果。

## 建议阅读顺序

| 顺序 | 文档 | 用途 |
|---|---|---|
| 1 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | 理解毕设背景、两项研究的关系、术语和证据边界 |
| 2 | [RESEARCH_STATUS.md](RESEARCH_STATUS.md) | 查看当前真实进度、已冻结决策、阻塞项和下一步 |
| 3 | [RCA_RESEARCH_DESIGN.md](RCA_RESEARCH_DESIGN.md) | 查看 Standalone RCA Research Design V0.2 |
| 4 | [P2_EXPERIMENT_PLAN.md](P2_EXPERIMENT_PLAN.md) | 执行当前 P2 表征、相对比较、结构假设的受控实验计划 V0.2 |
| 5 | [P2_METRIC_FEATURES.md](P2_METRIC_FEATURES.md) | 查看 P2-G1 metric schema、全量提取、coverage 与 onset 支持性审计 |
| 6 | [P2_C0_M_RESULTS.md](P2_C0_M_RESULTS.md) | 查看 P2-G2 nested OOF、P1 B2 对照与确定性证据 |
| 7 | [P2_MODALITY_SCHEMA_AUDIT.md](P2_MODALITY_SCHEMA_AUDIT.md) | 查看下一步 L0/T0 的 raw 字段与实体映射边界 |
| 8 | [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) | 查看已完成的 P1 统一 Event-level / Service-level RCA 协议 V0.5 |
| 9 | [DATASET_AUDIT.md](DATASET_AUDIT.md) | 查看 GAIA / RE2-OB 的真实本地布局、初步计数和未决风险 |
| 10 | [GAIA_INCLUSION_AUDIT.md](GAIA_INCLUSION_AUDIT.md) | 查看 13,470-case 主 cohort、2,730-case 并发 sensitivity 与 G1 证据 |
| 11 | [TELEMETRY_DIAGNOSTICS.md](TELEMETRY_DIAGNOSTICS.md) | 查看 G8 全量 coverage/missingness/topology 诊断与窗口证据 |
| 12 | [SPLIT_DIAGNOSTICS.md](SPLIT_DIAGNOSTICS.md) | 查看实际 5-fold assignment、候选比较与 G6 证据 |
| 13 | [BASELINE_RESULTS.md](BASELINE_RESULTS.md) | 查看 B0/B1/B2 的完整 ranking、macro 结果与 Label Firewall 证据 |
| 14 | [P1_REPRODUCIBILITY_AUDIT.md](P1_REPRODUCIBILITY_AUDIT.md) | 查看 RCAEval 内容身份、最终自动门禁和 P1 收口证据 |
| 15 | [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | 查看旧 Pilot 证据及后续实验记录规范 |

## 当前一句话状态

**P1 已完成；P2-G1 metric 子门与 P2-G2 C0-M 已完成。** 全量 170 维 metric
bundle 覆盖 GAIA main 13,470 cases/134,700 service rows 与 RE2-OB 90/990，
checksum、Label Firewall 和 coverage audit 通过。C0-M nested OOF 已保存完整
rankings，并通过 80-inner-fit 泄漏审计和核心文件确定性复跑；L0/T0 synthetic
tests 与真实双数据集 smoke 也已通过，下一步为全量流式 extraction。metric onset
仅允许 60/120 s，30 s 因 GAIA
采样不足只保留 unsupported control。

## 新对话启动方式

在新对话中可直接使用以下提示：

```text
请先完整阅读 docs/README.md、docs/PROJECT_CONTEXT.md、
docs/RESEARCH_STATUS.md、docs/RCA_RESEARCH_DESIGN.md、docs/P2_EXPERIMENT_PLAN.md、
docs/P2_METRIC_FEATURES.md、docs/P2_C0_M_RESULTS.md、docs/P2_MODALITY_SCHEMA_AUDIT.md、
docs/BENCHMARK_PROTOCOL.md、docs/DATASET_AUDIT.md、
docs/GAIA_INCLUSION_AUDIT.md、docs/TELEMETRY_DIAGNOSTICS.md、docs/SPLIT_DIAGNOSTICS.md 和
docs/BASELINE_RESULTS.md、docs/P1_REPRODUCIBILITY_AUDIT.md 和
docs/EXPERIMENT_LOG.md。
以 docs/RESEARCH_STATUS.md 的当前状态为准，从 P2 的下一项未完成任务继续；
不要把 conversation-reported 的旧 Pilot 结果当作当前分支已复现结果，
不要回改 P1 冻结边界，也不要跳过 P2 的单模态、独立 scorer 与受控消融直接堆叠复杂网络。
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
