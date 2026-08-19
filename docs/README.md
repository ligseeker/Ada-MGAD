# 研究连续性文档入口

> 状态版本：`research_context_v7`
> 最近更新：2026-08-20
> 当前分支：`claudecode`（git worktree；PR 目标 `main`）
> 基准提交：`64bb681328fa1793014615a20ba2bc1fdf33c3b7`
> 当前 Gate：P2-G4（M1-S）已完成，判定 **`no-go`**；下一步待用户决策

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
| 7 | [P2_G3_MODALITY_RESULTS.md](P2_G3_MODALITY_RESULTS.md) | 查看 P2-G1 全量 L0/T0 coverage 与 P2-G3 的 B2/C0-M/C0-L/C0-T/C1-I 对照、配对 bootstrap 与门禁判定 |
| 8 | [P2_G4_STAGE_RESULTS.md](P2_G4_STAGE_RESULTS.md) | 查看 P2-G4 M1-S 的 210 列 design、逐 fold 结果、配对 bootstrap 与 `no-go` 判定依据 |
| 9 | [P2_MODALITY_SCHEMA_AUDIT.md](P2_MODALITY_SCHEMA_AUDIT.md) | 查看 L0/T0 的 raw 字段与实体映射边界 |
| 10 | [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) | 查看已完成的 P1 统一 Event-level / Service-level RCA 协议 V0.5 |
| 11 | [DATASET_AUDIT.md](DATASET_AUDIT.md) | 查看 GAIA / RE2-OB 的真实本地布局、初步计数和未决风险 |
| 12 | [GAIA_INCLUSION_AUDIT.md](GAIA_INCLUSION_AUDIT.md) | 查看 13,470-case 主 cohort、2,730-case 并发 sensitivity 与 G1 证据 |
| 13 | [TELEMETRY_DIAGNOSTICS.md](TELEMETRY_DIAGNOSTICS.md) | 查看 G8 全量 coverage/missingness/topology 诊断与窗口证据 |
| 14 | [SPLIT_DIAGNOSTICS.md](SPLIT_DIAGNOSTICS.md) | 查看实际 5-fold assignment、候选比较与 G6 证据 |
| 15 | [BASELINE_RESULTS.md](BASELINE_RESULTS.md) | 查看 B0/B1/B2 的完整 ranking、macro 结果与 Label Firewall 证据 |
| 16 | [P1_REPRODUCIBILITY_AUDIT.md](P1_REPRODUCIBILITY_AUDIT.md) | 查看 RCAEval 内容身份、最终自动门禁和 P1 收口证据 |
| 17 | [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | 查看旧 Pilot 证据及后续实验记录规范 |

## 当前一句话状态

**P1 已完成；P2-G1（metric + 全量 L0/T0）、P2-G2 C0-M、P2-G3（C0-L / C0-T / C1-I）
与 P2-G4（M1-S）均已完成；P2-G4 判定为 `no-go`，下一步需用户在三条互斥路线中决策。**
全量 metric bundle 覆盖 GAIA main 13,470 cases/134,700 service rows 与 RE2-OB
90/990；L0/T0 全量 extraction 已完成（GAIA 各 134,700 × 50/80，RE2 各 990 ×
50/80），checksum、Label Firewall 与 coverage audit 通过。C1-I 相对 best single
modality（两数据集均为 C0-M）的配对 bootstrap 判定为 `exploratory_signal=true`、
`claim_ready=false`；M1-S 相对 C1-I 的配对 bootstrap 判定为
`exploratory_signal=false`、`claim_ready=false`、`p2_g4_decision="no-go"` —— GAIA
双端点为正且 CI 下界 > 0，但 RE2-OB 主端点 −0.002222 为负、次端点 −0.011111 越过
−0.01 guardrail，全部退化溯源为单个 case。metric onset 仅允许 60/120 s；RE2 log 因
content-complete 0.794900 < 0.80 被排除出 M1-S 的 staged 通道。GAIA 上所有 learned
方法主端点仍低于未学习的 P1 B2（M1-S 为 −0.0767）。**H1 在冻结门禁下未获支持；
H2/H3 均未检验。**

## 新对话启动方式

在新对话中可直接使用以下提示：

```text
请先完整阅读 docs/README.md、docs/PROJECT_CONTEXT.md、
docs/RESEARCH_STATUS.md、docs/RCA_RESEARCH_DESIGN.md、docs/P2_EXPERIMENT_PLAN.md、
docs/P2_METRIC_FEATURES.md、docs/P2_C0_M_RESULTS.md、docs/P2_G3_MODALITY_RESULTS.md、
docs/P2_G4_STAGE_RESULTS.md、docs/P2_MODALITY_SCHEMA_AUDIT.md、
docs/BENCHMARK_PROTOCOL.md、docs/DATASET_AUDIT.md、
docs/GAIA_INCLUSION_AUDIT.md、docs/TELEMETRY_DIAGNOSTICS.md、docs/SPLIT_DIAGNOSTICS.md 和
docs/BASELINE_RESULTS.md、docs/P1_REPRODUCIBILITY_AUDIT.md 和
docs/EXPERIMENT_LOG.md。
以 docs/RESEARCH_STATUS.md 的当前状态为准。P2-G4 已完成并判定 no-go，
因此**不要**自行选择下一步：先向我确认 docs/RESEARCH_STATUS.md §7 末尾三条互斥路线
（维持现状把 H1 记为未获支持 / 重议是否扩展 RE2-TT / 对 inner selection objective
做独立 protocol version bump + sensitivity experiment）中采用哪一条。
不要把 conversation-reported 的旧 Pilot 结果当作当前分支已复现结果，
不要回改 P1/P2 冻结边界，不要缩减 C/onset 网格或改动 inner selection objective，
不要为绕开 RE2-OB 天花板效应擅自引入 RE2-TT 或放宽 Go/No-Go 阈值，
也不要跳过受控消融直接堆叠复杂网络；
M2-R/M2-D/M3-G 在上述决策产生前不实现；H2/H3 未检验，不得预判其结论。
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
- 每完成一个 P1 / P2 Gate，同时更新 [RESEARCH_STATUS.md](RESEARCH_STATUS.md) 的状态与证据路径。
- 研究设计发生实质变化时递增版本号，并在决策日志说明原因。
- 任何数值结果必须附数据版本、split、seed、命令、提交哈希和产物路径。
- 外部资料注明核验日期；无法核验的内容明确标为“待验证”。
