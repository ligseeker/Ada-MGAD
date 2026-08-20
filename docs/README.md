# 研究连续性文档入口

> 状态版本：`research_context_v8`
> 最近更新：2026-08-20
> 当前分支：`claudecode`（git worktree；PR 目标 `main`）
> 基准提交：`c2af48f2be4066de7363d7e5f0871052e8564301`（RE2-TT extension 的
> driver 与文档尚未提交，其数字以 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) §23
> 记录的逐文件 SHA-256 为准）
> 当前 Gate：P2-G4（M1-S）已完成，判定 **`no-go`**；RE2-TT protocol extension
> 的 E1–E5 已执行完毕，**E4 预登记 headroom 门禁 `fail`，E6/E7 被阻塞**；
> 下一步待用户决策

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
| 9 | [RE2TT_EXTENSION_PROTOCOL.md](RE2TT_EXTENSION_PROTOCOL.md) | 查看 RE2-TT 独立 protocol extension V0.2：隔离规则、E1–E7 阶段设计、预登记 headroom 门禁，以及 §9 的 E1–E5 实际结果与 E6/E7 阻塞原因 |
| 10 | [P2_MODALITY_SCHEMA_AUDIT.md](P2_MODALITY_SCHEMA_AUDIT.md) | 查看 L0/T0 的 raw 字段与实体映射边界 |
| 11 | [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) | 查看已完成的 P1 统一 Event-level / Service-level RCA 协议 V0.5 |
| 12 | [DATASET_AUDIT.md](DATASET_AUDIT.md) | 查看 GAIA / RE2-OB 的真实本地布局、初步计数和未决风险 |
| 13 | [GAIA_INCLUSION_AUDIT.md](GAIA_INCLUSION_AUDIT.md) | 查看 13,470-case 主 cohort、2,730-case 并发 sensitivity 与 G1 证据 |
| 14 | [TELEMETRY_DIAGNOSTICS.md](TELEMETRY_DIAGNOSTICS.md) | 查看 G8 全量 coverage/missingness/topology 诊断与窗口证据 |
| 15 | [SPLIT_DIAGNOSTICS.md](SPLIT_DIAGNOSTICS.md) | 查看实际 5-fold assignment、候选比较与 G6 证据 |
| 16 | [BASELINE_RESULTS.md](BASELINE_RESULTS.md) | 查看 B0/B1/B2 的完整 ranking、macro 结果与 Label Firewall 证据 |
| 17 | [P1_REPRODUCIBILITY_AUDIT.md](P1_REPRODUCIBILITY_AUDIT.md) | 查看 RCAEval 内容身份、最终自动门禁和 P1 收口证据 |
| 18 | [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | 查看旧 Pilot 证据及后续实验记录规范 |

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

**RE2-TT protocol extension（并行、物理隔离在 `artifacts/ext/re2tt/`）已按用户
指定的 audit-first 顺序执行完 E1–E5：** E1 manifests（90 cases / 68 候选服务 /
5 root × 18 与 6 fault × 15 完全平衡 / RE2-OB manifest 四文件字节级回归
`passed`）、E2 telemetry（129,690 metric 行精确 1000 ms 采样、三模态必填字段
无缺失、±300 s label-free presence ratio 为 metric 1.000000 / log 0.654739 /
trace 0.383333）、E3 splits（18×5、root TV 0.066667 < 0.10、overlap 全 0、
复跑逐字节一致）与 E5 独立审计（`integrity_gates_passed=true`）**全部通过**；
但 **E4 的预登记 headroom 门禁 H-1 不通过**——B2 metric-change 的 root-macro
Avg@5 = **0.904444** 超出 0.90 上限（超出 0.004444，恰为 2 个 case 量子），故
`decision="fail"`、`failed_gates=["H-1"]`、`blocked_stages=["E6","E7"]`，
**RE2-TT 上的 C1-I → M1-S H1 replication 未执行**。两项可直接引用的结构性结论：
候选空间从 11 扩到 68 使 B0 从 0.255556 崩到 0.037778，但 B2 只降 0.028889，
说明 RE2 天花板来自单点注入 + metric 全覆盖的构造而非候选空间；B1 在 RE2-TT 与
RE2-OB 上逐位相同，说明 5-root frequency prior 的退化并未被更大候选空间修复。

## 新对话启动方式

在新对话中可直接使用以下提示：

```text
请先完整阅读 docs/README.md、docs/PROJECT_CONTEXT.md、
docs/RESEARCH_STATUS.md、docs/RCA_RESEARCH_DESIGN.md、docs/P2_EXPERIMENT_PLAN.md、
docs/P2_METRIC_FEATURES.md、docs/P2_C0_M_RESULTS.md、docs/P2_G3_MODALITY_RESULTS.md、
docs/P2_G4_STAGE_RESULTS.md、docs/RE2TT_EXTENSION_PROTOCOL.md、
docs/P2_MODALITY_SCHEMA_AUDIT.md、docs/BENCHMARK_PROTOCOL.md、docs/DATASET_AUDIT.md、
docs/GAIA_INCLUSION_AUDIT.md、docs/TELEMETRY_DIAGNOSTICS.md、docs/SPLIT_DIAGNOSTICS.md 和
docs/BASELINE_RESULTS.md、docs/P1_REPRODUCIBILITY_AUDIT.md 和
docs/EXPERIMENT_LOG.md。
以 docs/RESEARCH_STATUS.md 的当前状态为准。P2-G4 已完成并判定 no-go；
RE2-TT protocol extension 的 E1–E5 已执行完毕，E4 预登记 headroom 门禁判定
fail（B2 root-macro Avg@5 = 0.904444 > 0.90），E6/E7 被阻塞、未执行。
因此**不要**自行选择下一步：先向我确认 docs/RESEARCH_STATUS.md §7 末尾三条互斥路线
（维持现状把 H1 与「三个 benchmark 都饱和」写成 limitation / 显式处置 RE2-TT 的
H-1 失败 / 对 inner selection objective 做独立 protocol version bump +
sensitivity experiment）中采用哪一条。
不要把 conversation-reported 的旧 Pilot 结果当作当前分支已复现结果，
不要回改 P1/P2 冻结边界，不要缩减 C/onset 网格或改动 inner selection objective，
不要就地放宽 RE2-TT 的 0.90 headroom 上限或任何 Go/No-Go 阈值，
不要删除 RE2-OB，也不要往 artifacts/p1/** 或 artifacts/p2/** 写入 extension 产物，
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
- 每完成一个 P1 / P2 / EXT Gate，同时更新 [RESEARCH_STATUS.md](RESEARCH_STATUS.md) 的状态与证据路径。
- protocol extension（当前只有 RE2-TT）的产物一律写在 `artifacts/ext/<name>/`，
  不得写入 `artifacts/p1/**` 或 `artifacts/p2/**`；其结果与主协议结果分表报告。
- 研究设计发生实质变化时递增版本号，并在决策日志说明原因。
- 任何数值结果必须附数据版本、split、seed、命令、提交哈希和产物路径。
- 外部资料注明核验日期；无法核验的内容明确标为“待验证”。
