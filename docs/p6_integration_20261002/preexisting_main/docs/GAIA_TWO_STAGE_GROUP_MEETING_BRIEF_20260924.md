# GAIA 两阶段故障检测与定位：组会汇报简报

**整理日期：** 2026-09-24  
**数据截止：** 以 `GAIA_P5_CURRENT_CONTEXT.md`（2026-09-23 更新）和已归档实验产物为准。

## 一句话结论

GAIA 上的两阶段流水线已在 P5 完整跑通；当前主要短板是 Stage 1 的事件漏检和检测锚点对 RCA 的影响。P6-C0 的独立 Stage-1 触发器把事件级 F1 提高到 0.8395，但预注册结论仍是 **BORDERLINE**，且没有运行 Stage 2/RCA，因此目前不能声称完整 E2E 已提升。

## 方法与数据

流程是 **Stage 1：多模态异常/事件检测 → Stage 2：基于检测锚点的故障服务排序**。Ada-MGAD 输入包含 Metric `[T,10,48]`、Log `[T,10,32]`、Trace edge `[T,10,10,8]`；Stage 2 使用冻结的 68 维 Z2 表示，对 10 个服务排序。GAIA 标签表示注入/故障服务，不应表述为独立验证过的因果根因。

P5 是当前唯一完成的全量两阶段 E2E baseline：按时间 70/30 划分，预处理与 RCA scaler 由 Train 拟合，事件阈值只用 Train 选择。Test 有 26,127 个窗口、5,787 个完整 GT 事件。

## P5：完整两阶段基线

| 层级 | Test 结果 | 分母/口径 |
|---|---|---|
| Ada-MGAD 节点检测 | AUC **0.9328**；AP **0.8199**；F1 **0.8693**；Precision **0.9692**；Recall **0.7881** | 261,270 个节点行 |
| Stage 1 事件检测 | TP/FP/FN **3,703/65/2,084**；P/R/F1 **0.9827/0.6399/0.7751** | 全部 5,787 个 Test GT 事件 |
| 检测延迟 | 均值 **24.11 秒**；P95 **39.51 秒** | 3,703 个匹配事件 |
| Stage 2，GT anchor | AC@1/3/5 **0.9125/0.9919/0.9968**；MRR **0.9525** | 3,702 个 matched 且 W300 合法案例 |
| Stage 2，detected anchor | AC@1/3/5 **0.4657/0.9211/0.9830**；MRR **0.6988** | 同一批 3,702 个案例 |
| 完整 E2E Diagnosis | F1@1/**@3**/**@5** **0.3611/0.7145/0.7623** | 5,781 个 W300 合法 GT 事件；包含检测漏报、误报及排序失败 |

**如何解释：** Oracle/GT anchor 下 Stage 2 的 AC@1 为 0.9125；换成检测器实际产生的 anchor 后为 0.4657，显示定位结果对锚点时间上下文很敏感。它支持“anchor/context 是重要问题”的判断，但单凭该差值不能确定唯一因果机制。Stage 2 优化器已收敛；当前结果不像是单纯的优化器未收敛。总体分布不均衡：mobservice1/2 约占匹配案例 98.2%，login_failure 约占故障类型 97.8%；汇报时应同时看 weighted、root-macro 和 fault-macro 指标。

## P6：Stage 1 后续进展（不是新的完整 E2E）

P6-C0 测试了**根服务标签无关的系统级事件监督**：不使用节点异常标签或根服务标签，也不加载 Ada-MGAD checkpoint；它仍使用各服务 telemetry 和服务图，因此不是“完全没有服务身份信息”的无监督模型。P6-C0 使用 50/20/30 Fit/Validation/Test，checkpoint 与阈值由 Validation 选择。

| 指标 | P5 历史事件检测 | P6-C0 修正版 |
|---|---:|---:|
| Precision | 0.9827 | **0.9962** |
| Recall | 0.6399 | **0.7254** |
| F1 | 0.7751 | **0.8395** |
| TP/FP/FN | 3,703/65/2,084 | 4,198/16/1,589 |
| 平均 / P95 延迟 | 24.11 / 39.51 秒 | 23.77 / 39.15 秒 |

P6-C0 的三个总体工程门槛均通过（P≥0.90、R≥0.60、F1≥0.72），但正式 verdict 是 **BORDERLINE**：`>300s` 事件只有 **32/229** 被检出（Recall 0.1397）；`memory_anomalies` 为 **30/204**（Recall 0.1471）。按后续勘误，204 个 memory 事件是 229 个长时事件的子集，应视为重叠分层，不能算成两个独立失效机制。多 onset 的 Recall 为 0.4141，低于 single-onset 的 0.7769。

**比较边界：** P5 与 P6-C0 的监督标签、划分和阈值选择协议不同，且使用同一批 5,787 个 Test 事件；差值只能作为历史参考，不能当作严格控制的模型优劣结论。P6-C0 只输出系统级触发分数，没有服务排序，本轮 **未运行 RCA**。

P6-C0 首轮曾出现 30 秒标签网格错位，修正后才得到上表正式结果；首轮 F1=0.7021 的产物保留为历史记录。后续 C0F 更正审计没有重新训练或做 Test 推理；它把 Test 漏检位置分为 below-threshold 394、no-new-episode 509、matching-competition 683、no-legal-prediction 3。这些是损失发生位置的分类，不是已确认的因果解释。C0F 给出的协议级 episode 上界为 5,173/5,787，但不代表当前模型可达到。

## 当前状态与下一步

- **已完成：** P5 全量两阶段 baseline；P6-A/B0 审计；P6-C0 Stage-1 实验；P6-C0F 更正审计；C1 G1 静态可行性账本。
- **尚未完成：** P6-C1 受控 anchor-domain RCA 对照，以及 P6-C2 完整 E2E 评价。
- **C1 门槛：** G1 静态盘点给出 Fit-only 三折的 GT 支持上界 **4,254**；这不是实际生成的 OOS anchor 或可训练案例数。真实 OOS anchors 和共同 Train cohort 仍未知，当前状态为 **`EXECUTION_NO_GO_AS_IS`**。需先冻结逐折前向时间协议、输入 schema/拟合边界、共同案例集、scaler 和评价方案。
- **测试集限制：** 当前 Test 已用于研究设计与问题发现；后续在同一 Test 上的结果应明确称为复用 Test 评价，不能称为独立确认。
- **P5 归档限制：** P5 数值产物已生成，但结果包仍有 provenance 保留项（例如 final report 未记录 pytest、若干重写文件的旧 hash 未更新、detected feature hash 缺失）。这不改变本简报引用的指标，但正式发布时应披露并单独处理。

## 组会上可直接使用的 40 秒摘要

> 我们已经完成 GAIA 上 Ada-MGAD 检测接 Ada-RCA 排序的全量两阶段基线。P5 的事件检测 precision 很高，为 0.983，但 recall 是 0.640；RCA 在 GT 锚点下 AC@1 是 0.912，在检测锚点下降到 0.466，完整诊断 F1@1 为 0.361，说明当前瓶颈主要落在事件覆盖和锚点上下文。P6-C0 改用不含根服务标签的系统级事件监督后，事件 F1 达到 0.840、recall 达到 0.725，但长时事件召回只有 0.140，所以预注册 verdict 仍是 BORDERLINE，而且没有做 RCA。下一步不是直接宣布端到端提升，而是先冻结严格的前向 OOS 锚点与同案例 RCA 对照协议，再决定是否进入完整 E2E 评价。

## 主要证据

- [当前实验上下文](GAIA_P5_CURRENT_CONTEXT.md)
- [P5 完整 E2E 报告](../experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440/final_report.md)
- [P6-C0 Stage-1 结果](P6_C0_SYSTEM_EVENT_TRIGGER_RESULT.md)
- [P6-C0F 更正审计结果](../experiments/p6/c0f_failure_audit/c0f-correction-20260923T0755Z/final_report.md)
- [P6-C1 G1 静态可行性报告](../experiments/p6/c1_feasibility/c1-g1-20260923T105441Z/final_report.md)
- [P6 后续路线与勘误](P6_RESEARCH_ROADMAP.md)
