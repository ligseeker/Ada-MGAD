# P6-C0 逐窗口动态图修复与开发验证

日期：2026-10-01。状态：实施前协议；本次用户已授权继续研究、训练检测器，不执行全量预处理。

## 问题与假设

旧 `DynamicGraphLearner._summarize_inputs` 对 batch 求均值，生成一张共享图。连续的 32 个窗口相隔 30s，最早窗口的 logit 可受其后 930s 的输入影响。合成反例已确认同行变化会改变固定首样本的 logit。修复后的图只依赖本窗口，并保持原参数、图融合、输入、监督和训练选择规则。

这是正确性修复，是否有效不由 F1 上升决定。科学假设另列：移除批内前视后，真实窗口仍能提供足够 onset 信号；新因果基线是否达到原工程门槛，需要重新训练后测量。旧 checkpoint 换图推理只用于影响诊断。

## Material Passport

| 项目 | 固定来源/边界 |
|---|---|
| Base | `experiment/p6-c0-window-causal`，来自 `f275156cc5bdd43ab4d3649183e3562796d678f0` |
| GT | canonical registry SHA `7bc1b8b9c99164d0df004b72f47977b5413070c033834babac33931a27a0324b` |
| 输入 | 主工作树共享只读 `data/p5/v3_preprocessing_v2/ad/train/{timestamps,metric,log,trace}.npy` 和 `graph.npy` |
| 时间 | 原 50/20/30，Fit 与 Validation 事件分母 7443/2901；历史窗口/预测时刻/三态标签逐条对账 |
| 预处理等级 | 原 70% Train 拟合的 schema/scalers/vocabulary/graph，包括 detector Validation 的无标签输入；仅开发筛查，不是严格 preprocessing-OOS |
| 标注读取 | registry 区间列用于时间路由；Test root/fault 行在解析前跳过；跨界事件保留标签状态但 purge 事件指标总体 |
| 新配置 | `configs/e2e/gaia_p6_c0_window_causal_v1.json`；graph_batch_scope=`window` |
| 修改变量 | batch 共享动态图 → 逐窗口动态图；无其他模型、特征、损失、标签或 decoder 候选变化 |
| 训练 | V100 GPU、seed42、原 AdaBelief/学习率/BCE/pos_weight、batch32、最多30epoch、patience8 |
| 选择 | 原 Validation exact threshold sweep；checkpoint 按 event F1/recall/threshold；不从 Test 选择 |
| 存储 | `experiments/p6/c0_window_causal/{checkpoint-impact-v1,retrained-v1-seed42}/`，入口拒绝已有目录 |

## 顺序与门槛

1. 合成 gate：固定首样本，修改同行/重排/batch1、2、32，不改变 logit；finite backward；singleton graph 和 regularizer 对齐旧实现。
2. 开发 loader gate：仅 Train 数组，禁止 node labels/Test 数据；与归档 Fit/Validation 的 sample_index、prediction_available_time、trigger_label 精确一致。保留跨 Fit/Validation 和 Validation/Test 的 ongoing state。
3. 旧 GPU control checkpoint 只在 Validation 进行影响审计。`batch` 模式 TP/FP/FN 必须精确重放 2103/3/798；`window` 分别报告固定旧阈值和重新选择 Validation 阈值的结果，均不能称为重训修复结果。
4. 真实输入因果 gate：固定32个等距 Validation 样本，同行扰动、重排、batch1/2/32；绝对 logit 容差固定为 `1e-5`，相对容差0。超出即停止，不根据结果放宽容差。
5. 在同一输入上重训逐窗口图。选定 checkpoint 的 CSV 必须重放其所选事件指标；运行末再次核输入/源码 SHA 与 execution commit；全部产物写 completion manifest。
6. 报告修复后的 Validation P/R/F1、TP/FP/FN、paired gained/lost、fault/duration/overlap 分组和失败位置。历史 comparator 仅表明修复影响，不能支持“在线性能超过旧方法”。工程参考门槛 P≥.90、R≥.60、F1≥.72，达到后才具备继续研究的基线。

本协议不含任何 Test inference action。若要重新评估 Test 或重建 C1 OOS anchors，必须另立新 run、预测锁及更正报告。当前 Test 已反复观察，只能报告 reused-Test 描述性证据。

## 后续研究范围

修复基线后最多实施一个新的模型候选，优先评估仅历史窗口的 XGBoost trigger，或先做多 onset 可辨识性审计。不得把对称 ±300s 的 RCA 68D 直接用于在线 detector。模型候选与 raw-observability mask 修复分别比较，不捆绑变量。预处理只准备小规模验证与完整手动命令，不运行全量。

旧结论的适用范围见 `P6_BATCH_GRAPH_CAUSALITY_CORRECTION.md`。
