# GAIA P5/P6 当前实验上下文

> 本文件是给新的 code-agent CLI 的单一入口。它只保留当前有效的背景、状态、约束和下一步；完整审计与实现细节继续放在现有文档中，不在这里复制。

更新时间：2026-10-01（Asia/Shanghai；本独立锚点回溯工作树）

**2026-10-01 本独立锚点回溯工作树最新增量：** 用户继续授权以提高
两阶段 RCA AC@1，并在 `experiment/p6-mob-pair` 建立独立工作树。
fold 1 Train OOS 延迟中位数固定为 `25.621s`，将 RCA 68D 特征的
Train/Test 窗口中心统一设为 `t_detected−25.621s`，XGBRanker、检测器、
候选、matching 和分母不变。Train 时间前推 fold 2/3 AC@1 为
`0.8938/0.8975`（原 XGB `0.7226/0.7701`）。单次已复用 Test 的同一
4,197 个 legal matched case 上，AC@1 为 `3735/4197=0.8899`，
原 XGB 为 `3093/4197=0.7370`；完整 E2E F1@1 为 `0.7469`。
Test 预测锁含 4,214 episode、4,213 legal ranking、1 invalid context，
无 ranking failure。原评价入口因 matching CSV 的 NA 解析错误在写结果
前失败；独立更正评价只读原预测锁并完成精确 baseline replay 和独立
计数复算。详见[结果](P6_C1_ANCHOR_BACKDATE_RESULTS.md)与
[计划](P6_C1_ANCHOR_BACKDATE_DEV_AND_TEST_PLAN.md)。证据等级是多次
复用 Test 的探索性结果；改善主要来自 mobservice1/2 的 login_failure，
非主服务与 memory 故障仍弱。当前不再在同一 Test 调参；完整 E2E
Recall@1 仍受 Stage-1 recall `0.7254` 限制。

**前一独立 XGB 工作树增量：** 用户另行授权的 68D + XGBRanker RCA 单变量探索已在
`experiment/p6-z2-xgb` 完成，冻结执行源码 `e08f3a0`，独立 run
`experiments/p6/c1_z2_xgb/c1-z2-xgb-v1-seed20260826/`。同一 4,197
个 legal matched Test case 上，XGB 对原 C1-v2 Arm C Conditional Logit 的
配对 AC@1 为 `+0.04241`；完整 E2E F1@1 为 `0.6185` 对 `0.5829`。
这是已复用 Test 的**探索性**对照，不能升格为独立确认，也不覆盖下面
原 C1-v2/C2 的正式结论；见[新协议](P6_C1_Z2_XGB_PROTOCOL_V1.md)与
[结果](P6_C1_Z2_XGB_RESULTS.md)。其后用户另行授权的固定锚点回溯
研究见本文件顶部；该独立分支不再用同一 Test 继续调参。

**后续两阶段开发检验（2026-09-30）：** 独立 C0R2 工作树仅在
Fit/Validation 诊断固定检测分数的再触发语义，Validation recall
`0.7322→0.7735`，但新增命中以重叠的短时 login_failure 为主，
原 C0 verdict 和 Test 结果不变。当前 XGB 工作树完成
[Train-only RCA 时间前推检验](P6_C1_XGB_FORWARD_DEV_RESULT.md)：fold 2 / 3
的 paired AC@1 增益 `+0.1772 / +0.0312`，证据等级是后验开发检验，
无新 Test 评价；XGB 可留作独立新数据的候选，当前不继续调参。

**当前进度：P5 完整 E2E baseline 已完成；P6-A/B0 审计、P6-C0 系统级触发和 C0F 更正审计已完成。P6-C1-v2（`P6-C1-SUPERVISION-OOS-v1`，run `c1-supervision-oos-v1-seed42`）已完整执行并归档：三折 detector、OOS matching、共同 Train cohort 3,225 case、共享 scaler 的 B/C RCA 臂、label-free Test 预测锁和 C1/C2 评估全部完成，主结果 `Delta AC@1 = AC@1(C) - AC@1(B) = +0.19657`（95% descriptive interval [0.17041, 0.22467]），outcome 为 `ANCHOR_ALIGNMENT_SUPPORTED`；详见 [C1-v2/C2 结果](P6_C1_V2_AND_C2_RESULTS.md)。旧的 strict prefix C1 run `c1-prefix-oos-v1-seed42` 仍为 `PREFIX_SCHEMA_NO_GO`，其失败记录保留且不得复用。**

**历史（已停用的 strict-prefix C1，保留且不得复用）：P6-C0 的 aggregate gate 为 PASS，正式 verdict 仍为 BORDERLINE，尚未运行新 RCA/E2E。P6-C0F 独立更正审计已完成并归档，状态为 `COMPLETE_WITH_DECLARED_LIMITATIONS`。C1 的 G1 静态账本和 G2 设计锁已完成，G2 Python 3.8 更正使用 v1.1 配置。G3 的六类执行接口、阶段封存、合成测试及真实 adapter 的 1/24-worker 合成 raw smoke 已通过；[原手动正式运行方案](P6_C1_G3_MANUAL_RUN.md)已因真实 prefix 门槛失败而停用。2026-09-29 全量只读输入预检通过后，正式 fold 1 Fit 因真实 Metric 合格槽位仅 27 个、低于冻结的 45 个预算而停止；本 run 为 `PREFIX_SCHEMA_NO_GO`，详见[失败归档](P6_C1_G3_FOLD1_PREFIX_SCHEMA_NO_GO_20260929.md)。随后的[独立 Metric 审计](P6_C1_FOLD1_METRIC_PREFIX_AUDIT_RESULT_20260929.md)已完成封存：477 个候选中 45 个通过质量门槛，18 个相关性去重，剩余 27 个；384 个 host 候选在首个目标上的 Fit 覆盖率为零。没有完成任何 fold，也没有 OOS 锚点、共同 Train cohort、预测锁或 C1/C2 结果；不得继续执行此 run。**

后续 coding agent 先读本文件、[C1-v2 协议](P6_C1_V2_SUPERVISION_OOS_PROTOCOL.md) 与 [C1-v2/C2 结果](P6_C1_V2_AND_C2_RESULTS.md)、[C1 G2 Python 3.8 更正记录](../experiments/p6/c1_protocol/c1-g2-py38-correction-20260924T0831Z/correction_report.md)、[C1 G2 设计锁](P6_C1_G2_FROZEN_DESIGN.md)、[C1 G1 静态可行性报告](../experiments/p6/c1_feasibility/c1-g1-20260923T105441Z/final_report.md)、[C0F 归档说明](P6_C0F_CORRECTION_ARCHIVE.md)、[C0F 更正报告](../experiments/p6/c0f_failure_audit/c0f-correction-20260923T0755Z/final_report.md) 与 [更正协议](P6_C0F_CORRECTION_PROTOCOL.md)，再读 [后续实验方案](P6_NEXT_EXPERIMENT_PLAN.md)、[P6 研究路线](P6_RESEARCH_ROADMAP.md) 和 [C0F 原实施方案](P6_C0F_FAILURE_MECHANISM_AUDIT_PLAN.md)。路线图和原方案的“C0F 尚未执行”段落是 2026-09-18 的历史状态，不是当前状态。

## 1. 项目身份与研究边界

- 工作目录：`/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2`
- 当前分支：`e2e-v2`
- 独立 XGB 探索工作树：`/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2-xgb`，分支 `experiment/p6-z2-xgb`；其新增源码和产物只属于此分支，不回写原 run。
- C0F 更正执行时的 base HEAD 是 `d518c17f4a76ddb30e9bfaeb33fecc982fad3f21`；归档 commit 是 `eac0f44677ba1baba6b2ecac2e7b8e0e606856bd`。更正 run 的实际执行源码由 `source_snapshot.json` 绑定，source digest 为 `afbcf349692e2746ab99e961b15bb3bf129b8205c3ca9d62ee6e3f3d7716b720`，不能把 base HEAD 误写为完整执行源码。P5 执行 commit 为 `7dea779fc5196218a43d86a9777b591d290f047b`，P6-C0 修正版执行 commit 为 `cedc4a2bd7492933a8295067c8075e631cbf3df9`。后续会话仍需现场核对 HEAD、工作树和哈希。
- 任务：在 GAIA/MicroSS 上运行 Ada-MGAD 多模态异常检测，并通过冻结的 Ada-RCA 适配器完成两阶段事件检测与故障服务定位。
- Ada-MGAD 和 Ada-RCA 的核心模型、损失和 68D Z2 表示不在本轮重新设计；本仓库主要负责 GAIA 适配、预处理、事件协议、编排、指标和 provenance。
- P6-C0 是独立训练的 system-level trigger，不加载 Ada-MGAD checkpoint；使用 root-service-label-agnostic supervision，但仍消费具体服务的 telemetry 与 graph，不能说模型完全没有服务身份信息。
- RCA 的 canonical source commit：`a2c620922e7c0ab3615d34654d4a3690d1b22c8e`。不要把本仓库的 E2E adapter 结果表述为独立因果根因证明；标签语义是 labelled injected/fault service。

## 2. 先读哪些文件

按以下顺序读取即可，不要从旧文档的历史状态重新推断当前进度：

1. 本文件；
2. [C1 G2 Python 3.8 更正记录](../experiments/p6/c1_protocol/c1-g2-py38-correction-20260924T0831Z/correction_report.md) → [G2 设计锁](P6_C1_G2_FROZEN_DESIGN.md) → [G1 静态报告](../experiments/p6/c1_feasibility/c1-g1-20260923T105441Z/final_report.md) → [C0F 归档说明](P6_C0F_CORRECTION_ARCHIVE.md) → [更正报告](../experiments/p6/c0f_failure_audit/c0f-correction-20260923T0755Z/final_report.md) → [更正协议](P6_C0F_CORRECTION_PROTOCOL.md) → [后续实验方案](P6_NEXT_EXPERIMENT_PLAN.md) → [P6 研究路线与交接](P6_RESEARCH_ROADMAP.md)：当前来源绑定、设计、静态可行性、审计结论与历史决策；
3. [P6-C0 结果](P6_C0_SYSTEM_EVENT_TRIGGER_RESULT.md) 和 [原协议](P6_C0_SYSTEM_EVENT_TRIGGER_PROTOCOL.md)，并按路线图第 8 节读取解释性勘误；数值以对应 JSON 为准；
4. [P6-A 失败审计](P6_E2E_FAILURE_AUDIT.md) 与 [P6-B0/B0R 分数分解](P6_B0_SCORE_DECOMPOSITION_AUDIT.md)；
5. P5 baseline 的 [`final_report.md`](../experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440/final_report.md) 和 [`GAIA_V3_IMPLEMENTATION.md`](GAIA_V3_IMPLEMENTATION.md)：既有完整 E2E 与入口说明；
6. [`P5_GAIA_E2E_PROTOCOL_V2_AUDIT.md`](P5_GAIA_E2E_PROTOCOL_V2_AUDIT.md)、[`GAIA_MULTIMODAL_PREPROCESSING_AUDIT_V1.md`](GAIA_MULTIMODAL_PREPROCESSING_AUDIT_V1.md)、[`GAIA_MULTIMODAL_PREPROCESSING_CHANGE_PLAN_V1.md`](GAIA_MULTIMODAL_PREPROCESSING_CHANGE_PLAN_V1.md)：历史协议与预处理决策依据。

`GAIA_PREPROCESSING_V2_ALIGNMENT_REVIEW.md` 和早期 runbook 中出现的“尚未正式运行/NO-GO”是运行前的历史快照；不要删除历史证据，也不要把这些旧状态当成当前状态。

## 3. P5 完整 baseline 与 P6 当前进度

### 3.1 已完成的 P5 完整 E2E baseline

当前完整两阶段 baseline 的引用目录是：

```text
experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440/
```

状态证据：

- `run_state.json`: `COMPLETE`；
- `final_report.md`: `FORMAL_FULL_DATA_COMPLETE`；
- required artifact 均已生成；
- 产物已完成；如需判断当前是否有进程，必须现场检查，不能只依据本状态快照；
- 运行是从之前被杀掉的进程续跑完成的，不是一次连续执行。

阶段结果：

| 阶段 | 状态 | 备注 |
|---|---|---|
| Ada-MGAD train/inference | 完成 | 20 epoch，主 checkpoint 按 Train F1 选择 |
| Event detection | 完成 | 阈值只由 Train 选择 |
| Detected-anchor RCA features | 完成 | 68D，使用过已校验的 shard cache |
| Ada-RCA training/inference | 完成 | Train-only scaler，优化器收敛 |
| E2E evaluation/finalization | 完成 | 已输出四层比较和完整 failure semantics |

### 3.2 P6 后续阶段

| 阶段 | 状态 | 产物/解释 |
|---|---|---|
| P6-A E2E failure audit | 完成 | anchor 时间上下文敏感；具体机制仍有 Hypothesis |
| P6-B0/B0R score decomposition | 完成，Outcome C | `experiments/p6/score_decomposition/`；严格 `1e-6` replay gate FAIL，事件层复现一致 |
| P6-C0 修正版 system trigger | 完成，BORDERLINE | `experiments/p6/system_event_trigger/`；仅 Stage 1，未运行 RCA |
| P6-C0 首轮 label-lag 历史 | 保留 | `experiments/p6/system_event_trigger_v1_label_lag/`；不能替代修正版结果 |
| P6-C0F | 原 run 完成；独立更正 run `COMPLETE_WITH_DECLARED_LIMITATIONS` | `experiments/p6/c0f_failure_audit/c0f-correction-20260923T0755Z/`；不训练、不改阈值、不运行 Test 模型推理或 RCA；旧 Fit/Validation 推理源码仍 `UNVERIFIED` |
| P6-C1 G1 静态可行性 | 完成，`EXECUTION_NO_GO_AS_IS` | `experiments/p6/c1_feasibility/c1-g1-20260923T105441Z/`；Fit-only 三折的静态 GT/窗口上界 4,254，实际 OOS 锚点与共同 Train cohort 未知 |
| P6-C1 G2 设计锁 | 完成设计；Python 3.8 兼容更正 | `docs/P6_C1_G2_FROZEN_DESIGN.md`、`configs/e2e/gaia_p6_c1_g2_v1_1.json`、独立更正记录；三折 prefix-fit detector、固定 RCA 表示与 transductive 文件名目录限制、共同 Train 队列下限、预测锁和评估规则已固定 |
| P6-C1 G3 逐折实现与首次正式尝试 | `PREFIX_SCHEMA_NO_GO`，停止 | [失败归档](P6_C1_G3_FOLD1_PREFIX_SCHEMA_NO_GO_20260929.md)；代码、合成 smoke 与只读输入预检通过，但真实 fold 1 Fit 仅有 27/45 个合格 Metric base slots。失败目录保留 `INCOMPLETE`；无完成 fold、模型结果或 Test 评价 |
| P6-C1 fold 1 Metric 候选审计 | `COMPLETE`，仅诊断 | [结果与后续方案](P6_C1_FOLD1_METRIC_PREFIX_AUDIT_RESULT_20260929.md)；封存产物重现 27/45：质量通过 45、相关性排除 18；不改变原 C1 verdict，也不是模型效果结果 |
| P6-C1-v2 supervision-OOS + P6-C2 | 完成，`ANCHOR_ALIGNMENT_SUPPORTED` | `experiments/p6/c1_supervision_oos/c1-supervision-oos-v1-seed42/`；证据等级为 supervision-OOS，不是 prefix-preprocessing-OOS；共同 Train cohort 3,225（935/1168/1122 ≥ 596/765/767）；Test 预测锁 4,214 episode（4,213 legal，0 ranking failure）；主结果 `Delta AC@1 = +0.19657`（[0.17041, 0.22467]）；C2 full diagnosis F1@1 B 0.4180 / C 0.5829；见 [结果归档](P6_C1_V2_AND_C2_RESULTS.md) |
| P6-C1 68D + XGBRanker 独立探索 | 完成，reused-Test descriptive | 新工作树 `experiment/p6-z2-xgb`，冻结 commit `e08f3a0`，独立 run `experiments/p6/c1_z2_xgb/c1-z2-xgb-v1-seed20260826/`；原 C1 Arm C 对 XGB 的 paired AC@1 0.6945→0.7370（+0.04241），full E2E F1@1 0.5829→0.6185；无 Test 调参，见[结果](P6_C1_Z2_XGB_RESULTS.md) |
| P6-C1 固定锚点回溯 + 原 XGB | 完成，reused-Test exploratory | 本独立工作树；fold 1 Train OOS delay median 固定回溯 25.621s，fold 2/3 前推 AC@1 `0.8938/0.8975`；Test 同一 4,197 legal matched case AC@1 `0.7370→0.8899`，full E2E F1@1 `0.6185→0.7469`；更正评价只读原预测锁，见[结果](P6_C1_ANCHOR_BACKDATE_RESULTS.md) |
| P6-C0R2 后续 | 未执行，仍是独立支线 | 不再由本轮自动启动；如需进行必须先单独冻结协议，不得用已看过的 Test 选配置 |

## 4. 数据、预处理和输入契约

共享输入根目录（只读）：

```text
data/p5/v3_preprocessing_v2/ad/
artifacts/p5/v3_preprocessing_v2/ad/
artifacts/p5/v3_preprocessing_v2/protocol/
data/p5/v3/rca_raw_index/
data/p5/v3/rca_features_gt/
```

配置：`configs/e2e/gaia_p5_v3_preprocessing_v2.json`

当前 Ada-MGAD 输入维度：

```text
node metric: [T, 10, 48]
log:         [T, 10, 32]
trace edge:  [T, 10, 10, 8]
```

10 个服务使用固定 canonical order；所有节点 tensor 维度相同，但这不表示每个服务真实拥有完全相同的原始指标。缺失/不适用指标通过冻结 schema 和 mask 语义处理，不能在 Test 上重新选 feature、scaler、词表或 graph。

P5 baseline 的核心边界（不要直接套用为 P6-C0 的选择规则）：

- Metric schema、quality/redundancy、normalization、Log vocabulary/scaling、Trace graph/statistics 均由 Train 决定；Test 只能 transform；
- Ada-MGAD checkpoint 主选择指标是 Train F1；Test 每 epoch 只观察，不参与 fit、early stopping 或 checkpoint 选择；
- Event threshold 使用 Train-only exact score sweep；
- Ada-RCA StandardScaler 只在 Train candidate rows 上拟合；
- Test label 只能用于最终评估，不得用于 preprocessing 决策或后验调参；
- 不得将 raw GT 合并成“更容易命中”的标签，也不得把诊断排序失败从分母删除。

P6-C0 复用上述冻结预处理，但采用 50/20/30：Detector-Fit 进行梯度更新，Detector-Validation 选择 checkpoint/threshold，最后 30% Test 不参与选择。共享 preprocessing 仍来自原 70% Train；因此未来 C1 的严格 forward-OOS 需要单独处理跨 fold 拟合信息，不能直接把当前 Fit/Validation 预测当作严格 OOS anchors。

## 5. 已完成结果摘要

### 5.1 Ada-MGAD 节点检测

| 指标 | Train | Test |
|---|---:|---:|
| AUC | 0.9204 | 0.9328 |
| AP | 0.7519 | 0.8199 |
| F1 | 0.7668 | 0.8693 |
| Precision | 0.7403 | 0.9692 |
| Recall | 0.7953 | 0.7881 |

`best_train_f1.pt` 对应 epoch 10；最低 Train loss 的 checkpoint 在 epoch 17，仅作为辅助诊断。Test F1 后期曾达到约 0.8746，但不能据此替换主 checkpoint。

### 5.2 Event detection

Test：GT 5,787，TP 3,703，FP 65，FN 2,084，Precision 0.9827，Recall 0.6399，F1 0.7751；平均检测延迟 24.11 秒，P95 39.51 秒。

解释：检测器很保守，误报少但漏掉约 36% 的异常事件。事件召回是 E2E 的第一主要瓶颈。

### 5.3 RCA 与 E2E

在 3,702 个成功匹配且通过 W300 边界清洗的 Test 案例上：

| 方法 | AC@1 | AC@3 | AC@5 | MRR |
|---|---:|---:|---:|---:|
| Detector-only | 0.9384 | 0.9814 | 0.9935 | 0.9621 |
| Ada-RCA，GT anchor | 0.9125 | 0.9919 | 0.9968 | 0.9525 |
| Ada-RCA，detected anchor | 0.4657 | 0.9214 | 0.9830 | 0.6989 |
| Train-only root-frequency | 0.4843 | 0.9851 | 0.9916 | 0.7368 |

完整 detected-anchor 诊断 F1：`@1=0.3611`、`@3=0.7145`、`@5=0.7623`。Detected 与 oracle 的 AC@1 差为 `-0.4468`，说明当前最大问题在 detected anchor/时间上下文与事件召回，而不是 RCA 优化器未收敛。

当前分布高度不均衡：mobservice1/2 约占匹配案例 98.2%，login_failure 约占故障类型 97.8%。因此必须同时报告 weighted overall、root macro、fault macro 和各组样本数；不能只引用整体 AC@1。

RCA 特征健康：总案例 14,045、candidate rows 140,450、68D、finite ratio 1.0、全零行 0、全 Mask 行 0、constant features 0。RCA L-BFGS-B 已收敛（551 iterations，gradient norm `2.16e-9`）。

### 5.4 最新 P6-C0 Stage-1 结果

以下来自修正版 `experiments/p6/system_event_trigger/test_metrics.json`，不是新完整 E2E：

| 指标 | Validation | Test |
|---|---:|---:|
| Event Precision | 0.9939 | 0.9962 |
| Event Recall | 0.7322 | 0.7254 |
| Event F1 | 0.8432 | 0.8395 |
| TP / FP / FN | 2124 / 13 / 777 | 4198 / 16 / 1589 |

固定 checkpoint 的 recorded epoch index 为 8，threshold 为 `0.9998264908790588`；完整 SHA 见 C0F 方案。Test mean/P95 delay 为 23.77/39.15 秒。长时事件召回 `32/229=0.1397`，memory 召回 `30/204=0.1471`；这两个分组重叠但不是同一批 229 个事件。Test multi-onset recall 为 0.4141，single-onset 为 0.7769。

总体门槛通过不改变 `BORDERLINE`。标签直接解码参考不是模型可达召回上限；晚分数响应不能独自证明 telemetry 可观测性时延；持续低分不能直接确诊 representation failure。C0F 更正审计继续保持这些解释边界。

### 5.5 P6-C0F 独立更正审计（不是新检测器实验）

当前引用目录：`experiments/p6/c0f_failure_audit/c0f-correction-20260923T0755Z/`。
`completion_manifest.json` 状态为 `COMPLETE`，SHA-256 为
`faabd0a5e6486a89c6e6e3cab1af8f146af6cdab9fe5e728c027f5d18f80745a`；
审计决定状态为 `COMPLETE_WITH_DECLARED_LIMITATIONS`。完成清单列出 184 个输出，
已在本次会话逐项复核大小与 SHA；冻结输入的前后校验为 `PASS`（564 个文件）。
独立 `check-code` 与 run 内复验均为 119 tests passed；Validation replay 为 `MATCH`，
Test 既有预测/episode/matching 复核为 `PASS`。

| split | complete GT | observed TP | U_episode（精确） | U_grid（松弛） |
|---|---:|---:|---:|---:|
| Fit | 7,443 | 5,626 | 6,841 | 7,427 |
| Validation | 2,901 | 2,124 | 2,607 | 2,890 |
| Test | 5,787 | 4,198 | 5,173 | 5,766 |

三段 witness 均经冻结 episode/matching 流程重放，`observed <= U_episode <= U_grid`。
旧 run 的 Test `decision.json` 错把标签解码召回 `0.593053...` 写为网格上界；
更正后 Test `U_grid=5766/5787=0.996371...`，标签解码参考仍单独报告。
新旧 16,131 个事件的身份、正式失败类别、候选计数与匹配时间一致；
修正了 2 个 Test 事件的 `never → censored`、13 行删失标记和 5 行并发活动计数。
旧 `1.5x` 路线选择规则已退为 `DESCRIPTIVE_ONLY`，不自动选择 C0R2 或 C1。

更正 run 复用旧 Fit/Validation 分数的固定字节，未重新运行模型；旧 run 未归档其
执行时源码字节，因此历史 Fit/Validation 推理源码仍为 `UNVERIFIED`。更正 run 绑定
自己的实际源码、测试、输入和产物，不能把这种绑定追溯赋予旧推理。Test 已参与研究设计，
本次不是独立确认；C0 原 verdict 仍为 `BORDERLINE`。

## 6. 已知限制（不要静默修复）

这些不影响本次数值已经生成，但使结果包尚未达到“无审计保留”的发布状态：

1. `rca/rca_train_manifest.json` 中 `rca_detected_predictions.csv` 和 `rca_metrics.json` 的 SHA-256 是 E2E 重写文件前的旧值；实际最终文件 hash 不同。
2. `rca/e2e_diagnosis_metrics.json` 的 detected feature hash 为 `null`，虽然实际 feature bundle 有 hash。
3. `final_report.md` 的 Pytest 字段为 `not recorded`；不要声称本 run 已把测试结果写入报告。
4. `run_manifest.json` 中若干 reusable shared input 的 config hash 与 execution config 不同，这是 legacy preprocessing binding 的 provenance 信号，不自动等于数据泄漏，但需要在正式发布前明确记录。
5. 日志中的 AdaBelief 版本提示和 NumPy non-writable warning 非致命；后者应在后续代码维护中清理，但不是本次训练失败原因。

以上为 P5 结果包限制。P6-C0 正文另有计数/表述偏差，见 [路线图第 8 节](P6_RESEARCH_ROADMAP.md#8-历史解释勘误与文档一致性)；保留历史报告，未重写原产物。后续 agent 不得把旧正文的 Fit negative=25970、purged labels=2/6/3、memory 与 long 为同一总体等表述作为真实输入。C0F 的更正结果与历史来源限制见第 5.5 节；同一 Test 已被观察，后续研究必须披露复用限制。

## 7. 新 agent 的操作规则

- 首先运行 `git status --short`，保留用户已有的 untracked `artifacts/` 和实验输出；不要 reset、删除或覆盖它们。
- 读取当前 run 的 JSON/CSV 和日志时优先使用上述独立 run 目录，不要从默认旧 `artifacts/p5/v3` 路径推断新结果。
- 新实验/审计必须创建所属阶段的唯一新目录。P5 新训练使用 `experiments/p5/gaia_v2/<run_id>`；C0F 建议使用 `experiments/p6/c0f_failure_audit/<run_id>`。不得复用已完成目录的可变输出；C0F 可以只读引用本方案锁定的 checkpoint/预测，不能覆盖它们。
- 没有用户明确授权时，不启动 full preprocessing、full training、Test evaluation 或 RCA training；先做静态检查或小型 smoke。
- 任何新 preprocessing 选择都必须在 Train-only、label-free、固定 schema 下完成；不能根据 Test F1 反向选择。
- 修改 Ada-RCA 之前先确认用户是否要改变冻结的 68D 表示；默认只修 adapter/provenance，不重设计模型。
- 结果报告要区分：检测器节点指标、事件指标、matched-case RCA 指标和包含漏检的 full diagnosis 指标。
- 原 C0F 执行包含冻结模型的 Fit/Validation 前向推理，Test 只读已有 CSV；2026-09-23 更正 run 只复用原分数并标明历史源码限制。任何新 C0F 推理都不能调用会重跑 Test 并写回原目录的 C0 `evaluate` 入口。
- C0F 原 run 与独立更正 run 均已完成，任何新计算必须用唯一新目录；不覆盖这两个完成目录。更正 run 的 `git_commit` 是 base HEAD，实际执行源码以归档快照及 source digest 绑定，不能把未提交源码说成该 commit 的内容。
- C0 原 BORDERLINE 不追溯性改为 GO。后续 C1 若开展，属于另行冻结协议、明确接受 Stage-1 限制的接口研究。

## 8. 推荐下一步（不自动执行）

1. **本分支固定锚点回溯研究已完成，建议停止在同一 Test 继续优化。** Matched-case RCA AC@1 已达 `0.8899`，完整 E2E F1@1 为 `0.7469`；改善集中在高频 mobservice1/2 的 login_failure。Stage-1 仍漏报 1,589 个事件；下一步应整理论文结果，并在独立新数据确认固定方案。P6-C1-v2/P6-C2 的正式结论和旧 Test 复用限制保持原样，见 [原结果](P6_C1_V2_AND_C2_RESULTS.md) 与 [本分支结果](P6_C1_ANCHOR_BACKDATE_RESULTS.md)。
2. C0F 更正、旧 C0F run 与原 strict-prefix C1 失败 run 继续作为独立历史记录保留；不得复用 `c1-prefix-oos-v1-seed42`，也不得把其失败当作降低任何门槛的依据。
3. C1-v2 的证据等级是 supervision-OOS（共享预处理来自原 Train 的 label-free 拟合），不是 prefix-preprocessing-OOS；若未来要主张严格 forward-OOS，需要另行冻结协议，且必须先通过独立的前缀 schema 可行性审计。
4. 已知未解决项只作为 limitation 记录，不作为新的自动实验：多 onset 事件上 C 反而低于 B（onset 2/3 分组）、memory 分组 AC@3/5 回退、C0 Stage-1 长时/内存事件召回低、Test 已被多次复用。
5. C0R2 仍是独立条件支线，不根据已看过的 Test 挑选配置；如要开展需单独冻结协议。
6. 用户另行授权的 XGB scorer 和固定 Train-delay RCA anchor 回溯已分别在独立工作树完成。回溯 Test 结果达到本轮 matched-case AC@1 超过 0.8 的目标；不继续在同一 Test 选择 offset、特征或模型。若需证明泛化优势，只能在独立数据上固定方案后验证。
7. P5 provenance 修订属于独立归档工作，不通过修改本轮实验数值解决；保留原输入/结果和更正记录。

## 9. 只读快速检查命令

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
git status --short
RUN=experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440
grep -E '"status"|"updated_at"' "$RUN/run_state.json"
sed -n '1,80p' "$RUN/final_report.md"
```

上述命令只检查 P5 baseline。P6-C0 可只读检查 `experiments/p6/system_event_trigger/test_metrics.json`、`validation_selection.json` 和 `manifest.json`。C0F 更正可只读检查 `experiments/p6/c0f_failure_audit/c0f-correction-20260923T0755Z/{completion_manifest.json,decision.json,correction_summary.json,final_report.md}`。

P5 的入口和 worker 建议见 [`GAIA_V3_IMPLEMENTATION.md`](GAIA_V3_IMPLEMENTATION.md)；不要复用该 baseline 目录运行新实验。C0F 更正执行命令已记录在新 run 的 `completion_manifest.json:commands`；不得把 C0 的 audit/train/evaluate 命令充当 C0F 命令。

P6-C1-v2 run 的只读检查（不改动任何产物）：

```bash
RUN=experiments/p6/c1_supervision_oos/c1-supervision-oos-v1-seed42
python -c "import json;print(json.load(open('$RUN/evaluation/c1_results.json'))['primary'])"
ls $RUN/{folds,train_cohort,rca,predictions,evaluation}
```
