# GAIA P5 当前实验上下文

> 本文件是给新的 code-agent CLI 的单一入口。它只保留当前有效的背景、状态、约束和下一步；完整审计与实现细节继续放在现有文档中，不在这里复制。

更新时间：2026-09-16（Asia/Shanghai）

## 1. 项目身份与研究边界

- 工作目录：`/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2`
- 当前分支：`e2e-v2`
- 当前 HEAD：`7dea779fc5196218a43d86a9777b591d290f047b`
- 任务：在 GAIA/MicroSS 上运行 Ada-MGAD 多模态异常检测，并通过冻结的 Ada-RCA 适配器完成两阶段事件检测与故障服务定位。
- Ada-MGAD 和 Ada-RCA 的核心模型、损失和 68D Z2 表示不在本轮重新设计；本仓库主要负责 GAIA 适配、预处理、事件协议、编排、指标和 provenance。
- RCA 的 canonical source commit：`a2c620922e7c0ab3615d34654d4a3690d1b22c8e`。不要把本仓库的 E2E adapter 结果表述为独立因果根因证明；标签语义是 labelled injected/fault service。

## 2. 先读哪些文件

按以下顺序读取即可，不要从旧文档的历史状态重新推断当前进度：

1. 本文件；
2. 当前 run 的 [`final_report.md`](../experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440/final_report.md)；
3. [`GAIA_V3_IMPLEMENTATION.md`](GAIA_V3_IMPLEMENTATION.md)：当前入口、目录和分阶段命令；
4. [`P5_GAIA_E2E_PROTOCOL_V2_AUDIT.md`](P5_GAIA_E2E_PROTOCOL_V2_AUDIT.md)：协议、标签和边界约束；
5. [`GAIA_MULTIMODAL_PREPROCESSING_AUDIT_V1.md`](GAIA_MULTIMODAL_PREPROCESSING_AUDIT_V1.md) 与 [`GAIA_MULTIMODAL_PREPROCESSING_CHANGE_PLAN_V1.md`](GAIA_MULTIMODAL_PREPROCESSING_CHANGE_PLAN_V1.md)：预处理决策依据。

`GAIA_PREPROCESSING_V2_ALIGNMENT_REVIEW.md` 和早期 runbook 中出现的“尚未正式运行/NO-GO”是运行前的历史快照；不要删除历史证据，也不要把这些旧状态当成当前状态。

## 3. 当前正式 run 状态

当前唯一需要引用的完整实验目录是：

```text
experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440/
```

状态证据：

- `run_state.json`: `COMPLETE`；
- `final_report.md`: `FORMAL_FULL_DATA_COMPLETE`；
- required artifact 均已生成；
- 当前没有该 run 的活动训练/评估进程；
- 运行是从之前被杀掉的进程续跑完成的，不是一次连续执行。

阶段结果：

| 阶段 | 状态 | 备注 |
|---|---|---|
| Ada-MGAD train/inference | 完成 | 20 epoch，主 checkpoint 按 Train F1 选择 |
| Event detection | 完成 | 阈值只由 Train 选择 |
| Detected-anchor RCA features | 完成 | 68D，使用过已校验的 shard cache |
| Ada-RCA training/inference | 完成 | Train-only scaler，优化器收敛 |
| E2E evaluation/finalization | 完成 | 已输出四层比较和完整 failure semantics |

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

核心边界：

- Metric schema、quality/redundancy、normalization、Log vocabulary/scaling、Trace graph/statistics 均由 Train 决定；Test 只能 transform；
- Ada-MGAD checkpoint 主选择指标是 Train F1；Test 每 epoch 只观察，不参与 fit、early stopping 或 checkpoint 选择；
- Event threshold 使用 Train-only exact score sweep；
- Ada-RCA StandardScaler 只在 Train candidate rows 上拟合；
- Test label 只能用于最终评估，不得用于 preprocessing 决策或后验调参；
- 不得将 raw GT 合并成“更容易命中”的标签，也不得把诊断排序失败从分母删除。

## 5. 当前结果摘要

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

## 6. 当前已知限制（不要静默修复）

这些不影响本次数值已经生成，但使结果包尚未达到“无审计保留”的发布状态：

1. `rca/rca_train_manifest.json` 中 `rca_detected_predictions.csv` 和 `rca_metrics.json` 的 SHA-256 是 E2E 重写文件前的旧值；实际最终文件 hash 不同。
2. `rca/e2e_diagnosis_metrics.json` 的 detected feature hash 为 `null`，虽然实际 feature bundle 有 hash。
3. `final_report.md` 的 Pytest 字段为 `not recorded`；不要声称本 run 已把测试结果写入报告。
4. `run_manifest.json` 中若干 reusable shared input 的 config hash 与 execution config 不同，这是 legacy preprocessing binding 的 provenance 信号，不自动等于数据泄漏，但需要在正式发布前明确记录。
5. 日志中的 AdaBelief 版本提示和 NumPy non-writable warning 非致命；后者应在后续代码维护中清理，但不是本次训练失败原因。

## 7. 新 agent 的操作规则

- 首先运行 `git status --short`，保留用户已有的 untracked `artifacts/` 和实验输出；不要 reset、删除或覆盖它们。
- 读取当前 run 的 JSON/CSV 和日志时优先使用上述独立 run 目录，不要从默认旧 `artifacts/p5/v3` 路径推断新结果。
- 新实验必须创建新的唯一 `experiments/p5/gaia_v2/<run_id>`；已完成 run 不得复用 checkpoint、calibration、prediction 或日志。
- 没有用户明确授权时，不启动 full preprocessing、full training、Test evaluation 或 RCA training；先做静态检查或小型 smoke。
- 任何新 preprocessing 选择都必须在 Train-only、label-free、固定 schema 下完成；不能根据 Test F1 反向选择。
- 修改 Ada-RCA 之前先确认用户是否要改变冻结的 68D 表示；默认只修 adapter/provenance，不重设计模型。
- 结果报告要区分：检测器节点指标、事件指标、matched-case RCA 指标和包含漏检的 full diagnosis 指标。

## 8. 推荐下一步（不自动执行）

1. 保留当前 run 作为 baseline；修复 manifest/hash 回写和测试记录后再生成干净的 provenance 包。
2. 对 event recall、detected-vs-oracle anchor、delay band、channel/mask coverage、service/fault macro 做定向诊断；这不需要重新训练。
3. 若确实要改阈值/事件聚合或 RCA anchor 处理，所有决策仍只使用 Train，并用新的 run ID 做隔离实验。
4. 只有在 schema、代码 smoke、manifest 和运行命令冻结后，才提交新的 full run。

## 9. 只读快速检查命令

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
git status --short
RUN=experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440
grep -E '"status"|"updated_at"' "$RUN/run_state.json"
sed -n '1,80p' "$RUN/final_report.md"
```

新的正式 run 命令和 worker 建议以 [`GAIA_V3_IMPLEMENTATION.md`](GAIA_V3_IMPLEMENTATION.md) 为准；不要把该 run 目录直接填回命令中。
