# P6-A E2E Failure Audit

## 1. Executive summary

本报告是对已完成的 GAIA P5 E2E run 的只读失败审计。审计没有重新训练、重新预处理、重新选择阈值、修改 event matching、修改 RCA 特征或修改模型；新增的 `scripts/p6/audit_e2e_failure.py` 只读取已有 artifact，并在内存中做 anchor-offset 分析。

审计对象：

```text
run:    experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440
commit: 7dea779fc5196218a43d86a9777b591d290f047b
config: configs/e2e/gaia_p5_v3_preprocessing_v2.json
```

核心结论如下。

1. `Ada-RCA` 的 GT-anchor 到 detected-anchor 的 AC@1 从 `0.9125` 降到 `0.4657`，主要是 detected anchor 改变了 W300 原始 telemetry 上下文，造成 RCA representation/ranking 对时间位置高度敏感；不是 L-BFGS 未收敛或特征数组损坏。
2. 检测延迟的“大小”本身只解释一小部分差异：delay 与 reciprocal rank 的 Pearson/Spearman 均约 `-0.13/-0.10`，30 秒前后 AC@1 只差约 4.59 个百分点。可是固定 GT anchor 向后偏移 `+15s` 的非正式模拟就把 AC@1 降到 `0.4165`，说明时间上下文位置比一个单调的 delay 大小效应更关键；offset 曲线也不是单调的，不能据此引入后验校准参数。
3. Oracle 正确的 3,378 个 Top-1 案例中，detected ranking 仍有 `3,157/3,378=93.46%` 落在 Top-3、`3,339/3,378=98.85%` 落在 Top-5，但只有 `1,689/3,378=50.00%` 保持 Top-1。也就是说，RCA 多数情况下仍找到了正确候选集合，只是排序受 anchor 影响；这与“完全找不到 root service”不同。
4. full E2E 还受到事件召回限制：Test event recall 为 `0.6399`，`2,084` 个 GT event 未匹配（RCA W300 清洗后保留 `2,079` 个 miss）。这解释 full diagnosis F1 的下降，但不能解释 matched-case 内部的 `0.4468` AC@1 差值。
5. matched Test case 高度偏斜：`mobservice1/2` 占 `3,636/3,702=98.22%`，`login_failure` 占 `3,619/3,702=97.76%`。因此 overall 是加权结果，非 mob/fault strata 只能作为小样本诊断，不应作强泛化结论。

综合判断：当前下降的首要接口问题是“`t_hat=prediction_available_time` 直接作为 RCA anchor”与 RCA 的 pre/post temporal representation 之间的敏感性；次要但影响 full E2E 的问题是检测器事件召回；类别/服务不均衡是解释范围和泛化能力的风险，而不是本次 matched AC@1 崩落的充分原因。

## 2. Evidence and scope

### 2.1 Evidence grading

报告中的结论使用以下等级：

| 等级 | 含义 |
|---|---|
| **Confirmed** | 由本次脚本对最终 artifact 重算并通过身份/数量校验的事实。 |
| **Observed** | 在固定 run 的分组或模拟结果中观察到的关系；不等于因果证明。 |
| **Hypothesis** | 需要新 protocol/run 才能验证的机制解释。 |
| **Open question** | 当前 artifact 无法回答，不能补推。 |

### 2.2 读取的 artifact 与实现位置

主要输入为：

```text
events/event_matching.csv
events/event_detection_metrics.json
rca/rca_oracle_predictions.csv
rca/rca_detected_predictions.csv
rca/detector_only_predictions.csv
rca/e2e_diagnosis_metrics.json
rca/e2e_layered_report.json
rca/rca_metrics.json
rca/detected/rca_feature_manifest.json
rca/detected/rca_feature_health.json
rca/detected_features/z2_features.npy
rca/conditional_logit.npz
```

与语义直接相关的代码位置：

- `docs/GAIA_V3_IMPLEMENTATION.md:9-11`：Train-only、`t_hat=prediction_available_time`、oracle/detected anchor 协议；
- `src/e2e/e2e_evaluation.py:22-80`：完整 ranking 解析及 detector-only ranking；
- `src/e2e/e2e_evaluation.py:190-234`：已有 delay/ranking 关系函数；本审计脚本在此基础上增加了题目要求的四个 delay bucket 及 AC@3/5；
- `src/e2e/e2e_evaluation.py:395-428`：RCA diagnosis 及边界清洗后的分层指标；
- `src/e2e/gaia_rca_adapter.py:253-295`：以 anchor 构造 metric、log、trace-error、trace-latency 四个原始通道；
- `src/e2e/rca_features.py:179-255,371-379`：每通道 8 个 base + 9 个 morphology，四通道 flatten 为 68D Z2；
- `scripts/p5/run_i1_rca_features.py:166-172,202-250,259-295`：Train 使用 GT start，Detected Test 使用匹配到的 `prediction_available_time`，并按 W300 清洗；
- `src/e2e/protocol.py:35-50,312-341`：半开时间块和 `[anchor-300s, anchor+300s)` 边界规则；
- `src/e2e/rca_model.py:332-365,433-470`：候选排序和 RCA 指标。

审计脚本在运行时校验 `run_state.status == COMPLETE`、ranking identity、matching identity、anchor/delay 一致性、feature bundle 的 `(case,10,68)` 形状和 case ID 唯一性。所有正式指标仍来自原 run；offset simulation 不写回任何正式目录。

## 3. Current run and population definitions

| population | count | definition |
|---|---:|---|
| Test matching rows | 5,852 | event matching 中的 Test rows |
| Test GT events in event metric | 5,787 | event detection artifact 的完整 Test injection population |
| GT events after RCA W300 purge | 5,781 | e2e diagnosis 的边界合法 GT population |
| matched events after purge | 3,702 | 同时存在 detected RCA ranking 的 case |
| RCA misses after purge | 2,079 | 未匹配 GT event |
| false alarms after purge | 65 | 未匹配 prediction |
| matched detected ranking rows | 3,702 | 本报告所有 oracle/detected matched-case 比较的分母 |

事件 metric 和 E2E diagnosis 的 GT 分母不同是协议设计，不是重复计数错误：event metric 在所有完整 Test injection 上计算，RCA 只保留 W300 context 不跨 Test 边界的 case。清洗掉 `1` 个 detected-anchor context crossing case 和 `5` 个 GT W300-ineligible rows。

## 4. Baseline results reproduced from artifacts

### 4.1 Detector and event layer

Ada-MGAD node-level metrics：

| split | AUC | AP | F1 | precision | recall |
|---|---:|---:|---:|---:|---:|
| Train | 0.9204 | 0.7519 | 0.7668 | 0.7403 | 0.7953 |
| Test | 0.9328 | 0.8199 | 0.8693 | 0.9692 | 0.7881 |

Event detection Test：`TP=3,703`, `FP=65`, `FN=2,084`, precision `0.9827`, recall `0.6399`, F1 `0.7751`；匹配 delay mean `24.106s`、median `24.109s`、P95 `39.514s`。

### 4.2 Matched-case RCA layers

| method | AC@1 | AC@3 | AC@5 | MRR | cases |
|---|---:|---:|---:|---:|---:|
| Detector-only at `t_hat` | 0.9384 | 0.9814 | 0.9935 | 0.9621 | 3,702 |
| Ada-RCA, GT anchor | 0.9125 | 0.9919 | 0.9968 | 0.9525 | 3,702 |
| Ada-RCA, detected anchor | 0.4657 | 0.9214 | 0.9830 | 0.6989 | 3,702 |

Detected minus oracle degradation：AC@1 `-0.4468`、AC@3 `-0.0705`、AC@5 `-0.0138`、MRR `-0.2536`。Detector-only 与 Ada-RCA 不是同一任务：前者按异常分数做异常服务定位，后者使用四通道 raw telemetry 的条件 logit 做 labelled fault service ranking；不能直接用两者的差值宣称某模型优劣。

## 5. Audit 1 — detection delay and ranking

delay 重新定义为：

```text
delay_seconds = (t_hat - gt_start_ms) / 1000
t_hat = prediction_available_time
```

### 5.1 Correlation

在 3,702 个 matched case 上，delay 范围 `0.088s–59.673s`，mean `24.1086s`，median `24.109s`；没有负 delay，也没有大于 60 秒的 case。

| relation | Pearson | Spearman |
|---|---:|---:|
| signed delay vs reciprocal rank | -0.131810 | -0.102041 |
| absolute delay vs reciprocal rank | -0.131810 | -0.102041 |

该负相关方向与“延迟越大，排名略变差”一致，但绝对值较小，不能把全部 AC@1 崩落归因于 delay 数值本身。

### 5.2 Delay buckets

| bucket | n | mean delay (s) | AC@1 | AC@3 | AC@5 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| `<0` | 0 | — | — | — | — | — |
| `0–30` | 2,530 | 18.6424 | 0.4802 | 0.9261 | 0.9874 | 0.7088 |
| `30–60` | 1,172 | 35.9085 | 0.4343 | 0.9113 | 0.9735 | 0.6776 |
| `>60` | 0 | — | — | — | — | — |

30 秒前后 AC@1 差 `0.0459`、MRR 差 `0.0311`。因此 delay 是次要的连续影响因素，而不是唯一根因。

## 6. Audit 2 — oracle-to-detected rank transitions

### 6.1 Oracle Top-1 cases

在 oracle Top-1 的 `3,378` 个 case 中：

| transition | cases | conditional rate |
|---|---:|---:|
| Top-1 → Top-1 | 1,689 | 50.00% |
| Top-1 → detected Top-3 (cumulative) | 3,157 | 93.46% |
| Top-1 → detected Top-5 (cumulative) | 3,339 | 98.85% |
| Top-1 → outside Top-5 | 39 | 1.15% |

全体 3,702 case 中 oracle Top-1 service 与 detected Top-1 service 直接相同的比例为 `0.525932`。这表明 detected anchor 通常没有把 root service 完全排除，而是显著改变了候选之间的相对分数。

### 6.2 Full rank transition matrix

单元格是 `oracle rank → detected rank` 的 case 数；列为 detected rank。

| oracle \\ detected | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1689 | 1269 | 199 | 135 | 47 | 19 | 10 | 4 | 2 | 4 | 3378 |
| 2 | 33 | 152 | 43 | 25 | 8 | 1 | 4 | 0 | 0 | 0 | 266 |
| 3 | 1 | 6 | 11 | 5 | 1 | 1 | 1 | 2 | 0 | 0 | 28 |
| 4 | 1 | 0 | 5 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 10 |
| 5 | 0 | 1 | 0 | 1 | 3 | 0 | 0 | 1 | 1 | 1 | 8 |
| 6 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 4 | 0 | 0 | 7 |
| 7 | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | 4 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 7. Audit 3 — RCA feature drift

本节直接比较同一 case 的 GT-anchor feature 与 detected-anchor feature；没有重新生成正式 feature artifact。两者都是 `(10,68)` 的 Z2 表示，feature health 还显示 `finite=1.0`、全零 row `0`、全 mask row `0`、constant feature `0`，所以“数组损坏/全零输入”没有证据支持。

### 7.1 Cosine similarity

| scope | mean | median | P05 | P95 | min |
|---|---:|---:|---:|---:|---:|
| overall 10 services × 68D | 0.8871 | 0.9584 | 0.4816 | 0.9993 | 0.0050 |
| root service 68D | 0.8901 | 0.9416 | 0.5934 | 0.9989 | -0.0375 |
| non-root services 9 × 68D | 0.8980 | 0.9646 | 0.4851 | 0.9997 | 0.0046 |

root service 的均值略低于 non-root，但差异不是数量级差异；真正明显的变化集中在幅度和 metric channel，而不是 mask 是否存在。

### 7.2 Feature groups

`magnitude/temporal/persistence/mask_channel` 是按现有 68D Z2 位置分组；`channel_*` 是四个 raw channel 的全部 17D。

| group | mean cosine | mean relative-L2 delta |
|---|---:|---:|
| magnitude | 0.8379 | 0.9836 |
| temporal | 0.9570 | 0.2734 |
| persistence | 0.8947 | 0.4462 |
| mask_channel | 0.9951 | 0.0741 |
| channel_metric | 0.8812 | 0.7921 |
| channel_log | 0.9617 | 0.2106 |
| channel_trace-error | 0.9810 | 0.1114 |
| channel_trace-latency | 0.9432 | 0.3029 |

**Observed：** detected anchor 保留了大部分可用性/mask 结构，但 magnitude 与 metric channel 的数值形态发生大幅变化。**Hypothesis：** RCA conditional-logit 依赖这些 pre/post magnitude 对比，因而对 `t_hat` 的位置变化敏感。该机制需要在新的、预先冻结的 anchor protocol 下验证，当前审计不把它写成模型因果证明。

## 8. Audit 4 — fault type and root service stratification

### 8.1 Fault type

以下 oracle/detected 指标都使用同一 `3,702` 个 matched Test case；event recall 使用全部 Test GT event，分母单独列出。

| fault type | matched n | oracle AC@1 | detected AC@1 | oracle MRR | detected MRR | event GT n | event recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| login_failure | 3,619 | 0.9251 | 0.4708 | 0.9618 | 0.7045 | 5,546 | 0.6527 |
| memory_anomalies | 71 | 0.2958 | 0.1690 | 0.5019 | 0.3967 | 204 | 0.3480 |
| file_moving | 6 | 0.8333 | 0.8333 | 0.9167 | 0.9167 | 16 | 0.3750 |
| access_permission_denied | 4 | 0.7500 | 0.5000 | 0.8333 | 0.6875 | 5 | 0.8000 |
| cpu_anomalies | 2 | 0.5000 | 0.5000 | 0.6000 | 0.6000 | 12 | 0.1667 |
| normal_memory_freed | 0 | — | — | — | — | 4 | 0.0000 |

`login_failure` 的 matched n 为 3,619，而 event recall matched count 为 3,620，是 1 个边界清洗/总体 population 差异，不是 case identity 重复。`normal_memory_freed` 没有进入 detected matched RCA 分母，不能从空样本推断 RCA 能力。

### 8.2 Root service

`small-n` 定义为 matched n `<30`；小样本行只用于定位风险，不支持稳定排序结论。

| root service | n | small-n | oracle AC@1 | detected AC@1 | oracle MRR | detected MRR |
|---|---:|:---:|---:|---:|---:|---:|
| dbservice1 | 7 | yes | 0.4286 | 0.1429 | 0.6048 | 0.4452 |
| dbservice2 | 11 | yes | 0.0909 | 0.0909 | 0.3540 | 0.3045 |
| logservice1 | 7 | yes | 0.1429 | 0.1429 | 0.3738 | 0.3000 |
| logservice2 | 4 | yes | 0.0000 | 0.0000 | 0.2440 | 0.1979 |
| mobservice1 | 1,793 | no | 0.9258 | 0.4518 | 0.9620 | 0.6919 |
| mobservice2 | 1,843 | no | 0.9246 | 0.4916 | 0.9614 | 0.7182 |
| redisservice1 | 7 | yes | 0.1429 | 0.0000 | 0.4167 | 0.2845 |
| redisservice2 | 6 | yes | 0.0000 | 0.0000 | 0.2655 | 0.2294 |
| webservice1 | 8 | yes | 0.2500 | 0.2500 | 0.4970 | 0.4635 |
| webservice2 | 16 | yes | 0.3750 | 0.1875 | 0.5756 | 0.4280 |

`mobservice1/2` 合计占 matched population `98.22%`，并且两者均呈现接近的 oracle→detected AC@1 跌落。因此整体下降不能归结为某一个 mob 实例单独异常；但非 mob 结论受样本量强烈限制。

## 9. Audit 5 — detector-only versus Ada-RCA

| track | task | AC@1 | AC@3 | AC@5 | MRR |
|---|---|---:|---:|---:|---:|
| Detector-only | `t_hat` 时按 Ada-MGAD anomaly score 排服务 | 0.9384 | 0.9814 | 0.9935 | 0.9621 |
| Ada-RCA, oracle | GT start anchor 的 raw telemetry RCA ranking | 0.9125 | 0.9919 | 0.9968 | 0.9525 |
| Ada-RCA, detected | `t_hat` anchor 的 raw telemetry RCA ranking | 0.4657 | 0.9214 | 0.9830 | 0.6989 |

Detector-only 的高分说明异常分数在 matched `t_hat` 上能把 labelled service 排到前面；Ada-RCA detected 的低 AC@1 说明跨过 anchor 接口后排序稳定性不足。这三个 track 的标签/输入语义不同，不能直接当成一个统一的分类器比较。

## 10. Audit 6 — anchor sensitivity simulation

这是只读、非正式的能力分析：对同一批 matched case 使用现有 raw index、现有已拟合 conditional-logit（不 refit），将 GT anchor 固定偏移后重新提取内存 feature。所有偏移 case 即使上下文跨边界也保留，因此不能作为新的正式 Test 指标。

| offset from GT anchor | AC@1 | AC@3 | AC@5 | MRR | W300 crossing cases |
|---:|---:|---:|---:|---:|---:|
| 0s | 0.9125 | 0.9919 | 0.9968 | 0.9525 | 0 |
| +15s | 0.4165 | 0.9155 | 0.9819 | 0.6698 | 1 |
| +30s | 0.4363 | 0.9149 | 0.9803 | 0.6808 | 1 |
| +45s | 0.4449 | 0.9198 | 0.9835 | 0.6870 | 1 |
| +60s | 0.4460 | 0.9219 | 0.9822 | 0.6878 | 1 |

**Confirmed/Observed：** 0 秒重算与 oracle artifact 完全一致；只移动 anchor 就造成约 0.47–0.50 的 AC@1，证明 RCA ranking 对时间上下文有强敏感性。曲线在 +15s 后略回升而非单调下降，所以不能把它简化为“延迟每增加一秒，排名固定减少多少”。实际 detected anchors 是每案不同的 `t_hat`，模拟曲线用于识别接口风险，不用于选择 offset、阈值或新模型参数。

## 11. Failure diagnosis and priority

### P0 — detected anchor 与 RCA temporal context 的接口敏感性（Confirmed mechanism / Hypothesis for causality）

- 证据：GT→detected AC@1 `0.9125→0.4657`；oracle Top-1 保持率只有 50%，但 Top-5 保持率 98.85%；+15s 模拟 AC@1 `0.4165`；magnitude mean cosine `0.8379`、metric channel `0.8812`，mask cosine `0.9951`。
- 判断：主要损失发生在 anchor 改变后的 raw context / magnitude representation 和候选相对排序，而不是 feature shape、全零输入或优化器失败。
- 边界：offset simulation 是性能盲的诊断，不证明某一个具体 bin 或某一类 telemetry 是唯一因果源。

### P1 — 事件召回不足（Confirmed full-E2E bottleneck）

- 证据：event recall `0.6399`，FN `2,084`；W300 后仍有 `2,079` 个 GT miss。
- 判断：它显著压低包含漏检的 diagnosis F1（例如 Diagnosis F1@1 `0.3611`），但不解释 3,702 个 matched case 内 oracle/detected AC@1 的差值。

### P1/P2 — fault/root 分布偏置（Confirmed distribution risk）

- 证据：mobservice1/2 `98.22%` matched cases，login_failure `97.76%`；小类 event recall 在 `0–0.8` 间波动，多个组 n≤16。
- 判断：weighted overall 主要是 mob/login_failure 的表现；小类结果不稳定，不能将整体结论推广到所有服务/故障类型。它是评估覆盖和泛化风险，不是已证实的 anchor drop 唯一原因。

### P2 — provenance/发布完整性限制（Confirmed bookkeeping issue）

1. `rca/rca_train_manifest.json` 中 detected prediction 与 `rca_metrics.json` 的 SHA 是 E2E 重写前的旧值，和最终文件 hash 不同。
2. `rca/e2e_diagnosis_metrics.json` 的 detected feature hash 为 `null`，虽然 `z2_features.npy` 有实际 hash。
3. `final_report.md` 的 Pytest 字段是 `not recorded`。
4. `run_manifest.json` 中部分 reusable shared input 的 config hash 与执行 config 不同；这是 legacy preprocessing binding 的 provenance 信号，当前没有证据把它等同于泄漏。

这些问题不改变本审计重算的数值，但在发布/论文归档前必须修正或明确声明。

### Not supported by evidence

- AdaBelief 版本提示、NumPy non-writable warning 是非致命 warning，不是本次 E2E 下降的数值根因。
- RCA L-BFGS-B 已收敛：551 iterations，gradient norm `2.16e-9`，因此“优化器没有训练好”不成立。
- 当前 artifact 不能证明某个 fault label 直接导致 anchor sensitivity，也不能证明调整阈值就能恢复 RCA。

## 12. Minimal recommendations

本报告不实施以下建议，也不改变当前正式 baseline：

1. 保留本 run 作为不可变 baseline，继续分开报告 event recall、matched oracle RCA、matched detected RCA 和含漏检的 full diagnosis；不要删除 miss 或用 Test F1 反调阈值。
2. 下一次若要改善接口，先在新 run protocol 中预注册并固定 anchor 语义（例如明确 `prediction_available_time` 对 W300 pre/post 的位置），再只用 Train 设计 anchor calibration/映射；不能使用本 Test offset 曲线选择一个最优偏移。
3. 任何候选方案都应先做 Train-only 的 anchor smoke 和 feature drift 检查，再用全新 run ID 运行；不修改 Ada-MGAD、Ada-RCA loss、融合网络或冻结 68D 表示作为第一步。
4. 修复 manifest hash 回写、detected feature hash 和 pytest 记录后，再生成可发布 provenance 包。
5. 由于类别分布极偏，后续报告至少保留 weighted overall、root/fault macro、各组 n 和 event recall；小样本 strata 不做强结论。

## 13. Reproduction command

命令只生成审计 JSON 和终端报告，不触碰正式 run artifact：

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

/home/zhangll24/miniconda3/envs/DAG/bin/python -u \
  scripts/p6/audit_e2e_failure.py \
  --run-dir experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440 \
  --config configs/e2e/gaia_p5_v3_preprocessing_v2.json \
  --gt-feature-root data/p5/v3/rca_features_gt \
  --index-manifest data/p5/v3/rca_raw_index/index_manifest.json \
  --workers 24 \
  --chunk-size 64 \
  --output-json /tmp/p6_e2e_failure_audit_full.json
```

`--skip-offset-simulation` 可用于只做 artifact 重算；默认会执行本报告第 10 节的非正式内存模拟。所有 preprocessing/training/Test threshold 和 RCA model artifacts 均保持不变。

## 14. Open questions

- detected anchor 的 degradation 是否主要由 metric magnitude、某个 trace statistic、log level timing，还是三者交互造成？当前 drift 分组只能定位到 representation group，不能做 label-free 的唯一因果分解。
- 是否存在不依赖 Test 标签、同时不改变 68D 表示的 Train-only anchor calibration，需另行冻结协议和新 run 验证。
- 修复上述 manifest/hash 记录后，是否需要重打包而非重算数值，属于发布治理决定，不在本审计中执行。
