# P6-C0R3 同告警预算对照结果

**状态：`COMPLETE_DEVELOPMENT_ONLY`；预设的独立起点信息门槛 `NO-GO`。**
固定 C0 checkpoint 分数、阈值 `0.9998264908790588`、原始 GT、30 秒
网格和精确 60 秒因果匹配。执行源码 commit
`5656e750c67cd6beb46a4dd7e8cfa913057f3972`，唯一输出
`experiments/p6/c0r3_equal_alert_control/c0r3-v1/results.json`，
协议见[P6_C0R3_EQUAL_ALERT_CONTROL_PROTOCOL.md](P6_C0R3_EQUAL_ALERT_CONTROL_PROTOCOL.md)。
原 C0 baseline 的每个 `(case_id,t_hat)` 和 C0R2 的 TP/FP/FN、告警数
均精确重放。只读 Fit/Validation，没有读取或重跑 Test。

## 同等告警预算的 64 个位置对照

每个对照保持高分段起点不变，并在**相同段长度组**内随机放置与
logit 上升方案相同数量的额外告警。Fit、Validation 的额外告警数
分别固定为 419、122；64 个 PCG64 种子预先固定且全部报告在 JSON。

| 数据段 | decoder | TP | FP | R | F1 | 干净事件召回 |
|---|---|---:|---:|---:|---:|---:|
| Fit | 原 decoder | 5,626 | 44 | 0.7559 | 0.8581 | 2,827/3,005 = 0.9408 |
| Fit | logit 上升 | 6,009 | 80 | 0.8073 | 0.8881 | 2,827/3,005 = 0.9408 |
| Fit | 同预算对照中位数 | 6,000 | 89 | 0.8061 | 0.8868 | 2,827/3,005 = 0.9408 |
| Validation | 原 decoder | 2,124 | 13 | 0.7322 | 0.8432 | 899/952 = 0.9443 |
| Validation | logit 上升 | 2,244 | 15 | 0.7735 | 0.8698 | 899/952 = 0.9443 |
| Validation | 同预算对照中位数 | 2,241 | 18 | 0.7725 | 0.8686 | 899/952 = 0.9443 |

对照 F1 的 95 分位数：Fit `0.8878`、Validation `0.8694`；
logit 上升的 F1 分别为 `0.8881`、`0.8698`，略高于这 64 个对照的
95 分位数。但其对**干净事件**的召回与原 decoder 和全部 64 个
对照完全相同。干净事件在这里仅按同一段内完整 GT 排除邻近/持续
的其他事件；跨段被 purge 的事件仍是限制。预设门槛要求 F1 与
干净召回都超过对照，因此为 `NO-GO`。结论是：大部分表观召回增益
来自增加告警预算，分数上升可能在拥挤事件中有很小的择时优势，
但没有证据表明它新增识别了独立起点。此门槛不是显著性检验；
Validation 和候选方案都已在此前被研究使用。

## 当前模型对 memory 起点的分数响应

只在干净上下文内，现有冻结分数在起点后 60 秒达到原阈值的病例：

| 数据段 | login_failure | memory_anomalies |
|---|---:|---:|
| Fit | 2,823/2,851 | 2/136 |
| Validation | 899/915 | 0/31（30 案有合法分数格） |

这说明当前模型和阈值几乎没有覆盖干净 memory 起点；不能据此认定
原始遥测没有信号，也不能把低分直接归咎于 BCE、图编码器或阈值。
另一个独立的 Train 原始输入诊断位于
`/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2-c0raw/`。

## 决策

停止把“高分段内 logit 上升再触发”作为候选正式 detector 改进；
不再在同一 GAIA Test 上试它。下一步若训练新检测器，应瞄准
**干净长时/memory 起点的分数缺口**，一次只改变一个训练机制，
并把新实验限定为 Train/Validation 开发证据。原 C0 `BORDERLINE`
和全部历史产物不变。C0F Fit/Validation 分数的历史推理源码身份
仍为 `UNVERIFIED`。
