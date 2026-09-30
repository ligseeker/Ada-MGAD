# P6-C0R2 检测阶段开发集敏感性结果

**结论：固定分数下更改 episode 输出可提高 Fit/Validation 的事件召回；
该结果是开发集、后验选择的协议敏感性，尚不足以重训或发布新检测器。**

独立工作树 `Ada-MGAD-e2e-v2-c0r2`，分支
`experiment/p6-c0r2-trainval`。执行源码 commit `72c1869`，
结果 `experiments/p6/c0r2_trainval_dev/positive-logit-rise-v2/results.json`。
协议、输入 SHA 和证据等级见 [协议](P6_C0R2_TRAINVAL_DEV_PROTOCOL.md)。
原 P6-C0、C0F 与 Test 产物未被修改或重新评价。

## 单变量干预

保留 C0 已有 Fit/Validation score、checkpoint 对应的固定阈值
`0.9998264908790588`、30 秒网格、原始 GT、60 秒因果一对一精确匹配。
原 decoder 把连续高分 bin 合成一个 episode，仅在第一个 bin 触发。
候选仍在首个 bin 触发；同一高分段内，只有当前 logit **严格高于**
前一 bin 时才再触发一次。不使用标签来确定触发点，也不重训模型。

运行器在 Fit 和 Validation 上都先重放原 decoder，逐案
`(case_id, t_hat)` 与 C0F 封存 matching **完全一致**，再评价候选。
所有输入和分析源码都按字节绑定；新结果状态为
`COMPLETE_DEVELOPMENT_ONLY`，`test_read=false`。

| 开发段 | decoder | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Fit | 原 | 5626 | 44 | 1817 | 0.9922 | 0.7559 | 0.8581 |
| Fit | 正向 logit 再触发 | 6009 | 80 | 1434 | 0.9869 | 0.8073 | 0.8881 |
| Validation | 原 | 2124 | 13 | 777 | 0.9939 | 0.7322 | 0.8432 |
| Validation | 正向 logit 再触发 | 2244 | 15 | 657 | 0.9934 | 0.7735 | 0.8698 |

候选在 Fit/Validation 分别增加 419/122 个触发点，净多匹配
383/120 个 GT case；没有丢掉原有已匹配 case。新增 Validation 命中
中 **119/120 是 login_failure**，仅 **1/120 是 memory_anomalies**。
进一步只读语义诊断显示，新增 Validation 命中的 120 案里 104 案在
GT onset 前已有 active episode，103 案在 60 秒邻域内还有其他 onset，
24 案与另一 onset 落在同一 30 秒 bin。这些计数说明主要收益集中在
重叠注入，而不证明分数上升确实识别了独立的第二个故障。

## 科学决策

原 detector 的 episode 合并规则确实影响观察到的召回；在开发数据
上允许有限再触发可以提高事件 F1。与此同时，候选几乎没有改善长时
或 memory 事件，且新增命中高度依赖重叠注入的匹配。当前证据支持
**继续研究事件表示和触发语义**，不支持把此候选直接写成新的正式
Test 改进，也不支持仅凭这一点重训更复杂的 detector。

本轮在冻结协议之前已查看候选的 Validation 聚合结果；因此上述
Validation 表只能作为开发诊断，不能当作预注册确认。沿用的 C0F
Fit/Validation score 字节已封存，但其历史推理执行源码身份仍为
`UNVERIFIED`。若未来要正式改检测器，应先定义独立 onset 信号或
持续异常目标、固定评估和误报预算，并获取未参与本轮选择的时间段
或数据集；当前已看过的 Test 不用于选规则或验证收益。
