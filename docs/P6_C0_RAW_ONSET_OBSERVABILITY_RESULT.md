# P6-C0 原始输入起点变化诊断

**状态：`COMPLETE_DEVELOPMENT_ONLY`。** 固定 Train 侧预处理 Metric、Log、
Trace 张量，在干净的 Fit/Validation 故障起点前取 300 秒基线，
比较起点后 60 秒内的输入变化；与各段 512 个无事件伪起点比较。
本结果只说明预先固定的简单变化统计量是否有描述性区分度，
不等同于训练出的 detector 准确率或因果可观测性。

独立工作树 `Ada-MGAD-e2e-v2-c0raw`、分支
`experiment/p6-c0-raw-observability`；执行源码 commit
`9e1580208d858bf297da2bdc0e100787bc50158f`。协议见
[P6_C0_RAW_ONSET_OBSERVABILITY_PROTOCOL.md](P6_C0_RAW_ONSET_OBSERVABILITY_PROTOCOL.md)，
唯一输出 `experiments/p6/c0_raw_onset_observability/raw-onset-v1/`。
结果 JSON 绑定输入/源码 SHA，四个 CSV 保存每案与伪起点统计量。
不读取 Test，不训练模型，不选择任何特征或阈值。

## 样本及主要观察

| 数据段 | 干净 GT | 干净 memory（有完整输入） | 无事件候选池／固定抽样 |
|---|---:|---:|---:|
| Fit | 3,005 | 136/136 | 4,551／512 |
| Validation | 952 | 30/31 | 4,175／512 |

对所有服务、Metric 维度计算的**平均绝对变化**，与本段无事件
伪起点比较：

| 数据段 | memory 变化中位数 | 无事件中位数 | 描述性 AUC | login_failure AUC |
|---|---:|---:|---:|---:|
| Fit | 0.0213 | 0.0161 | 0.7984 | 0.6092 |
| Validation | 0.0229 | 0.0183 | 0.6795 | 0.7088 |

因此预处理后的 Metric 输入对部分干净 memory 起点有**可能可用**的
变化，而当前 C0 分数在同条件下仅 Fit 2/136、Validation 0/31
达到固定阈值。两段 AUC 差异明显，Validation memory 仅 30 案；
不能据此推断一个新分类器必然提高事件召回。

Log/Trace 对 memory 的变化更不稳定：Fit 的 mean-absolute AUC
分别为 `0.4870/0.4639`，Validation 为 `0.8285/0.8204`。
Validation 无事件参考的 Log/Trace 变化中位数均为零，说明参考池
与事件附近的活动状态可能不同；后者的较高 AUC 不能直接归因为
memory 故障。全部六个预设统计量和分组样本数保留在 `results.json`。

## 决策边界

原始输入分析为**有限支持**：可继续设计一个只改变训练机制的
Train-only memory 起点实验，但先要控制无事件参考的时段/活动差异，
明确“干净 memory”样本量和至少一个不依赖 Test 的误报预算。
当前不选择 Metric 阈值、不训练或发布新检测器。GAIA 的故障类型、
服务和持续时间高度相关；本次上下文排除仅使用各段完整 GT，
不能保证没有跨边界被 purge 的重叠事件。历史 C0F Fit/Validation
分数推理源码身份仍为 `UNVERIFIED`，原 C0 Test 和 `BORDERLINE`
verdict 不变。
