# P6-C1 68D RCA 时间前推 Train 开发检验

**状态：`COMPLETE_DEVELOPMENT_ONLY`。** 固定 68D 特征、病例、十服务候选顺序、
Conditional Logit 和 XGBRanker 参数后，XGB 在两个时间前推 Train 对照中均提高
detected-anchor AC@1。增益从 fold 2 的 17.72 个百分点降至 fold 3 的
3.12 个百分点。结果支持将 XGB 保留为下一份独立数据的候选评分器，
不构成在已复用 Test 上继续选型或调参的依据。

## 来源和边界

- 协议：[P6_C1_XGB_FORWARD_DEV_PROTOCOL.md](P6_C1_XGB_FORWARD_DEV_PROTOCOL.md)。
- 独立工作树 `Ada-MGAD-e2e-v2-xgb`，分支 `experiment/p6-z2-xgb`，
  执行源码 commit `8ed6d40ba42a6d52735e5df9a486bfbb8ddd699a`。
- 唯一 run：`experiments/p6/c1_xgb_forward_dev/c1-xgb-forward-v1/`。
  `run_lock.json` 绑定源码、环境、原始 Train cohort 的 SHA、排序文件和
  两个前推切分各自的模型状态；`results.json` 绑定预测锁。
- 只使用 C1-v2 已封存的 3,225 个 **Train** case。按前折训练、后折评价，
  两评分器共享较早折的病例及标签；CL scaler 只在较早折 GT-anchor
  候选行拟合，XGB 不用 scaler。后折标签仅在独立 `evaluate` 阶段解码。
  训练阶段对标签文件做整文件原始字节哈希，因此标签边界是代码使用
  边界，不是物理隔离。未读取、重跑或重评价 Test。
- 内部时间折仍共用原 70% Train 的无标签预处理 schema，因此不是严格
  prefix-preprocessing-OOS。此前 Test 已被看过，本检验是后验开发研究，
  不是独立确认。

## 全量配对结果

| 后折（训练前折） | n | CL AC@1/3/5 | CL MRR | XGB AC@1/3/5 | XGB MRR | AC@1 差值 |
|---|---:|---|---:|---|---:|---:|
| fold 2（fold 1） | 1,168 | 0.5454 / 0.8964 / 0.9983 | 0.7182 | 0.7226 / 0.9914 / 0.9991 | 0.8557 | +0.1772 |
| fold 3（fold 1+2） | 1,122 | 0.7389 / 0.9911 / 0.9955 | 0.8661 | 0.7701 / 0.9955 / 1.0000 | 0.8836 | +0.0312 |
| 合并，仅两后折 | 2,290 | 0.6402 / 0.9428 / 0.9969 | 0.7906 | 0.7459 / 0.9934 / 0.9996 | 0.8694 | +0.1057 |

fold 2 Top-1：两者正确 536、仅 CL 101、仅 XGB 308、两者错误 223；
fold 3 分别为 740、89、124、169。合并净多正确 242 案，但两次前推
增益差异较大，不能把合并值当作稳定效应大小。

## 结构性检查

合并后 `login_failure` 为 2,275/2,290 案；XGB 与 CL 的 Top-1
分别为 1,704/2,275 和 1,464/2,275。`memory_anomalies` 只有
15 案，分别为 4/15 和 2/15，无法支持该故障类型的泛化结论。
两大服务分别是 `mobservice1` 1,113 案（XGB 812、CL 697）和
`mobservice2` 1,167 案（XGB 895、CL 769）；其余服务每类仅 1–2 案。
在增益较小的 fold 3，两大服务仍分别净多正确 13 和 22 案；
四个 UTC 日期块的净增益为 4、13、12、6 案。这些是**后验描述**，
日期只有四块，不能当独立统计检验。

## 决策

预先写下的“两折 AC@1 均为正、增益不只来自少数类”最低开发门槛通过；
它说明固定 XGB scorer 在主要的短时 login_failure / mobservice 数据上
比 CL 更适合 detected-anchor 68D 表示。fold 3 的效应弱得多，稀少故障
和服务基本没有证据。当前不再用同一 GAIA Test 选模型或微调 XGB。
若未来取得独立时间段或数据集，可预先冻结 XGB/CL 对照后只做一次检验。

检测阶段另一个独立工作树 `Ada-MGAD-e2e-v2-c0r2` 的
`docs/P6_C0R2_TRAINVAL_DEV_RESULT.md` 显示，固定分数的再触发规则
可以在开发集提高召回，但增益高度集中于重叠的短时 login_failure，
不能视为新的独立事件识别能力。当前没有据此重训检测器的依据。
