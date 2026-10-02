# P6-C1 68D + XGBRanker 固定锚点回溯结果

**状态：完成，matched-case AC@1 达到 0.8899；证据等级为已复用 Test 的探索性对照。**
独立工作树 `experiment/p6-mob-pair`。Train/label-free Test 预测的执行源码
commit 为 `b86a8f0d282480dad837261bb44e9b9b9bb23fef`，独立更正评价源码
commit 为 `e1684e69251607cae142b2c85b1c7eb4ce9bcbd2`。方案与单次 Test
决策见 [开发与 Test 计划](P6_C1_ANCHOR_BACKDATE_DEV_AND_TEST_PLAN.md)。

## 改动和输入

保持 P6-C0 检测器、Test episode、60s matching、十服务候选、W300-B15 的
68D Z2 表示和固定的 XGBRanker 参数不变。唯一选中的方法变量是 RCA
特征窗口中心：`t_rca = t_detected - 25.621s`。回溯量为 C1-v2 Train
fold 1 的 OOS 检测延迟中位数，未由 Test 选择。Train 和 Test 都从冻结的
raw index 在该锚点重新提取特征；原始 detected 锚点的抽样特征重放与封存
数组逐值相同。XGB 无 scaler；Train 3,225 个 case、10 个候选、根因索引
和原 C1 cohort 顺序均核对一致。回溯后 Train 最晚原始特征窗口终点
`1626440344379ms` 早于 Test 边界 `1626963120000ms`；Test 首个窗口
起点 `1626963154379ms` 不早于该边界，原始窗口不跨 Train/Test。

Train-only 时间前推检验见
`experiments/p6/c1_anchor_backdate_dev/c1-anchor-backdate-forward-v1/`：

| 比较 | n | 原 detected-anchor XGB AC@1 | 固定回溯后 AC@1 | 净多正确 |
|---|---:|---:|---:|---:|
| fold 1 → fold 2 | 1,168 | 844/1168 = 0.7226 | 1044/1168 = 0.8938 | +200 |
| folds 1+2 → fold 3 | 1,122 | 864/1122 = 0.7701 | 1007/1122 = 0.8975 | +143 |

两个后折的八个 UTC 日期块净增益均为正。只移动评价锚点、保留原
detected-anchor 训练的 AC@1 为 0.5214/0.6248；Train/Test 窗口语义
必须一致。其他 Train-only 小对照未形成同等稳定收益：直接 mob1/2
二分类、事件相对特征和 GT/detected 混合训练在 fold 3 均未超过原 XGB；
仅删掉 Metric 通道为 0.7286/0.7799，增益较小。移除 Trace Latency
通道使两折明显退化。

## 同一 Test legal matched cohort 的 RCA

预测锁：
`experiments/p6/c1_anchor_backdate/c1-anchor-backdate-v1-delta25621/predictions/prediction_lock.json`。
所有 4,214 个 C0 episode 均保留，其中 4,213 个 legal 完整排序、1 个
illegal context、0 个 ranking failure；Test matching/GT 在锁生成前未读取。
旧 XGB 排序与封存 scope 逐案例相同。以下共同分母 **4,197**，完整结果在
`experiments/p6/c1_anchor_backdate_eval_correction/c1-anchor-backdate-eval-v1/`。

| 指标 | 原 detected-anchor XGB | 固定回溯 XGB | 回溯 − 原方法 |
|---|---:|---:|---:|
| AC@1 | 3093/4197 = 0.7370 | 3735/4197 = 0.8899 | +0.15297 |
| AC@3 | 4189/4197 = 0.9981 | 4191/4197 = 0.9986 | +0.00048 |
| AC@5 | 4194/4197 = 0.9993 | 4193/4197 = 0.9990 | −0.00024 |
| MRR | 0.86646 | 0.94373 | +0.07727 |

Top-1 配对状态：两者均正确 2,855、仅原 XGB 正确 238、仅回溯正确 880、
两者均错误 224，净多正确 642。按十个 UTC onset day 做固定 10,000 次
cluster bootstrap，配对差的**描述性** 95% percentile 区间为
`[+0.13598, +0.16694]`；Test 已多次复用，不能将其视为独立确认置信区间。
十个 Test 日期块的净增益均为正。延迟 10–40s 的分组净增益较大；
40–50s 的 114 案净退化 12 案，固定回溯对过晚检测并非普遍有益。

改善集中于高频群：`mobservice1` 从 1,525/2,066 到 1,868/2,066，
`mobservice2` 从 1,568/2,105 到 1,867/2,105；
`login_failure` 从 3,089/4,164 到 3,732/4,164。其余 26 个服务根因
案例两方法 Top-1 均为 0/26；`memory_anomalies` 从 4/30 降至 3/30。
因此 0.8899 主要说明主流 mob1/2 短时 login 故障的排序改善，
不能宣称解决跨服务或少数故障。

## 完整两阶段 Diagnosis

Stage 1 保持 TP 4,198 / FP 16 / FN 1,589，GT 5,787、预测 episode
4,214。完整 E2E 的 TP 是 legal matched 且根因在 Top-k 的事件；错误或
缺失排名仍计入 FP/FN。旧 XGB 的 C1 cohort 和 C2 @1/3/5 指标在本次
更正评价中精确 replay。

| k | 方法 | TP | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| @1 | 原 XGB | 3,093 | 0.7340 | 0.5345 | 0.6185 |
| @1 | 回溯 XGB | 3,735 | 0.8863 | 0.6454 | 0.7469 |
| @3 | 回溯 XGB | 4,191 | 0.9945 | 0.7242 | 0.8381 |
| @5 | 回溯 XGB | 4,193 | 0.9950 | 0.7246 | 0.8385 |

回溯方法的 failure ledger：`EVENT_MISSED=1589`、`EVENT_FALSE_ALARM=16`、
`RCA_CONTEXT_INVALID=1`、`RCA_RANKING_MISSING=0`、`ROOT_OUTSIDE_TOP1=456`
（rank 2–3）、`ROOT_OUTSIDE_TOP3=2`（rank 4–5）、`ROOT_OUTSIDE_TOP5=4`、
`SUCCESS_TOP1=3735`。Stage-1 漏报仍是最大的 E2E 失败类别；在当前固定
检测器下，即使 RCA 完美，Diagnosis Recall@1 也不超过 `4198/5787=0.7254`。
若目标是**完整 E2E Recall@1 超过 0.8**，必须改变检测阶段；本轮达到
0.8 的指标是 matched-case RCA AC@1。

## 有效性与更正记录

原 `scripts/p6/run_c1_anchor_backdate_test.py evaluate` 在读取
`test_matching.csv` 时错误地设置 `keep_default_na=False`，将含空值的
`t_hat` 列读成字符串；评价器的整数转换立即失败，原预测 run 没有生成
`evaluation/`。Train 模型、85 个 Test 特征分片和排序锁均未改动。
独立的 `scripts/p6/evaluate_c1_anchor_backdate_correction.py` 使用与旧
XGB 评价入口相同的 pandas 默认 NA 读取语义，只读锁定预测并写入新的
更正目录。该目录的 completion manifest 所列哈希、原模型/预测锁绑定、
逐案例旧 XGB 排序 replay 和独立 CSV join 的 AC@1/3/5、MRR、完整
P/R/F1 整数计数复算均通过。

本研究仍受已复用 Test、单 seed、GAIA 30s 网格、多事件重叠、类别极度
偏斜以及 supervision-OOS 共享预处理限制。固定 offset 来自本数据集
Train 延迟分布，不可当作其他系统的通用常数。当前 Test 不再用于调整
offset、阈值、特征子集或模型；若需主张泛化，应在独立数据上冻结本方案
后一次性验证。

本分支归档了脚本、Train/Test 锁、模型、完整排序、评价和更正报告；
重提的 `.npy` 特征张量与分片留在本地独立 run 目录，按仓库现有规则
不提交 Git。完整特征级重放需原始 raw index 和这些本地产物，或按
锁定源码重新物化；指标级整数计数可由已归档排序与来源 matching 复核。
