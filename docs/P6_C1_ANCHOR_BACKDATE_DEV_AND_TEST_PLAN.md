# P6-C1 RCA 锚点回溯：开发检验与一次 Test 方案

## 研究问题

P6-C1 68D + XGBRanker 在 detected-anchor Test 的 matched-case AC@1 为
`3093/4197 = 0.7370`。Train fold 1 的 OOS 检测时间比故障起点晚的中位数为
`25.621s`。W300-B15 特征的前 20 个 bin 用于基线估计，约 25 秒的延迟可把
短时异常放到 anchor 前。假设：**仅把 RCA 特征窗口中心设为
`detected_anchor_ms - 25621`，并用同样窗口训练与推理，能改善主服务之间的
Top-1 排序。** 检测器、episode、matching、候选服务、68D 算法、XGB 参数和
评价分母保持不变。

## Train-only 开发结果与 Test 决策

独立工作树 `experiment/p6-mob-pair` 的
`experiments/p6/c1_anchor_backdate_dev/c1-anchor-backdate-forward-v1/` 只使用
封存的 3,225 个 Train case 和只读 raw index。`δ=25.621s` 只来自 fold 1
OOS 延迟的中位数；fold 2/3 根因标签没有用于选 δ。每个新特征以原始索引
重新计算，抽样的原锚点特征逐值重放成功；65 个 Train 分片和最终张量已
记录 SHA-256。Train 时间前推结果如下：

| 前推比较 | n | 原 detected + XGB | 回溯 Train/评价 + XGB | 配对净多正确 |
|---|---:|---:|---:|---:|
| fold 1 → fold 2 | 1168 | 844/1168 = 0.7226 | 1044/1168 = 0.8938 | +200 |
| folds 1+2 → fold 3 | 1122 | 864/1122 = 0.7701 | 1007/1122 = 0.8975 | +143 |

分别有 `277/77` 和 `214/71` 个旧错误修正/新错误。只在评价阶段回溯、
仍用原 detected 特征训练的 AC@1 为 `0.5214/0.6248`，说明训练与
推理锚点语义必须一致。GT 特征训练、回溯特征评价为 `0.8365/0.8360`。
这些比较是已观察 Test 后的开发研究；共享 Train 预处理并非严格
prefix-preprocessing-OOS。

既定 Test GO 条件为：固定单一 δ、两次前推均超过各自原 XGB，并且
fold 3 超过 0.8；当前已满足。现只运行一个固定候选的 Test 评价，
不使用 Test 结果改变 δ、阈值、模型或特征子集。若 Test 未达 0.8，
如实报告并停止在同一 Test 上调参。

## Test 执行边界

新目录：`experiments/p6/c1_anchor_backdate/c1-anchor-backdate-v1-delta25621/`。
`scripts/p6/run_c1_anchor_backdate_test.py` 分三步：

1. `train`：在全部 3,225 个 Train 回溯特征上拟合原固定 XGBRanker；绑定
   Train 开发结果、源码和模型 SHA。
2. `predict`：读取原 C0 的 4,214 个 label-free Test episode 和既有 XGB
   排序；对原 4,213 个 legal episode 用固定 δ 重提 68D，另 1 个非法
   context 保持原状态。锁定完整排序后才允许读取 Test matching/GT。
3. `evaluate`：复用原 C1/C2 评价器，同一 4,197 个 legal matched case
   计算 AC@1/3/5、MRR 和配对迁移；对 5,787 个 GT 事件计算完整诊断
   P/R/F1@1/3/5 和失败账本。旧 XGB 指标必须精确 replay。

本次 Test 是已复用数据上的**探索性描述**，不构成独立泛化确认。
尤其不把 matched-case RCA AC@1 当成完整 E2E 召回。若需论文主张
新数据上的改进，必须对独立时间段或数据集固定此方案后再次验证。
