# 当前两阶段方法与保存的研究资产

更新时间：2026-10-03。代码与文档统一回原工作树 `Ada-MGAD-e2e-v2` 的 `e2e-v2` 分支。
本文件说明当前可运行的方法，不把历史开发候选或失败实验当成正式改善。
当前状态：**用户已确认Scientific Freeze v1**；代码、配置、同一冻结scorer和全部seed1–10结果封存。
见[正式冻结记录](P6_TWO_STAGE_SCIENTIFIC_FREEZE_V1_20261003.md)；暂停当前方法搜索，保留历史NO-GO与限制。

## 1. 检测阶段

输入是冻结的 GAIA 多模态序列：每个 30s bin 的 Metric `[10,48]`、Log
`[10,32]`、Trace `[10,10,8]`。一个窗口取最近 10 个已完成 bin，共 300s。
复用原 70% Train 拟合的预处理；本次不重建 schema、词表、scaler 或静态图。

检测器是监督的 **WindowCausalTCNTrigger**，不加载 Ada-MGAD checkpoint。
多模态 embedding、逐窗口动态图、入向/出向 Trace 汇聚后，使用服务间共享的
小型因果 TCN：hidden 32、kernel 3、dilation 1/2/4、每块两层卷积、dropout .2、
逐时刻 LayerNorm、仅左侧 padding。最后时刻的十个服务表示经过
mean/max/std pooling 和系统级 head，输出一个起点分数。

监督目标是 `[t−30s,t)` 内是否存在事件起点；保留原 recent-onset60 的 IGNORE
loss mask。损失是带 Fit-only pos_weight 的 masked BCE，加原动态图稀疏项。
不使用 root service 标签、节点异常标签或 fault-type 权重。

时间划分为 Detector-Fit / Detector-Validation / Test = 50% / 20% / 30%。
窗口必须完全在所属 block 内，GT 必须完整属于一个 block。
使用 AdaBelief，lr .001、weight decay .0005；StepLR 每 10 epoch 乘 .5；
batch 32；**max_epochs 30、patience 8**。正常 early stopping 可能在 30 epoch 前停止。

checkpoint 和 threshold 沿用原 **merged Validation event F1 → recall → threshold**
选择规则。接入时使用 **独立 bin decoder**：每个分数 ≥ 冻结 threshold 的 bin
都生成一个候选，`t_hat = bin end`，不合并相邻阳性 bin。新增重复告警完整计 FP。
本次不改为 bin 专用 checkpoint/threshold selector。

入口：[十次重复协议](P6_TWO_STAGE_REPEATS_PROTOCOL.md)、
[检测器配置](../configs/e2e/gaia_p6_two_stage_detector_v1.json)、
[TCN 实现](../src/e2e/system_trigger_tcn.py)、
[训练入口](../scripts/p6/repeat_detector.py)。

## 2. RCA 阶段

检测时间经过固定 Train-derived backdate：`anchor = t_hat − 25621ms`。
每个预测候选提取十个服务各自的 **68D Z2**：Metric、Log、Trace-error、
Trace-latency 四通道，每通道 8 个幅度/起点/持续性/覆盖统计，加 9 个形态特征。
窗口是 anchor 前后各 300s，15s 分箱，共 40 bin。
特征内部的 robust deviation 使用每个窗口的 pre-anchor 观测；没有新拟合全局 scaler。

使用已冻结 **XGBRanker**：rank:pairwise，200 trees，depth 3、lr .05、lambda 1、
hist、训练 seed 20260826。每个事件的十个服务构成一个 ranking group。
所有重复实验共用同一个 scorer，SHA256
`578b1baf20ff9958a990b95b724a4f8afb83aaa801ce0762ccc56a59d25f0af2`。
它来自原 C1 的 3,225 个共同 Train case（supervision-OOS），不是针对新 detector seed
重新生成 OOS anchors、重训 RCA 的结果。

预测时只传 detected anchor 和 telemetry，不传 GT 时间、服务或匹配关系。
只有 anchor ±300s 完全属于 Test block 的候选有合法上下文。
边界无效候选仍保留在完整 E2E 分母中。RCA 需要 post-anchor 数据，不能声称零延迟在线诊断。

入口：[RCA 配置](../configs/e2e/gaia_p6_two_stage_repeats_v1.json)、
[特征提取](../src/e2e/c1_common_cohort.py)、[接入与评价](../scripts/p6/repeat_rca.py)。

## 3. 评价与结论边界

沿用 causal maximum-cardinality / minimum-delay 一对一匹配，容差 60s。
分别报告：检测 P/R/F1 和 TP/FP/FN；matched legal RCA 的 AC@1/3/5、MRR 与 n；
完整 GT 5,787 的 Diagnosis P/R/F1@1/3/5；detector 和 diagnosis failure ledger；
服务、故障类型、持续时间、起点重叠分层。

**已完成的 seed 1–10 是冻结 scorer transfer 的初始化稳定性实验。**
它不追溯修改旧 TCN replication 和训练预算实验的 NO-GO。
所有十个结果报告均值、样本标准差和范围，不选最佳 seed。GAIA Test 已反复使用，
这组标准差只描述固定数据上的初始化变化，不能当成独立数据确认或 SOTA 显著性。
70% Train preprocessing 含 Detector-Validation，旧 C1 batch graph 存在回顾性限制，
Trace parent availability 仍未证明；这些限制随冻结 scorer 继续声明。

## 4. 已整合资产

21 个 P6 派生工作树的 HEAD 均已作为当前分支祖先保存。历史 baseline、
逐窗口图修复、onset30、TCN、独立 bin decoder、68D XGB、anchor backdate、
Metric drift、flat XGB、正常预测、raw observability/mask、复制实验、训练预算、
完整 Test 比较、审计与 failure analysis 的代码/config/docs 均保留。
冲突时保留当前主路径；不同版本另存于
[source_variants](p6_integration_20261002/source_variants/)，历史源码也可按原 commit 检出。
旧 exact-source/AST replay 的拒绝条件未放宽；统一入口使用独立的新协议。

按用户后续要求，本次只合并代码、文档和数据来源索引，**不复制大缓存、
旧 detector checkpoint、原始分数、逐 case 特征或历史图表**。这些数据仍在原工作树。
索引记录 21 个工作树的 4,274 个文件条目及 1,083 个独立内容哈希；
原正式状态仍以每个 run 自己的 completion/status 为准。

为使十次重复命令可以从原主工作树接入 scorer，额外保留唯一必需的小模型包：

```text
assets/p6/frozen_rca/backdate25621-v1/
  model_state.ubj                       # 233,731 bytes
  train_completion_manifest.json        # 原字节
  original_run_lock.json                # 原字节
  ORIGIN.json                           # 来源与哈希
```

之前因按“代码、数据、文档”范围复制了约 4.13 GB 产物，其中两个 flat-XGB
缓存约 3.60 GB。用户收窄范围后，已核对所有独立内容的原文件仍存在且 SHA 相同，
撤掉这次新增的全部大副本；没有删除任何原实验文件。共享 preprocessing 留在原工作树。
未封存的 thesis-freeze 草稿不进入本次整合。不删除原工作树、不重写历史绝对路径。

可核查 [归档清单](p6_integration_20261002/asset_catalog.csv)、
[来源与哈希](p6_integration_20261002/asset_inventory.json)、
[原主分支未提交文档的原字节备份](p6_integration_20261002/preexisting_main.json)。
Git 保存代码、协议、catalog 和小 scorer 包；大产物仍在本机原工作树。
删除派生工作树前必须先单独处理这些本地数据；来源索引不是数据备份。
