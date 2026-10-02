# GAIA P5/P6 当前研究入口

更新时间：2026-10-02（Asia/Shanghai）。本文件是当前有效状态的单一入口。
历史各分支入口的原字节保存在 `docs/p6_integration_20261002/source_variants/`，
不要把其中 RUNNING、待执行或旧主分支 HEAD 当成现在的状态。

## 1. 当前任务与仓库

原工作树：`/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2`，原主分支：`e2e-v2`。
本次从 21 个 P6 派生分支保留 Git merge ancestry、代码/config/docs 和来源索引，
统一入口在集成工作树开发后合回原主分支。实际最新 HEAD 需现场 `git rev-parse HEAD`。
原主分支已有两份未提交文档已按原字节保存并提交，见 `preexisting_main.json`。

用户最新收窄保存范围：**只合并代码、文档和数据来源索引，不额外复制大数据**。
旧工作树的 checkpoint、分数、特征仍在原处；清点 4,274 个文件/1,083 个独立哈希。
唯一额外的小模型包为 `assets/p6/frozen_rca/backdate25621-v1/`，
模型 233,731 bytes，来源 metadata 同包保存。先前新复制的约 4.13 GB 副本已撤掉，
所有原实验文件未删除。数据索引不是数据备份，不能直接删除派生工作树。

当前用户授权：整理完成后提供 **seed 1–10 的完整手动重复命令**。
用户已选定：每个 seed 重训 detector，接入同一个现有冻结 68D＋XGBoost RCA。
**本整理任务不运行训练、Test 推理、Test 评价或全量 preprocessing。**
只执行环境/metadata 预检、源码检查和合成单元测试。不新增图或论文正文。

先读：

1. [核心方法与资产](P6_TWO_STAGE_CORE_METHOD.md)；
2. [十次重复协议与完整手动命令](P6_TWO_STAGE_REPEATS_PROTOCOL.md)；
3. [重复配置](../configs/e2e/gaia_p6_two_stage_repeats_v1.json)；
4. [统一编排入口](../scripts/p6/run_two_stage_repeats.py)；
5. [来源/哈希索引](p6_integration_20261002/asset_inventory.json)。

## 2. 当前两阶段接入方案

Stage 1：监督 WindowCausalTCNTrigger，最近十个已完成 30s bin 的 Metric/Log/Trace，
逐窗口动态图、共享小型因果 TCN、十服务 mean/max/std pooling、system head。
目标为完成 bin `[t−30s,t)` 的事件起点；保留 recent-onset60 IGNORE mask。
Fit/Validation/Test = 50/20/30；全历史窗口和完整 GT 必须在所属 block 内。
原 AdaBelief/StepLR/损失/batch 固定；**max_epochs=30、patience=8**。
checkpoint/tau 沿用 merged Validation event-F1/recall/tau 选择；
接入 decoder 是独立 bin，沿用上述冻结 tau，不使用 bin 专用 selector。

Stage 2：`anchor=t_hat−25621ms`（固定 Train-derived），W300-B15 的四通道
每服务 68D Z2，使用冻结 XGBRanker（pairwise、200/depth3/lr.05、seed20260826）。
共同 Train cohort 3,225，模型 SHA
`578b1baf20ff9958a990b95b724a4f8afb83aaa801ce0762ccc56a59d25f0af2`。
每个 detector seed 使用同一 scorer，不生成新 seed 原生 Fit-OOS RCA supervision。

预测语义标签隔离：Train 阶段只使用 Fit/Validation annotation；global registry
interval/domain 列用于 split 路由，文件 SHA 是 opaque provenance，不等同语义标签 join。
Test 输入专用 dataset 无 label API；RCA 只接 predicted anchor/telemetry。
全十个 detector/RCA stage seal + global prediction lock 都完成并提交 Git 后才读取 Test GT。
报告检测、matched legal RCA 的 n/AC/MRR、完整 GT5787 的 Diagnosis@1/3/5 和 failure/strata。
全部十个 seed 报告均值/样本标准差/范围，不挑最佳 seed。

## 3. 已完成研究，保持原结论

- P5 完整 E2E baseline 完成：`experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440/`。
  [P5/V3实现](GAIA_V3_IMPLEMENTATION.md)、[旧完整入口快照](p6_integration_20261002/source_variants/pre_consolidation_context.md)。
- P6-C0 原 system trigger 完成，BORDERLINE；C0F 更正已完成且声明限制。
- C1 strict-prefix run `c1-prefix-oos-v1-seed42` = PREFIX_SCHEMA_NO_GO，fold1 Metric 27/45，
  不得通过 padding、放宽门槛或复用失败目录继续。
- C1-v2/C2 supervision-OOS 完成：三折、共同 Train3,225；C-B AC@1 +.19657，
  原 C2 full F1@1 B .4180 / C .5829。见 [正式结果](P6_C1_V2_AND_C2_RESULTS.md)。
  这不是 strict prefix preprocessing-OOS，旧动态图还须按下列 causal 更正限制。
- 逐窗口图修复、flat-XGB、onset30、Metric drift、正常预测等候选及失败记录均保留；
  不能从代码已经合入推断它们效果 GO。raw-observability mask 全量 preprocessing 仍未执行。
- TCN seed42 开发时固定 bin Val2577/14/324，P/R/F1 .994597/.888314/.938456；
  原 seed17/2026 复制结论 STABILITY_NO_GO，保持不变。
- 先前完整 Test 比较含19个 detector arms、40个 RCA arms，已完成并审计；
  原 TCN42/17/2026 bin detector TP/FP/FN 为5094/16/693、5188/89/599、4828/124/959；
  backdate transfer full F1@1 为.839497/.842191/.816463。
  这是 reused-Test exploratory 比较，不使用其指标选择新的 threshold/offset/hyperparameter。
- 三 seed 30epoch 训练预算 prefix 诊断已完整封存：late recovery 均False，
  budget/selector all-seed gate 均False，无新的 native E2E。
  [最终复盘](P6_TRAINING_BUDGET_FINAL_RESULTS_20261002.md)保留 Scientific Freeze 建议。
  当前十次重复是用户另行授权的 transfer 稳定性观察，不追溯改成 GO。

历史产物仍按原文档列出的原工作树路径读取，且以 completion/status/hash 为证据。
对 partial、failed replay、cancelled pair 等只作实施诊断，不认定为完成的效果实验。

## 4. 输入与可信度边界

共享只读输入：

```text
data/p5/v3_preprocessing_v2/ad/{train,test}/
artifacts/p5/v3_preprocessing_v2/ad/ad_data_manifest.json
artifacts/p5/v3/protocol/gt_event_registry.csv
data/p5/v3/rca_raw_index/index_manifest.json
```

基配置：`configs/e2e/gaia_p5_v3_preprocessing_v2.json`。
Metric `[T,10,48]`、Log `[T,10,32]`、Trace `[T,10,10,8]`，固定十服务顺序。
共享预处理仅由原70% Train拟合，包含 Detector-Validation；新重复不宣称严格 preprocessing-OOS。

旧 P5/C0/C1-v2 graph 沿 batch 求均值，batch32 最早窗口可受到之后930s输入影响。
[causality 更正](P6_BATCH_GRAPH_CAUSALITY_CORRECTION.md)继续有效；历史数值算术可保留，
时序/OOS声明受限为 batch-conditioned retrospective。新 TCN 使用 window graph并核 batch independence，
冻结旧 RCA scorer 的训练 anchors 仍继承原范围。Trace parent availability 未证明。
RCA ±300s 使用 post-anchor context，不能称零延迟在线。GAIA 30s、多事件重叠、服务/故障偏斜
与固定完整GT分母继续声明。Test已多次使用；十次seed的离散度不等同独立数据确认。

## 5. 执行规则

不覆盖现有 run、原工作树数据或共享预处理。新实验必须用新目录，所有选择在 Train/Validation。
历史 exact-source/AST guard 不放宽；如需历史重放，按原 commit 在隔离树执行。
统一入口另有独立协议、源码/config/input绑定、required文件和 inner-seed核对、提交的预测锁。
恢复只跳过已封存完整 stage；没有完整 optimizer/RNG state 的 partial stage 不声称精确续训。
未经用户新的明确授权，不从这里启动训练、重跑Test、全量预处理或增加方法模块。
