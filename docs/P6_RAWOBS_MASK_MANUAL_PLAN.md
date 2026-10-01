# 原始可观测率修复：手动预处理与开发实验

2026-10-01。工作树 `Ada-MGAD-e2e-v2-c0rawmask`，分支 `experiment/p6-c0-rawobs-mask`。完整预处理未执行。

## 假设与修改

旧 Metric mask 对≤60s forward fill 后的值取 finite，夸大 raw availability。新版本显式指定 observation_mode=raw，供 global/host 两个 observed fraction 使用；host_applicable不变。45个数值槽、ffill、scaler、网格、Log、Trace、监督和逐窗口图检测器均不改变。

该修复必须先通过语义 gate；不承诺它提高 recall。模型假设是显式缺失状态可减少填充后伪稳定输入造成的 onset 误判。准确性决定 GO/NO-GO，不能反向改变 raw availability 的定义。

## Material Passport 与防护

- Base causal 实现 `37eed5c`；新配置/策略均单独版本，schema和data/artifact根全部新建。
- canonical registry仍为原SHA；预处理仅从原70% Train拟合；C0在其中做50/20开发。Validation输入参与共享预处理，因此仍是开发筛查。
- 额外工程防护不改变特征值：已有data/artifact根在任何fit/write前拒绝；同维度的slots/scalers/Drain/edges必须逐字段对齐冻结schema；新策略对原始CSV内容做SHA而不是仅路径和大小。全文件内容哈希包含Test telemetry文件字节，仅用于provenance，不用于fit/选择；不等于严格Train-only内容目录。
- 新策略的raw_binding_mode不同于旧binding；不能混用旧schema。
- Counter的“先binmean再差分”风险本轮只记录，不与mask混合修改；没有足够实际证据把memory失败归因于它。

## 完整手动命令

以下命令由用户手动执行。第一步较耗时，生成Train/Test预处理数组并做形状/有限值验收，但不训练或推理；不要用 `all`、`smoke` 或旧共享输出根。

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2-c0rawmask

PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p5/run_i1_ad.py preprocess \
  --config configs/e2e/gaia_p5_v3_preprocessing_v2_rawobs_mask_v1.json \
  --raw-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS \
  --data-root data/p5/v3_preprocessing_v2/rawobs-mask-v1-20261001/ad \
  --artifact-root artifacts/p5/v3_preprocessing_v2/rawobs-mask-v1-20261001/ad \
  --checkpoint-dir data/p5/v3_preprocessing_v2/rawobs-mask-v1-20261001/checkpoint \
  --chunk-rows 150000 --workers 1 \
  --metric-workers 2 --log-workers 2 --trace-workers 2 \
  --start-method spawn --gpu false

PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/check_rawobs_train_arrays.py \
  --baseline-data /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/data/p5/v3_preprocessing_v2/ad \
  --candidate-data data/p5/v3_preprocessing_v2/rawobs-mask-v1-20261001/ad \
  --output-dir experiments/p6/c0_rawobs_mask/train-array-gate-v1

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c0_window_causal.py train \
  --config configs/e2e/gaia_p6_c0_rawobs_mask_v1.json \
  --data-root data/p5/v3_preprocessing_v2/rawobs-mask-v1-20261001/ad \
  --artifact-root artifacts/p5/v3_preprocessing_v2/rawobs-mask-v1-20261001/ad \
  --registry /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/artifacts/p5/v3/protocol/gt_event_registry.csv \
  --output-dir experiments/p6/c0_rawobs_mask/retrained-v1-seed42 \
  --gpu --threads 8 --threshold-workers 8
```

逐条执行：只有第一步成功且Train-array gate PASS才启动第三步；若fit选择不足45 real Metric槽或任一identity断言失败，停止，保留INCOMPLETE，不补槽、不放宽门槛。工具拒绝复用输出目录；中断后使用新的版本目录，不删历史。

## 验证与 GO/NO-GO

1. 工程：仅原始global/host可观测率降低，其他Train数组/45数值槽/host applicability精确一致；shape48/32/8有限；窗口/标签/cohort与旧归档精确一致；真实输入逐窗因果gate PASS。
2. 与同输入范围、同GPU/seed/训练选择规则的逐窗口 causal baseline 对比，唯一数据变量为mask。两者各自在Validation按同一exact event-F1阈值规则选择；不使用Test。
3. 预设效果GO：Validation event F1绝对提升≥0.005，recall不下降，precision≥0.90；报告paired gained/lost、单/多onset与fault样本数。若仅n很小的小组改善，不判GO。否则效果NO-GO并停止这条性能路线；raw mask正确性修复与性能结论分开。
4. 本轮不含Test模型评估，不声称独立统计显著或完整E2E提升。通过开发门槛后才另立RCA/E2E计划。
