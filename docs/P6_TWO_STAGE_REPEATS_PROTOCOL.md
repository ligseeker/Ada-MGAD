# P6 两阶段 seed 1–10 重复实验：手动运行协议

协议：`P6-TWO-STAGE-FROZEN-XGB-SEEDS1-10-V1`，冻结日期 2026-10-02。
用户已选择：**每个 seed 重训检测器，接入现有冻结 68D＋XGBoost**。
方法细节与历史资产见 [核心方法](P6_TWO_STAGE_CORE_METHOD.md)。

## 要验证的假设

在同一冻结数据和 RCA scorer 上，只改变 detector seed 为 1–10，考察当前
TCN onset30＋固定独立 bin 接入方案的检测、matched RCA 和完整 E2E 波动。
预期输出是十个 seed 的完整结果及离散程度；本协议不设看过 Test 后再挑模型的 GO 门槛。
如果仍不稳定，报告实际波动后再决定是否有新的 Train/Validation-only 研究依据。

## 固定与变化

| 项目 | 本次规定 |
|---|---|
| 唯一实验变量 | detector random seed = 1,2,3,4,5,6,7,8,9,10 |
| detector 架构/输入/目标 | 冻结 TCN、逐窗口图、onset30、原 IGNORE mask |
| 训练 | max_epochs=30、patience=8；原 optimizer/scheduler/loss/batch |
| checkpoint 与 tau | 原 merged Validation event F1/recall/tau 规则；每 seed 仅 Validation 选择 |
| 接入 decoder | 独立 bin，使用上行选择的 tau；不重新在 bin/Test 上调 tau |
| RCA | 同一冻结 68D＋XGB；不重训、不加 scaler、不变候选顺序 |
| anchor | 固定 `t_hat−25621ms`，不按 seed/Test 调 offset |
| Test | 同一完整 GT=5787，一对一 causal 匹配、60s；所有预测锁后评价 |
| 结论 | scorer transfer、固定数据初始化稳定性、reused-Test descriptive |

本协议独立于历史候选的 all-seed GO 门槛。它不是重开旧 NO-GO 的 native Fit-OOS
RCA 路径，也不宣称 detector seed 改变后的 scorer 是原生训练匹配的。

## 执行阶段与封存

1. `preflight`：核配置、scorer/source 哈希、两个环境和输入文件存在性，不打开 telemetry 数组/标签。
2. `predict`：新建唯一 collection，冻结源码/config、输入哈希、runtime 和 seed family。
   记录两个环境实际版本；detector 子进程启动前设置对应 PYTHONHASHSEED，RCA 固定为20260826。
   对每个 seed 执行 Fit/Validation 正常训练；核 Val 首/中/尾 checkpoint replay 和真实窗口 batch independence；
   保存 Validation 原始分数与 bin 指标；用输入专用 dataset 生成 Test scores/episodes；
   仅凭预测 anchor 提取 68D 并使用同一 frozen scorer 排十个服务。
3. 全十个 seed 的 detector/RCA stage complete 后，全量校验并写 `global_prediction_lock.json`。
   `scope_lock.json` 和 prediction lock **提交到 Git 后**才能进行 Test GT join。
4. `evaluate`：先核已提交 lock 与全十个 stage 的哈希，再读取 Test 标签；
   所有 seed 分别评价完整 GT、failure/strata，自动生成十次汇总。
   resume/evaluate 均复验环境；预测期间每阶段核文件大小/mtime，global lock 和评价前核全字节 SHA。

所有写入都在新 collection；已有 stage、run 或共享输入不会覆盖。
checkpoint 从 0 起算 epoch；early stopping 并不要求每次跑满 30。
预测 stage 的 `test_gt_or_matching_read:false` 表示未读 Test 语义标签/匹配，
训练仍读 global registry interval/domain 路由列，不能解释成从未读 registry metadata。
RCA 的 ±300s 窗口仍然包含 post-anchor 数据。
保留原训练的 CUDA 默认设置，不新增 deterministic-algorithms 变体；各 seed 保存实际 CUDA/cuDNN
版本、device 和 flags。标准随机种子控制不等于跨硬件或 CUDA kernel 的完全逐 bit 重现。

## 完整手动命令

直接复制整段执行；GPU 默认使用已有 `CUDA_VISIBLE_DEVICES`，未设时为 0。
无需切 conda 环境：编排器使用 DAG Python，RCA 子进程自动使用专用 XGBoost 环境。
代码与 config 整理时只做预检和合成测试，**未启动这里的训练或 Test**。

```bash
bash <<'P6_RUN'
set -euo pipefail
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

P6_REPEAT_PY=/home/zhangll24/miniconda3/envs/DAG/bin/python
P6_REPEAT_CONFIG=configs/e2e/gaia_p6_two_stage_repeats_v1.json
P6_REPEAT_RUN="$PWD/experiments/p6/two_stage_repeats/seeds1-10-v1-$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)"

"$P6_REPEAT_PY" scripts/p6/run_two_stage_repeats.py preflight \
  --config "$P6_REPEAT_CONFIG"

"$P6_REPEAT_PY" scripts/p6/run_two_stage_repeats.py predict \
  --config "$P6_REPEAT_CONFIG" --output-dir "$P6_REPEAT_RUN" \
  --gpu --threads 8 --threshold-workers 8 --feature-workers 8

git add -f -- "$P6_REPEAT_RUN/scope_lock.json" "$P6_REPEAT_RUN/global_prediction_lock.json"
git commit --only -m "lock(p6): ten-seed frozen-XGB transfer predictions before Test labels" \
  -- "$P6_REPEAT_RUN/scope_lock.json" "$P6_REPEAT_RUN/global_prediction_lock.json"

"$P6_REPEAT_PY" scripts/p6/run_two_stage_repeats.py evaluate \
  --config "$P6_REPEAT_CONFIG" --output-dir "$P6_REPEAT_RUN"

printf 'Results: %s/summary/per_seed.csv\n' "$P6_REPEAT_RUN"
P6_RUN
```

运行前工作树须无未提交代码/config；预检会明确拒绝漂移。所有 seed 顺序执行，避免 GPU
并发改变资源条件。`threshold-workers` 只用于 Validation exact sweep，`feature-workers`
只用于 label-free RCA 提取，不改变优化器或数据 cohort。

## 恢复规则

中断后先看真实 PID 和 stage completion，不自动重启另一个实验。
同一 collection 的 `predict --resume` 仅跳过哈希验证通过的完整 stage；runtime/config 必须完全相同。
未封存的训练/提取 stage 会拒绝继续，保留现场，因为未保存完整 optimizer/RNG 中间状态，
不能声称精确续训。不要删除旧 stage 后原地重跑；核原因后新建 collection。
如果 global lock 已生成，执行上述 Git 封存与 `evaluate`，不再次 `predict`。
已有完整 evaluation 不再评价；若只缺 summary，可运行 `summarize`，已有 summary 也不会覆盖。

## 输出与算术

```text
<collection>/
  scope_lock.json
  seed-01 ... seed-10/
    detector/ # checkpoint、Val selection/分数/bin 指标、Test scores/episodes、gate、seal
    rca/      # features、valid、完整十服务 rankings、固定 scorer 记录、seal
  global_prediction_lock.json
  evaluation/
    test_gt.csv、purged_gt.csv
    seed-01 ... seed-10/ # detector/matched RCA/E2E、failure ledger、分层
  summary/
    per_seed.csv
    statistics.json
```

detector TP/FP/FN 分母与完整 GT 一致。matched RCA 单独列合法 matched `n`、AC@1/3/5、MRR。
完整 diagnosis 对每个 k：`TP=matched 且 root rank≤k`，`FP=全部预测−TP`，
`FN=全部 GT−TP`；非法上下文和排名失败仍计入完整分母。
十次的样本标准差使用 ddof=1，列出全部 seed、mean/min/max 与 defined_seeds。
如果 matched n=0，条件 RCA 指标为 null，不当作成功，也不从 E2E 删除。
不从最好的 Test seed 推导方法改善，不把十次重复当成十套独立数据。
