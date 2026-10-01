# P6 检测研究：30s 起点监督与独立输出消融

## Material Passport

- Mode: run / controlled development experiment
- Source: repaired window-independent C0 baseline; frozen Train telemetry and canonical registry
- Human read status: not attested
- Evidence grade: reused Train/Validation development; no new Test confirmation
- Date: 2026-10-01

## 授权、问题与变量

用户授权按照研究方案逐步实施，文件探索由 gpt-6-luna 完成。此次先验证：
更精确的起点监督能否减少相邻事件在连续阳性响应中的合并，改善完整事件召回。
新工作树 `Ada-MGAD-e2e-v2-c0onset`、分支 `experiment/p6-c0-onset30`；
不覆盖主工作树或既有 run，不执行全量预处理或 Test。

E1 唯一方法变量为监督目标；当前输入、逐窗口图模型、初始化 seed42、优化器、
训练预算、Validation 选择规则、连续阳性 episode 合并、一对一匹配全部固定。
Fit43550 窗/7443 完整事件；Validation17414 窗/2901 完整事件。
基线：`c0causal/experiments/p6/c0_window_causal/retrained-v1-seed42`，
Validation TP/FP/FN2103/19/798、P/R/F1 .991046/.724922/.837348。

## E1 起点目标和归因边界

- 预测时刻为 t=target_bin_end；输入仍为 [t-300s,t)。
- POSITIVE：存在 start 满足 t-30000 <= start < t。起点恰好等于 t 时，
  首次在 t+30000 被监督；不把未完成 bin 的起点信息放入输入。
- IGNORE：精确保留旧 60s recent-onset 目标的 IGNORE 向量；不是改为 30s IGNORE。
- NEGATIVE：没有本 bin 新起点且不是旧 IGNORE，包括原 POSITIVE 中较早的起点。
  因而这是起点监督，不是异常活动状态分类。
- 同 bin 多事件仍仅一个 binary positive；不读取 GT 计数生成多个预测。
- 先检查旧标签与归档精确一致，再允许唯一 POSITIVE→NEGATIVE 转换；
  window index、prediction time、GT ID/metadata 和 IGNORE mask 必须相同。
- pos_weight 公式仍为 Fit negatives / Fit positives。目标改变后该派生数值自然变化；
  不是手动调参，也不能把结果归因为纯粹更换时间边界而忽略派生类权重。
- 起点时间窗缩短也提高了时间精度要求；负结果不证明所有监督起点模型均无效。

## 实验顺序

1. E1：固定架构/decoder，起点目标重训，封存模型与 Validation 分数。
2. E2：E1 分数封存后，独立验证每个阳性 bin 一个候选，取消连续合并；
   不重训、不更改GT或60s匹配容差，所有新增候选均进入 FP 计数。
   阈值操作条件在解封E1分数前锁定；先使用E1所选阈值，另报告预先声明的
   Validation-only独立阈值选择作为不同operating-point结果，不混成单变量结论。
3. E1/E2 未达到效果门槛时，可在独立 run 比较小型因果 TCN；只更换编码器，
   不同时修改起点目标、预处理、权重或 decoder。具体 TCN 配置须另锁后运行。
4. 同 bin count 输出先做 Fit 内时间留出可辨识性验证。半监督正常预测分支也先
   做 Fit 内验证，再独立运行；不根据本轮Validation gained/lost制定救分规则。
5. 检测候选达标后才重建 detector-OOS 训练 anchors 和修复后 68D+XGB 开发基线，
   做全GT分母的Diagnosis评估；不自动重评Test。

## GO / NO-GO（在新实验结果前冻结）

- E1 对修复后基线：Recall增量>=.05，event F1增量>=.02，Precision>=.98。
- 同时报告 single/multiple onset、fault/root macro、memory、长事件样本数与退化；
  single-onset recall下降不得超过.03；小样本分组不作显著性结论。
- 若进入 E2，必须同时比较E2 vs E1（decoder归因）和E2 vs基线（总效果）；
  仅改变matching次数或重复告警不构成完整E2E成功。
- Full E2E最终开发门槛：同完整cohort Diagnosis F1@1提升>=.02，报告@3/@5和失败。
- 初步GO后才增加seed和新时间块复核；未触碰时段才可提供独立确认。
- NO-GO不追加本候选的threshold/offset/tree depth或其他后验参数搜索。

## 完整性与限制

run入口只有check/train；不导入旧ProtocolState实例，不打开Test数组/node labels。
registry全文件SHA用于绑定，interval列仅用于路由，Test注释行在解析前跳过。
before/after source/input hashes、实际execution commit、模型参数/训练配置及旧GT
身份必须一致；完成清单封存每个输出。真实窗口扰动/重排/batch1/2/32 gate atol1e-5。
原始45槽与filled mask保持既有冻结输入。raw-observability修复为独立变量；
没有新预处理产物时不把它混入目标实验。共享70% Train preprocessing含Validation
无标签输入；Trace parent可用性仍待证明；本轮不宣称严格在线或独立OOS验证。

## 产物

- `configs/e2e/gaia_p6_c0_onset30_v1.json`
- `scripts/p6/run_c0_onset_development.py`
- `experiments/p6/c0_onset_development/preflight-v1/`
- `experiments/p6/c0_onset_development/onset30-v1-seed42/`
- `experiments/p6/c0_onset_development/onset30-vs-baseline-v1/`

## 启动前核验与执行命令

`preflight-v1`通过：Fit新POS7027/NEG31841/IGNORE4682；Validation新POS2709/
NEG13029/IGNORE1676。唯一转换分别5870/2178个POS→NEG。模型/优化代码与修复后
基线hash一致；完整GT身份、窗口、时间和mask精确一致。此预检在提交前执行，
记录base HEAD加实际source hashes，不把base HEAD说成完整执行源码。
另修复未消费的70%边界标签：填IGNORE sentinel，不再为Test-owned目标构造标签；
所有实际Fit/Val样本与原基线一致。

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2-c0onset
env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c0_onset_development.py train \
  --config configs/e2e/gaia_p6_c0_onset30_v1.json \
  --output-dir experiments/p6/c0_onset_development/onset30-v1-seed42 \
  --data-root /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/data/p5/v3_preprocessing_v2/ad \
  --artifact-root /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/artifacts/p5/v3_preprocessing_v2/ad \
  --registry /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/artifacts/p5/v3/protocol/gt_event_registry.csv \
  --gpu --threads 8 --threshold-workers 8
```

E2使用`scripts/p6/analyze_c0_bin_decoder.py`，只读取完成的E1 Validation分数。
固定阈值对比和Validation重选阈值的operating-point对比分开。
