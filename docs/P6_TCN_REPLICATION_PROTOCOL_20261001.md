# TCN + independent bins：初始化及时间稳定性复制

## Material Passport

- Date: 2026-10-01
- Scope: registered after seed42 Validation GO, before seed17/2026 results
- Evidence grade: reused-Validation development; not independent confirmation
- Model changes: random initialization only; no new preprocessing or Test inference

## 假设及固定方法

seed42 的 2577/14/324、P/R/F1 .994597/.888314/.938456 是值得复制的候选。
检验：同一监督起点 TCN + independent-bin 算法能否在两个新初始化下持续改善
已修复的 detector baseline。seeds 固定为17和2026，seed42保留主模型，不挑最佳种子。
这不是 seed-paired encoder attribution；baseline 固定seed42，其种子方差未估计。

输入/标签/IGNORE/窗口graph/encoder/优化器/损失/预算全部与seed42相同。
max_epochs30、patience8、同一 merged event F1 checkpoint/threshold selector；
每个seed选出的merged阈值随后冻结，用于独立bin主结果。不同模型不共享数值阈值，
因为其分数标度可能不同；完整选择算法相同。预声明 bin 阈值重选仅作辅助，
不得替代主结果或选择最佳seed。训练不会强行固定为seed42实际停下的10epoch。

## 冻结门槛

每个seed42/17/2026主结果必须同时满足原开发门槛：

- P≥.98；R相对同一 repaired baseline ≥+.05；F1≥+.02；
- 同30s bin单起点组召回下降≤.03。

时间检查：将冻结Validation区间按**相同持续时间**分成前/后两半；
按GT起点归属计算paired recall，两个时段各要求≥+.03。
对完整Validation执行一次原匹配，保留其全局命中集合与全部未匹配预测；
不在人工分界处重新匹配/裁剪预测。FP/day按全局未匹配预测的t_hat归属，
仅描述，不增加未声明的阈值选择。seed42时间分层此前未查看，门槛先于查看固定。
时间分层同样复用Validation，并非新的forward holdout或显著性检验。

全部通过 → GO_TO_FIT_OOS_AND_VALIDATION_E2E；任一失败 →
STABILITY_NO_GO，不按该seed结果修参数，停止推进其完整E2E，先复盘失败来源。
共享预处理含Validation协变量、原filled-mask及Trace parent可用性边界继续保留。

## 输出及执行

独立工作树 `experiment/p6-c0-onset-tcn-replication`；训练、分析用新目录，
不修改已封存TCN源或产物。运行期间HEAD、源、配置、输入须保持不变。
每个训练进程启动前设置对应PYTHONHASHSEED。源码/输入/GT/cohort全部绑定，
实际batch独立gate必须通过才封存。
全局GT registry的interval/domain列先用于路由，之后只解析Train/Validation
service/fault字段；没有Test数组、预测、指标使用。旧报告的test_read:false只可
解释为无Test遥测/推理/评价，不能解读为没有扫描任何Test行的路由metadata。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=17 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c0_tcn_development.py \
 --config configs/e2e/gaia_p6_c0_tcn_replication_v1.json --replication-seed 17 \
 --output-dir experiments/p6/c0_onset_tcn_replication/seed17-v1 \
 --data-root ../Ada-MGAD-e2e-v2/data/p5/v3_preprocessing_v2/ad \
 --artifact-root ../Ada-MGAD-e2e-v2/artifacts/p5/v3_preprocessing_v2/ad \
 --registry ../Ada-MGAD-e2e-v2/artifacts/p5/v3/protocol/gt_event_registry.csv \
 --gpu --threads 4 --threshold-workers 4
```

seed2026用同一命令替换seed及唯一输出目录。完成后分别运行
`analyze_c0_bin_decoder.py`，再 `analyze_c0_tcn_replication.py`；不运行Test。

## 下一阶段及效果定义

只有复制通过才建立新的 window-independent Fit detector-OOS anchors；不得
重用旧batch-conditioned C1 anchors。固定68D及XGB配方，仅替换检测输入，
对全部Validation GT及预测报告完整诊断P/R/F1@1/3/5，漏检和额外候选都保留。
full E2E F1@1至少+.02才支持继续。这个比较须另行冻结训练cohort及anchor
对照，不能将检测召回差直接当作完整诊断收益。
稀有memory/长事件当前仍弱；若主路线稳定且后续确需改善，单独冻结
正常预测Fit筛查/长故障检测假设，不与本次复制同时改目标或融合分数。
