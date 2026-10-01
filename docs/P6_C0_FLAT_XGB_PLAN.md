# P6-C0 历史窗口 XGBoost detector 开发方案

2026-10-01，模型运行前冻结。用户授权探索全流程，预处理全量不执行。此候选仅替换检测模型；不修改共享输入、原三态标签、60s onset目标、episode decoder或matching。

## 假设与 Material Passport

问题：逐窗口因果修复后，固定历史输入是否仍有神经网络难以学到的onset边界？假设浅树对已有数值的非线性阈值划分能改善事件召回。反证是相同输入/标签下XGB仍不能提高开发事件指标；不能据此证明原始telemetry完全不可辨识。

输入为每行自身 `[t_hat-300s,t_hat)` 的10个30s bin，Metric48、Log32、Trace8，按canonical service/edge顺序无损展开为16000D。禁止使用RCA的对称±300s 68D作为在线detector。无新的fit/scaler/feature选择。只读主工作树Train数组，原50/20分割，Fit43550窗/7443完整事件，Validation17414窗/2901完整事件；IGNORE Fit行去除，38868训练行。单个feature矩阵约2.3GiB+1.0GiB，不是全量raw预处理。

环境固定 `/home/zhangll24/.venvs/ada-rca-supervised-baselines/bin/python`：Python3.8.20/XGBoost2.1.4，CPU hist单线程、PYTHONHASHSEED=42。这是独立detector配置，不复用RCA已锁run的参数/结果身份。

固定200 trees、depth3、eta.05、subsample1、colsample1、L2=1、seed42、binary:logistic，scale_pos_weight=Fit negatives/positives。树结构设置来自既有RCA XGB冻结实现的保守配置；无search/early stopping/eval_set。Fit worker不接收Validation标签，固定boosting轮数不能称为与NN checkpoint搜索等预算。只有一个候选模型；全体Validation历史窗评分后，按原exact event-F1 threshold sweep选择一个开发阈值。

## Gate与 GO/NO-GO

- 工程gate：Train/Validation索引、预测时刻、标签与归档精确一致；Test数组/标签不读；flatten不跨行，同行改变/重排/singleton的XGB概率精确相同。模型与完整Validation scores先写prediction lock，再做阈值选择/事件匹配；运行前后源码与输入digest一致。
- 比较对象仅为新的逐窗口C0重训基线；旧batch-conditioned结果只作更正影响说明。两者共享原70% Train-fitted preprocessing，所以Validation仍是复用开发筛查，不是独立OOS证明。
- 固定开发GO：相对修复NN基线，Validation recall绝对提升≥.02、event F1绝对提升≥.005、precision≥.90；单onset组recall不下降超过.03。报告paired gained/lost及样本数，低频n<30仅描述。任一失败则NO-GO，不加深树、不加轮数、不改阈值目标或换特征挽救。
- GO只允许另立完整RCA/E2E开发验证，不等于E2E已改善。本轮无Test inference action；Test重评须另行锁新协议/模型/预测，不用Test模型选择。

## 命令

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2-c0flatxgb
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c0_flat_xgb.py \
  --config configs/e2e/gaia_p6_c0_flat_xgb_v1.json \
  --output-dir experiments/p6/c0_flat_xgb/fixed-v1-seed42 \
  --data-root /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/data/p5/v3_preprocessing_v2/ad \
  --artifact-root /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/artifacts/p5/v3_preprocessing_v2/ad \
  --registry /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/artifacts/p5/v3/protocol/gt_event_registry.csv \
  --threshold-workers 8
```
