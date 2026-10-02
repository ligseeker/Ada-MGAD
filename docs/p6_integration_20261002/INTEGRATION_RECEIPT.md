# P6 工作树整合与手动重复入口验收

日期：2026-10-02。原工作树 `Ada-MGAD-e2e-v2` / `e2e-v2`。

## 保存范围与来源

从预算复盘 bf423f9 出发，在 `integration/p6-assets-and-seeds1-10` 保留真实 merge
parents，合入 Test report、mobpair/XGB、C0R2、Metric drift、flat/onset 初始实现、
normal fallback、budget 失败/取消诊断、budget prefix、detector Test review 和原主分支。
清点的 21 个来源 HEAD 均为集成树祖先；完整列表见 `asset_inventory.json`。
不需要把旧工作树的 ignored 数据复制到 Git 才能保留其代码历史。

原主分支原 HEAD8d69eba 的两份用户已有文档已按原字节保存并提交，哈希见
`preexisting_main.json`，最终活动路径的字节仍吻合。历史 conflict variant 和整合前
canonical context另存 `source_variants/`。未删除原工作树或原实验产物。

按后续“只保存文档和代码”的范围更新，撤掉本轮新增的 4,127,748,302-byte 大副本。
删除前，逐个独立内容核对原工作树文件的 SHA；全部 1,083 个独立原内容仍在。
只保留约 0.239 MB 的必需 frozen RCA 包及约 0.960 MB 的文字来源索引。
大特征缓存和 detector 权重不进本次整理。源数据索引不是数据备份。

## 冲突处理

- 保留逐窗口图的当前数值路径；Metric drift 作为默认关闭的可选模块保留。
- 保留当前训练 factory、epoch observer、Fit-only weighting hooks；正常 TCN 不开启其他候选。
- 保留当前 corrected flat-margin 导出、window factory 和 Test annotation 边界 guard。
- 旧 exact-source/AST 条件不放宽。旧 runner 的历史重放使用其锁定 commit；
  新手动重复使用独立协议、独立目录和 commit/source/input 绑定。

## 核验（没有真实训练/新 Test）

已核当前源码、配置、历史小 manifest、21-source ancestry、模型/feature-source SHA、
原用户文档 SHA 和独立内容原文件存在/哈希。
两个实际环境的 import-only 严格版本预检通过（Python3.8.20；DAG Torch1.12.0；
RCA XGB2.1.4；其余包版本由 config 固定）。Python3.8 AST parse、`--help` 和 diff check 通过。

合成/单元验证 **65 项通过**，执行命令分环境为：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /home/zhangll24/miniconda3/envs/DAG/bin/python -m pytest -q \
 tests/test_p6_c0_tcn_model.py tests/test_p6_c0_window_causality.py \
 tests/test_p6_c0_metric_drift.py tests/test_p6_c0_onset_targets.py \
 tests/test_p6_c0_trigger_model.py tests/test_p6_c0_long_onset_weight.py \
 tests/test_frozen_test_review.py

PYTHONDONTWRITEBYTECODE=1 \
 /home/zhangll24/.venvs/ada-rca-supervised-baselines/bin/python -m pytest -q \
 tests/test_p6_two_stage_repeats.py tests/test_frozen_rca_test_review.py \
 tests/test_p6_bin_trigger_decoder.py
```

覆盖 TCN/window graph、默认关闭的 Metric drift 兼容、onset/IGNORE、输入 label firewall、
因果一对一/独立 bin、GT/prediction 分母、非法 RCA context、空预测、全十 stage required
files/inner-seed/hash 锁、已提交锁、子进程启动 hash seed、全部 seed 汇总/ddof=1。
不表示真实 CUDA 训练或新的十次实测已经通过；真实 Validation replay/batch gate 在手动运行中执行。

新入口不训练 RCA、不使用 Test 调参/选 seed、不覆盖现有 run。
全十 predictions lock 提交后才语义读取 Test GT；metadata/文件字节 SHA 与语义标签 join 分开说明。
命令、恢复范围、真实环境限制和统计口径详见 `../P6_TWO_STAGE_REPEATS_PROTOCOL.md`。
