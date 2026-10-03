# P6 两阶段方法 Scientific Freeze v1

## 决定与版本

- 状态：**SCIENTIFIC_FREEZE，用户已确认**。
- 确认日期：2026-10-03（Asia/Shanghai）。
- 用户决定：`可以，Scientific Freeze 当前版本`。
- 方法版本：`P6-TWO-STAGE-SCIENTIFIC-FREEZE-V1`。
- 仓库分支：`e2e-v2`。
- 本地 Git 标签：`p6-two-stage-scientific-freeze-v1-20261003`。
- 方法执行源码：`3e1fdea2134b1d48dd8d2ef59f64a04f11bc6fcd`。
- 全部十次预测的已提交 lock：`98aaf10a8827a9c90f99ba3be4ef975f5876f973`。
- 实际封存提交由上述 Git 标签定位；冻结清单自身由该提交保存。

本决定结束当前数据与协议上的方法优化、阈值/offset/模型搜索和额外重复训练。
后续可以基于已冻结产物整理证据、文档和结果表；用户当前要求不生图、不写论文正文。
重新开启方法研究须有新的明确授权、单独冻结的假设/协议与独立输出目录；
独立效果确认需要未使用的数据。原始实验与本版本继续保留。

## 冻结的核心方法

| 项目 | 冻结内容 |
| --- | --- |
| 检测器 | 监督 WindowCausalTCNTrigger，逐窗口图，多模态输入，因果TCN与system head |
| 输入 | 最近10个完成的30s bin；Metric/Log/Trace，原冻结preprocessing |
| 监督 | onset30；原recent-onset60 IGNORE mask；Fit-only pos_weight masked BCE |
| 数据划分 | 时间顺序Fit/Validation/Test=50/20/30，完整block内窗口和GT |
| 训练 | AdaBelief/原StepLR，batch32，max_epochs30，patience8 |
| checkpoint/tau | merged Validation event F1/recall/tau；每seed仅Validation选择 |
| 接入 | 独立bin decoder，沿用对应seed的冻结tau |
| RCA anchor | `t_hat−25621ms`，原Train-derived固定backdate |
| RCA表示/模型 | 每服务68D Z2，W300-B15（±300s），同一冻结XGBRanker，无额外scaler |
| 评价 | causal60s一对一最大匹配/最小延迟；完整GT5787；检测、matched RCA和Full E2E分别报告 |
| 重复实验 | detector seeds1–10；RCA不重训；完整报告十次，不以Test选最佳seed |

固定 scorer SHA256：
`578b1baf20ff9958a990b95b724a4f8afb83aaa801ce0762ccc56a59d25f0af2`。
模型包：`assets/p6/frozen_rca/backdate25621-v1/`。
详见[核心方法](P6_TWO_STAGE_CORE_METHOD.md)与[原重复协议](P6_TWO_STAGE_REPEATS_PROTOCOL.md)。

冻结范围包括完整方法、全部十个seed的产物和统计；未指定部署用的单一detector checkpoint。

## 冻结的实验与结果

集合：`experiments/p6/two_stage_repeats/seeds1-10-v1-20261002T213410/`。
10 detector + 10 RCA stage，以及evaluation/summary均COMPLETE。
以下为十次mean±sample SD；百分比指标的SD单位为百分点：

| 指标 | 冻结结果 |
| --- | ---: |
| Detector P/R/F1 (%) | 98.12±1.28 / 87.07±3.08 / 92.25±2.11 |
| matched RCA AC@1/3/5 (%) | 90.03±.44 / 99.83±.03 / 99.90±.01 |
| matched RCA MRR | .9488±.0023 |
| Full E2E F1@1/3/5 (%) | 83.01±1.59 / 92.05±2.07 / 92.12±2.09 |
| macro root/fault E2E R@1 (%) | 17.95 / 22.46 |

完整P/R/F1、每seed、cohort、失败分解与统计解释保存在
[十次复盘](P6_TWO_STAGE_SEEDS1_10_RESULTS_20261003.md)。
该复盘作为作出决定前的分析记录原字节保留；其E/F未来方案当前不执行。
既有预算/selector/replication NO-GO与历史失败记录保留。

## 证据等级与限制

证据等级：**固定数据与冻结RCA scorer条件下的初始化波动，reused-Test descriptive**。
产物完整性和指标算术通过；本轮封存未重训、未做模型推理或重新评价Test。

- Test已反复使用，十个seed不构成十套独立数据。
- 原70% Train preprocessing包含Detector-Validation协变量。
- 固定RCA训练anchors继承旧C1的batch-conditioned限制；本次是transfer，未建立新seed原生Fit-OOS RCA。
- RCA使用post-anchor数据；Trace parent当时可用性仍未证明。
- 当前Validation选择采用merged单位，最终输出采用bin单位，稳定性问题仍保留。
- Test极度偏斜，少数fault/root能力弱；micro高分和近满分AC@3/5不代表均衡泛化。
- 标准seed控制与当前CUDA默认设置不保证跨硬件逐bit重现。

## 封存与存储边界

机器可读清单：[freeze manifest](p6_two_stage_scientific_freeze_v1_20261003.json)。
清单绑定130份执行源码/config、原scope/prediction lock、20份stage completion manifest、
evaluation/summary清单、小模型包、复盘及其审计附件、当前方法与研究入口。
原scope保存输入SHA；原stage/evaluation/summary清单保存其完整成员SHA。
这些绑定共同覆盖现有checkpoint、分数、排名、特征、GT与failure账本。

Git新增保存文档、小型审计文件和哈希清单。大型实验产物仍在原本地目录，
原派生工作树仍按已有来源索引保存；清单不是数据备份。
本次不复制或删除这些文件，也不改变已封存实验目录。

封存前重新核验了执行源码/config、3,803个输入的size/mtime、全部stage/evaluation/summary
字节哈希、已提交预测锁与分析附件。全输入字节哈希已在前一轮结果审计通过，
本次未重新扫描全部3.15GB输入或复现完整训练。
