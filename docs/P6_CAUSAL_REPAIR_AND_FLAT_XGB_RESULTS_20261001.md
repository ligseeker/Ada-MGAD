# 检测全流程探索：逐窗口图修复与固定XGB结果

2026-10-01。结论：固定历史窗口XGB替换为`NO_GO_DEVELOPMENT`；逐窗口图修复完成；raw-observability修复及手动命令已准备。没有执行全量预处理或新的Test inference。

## A. 当前最终实验结果

主分支及remote `e2e-v2` 都为`8d69ebaf3a8942a790768a17edff48954fe7d7b4`，用户已有文档改动未触碰。新工作树独立保存实验。

| Run | execution commit | 状态 |
|---|---|---|
| c0causal/checkpoint-impact-v1 | `37eed5c21a6ed472ba51d5fad8d017fc2035b273` | COMPLETE，旧checkpoint Validation影响审计 |
| c0causal/retrained-v1-seed42 | 同上 | COMPLETE_DEVELOPMENT_ONLY；15epoch，选epoch6（从0计数） |
| c0flatxgb/fixed-v1-seed42 | `3ade981`，完整SHA见输入锁 | 固定Fit/model/probability成功；driver缺logits，INCOMPLETE_EXPORT_ERROR，原目录保留 |
| c0rawmask/finalized-v1-seed42 | `ae7e7a870acadeb1da55b1e074980a3709fc9121` | COMPLETE_DEVELOPMENT_ONLY；只补margin导出，无refit |
| c0rawmask/nn-vs-xgb-v1 | 同上 | 完整Validation总体比较、逐案ledger、来源核验 |

同样的旧冻结输入、原50/20/30划分：Fit43550窗/7443事件；Validation17414窗/2901完整事件。Fit标签positive12897、negative25971、IGNORE4682。IGNORE仅屏蔽训练loss，不缩小事件分母。

| Validation模型 | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 逐窗口图NN重训 | 2103 | 19 | 798 | .991046 | .724922 | .837348 |
| 16000D历史窗口XGB | 1999 | 4 | 902 | .998003 | .689073 | .815253 |

NN阈值`.9995630383491516`，checkpoint SHA `d4f0460649fe2bc586fa4d79f141fe939a2cf16bb1127b27603b7dc0d1ad3f0e`。XGB阈值`.9253149032592773`，固定200 trees/depth3/eta.05，CPU单线程，XGBoost2.1.4。两者阈值均由相同Validation exact event-F1 sweep选择；不同模型阈值大小不能直接比较。NN进行原checkpoint搜索，XGB固定轮数、无search/eval_set/early stopping，选择预算不等价。

旧GPU control原batch图精确重放2103/3/798、F1 .840024。同权重和旧阈值仅换逐窗口图为2097/3/804、F1 .838632；重新选Validation阈值为2098/3/803、F1 .838864。这些仅为旧权重影响诊断，不能代替一致重训。

**本轮没有修复后Test、三折OOS anchor、68D RCA或Full E2E新结果。**旧P6-C1/C2的B/C以及其他68D+XGB结果须按原批条件回顾性pipeline解释，不能自动移植到新基线。

## B. 结果是否可信

已通过：只打开Train arrays；Test root/fault行在registry解析前跳过；Fit/Val index、预测时刻、标签与归档逐条一致；完整Val GT的case ID/service/fault/start/end与归档一致；每个GT恰好一行matched/miss，FP单列。没有以共同matched交集代替分母。

source/config/input/checkpoint与completion manifest均已核验；导出CSV精确重放所选TP/FP/FN/F1。CSV使用round-trip float解析，防止阈值等值处漂移。真实32个Val窗口同行扰动/重排/batch1/2/32的NN logit最大差≤`9.54e-7`，低于预设绝对容差`1e-5`；XGB概率差精确为0。81项集成测试通过。

XGB首次CSV导出字段错误不影响已封存fit或概率。保留错误目录，新目录核验原source/features/model/probability后补齐真实booster margin；重新加载模型概率逐项精确相等，没有重训、改参数或用失败结果选模型。

必须保留的边界：

- **确定的旧batch前视：**P5/C0/C1-v2沿batch均值生成共享图，顺序batch32最早窗口可依赖其后930s输入。不是GT标签泄漏，但在线时刻/延迟/因果OOS anchor声明不成立。固定scores的指标算术可保留为batch-conditioned retrospective。见[更正](P6_BATCH_GRAPH_CAUSALITY_CORRECTION.md)。
- preprocessing在完整70% Train拟合，包括detector Validation的无标签输入；这是复用Validation开发筛查，不是strict forward/preprocessing-OOS或独立确认。单seed差异不支持显著性声明。
- Metric限龄ffill、冻结Drain匹配、Trace end-time分桶未发现确定前视。Trace parent-service索引覆盖整个split且无parent时间约束，严格在线可用性未证明；少量raw头部样本无配置Train行，未找到实际witness，不能断言现存arrays已前视或此gate已通过。逐窗口修复认证的是batch独立性。
- 旧schema复用仅核绑定/维度，raw binding仅路径/大小，存在fail-open缺口；未发现现存tensor已错配。手动分支加逐字段对账、原始内容SHA和任何fit/write前的输出根保护。

## C. 为什么XGB没有改善；旧B/C怎样解释

2901总体paired transition：both-hit1945、NN-only158、XGB-only54、both-miss744。净损失104，Delta R`-.035850`，Delta F1`-.022095`。login_failure新增53/丢失157；memory仅14→15/110。

| 组 | n | NN TP | XGB TP | 新增/丢失 |
|---|---:|---:|---:|---:|
| single-onset bin | 2521 | 1947 | 1847 | 52/152 |
| multiple-onset bin | 380 | 156 | 152 | 2/6 |
| memory | 110 | 14 | 15 | 1/0 |
| duration>300s | 125 | 16 | 16 | 1/1 |

XGB正预测bin更多（2732 vs2590），episode更少（2003 vs2122）；多bin episode更多（554 vs380），长度P95为3 vs2，最大10 vs6。观察支持正响应更易连接、可匹配的新anchor减少；不是所有数据上树模型都不适用的证明，也不是telemetry不可辨识的证明。

NN/XGB root-macro recall .256826/.251855；有支持fault的macro .255815/.210140。Validation仅5类fault有样本，后三组n=8/2/5，CPU n=0；必须保留空组和小样本边界。

旧C1 B/C可保留给定回顾性anchors下的配对描述；不能继续用作因果在线RCA改善证明。修复后C是否仍改善，须另立新逐窗口detector OOS anchors、共同cohort、共享scaler及预测锁，不能复用旧anchors或根据已看Test救参。本轮未重建这条链路。

## D. 当前最大瓶颈

| Failure location | NN | XGB |
|---|---:|---:|
| MATCHED | 2103 | 1999 |
| NO_LEGAL_PREDICTION | 1 | 1 |
| BELOW_THRESHOLD | 137 | 180 |
| NO_NEW_EPISODE | 330 | 400 |
| MATCHING_COMPETITION | 330 | 321 |
| 完整GT | 2901 | 2901 |

NN漏检中660/798=82.7%位于没有新episode或匹配竞争。优先问题是binary recent-onset、连续episode与多事件评价之间的信息压缩，以及罕见fault的可观测信号/数据支持。RCA排序不能补回Stage1没有输出的事件；本轮没有修复后RCA，不能声称已量化其当前性能。

确定实现问题：batch前视、filled/raw mask语义不一致、schema/provenance fail-open、已修复的XGB导出字段遗漏。可改模型/协议：正响应持续形态、onset目标、episode生成和同bin输出容量；不能把所有NO_NEW_EPISODE视为不可改的数据硬上限。

GAIA结构：30s量化、60s onset OR、多注入重叠；注入开始不一定等于可观察异常开始。login_failure2776/2901，mobservice1/2共2803/2901；长事件与memory/非mob混杂。没有证据认定具体registry错标；不能调Test tolerance/offset让事件容易命中。failure类别标位置，不是完整因果归因。

## E. 最值得尝试的最小方向

1. **近期只验证raw-observability mask。**固定45数值槽、scaler、30s grid、Log/Trace及逐窗口NN，只纠正global/host observed fraction的raw finite定义。代码、独立配置、Train数组gate已准备；[完整手动方案](P6_RAWOBS_MASK_MANUAL_PLAN.md)。全量预处理未执行。
2. 若要新方法研究，可单独立项onset count输出，先在Fit内部时间留出验证同bin计数可辨识性。它需改监督/事件输出，属于新协议；不从当前Val gained/lost或已看Test定义rescue规则。本轮不启动第三候选。

Trace parent可用性属于在线解释完整性条件；未证实实际影响前不包装成性能优化。Counter先binmean再差分的风险先做Train-only多采样/reset桶统计，不与mask同时修改。

## F. 验证与 GO/NO-GO

XGB预声明四门槛：Delta R≥.02、Delta F1≥.005、P≥.90、single-onset R下降≤.03。只有P通过，**NO_GO_DEVELOPMENT**；不追加轮数、树深、search、ensemble或Test。

Mask工程gate：原45数值槽、Log/Trace/timestamps/graph、host_applicable精确一致，只有两个observed fraction降低；维度48/32/8有限；窗口/标签/cohort/逐窗独立性一致。相同GPU/seed/训练选择下，event F1提升≥.005、R不下降、P≥.90才效果GO；任一identity失败先停止，不放宽gate。效果NO-GO不改变raw mask应如实定义的正确性结论。

Count尚无具体新执行协议：先仅Fit内部选一种目标/输出，锁后单次Val筛查。拟议R提升≥.05且P≥.90；进入完整RCA/E2E开发时，全GT保留漏检的Diagnosis-F1@1提升≥.03，共同合法matched AC@1下降≤.02。未通过不重评Test或堆模块。

## G. 是否继续或 Scientific Freeze

建议冻结68D+XGB RCA scorer结构，停止本轮detector替换和加模块；保留逐窗口NN开发基线。近期只做raw mask修复验证，并补论文的batch前视/Trace可用性/共享预处理/复用Test边界。若mask也NO-GO，进入方法结构的Scientific Freeze；有效增量需要未触碰时间块、新故障支持或新数据。

不能把旧含batch前视的在线/E2E声明原样冻结。如果论文必须报告修复后完整E2E，先另立一致重训/OOS anchor/RCA cohort/预测锁更正方案，再做明确reused-Test的最终评价。本轮未自动执行。

## 原始记录

- NN：`/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2-c0causal/experiments/p6/c0_window_causal/retrained-v1-seed42/`
- XGB Fit：`/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2-c0flatxgb/experiments/p6/c0_flat_xgb/fixed-v1-seed42/`
- XGB最终导出：`experiments/p6/c0_flat_xgb/finalized-v1-seed42/`
- 完整比较及逐案ledger：`experiments/p6/c0_causal_comparison/nn-vs-xgb-v1/`
- [修复协议](P6_C0_WINDOW_CAUSAL_PLAN.md)、[XGB方案与导出更正](P6_C0_FLAT_XGB_PLAN.md)、[手动预处理命令](P6_RAWOBS_MASK_MANUAL_PLAN.md)。

large arrays/checkpoint/逐窗逐案CSV保持本地原目录，Git仅保存轻量指标/来源/报告。实际执行commit与后续归档commit分开记录。
