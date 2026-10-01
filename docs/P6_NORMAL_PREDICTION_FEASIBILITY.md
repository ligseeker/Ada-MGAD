# 半监督正常行为预测：后续 Fit-only 可行性检查

## Material Passport

- Mode: plan / file-grounded feasibility
- Source: gpt-6-luna read-only exploration of frozen Train arrays and existing diagnostics
- Human read status: not attested
- Evidence grade: proposal; no predictor trained, no new Validation/Test result
- Date: 2026-10-01

## 问题与顺序

E1起点监督、E2独立bin输出与条件性TCN先逐步验证。半监督方向是第二支线，
检验仅学习正常轨迹的预测残差能否提供当前分类器欠缺的早期异常信号，特别是
memory/长事件；不从已经看的Validation gained/lost定义特例。

现有数据不是每槽原始观测：Metric45数值槽已限龄ffill，另有3个聚合mask通道；
无逐feature raw-valid mask。逐槽raw-observation masked residual需要新Train
sidecar，目前未生成；这不等于无法研究冻结的filled-series residual。
后者必须如实命名和解释，按缺失/观测率分组检查误报，不能声称只在真实观测
槽上打分。rawobs修复仍仅两个aggregate fractions；不自动全量预处理。

## 固定输入和预测时刻

为保持当前300s总信息窗口及原sample-index cohort，建议在同一10bin窗内：
前9bin[t-300s,t-30s)预测最后1bin[t-30s,t)，观察目标bin后于t打分。
不能用完整10bin输入预测其已知最后一bin；也不能将下一bin误差提前记为t可用。
若另采用10历史bin预测下一bin，则总足迹330s，须另声明额外30s上下文并处理
边界不合法窗口；不混入本最小比较。

预测对象仅10service×45 numerical slots，不预测3个聚合mask；过去的mask可作输入。
首候选固定两个causal residual blocks、hidden32、kernel3、dilation1/2、Huber loss。
分数候选为每service的45槽平均绝对残差，再取十service最大值；不输入fault/root。
不得根据Validation选择聚合方式、窗口、模型宽度或阈值。

## Fit 内检查

在原Fit时间块固定60/20/20：正常训练/正常校准/完整事件留出。边界按300s
总足迹purge。normal必须是整个[t-300s,t)与所有已知label-legal GT区间无交集；
包括跨子块仍active事件。recent-onset NEG不能当normal，IGNORE不能当normal。
正常只表示无注册事件的窗口，不能保证不存在未标注异常。

校准固定正常分数99.5分位；留出按照原60s容差、一对一匹配和完整GT报告
event P/R/F1、首60s响应、memory/长事件、缺失观测率分组和分母。
校准/留出正常格各>=512、干净memory起点>=30才解释稀有故障支持；不足则
记为证据不足，不改划分或把交叠事件删出完整指标。正式训练/轮数/seed及
回退条件须在该支线执行前单独锁定，这份可行性说明不是已执行协议。

此前raw变化的Fit描述AUC .7984（clean-memory136）仅支持可辨识性筛查。
global Metric drift候选既有NO-GO也不能证明预测残差一定无效。低样本memory
支线应同时计算整体收益容量，不把macro改善当作已通过完整E2E门槛。
仍沿用共享70%Train preprocessing的开发限制，独立确认需未触碰时间块。
