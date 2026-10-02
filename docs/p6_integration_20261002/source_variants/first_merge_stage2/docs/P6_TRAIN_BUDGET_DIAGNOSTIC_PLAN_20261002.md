# P6 TCN 训练预算诊断方案（实施版）

日期：2026-10-02。用户提出三个 seed 可能因训练时间不足而不稳定。
本方案此前已提供审查，用户随后授权继续研究。本新工作树只实施
Train/Validation 预算诊断；不打开新的 Test 推理或选择。

## 1. 要验证的假设

**H1：patience=8 让 TCN 在较低学习率阶段得到充分训练之前停止；在原定
30 epoch 预算内继续训练，可以恢复 Validation 事件效果并降低三 seed 差异。**

现存日志支持研究这个假设，但尚未支持其成立：

| Seed | selected epoch（从0计） | completed epochs | Train loss：选中→最后 | Val merged F1：选中→最后 |
|---|---:|---:|---:|---:|
| 42 | 1 | 10 | .9384→.6904 | .8451→.7534 |
| 17 | 2 | 11 | .8695→.6830 | .8346→.7152 |
| 2026 | 2 | 11 | .7952→.6775 | .8300→.7483 |

三次均因连续8轮 merged Validation key 未改善而停止。
StepLR(step_size=10, gamma=.5) 在每 epoch 训练后更新；根据配置和调用位置，
epoch0–9使用lr=.001，epoch10开始使用.0005。seed42没有训练降学习率后的epoch，
另两个seed仅训练了一个。日志没有逐轮实际LR；以上是调度推算。
Train loss下降且Val F1下降也符合泛化退化，不能把loss下降直接当成效果未收敛。

## 2. 唯一研究变量

将 **patience: 8 → 30**，保持 **max_epochs: 30**，在当前上限内观察完整曲线。
不同时扩大到50/60epoch，不改学习率、优化器、scheduler、loss、class weight、
onset30目标、IGNORE、输入、TCN结构、graph scope、decoder、matching或selector。

仍使用原 merged-event Validation threshold sweep 和 checkpoint key
（F1、recall、threshold）；当前研究不按bin指标重选checkpoint。
新模型仍必须按原算法在Validation选择自己的阈值，不能套用旧seed的数值阈值。

固定seeds42/17/2026，每个均执行，不增加seed或挑最佳seed。
相同冻结Fit/Validation输入和时间切分，不读取Test遥测、预测、标签或指标。
另开工作树/协议及三个新run目录，不覆盖现有run或checkpoint。

## 3. 需要保存的诊断产物

- 每轮checkpoint、原始Validation logits/score/time/index、当轮merged阈值和指标。
- 实际LR、Train BCE与graph正则、有效bin计数；保留原batch平均loss口径。
- 每轮使用当轮原merged阈值的bin指标，作为诊断，**不参与本实验选择/早停**。
- POS/NEG/IGNORE高分比例、连续高分段持续时间、FP/day、完整GT及分层计数。
- 各seed的选择结果和两个预定时间半段结果，不只报告全局最好的seed。
- source/config/input/weight hashes、完整预测锁及失败账本。

保存产物是审计补充；不得改变RNG顺序、loader/batch补齐或训练数值路径。
前三个seed原已完成epoch的数值若无法复现，要先解释差异，不能直接归为预算收益。

## 4. GO / NO-GO

沿用复制前已有开发门槛，基准为window-causal C0在同一Validation的
P/R/F1=.991046/.724922/.837348，GT=2901：

1. 三个seed**各自**在最终冻结bin输出上P≥.98、R增益≥.05、F1增益≥.02。
2. single-onset R下降≤.03；两个既定时间半段R增益各≥.03。
3. 不因GT/context/ranking failure删分母；batch/peer独立、窗口时间和输入gate通过。

只有旧停止点之后的checkpoint带来可复现的Val恢复，才支持“早停过早”解释。
仅Train loss继续下降、某个seed改善或同一Val上挑中好epoch，都不足以宣布稳定提升。
若三seed仍不能共同过门槛，预算方向NO-GO，不根据Test追加epoch、改阈值或挑seed。
通过也只是开发GO；下一阶段仍需未使用的时间段/数据确认。

## 5. 后续独立方向：checkpoint selector（条件执行）

只有预算诊断完成后，再预先冻结 **仅改变checkpoint选择指标** 的方案。
优先复用同一30epoch轨迹已封存的全部权重和Val logits：每epoch阈值仍用原
merged算法，分别以merged F1与最终bin F1选择checkpoint，其余规则相同。
这样比较相同训练轨迹的选择规则，不能同时修改阈值算法或训练budget。
若要同时改为bin threshold sweep，应作为再下一项单独实验，不混入selector效果。

当前旧run只保存一个best checkpoint及其Val logits，无法在现有产物上完成
逐epoch selector比较，也不能宣称已经证明bin-best epoch与merged-best epoch不同。
全部开发选择均复用Validation，应如实披露；不使用本轮Test选择哪个后续方案。

## 6. 停止条件

本轮已有模型Test比较完成后，冻结其结果。下一轮最多先执行以上一个预算诊断，
再按条件评估一个selector变量。如果仍不稳定、或无新的未用验证数据，建议
Scientific Freeze并转入论文中的失败机制、因果性边界与GAIA结构限制说明。
当前68D+XGB RCA及Train导出的25.621s回溯配方维持已冻结的历史候选身份。

## 7. 实际来源

- `../Ada-MGAD-e2e-v2-c0onsettcn/experiments/p6/c0_onset_tcn/tcn-v1-seed42/{training_log,validation_selection}.json`
- `../Ada-MGAD-e2e-v2-c0replica/experiments/p6/c0_onset_tcn_replication/{seed17-v1,seed2026-v1}/{training_log,validation_selection}.json`
- `../Ada-MGAD-e2e-v2-c0onsettcn/scripts/p6/run_c0_trigger.py`
- `../Ada-MGAD-e2e-v2-c0replica/configs/e2e/gaia_p6_c0_tcn_replication_v1.json`

这些日志的digest和逐epoch曲线由最终比较报告另行记录。

## 8. 实施绑定

新工作树 `Ada-MGAD-e2e-v2-c0budget`，分支 `experiment/p6-tcn-training-budget`，
基于旧复制源码 `2e16c8e`。每个seed保留原进程的线程数和threshold workers：
42为8，17/2026为4；分别设置PYTHONHASHSEED，独立进程。
原训练源码仅新增可选observer接口及读取LR；AST删除这三项明确审计补充后
必须与原源码完全相同。Observer每轮验证Python/NumPy/Torch/ CUDA RNG不变。
旧epoch的阈值、selection key、merged指标和旧选中epoch的Val logits必须精确复现；
loss和权重固定绝对容差1e-6，同时报告是否exact。任何gate失败均保持INCOMPLETE，
先调查来源或实现差异。禁止将早期漂移解释成预算收益。
运行中不改变本树HEAD、config、源码或输入。源在执行前提交。
