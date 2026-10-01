# 确定性环境下的配对训练预算诊断

## Material Passport

- Date: 2026-10-02；Train/Validation development，非独立确认。
- 原预算执行源码：`f987a1b`；3个run均epoch0因历史轨迹复现门槛FAIL而停止。
- 当前问题：同seed、同真实Fit batch的GPU训练自身是否确定，以及早停是否过早。
- 所有旧输出、源码和失败现场保留；原Test结果和原稳定性NO-GO不改写。

## 诊断证据与解释边界

在V100/Torch1.12/CUDA11.3上，8个真实Fit batch、相同初始化、相同seed重复两次：
原默认设置下初始权重差=0、第一步logit差=0；后续最大logit差1.043081283569336e-7，
最终权重最大差1.4901161193847656e-8。开启确定性算法后上述差异全为0。
原`seed_everything`只设置种子，没有开启确定性算法。这个probe确认内部非确定性存在；
不能证明它解释了全部历史epoch0差异或跨seed稳定性差异。
旧运行未归档完整GPU/runtime/算法环境；旧seed不能作为当前环境的精确训练对照。

## 新的受控比较

每个seed42/17/2026，独立运行两个fresh arm：

| arm | max_epochs | patience | checkpoint/threshold |
|---|---:|---:|---|
| control | 30 | 8 | 原merged Validation exact sweep与原F1/recall/threshold key |
| budget | 30 | 30 | 完全相同 |

两个arm共同固定`torch.use_deterministic_algorithms(True)`、`cudnn.deterministic=True`、
`cudnn.benchmark=False`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`；GPU/线程/输入/源均一致。
这些设置是新配对实验共同的复现条件，不把与旧run的变化归因于训练时间或模型提升。
42 threads/workers=8，17/2026=4，OMP/MKL与torch线程数相同。
不改模型、目标、IGNORE、loss、class weights、optimizer、LR/scheduler、decoder或matching。
不扩大30epoch，不改变seed列表，不读取Test、不选best seed。

每轮记录既有Validation输出、权重、实际LR及batch损失口径；observer必须不改RNG。
历史轨迹比较保存为`historical_trajectory_comparison`，仅描述，不是当前paired gate。
两个fresh arm重叠的每个epoch，loss、阈值、metrics、logits和权重必须**精确相同**。
任何确定性操作不支持、peer轨迹不一致或source/input变更均为INCOMPLETE，优先诊断。

## GO / NO-GO与后续

早停不足得到支持，要求budget在control实际停止点之后得到更好的merged选择结果，
并且三个seed最终bin结果共同通过原开发门槛（P≥.98、Rgain≥.05、F1gain≥.02，
single-onset退化≤.03、两个预定半段Rgain各≥.03）。否则预算方向NO-GO。
若两个arm选择相同早期checkpoint，不能把确定性control结果本身的变化称为预算收益。
所有开发GO还需同一Fit-only冻结scorer的Validation transfer或单独native Fit-OOS RCA；
完整E2E F1@1增益≥.02才支持两阶段改善。旧scorer只有Fit根因标签，但其训练锚点仍为
旧batch-conditioned retrospective；transfer和native结果必须分开命名。

预算完成后最多做一项条件checkpoint selector研究，复用同一30epoch轨迹，
只将merged F1 selector改为bin F1，保持每epoch merged-derived threshold。
旧历史三seed、两个当前arm都完整报告。不根据Test调threshold、offset或选epoch。
预算和selector均失败时停止新增模型，建议Scientific Freeze。
