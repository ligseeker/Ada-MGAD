# 同一训练轨迹中的早停预算比较

## Material Passport

- 2026-10-02；Train/Validation development，三个固定seed42/17/2026。
- 首次历史精确复现尝试：f987a1b，三个run在epoch0停止，INCOMPLETE。
- 确定性fresh-pair尝试：4c1abff，仅启动seed17 control/budget；因性能回退取消，
  尚无完整epoch，不作效果结论，原目录/输入锁/日志保留。
- 原默认CUDA有微小跨运行非确定性；这不使已锁定预测的旧指标算术无效。

## 最小研究设计

使用原backend默认值（cudnn deterministic/benchmark及deterministic_algorithms均False），
保留原模型、优化器、loss、targets、IGNORE、graph、LR/scheduler、数据、selector和matching。
每seed只训练一条完整30epoch轨迹，patience30，保存全部每轮权重和既有Val原始分数。
用原patience8逻辑在这条**同一实测轨迹**的前缀计算停止点和选中的checkpoint。
两种预算共享完全相同的随机实现/数值路径，不需要在两次物理CUDA运行间复现。
这是同一轨迹中的预算对照，不能称两个独立训练run，也不是旧run的继续训练或精确复现。

数学依据：在原训练循环中patience只用于epoch末的break；任何停止点以前的更新、
阈值、checkpoint key与worse_count逻辑均相同。AST剥除记录接口后须与原训练源完全一致。
Observer必须不消耗RNG、不重跑前向。每轮先保存权重和原始分数，后记录诊断。
前缀重放仅消费原merged key和原tau；绝不按bin指标或Test选择停止点。

prefix_control_selection必须逐字段重复重放一致，checkpoint及raw score哈希绑定到
停止前的实际epoch；不重新优化threshold，不把GPU跨运行非确定性藏为exact replay。
原默认算法的跨运行复现限制明确保留。旧seed输出只作历史参考。

## 决策

先比较patience8前缀与完整30epoch的原merged selector结果。
预算解释要求停止点后的checkpoint发生恢复，并且三个seed各自通过原门槛：
P≥.98、相对逐窗口C0 Rgain≥.05、F1gain≥.02，single-onset R降幅≤.03，
预定两个Validation时间半段Rgain各≥.03。GT=2901，全部GT/FP/failure保留。
若完整预算仍选相同早期epoch，不能把Train loss下降称为早停不足。

预算NO-GO后只允许一次已计划的bin checkpoint selector比较；沿用每epoch原merged tau。
该selector和预算均是复用Validation的开发结果，不作独立统计确认，不选择best seed。
检测三seed未GO时不启动新的native Fit-OOS RCA；仍报告此前冻结完整E2E结果。
如GO，先用同一Fit-only冻结回溯XGB做Val transfer，另行区分native Fit-OOS。
预算/selector均NO-GO后建议Scientific Freeze。没有新Test推理或Test调参。
