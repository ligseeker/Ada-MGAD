# 30 epoch预算复盘与条件selector诊断

## Material Passport

- Mode: experiment-agent run/validate；development-only。
- Scope: 三seed冻结Train/Validation轨迹；不读取Test数组、分数或指标。
- Budget source: `fd25bff`（同一实测轨迹前缀对照；f987a1b历史复现失败、4c1abff性能回退取消均保留），独立`c0budget`树。
- 本协议在完整预算结果可用前冻结；预算源码/HEAD保持不变。

## 顺序与对照

1. 先验证三个训练run均完成30epoch、同一轨迹patience8前缀重放gate、输入/source gate、真实batch独立gate。
2. 用相同原merged selector对照同一实测轨迹的patience8前缀control与patience30 budget；旧run仅历史参考。每epoch阈值仍由merged exact sweep产生。
3. 如果预算不能使三seed共同通过，则仅在已封存的同一30epoch轨迹比较checkpoint selector：
   merged F1→bin F1，次级key仍为recall、threshold，完全相同key保留最早epoch。
   只读取权重/Val分数，不增加训练、不改threshold sweep或decoder。
4. 若预算已经GO，则selector候选不启用；可先做一次固定scorer Val transfer，再另行冻结native Fit-OOS RCA。

## 门槛与评价

Validation完整GT=2901；基准逐窗口C0为P/R/F1=.991046/.724922/.837348。
每个seed各自要求：P≥.98、R增益≥.05、F1增益≥.02；single-onset R退化≤.03；
两个预定等时长半段各R增益≥.03。三个全部通过才为开发GO，不选择best seed。
所有GT及预测保留；记录独立bin候选matching、failure ledger、FP/day、fault/root/duration分层。
时间分层复用全局matching，不在半段边界重新匹配。

旧停止点后的merged最佳checkpoint改善、以及三seed稳定性恢复才支持早停过早。
Train loss下降或bin selector开发效果改善不能倒推证明预算假设。
权重和分数在label join前做完整scope/prediction lock；Validation本身已用于选择，
因此所有变化为开发结果，interval不作为独立显著性证据。

## 两阶段与停止

已核验旧XGB的3225-case cohort全部早于Detector-Fit边界，没有使用Detector Val根因标签。
它可以用于固定scorer Val transfer，但训练锚点来自旧batch图，仅为开发/描述性诊断；
不能据此宣称native window-independent Fit-OOS RCA。
检测稳定性未GO时，不启动这项额外训练；既有Test完整E2E比较仍是历史冻结观察。
预算和单变量selector均NO-GO时建议Scientific Freeze，不追加epoch、seed或模块。

该control不是第二个独立训练run；不能声称跨运行逐epoch精确复现。历史复现失败和
确定性慢运行是诊断/INCOMPLETE记录，不算模型NO-GO。实际三条30epoch源码/输入固定。
