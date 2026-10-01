# 起点监督与独立bin输出：首轮结果

## Material Passport

- Mode: run / descriptive development evaluation
- Source: completed onset30 run ac94227; decoder analysis d4bf971; frozen repaired NN
- Human read status: not attested
- Evidence grade: reused Validation development; no Test/RCA/E2E result
- Date: 2026-10-01

## 已完成结果

Validation GT2901、window17414，GT身份和窗口完全相同，包含所有漏检。

| 方案 | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 修复后recent-onset60 NN |2103|19|798|.991046|.724922|.837348|
| E1 onset30，原merged decoder |2057|131|844|.940128|.709066|.808410|
| E2独立bin，固定E1阈值 |2320|3484|581|.399724|.799724|.533027|
| E2独立bin，预声明Val重选阈值 |2369|3554|532|.399966|.816615|.536945|

E1训练10epoch，epoch1（从0计数）为最佳；threshold .2392304241657257。
代码/输入执行前后SHA核验、完成清单逐文件校验和指标重放通过。真实32窗
batch1/2/32最大logit误差7.15e-7，低于预声明atol1e-5，同行扰动误差0。
44项范围测试通过。E1/E2均未满足原定效果门槛，保留为NO-GO开发结果。

E1对基线paired：both-hit1974、baseline-only129、candidate-only83、both-miss715，
净损失46，DeltaR -.015857、DeltaF1 -.028938。memory14→15/110、长事件
16→17/125不构成可靠改善；single-onset1947→1903/2521，multi156→154/380。

## 失败位置及解释

E1：matched2057、below-threshold273、no-new-episode250、competition320、
no-legal1。相对基线，no-new减少80、competition减少10，却被below-threshold
增加136抵消。这些是失败发生位置，不能直接诊断telemetry不可辨识或唯一模型机制。

独立bin输出把一个持续响应转换为多次预测。每次预测照常参与一对一匹配与FP，
召回提升不代表新事件识别已改善：固定点告警5804，FP3484，整体F1下降。
E2只使用E1封存checkpoint；E1 checkpoint是在merged decoder下选择的。
不能将这个负结果推广为所有按独立bin协议训练/选择checkpoint的方法都无效。

补充输出覆盖核查（仅开发诊断）：5804阳性bin中old IGNORE61、新onset POS1724、
old POS但已转NEG646、old normal NEG3373。IGNORE不是主要重复输出位置。
所以不自动重训IGNORE→NEG消融，也不将“覆盖监督缺口”当成已证实原因。
标签任务的时间精度、模型的分数响应形态与输出选择协议仍值得区分验证。

## Fit-only标注解码参考

完整Fit GT7443，固定标签分数0/1、阈值.5，无模型推理：

| 标注解码方式 | TP | FP | FN | R |
|---|---:|---:|---:|---:|
| recent60 merged |4922|0|2521|.661292|
| onset30 merged |5870|0|1573|.788660|
| onset30 independent bins |7027|0|416|.944109|

这只显示目标/decoder如何压缩标注。不是模型性能，也不是可达上界；模型能在
IGNORE或其他非标注正bin产生预测，完整协议的最大匹配容量还需另行求解。
该诊断在提交前执行，具体script/input hashes封存，不把base HEAD当完整执行源码。

## 后续动作

按已准备协议进入固定小型TCN编码器比较：保持onset30目标、旧IGNORE、输入/
training/merged decoder，单独更换encoder。TCN分数封存后再独立复用E2输出消融。
若候选通过检测门槛，再做seed/时间复核以及完整cohort的68D+XGB E2E开发。
半监督正常预测支线先做Fit内部可行性筛查，不混入当前候选，不重评Test。

限制：共享70%Train preprocessing含Validation无标签输入；复用Validation；
Trace parent可用性未完全证明；old filled mask语义保持冻结。没有新Test、
detector-OOS anchors或RCA/E2E结果，不能将旧回顾性E2E指标视为本轮新结果。

## 原始产物

- E1：`/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2-c0onset/experiments/p6/c0_onset_development/onset30-v1-seed42/`
- E1对比：同worktree `onset30-vs-baseline-v1/`
- E2：同worktree `bin-decoder-v1/`
- Fit标注参考：本worktree `experiments/p6/c0_onset_development/fit-label-decoding-reference-v1/`
