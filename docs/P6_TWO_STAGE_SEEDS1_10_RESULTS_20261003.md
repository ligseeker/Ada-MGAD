# 两阶段 seed 1–10 重复实验复盘

## Material Passport

- Origin: academic-research-suite / experiment-agent，validate 模式，角色在本进程内执行。
- Date: 2026-10-03（Asia/Shanghai）。Version: validation_v1。
- Status: **ANALYZED**。产物完整性和指标算术通过；没有重新训练或复现完整实验。
- Evidence grade: 固定数据、固定 RCA scorer 条件下的初始化波动；reused-Test descriptive。
- 当前建议：**Scientific Freeze 当前版本**；不选择最佳 Test seed，不自动开启后续实验。
- 本轮仅分析已完成产物、保存小型审计记录；没有新模型推理、阈值搜索、图表或论文正文。

## A. 当前最终实验结果

### A1. Run、源码与完成状态

运行集合：`experiments/p6/two_stage_repeats/seeds1-10-v1-20261002T213410/`。
协议：`P6-TWO-STAGE-FROZEN-XGB-SEEDS1-10-V1`。
分析开始时 branch=`e2e-v2`，HEAD=`98aaf10a8827a9c90f99ba3be4ef975f5876f973`，工作树干净。
训练/预测执行 commit=`3e1fdea2134b1d48dd8d2ef59f64a04f11bc6fcd`。
HEAD 是十次预测全部封存、Test label join 之前的 lock commit。

十个 detector stage、十个 RCA stage、统一 evaluation 和 summary 均 COMPLETE。
未把 partial run、旧 seed42/17/2026 或历史模型混入这十次统计。

唯一变化是 detector seed=1–10。固定监督 WindowCausalTCNTrigger、onset30、原 IGNORE、
逐窗口图、AdaBelief/StepLR、max_epochs=30、patience=8、batch=32。
checkpoint/tau 按 merged Validation event F1/recall/tau 选择；最终报警为独立 bin。
每次均使用同一冻结 68D XGB、同一 Train-derived `anchor=t_hat−25621ms`；RCA 不重训、无额外 scaler。

| cohort | 每 seed 的固定数量 |
| --- | ---: |
| Detector-Fit 完整 GT / 输入窗口 | 7,443 / 43,550 |
| Detector-Validation 完整 GT / 输入窗口 | 2,901 / 17,414 |
| Test 完整 GT / 输入窗口 | 5,787 / 26,127 |
| Test detected TP | 4,751–5,215，平均 5,039 |
| 合法 matched RCA n | 4,750–5,212，平均 5,036.8 |

### A2. 十次总体结果

下表均为十个 seed **先各自计算，再取均值和样本标准差（ddof=1）**。
单位为 %；± 后是百分点的标准差，MRR 也按 ×100 显示。
这些不是独立数据置信区间，不把重复使用的 GT 扩成 57,870 个独立事件。

| 指标 | mean ± SD | min–max |
| --- | ---: | ---: |
| Detector P | 98.12 ± 1.28 | 95.26–99.65 |
| Detector R | 87.07 ± 3.08 | 82.10–90.12 |
| Detector F1 | 92.25 ± 2.11 | 88.94–93.99 |
| matched RCA AC@1 | 90.03 ± 0.44 | 89.58–91.11 |
| matched RCA AC@3 | 99.83 ± 0.03 | 99.79–99.87 |
| matched RCA AC@5 | 99.90 ± 0.01 | 99.88–99.92 |
| matched RCA MRR | 94.88 ± 0.23 | 94.64–95.42 |
| Full E2E P@1 | 88.30 ± 0.76 | 86.78–89.44 |
| Full E2E R@1 | 78.35 ± 2.53 | 73.82–80.84 |
| Full E2E F1@1 | 83.01 ± 1.59 | 80.48–84.31 |
| Full E2E P@3 | 97.91 ± 1.26 | 95.11–99.39 |
| Full E2E R@3 | 86.89 ± 3.04 | 81.98–89.87 |
| Full E2E F1@3 | 92.05 ± 2.07 | 88.80–93.74 |
| Full E2E P@5 | 97.98 ± 1.26 | 95.16–99.49 |
| Full E2E R@5 | 86.95 ± 3.05 | 82.01–89.96 |
| Full E2E F1@5 | 92.12 ± 2.09 | 88.85–93.83 |

### A3. 全部 seed，不筛选结果

| seed | Test TP/FP/FN | legal RCA n | Detector F1 | RCA AC@1 | E2E F1@1 | E2E F1@3 | E2E F1@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5165/92/622 | 5162 | 93.53 | 89.89 | 84.03 | 93.34 | 93.39 |
| 2 | 5100/23/687 | 5098 | 93.49 | 89.88 | 84.00 | 93.31 | 93.36 |
| 3 | 5105/76/682 | 5103 | 93.09 | 89.85 | 83.61 | 92.87 | 92.94 |
| 4 | 4751/78/1036 | 4750 | 89.51 | 89.94 | 80.48 | 89.37 | 89.41 |
| 5 | 4827/240/960 | 4826 | 88.94 | 91.11 | 81.02 | 88.80 | 88.85 |
| 6 | 5183/92/604 | 5180 | 93.71 | 89.90 | 84.20 | 93.46 | 93.55 |
| 7 | 4786/160/1001 | 4785 | 89.18 | 90.49 | 80.69 | 89.03 | 89.09 |
| 8 | 5101/18/686 | 5098 | 93.54 | 89.58 | 83.75 | 93.31 | 93.40 |
| 9 | 5215/95/572 | 5212 | 93.99 | 89.75 | 84.31 | 93.74 | 93.83 |
| 10 | 5157/86/630 | 5154 | 93.51 | 89.93 | 84.04 | 93.31 | 93.36 |

完整原始结果：[per_seed.csv](../experiments/p6/two_stage_repeats/seeds1-10-v1-20261002T213410/summary/per_seed.csv)、
[statistics.json](../experiments/p6/two_stage_repeats/seeds1-10-v1-20261002T213410/summary/statistics.json)。

### A4. Failure decomposition

每 seed 以全部 GT=5,787 为基础。下面的平均值可以为小数；没有把假警加入 GT 漏诊比例。

| 完整 Top-1 诊断位置 | 平均事件数 | 占 Top-1 FN |
| --- | ---: | ---: |
| 检测漏检 EVENT_MISSED | 748.0 | 59.71% |
| 已匹配、合法上下文，但 root rank=2–10 | 502.5 | 40.11% |
| 已匹配但 RCA context invalid | 2.2 | 0.18% |
| 合计 Top-1 FN | 1,252.7 | 100% |
| Top-1 SUCCESS | 4,534.3 | — |
| Detector false alarm（单独统计） | 96.0，范围18–240 | — |

检测 FN=748 的进一步定位：BELOW_THRESHOLD 平均251.8、MATCHING_COMPETITION
493.2、NO_LEGAL_PREDICTION 3、NO_NEW_EPISODE 0。
这些是账本中的失败位置，不能直接解释为因果机制。

RCA 排名错误=502.5，其中 **rank2–3=493.9，rank4–5=3.6，rank6–10=5.0**。
原账本 `ROOT_OUTSIDE_TOP1` 仅表示 rank2–3；不能单用这一列当作全部 Top-1 排名失败。
@3、@5 距离 detector F1 的均值仅约0.20、0.13个百分点，已经主要受检测限制。

### A5. 训练是否完整、是否只是时间不足

| seed | selected epoch index（从0起） | 实际完成 epoch 数 | 停止原因 | 原 merged Val F1 | 固定 tau 下 Val bin F1 |
| --- | ---: | ---: | --- | ---: | ---: |
| 1 | 1 | 10 | patience8 | 84.32 | 58.89 |
| 2 | 1 | 10 | patience8 | 84.49 | 93.31 |
| 3 | 2 | 11 | patience8 | 83.38 | 87.02 |
| 4 | 1 | 10 | patience8 | 83.75 | 90.81 |
| 5 | 1 | 10 | patience8 | 82.12 | 56.16 |
| 6 | 2 | 11 | patience8 | 83.77 | 87.55 |
| 7 | 1 | 10 | patience8 | 82.92 | 56.25 |
| 8 | 2 | 11 | patience8 | 83.32 | 92.19 |
| 9 | 1 | 10 | patience8 | 84.12 | 87.46 |
| 10 | 1 | 10 | patience8 | 83.29 | 90.35 |

共完成103个训练 epoch。每次在记录内最佳 epoch 后连续8次没有更好的 merged key，
随后正常 early stop；最佳权重已按 checkpoint SHA/replay 绑定。
Train loss 首轮约1.162–1.164，末轮约.676–.687；下降的训练 loss 未对应更好的选择指标。
这十次没有运行到30轮，不能据此宣称十个 seed 在后期绝无恢复。
但此前三个 seed 的完整30轮同轨迹预算实验已覆盖两次降LR，仍没有 late recovery；
因此当前没有积极证据支持单纯增加 epoch/patience。
见[既有预算实验](P6_TRAINING_BUDGET_FINAL_RESULTS_20261002.md)。

### A6. 与已经封存结果的关系

历史 window-causal C0 + backdate transfer detector F1=.833450、E2E F1@1=.740689。
新十次均值分别为.922501、.830124，描述性差值约+8.91、+8.94个百分点。
这是不同完整方法/训练运行的历史比较，包含 TCN、onset 目标与 decoder 的变化，
不能归因为某一个模块的单变量改善，且双方都使用同一个反复查看的 Test。

历史 TCN42/17/2026 bin 的 backdate E2E F1@1=.839497/.842191/.816463。
新十次落在相近水平，并更完整暴露了 seed 波动；不是又一次新方法提分。
原 C1 A/B/C 及 C−B paired 改善属于旧 conditional-logit 三臂协议；
本次没有 B/C 成对干预，不能从十次 transfer 结果重新推导 C−B 的因果收益。
历史表见 [detector19](p6_frozen_test_comparison_20261002/detector_19.csv)、
[RCA/E2E40](p6_frozen_test_comparison_20261002/rca_e2e_40.csv)、
[C1/C2](P6_C1_V2_AND_C2_RESULTS.md)。旧 NO-GO 不追溯改成 GO。

## B. 结果是否可信

**当前运行完整、指标算术有效。证据范围是 frozen-scorer transfer 稳定性。**

1. 在读取数值结果之前调用 `check_family`：130份源文件、3,803个输入文件约3.15GB
   全字节哈希、环境、scope/global committed lock、20个 stage seal 均通过。
   evaluation 42个文件、summary 2个文件的 seal 和文件集合也通过。
2. 预测锁文件生成于2026-10-03 04:29:38 +0800，提交于04:29:45；
   evaluation 在之后生成并于05:25:12完成。现存文件时间与源码的先锁后 label join 路径一致。
   文件 mtime 是辅助证据，不能替代 committed lock/源码控制，也不是外部可信时间戳。
3. 十次 GT 身份、服务、fault 与冻结 registry 完全一致；没有只保留成功检测 cohort。
   Test window indices/timestamps 同一；逐预测检查 threshold positive 集合、
   ranking ID/anchor、`t_hat−25621ms`、Test ±300s context 边界、十服务完整排名。
4. 特征全部为有限值且 shape=`[N,10,68]`；非法上下文保留在预测与 E2E 分母。
   model SHA、scaler=None、offset、seed、30/8、model args/cohort/runtime 跨十次一致。
5. 每个 seed 的实际 batch independence 和 selected Val checkpoint replay 均有通过记录。
   审计重放记录中的 checkpoint key/early stopping，核对已有 merged Val matching 计数。
   未新增逐 epoch 阈值 sweep；此项不宣称重新优化验证所有历史 threshold 候选。
6. 独立从匹配表和完整排名重建 root rank、GT/prediction 闭合、P/R/F1/AC/MRR、
   全 failure counts，并复算 CSV 与全部 summary mean/std/min/max，一致。
   所有候选仍按预注册 causal60s 一对一规则评价，未根据结果改 GT 或容差。
7. 未发现本次实现中的 cohort、scaler、anchor 使用错误、源漂移或新的 Test 模型选择。
   源码/元数据审查不能证明主机外的所有人为行为；不作此类额外保证。

仍需保留的限制：

- Test 已多次复用；十次不是十套新数据，没有独立确认性显著性检验。
- 70% Train 的冻结 preprocessing 包含 Detector-Validation 的协变量，
  因而不是严格 Detector-Fit-only preprocessing OOS；本次没有偷偷重拟合 scaler/schema。
- RCA scorer 来自旧 C1 的3,225个 supervision-OOS Train case，其 anchors 继承旧 batch graph 限制。
  新 detector 的逐窗口 gate 通过，不会修复旧 RCA training anchors 的历史身份。
- RCA 未按新 seed 重建 Fit-OOS anchors 或重训；只能称冻结 scorer 的 transfer。
- RCA 使用 post-anchor 300s 数据；Trace parent 当时可用性未证明；不能称零延迟严格在线诊断。
- 同一 V100/CUDA11.3/cuDNN8302，deterministic/benchmark/algorithms flags 均false。
  seed 控制不等于跨硬件/跨 kernel 的逐 bit 重现。

完整审计：[integrity_receipt](p6_two_stage_repeats_20261003/integrity_receipt.json)、
[audit.json](p6_two_stage_repeats_20261003/audit.json)、[只读审计源码](p6_two_stage_repeats_20261003/audit.py)。
所有新增分析文件在原 immutable run 之外，未覆盖原产物。

## C. 为什么检测和完整 E2E 波动，而 matched RCA 相对稳定

完整召回逐 seed 满足：

`E2E R@k = detector R × (legal matched / detector TP) × matched AC@k`。

这是一条指标恒等式，不是因果估计。detector R 的 SD 为3.08pp，AC@1 的 SD 仅.44pp，
context 损失每次只有1–3个；目前总体初始化变化主要体现在检测召回。
seed5 的 AC@1最高却不是最佳 E2E，直接说明更容易的 matched cohort 不能补偿漏检。

对相同 anchor 有更直接的证据：5,650个独立合法 anchor 中累计45,678次跨 seed 重复比较，
**68D feature bytes 与完整排名全部相同**。没有发现 RCA 提取随着 seed 随机抖动。
十次共同 legal-matched 的4,512个 GT 中，4,221个 anchor 完全相同，291个跨 seed 改变；
这个共同 cohort 上 AC@1范围90.69%–91.31%。
共同 cohort 是事后形成的容易检测子集，只用于条件诊断，不能替代完整 GT 指标。
该证据也不能证明所有幅度的 anchor shift 都鲁棒；逐维68D Train-only shift 敏感性仍未完成。

Validation 上选择规则和最终报警单位仍错位：seed1/5/7 的 merged Val F1约.82–.84，
对应最终 bin F1仅.56–.59，FP=3326/3463/3432。
其 merged 候选数2125/2211/2176，独立 bin 候选数5925/5948/5910；
连续高分在 merged 中只承担一次告警成本，接入 bin 后每个阳性 bin 都承担匹配/假警成本。
本次严格沿用提前冻结的规则，属于已知设计限制，没有更改协议来挽救结果。
只有原始分数分布/时间差异的观察，不能凭这些计数将所有新增假警归因于某个输入故障。
Train/Val 的 NEG 表示没有该目标的新 onset，不等于系统健康。

## D. 当前最大瓶颈

### D1. 对完整微平均效果：检测优先，RCA Top1仍有残差

Top-1漏诊中约60%位于检测，约40%位于排名。@3/5基本贴近检测指标。
检测有540个 GT 在全部十个 seed 中都未检出，733个在部分 seed 检出；
这说明同时存在持续失败与初始化敏感案例，不能只通过选 seed 消除。
RCA rank2–3仍占几乎全部排序错误，68D XGB 的 Top1判别尚非完美；
但当前没有新的受控实验显示换表示/加模块会改善完整两阶段结果。

### D2. 对科学泛化：偏斜和少数类型支持比总体分数更严重

Test login_failure=5,546/5,787（95.84%）；mobservice1/2根因=5,586/5,787（96.53%）。
因此 AC@3/5≈100%同时受到候选容易性和 cohort 偏斜影响。

| fault | 完整 GT n | 平均 detector R | 平均 E2E R@1 |
| --- | ---: | ---: | ---: |
| login_failure | 5546 | 89.97% | 81.58% |
| memory_anomalies | 204 | 17.50% | 2.16% |
| cpu_anomalies | 12 | 8.33% | 0% |
| file_moving | 16 | 66.25% | 25.00% |
| normal_memory_freed | 4 | 0% | 0% |
| access_permission_denied | 5 | 42.00% | 26.00% |

完整 E2E macro root R@1=**17.95%±1.39pp**（10组均有GT），
macro fault R@1=**22.46%±7.03pp**（6组均有GT）。这些是宏召回，不是宏F1。
micro R@1=78.35%不能据此推广成对十服务/六fault都可靠。
少数类型 n=4–16，单个 case 会显著影响宏指标；它们的差异不作显著性或泛化宣称。
memory 的平均成功检测35.7例却仅平均Top1正确4.4例，表明这类故障检测和RCA都弱。
完整分层见 [均值表](p6_two_stage_repeats_20261003/strata_mean.csv)、
[十次逐层表](p6_two_stage_repeats_20261003/strata_per_seed.csv)。

### D3. 30s网格/重叠是结构限制，但不能替模型承担全部责任

一对一账本中平均493.2个 competition 并不等于493.2个不可避免的结构损失。
作为只读协议容量诊断，允许**全部26,127个30s slot都发预测、完全忽略FP代价**，
按等宽 causal60s interval 的最早截止贪心匹配，最多可匹配5,766/5,787个 GT。
在同一 slot 集和匹配协议下至少21个GT无法被一对一覆盖；实际748个平均FN远多于这个下界。
全部实际阳性 slot 的相同独立计数也逐 seed 重放了正式 detector TP。

这是用GT计算的乐观协议容量，**不是可部署模型或99.64%召回的效果承诺**：
全部slot报警有巨大FP代价，也没有考虑真实信号可观测性与分类难度。
它只反驳“所有competition都必然由30s粒度造成”。短事件的信号稀释、多事件混合、
同一时间窗多个根因、缺少Train支持仍可能限制检测与Top1；需要Train/Val证据分辨，不能仅由Test归因。

## E. 最值得保留的1–2个最小方向

**当前建议停止模型搜索。**若以后确有研究资源，按以下次序重新立项；下列实验均未执行。

1. **仅改变阈值选择的报警单位。**保持每 seed 已选 checkpoint、原分数、训练、RCA/offset不变，
   只将该 checkpoint 的 threshold selector 从 merged event F1改成最终独立 bin F1。
   假设：连片高分在 merged 目标中被低估的告警代价，导致部分 seed 的 bin precision 崩溃。
   先用已保存 Train/Validation 产物验证，无需重训，不读取Test分数或指标来选tau。
   先前预算实验改的是 checkpoint key并沿用merged tau；此前的bin auxiliary则已经在旧运行中
   实际做过这个threshold改动，没有获得共同稳定改善。当前十次尚未做此辅助对照，
   补做只能增加初始化覆盖，不能包装成新方法；**因此不建议为了刷分再重复它**。
2. **只审计少数类型的训练支持和可观测性。**先检查Fit/Val每fault/root的合法样本数、
   缺测与原始信号、已有68D的分布和anchor偏移，而不马上增加模型模块。
   假设：少数类型的持续漏检/错排主要受训练支持或可观测性约束。
   若确有可学习的Fit信号，再另行冻结一种单变量改动；目前没有足够证据推荐具体RCA特征删除或重训配方。

这些方向是在复盘当前已查看数据后提出的，现有Validation上的改善只能算开发证据。
最终泛化结论需要真正未使用的时段/数据；不能继续用当前Test选择胜者。

## F. 每个方向的验证和GO/NO-GO

以下是**未来立项方案**，需要在执行之前冻结；不是本轮预注册门槛，不追溯给十次结果判GO。

| 方向 | 唯一变量/验证 | 建议GO条件 | NO-GO条件 |
| --- | --- | --- | --- |
| H1 报警单位一致的tau | 固定十个checkpoint，仅在对应已有Validation分数上换为bin F1/highest-tau规则；保留全部seed、完整2901GT及固定时间前后半段；属于旧辅助试验的补覆盖 | 全seed P≥.98；相对本次固定tau，各seed bin F1不下降、十次平均F1至少+.02；前/后半段与single-onset R均不下降超过.03。通过也只进入新holdout验证 | 任一seed不满足；改善只在重挑checkpoint、改epoch/loss/offset或查看Test后才出现；停止，不反复加约束刷Val |
| H2 支持/可观测性审计 | 仅Fit/Val分层及已有原始信号/特征审计，无训练改动 | 对拟优化的类别，在Fit和Val都存在合法、可观测且可区分的信号；事先冻结所需样本量/不确定性预算后，再提出一种单变量实验，并具备未使用的确认数据 | 类别缺Train支持、没有可观测信号、少量case不足以判断，或没有新确认数据；优先补数据，不堆RCA模块 |

H2 尚未指定效应检验所需样本量，因此**不能据当前计数批准模型实验GO**；
数据支持审计的输出应先给出该设计所需样本量与可行性。这是开展审计前允许保留的未知量。
不自动运行上述任一方向；不再次跑当前Test。

## G. 是否继续优化还是Scientific Freeze

**建议冻结当前两阶段方法及结果，结束本数据上的epoch/seed/模块搜索。**
十次补足了初始化稳定性的描述：micro结果较强，但检测仍有约8pp召回范围，
Validation独立bin precision存在明显seed/时间问题，minority能力很弱。
这支持“当前接入方法在该偏斜Test上达到稳定量级的总体效果”，
不支持“跨类型可靠”“原生新detector-RCA OOS”“严格在线”或“独立确认优于已有方法”。

freeze是保存当前方法、全部十次和失败证据，不是挑seed9作为最终模型。
若必须确定部署用的单一checkpoint，应使用提前规定的seed或新的Validation规则，
另行说明部署选择；不得由这个Test表选出最好seed。
科学上更值得重新开启的是新的时间块/数据与明确的告警单位、少数类型支持。
本轮按用户要求只保存分析，不生图、不写论文正文。

## 统计推断检查（11项覆盖）

| 模式 | 本轮检查与处理 |
| --- | --- |
| Simpson's paradox | 并列micro、fault/root分层与宏召回；没有把总分推出所有组改善 |
| Ecological fallacy | seed均值和group均值不代替个体case能力；附分母与范围 |
| Selection/Berkson bias | matched/co-common仅条件诊断；完整GT保留全部漏检 |
| Collider bias | conditioning on detected可能关联anchor/难度；不以matched AC推导总体因果收益 |
| Base-rate neglect | 明列95.84% login与96.53%两mob根因，少数类别n单列 |
| Regression to the mean | 全seed1–10，不选最好seed，不由历史seed42推广必然稳定 |
| Survivorship bias | 十次完整，失败context/FP/FN不删除；partial历史不并入 |
| Look-elsewhere effect | Test已反复使用；没有新阈值、offset、feature搜索或最优Test模型选择 |
| Garden of forking paths | 固定协议后全量评价；新假设单列为事后未来开发，不回填预注册 |
| Causal language from observational data | seed归因使用固定控制与指标恒等式，历史方法差值仅描述性；失败位置不称根因 |
| Reverse causality | 先封存预测再标签匹配；未将GT时间/服务送入本次RCA预测，未由Test反向选tau |

Coverage: **11/11检查完成**；发现并声明条件cohort、base rates、重复Test与验证报警单位风险。
未计算p值/独立数据CI，不作显著性和样本外泛化宣称。

## 算术审计复现

以下仅复算已有CSV/JSON/特征一致性，不训练、不做模型推理、不搜索tau。
必须选新输出目录，脚本拒绝覆盖：

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
PYTHONDONTWRITEBYTECODE=1 /home/zhangll24/.venvs/ada-rca-supervised-baselines/bin/python \
  docs/p6_two_stage_repeats_20261003/audit.py \
  --run experiments/p6/two_stage_repeats/seeds1-10-v1-20261002T213410 \
  --output /tmp/p6-two-stage-arithmetic-audit-new
```

该脚本应在完整seal/hash校验后使用；本轮已做的全字节校验与环境/committed lock核验
另存于integrity_receipt，算术脚本不替代`check_family`。
