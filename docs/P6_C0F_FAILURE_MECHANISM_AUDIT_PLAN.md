# P6-C0F Failure Mechanism Audit 实施方案

- 日期：2026-09-18（Asia/Shanghai）。
- 设计版本：`p6_c0f_plan_v1`；状态：`DESIGN_READY_FOR_IMPLEMENTATION / EXECUTION_NOT_STARTED`。
- 依据：用户审阅的下一步设计及本轮修订；配套 [P6 研究路线](P6_RESEARCH_ROADMAP.md)。
- 本文件定义下一轮审计实现，不代表已运行审计，也不将已见 Test 后的设计称为原 C0 的事前预注册。
- 当前任务仅保存文档；未来 implementation/run 以当次用户指令为准，不由本文件自动启动。

## 1. 目标与非目标

目标：在 C0 模型、阈值和原评价协议不变的情况下，拆分低召回的发生位置，量化标签/episode/匹配的结构限制，登记证据支持的现象与仍未确定的机制。

不以“证明 IGNORE 有问题”“证明模型已达到上限”或“确保进入 C1”为目标。允许结论为混合机制或证据不足。C0 的正式 verdict 始终保留 BORDERLINE。

允许：只读现有输入；新目录写审计输出；补齐冻结模型的 Fit/Validation 前向分数；对既有 Test 预测和 GT 做明确标记的诊断计算；合成 fixture 测试。

禁止：训练/反向传播、新 checkpoint、checkpoint 选择、校准重拟合、替代阈值扫描或 selection objective 试验、架构/标签/episode/matching 修改、raw preprocessing、Test 模型推理、RCA、正式指标覆盖或原 verdict 回写。

这里“只读”指 frozen inputs 和原实验不被改变，不是禁止审计目录写文件；“不重训”也不等于完全没有前向推理。

## 2. 输入身份与源版本

### 2.1 固定源

```text
repository: /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
inspection HEAD: b3a3ceb61bd22af594fdbdb3721380e9679f70f2
C0 recorded execution commit: cedc4a2bd7492933a8295067c8075e631cbf3df9
C0 root: experiments/p6/system_event_trigger/
first-run history: experiments/p6/system_event_trigger_v1_label_lag/
P5 baseline: experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440/
```

| 绑定对象 | 固定值 |
|---|---|
| checkpoint | `checkpoint/best_validation_event_f1.pt`（相对 C0 root） |
| checkpoint SHA-256 | `6a4317d6414774e1ca6be4e418ed3d47d678adfab1f0155e96e04aa3fa9981f5` |
| selected epoch | `8`，产物中的零基 epoch index；本轮不再选 epoch |
| threshold | `0.9998264908790588` |
| trigger config | `configs/e2e/gaia_p6_c0_system_trigger.json` |
| trigger config SHA-256 | `a2c42beb19f5a234ab74958cf3db32301524f909d5605001bdaeb9c141e3b4f8` |
| base config | `configs/e2e/gaia_p5_v3_preprocessing_v2.json` |
| base config SHA-256 | `25abf4e8008e90e19a51131f1a2ee7420b337fd275dbfbacf9504c2bc5e2ee35` |
| AD data manifest SHA-256 | `3b228061fd66b659bfa1f7564d4b9ffe5129be06b702b58573f67003d61634c2` |
| GT registry | `artifacts/p5/v3/protocol/gt_event_registry.csv` |
| GT registry SHA-256 | `7bc1b8b9c99164d0df004b72f47977b5413070c033834babac33931a27a0324b` |
| grid / history / tolerance | `30s / 300s / 60s` |
| prediction time | `prediction_available_time = target_bin_end` |
| episode rule | 连续 positive prediction timestamps 合并，起点为首个 positive |
| matching | causal maximum-cardinality、minimum-delay、一对一；`0 <= delay <= 60s` |
| inference configuration | 复用原记录：CPU、seed 42、torch threads 8、batch size 32；完整环境与 sampler 设置另行记录 |

原 C0 文件的文档化 SHA-256 快照：

```text
manifest.json             cb222bb5c21d20d8a7546e8b1ee1b4882daaac5d4a397d0e07655fe646cf9794
validation_selection.json b5167c8d5df9cd9eebd7812cb76733799aa3f7829f3fe1a2c5cdb0ec51715a24
test_metrics.json         2968632cc4da1b807d976d8bb1c449f9d755cb1c15d2ee43bcd440b74946ce5b
split_audit.json          e5c60574f4c36ba5c7ebddbc869ede70f6f9aa83cf35c46fd611dbfba310463e
split_manifest.json       1cad441a2d207379ffd8838bc2d655e9c68f3e03b6f4f7e666413a0b33aca193
test_predictions.csv      985623e06f5d56db9c2c536b8dd65226967ae1c25f36c9a4924ae9e38d5bafab
test_episodes.csv         30bacbeecbee4b9f738fc6330fa9d8df5c1adc08e37fb1ee8476f1b47735399b
test_matching.csv         011332f3537ee6f7d044e9da23df1d88f617fb71d3a552354343db6fd91c9c77
```

执行前核对真实文件；不匹配时停止受影响路径，不能把文档哈希静默改成当前值。其他数组/graph 的完整绑定从 C0 `manifest.json.source_artifacts` 读取并记录验证状态；未校验的对象不能标作通过。不得用“路径相似”代替同一来源。

审计实现应记录自身 commit、源码文件 SHA、配置、依赖版本、线程数和实际启动参数。当前 HEAD 与原执行 commit 不同，应保留两者；不能误称新审计运行在原 commit。

### 2.2 数据总体

| split | 时间块 `[start_ms,end_ms)` | windows | complete GT | `>300s` GT |
|---|---|---:|---:|---:|
| Fit | `[1625133600000,1626440400000)` | 43550 | 7443 | 362 |
| Validation | `[1626440400000,1626963120000)` | 17414 | 2901 | 125 |
| Test | `[1626963120000,1627747200000)` | 26127 | 5787 | 229 |

标签仍从原 label-legal event population 构造；完整 event metric population 仍使用原分配/跨界 purge。两者的差别必须保留，不能为了便于分析只用 matched events 或忽略跨界事件对标签状态的作用。

Test 官方计数：`TP=4198, FP=16, FN=1589, episodes=4214`。Validation 官方计数：`TP=2124, FP=13, FN=777, episodes=2137`。这些数字是源产物的核对值，不是通过修改解析/过滤去满足的硬编码目标。

## 3. 实施阶段与运行边界

### C0F-0：静态准备、源码与合成测试

1. 读当前上下文，核对 git status，保留已有文件与实验目录。
2. 按第 2 节锁定输入、完整源码依赖和环境；列出缺失数据，先实现纯函数与合成测试。
3. 定义完整输出 schema、nullable 字段、排序/tie-break、精确上界求解策略及资源预算。
4. 将实现代码、测试、schema 和实际命令固定后，再进入真实数据计算；审计实现不得以 Test 性能选择分析规则。

### C0F-1：冻结 Fit/Validation 分数补全

当前仅存在 `test_predictions.csv`；`training_log.json` 和 `validation_selection.json` 是历史汇总，不是逐窗口分数，也不能从它们伪造轨迹。

独立 loader/driver 只读取 frozen input，复用原模型的 eval/no_grad、窗口顺序、padded sampler 去重和 score 算法，为 Fit/Validation 写每个合法窗口的原始 logit 与 score。禁止调用会写回 C0 root 的 `run_evaluation`，禁止附带执行 Test inference。

Validation replay：固定同一 threshold，不重新枚举阈值；TP/FP/FN/episodes 应与选定记录完全一致，P/R/F1 用原指标算术核对（浮点显示误差限预先固定为绝对 `1e-12`）。保存 delay 的比较结果。若主计数或阈值绑定不一致，停止后续基于新分数的合并结论，不修改容差或改选 epoch。原 Validation 的逐行分数/匹配未保存时，应明确逐行 replay 无法验证，不能用汇总一致冒充逐行一致。

Fit 分数用于 in-sample 诊断；Validation 分数用于已参与选择的开发期诊断。二者都不是新的独立测试证据，也不是严格 OOS anchors。

### C0F-2：既有 Test 预测审计

直接使用已有 Test score/logit/binary prediction、episode 和 matching。可以按原规则在审计目录核对它们的算术与身份，但不得重跑 Test 模型或把核对输出写回原正式产物。

若 Test 的记录本身不一致，记 STOP，不用重新生成的结果替换它。可继续独立静态分析，但整体状态必须保持 PARTIAL/STOP。

### C0F-3：分解、轨迹、oracle 与解释

只有相应输入通过身份检查后，执行第 4–7 节。所有阶段均写入新 run 目录；完成标记在 required outputs 验证后原子发布。失败、超时或部分输入缺失保留显式状态，不能产生完整成功声明。

## 4. 漏检账目：先分清失败发生在哪一层

对各 split 的每个 complete GT，使用完整系统时间轴与全局 matching，按以下顺序分配唯一类别：

| code | 判定 |
|---|---|
| `MATCHED` | 原协议匹配成功 |
| `NO_LEGAL_PREDICTION` | `[onset,onset+60s]` 内没有合法 prediction timestamp |
| `BELOW_THRESHOLD` | 有合法 timestamp，但没有 `score >= frozen_threshold` |
| `NO_NEW_EPISODE` | 有 positive timestamp，但区间内没有新的 episode 起点 |
| `MATCHING_COMPETITION` | 有合法 episode 起点，但全局一对一 matching 未分配给该 GT |

这些是可核查的发生位置，不是全部因果解释。为每个类别保存依据：合法 timestamp 数、positive 数、候选 episode IDs、实际匹配对象、onset 前 episode 是否已开始。

不允许逐事件重新开始 episode construction，或对每个 fault 分组独立重做正式 matching。那会切断原 episode、释放被其他事件占用的匹配槽，改变正式问题。

每个 split 必须满足：类别计数之和 = complete GT 数；`MATCHED=TP`；其余类别总和 = FN；unmatched episodes = FP。非法时间/重复身份/不可能状态属于 implementation error，不能塞进某个失败类别。

## 5. 分数轨迹、响应时间与并发标记

### 5.1 时间和统计定义

每个事件使用实际 `delta_ms=prediction_available_time-onset_ms`，保留所有可用网格点。至少包含 onset 前 300 秒到 onset 后 300 秒，并为 long event 追加整个持续区间的可用观测。

主窗口：`[-300s,0)`、`[0,60s]`、`(60s,120s]`、`(120s,300s]`、`[onset,end)`。每个窗口输出有效观测数、max、median、score/logit 分位数、positive 数及 missing/censor 状态。

展示可包含 first causal prediction 及其后第 1/2/3/4/6/10 个网格步，但必须携带真实 delta；不能把网格起点强行对齐 onset，不插值伪造精确 `+30s/+60s` 样本。主结论基于上述时间区间，而非人为对齐的插值点。

主分层包括 `<=15s`、`>300s`、所有 fault types（重点列出 memory/login）、single/multi-onset、重叠/相对隔离事件；空组报告 `n=0, metric=null`。

### 5.2 三种时间不能混用

分别记录：

```text
first_positive_time      = onset 后第一个 score >= threshold 的合法时间
first_new_episode_time   = onset 后第一个新 episode 起点
official_matched_time    = 原全局 matching 分配的 t_hat（可能没有）
```

原始 binary decision 继续按保存的 score 和冻结阈值核对。logit 和 `logit(threshold)` 的 margin 只作数值尺度诊断，不用重新计算的 logit 比较替换原 float32/CSV decision。

long-event 首次响应分带：`[0,60]`、`(60,120]`、`(120,300]`、`(300,event_end)`；另记完整可观察区间内未响应及右删失。positive 与 new-episode 分别给表，不能把 late positive 计入原 TP。

### 5.3 边界与最大值偏差

- 短事件 `[onset,end)` 内可能完全没有网格点；记 NA/no-grid-observation，不记 score=0 或 never。
- 正式 60 秒响应窗口不因短事件结束而提前截断。
- 长事件观测到 `min(event_end,split_end,score_coverage_end)`；不跨 split 拼接分数，不填补缺失。
- “Never”只能针对明确、完整的观察区间；不完整者是 censored，须列 n 和原因。
- `max during event` 随持续时间和观测点数增长，不能直接跨 duration 解释为更强可检测性；固定时长窗口为主要比较，整个 event 的 max 为辅助。

### 5.4 防止把其他事件的响应归给长事件

为每个 trajectory timestamp 标记：其他事件在过去 60 秒内的 onset 数、其他 active events 数、其 IDs/类型仅作为审计元数据、当前窗口是否处于其他事件的 recent-onset 区间。

至少报告全部事件和无其他 recent-onset 干扰的可观察片段/案例。相对隔离子集不得替代正式总体，也不能据小样本外推所有 long events；给出覆盖数。既有 frozen mask/observability channels 如可直接解读，可作辅助缺失统计；不新增特征选择或 raw 全量扫描，不把“mask 存在”解释为故障信息可辨识。

晚分数响应最多支持“冻结模型在该上下文出现较晚高分”。若重叠污染或数据不足，归属为 UNKNOWN。持续低分也不能排除输入不可辨识、目标不合适或模型容量等多种解释。

## 6. 结构诊断：两个上界和一个标签解码参考

### 6.1 先核对碰撞总体

精确记录 onset 的整数毫秒、grid origin、bin ID、是否恰落网格边界；分别统计 label-legal 全域与各 split complete-GT 总体。

当前 Test 报告有 821 multi-onset events，histogram 中 782 个事件来自 2-onset bins、39 个来自 3-onset bins；`782/2+39/3=404` 仅为待按身份核验的 bin 数。`404/821` 不是未经证明即可套用的完整协议上限。

不把 onset bin 等同于 episode 起点 bin；考虑合法预测时刻、恰在边界的 onset、跨 bin 竞争与相邻 positive 合并。不能人为增加“每 onset bin 只能一个 TP”的额外规则，再称其为原协议上限。

### 6.2 R_label：标签直接解码参考，不是性能上限

用原 prediction-time grid 和原三态标签构造：POSITIVE→1、NEGATIVE→0、IGNORE→0，再复用原 episode/matching。IGNORE→0 是明确指定的解码约定，不是“完美模型必须输出 0”。保留时间轴，不能删除 IGNORE 行。

输出 episode 数、P/R/F1、匹配和碰撞分层，仅解释标签直接解码与事件评价是否对齐。若另研究 IGNORE 自由赋值/不同约定，须在运行前固定并单独标记，不能挑选最有利的结果。

反例必须进测试：事件 onset=10s/70s，30 秒网格上的 recent-onset 标签在 30/60/90/120s 全为 1，直接解码仅一个 episode；预测 1/0/1/0 可得到两个合法 episode。故 R_label 不能当作模型召回上限。

现有 metadata 记录 Test positive regions=3432，而实际模型 TP=4198，进一步说明二者不是同一优化目标；C0F 仍需按实际合法网格复核，不把该 metadata 比较当作已执行的 oracle 结果。

### 6.3 U_grid：合法预测时刻的松弛容量上界

每个合法 prediction timestamp 是一个容量 1 的预测槽，与满足 `0 <= t-onset <=60s` 的 GT 连边，做 maximum-cardinality 一对一匹配。暂不施加相邻 positive 的 episode 合并约束。

这是时间网格与匹配层的松弛上界，通常不能被一个真实 binary sequence 同时实现。记录匹配 witness；不误称可以部署的 detector。

### 6.4 U_episode：完整协议的可达上界

定义：在同一个合法 prediction grid 上枚举/优化 binary sequence，经原 episode construction 和 matching 后，使 TP 最大。

```text
U_episode = max_binary_sequence TP(match(episodes(binary_sequence), GT)) / N_GT
```

可以用精确 DP/整数规划等实现，但必须证明约束与 episode 起点等价。例如完整连续 30 秒网格上，两个独立 episode 起点不能相邻；pulse-only 序列可为候选起点集合提供 witness。对时间缺口与 split 边界按实际 episode 实现处理，不能套用不成立的简化。

求解输出包含 candidate slots、assignment、binary witness、原 matching 重放结果、求解器版本、状态、上下界、gap、runtime。只有证明最优并通过原 pipeline witness 检查才标 `EXACT`。超时可报告 `BOUNDED`，保留可行下界和有效上界；不得把 incumbent 当作精确 ceiling。

应满足（相同总体、未删事件）：`observed recall <= U_episode <= U_grid <= 1`。若只有 bounds，使用对应区间；若实际模型违反声称的上界，应停止并排查求解/总体契约，不能删除反例。

每个 split 先求全局结果，再分层汇报同一 witness 的匹配。总体最优解可能不唯一，分组匹配数不能天然视为该组独立上界。另求 multi-only 上界时明确它忽略了其他组竞争，各组独立最大值不能相加或声称同时可达。

使用 GT 求上界是诊断本身，不产生可用于推理的 oracle trigger，也不能修改原 episode/tolerance/GT 来改善上界。求解预算/停止规则必须在真实数据求解前固定；无法在预算内精确求解是允许的限制。

## 7. Validation 一致性、证据分级与决定

所有账目/轨迹/结构结果都分 Fit/Validation/Test 输出。Validation 已用于选择，因此“一致”只说明该现象在开发段也存在，不构成独立复现或泛化证明。

同时给出 fault × duration × service 的交叉计数与空单元；`memory n=204` 与 `long n=229` 的重叠、CPU 的分布和稀有组必须按实际总体解释。

| 证据级别 | 内容 |
|---|---|
| `CONFIRMED` | 核对通过的身份、计数、匹配位置、标签逻辑、已证明的结构界 |
| `OBSERVED` | score/logit 轨迹、late response、margin、开发段与 Test 的模式 |
| `HYPOTHESIS` | IGNORE、目标、表示、可观测性或不均衡导致上述现象 |
| `UNRESOLVED` | 并发污染、缺失、删失、小样本或多解释无法区分 |

阈值附近高分只能作为 operating-point 候选证据；不运行替代阈值扫描，也不据此保证 R2 成功。不将整段低分自动分类成已证实 representation failure。

审计输出可以包含多个机制标签；不要求三选一。结果按 [路线图第 4 节](P6_RESEARCH_ROADMAP.md#4-c0f-之后的决定) 提出下一步：STOP、结构限制下的 C1 提案、C0R2 候选，或保留未决。所有情况下 C0 原 `BORDERLINE` 不变。

不新增“离上界差 5% 就 GO”之类基于已见 Test 的验收阈值。若未来 C1 需要准入规则，另行定义研究范围及接受的限制，不能重写 C0 的历史 gate。

## 8. 建议代码边界与现有复用点

以下新路径是建议接口，尚不存在；实施者可在不改变契约的情况下调整布局，提交时记录最终路径。

```text
scripts/p6/audit_c0_failure.py            独立审计 driver，显式 stage/output-dir
src/e2e/system_trigger_failure_audit.py   纯账目/轨迹/并发分析
src/e2e/system_trigger_capacity.py        标签参考/结构界/witness 验证
tests/test_p6_c0f_failure_audit.py         合成契约测试
tests/test_p6_c0f_capacity.py              小实例穷举与精确界交叉验证
```

复用时先直接阅读源码：

- `src/e2e/system_trigger.py`：prediction-time label、split、onset density、固定阈值评价。
- `src/e2e/system_trigger_data.py`：frozen arrays、sample identity、prediction times。
- `src/e2e/system_trigger_model.py`：原模型与 scalar head。
- `scripts/p6/run_c0_trigger.py`：`infer_trigger_logits`、`score_dataset`、sampler 去重与原 Validation 选择记录。
- `src/e2e/event_detection.py`：原 episode、matching 和 event metrics。
- `src/e2e/protocol.py`：原 event assignment 与边界规则。

尽量复用无写入的纯函数；不得直接调用训练/正式 evaluate 主流程。若提取公共 helper，必须验证原 C0 路径语义未改变，不能顺手重构模型或修正式产物。

建议 driver 分为 metadata prepare、Fit/Validation score extraction、artifact analysis、capacity solve、finalize；默认动作不得隐式推理/训练。输出目录必须显式传入且为新目录，拒绝 C0/P5/共享输入树和已有完整结果目录。恢复若实现，须严格绑定输入、配置和已完成 shard，不能近似复用。

## 9. 审计产物契约

新输出根建议为 `experiments/p6/c0f_failure_audit/<unique_run_id>/`；这只是目录约定，不是本次已创建的实验目录。

```text
run_state.json
input_manifest.json
audit_config.json
source_integrity.json
scores/{fit,validation}.csv
validation_replay.json
split_population_checks.json
event_failure_ledger.csv
event_score_trajectories.csv
event_response_summary.csv
stratified_summary.json
confounding_tables.json
oracle/{label_reference,grid_bound,episode_bound}/
    summary.json
    witness files
    replay_validation.json
evidence_ledger.json
decision.json
final_report.md
completion_manifest.json
```

`scores` 至少包含 split、sample_index、prediction_available_time、window bounds、原始 logit、score、固定 threshold/binary、三态 label。预测输入/输出与审计 GT 元数据分开，禁止把 root/fault 字段传入模型。

`event_failure_ledger` 一行一个 complete GT，包含 case_id、split、onset/end、分类依据、三种响应时间、eligible/positive 数、候选/已匹配 episode IDs、并发与删失状态。原 case_id 和 GT 不改；空值用明确 nullable 语义，不用 0/-1 冒充真实时间。

`event_score_trajectories` 保留 actual delta、score/logit、窗口覆盖与其他事件标记。`confounding_tables` 保留各单元计数，空组也显式输出。

`evidence_ledger` 至少包含 claim_id、statement、grade、split/population、artifact+field 定位、limitations。`decision` 分别记录 `c0_original_verdict=BORDERLINE`、audit completion、mechanism findings、unresolved、recommended_next_scope；不能只有一个把所有含义混在一起的 passed 布尔值。

`completion_manifest` 包含 required outputs、大小/SHA、输入绑定、实际命令、依赖/线程、测试结果及日志引用、每个 oracle 的 exact/bounded 状态。先校验后发布；bounded oracle 或无法归因的机制可以作为显式限制完成审计，身份/实现校验失败不能标 COMPLETE。

## 10. 必要验证与停止条件

### 合成测试（先于真实审计）

1. prediction-time 与 bin-start 错位 30 秒能被识别；毫秒整数和恰落边界的 onset 保留。
2. 同 bin 多事件、相邻 bin 多事件、跨 split/时间缺口的 episode 语义正确。
3. onset 前已开始的 positive episode 与新的阈值 crossing 区分。
4. 一个 episode 可候选多个事件时仍使用同一全局匹配，竞争类别可核查。
5. IGNORE 保留时间轴；短事件无网格点是 NA；不跨 split 填值。
6. late response 来自另一个 onset 时被标记，不能变成原事件 TP。
7. 右删失与完整区间无响应区分，类别账目保存完整分母。
8. 第 6.2 节反例验证 R_label 不是上限。
9. 小实例穷举 binary sequence，对照 U_episode 的 optimum/witness；验证 U_episode <= U_grid，及多个总体最优解的分层解释。
10. 缺失/改动输入、重复 ID、已有输出目录、误指向 frozen root、求解超时均显式失败或降为 BOUNDED，不静默覆盖。

测试不得触发 full model training、全量推理、真实 Test evaluation 或原目录写入。不得用完整旧测试套件的副作用绕过运行边界。

### 实际运行检查

- 输入 SHA、配置和 checkpoint 精确绑定，执行前后 frozen artifacts 未改变。
- 三段 window/GT identities 与原 split manifest 一致；计数是核对结果，不是通过删行满足的目标。
- 补齐分数每个合法窗口唯一；Validation replay 主计数一致，Test 只复用已有 score。
- 原 Test episode/matching/TP/FP/FN 在审计中得到精确账目。
- 轨迹空值、边界、并发和分层 n 保留，正式分母未替换。
- witness 经原 episode/matching 验证；未证明最优不标 exact。
- 结论明确区分证据等级，C0 verdict 未变化，无未声明的 Test-dependent tuning。

身份/时间/实现契约失败：STOP。缺少某个 split 的完整分数：PARTIAL，不声称三段一致性已完成。求解器未证明最优：BOUNDED 并报告 gap。无因果识别证据：UNRESOLVED，不伪装成审计失败，也不强行给出机制结论。

## 11. 交付给下一位 coding agent 的执行说明

先实现并测试 C0F 所需的隔离 driver 和纯分析模块；提交可审阅的源码、schema、验证记录和实际可执行命令。**当前仓库没有 C0F 执行入口，不得把建议路径写成已经可运行的命令。**

得到执行 C0F 的任务后，按 C0F-0 → C0F-1 → C0F-2 → C0F-3 顺序推进，使用唯一 output directory，记录运行阶段和资源预算。不在本轮添加 R2/C1/RCA 训练，不因审计较慢或上界难求而放宽原协议。

最后报告：改动文件、测试、实际完成阶段、产物路径/哈希、未完成项、原 C0 verdict、证据支持的下一步。用户若只要求实现，不把“实现完成”解释成已授权真实数据全量运行。
