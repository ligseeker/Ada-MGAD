# P6 后续研究路线与 coding-agent 交接

- 更新日期：2026-09-18（Asia/Shanghai）。
- 交接所依据的仓库 HEAD：`b3a3ceb61bd22af594fdbdb3721380e9679f70f2`，分支 `e2e-v2`。
- 文档性质：经用户要求保存的研究设计与实施交接；不是新增实验结果，也不是已执行的预注册实验。
- 本次工作范围：仅文档化。C0F 尚未实现或执行；C0R2、C1、C2 均未启动。
- 下一步：优先实现 [P6-C0F Failure Mechanism Audit](P6_C0F_FAILURE_MECHANISM_AUDIT_PLAN.md)。
- 阅读顺序：[当前上下文](GAIA_P5_CURRENT_CONTEXT.md) → 本文 → C0F 实施方案。

## 1. 当前研究状态与结论边界

| 阶段 | 状态 | 已有证据 | 不能据此声称 |
|---|---|---|---|
| P5 完整 E2E | 已完成 | event F1 0.7751；matched GT/detected-anchor RCA AC@1 0.9125/0.4657；Diagnosis F1@1 0.3611 | E2E 定位问题已解决 |
| P6-A 失败审计 | 已完成 | anchor 改变伴随明显特征/排序变化；offset 模拟支持时间上下文敏感性 | 找到了唯一因果机制或可直接采用 Test 最优 offset |
| P6-B0/B0R 分数分解 | 已完成，Outcome C | reconstruction objective 本身受节点标签条件化；只换分数不能解除标签耦合 | reconstruction-only 是无监督通道；严格数值 replay 已通过 |
| P6-C0 修正版 | 已完成，aggregate gate PASS，正式 verdict BORDERLINE | Test P/R/F1 = 0.9962/0.7254/0.8395；长时事件和 memory 分层不通过 | 原验收 GO；所有故障均改善；新完整 E2E 已验证 |
| P6-C0F | 设计已保存，待实现/执行 | 本文及独立实施方案 | 审计结果或机制已经确定 |
| P6-C0R2 | 条件性后续选项，未冻结实验细节 | 仅保留修订方向 | 现在就可以改阈值、目标或训练 |
| P6-C1/C2 | 研究框架已保存，未冻结完整执行协议 | anchor-domain 对照与 E2E 评价设计 | 已获准运行，或能够直接复用旧结果作为新结果 |

P6-C0 的精确定义是 **root-service-label-agnostic supervision**：训练目标不含根服务身份，不读取节点异常标签、不加载 Ada-MGAD checkpoint；仍消费十个具体服务的 telemetry 与服务图。它有系统级事件监督，不是无监督方法，也不能声称模型完全不知道服务身份。

P6-C0 的模型/阈值由 Detector-Validation 选择，采用 50/20/30 时间划分；P5 采用 70/30、Train 选择。两者的总体差值是历史对照，不能单独归因于取消根服务监督。当前正式 Test 是同一批 5,787 个 GT events。

当前 Test 已用于发现问题，包括 C0 首轮的 30 秒标签错位。后续设计不得宣称在从未见过 Test 的条件下产生；只用 Fit/Validation 选择新参数，也不会恢复这批 Test 的独立确认地位。

## 2. 研究路线

```text
P6-C0：保留原正式 BORDERLINE
        |
        v
P6-C0F：冻结模型的失败机制审计
        |
        +-- 身份/实现契约错误 --> 停止相关分析；单独记录修复方案
        |
        +-- Validation 支持选择规则修订候选 --> 另行冻结 P6-C0R2
        |
        +-- 持续响应不足/机制未决 --> 登记假设，必要时另立 P6-C0R2
        |
        +-- 可核查的结构限制与已知能力边界
                    |
                    v
           可提出 P6-C1 接口研究协议
           明确接受 Stage-1 限制；不把 C0 改判 GO
                    |
                    v
           P6-C1：受控 anchor-domain RCA 对照
                    |
                    v
           P6-C2：冻结预测后的完整 E2E 评价
                    |
                    v
           证据包归档与结论冻结（可以包含失败和限制）
```

不存在“最可能 F-A，所以默认进入 C1”的出口。也不要求先证明所有失败的唯一因果机制；要求的是明确已知事实、未决解释、已知实现问题和下一阶段接受的限制。

## 3. 下一阶段 C0F 的范围

C0F 不训练、不改 checkpoint、不扫描替代阈值、不改标签/架构/episode/matching，不运行 RCA。冻结输入只读，允许在新的独立审计目录生成产物。

主要交付物：

1. 正式漏检的互斥账目：无预测机会、低于阈值、episode 起点受阻、匹配竞争。
2. Fit/Validation/Test 分数与 logit 轨迹、首次 positive/new-episode 时间、边界与并发事件标记。
3. 标签直接解码参考、时间网格松弛上界、完整 episode 协议可达上界及可核查 witness。
4. duration/fault/service/onset-density 的分层与交叉计数。
5. 证据分级、未决问题与后续建议；保留 C0 原 verdict。

当前只持久化了 Test 的逐时间点预测；Fit/Validation 需要冻结模型前向推理才能完成轨迹审计。不得运行原 C0 `evaluate` 入口取得这些分数：该入口还会重跑 Test 并写回原目录。执行边界、数据结构、测试和停止条件见 [C0F 实施方案](P6_C0F_FAILURE_MECHANISM_AUDIT_PLAN.md)。

## 4. C0F 之后的决定

| 证据状态 | 允许提出的后续方案 | 保留的限制 |
|---|---|---|
| 身份、时间、标签或 Validation replay 出错 | 独立 bug 修复/复核计划；受影响分析停止 | 不静默修复 frozen artifacts，不用新结果覆盖旧结果 |
| 精确容量分析支持结构损失 | 固定当前 trigger，单独制定 C1 接口研究协议 | C0 仍为 BORDERLINE，低召回仍计入 E2E |
| Validation 存在阈值附近响应，支持 operating-point 候选解释 | 单独制定 C0R2 的选择规则实验 | 不能从近阈值直接推断降低阈值会改善整体效用 |
| 当前模型持续响应不足 | 登记 objective/representation/observability 候选解释，必要时制定 C0R2 | 不能只凭低分确诊 representation failure |
| 混合机制或证据不足 | 保留 UNRESOLVED，可结束审计并报告限制 | 不强制归入 F-A/F-B/F-C，不为得到明确出口补调参数 |

原 [C0 协议](P6_C0_SYSTEM_EVENT_TRIGGER_PROTOCOL.md) 的 BORDERLINE 规则只允许该轮 failure audit，不允许 RCA。C1 必须是单独的后续研究协议与执行任务，不是追溯性放宽 C0 gate。本文件的保存不启动任何阶段；后续以用户当次授权和已经授权的范围为准，不重复索取已有授权。

## 5. 条件性 C0R2：只保留方向，不能现在实现

候选方向包括：

- selection objective：在开发数据上固定宏平均或最小组召回约束；必须同时约束误报和总体表现。
- system-level objective：onset 与 sustained abnormal-state 的辅助目标；仍不使用 node/root-service supervision。

这些不是已选方案。R2 前必须冻结确切目标、分组依据、小样本处理、选择规则、候选集合、指标、split、停止条件和新 run ID。若采用 root/fault 标签参与开发期选择，应明确它们进入了哪一条选择路径，不能继续宣称所有开发决策都不含这些身份。

Fit/Validation 决定 R2 的方案与参数；当前 Test 不用于比较候选并挑选赢家。同一 Test 的后续结果应作为复用测试集上的后续评价报告。独立确认需要另有未参与开发的时间段/数据或预先隔离的外层评价设计；如不具备，就保留证据限制，不伪造新 holdout。

## 6. P6-C1：Detector-aligned RCA 受控研究

### 6.1 研究问题和固定项

研究问题：在固定 Stage-1、相同案例集合和相同 RCA 模型形式下，训练与评价的 anchor domain 对齐能否改善 detected-anchor 定位？

固定 68D Z2 特征定义、四通道、服务候选全集及顺序、Conditional Logit 形式与超参数、原始数据处理规则、RCA 窗口和 failure semantics。这里的“Conditional Logit 不变”指模型形式/超参不变；比较不同训练 anchor 时，模型权重仍需分别在 Train 上拟合。

P6-C1 的新 detected anchors 来自新的 trigger，案例总体也可能不同。不得直接把新对照称为对 P5 `0.9125 → 0.4657` 数值差的复现，或宣称解释了 P5 下降的全部原因。

### 6.2 OOS anchors 的真实性

当前 checkpoint 对 Fit 的预测属于 in-sample；对 Validation 的预测也受该集合的 checkpoint/threshold 选择影响。两者不能直接标作严格 OOS anchor。

严格 forward-temporal anchor generation 的框架：

```text
过去的 detector fitting 区间
    -> 过去的 detector selection 区间
    -> 后续未参与 fitting/selection 的 anchor-generation 区间
```

每个 fold 的模型、阈值、校准及 schema/scaler/vocabulary/graph 等拟合来源只能位于该 fold 的过去。当前共享 preprocessing 在原 70% Train 上拟合，可能包含 fold 的未来段；复用它不能自动证明完整流水线严格遵守前向时间隔离。

C1 前必须明确选择：建立严格前缀拟合的隔离输入，或仅研究 supervision-OOS 并披露 frozen preprocessing 的跨 fold 信息。后者不能冒充完整 forward-OOS；不得为满足前者而未经授权重做共享预处理。

辅助 fold detector 可能需要训练；这属于另行授权和冻结的 C1 工作，不属于 C0F。最终用于 Test 的 Stage-1 checkpoint/threshold 继续固定。fold 数、时间范围、purge、训练/选择规则、少数类不足处理都必须在 C1 实验前明确，不能按 Test 表现决定。

### 6.3 同 cohort 的主对照

先锁定一批具有合法 OOS-detected anchor，且 GT/detected 两种 RCA context 都合法的 Train cases。主比较的两种训练输入使用完全相同的 case IDs、标签、样本权重与候选服务；不能用“全部 GT Train”对照“只被检出的 detected Train”并把差异全部归因于 anchor。

| arm | RCA Train anchor | RCA evaluation anchor | 作用 |
|---|---|---|---|
| A | GT | GT | 同 cohort 的 oracle-anchor 诊断 |
| B | GT | detected | anchor-domain 不匹配对照 |
| C | OOS detected | detected | anchor-domain 对齐对照 |

A/B 使用同一个 GT-anchor 训练模型；B/C 使用同一批 Test matched cases 做配对比较。所有 primary comparison 使用共同合法 cohort；原始完整总体及各类排除仍单独报告。

scaler 的建议主设计：在上述共同 Train cohort 的 GT-anchor candidate rows 上拟合一次，同一冻结 scaler 用于两种训练 arm 和全部评价输入，使主干预限于 anchor domain。若另设各 arm 独立 Train-fit scaler 的方案，应在实验前列为次要对照，并说明它共同改变了 domain scaling 与正则化尺度，不混入主对照。

可另设 OOS-detected → GT 的对称诊断，或全部 GT Train 的历史参考；必须预先列出，不能看 Test 后追加有利结果。

### 6.4 预测、标签与窗口边界

- Train 的 OOS episodes 可在 Train 内做时间匹配，以获得 RCA 监督标签；保留未匹配 episodes 的账目。
- Test 先为所有合法 detected episodes 生成完整 ranking，并锁定预测/输入/模型，再加入 Test 标签做评价；不能先筛出 GT-matched episodes 才运行 RCA。
- root/fault 标签不得进入 Test 特征、ranking、anchor 校正或参数选择；GT-anchor arm 是明确使用 oracle anchor 的诊断，不是部署路径。
- primary matched-case AC@k/MRR 与全量 E2E 分母分开。missing/invalid ranking 不得从诊断分母删除。
- `[anchor-300s, anchor+300s)` RCA context 与 AD 的 300 秒历史窗是两个不同约束。OOS folds 和 Train/Test 边界需要同时考虑两者及 raw injection 跨界规则。
- RCA 使用 anchor 后 telemetry，故诊断可用时间至少包含相应后视等待；不能把 Stage-1 的 `t_hat` 直接报告成完整 RCA 的实时输出时刻。

### 6.5 C1 实施前还需冻结的项目

具体 folds、输入拟合范围、运行 ID、模型/代码/输入哈希、共同 Train cohort、scaler 选择、Test 预测总体、边界与失败分类、主指标、paired 分析方法、计算预算与停止条件。

上述项目未冻结时，只能准备代码接口、静态验证和合成 fixture；不能把本路线图当成已存在的正式 C1 执行命令。

## 7. P6-C2：完整 E2E 评价与科学结论冻结

C2 消费固定的 Stage-1 与 RCA 预测。评价前锁定预测、case universe、模型与表示、阈值、匹配规则、指标和 source hashes。

必须分开报告：

1. Stage-1 event P/R/F1、TP/FP/FN、延迟与分层覆盖。
2. 同一 matched-case cohort 的 RCA AC@1/3/5、MRR、root/fault macro 与各组 n。
3. 包含漏检、误报及 ranking failure 的完整 Diagnosis P/R/F1@k 和原始计数。
4. 从 onset 到 detector 输出、RCA 数据就绪、最终诊断输出的不同延迟口径。
5. paired 差值和证据限制；若报告不确定性，应预先考虑事件重叠/时间相关，不能默认所有事件独立。

Oracle-anchor 结果仅是诊断参考。不得合并 raw GT、放宽 tolerance、删掉难例或改变后验报告总体来提升分数。归档可以得到“有条件支持”“无增益”或“失败”；科学冻结指证据与结论固定，不等于方法成功。

## 8. 历史解释勘误与文档一致性

以下为对既有报告的解释性更正，数值产物与原 verdict 不改：

- `memory_anomalies n=204` 是 `>300s n=229` 的子集，不是完全相同的 229 个事件；重叠不能证明同一机制。
- Test 中 `<=15s` 包含 login_failure 和 cpu_anomalies，不能写只有 login_failure。duration/fault/service 关系用交叉计数逐项表述。
- “root-agnostic”限定于监督路径，不等于 telemetry/graph 不含服务身份。
- late score response 不是 telemetry observability 的证明；IGNORE 导致漏检仍是 Hypothesis。
- `3432` 个 Test positive regions 不是模型可达 TP 上限；模型已有 `4198` 个 TP，标签直接解码与 event recall 优化不是同一任务。
- 原报告的 `996` multi-onset bins 是 label-legal 全时间域统计，不能当成 Test-only bins；Test 的 821 multi-onset events 与约 404 个相关 bins 必须保留各自总体定义并在 C0F 核对。
- corrected C0 JSON 中 Fit negative bins 为 `25971`，pos_weight 为 `2.0137241218888113`；purged labels 为 positive/ignore/negative = `2/7/2`。旧正文的 `25970`、`2.0135`、`2/6/3` 不应继续作为实施输入。
- P6-B0 的 `1e-6` 数值 replay gate 仍为 FAIL。事件计数/指标一致不等于 raw scores 或阈值数值完全相同。

本次不重写历史结果报告，以保留原记录；新 agent 应使用本节解释勘误和机器可读产物，不能照抄上述错误表述。正式更正历史报告时应另留更正记录，不能修改原实验数值或伪装成当时已经具备的证据。

## 9. 交接完成标准

- 当前上下文明确指向本路线图和 C0F 方案。
- 下一位 agent 能区分：已有结果、待实现审计、仅有框架的后续实验。
- C0F 实现时依据独立方案交付代码、合成测试、准确命令和审计产物；执行与否服从用户当次任务范围。
- 不将本文件的保存视为全量预处理、训练、Test 推理、RCA 或 C1/C2 的执行授权。
- 没有用户另行请求时，不更新个人 memory、不提交或推送研究结果。

## 10. 来源

- [P6-C0 原协议](P6_C0_SYSTEM_EVENT_TRIGGER_PROTOCOL.md)
- [P6-C0 原结果报告](P6_C0_SYSTEM_EVENT_TRIGGER_RESULT.md)
- [P6-B0/B0R 审计](P6_B0_SCORE_DECOMPOSITION_AUDIT.md)
- [P6-A E2E 失败审计](P6_E2E_FAILURE_AUDIT.md)
- [C0 manifest](../experiments/p6/system_event_trigger/manifest.json)
- [C0 Test metrics](../experiments/p6/system_event_trigger/test_metrics.json)
- [C0 split audit](../experiments/p6/system_event_trigger/split_audit.json)
- [C0 Validation selection](../experiments/p6/system_event_trigger/validation_selection.json)

设计审查记录：Origin Skill `academic-research-suite / experiment-agent`；Mode `validate`；Date `2026-09-18`；Verification Status `ANALYZED`；Version Label `p6_roadmap_v1`。这是设计和现有证据分析，不是新增运行验证。
