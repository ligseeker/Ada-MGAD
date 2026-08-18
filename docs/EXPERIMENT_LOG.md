# 实验日志

> 日志版本：`experiment_log_v1`  
> 最近更新：2026-08-18  
> 原则：历史只追加，不用新叙事覆盖旧限制

## 1. 证据等级

- `reproduced-current`：在当前分支、记录的提交与数据 manifest 上可复现。
- `artifact-verified`：存在完整原始产物和配置，但本轮未重新运行。
- `conversation-reported`：参考对话报告的结果，当前工作区缺少对应代码/产物。
- `planned`：实验设计，尚未执行。

任何论文表格只应直接使用前两类；第三类只能作为研究动机或待复现线索。

## 2. Legacy RCA Pilot（历史线索）

**证据等级：`conversation-reported`。**

参考对话称旧 `rca` 分支基于冻结 Ada-MGAD evidence 做窗口级 RCA，比较了 detector ranking、逐节点 MLP 与 Set-Attention。当前 clone 看不到该分支、配置、数据 split、seed、per-case predictions 或原始日志，因此以下数字尚不能由本仓库复核。

| 方法 | Macro HR@1 | Macro MRR | Login HR@1 |
|---|---:|---:|---:|
| Detector score | 0.9324 | 0.9500 | 0.776 |
| MLP APT | 0.9400 | 0.9571 | 0.812 |
| Set-Attention A | **0.9540** | **0.9730** | **0.873** |
| Set-Attention APT | 0.9534 | 0.9726 | 0.872 |

参考对话还报告了以下现象：

- anomaly-only 的 Set-Attention 与 APT 版本几乎相同，P/T evidence 未提供稳定增量；
- login failure 是主要困难类型，`mob1` / `mob2` 存在显著混淆；
- 跨节点相对比较优于逐节点独立评分，构成 H2 的前期动机；
- 窗口级标签效率很快饱和，但独立事件数可能远小于窗口数。

### 不能从该 Pilot 推出的结论

- 不能证明 event-level Standalone RCA 的性能；
- 不能证明 H1/H2/H3 已成立；
- 不能证明 propagation/temporal/structure 一定无效；
- 不能把窗口数当作独立故障事件数；
- 不能把 GAIA detector score 的高排名解释为跨数据集 RCA 泛化；
- 不能把这些数字写成当前 `rca-standalone` 分支结果。

### 已识别的 Pilot 局限

1. 样本单位是高度重叠窗口，存在 pseudo-replication 风险。
2. GAIA 注入节点、节点异常标签和根因标签语义耦合，可能形成任务捷径。
3. 旧 P/T evidence 依赖 Ada-MGAD 表征，不是独立 RCA 输入。
4. 当前 Ada-MGAD `DynamicGraphLearner` 在 batch 维度上汇总节点/日志表示并生成共享边权，不等同于逐故障事件结构。
5. Temporal evidence 的历史压缩方式可能不足以表达短故障的传播顺序。

## 3. Standalone RCA 实验状态

当前没有 `reproduced-current` 的 Standalone RCA 实验。

| 实验 ID | 内容 | 状态 | 结果 |
|---|---|---|---|
| P1-EVAL-TOY | AC@k / Avg@5 / MRR toy cases | planned | — |
| P1-GAIA-DIAG | GAIA event-level diagnostics | planned | — |
| P1-RE2OB-DIAG | RE2-OB 90-case diagnostics | planned | — |
| P1-B0 | Random baseline | planned | — |
| P1-B1 | Train-fold Root Frequency Prior | planned | — |
| P1-B2 | Metric Change Score | planned | — |

## 4. 新实验记录模板

复制以下模板追加，不覆盖旧记录：

```markdown
## YYYY-MM-DD — EXPERIMENT_ID

- Evidence level: reproduced-current | artifact-verified
- Objective:
- Hypothesis / gate:
- Git branch:
- Git commit:
- Working tree status:
- Dataset source/version/manifest:
- Included/excluded cases:
- RCACase schema version:
- Split manifest + seed:
- T_pre / T_post:
- Preprocessing fit scope:
- Candidate-set rule:
- Label Firewall test result:
- Command:
- Config path:
- Output artifact path:
- Per-case prediction path:
- Overall AC@1/3/5, Avg@5, MRR:
- Fault-type macro:
- Root-service macro:
- Runtime:
- Sanity checks:
- Result interpretation:
- Limitations / anomalies:
- Decision: continue | revise | no-go
```

## 5. 结果写作规则

- 报告均值前先说明统计单位是 case/event，不是窗口。
- 同时报告 overall、fault-type macro 和 root-service macro。
- 所有学习型统计量必须证明只在训练 fold 拟合。
- 保存 per-case ranking，不能只保存聚合指标。
- 负结果照常记录；不得因为结构模块无增益而只保留最优 seed。
- H1/H2/H3 的结论至少需要 GAIA 与 RE2-OB 的一致趋势或明确解释数据集差异。
- Oracle 与 Detected-trigger 结果分表，不能混用标题或 claim。

