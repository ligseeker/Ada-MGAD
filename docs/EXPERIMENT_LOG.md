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

当前没有 `reproduced-current` 的 Standalone RCA 性能实验；P1 协议与数据工程结果单列如下。

| 实验 ID | 内容 | 状态 | 结果 |
|---|---|---|---|
| P1-EVAL-TOY | AC@k / Avg@5 / MRR toy cases | reproduced-current | 13/13 test suite passed；G4 completed |
| P1-GAIA-DIAG | GAIA event-level diagnostics | reproduced-current / preliminary | 16,200 candidate cases；G1 partial |
| P1-RE2OB-DIAG | RE2-OB 90-case diagnostics | reproduced-current / preliminary | 90/90 valid；G2 completed |
| P1-MANIFEST-SPLIT | 分离 manifest + overlap grouping | reproduced-current | GAIA 13,077 groups；23/23 tests；G6 partial |
| P1-B0 | Random baseline | planned | — |
| P1-B1 | Train-fold Root Frequency Prior | planned | — |
| P1-B2 | Metric Change Score | planned | — |

## 4. 2026-08-18 — P1-EVAL-TOY

- Evidence level: reproduced-current
- Objective: 验证统一 schema、完整 ranking invariants、AC@1/3/5、Avg@5、MRR 与最小 Label Firewall。
- Hypothesis / gate: P1 G4；G7 的最小接口部分。
- Git branch: rca-standalone
- Git commit: ab31282055b625838f41a1a3c91e51098175f254 + 当前未提交 P1 变更
- Dataset source/version/manifest: 手工 toy cases，无外部数据。
- Split manifest + seed: 不适用。
- Label Firewall test result: prediction boundary 只接收 RCACaseInput；尝试读取 root_service 抛出 AttributeError；嵌套敏感 metadata 被拒绝。
- Command: python -m unittest discover -s tests -v
- Output artifact path: tests/test_schema.py、tests/test_metrics.py、tests/test_adapters.py
- Result: 13/13 tests passed；其中 G4 指标与非法 ranking cases 全部通过。
- Sanity checks: 正确首位、Top-k 命中、缺失根因、重复/不完整/额外候选、case id 不匹配。
- Limitations / anomalies: 尚无 preprocessing 与 baseline prediction pipeline，故 G7 只记 partial。
- Decision: continue

## 5. 2026-08-18 — P1-DATA-ADAPTER-AUDIT

- Evidence level: reproduced-current / preliminary
- Objective: 在真实本地数据上验证 GAIA 与 RE2-OB adapter 覆盖和 schema invariants。
- Hypothesis / gate: P1 G1、G2、G8 前置诊断。
- Git branch: rca-standalone
- Git commit: ab31282055b625838f41a1a3c91e51098175f254 + 当前未提交 P1 变更
- Dataset source/version/manifest: 用户提供的本地 GAIA 与 RCAEval 目录；发布版本与 checksum 尚未固定。
- Included/excluded cases: RE2-OB 90/90、排除 0；GAIA 16,200 候选注入、排除 952 条非 case 记录。
- RCACase schema version: 首版代码 schema；序列化 manifest version 尚未冻结。
- Split manifest + seed: 尚未生成。
- T_pre / T_post: 尚未冻结。
- Preprocessing fit scope: 本轮只读取事件表、文件存在性和 CSV header，无学习型拟合。
- Candidate-set rule: RE2-OB 为 CPU/Memory 应用服务实体并排除辅助容器，共 11；GAIA 为 10 个 application service instances。
- Label Firewall test result: input/label/source 分离；RE2 原始标签目录名不进入预测可见 URI。
- Command: python scripts/audit_p1_datasets.py --gaia-path /home/zhangll24/project_2/MultimodalAD/MSTGAD-GAIA/data/GAIA/MicroSS --re2ob-path /home/zhangll24/RCA_project/datasets/RCAEval/RE2-OB --output artifacts/p1/dataset_audit.json
- Output artifact path: artifacts/p1/dataset_audit.json、docs/DATASET_AUDIT.md
- Overall AC@1/3/5, Avg@5, MRR: 不适用，尚未运行 baseline。
- Sanity checks: RE2 root_service 全在 11 个候选中；GAIA 10 个候选固定；旧 CPU 浮点时长漏标已在 RCA adapter 修复。
- Result interpretation: G2 completed；G1 partial；G8 pending。
- Limitations / anomalies: GAIA 有 4,037 个 case 参与 3,429 对区间重叠；RCAEval 版本未固定；缺少完整 missingness、相对时间覆盖与 topology 诊断。
- Decision: continue

## 6. 2026-08-18 — P1-MANIFEST-SPLIT

- Evidence level: reproduced-current
- Objective: 生成确定性、物理标签分离的 case bundles，并建立相关 case 不跨 split 的可执行约束。
- Hypothesis / gate: P1 G1/G6/G7 的 manifest 与防泄漏部分。
- Git branch: rca-standalone
- Git commit: ab31282055b625838f41a1a3c91e51098175f254 + 当前未提交 P1 变更
- Working tree status: 本轮代码、测试、文档和 artifacts 未提交。
- Dataset source/version/manifest: 用户提供的本地 GAIA 与 RE2-OB；bundle schema `p1_rca_manifest_v1`；raw release/checksum 尚未固定。
- Included/excluded cases: GAIA 16,200/952；RE2-OB 90/0。
- Split manifest + seed: 已生成原子 group manifest；实际 partition/fold assignment 与 seed 尚未冻结。
- Preprocessing fit scope: 无学习型拟合。
- Label Firewall test result: input/label/source 物理拆分；input manifest 标签字段、原始路径、relative directory、source index 零命中；每个文件有 SHA-256。
- Command: `python scripts/prepare_p1_manifests.py --gaia-path /home/zhangll24/project_2/MultimodalAD/MSTGAD-GAIA/data/GAIA/MicroSS --re2ob-path /home/zhangll24/RCA_project/datasets/RCAEval/RE2-OB --output-root artifacts/p1/manifests`
- Output artifact path: `artifacts/p1/manifests/gaia/`、`artifacts/p1/manifests/re2ob/`
- Runtime: 9.49 s（manifest generation；当前机器）
- Sanity checks: 23/23 tests；compileall；bundle checksum verification；diff check；无 patch 残留。
- Result: GAIA 13,077 groups，914 个非单例 group 覆盖 4,037 cases，最大组 39；RE2-OB 90 singleton groups。
- Result interpretation: grouping primitive 可用于后续 grouped/time-aware split；尚无实际 assignment，G6 保持 partial。
- Limitations / anomalies: RCAEval release 未固定；T_pre/T_post、GAIA split 类型和完整 telemetry diagnostics 未冻结。
- Decision: continue

## 7. 新实验记录模板

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

## 8. 结果写作规则

- 报告均值前先说明统计单位是 case/event，不是窗口。
- 同时报告 overall、fault-type macro 和 root-service macro。
- 所有学习型统计量必须证明只在训练 fold 拟合。
- 保存 per-case ranking，不能只保存聚合指标。
- 负结果照常记录；不得因为结构模块无增益而只保留最优 seed。
- H1/H2/H3 的结论至少需要 GAIA 与 RE2-OB 的一致趋势或明确解释数据集差异。
- Oracle 与 Detected-trigger 结果分表，不能混用标题或 claim。

