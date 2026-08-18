# 实验日志

> 日志版本：`experiment_log_v1`  
> 最近更新：2026-08-19
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

当前没有 `reproduced-current` 的 Standalone RCA 新方法实验；P1 协议、数据工程
与 sanity baseline 结果单列如下。

| 实验 ID | 内容 | 状态 | 结果 |
|---|---|---|---|
| P1-EVAL-TOY | AC@k / Avg@5 / MRR toy cases | reproduced-current | 13/13 test suite passed；G4 completed |
| P1-GAIA-DIAG | GAIA event-level diagnostics | reproduced-current / preliminary | 16,200 candidate cases；G1 partial |
| P1-RE2OB-DIAG | RE2-OB 90-case diagnostics | reproduced-current / preliminary | 90/90 valid；G2 completed |
| P1-MANIFEST-SPLIT | 分离 manifest + context grouping | reproduced-current | GAIA 322 groups；33/33 tests；G6 partial |
| P1-TELEMETRY-DIAG | 全模态 coverage/missingness | reproduced-current | 33/33 tests；全量 349,120,558 行；G8 completed |
| P1-SPLIT-ASSIGN | 实际 5-fold assignment 与候选比较 | reproduced-current | GAIA grouped-stratified、RE2 singleton stratified；G6 completed |
| P1-GAIA-INCLUSION | 主 cohort 与 multi-root sensitivity | reproduced-current / artifact-verified | 13,470 main + 2,730 sensitivity；G1 completed |
| P1-B0 | Random baseline | reproduced-current / artifact-verified | GAIA/RE2 AC@1=0.1009/0.0556 |
| P1-B1 | Train-fold Root Frequency Prior | reproduced-current / artifact-verified | GAIA/RE2 AC@1=0.4807/0.1667；GAIA shortcut warning |
| P1-B2 | Metric Change Score | reproduced-current / artifact-verified | GAIA/RE2 AC@1=0.1414/0.8556；G3/G5/G7 completed |

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

## 7. 2026-08-18/19 — P1-TELEMETRY-DIAG

- Evidence level: reproduced-current
- Objective: 对 GAIA 与 RE2-OB 原始 metrics/logs/traces 做全量 coverage、timestamp、missingness 与 topology-source 诊断。
- Hypothesis / gate: P1 G8；为 `T_pre/T_post`、context-aware split 与 missing-data policy 提供前置证据。
- Git branch: rca-standalone
- Git commit: ab31282055b625838f41a1a3c91e51098175f254 + 当前未提交 P1 变更
- Working tree status: P1 代码、测试、文档和 artifacts 未提交。
- Dataset source/version/manifest: 用户提供的本地 GAIA MicroSS 与 RCAEval RE2-OB；release pin 尚未完成；manifest schema `p1_rca_manifest_v1`。
- Included/excluded cases: GAIA 16,200/952；RE2-OB 90/0。
- T_pre / T_post: 本实验诊断 ±30/60/300/600/1800 s；后续执行修订据此冻结 ±300 s 主窗口。
- Preprocessing fit scope: 无学习型拟合；10 万行分块流式统计。
- Candidate-set rule: GAIA 10 个 application service instances；RE2-OB 11 个应用服务。
- Label Firewall test result: scanner 消费 label-separated adapter 结构；预测可见 manifest 保持无 root/fault 字段。
- Commands:

```text
python scripts/diagnose_p1_telemetry.py --gaia-path /home/zhangll24/project_2/MultimodalAD/MSTGAD-GAIA/data/GAIA/MicroSS --re2ob-path /home/zhangll24/RCA_project/datasets/RCAEval/RE2-OB --output artifacts/p1/telemetry_diagnostics_smoke.json --chunk-rows 100000 --windows-seconds 30,60,300,600,1800 --max-gaia-files-per-modality 1 --max-re2ob-cases 1 --progress-every 1
python scripts/prepare_p1_manifests.py --gaia-path /home/zhangll24/project_2/MultimodalAD/MSTGAD-GAIA/data/GAIA/MicroSS --re2ob-path /home/zhangll24/RCA_project/datasets/RCAEval/RE2-OB --output-root artifacts/p1/manifests
python scripts/diagnose_p1_telemetry.py --gaia-path /home/zhangll24/project_2/MultimodalAD/MSTGAD-GAIA/data/GAIA/MicroSS --re2ob-path /home/zhangll24/RCA_project/datasets/RCAEval/RE2-OB --output artifacts/p1/telemetry_diagnostics.json --chunk-rows 100000 --windows-seconds 30,60,300,600,1800 --progress-every 100
```

- Output artifact path: `artifacts/p1/telemetry_diagnostics_smoke.json`、`artifacts/p1/telemetry_diagnostics.json`、`artifacts/p1/context_group_diagnostics.json`、`artifacts/p1/manifests/{gaia,re2ob}/`、[TELEMETRY_DIAGNOSTICS.md](TELEMETRY_DIAGNOSTICS.md)。
- Runtime: smoke 约 2.1 min；manifest regeneration 约 9 s；full scan 约 67 min（当前机器）。
- Sanity checks: 三条命令均 exit 0；六个数据集×模态组合未完成文件均为 0；全量 JSON SHA-256 `957ef5f94f50e096b53e7f0107920361cdb70ad5b0610a87640bbacbc327c6b6`。
- Result: 共读取 349,120,558 行；G8 completed。
- GAIA: 5,724 个候选 metric 文件、10/10 logs、10/10 traces；metric 网格推断缺口率 16.18%；±300 s 有 metric 活动 14,358/16,200。
- RE2-OB: 90/90 case 三模态覆盖 `t0`；metric value-cell missingness 0.6408%；±300 s 三模态平均 bin 覆盖均 ≥99.47%。
- Window/split interpretation: 注入区间与上下文并集在 ±300 s 形成 322 组、最大 471；±600 s 仅 23 组、最大 3,206，因此优先推进 ±300 s。
- Limitations / anomalies: RCAEval release 未 pin；未算完整 telemetry 内容哈希；动态图未物化；窗口、无观测 fallback 与实际 split assignment 尚未实现。
- Decision: continue

## 8. 2026-08-19 — P1-CONTEXT-GROUP

- Evidence level: reproduced-current
- Objective: 将 G8 的窗口覆盖证据转化为不会共享 telemetry 跨 split 的可执行分组。
- Hypothesis / gate: P1 G1/G6；比较候选窗口的分组可行性并冻结主上下文。
- Git branch/commit: rca-standalone；ab31282055b625838f41a1a3c91e51098175f254 + 当前未提交 P1 变更。
- Inputs: `artifacts/p1/manifests/gaia/{inputs,event_audit}.jsonl` 与 `artifacts/p1/telemetry_diagnostics.json`；不读取 labels 做分组。
- Interval semantics: 毫秒半开区间；GAIA 分组区间为“实际注入区间 ∪ [t0-300 s,t0+300 s)”。
- Commands:

```text
python scripts/diagnose_p1_context_groups.py --gaia-manifest-dir artifacts/p1/manifests/gaia --output artifacts/p1/context_group_diagnostics.json --windows-seconds 30,60,300,600,1800
python scripts/prepare_p1_manifests.py --gaia-path /home/zhangll24/project_2/MultimodalAD/MSTGAD-GAIA/data/GAIA/MicroSS --re2ob-path /home/zhangll24/RCA_project/datasets/RCAEval/RE2-OB --gaia-context-seconds 300 --output-root artifacts/p1/manifests
```

- Output: GAIA 322 groups；306 个非单例组覆盖 16,184 cases；最大组 471（2.91%）；direct overlap pairs 64,956。
- Alternatives: ±600 s 仅 23 组、最大组 3,206（19.79%）；±1800 s 仅 5 组、最大组 8,783。
- Manifest SHA-256: `69fe14c41e165a9a734649586fa7ab8df09685607ad51ef0cd5c4a99e77e1cbd`。
- Groups SHA-256: `adc68a6267bb3a75788b5944a26d656334e82c6d2e035e4e547e68355cacbdf9`。
- Context diagnostic SHA-256: `6dca047930d342bbc7902ca686dd55e8e1cb53cb43a889e9d8f67bb597efec12`。
- Sanity checks: 33/33 tests；compileall；bundle verification；半开边界、传递连通、非法区间测试。
- Decision: 冻结 `T_pre=T_post=300 s` 与 context-aware grouping；actual split assignment 仍 pending。

## 9. 2026-08-19 — P1-SPLIT-ASSIGN

- Evidence level: reproduced-current / artifact-verified
- Objective: 比较 GAIA grouped-stratified 与 context-purged temporal-block
  5-fold，并为 GAIA/RE2-OB 生成实际 OOF assignment。
- Hypothesis / gate: P1 G6；原子 group 不跨 fold，并在可行条件下保持每折
  root/fault 覆盖。
- Git branch/commit: `rca-standalone`；
  `ab31282055b625838f41a1a3c91e51098175f254` + 当前未提交 P1 变更。
- Dataset source/version/manifest: `p1_rca_manifest_v1`；GAIA manifest SHA-256
  `69fe14c41e165a9a734649586fa7ab8df09685607ad51ef0cd5c4a99e77e1cbd`；
  RE2-OB manifest SHA-256
  `a28dba97ab089ccfbfe963b0054a84d0c02fb1124e196070f2df4cda74327343`。
- Included/excluded cases: GAIA 16,200/952；RE2-OB 90/0。
- Split manifest + seed: `p1_split_manifest_v1`；5 folds；seed `20260819`；
  root/fault/joint weights 1/1/0.25。
- T_pre / T_post: GAIA ±300 s；半开区间；注入区间与 context union groups。
- Preprocessing fit scope: 本实验不拟合 telemetry preprocessing；后续每个
  baseline 必须只在 train folds 拟合。
- Label Firewall: labels 只由 splitter/diagnostics 读取，prediction input 与
  assignment 分离；split manifest 绑定 input/label/group hashes。
- Command:

```text
python scripts/prepare_p1_splits.py --manifest-root artifacts/p1/manifests --output-root artifacts/p1/splits --diagnostics-output artifacts/p1/split_diagnostics.json --folds 5 --seed 20260819
```

- Output: `artifacts/p1/split_diagnostics.json`、
  `artifacts/p1/splits/{gaia,re2ob}/`、[SPLIT_DIAGNOSTICS.md](SPLIT_DIAGNOSTICS.md)。
- Runtime: 首次质量门失败前约 4.1 s；修正后最终生成约 9.3 s。
- GAIA grouped result: fold sizes 3239/3239/3239/3241/3242；max root TV
  0.00426；max fault TV 0.00031；可行 root/fault 每折覆盖全部通过。
- GAIA temporal result: fold sizes 3247/3262/3263/3222/3206；max root/fault TV
  0.01024/0.01013，但 12 条 CPU fault 全部位于 fold-4，fold-0 至 fold-3
  缺失 CPU，fold-0 还缺失 access-permission fault，因此不 eligible。
- RE2-OB result: 每折 18 cases；每种 fault 每折 3 条；每个 root 每折 3–4 条；
  max fault/root TV 为 0/0.06667。
- Anomaly: 首版纯贪心 RE2 assignment 的 max root TV 为 0.12632，质量门 exit 1，
  未写成正式 assignment；加入确定性 move/pair-swap 局部改进和 90-case 网格
  测试后通过，未静默保留失败结果。
- Artifact SHA-256: diagnostics
  `bd174e785a6e44eb3b7f366defc232b63f35d01f82ac96b35b7f7f06646fceb7`；
  GAIA selected `996a03698f77fa8128c737ffc6547fe767e3f5d9d8fd7b430d86341c6dac7c90`；
  RE2 selected `c2205f39fa4742e7a22e474664cfccd70f3827e154534965760f0d16caf30c78`。
- Sanity checks: 37/37 tests；group integrity；source binding；逐文件 row count/SHA-256；
  fold coverage；确定性 tie-breaking。
- Decision: GAIA 主 split 冻结为 grouped-stratified 5-fold；时间块保留为
  temporal-shift sensitivity；RE2-OB 冻结为 singleton stratified 5-fold；
  G6 completed。

## 10. 2026-08-19 — P1-SANITY-BASELINES

- Evidence level: reproduced-current / artifact-verified
- Objective: 在冻结 split 上运行 B0 Random、B1 train-fold Root Frequency、
  B2 within-case Metric Change，并保存完整 rankings 与 macro metrics。
- Hypothesis / gate: P1 G3/G5/G7；baseline 必须输出完整候选排列，学习型统计
  只使用 train folds，prediction path 不读取 test root/fault。
- Git branch/commit: `rca-standalone`；
  `ab31282055b625838f41a1a3c91e51098175f254` + 当前未提交 P1 变更。
- Dataset/manifest: GAIA 16,200 cases；RE2-OB 90 cases；
  `p1_rca_manifest_v1` + `p1_split_manifest_v1`。
- Split + seed: GAIA grouped-stratified 5-fold；RE2 singleton stratified
  5-fold；seed `20260819`。
- T_pre/T_post: B2 为 ±300 s 半开 pre/post windows。
- Preprocessing fit scope: B0 none；B1 每个测试 fold 只从另外四折统计 roots；
  B2 none，每个 case 只使用自身 metrics。
- B2 config: duplicate timestamp finite mean；pooled-std standardized absolute
  mean shift；min 2 samples/side；cap 20；service Top-5 feature mean；缺失服务
  alphabetical-last fallback。
- Commands:

```text
python scripts/run_p1_sanity_baselines.py --manifest-root artifacts/p1/manifests --split-root artifacts/p1/splits --output-root artifacts/p1/baselines --summary-output artifacts/p1/baseline_summary.json --random-seed 20260819

python scripts/run_p1_metric_change.py --manifest-root artifacts/p1/manifests --split-root artifacts/p1/splits --output-root artifacts/p1/baselines --summary-output artifacts/p1/baseline_summary.json --window-seconds 300 --min-samples-per-side 2 --top-k-features 5 --score-cap 20 --progress-every 100
```

- Output: `artifacts/p1/baseline_summary.json`、
  `artifacts/p1/baselines/{gaia,re2ob}/{random,root_frequency,metric_change}/`、
  [BASELINE_RESULTS.md](BASELINE_RESULTS.md)。
- GAIA B0 overall AC@1/Avg@5/MRR: 0.1009/0.3015/0.2941；root-macro
  AC@1/Avg@5=0.0971/0.3138。
- GAIA B1 overall AC@1/Avg@5/MRR: 0.4807/0.8738/0.7292；root-macro
  AC@1/Avg@5=0.0997/0.2997，确认 class-frequency shortcut。
- GAIA B2 overall AC@1/Avg@5/MRR: 0.1414/0.3589/0.3410；fault-macro
  0.4413/0.5914/0.5751；root-macro 0.5285/0.7060/0.6620。
- RE2 B0 AC@1/Avg@5/MRR: 0.0556/0.2556/0.2547。
- RE2 B1 AC@1/Avg@5/MRR: 0.1667/0.5667/0.4241。
- RE2 B2 AC@1/Avg@5/MRR: 0.8556/0.9333/0.9046。
- Coverage: GAIA B2 1,895 all-service fallback、3,234 any-service fallback、
  14,305 metric-observed；RE2 90/90 无 fallback。coverage slices 已写入 metrics。
- Label Firewall: prediction files 敏感字段零命中；B1 每折 train/test overlap=0
  并保存 IDs digest；B0/B2 predictor 只接收 `RCACaseInput`；B2 不解析
  label-bearing RE2 directory names。
- Reproducibility: B0/B1 重跑后所有相关 outputs 与 B2 artifacts/summary 的 bytes
  均未变化；B2 重新全量读取 2,862 GAIA logical series/5,724 shards 与 90 RE2
  cases，summary、prediction、metrics、audit、run-manifest hashes 均一致。
- Runtime: B0/B1 约 25 s；B2 smoke 约 17 s；B2 每次正式全量约 3.5 min。
- Summary SHA-256:
  `42353b9b9460f8463fc1d14d8536f04da87d849b520051d3e07d3f1e8ca5a855`。
- Limitations: B2 是强数据 sanity baseline，不是结构/传播方法；GAIA overall
  被 login/mobile 主导，且 1,895 cases 依赖完全 fallback；RE2 高分不能单独
  证明新方法空间已经饱和。
- Decision: G3/G5/G7 completed；继续 G1 GAIA 最终纳入/污染规则与 P1 Gate audit。

## 11. 2026-08-19 — P1-GAIA-INCLUSION

- Evidence level: reproduced-current / artifact-verified
- Objective: 冻结 GAIA service-level single-root 主 cohort，并确保 cohort-specific
  Frequency 不读取被排除的 multi-root labels。
- Hypothesis / gate: P1 G1；每条 supported event 均映射为 inventory case，主表
  在锚点时必须只有一个 root service。
- Git branch/commit: `rca-standalone`；
  `ab31282055b625838f41a1a3c91e51098175f254` + 当前未提交 P1 变更。
- Inventory: 16,200 supported operational anomaly events；排除 952 条均有原因。
- Main rule: 若另一个 root service 的事件满足
  `other.start <= t0 < other.end`，目标 case 进入 multi-root sensitivity；同 root
  并发保留。
- Cohorts: main 13,470；multi-root sensitivity 2,730；anchor 无并发 13,077；
  same-root 并发 393；actual-interval isolated 12,163；完整 context isolated 16。
- Split: 继承 context-group-safe inventory assignment；main fold sizes
  2693/2684/2711/2705/2677，全部 5 fault / 10 root 每折覆盖。
- Command:

```text
python scripts/finalize_p1_gaia_inclusion.py --manifest-root artifacts/p1/manifests --split-root artifacts/p1/splits --baseline-root artifacts/p1/baselines --baseline-summary artifacts/p1/baseline_summary.json --inclusion-output-root artifacts/p1/inclusion/gaia --diagnostics-output artifacts/p1/gaia_inclusion_diagnostics.json --random-seed 20260819
```

- Output: `artifacts/p1/inclusion/gaia/`、
  `artifacts/p1/gaia_inclusion_diagnostics.json`、
  `artifacts/p1/baselines/gaia_main/`、
  [GAIA_INCLUSION_AUDIT.md](GAIA_INCLUSION_AUDIT.md)。
- Main B0 overall/root-macro AC@1: 0.1019/0.0993。
- Main B1 overall/root-macro AC@1: 0.4808/0.0994；Frequency shortcut 保留。
- Main B2 overall/fault-macro/root-macro AC@1: 0.1405/0.4592/0.5311。
- Main B2 coverage: 11,843 metric-observed；1,627 all-service fallback。
- Label Firewall: purity flags 与 cohort 位于 trusted analysis sidecar，不进入
  `inputs.jsonl`；main B1 重新仅从 main-cohort training folds 拟合。
- Inclusion manifest SHA-256:
  `88cef66b3cff0633b1cbd40c70b42fb004b1db0674364b77a59c618c353b6cca`。
- Diagnostics SHA-256:
  `9b9ddf8a63973eeb8c0cee4fe07bc18a2465ff353d2482ae18498792f07f13be`。
- Decision: G1 completed；任务表述为 designated operational event root，
  不宣称其是整个 ±300 s context 中唯一因果故障。进入 P1 reproducibility
  closeout。

## 12. 2026-08-19 — P1-REPRODUCIBILITY-CLOSEOUT

- Evidence level: reproduced-current / artifact-verified
- Objective: 固定 RCAEval 实际实验字节并交叉验证全部 P1 gate artifacts。
- Hypothesis / gate: 若 case/source/split/cohort/baseline bindings 完整且原始输入可按
  内容寻址，则 P1 可以关闭并解除 P2 阻塞。
- Git branch/commit: `rca-standalone`；
  `ab31282055b625838f41a1a3c91e51098175f254` + 当前未提交 P1 变更。
- Source scope: 原始 `RE2-OB.zip` 1,191,025,569 bytes；90 cases 实际引用的
  metrics/logs/traces/inject-time 共 360 files、8,441,465,341 bytes。
- Source identity: content identity
  `cec0030da8b499914af7979283eb1a2cd4f2cad2430742c70130829cd28133c9`；
  snapshot manifest
  `61f369b5e8b0cb3aaf540decbf585fb1643b7d4c153cb0dff4af1f445c07fab3`；
  raw archive
  `0605a36cdcad8a6ae0107f2357c9c91ecee2c4ab5d72579bffea0372d9747513`。
- Commands:

```text
python scripts/pin_p1_rcaeval_source.py --case-manifest artifacts/p1/manifests/re2ob --source-root /home/zhangll24/RCA_project/datasets/RCAEval --archive /home/zhangll24/RCA_project/datasets/RCAEval/RE2-OB.zip --output artifacts/p1/source_snapshots/re2ob

python scripts/pin_p1_rcaeval_source.py --case-manifest artifacts/p1/manifests/re2ob --source-root /home/zhangll24/RCA_project/datasets/RCAEval --archive /home/zhangll24/RCA_project/datasets/RCAEval/RE2-OB.zip --output artifacts/p1/source_snapshots/re2ob --verify-only

python scripts/audit_p1_gates.py --artifact-root artifacts/p1 --raw-source-root /home/zhangll24/RCA_project/datasets/RCAEval --output artifacts/p1/gate_audit.json

python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
git diff --check
```

- Gate audit: G1–G8 全部 pass；GAIA inventory/main/RE2 case counts 为
  16,200/13,470/90；9 组 baseline outputs 均通过 checksum、完整 ranking、
  split、指标重算、Label Firewall 与 run-summary binding。
- Raw verification: `raw_re2_source_bytes_verified=true`；同一 source snapshot
  另行 verify-only 重读后通过。
- Code checks: 51/51 tests pass；compileall 与 diff check exit 0。
- Gate audit SHA-256:
  `a02625a0a37136f0d763c8c7fa83d39b003c15eeac033310f6990ab55ce1bf45`。
- Limitation: content identity 能确认实验输入逐字节一致，但本地资产没有保留其
  上游 commit/tag/DOI/download provenance。
- Decision: P1 completed；冻结 case/window/split/candidate/metrics/Label Firewall，
  进入 P2 representation experiment design。

## 13. 2026-08-19 — P2-EXPERIMENT-PLAN-V0.1

- Evidence level: official-source-audited / plan-only
- Objective: 将 H1 event stage、H2 cross-node relative ranking、H3 structural
  interaction 拆成可独立否证的双数据集实验。
- Frozen P1 inputs: GAIA main 13,470、RE2-OB 90、±300 s、seed `20260819`
  five-fold、完整 service ranking 与 Label Firewall。
- Primary endpoint: root-service macro Avg@5；关键次端点为 root-macro AC@1。
- Model selection: outer-train 四折轮换 inner validation；不得读取 outer-test；
  `C ∈ {0.01,0.1,1,10}`，并列优先简单/强正则配置。
- Statistics: seed `20260819`，10,000 次 paired bootstrap；GAIA 以 context group、
  RE2 以 case 为重采样单位。
- Controlled ladder: single-modal independent → multimodal independent → event-stage
  → case-relative → conditional DeepSets → structure。
- External audit: RCAEval main 固定为
  `4695aa69f4f1f57b9094ca04ff235908b73a8e24`。官方 runner 使用派生
  `simple_metrics/logts/tracets` 与 indicator→service projection，故 external
  reference 与统一协议内 Track C 分表。
- Code-level eligibility: 固定实现中 mmBARO 使用 metric/log/trace series；
  mmCIRCA 使用 metric+log；mmRCD 的实际计算路径只使用 metric，不能作为三模态
  主对照。
- Output: [P2_EXPERIMENT_PLAN.md](P2_EXPERIMENT_PLAN.md) V0.1。
- Result status: plan-only；H1/H2/H3 均未验证。
- Decision: P2.0 completed；进入 P2-G1 feature schema + metric extractor smoke。

## 14. 2026-08-19 — P2-G1-METRIC-FEATURES

- Evidence level: reproduced-current / artifact-verified
- Objective: 建立 label-free 固定宽度 metric bundle，并完成 GAIA main 与 RE2-OB
  全量 Whole/Stage extraction。
- Dataset/cohort: GAIA main 13,470 cases/10 services；RE2-OB 90 cases/11 services。
- Window: ±300 s 半开；stage onset 30/60/120 s；每段至少 2 样本。
- Representation: 10 comparison blocks × 17 features = 170；float64 统计、float32
  values、bool mask。
- Commands:

```text
python scripts/extract_p2_metric_features_smoke.py
python scripts/extract_p2_metric_features.py
```

- Output: `artifacts/p2/features/`、`artifacts/p2/metric_feature_summary.json`、
  `artifacts/p2/metric_feature_audit.json`、[P2_METRIC_FEATURES.md](P2_METRIC_FEATURES.md)。
- Full counts: GAIA 134,700 rows，扫描 2,862 logical series/5,724 files；RE2
  990 rows，扫描 33,020 candidate metric columns。
- Label Firewall: index/manifest 敏感字段零命中；values finite；masked value=0；
  完整 case-service coverage；独立 verifier pass。
- Reproducibility: 双数据集 smoke 连续两次 8 个文件逐字节一致；feature-name
  cache 优化后 hashes 不变。
- Anomaly: GAIA 30 s pre→onset/onset→impact 仅 0.092%/0.091% service rows
  observed；60/120 s 约 83.7%，RE2 全部 100%。
- Decision: metric 子门 completed；metric 调参候选收缩为 60/120 s，30 s 只保留
  unsupported control；继续 C0-M 与 L0/T0。

## 15. 2026-08-19 — P2-G2-C0-M

- Evidence level: reproduced-current / artifact-verified
- Objective: 建立只使用 Whole metric representation 的 learned independent
  scorer，作为后续单模态/融合对照。
- Split/selection: frozen 5-fold outer OOF；每个 outer-train 四折轮换；
  `C={0.01,0.1,1,10}`；root-macro Avg@5 调参；seed `20260819`。
- Fit scope: StandardScaler 与 L2 logistic 只在当前 fit rows；每 case root/non-root
  权重各合计 0.5；test labels 不参与 fit/selection。
- Commands:

```text
python scripts/run_p2_c0_metric.py
python scripts/audit_p2_c0_metric.py
```

- GAIA root-macro AC@1/Avg@5: 0.4567/0.5820；overall 0.1633/0.3953。
- RE2 root-macro AC@1/Avg@5: 0.9111/0.9756。
- Delta vs P1 B2 root-macro AC@1/Avg@5: GAIA −0.0743/−0.1266；RE2
  +0.0556/+0.0422。
- Audit: 每数据集 80 inner fits + 5 outer fits；case/group overlap 全为 0；完整
  rankings；prediction labels 零命中；metrics 精确重算；selected C 可重建。
- Reproducibility: 第二次全量运行的 predictions/metrics/training-audit 六个核心
  文件 SHA-256 全部一致；runtime 单列 informational。
- Code checks: 当前全套 61/61 tests、compileall 与 `git diff --check` 通过。
- Output: `artifacts/p2/runs/c0_m/`、`artifacts/p2/c0_m_summary.json`、
  `artifacts/p2/c0_m_audit.json`、[P2_C0_M_RESULTS.md](P2_C0_M_RESULTS.md)。
- Decision: P2-G2 completed；保留 GAIA 主端点负差异，不宣称统一优于 B2；
  在 L0/T0 完成前不进入 C1-I 或 H1。

## 16. 2026-08-19 — P2-L0-T0-SCHEMA-AUDIT

- Evidence level: reproduced-current / raw-schema-audited
- Objective: 在编码 log/trace extractor 前冻结可比较 raw 字段、时间单位、status
  语义与 service alias。
- Evidence: P1 全量 telemetry diagnostics + 四类实际 CSV header/sample 只读检查。
- GAIA: logs 87,974,871 rows，真实时间来自 message prefix；traces 28,681,438
  rows，10 services 均有 parent/trace/span/service 字段。
- RE2: logs 15,053,223 rows；`cluster_id/log_template` 为预派生字段；traces
  34,461,235 rows，直接只覆盖 7 类 entities，`parentSpanID` 缺失 2,112,216 rows。
- Decision: L0 仅使用 raw 无词表 rate/severity/message-shape；template features
  进入 train-fold-only L1。T0 允许静态 `frontendservice→frontend`，其余未出现
  candidates 必须 values=0/mask=false；图边留到 H3。
- Synthetic tests: 半开边界、空活动/缺失实体、逐字段 mask、0/1/domain 校验与
  alias candidate boundary 全部通过。
- Real smoke: GAIA `dbservice1` L0/T0 扫描 6,018,741/1,437,954 raw rows，
  窗口 1,510 events/382 spans，50/50 与 80/80 observed；RE2 1 case L0/T0
  observed 468/550 与 560/880 cells，后者严格对应 7/11 trace entities。
- Reproducibility: `artifacts/p2/event_features_smoke/` 四个 bundles 共 16 文件
  连续运行两次 SHA-256 全部一致。
- Code checks: 当前全套 66/66 tests、compileall 与 `git diff --check` 通过。
- Output: [P2_MODALITY_SCHEMA_AUDIT.md](P2_MODALITY_SCHEMA_AUDIT.md)。
- Next: 将 readers 改成可恢复的全量流式实现 → full extraction/coverage audit。

## 17. 新实验记录模板

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

## 18. 结果写作规则

- 报告均值前先说明统计单位是 case/event，不是窗口。
- 同时报告 overall、fault-type macro 和 root-service macro。
- 所有学习型统计量必须证明只在训练 fold 拟合。
- 保存 per-case ranking，不能只保存聚合指标。
- 负结果照常记录；不得因为结构模块无增益而只保留最优 seed。
- H1/H2/H3 的结论至少需要 GAIA 与 RE2-OB 的一致趋势或明确解释数据集差异。
- Oracle 与 Detected-trigger 结果分表，不能混用标题或 claim。
