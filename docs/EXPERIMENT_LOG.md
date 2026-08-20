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

## 19. 2026-08-19 — P2-G1-L0-T0-FULL-EXTRACTION

> 记录性质：治理补账（backfill）。产物先于本记录写入，本轮未重跑 extraction；
> 下列数值全部从既有 artifact 直读。

- Evidence level: artifact-verified
- Objective: 把 L0 log / T0 trace readers 换成可恢复的全量流式实现，并完成 GAIA
  main 与 RE2-OB 双数据集全量 extraction 与 coverage 审计。
- Hypothesis / gate: P2-G1 事件模态子门（不检验 H1/H2/H3）。
- Git branch: `claudecode`
- Git commit: `64bb681328fa1793014615a20ba2bc1fdf33c3b7`
- Working tree status: 仅未跟踪新增文件，无已跟踪文件修改。
- Dataset source/version/manifest: `artifacts/p1/manifests/{gaia,re2ob}`，
  `p1_rca_manifest_v1`；raw telemetry 只经 manifest 内 URI 访问。
- Included/excluded cases: GAIA main cohort 13,470（`artifacts/p1/inclusion/gaia/
  main_cohort.jsonl`）；RE2-OB 全部 90。
- RCACase schema version: `p2_feature_bundle_v2`；extractors `p2_log_l0_v1`（50 列）
  与 `p2_trace_t0_v1`（80 列）。
- Split manifest + seed: 本阶段不使用 split；extraction 与 fold 无关，seed 不适用。
- T_pre / T_post: ±300 s 半开、毫秒。
- Preprocessing fit scope: 无任何拟合量；L0 只用 raw 无词表 rate/severity/
  message-shape，template 特征留给 train-fold-only L1。
- Candidate-set rule: 每 case 的候选 service 完全取自 manifest inputs，
  未出现的 candidate 一律 values=0 / mask=false。
- Label Firewall test result: `labels_read_by_extractor=false`、
  `masked_values_zero=true`、`all_values_finite=true`、`all_bundles_verified=true`。
- Command:

```text
python scripts/extract_p2_event_features_smoke.py
python scripts/extract_p2_event_features.py
```

- Config path: 脚本内默认绑定（无外部 config 文件）；raw 路径不作为 flag 暴露。
- Output artifact path:
  `artifacts/p2/event_features/{gaia_main,re2ob}/{p2_log_l0_v1,p2_trace_t0_v1}/`、
  `artifacts/p2/event_feature_summary.json`（SHA-256
  `a2ba98d4e52c566639701e1c9a16df99894c27cc4a509622c6a17b666aa2aa3d`）、
  `artifacts/p2/event_feature_audit.json`（SHA-256
  `27367fb5d14a46d74551a767507c119be7aeb11491bd2bc69de3aaf86e1d1ff3`）。
- Per-case prediction path: 不适用（本阶段无 ranking 输出）。
- Full counts: GAIA log 扫描 87,974,871 raw rows → 134,700 × 50（10 checkpoints）；
  GAIA trace 28,681,438 → 134,700 × 80（10 checkpoints）；RE2 log 15,053,223 →
  990 × 50（90 checkpoints）；RE2 trace 34,461,235 → 990 × 80（90 checkpoints）。
- Coverage slices（`entity_observed_row_count` / 总 service rows）：GAIA log
  134,670/134,700 = 0.999777；GAIA trace 134,670/134,700 = 0.999777；RE2 log
  902/990 = 0.911111；RE2 trace 630/990 = 0.636364（严格对应 11 candidates 中
  7 个直接可观测的 trace entities）。
- Onset support（`absolute_min_complete_ratio`，阈值 absolute ≥ 0.80 且
  relative-to-whole ≥ 0.90）：GAIA log 30/60/120 s = 0.964833 / 0.964706 /
  0.964677，GAIA trace = 0.927051 / 0.926962 / 0.926962，RE2 trace = 0.995238 /
  0.988889 / 0.988889，均 supported；**RE2 log 三档同为 0.794900，低于 0.80 绝对
  阈值，全部 unsupported**（relative_to_whole 为 1.0，故判定只由绝对阈值触发）。
- Overall AC@1/3/5, Avg@5, MRR: 不适用（无 ranking）。
- Fault-type macro: 不适用。
- Root-service macro: 不适用。
- Runtime: 未记录（`event_feature_summary.json` 无 runtime 字段）。
- Sanity checks: coverage audit 由 `extract_p2_event_features.py` 在同一次运行内
  经 `verify_feature_bundle` 复验后写出，**不是**独立第二进程产物，引用时不得与
  `run_*`/`audit_*` 成对的独立审计混称。
- Result interpretation: 事件模态可用于 C0-L / C0-T / C1-I 的 `whole.*` 通道；
  RE2 log 因 unsupported 不得进入任何 staged 通道。
- Limitations / anomalies: RE2 trace 仅 0.6364 entity-observed，其融合贡献部分
  依赖 mask 通道；RE2 log 0.794900 的缺口来自 cohort 本身而非窗口截断。
- Decision: continue — P2-G1 事件模态子门 completed；M1-S 的 staged 通道排除 log，
  该排除写入 run manifest 的 `log_stage_excluded`。

## 20. 2026-08-19 — P2-G3-LINEAR-ABLATIONS（C0-L / C0-T / C1-I）

> 记录性质：治理补账（backfill）。产物先于本记录写入，本轮未重跑三个 run；
> 数值取自 `linear_ablation_summary.json` 与独立 `linear_ablation_audit.json`。

- Evidence level: artifact-verified
- Objective: 在与 C0-M 完全同一装置下建立两个事件单模态 scorer（C0-L / C0-T）与
  naive early-fusion 对照（C1-I），为 M1-S 提供合法单因素基线。
- Hypothesis / gate: P2-G3；本阶段不检验 H1（H1 需 M1-S vs C1-I）。
- Git branch: `claudecode`
- Git commit: `64bb681328fa1793014615a20ba2bc1fdf33c3b7`
- Working tree status: 仅未跟踪新增文件，无已跟踪文件修改。
- Dataset source/version/manifest: 同记录 19。
- Included/excluded cases: GAIA main 13,470；RE2-OB 90。
- RCACase schema version: 输入 `p2_feature_bundle_v2`；run schema
  `p2_nested_oof_v1`。
- Split manifest + seed: `artifacts/p1/splits/{gaia,re2ob}`，
  `p1_split_manifest_v1`，5-fold outer OOF，seed `20260819`。
- T_pre / T_post: ±300 s 半开、毫秒。
- Preprocessing fit scope: StandardScaler 与 L2 logistic 只在当前 fit rows 拟合；
  每 case root 权重 0.5、全部 non-root 合计 0.5；outer-test labels 不参与
  fit 或 selection。
- Candidate-set rule: 输出每 case 的完整候选排列，分数并列按 service name 升序。
- Design: 只用 `whole.*` 值列及等宽 masks —— C0-L 5/10，C0-T 8/16，
  C1-I 30/60（metric 17 + log 5 + trace 8 拼接）；不含 stage、service identity、
  fault type、root frequency。
- Label Firewall test result: predictions 归一化后 root/fault/ground-truth token
  零命中；`label_free_predictions=true`（三方法、两数据集）。
- Command:

```text
python scripts/run_p2_linear_ablations.py
python scripts/audit_p2_linear_ablations.py
```

- Config path: 脚本内默认绑定；`C ∈ {0.01,0.1,1,10}`；inner 目标唯一为
  root-service macro Avg@5；并列优先较小 C。
- Output artifact path: `artifacts/p2/runs/{c0_l,c0_t,c1_i}/{gaia_main,re2ob}/`、
  `artifacts/p2/linear_ablation_summary.json`（SHA-256
  `ef1d3aa46f961dbf77776882d3c2234e26f5e474090e047bee2711654b62a718`）、
  `artifacts/p2/linear_ablation_audit.json`（SHA-256
  `003655fc3207d7831f92d2d4d1631ab7bdeefbde17275acd5d67a717cdba75e3`）。
- Per-case prediction path:
  `artifacts/p2/runs/<method>/<dataset>/predictions.jsonl`。
- Overall AC@1/Avg@5: GAIA C0-L 0.0856/0.4141，C0-T 0.3566/0.5043，
  C1-I 0.3376/0.6691；RE2 C0-L 0.2889/0.4667，C0-T 0.6222/0.8156，
  C1-I 0.9778/0.9956（RE2 单例分层，三层同值）。
- Fault-type macro AC@1/Avg@5: GAIA C0-L 0.4180/0.5731，C0-T 0.4971/0.6096，
  C1-I 0.6020/0.7541。
- Root-service macro AC@1/Avg@5（主端点层）: GAIA C0-L 0.1622/0.3567，
  C0-T 0.1722/0.3549，C1-I 0.3656/0.6072；RE2 C0-L 0.2889/0.4667，
  C0-T 0.6222/0.8156，C1-I 0.9778/0.9956。
- Delta vs best single modality（两数据集均为 C0-M）root-macro AC@1/Avg@5:
  GAIA −0.0911/**+0.0253**；RE2 +0.0667/**+0.0200**。
  `c1_i_primary_improved=true`（两侧），
  `c1_i_exploratory_signal_both_datasets=true`。
- Delta vs P1 B2 metric-change root-macro AC@1/Avg@5: GAIA −0.1655/−0.1013；
  RE2 +0.1222/+0.0622。（B2 基数直读
  `artifacts/p1/baselines/{gaia_main,re2ob}/metric_change/metrics.json`：
  GAIA root-macro 0.531066/0.708573，RE2 0.855556/0.933333。）
- Runtime: C0-L GAIA 114.41 s / RE2 0.71 s；C0-T 131.06 s / 0.78 s；
  C1-I 848.98 s / 8.58 s（均 `informational_only`）。
- Sanity checks: 每方法每数据集 5 outer fits + 80 inner fits；outer train/test 的
  case overlap 与 group overlap 全为 0；inner fit/validation overlap 全为 0；
  rankings 经 evaluator 独立重算且与记录逐字相等；selected C 可由保存的 inner
  objective 重建；run manifest 内每个核心文件 SHA-256 校验通过。
- Per-fold（root-macro Avg@5）: GAIA C1-I 0.6276/0.6038/0.5986/0.5879/0.6135，
  五折无反转；GAIA 各 outer fold 选中 C 均为 10；RE2 C1-I 为
  1.0000/0.9767/1.0000/1.0000/1.0000。逐 fold 切片由 predictions + 冻结
  assignments 重算补齐，**未写回 summary artifact**。
- Code checks（补账时点）: 全套 106/106 tests、`compileall`、`git diff --check`
  通过。
- Result interpretation: naive 拼接在两数据集主端点均优于任一单模态，构成
  exploratory 级融合信号；但 GAIA 次端点 AC@1 明显退化，且 GAIA 上所有 learned
  方法主端点仍低于未学习的 P1 B2。
- Limitations / anomalies: (1) GAIA 两个事件单模态在主端点接近低信息水平
  （0.3567/0.3549），其 overall 层优于 metric-only 反映类别频率相关信号而非 root
  区分度；(2) RE2-OB C1-I 已达 0.9956，天花板效应使该数据集对后续方法区分度极低；
  (3) RE2-OB 单例分层使三层指标同值。
- Decision: continue — P2-G3 completed；不得宣称学习式融合优于 metric-change
  基线；GAIA AC@1 退化仅作为诊断结果，不作为修改 inner selection objective 的
  依据（用户已确认：改动会破坏 H1 的单因素可解释性）。

## 21. 2026-08-19 — P2-G3-C1-I-PAIRED-BOOTSTRAP

> 记录性质：治理补账（backfill）。数值直读 `artifacts/p2/c1_i_bootstrap.json`。

- Evidence level: artifact-verified
- Objective: 用冻结的配对 bootstrap 判定 C1-I 相对 best single modality 的
  exploratory / claim-ready 门禁状态。
- Hypothesis / gate: P2-G3 统计收口；不检验 H1。
- Git branch: `claudecode`
- Git commit: `64bb681328fa1793014615a20ba2bc1fdf33c3b7`
- Working tree status: 仅未跟踪新增文件，无已跟踪文件修改。
- Dataset source/version/manifest: 同记录 19；比较对象由
  `linear_ablation_audit.json` → `best_single_comparison` 决定，两数据集均为
  `c0_m`。
- Split manifest + seed: 同记录 20；bootstrap seed `20260819`，
  `numpy.default_rng/PCG64`，10,000 次。
- Resampling unit: GAIA 按 322 context group；RE2-OB 按 90 case。
- Preprocessing fit scope: 不适用（只对已保存 rankings 重采样，不重新拟合模型）。
- Label Firewall test result: 只读 predictions 与 labels 用于评估；无模型拟合，
  故无泄漏面。
- Command:

```text
python scripts/bootstrap_p2_c1_i.py
```

- Output artifact path: `artifacts/p2/c1_i_bootstrap.json`（schema
  `p2_c1_i_paired_bootstrap_v1`，SHA-256
  `f7399673340d7af540af2336e997ef5fffb4ca7495247a72e36597a95ca77212`）。
- Root-service macro Avg@5（主端点，C1-I 减 C0-M）: GAIA point +0.025261，
  95% CI [−0.003557, +0.056340]，P(Δ ≤ 0) = 0.0445；RE2 point +0.020000，
  95% CI [+0.002514, +0.040000]，P(Δ ≤ 0) = 0.0118。
- Root-service macro AC@1（关键次端点）: GAIA point −0.091132，
  95% CI [−0.132206, −0.049165]，P(Δ ≤ 0) = 1.0000；RE2 point +0.066667，
  95% CI [+0.002050, +0.133333]，P(Δ ≤ 0) = 0.0223。
- Gate 判定: `exploratory_signal = true`（两数据集主端点点估计均为正）；
  `claim_ready = false`，两项检查均不通过 ——
  `both_primary_ci_lower_bounds_positive = false`（GAIA CI 下界 −0.003557 ≤ 0），
  `both_secondary_point_deltas_at_least_minus_0_01 = false`（GAIA AC@1
  −0.091132 < −0.01）。
- Runtime: 未单列记录。
- Sanity checks: point estimate 与 OOF 报表逐位一致；重采样单位数与
  `resampling_unit_count`（GAIA 322、RE2 90）一致；root service 数 GAIA 10 / RE2 5。
- Result interpretation: C1-I 只达到 exploratory 级信号，不可用于任何论文级
  claim；GAIA 主端点 CI 跨 0 且次端点显著退化。
- Limitations / anomalies: GAIA CI 下界仅 −0.003557，接近但未越过 0，说明当前
  样本与重采样单位下融合增益不稳定；RE2 的正结果受天花板效应影响，不能单独支撑
  结论。
- Decision: continue — 以 `exploratory_signal=true / claim_ready=false` 收口
  P2-G3，进入 P2-G4（M1-S run → 独立 audit → M1-S vs C1-I 配对 bootstrap）；
  不因本结果改动冻结协议、不缩减 C/onset 网格、暂不引入 RE2-TT。

## 22. 2026-08-20 — P2-G4-M1-S（run + independent audit + paired bootstrap）

> 记录性质：实时记录。数值直读 `artifacts/p2/m1_s_summary.json`、
> `artifacts/p2/m1_s_audit.json`、`artifacts/p2/m1_s_bootstrap.json`。

- Evidence level: artifact-verified
- Objective: 在冻结的 nested OOF 装置内，以 event-stage 通道为**唯一**变化因素，
  检验 H1（stage-aware 表征是否优于 naive early fusion C1-I）。
- Hypothesis / gate: H1；P2-G4 go/no-go。
- Git branch: `claudecode`
- Git commit: `64bb681328fa1793014615a20ba2bc1fdf33c3b7`
- Working tree status: `scripts/run_p2_m1_stage.py` 与 `src/` 为该提交的未修改
  版本；工作树另有治理文档修改（`docs/{README,RESEARCH_STATUS,EXPERIMENT_LOG}.md`）
  与未跟踪新增文件（`scripts/{audit_p2_m1_stage,bootstrap_p2_m1_s}.py`、
  `tests/test_m1_stage_{audit,bootstrap}.py`、`docs/P2_G{3,4}_*.md`、`CLAUDE.md`）。
- Dataset source/version/manifest: GAIA main 13,470 cases
  (`dataset_manifest_sha256` `69fe14c41e165a9a734649586fa7ab8df09685607ad51ef0cd5c4a99e77e1cbd`)
  与 RE2-OB 90 cases；特征绑定为
  `artifacts/p2/features/<ds>/p2_metric_summary_v1` +
  `artifacts/p2/event_features/<ds>/{p2_log_l0_v1,p2_trace_t0_v1}`，
  manifest SHA-256 逐一记录在 run manifest 的 `source.features`。
- Split manifest + seed: `artifacts/p1/splits/{gaia,re2ob}`，
  `p1_split_manifest_v1`，5-fold OOF，seed `20260819`；GAIA
  `split_assignment_sha256` `996a03698f77fa8128c737ffc6547fe767e3f5d9d8fd7b430d86341c6dac7c90`。
- Design: C1-I 的 `whole.` metric+log+trace（30 值）加 selected-onset 的
  metric+trace `stage{60|120}.`（51 + 24 = 75 值），共 105 值 + 105 masks =
  **210 列**（audit `design_column_count` 两数据集均为 210）。log 的 staged 通道
  按已冻结决策排除，理由记于 run manifest `log_stage_excluded`。
- Grid: `C ∈ {0.01,0.1,1,10}` × `onset ∈ {60,120}`，未缩减；每 outer fold 32
  inner fits ⇒ 每数据集 **160 inner + 5 outer fits**（audit `inner_fit_count` = 160）。
- Preprocessing fit scope: StandardScaler 仅在当前 fit rows 上拟合；inner
  validation 唯一目标为 inner root-service macro Avg@5，tie-break 为较小 C →
  较短 onset（**未修改**）。
- Label Firewall test result: `label_free_predictions = true`、
  `label_firewall_flags_all_false = true`（两数据集）。
- Command:

```text
nohup setsid python -u scripts/run_p2_m1_stage.py > logs/m1_s_run.log 2>&1 < /dev/null &
python scripts/audit_p2_m1_stage.py
python scripts/bootstrap_p2_m1_s.py
```

- Output artifact path: `artifacts/p2/runs/m1_s/{gaia_main,re2ob}/`；
  `artifacts/p2/m1_s_summary.json`（`p2_nested_oof_v1`，SHA-256
  `d77b31436e1eb0c0ec6dac61ad917d31418d82988e1b819b61c84be17a5dc495`）；
  `artifacts/p2/m1_s_audit.json`（`p2_m1_stage_audit_v1`，SHA-256
  `930c3ae14269a3dc7ecb64187f18f2297efd37605bf6a0fd2d0b7fe247f82910`）；
  `artifacts/p2/m1_s_bootstrap.json`（`p2_m1_s_paired_bootstrap_v1`，SHA-256
  `65788c2dd7f8c79d82c4494255cf2b2295e64ab8f983f9ea48b33ca84f0b323a`）；
  run manifests `b774070c77b4fa4ae9e01bcc055111992fc52208aba1ca859bffa0de41667687`
  (GAIA) / `e96efb25ea73845ab01c368ca51ea692a141be1948d1eae1f33fddc3b785f0bb` (RE2)。
- Root-service macro（主层，OOF）: GAIA AC@1 0.405516 / Avg@5 0.631871；
  RE2-OB AC@1 0.966667 / Avg@5 0.993333。
- Overall / fault-type macro: GAIA overall 0.518486 / 0.808552，fault-macro
  0.646138 / 0.793613；RE2-OB 三层同值（单例分层）。
- 相对 C1-I 的主层差值（单因素对照）: GAIA AC@1 +0.039918 / Avg@5 +0.024646；
  RE2-OB AC@1 −0.011111 / Avg@5 −0.002222。
- 相对其他参照的主端点差值: vs C0-M GAIA +0.049907 / RE2 +0.017778；
  vs P1 B2 GAIA **−0.076702** / RE2 +0.060000。
- Selected 超参: GAIA 五折均为 C=10、onset=60 s；RE2-OB 为 C 1/1/10/0.1/0.1、
  onset 120/60/120/120/120 s。
- Paired bootstrap（10,000 次，seed `20260819`，GAIA 按 322 context group、
  RE2 按 90 case）: GAIA 主端点 point +0.024646，95% CI
  [+0.005371, +0.043954]，P(Δ ≤ 0) = 0.0070；GAIA 次端点 point +0.039918，
  95% CI [+0.008128, +0.070546]，P(Δ ≤ 0) = 0.0066；RE2 主端点 point −0.002222,
  95% CI [−0.007500, 0.000000]，P(Δ ≤ 0) = 1.0000；RE2 次端点 point −0.011111,
  95% CI [−0.037500, 0.000000]，P(Δ ≤ 0) = 1.0000。
- Gate 判定: `exploratory_signal = false`（RE2 主端点点估计为负）；
  `claim_ready = false`，两项检查均不通过 ——
  `both_primary_ci_lower_bounds_positive = false`（RE2 CI 下界 −0.007500 ≤ 0），
  `both_secondary_point_deltas_at_least_minus_0_01 = false`（RE2 AC@1
  −0.011111 < −0.01）。⇒ `p2_g4_decision = "no-go"`。
- Runtime: GAIA 12,326.37 s（≈3 h 25 min）、RE2-OB 24.01 s，均标记
  `informational_only=true` 且排除在核心 checksum 之外。
- Sanity checks: audit 独立重算 metrics 与记录逐位相等；
  `outer_fold_count`=5、`inner_fit_count`=160、`outer_fold_ids_match_frozen_split`
  与 `prediction_folds_match_frozen_split` 均 true；outer train/test case 与 group
  overlap、inner fit/validation case 与 group overlap 全部为 0；selected C 与
  onset 均可由保存的 inner objective 重建；run manifest 内每个核心文件 SHA-256
  校验通过；对照方 C1-I 亦经
  `comparator_metrics_exactly_recomputed=true` / `comparator_core_files_verified=true`
  复验；bootstrap 的 point delta 与 audit 的 `root_service_macro` delta 交叉一致
  （容差 1e-12）；第二次运行 audit 与 bootstrap（输出至 scratch 路径，未覆盖既有
  产物）得到逐字节相同 SHA-256。
- Result interpretation: **H1 在冻结门禁下未获支持。** GAIA 上 stage 通道同时改善
  主端点与次端点（两端点 CI 下界均 > 0），但 RE2-OB 上主端点符号为负且次端点越过
  −0.01 guardrail，因此不满足"两数据集同向"的 exploratory 条件。M1-S 与 C1-I 的
  门禁失败原因互补（C1-I 败于 GAIA 次端点，M1-S 败于 RE2 主端点与次端点），两个
  gate 均未出现两数据集同向证据。GAIA 上 M1-S 主端点仍低于未学习的 P1 B2
  （−0.0767，C1-I 时为 −0.1013），"learned 方法在 GAIA 主端点不优于 B2"的结论
  在本 gate 依然成立。
- Limitations / anomalies: RE2-OB 的全部退化溯源为**单个 case**
  `re2ob-c9c8f348d3f5974b`（fold_1，root `recommendationservice`）的 root 由
  rank 1 落到 rank 2 —— 逐 case 比对显示 89/90 个 case 仅发生尾部重排，只有
  1/90 个 case 的 root 排名位置变化；5 root × 18 cases 下该 case 使宏平均 AC@1
  降 (1/18)/5 = 0.011111、Avg@5 降 0.002222，与记录值逐位一致，且 AC@3/AC@5
  差值恰为 0.0。这是 P2-G3 已记录的 RE2-OB 天花板效应（C1-I 已达 0.9956）在门禁上
  的直接后果；该溯源**只作诊断，不改变判定**，也不构成放宽阈值的依据。本次
  guardrail 差值 −0.011111 与 −0.01 的距离远大于浮点误差，无边界歧义。另：
  两数据集 inner selection 落在不同 onset（GAIA 全 60 s、RE2 多为 120 s）；M1-S
  同时引入 stage 切分与 75 个新值列，无法分离"切分本身"与"维度增加"的贡献；逐 fold
  切片与单 case 溯源由 predictions 重算，未写回 summary artifact。
- Decision: stop — 按既定停止点（`M1-S run → independent audit → paired bootstrap
  → P2-G4 go/no-go`）在此收口。不因本结果改动任何冻结协议、不缩减 C/onset 网格、
  不修改 inner selection objective、不引入 RE2-TT、不实现 M2-R/M2-D/M3-G。
  后续路线（维持 H1 未获支持的叙述 / 重新讨论 RE2-TT / 对 selection objective 做
  独立 protocol version bump + sensitivity experiment）三者互斥，须由用户明确决定
  后另开 gate。H2 / H3 仍未检验。

## 23. 2026-08-20 — EXT-RE2TT-E1..E5（第三 benchmark 的 audit-first 验收）

- Evidence level: artifact-verified（E5 独立复算，不 import E1/E3/E4 driver 的计算 helper）
- Objective: 用户选择路线②——保留 P2-G4 的 `no-go` 与 H1「冻结门禁下未获支持」不变，
  新建独立 protocol extension，引入 RCAEval RE2-TT 作为第三 benchmark，**先**验证其在
  统一 Track C 下的数据质量、候选空间、三模态 coverage 与性能 headroom，审计全部通过后
  才在完全冻结的 C1-I → M1-S 单因素协议下做 H1 replication。
- Hypothesis / gate: E-G1…E-G5（协议见 `docs/RE2TT_EXTENSION_PROTOCOL.md`
  `re2tt_extension_protocol_v0.2` §6，判定规则在看到任何 E4 数值之前冻结）。
  核心为 E-G4 headroom 门禁：H-1 `B2 metric_change` root-macro Avg@5 ≤ 0.90（门禁）、
  H-2 同 AC@1 ≤ 0.90（门禁）、H-3 B2 在主/次端点均严格优于 B1（门禁）、
  H-4 B1 AC@5（仅报告）。
- Git branch: `claudecode`
- Git commit: `c2af48f2be4066de7363d7e5f0871052e8564301`
- Working tree status: **不干净** —— 扩展 driver 与新测试在运行时为 untracked/modified。
  为保证可核对，逐文件 SHA-256 记录于 `docs/RE2TT_EXTENSION_PROTOCOL.md` §9.0
  （`prepare_ext_re2tt_manifests.py` `5bee68ec…`、`diagnose_ext_re2tt_telemetry.py`
  `8b7b1a2d…`、`prepare_ext_re2tt_splits.py` `47c56980…`、
  `run_ext_re2tt_baselines.py` `eb00fed7…`、`audit_ext_re2tt_gates.py` `698bacd8…`，
  另有 `src/data/{rcaeval,telemetry_diagnostics,__init__}.py` 与
  `scripts/run_p1_metric_change.py` 的 artifact-neutral 改动）。
- Dataset source/version/manifest: `RCAEval-RE2-TT`，源根
  `/home/zhangll24/RCA_project/datasets/RCAEval/RE2`（`RE2-TT/` 与 `RE2-TT.zip` 同根）；
  `content_identity_sha256 = ad1396d0713fe21343a9db9233a60e51aa043026f8f39cd30fa1ba0fd5f15fc0`
  （`p1_source_snapshot_v1`，1 归档 2,801,345,134 B + 360 消费文件 22,752,639,186 B）；
  manifest `artifacts/ext/re2tt/manifests/manifest.json`
  `e7c0bb4c8fd3866f7af11db3c08078133a3f8557517ff8886bdd62b7743a7aa2`。
- Included/excluded cases: 90 / 0。候选服务 68 个，候选元组变体 1 个（全 case 同一候选
  集）；90 个 singleton group（`largest_group = 1`）；root `ts-auth/order/route/train/
  travel-service` 各 18；fault `cpu/delay/disk/loss/mem/socket` 各 15；
  root ∉ 候选集 = 0。
- RCACase schema version: `p1_rca_manifest_v1`（复用，未新造）
- Split manifest + seed: `artifacts/ext/re2tt/splits/split_manifest.json`
  （`p1_split_manifest_v1`，`grouped_stratified_5fold`），seed `20260819`，5 折；
  `assignments.jsonl` `6a1f165dbdbb3b651f7def513ea3447873da7d5276f685bb1aec8dc8a40f1662`、
  `split_manifest.json` `7e590bb408e163ab0b8474fba26162abbdd6d05fdee861b2adcad04e27aa684c`。
- T_pre / T_post: 300 s / 300 s，半开毫秒窗（未改动）
- Preprocessing fit scope: B0/B1 仅在训练折拟合（root 频率先验）；B2 无拟合，
  每个分数只用本 case 窗内数据（`fit_scope = "none; every score uses only its own
  case window"`）。E1–E5 不产出任何进入模型的特征。
- Candidate-set rule: 官方部署清单全集，未按 root 标签剔除任何实体（剔除需用标签，
  属泄漏）；因此 68 候选中包含大量 `ts-*-mongo`/`ts-*-mysql` 类永不为 root 的实体。
- Label Firewall test result: 90/90 input 通过 `assert_label_free`；
  `inputs.jsonl` 原始路径泄漏扫描 `raw_home_paths` / `condition_directory_names` /
  `release_directory` 全为 `false`；`verify_manifest_bundle` 通过；与冻结 RE2-OB 的
  case_id 冲突 0。E5 是本扩展中唯一被允许在扫描原始遥测时读标签的阶段（审计用途，
  不产出特征），该豁免已写入协议 §6.1。
- Command:
  ```bash
  python scripts/prepare_ext_re2tt_manifests.py    # E1
  python scripts/diagnose_ext_re2tt_telemetry.py   # E2
  python scripts/prepare_ext_re2tt_splits.py       # E3  (--folds 5 --seed 20260819)
  python scripts/run_ext_re2tt_baselines.py        # E4
  python scripts/audit_ext_re2tt_gates.py          # E5
  ```
- Config path: B2 配置与冻结 P1 逐键相同（300 s 半开窗、每侧 ≥ 2 样本、top-5 特征、
  cap 20.0），记录于 `artifacts/ext/re2tt/baseline_summary.json`
  → `metric_change_config`，等值性由
  `tests/test_ext_re2tt_extension.py::FrozenReuseTest` 以 golden literal 钉住。
- Output artifact path: `artifacts/ext/re2tt/`（gitignored；**未写入
  `artifacts/p1/**` 或 `artifacts/p2/**`**，由 `_assert_isolated` 在每个 driver 入口强制）
  —— `manifests/`、`source_snapshot/`、`telemetry_diagnostics.json`
  （`eb4cba13ca8b8d6c8f93143d8dd78ae739adc7ddf8ce08c0a5e369b9b2a19522`）、
  `splits/`、`split_diagnostics.json`、`baselines/{random,root_frequency,metric_change}/`、
  `baseline_summary.json`（`8488604fac2b02d30f78bd15d7ec27287d08b7989438603807be5c62066ca1bc`）、
  `headroom_gate.json`（`8f015bf2df84ee599f0d25ccd12d3147252410d6462b8caa236085b3ab881e18`）、
  `gate_audit.json`（`8895138ab617d49f8928072c571bf0bdb5a8cfadb0600ac29a6c528fda197cf4`）。
- Per-case prediction path: `artifacts/ext/re2tt/baselines/<baseline>/`
  （run manifest digest：B0 `f5bb15b2…e92c0c`、B1 `2ba40b4f…4aa32330`、
  B2 `7294f3c8…f0ba0a8b`）
- Overall AC@1/3/5, Avg@5, MRR: B0 0.000000 / 0.033333 / 0.077778 / 0.037778 /
  0.063523；B1 0.166667 / 0.555556 / 1.000000 / 0.566667 / 0.424074；
  B2 0.822222 / 0.922222 / 0.944444 / **0.904444** / 0.878432。
- Fault-type macro: 与 overall **逐位相同**。
- Root-service macro: 与 overall **逐位相同**。RE2-TT 的 root 5×18 与 fault 6×15 都是
  完全均衡设计，故三个报告层恒等；这一点本身是局限（见下），不是巧合。
  参照（不合并统计）：RE2-OB root-macro B0 0.255556 / B1 0.566667 / B2 0.933333；
  GAIA main root-macro B0 0.313242 / B1 0.296640 / B2 0.708573（候选数 68 / 11 / 10）。
- Runtime: E2 全扫（metrics 129,690 行 + logs 21,291,651 行 + traces 67,345,051 行）
  与 E4/E5 均为分钟级，无需 nohup 长作业；未纳入任何 checksum。
- Sanity checks: E5 独立重算——`manifests/` 5 个文件 digest 全部一致、
  `verify_manifest_bundle` 通过；source snapshot 的 `archives.jsonl` /
  `consumed_files.jsonl` digest 一致；重建 `SplitAssignment`/`CaseGroup` 后
  `validate_split_integrity` 通过且 `fold_label_consistent = true`、折间 case 与
  group 重叠均为 0、fold 大小 18/18/18/18/18、`max_case_count_relative_deviation
  = 0.0`、`max_fault_type_total_variation = 0.0`、
  `max_root_service_total_variation = 0.06666666666666665`（阈值 0.10）、
  每折含全部 5 root 与 6 fault；每条预测均为 68 服务的无重复全排列，
  三个报告层的每个指标与记录的 `metrics.json` 差 ≤ `1e-12`，OOF 覆盖完整；
  `integrity_gates_passed = true`；headroom 独立复判
  `matches_recorded_gate = true`。E3 重跑逐字节一致。
  **RE2-OB 逐字节回归 passed**（`inputs/labels/sources/groups.jsonl` 四文件，
  raw root `…/RCAEval/RE2-OB`，90 case）—— adapter 的 profile 化未扰动任何已冻结产物。
  三道闸门：`python -m unittest discover -s tests` 131 tests OK、
  `python -m compileall -q src scripts tests` OK、`git diff --check` OK。
- Result interpretation: **E-G4 的预登记 headroom 门禁 H-1 不通过**：
  B2 `metric_change` root-macro Avg@5 = **0.9044444444444444 > 0.90**，超出 0.004444
  （恰为 2 个 Avg@5 case 量子，1 量子 = 0.002222）。`decision = "fail"`、
  `failed_gates = ["H-1"]`。H-2 通过（0.822222）、H-3 通过（Avg@5 +0.337778、
  AC@1 +0.655556）、H-4 报告值 B1 AC@5 = 1.000000（与 RE2-OB 同值）。
  三项实质读数：(a) B0 从 0.255556 降到 0.037778，68 候选的干扰项扩张是真实且巨大的；
  (b) B1 与 RE2-OB **逐位同值**，已排除接线错误（run manifest 绑定
  `artifacts/ext/re2tt/{manifests,splits}` 且 digest 相符，预测排序全部 68 个 RE2-TT
  服务），同值是"5 root × 18 case 均衡设计 + B1 只用 root 频率先验（与候选总数无关）"
  的结构性后果 —— **RE2-TT 没有修复 5-root 先验退化**；(c) B2 只从 0.933333 降到
  0.904444（−0.028889），即干扰项从 6 个变 63 个后，一个无训练的 within-case metric
  shift 基线仍占据 0.904 的主端点。因此 **RE2-TT 的天花板不是候选空间造成的**，而是
  "注入式单点故障 + 完整 metric 覆盖"这一实验设计的共性；换 RCAEval 的另一个 release
  无法解决。按协议 §5 与 §6.3，`gate_audit.json` 记录
  `blocked_stages = ["E6","E7"]`，**E6/E7 未执行**。route ② 以"第三个 benchmark 也
  饱和"收口：P2-G4 的 `no-go` 与 H1「冻结门禁下未获支持」不变，且多出一条更强证据 ——
  RE2-OB 的天花板不是它自己的偶然缺陷。
- Limitations / anomalies: (1) **log 掩码与 root 类别完全混杂** —— E5 的
  root-conditioned coverage（±300 s 双侧可见）为 metrics 90/90、traces 90/90、
  **logs 72/90**，未覆盖的 18 个恰好且仅是 `ts-train-service` 的全部 18 个 case
  （11 个窗内完全无 root 日志、6 个仅 post 侧、1 个仅 pre 侧），其余 4 个 root 的 72
  个 case 全部双侧可见。故"root 的 log 通道被掩码"在 RE2-TT 上是"root **不是**
  `ts-train-service`"的近确定性指示器，任何用到 log observed mask 的模型都可能靠此
  捷径抬高 root-macro，且该捷径不迁移。(2) 三个报告层同值使 RE2-TT 无法暴露"总体好但
  某分组塌陷"的失效模式 —— 恰是 GAIA 上最有信息量的那种。(3) 90-case 粒度未改善
  （单 case 量子 AC@1 0.011111 / Avg@5 0.002222），与 RE2-OB 相同。(4) 候选空间跨数据
  集不可比（68 / 11 / 10），报告时必须并列候选数；68 候选含大量数据库实体，模型可学到
  "数据库实体不是 root"的静态先验。(5) trace 掩码同样携带静态实体类型信息（68 候选中
  仅 ~27 个有 span）。(6) 1 个 case（`ts-train-service_socket/1`）的 log 起点为
  +123.289 s，t0 之前无日志；按冻结的事件提取规则会被整段掩码而非剔除，无需新规则。
  (7) 单一应用、每场景 3 replicate、无级联故障，与 RE2-OB 同类局限。
  (8) 运行时工作树不干净（driver 未提交），已用逐文件 SHA-256 补偿，但严格意义上
  本记录的可复现性依赖那些 digest 而非单一 commit。
- Decision: no-go（对 E6/E7）—— 按预登记规则停在 E5。**不放宽 H-1 阈值**（0.904444 与
  0.90 只差 2 个量子，但门禁在看到数值之前已冻结，事后调整会使整条 route ② 失去证据
  价值）；不修改 inner selection objective；不删除或弱化 RE2-OB；不改写 P2-G4 的
  `no-go` 与 H1 结论；不为 H1 寻找第四个数据集；不实现 M2-R/M2-D/M3-G。
  Event-stage 的去留与是否进入 H2 仍待用户决定。H2 / H3 仍未检验。
