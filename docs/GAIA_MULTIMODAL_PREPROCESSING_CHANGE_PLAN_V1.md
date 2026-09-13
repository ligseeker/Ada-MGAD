# GAIA 多模态 Preprocessing 修改方案 V1

## Material Passport

- Origin Skill: `academic-research-suite/experiment-agent` + `codebase-design`
- Origin Mode: implementation planning
- Origin Date: `2026-09-14`
- Worktree: `/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2`
- Plan base HEAD: `0706957161f416de54cbf26940934336511013f4`
- Phase-1 implementation commit: `6bbf874f311030b7d6fbf7298f5dfd04159beec6`
- Parent method snapshot: `f405fce1107a9ed0ed3ced83de53c953f7bd6ff3`
- Parent audit: `docs/GAIA_MULTIMODAL_PREPROCESSING_AUDIT_V1.md`
- Parent audit SHA-256: `32b15d4f50188bc6ae80936a18ab850361a030d1f0aaa7f2e18ac9539bf74048`
- Current V3 config SHA-256: `121bdaddc7d8d70087a5ecd41ebc2070a3534d52ec13b76f9945dc916df90449`
- Verification Status: `IMPLEMENTATION IN PROGRESS; SYNTHETIC TESTS ONLY`
- Execution Status: `NO FORMAL PREPROCESSING OR TRAINING`
- Version Label: `gaia_multimodal_preprocessing_change_plan_v1`

## 1. 决策摘要

建议把当前“运行时一边选择 schema、一边 transform”的实现改成两个严格分离的阶段：

```text
Train-only audit / schema fit
        ↓
人工确认并冻结 FrozenPreprocessingSchema
        ↓
Train/Test deterministic transform
        ↓
split-local arrays + manifest validation
```

核心修改为：

| 模态 | 新表示 | 维数约束 | 核心模型修改 |
|---|---|---:|---|
| Metric | `global container representatives + host-of-node representatives + 3 observability slots` | 目标 48–64；exact slots 待 P1 冻结 | 不修改 |
| Log | `stable templates + 5 RARE + 5 UNK + 5 level counts` | `15 + K`，总维数 32–64 | 不修改 |
| Trace | 每个 status 的 `count + mean_latency` | 固定 `4×2=8` | 不修改，仅更新 `raw_edge` |

当前 `feature_node=16`、`feature_log=8`、`feature_edge=4` 保持不变。`raw_node/log_len/raw_edge` 继续从 preprocessing manifest 动态传给 `MyModel`。本轮不修改 reconstruction loss；先通过语义去重、维数预算和尺度统一控制隐式模态权重。

## 2. 必须先修正的实现结构

### 2.1 建立一个深 preprocessing module

当前 `src/e2e/ad_preprocess.py` 同时承担 registry、Train fit、transform、缓存、并行、manifest 和 label materialization，而且存在被后部定义覆盖的旧 `build_log_arrays` 与 `build_ad_data`。继续向该文件叠加规则会使调用者和测试必须理解过多内部状态。

建议新增：

```text
src/e2e/gaia_preprocessing/
  __init__.py
  schema.py       # policy/frozen schema 的读取、验证和 hash binding
  metric.py       # Metric registry、语义变换和 transform
  logs.py         # Drain3 fit、stable/RARE/UNK routing 和 transform
  traces.py       # split-local parent resolution、graph 和 8D transform
  materialize.py  # 唯一 orchestration implementation
```

外部 interface 只保留两个入口：

```python
freeze_preprocessing_schema(
    *, policy_path, audit_root, raw_root, config_path, output_path
) -> FrozenPreprocessingSchema

materialize_ad_inputs(
    *, schema_path, config_path, raw_root, data_root, artifact_root, runtime
) -> Mapping[str, object]
```

接口不允许传入临时 feature list、Test-derived threshold 或运行时 fallback。所有选择必须已经存在于 frozen schema。`src/e2e/ad_preprocess.py` 最终仅作为兼容 adapter 转发到新 module；旧的 shadowed implementations 在 characterization tests 建立后删除，而不是再保留第三套路径。

### 2.2 分离 policy、frozen schema 与 transform

建议定义三个不同概念：

1. `PreprocessingPolicy`：研究者批准的 label-free 规则、候选阈值和维数上限。
2. `FrozenPreprocessingSchema`：用 Train 数据在 policy 下得到的 exact slots、templates、scalers、graph 和 source hashes。
3. `TransformResult`：只使用 frozen schema 生成的 Train/Test arrays 与 manifest。

正式 materialization 中不得再次执行 feature selection、Drain cluster 创建、相关阈值选择或 graph 增边。

### 2.3 保留 V3，新增 preprocessing revision

不要原地改写 `configs/e2e/gaia_p5_v3.json` 或现有 `data/p5/v3`、`artifacts/p5/v3`。新增完整配置：

```text
configs/e2e/gaia_p5_v3_preprocessing_v2.json
```

它保持 GT、split、window、模型和训练参数不变，只新增/修改 preprocessing revision，并记录：

```text
parent_config_path
parent_config_sha256
preprocessing_policy_path
frozen_preprocessing_schema_path
preprocessing_schema_sha256
data_root = data/p5/v3_preprocessing_v2/ad
artifact_root = artifacts/p5/v3_preprocessing_v2/ad
checkpoint_root = data/p5/v3_preprocessing_v2/checkpoint
```

在 `scripts/p5/run_i1_pipeline.py` 尚未去除硬编码 V3 输出路径前，不允许用它执行新 revision 的正式流程；先通过 `run_i1_ad.py` 的显式 `--data-root/--artifact-root/--checkpoint-dir` 做 smoke。

## 3. Frozen schema 的最小契约

建议 schema 至少包含：

```json
{
  "schema_version": "gaia_ad_preprocessing_v2",
  "status": "FROZEN",
  "fit_split": "train",
  "decision_inputs": ["train"],
  "gt_labels_used": false,
  "test_used_for_selection": false,
  "source_binding": {},
  "policy_sha256": "...",
  "metric": {
    "ordered_slots": [],
    "source_entities": {},
    "semantic_transforms": {},
    "fill_parameters": {},
    "imputation_parameters": {},
    "scalers": {}
  },
  "logs": {
    "drain_config_sha256": "...",
    "drain_state_sha256": "...",
    "stable_cluster_ids": [],
    "ordered_slots": [],
    "scalers": {},
    "zero_support_fallbacks": {}
  },
  "traces": {
    "status_order": ["200", "300", "400", "500"],
    "statistic_order": ["count", "mean_latency"],
    "ordered_slots": [],
    "directed_edges": [],
    "symmetric_adjacency_sha256": "...",
    "scalers": {}
  }
}
```

校验器必须拒绝以下情况：

- `status != FROZEN`；
- source/config/policy/schema hash 不匹配；
- `decision_inputs` 包含 Test；
- Metric/Log slot 重名或顺序漂移；
- Trace slot 不严格等于 8 个固定槽；
- graph 或 scaler 在 Test transform 后发生变化；
- exact dimension 超出冻结预算；
- schema 中含 NaN/Inf 或未解析 fallback。

## 4. Metric 修改方案

### 4.1 Registry：从 strict intersection 改为 scope-aware unified schema

修改 `_logical_metric_schema()` 的职责：它只生成 registry，不再直接返回十节点交集。

每条 registry row 保存：

```text
raw source namespace
physical IP/host
target node(s)
logical feature
scope = global_container | host_of_node | quarantined
kind = gauge | counter | direct_rate
core aggregation
duplicate reduction
applicability
```

当前证据下：

- 90 个 candidate-owned Docker logical features 进入 `global_container` candidate pool；
- 384 个 system logical features 进入 `host_of_node` candidate pool；
- `redis@0.0.0.3` 与 `zookeeper@0.0.0.1` 保持 `quarantined`，没有部署拓扑证据时不得映射到 candidate nodes；
- `.3` 上的 `logservice1/webservice2` 对 host slots 为 not-applicable，不是低质量数据。

最终 tensor 仍为所有节点相同的 ordered slots。host slot 在有 telemetry 的节点写入 host 值；无对应 host telemetry 的节点写入 Train-neutral 值，同时 `host_applicable=0`。

### 4.2 先做语义变换，再做质量和冗余审计

处理顺序固定为：

```text
raw shard merge
→ exact duplicate reduction
→ 30s alignment
→ gauge/counter/direct-rate transform
→ multi-core aggregation
→ Train-only quality
→ Train-only redundancy
→ time-aware fill
→ Train median imputation
→ q01/q99 clipping + min-max
→ append observability slots
```

不能在 cumulative counter 上直接计算最终相关性和 min-max。Counter 的规则：

```text
rate_t = (value_t - value_prev) / elapsed_seconds
negative delta -> reset -> missing
elapsed > max_gap_for_rate -> missing
```

如果同一语义同时有 direct rate 和可差分 counter，代表列优先级为：

```text
direct gauge/rate
> reset-aware derived rate
> cumulative counter
```

累计 counter 原值默认不进入最终 schema。

### 4.3 Multi-core

- `core utilization pct/norm_pct`：保留 `mean` 和 `max` 两个候选；max 用于单核热点。
- `core ticks`：先聚合/差分后参与与 total CPU rate 的冗余审计；本轮抽样显示其 mean/max 近乎完全相关，不同时保留二者。
- 不默认增加 `std`。
- core 数量和缺失 core 数写入 registry/manifest，不能因不同实例 core 数不同而改变 slot 顺序。

### 4.4 Quality 与 representative selection

P1 前需要批准 exact policy；建议的跨实体规则候选为：

```text
global feature: 在至少 8/10 nodes 通过质量规则
host feature:   在至少 2/3 physical hosts 通过质量规则
future type feature: 在该 type 的 2/2 instances 通过质量规则
```

这三个比例是设计候选，不是正式预注册结果。单实体 failure 不再删除其他实体有信息的统一 slot。

冗余只在相同 semantic family 内判断，并使用 pairwise-complete Train bins；禁止因偶然跨族相关而删除物理意义不同的信号。P0 同时报告 Pearson/Spearman 在 `0.90/0.95/0.98/0.995` 的结果，P1 人工批准 exact representative list。Production transform 只读取该 list，不在运行时重新聚类或 top-K。

### 4.5 Missing、not-applicable 与 scaling

每个 source series 从 Train 估计 median positive sampling interval：

```text
max_fill_age = clip(2 × median_interval, 60s, 120s)
```

fill 必须 split-local。超过 max age 的位置保留 missing，之后用对应 Train median 填充。最终只增加三个 aggregate slots：

```text
global_observed_fraction
host_applicable
host_observed_fraction
```

不为每个 feature 增加 mask，避免维数翻倍。

缩放规则：

- container：按 `service×logical_feature` 使用 Train q01/q99 clipping 后映射到 `[0,1]`；
- host：按 `physical_host×logical_feature` 拟合，再复制到共宿 candidate nodes；
- Test 始终 clip 到冻结范围，不更新 quantile；
- `q99<=q01` 的 slot 在 P1 freeze 前失败，不在 transform 时静默修复。

### 4.6 Metric 输出

```text
[global_container_representatives]
[host_of_node_representatives]
[global_observed_fraction]
[host_applicable]
[host_observed_fraction]
```

目标预算：global 30–40、host 15–21、observability 3，总计 48–64。该区间是 hard budget；exact feature names 和 `raw_node` 只能由 P1 frozen schema 给出。

## 5. Log 修改方案

### 5.1 Vocabulary fit 与 stable selection

继续使用一个全节点共享的 Train-only Drain3。固定：Drain config hash、service/file/row 顺序、Train 半开区间和 miner state。

对每个 Train cluster 统计：

```text
records
active_node_bins
active_time_bins
active_days
active_services
level distribution
```

Stable eligibility threshold 和 `K_max` 在 P1 根据 Train 分布批准。若 eligible clusters 超过预算，使用以下 label-free 确定性顺序：

```text
active_node_bins desc
→ active_days desc
→ active_services desc
→ records desc
→ cluster_id asc
```

最终 schema 保存 exact stable cluster IDs；Test 不能触发替换或增加。

### 5.2 RARE、UNK、level 与 slot order

固定顺序：

```text
stable_template_<cluster_id> × K
RARE_INFO, RARE_WARNING, RARE_ERROR, RARE_DEBUG, RARE_UNKNOWN
UNK_INFO,  UNK_WARNING,  UNK_ERROR,  UNK_DEBUG,  UNK_UNKNOWN
level_INFO, level_WARNING, level_ERROR, level_DEBUG, level_UNKNOWN
```

路由规则：

- Train 见过且获得独立 slot：stable template slot；
- Train 见过但未获独立 slot：按消息 level 进入 RARE；
- frozen matcher 无匹配：按消息 level 进入 UNK；
- 每条消息同时进入一个 level count；
- `log_total` 只写入 audit stats，不进入模型。

Rare ERROR 不删除，只是不再让每个稀有文本占一维。Single-service template 允许保留；service identity 已由 node 轴表达，不创建 service-specific vocabulary。

### 5.3 Scaling 与零支持 UNK

所有 counts 先做 `log1p`，再用 Train positive node-bin q99 缩放并 clip 到 `[0,1]`。

UNK 在 Train 中按定义通常没有正样本，不能用 Test 估计 scale。必须冻结 fallback chain：

```text
UNK_<level>
→ RARE_<level> Train scale
→ level_<level> Train scale
→ pooled Train message-count scale
→ fixed scale 1.0
```

每个 fallback 的实际选择写入 frozen schema。Test UNK count、level 和比例仅进入 transform diagnostics，绝不回写 stable/K/scaler。

### 5.4 Log 输出

```text
log_len = K + 15
32 <= log_len <= 64
17 <= K <= 49
```

若 Train 审计无法在该预算下形成稳定 schema，P1 必须 NO-GO；不能因 Test 表现扩大 K。

## 6. Trace 修改方案

### 6.1 Split-local、trace-local parent resolution

把当前 full-month `span_id` lookup 改为每 split 独立的 combined key：

```text
key = stable_hash(trace_id + NUL + span_id)
```

建议沿用当前双 64-bit hash/memmap 的 bounded-memory思路，但索引必须包含 `trace_id` 和 split。解析 child 时：

1. 只查询与 child 相同 split 的 parent index；
2. 统计 0 match、1 match、>1 match；
3. 只有唯一 match 才形成 source service；
4. 跨 split、ambiguous、self-loop、unknown service 分别计数，不做 first-match fallback。

Train/Test 都可以被 transform，但只有 Train 成功解析的 cross-service edge 可以参与 graph freeze。

### 6.2 8D raw edge features

每个 30s bin、每条 directed edge、每个 status 维护：

```text
count
duration_sum
```

随后计算：

```text
mean_latency = duration_sum / count, if count > 0 else 0
```

最终固定顺序：

```text
200_count_log, 200_mean_latency_log,
300_count_log, 300_mean_latency_log,
400_count_log, 400_mean_latency_log,
500_count_log, 500_mean_latency_log
```

`duration_sum` 只作为中间聚合量和 audit 字段，不进入模型；第一版不加入 p95/max。

### 6.3 Scaling

- count：`log1p(count)` 后用 Train positive-bin q99；
- mean latency：`log1p(mean_latency_seconds)` 后用 Train positive-bin q99；
- scaler 首选 `directed edge×status×statistic`；
- positive bins 少于 P1 冻结阈值时，回退到 `status×statistic` pooled Train scale；
- 再无支持则回退到 `statistic` pooled Train scale；
- Test 不建立或更新任何 scaler。

### 6.4 Graph

P0 先输出每条 Train directed edge 的 calls、active bins 和 active days。P1 冻结 exact directed edge list；production 不在 transform 时应用新的频率阈值。

模型 adjacency 为兼容当前 GAT 继续对称化，但同时保存 directed registry：

```text
directed_train_edges.json
graph.npy                  # symmetric model adjacency
```

Test 新出现的 edge 不得修改 graph；应记录为 `test_unseen_directed_edge`。若该数量不可忽略，它是静态图方案的 limitation，不能用 Test 反向扩图。

## 7. Config 修改

新的 `ad_preprocessing` 至少包含：

```json
{
  "schema_version": "gaia_ad_preprocessing_v2",
  "fit_split": "train",
  "policy_path": "configs/e2e/gaia_p5_v3_preprocessing_v2.policy.json",
  "frozen_schema_path": "artifacts/p5/v3_preprocessing_v2/schema/frozen_preprocessing_schema.json",
  "metric": {
    "schema": "scope_aware_uniform_slots",
    "fill": "train_interval_max_age",
    "imputation": "train_median_plus_scope_observability",
    "scaling": "train_q01_q99_clip_minmax",
    "dimension_min": 48,
    "dimension_max": 64
  },
  "logs": {
    "schema": "stable_plus_level_rare_unk",
    "scaling": "log1p_train_positive_q99",
    "dimension_min": 32,
    "dimension_max": 64
  },
  "traces": {
    "parent_key": ["trace_id", "span_id"],
    "parent_scope": "split_local",
    "status_order": ["200", "300", "400", "500"],
    "statistics": ["count", "mean_latency"],
    "raw_edge": 8,
    "scaling": "log1p_train_positive_q99"
  }
}
```

Exact feature list、stable cluster IDs、graph edges 和 scaler 不直接散落在 config；它们属于 hash-bound frozen schema。

## 8. 代码文件修改清单

| 文件 | 修改 |
|---|---|
| `src/e2e/gaia_preprocessing/*` | 新增深 module，实现 schema freeze 与三模态 transform |
| `src/e2e/ad_preprocess.py` | 变为兼容 adapter；删除重复/被覆盖实现 |
| `src/e2e/log_templates.py` | 保留 Drain 封装，增加 cluster statistics 和 frozen routing；不负责选择 policy |
| `src/e2e/protocol.py` | 校验 preprocessing revision、frozen schema hash、独立输出根 |
| `scripts/p5/audit_gaia_multimodal_preprocessing.py` | P0 补齐 semantic counter transform 后的 quality/correlation/scaler candidates |
| `scripts/p5/freeze_gaia_preprocessing_schema.py` | 新增；只消费 approved policy + Train audit，原子写 frozen schema |
| `scripts/p5/run_i1_ad.py` | 接收/验证 schema path；移除 smoke 中硬编码的旧 config hash |
| `scripts/p5/run_i1_pipeline.py` | 后续改为 config-derived output roots 和 revision-specific lock；P3 前不用于新 revision |
| `configs/e2e/gaia_p5_v3_preprocessing_v2*.json` | 新增 policy 与完整 revision config；不改 V3 原文件 |
| `tests/test_gaia_preprocessing_v2.py` | 新的 interface-level contract tests |
| `scripts/p5/run_v3_preprocessing_smoke.py` | 复制/升级为 revision-specific smoke，不覆盖历史 V3 smoke artifact |

## 9. 测试方案

### 9.1 Schema firewall

- schema fit 只能接受 Train interval；传入 Test decision source 必须失败；
- frozen schema 在 Test transform 前后 hash 完全相同；
- source/config/policy/hash 任一漂移必须失败；
- ordered slots 在 serial/parallel、不同 chunk size 下完全相同。

### 9.2 Metric

- 统一 slot 长度相同，但 host N/A 与 missing、真实最小值三者可区分；
- `.3` 节点 `host_applicable=0`，其他节点为 1；
- 共宿节点使用相同 host 值与同一 physical-host scaler；
- counter reset 不产生负异常尖峰，长 gap 不跨越求 rate；
- fill 不跨 split、不超过 frozen max age；
- 单核热点能改变 core max 而不必改变 mean；
- Test extreme value 只被 frozen q01/q99 clip，不更新 scaler。

### 9.3 Log

- Train stable、Train rare、Test unseen 分别进入 stable/RARE/UNK；
- rare ERROR 进入 `RARE_ERROR`，不被删除；
- Test match 不创建 cluster，Drain state hash 不变；
- zero-support UNK 严格走冻结 fallback chain；
- `log_total` 不在模型 schema，但 audit total 与 level/template route 总数一致；
- `log_len=K+15` 且不超过 64。

### 9.4 Trace

- 相同 `span_id`、不同 `trace_id` 不得误连；
- 跨 split parent 默认 unmatched；
- 多 service 命中为 ambiguous，不采用 service-order first match；
- `duration_sum=count×mean_latency` 的算术在 fixture 中一致，但输出只保留 count/mean；
- 四种 status 的 slot order 固定；
- graph 只由 Train edge 构建，Test-only edge 不增图；
- `raw_edge=8`，serial/parallel 结果一致。

### 9.5 Integration/model smoke

- 输出严格为 Metric `[T,10,Rm]`、Log `[T,10,K+15]`、Trace `[T,10,10,8]`；
- 所有 arrays finite；Train/Test windows 不跨边界；
- `MyModel` 使用 manifest dimensions 完成一次 forward/backward smoke；
- smoke 不写正式 checkpoint，不计算正式 Test 指标；
- manifest 最后原子发布，失败运行不得留下 `COMPLETE` 状态。

## 10. Manifest 与缓存修改

新的 `ad_data_manifest.json` 必须额外包含：

```text
preprocessing_revision
policy path/hash
frozen schema path/hash
per-modality ordered slots/hash
source layout + run-table hash
decision_inputs=[train]
gt_labels_used_for_schema=false
test_used_for_selection=false
metric missing/applicability summary
log stable/RARE/UNK dimensions and frozen-state hash
trace directed edges, symmetric graph hash, parent-resolution counts
normalization fallback counts
serial/parallel execution metadata
validator result/hash
```

Cache binding 至少包含 source layout、grid、split、preprocessing schema hash 和实现 commit。旧 V3 cache 不得被新 revision 命中。中断运行只允许保留标记为 incomplete 的 staging/cache，completion manifest 必须最后发布。

## 11. Ada-RCA adapter 的同步边界

本轮 Ada-MGAD 修改不直接重做 Ada-RCA 68D。

可共享的只有 label-free raw metric registry：

```text
filename/source parsing
physical host mapping
logical name/scope
counter/gauge semantics
Train availability metadata
```

Ada-MGAD selector 生成统一 tensor slots；Ada-RCA selector 则允许每个 candidate 使用自己的可用 indicators，再由冻结的 Q90 形成 68D。后续单独修改 `src/e2e/gaia_rca_adapter.py`，去掉不必要的 10-service common intersection，但不改变四 channels、Q90、morphology、68D 或 scorer。该修改必须单独版本化和验证，不能混入首个 Ada-MGAD preprocessing patch。

## 12. 实施顺序与 gate

### P0A：补齐审计实现

- 让 Metric audit 使用 production-exact duplicate reduction；
- 在 counter/rate 与 core aggregation 后重新生成 quality/correlation；
- 完成 3 台 physical host 的质量与冗余统计；
- 完成 Train Drain3 distribution、Trace parent/edge/scale audit。

**Gate：** 完整机器可读输出、source/config hash 一致。否则 NO-GO。

### P0B：批准 policy

- 批准 Metric quality prevalence、correlation threshold/representatives；
- 批准 Log stable eligibility 和 K；
- 批准 Trace edge registry 与 scaler fallback support；
- 全过程不查看 Test F1 或 GT label。

**Gate：** `policy.status=APPROVED`，有独立 hash。否则 NO-GO。

### P1：冻结 schema

- 运行 freeze-only command；
- 输出 exact Metric slots、Log clusters、Trace edges/scalers；
- validator 检查维数、顺序、Train-only 字段和 source binding。

**Gate：** `FrozenPreprocessingSchema.status=FROZEN` 且 deterministic replay hash 一致。

### P2：实现代码

建议分四个可审查 patch：

1. module/schema/manifest seam + characterization tests；
2. Metric scope/missing/scaling；
3. Log stable/RARE/UNK/scaling；
4. Trace parent/8D/graph + integration。

每个 patch 只运行 synthetic/unit tests，不扫描全量数据。

### P3：preprocessing smoke

- 使用小时间窗和所有 10 个服务；
- 验证 schema、shape、finite、leakage firewall、determinism 和 model forward；
- 输出到 `artifacts/p5/v3_preprocessing_v2/smoke`，标记 `NOT FORMAL RESULT`。

### P4：正式 full preprocessing

- 独立输出根、revision-specific lock、single writer；
- 只读取 frozen schema；
- 完成后先运行 manifest validator。

### P5：正式 Ada-MGAD training

- 只有 P4 validator PASS 后启动；
- checkpoint/calibration/threshold 全部 Train-only；
- Test 只执行一次冻结推理；
- 性能不足时报告 limitation，不回头改 schema。

## 13. 本方案明确不做的事情

- 不修改 `src/model.py`、`src/model_util.py` 或 loss weighting；
- 不把 384 个 host features 全量加入；
- 不把 `redis/zookeeper` namespace 猜测映射到 candidate nodes；
- 不按 Test F1 选择 feature、K、correlation threshold、edge 或 scale；
- 不让 Test unseen template 创建新 Drain cluster；
- 不保留 `duration_sum + count + mean` 三套冗余 Trace slots；
- 不在本轮改变 Ada-RCA 68D；
- 不覆盖现有 V3 config、arrays、artifacts 或 checkpoints。

## 14. 当前结论

```text
CHANGE PLAN: P0/P1 GATES RETAINED
PRODUCTION CODE MODIFICATION: PHASE-1 SCHEMA/TRANSFORM CORE STARTED
FORMAL FULL PREPROCESSING/TRAINING: NO-GO
```

可以立即开始的是 P0A 审计补齐和 synthetic characterization tests。不能立即开始的是正式 full preprocessing、训练，以及任何依赖尚未冻结 exact Metric/Log schema 的结果比较。

## 15. Phase-1 实施记录（2026-09-14）

本轮已落地、且只用 synthetic/unit tests 验证的部分：

- 新增 `src/e2e/gaia_preprocessing/` 深 module；
- 新增 frozen schema loader，强制 `Train-only`、GT/Test firewall、Metric/Log 维数预算、固定 8D Trace 顺序以及 config/policy/audit hash binding；
- Metric：reset-aware counter rate、split-local time-aware fill、Train q01/q99 scaler、mean/max multi-core、统一 schema 与 3 个 observability slots 的 assembler；
- Log：`stable + 5 RARE + 5 UNK + 5 level` 路由、移除模型 `log_total`、`log1p + Train q99` scaler 与 UNK fallback chain；
- Trace：split-local `(trace_id, span_id)` parent resolution、ambiguous fail-closed、`count + mean_latency` 8D 聚合、Train-only edge/status/statistic scaler fallback；
- materialization 发布前 shape/dimension/finite validator；
- 旧 `build_ad_data` 对显式 V2 schema 声明 fail closed，避免静默回落到 V3 strict-intersection 实现；
- `run_i1_ad.py` synthetic smoke 的 config hash 改为绑定实际 `--config`。

尚未实施、且受 P0/P1 gate 阻止的部分：

- exact Metric representative slots、Log stable cluster IDs/K、Trace directed edge list/scalers；
- 真实 GAIA streaming adapters 与 V2 completion manifest；
- `configs/e2e/gaia_p5_v3_preprocessing_v2.json` 正式配置；
- 正式 full preprocessing、training、Test evaluation。

因此当前代码是可审查的 Phase-1 基础设施，不是可用于正式数据生成的 V2 pipeline。下一步必须先完成 P0A 机器可读审计并批准 P0B policy，再生成 P1 frozen schema，之后才能把三种 raw-data adapters 接到 `materialize.py`。
