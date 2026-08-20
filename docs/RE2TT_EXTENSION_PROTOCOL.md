# RE2-TT Protocol Extension V0.2

## Material Passport

- Origin: 本仓库只读侦察 + 现有冻结协议派生
- Origin Date: 2026-08-20
- Verification Status: 侦察结论（§3）= 仓库已验证；阶段设计与判定规则（§4–§8）= 已冻结；
  **E1–E5 执行结果（§9）= 仓库已验证 / artifact-verified**；E6–E7 = 未执行（被 §9.6 阻塞）
- Version Label: `re2tt_extension_protocol_v0.2`
- 当前扩展状态：**E1/E2/E3/E5 通过，E4 预登记 headroom 门禁 H-1 不通过 →
  `blocked_stages = ["E6","E7"]`**，详见 [§9](#9-e1e5-执行结果2026-08-20)
- 上游依据：[P2_G4_STAGE_RESULTS.md](P2_G4_STAGE_RESULTS.md)、
  [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)、
  [P2_EXPERIMENT_PLAN.md](P2_EXPERIMENT_PLAN.md)、[DATASET_AUDIT.md](DATASET_AUDIT.md)

## 1. 本扩展的目的与不做的事

P2-G4 在冻结门禁下判定 `no-go`：GAIA 双端点显著为正，RE2-OB 主端点 −0.002222、
次端点 −0.011111，且全部退化可溯源到单个 case。RE2-OB 的未学习基线 B2 已达
root-macro Avg@5 0.933333、C1-I 达 0.995556，可测量空间不足 0.005，小于两个单
case 量子（0.002222/量子）。因此 RE2-OB 上的负结果无法区分"H1 无效"与"数据集
饱和"。

本扩展引入 RCAEval RE2-TT（TrainTicket）作为第三 benchmark，用于**归因**这一
歧义，而不是用于翻案。

**不改变的内容（用户已确认，本扩展全程不得触碰）：**

- P2-G4 的 `p2_g4_decision = "no-go"` 与 H1"冻结门禁下未获支持"的记录永久保留，
  不重写、不加脚注式撤回；
- 不修改 inner selection objective（唯一目标仍为 root-service macro Avg@5）；
- 不放宽 Go/No-Go 阈值；
- 不删除、不弱化 RE2-OB，它继续是主判定读数的一部分；
- 不实现 M2-R / M2-D / M3-G；
- 不预判 H2/H3。

**物理隔离要求：** 本扩展不写入 `artifacts/p1/**` 与 `artifacts/p2/**`，且不重跑
`scripts/prepare_p1_manifests.py`、`scripts/prepare_p1_splits.py`、
`scripts/run_p1_sanity_baselines.py` —— 这三个脚本都在单次运行内同时写
GAIA 与 RE2-OB 子树，重跑即覆盖已冻结产物。

## 2. 权威边界说明

本目录下 `/home/zhangll24/RCA_project/datasets/RCAEval/RE2/RE2-TT_README.md` 是
2026-07-16 本地自建的分析文档，不是 RCAEval 官方数据说明，权威等级为 3–4 级。
本文件 §3 的每一项都由今日的本地只读扫描独立测得，不引用该文档的数值。

## 3. E0 只读侦察结论（仓库已验证，2026-08-20）

数据根：`/home/zhangll24/RCA_project/datasets/RCAEval/RE2/RE2-TT`（22 GB，
另有 `RE2-TT.zip` 2.7 GB）。

### 3.1 case 结构与标签分布

| 项 | RE2-TT | RE2-OB（对照） |
|---|---|---|
| condition 目录 | 30 | 30 |
| replicate | 1/2/3，全部存在 | 1/2/3 |
| case 总数 | 90 | 90 |
| 缺文件 / 解析失败 / `root_not_in_candidates` | 0 / 0 / 0 | 0 / 0 / 0 |
| root service | 5 × 18 case | 5 × 18 case |
| fault type | 6 × 15 case（cpu/delay/disk/loss/mem/socket） | 同 6 类 × 15 |

root service 为 `ts-auth-service`、`ts-order-service`、`ts-route-service`、
`ts-train-service`、`ts-travel-service`。`_parse_condition` 的
`rsplit("_", 1)` 在 RE2-TT 上原样可用：TrainTicket 服务名只含连字符、不含下划线。

### 3.2 候选空间（本扩展最重要的差异）

沿用**完全不变**的冻结候选规则（`simple_metrics.csv` 中 `_cpu`/`_mem` 后缀实体，
再减去 `_AUXILIARY_METRIC_ENTITIES`）：

- 原始派生：85 case 得 68 个实体，5 case 得 69 个；唯一差异实体是 `istio-init`，
  而它已在现有 `_AUXILIARY_METRIC_ENTITIES` 中；
- **过滤后 90/90 case 恒为同一 68 个候选实体**，因此
  `scripts/extract_p2_metric_features.py::_common_services` 的"每数据集单一候选集"
  约束成立；
- 90/90 case 的注入 root 均在候选集内。

候选空间从 RE2-OB 的 11 扩大到 68，而可能的 root 仍只有 5 个。这既是本扩展想要的
（干扰项从 6 个变成 63 个），也带来必须记录的边界：**5-root 先验退化没有被修复**
（见 §6.3）。

`simple_metrics.csv` 共 370 列，聚合后缀分布 cpu 68 / mem 68 / diskio 67 /
socket 68 / workload 28 / error 14 / latency-50 28 / latency-90 28。仅 28 个实体带
latency 聚合，但该文件只用于候选派生，Track C 特征一律来自 raw 三模态文件。

### 3.3 raw 三模态 schema 与时间戳单位

| 文件 | 列结构 | 时间戳列 / 单位 | 与 RE2-OB 是否一致 |
|---|---|---|---|
| `metrics.csv` | 1575 列 = `time` + 1269 `<entity>_container-<metric>` + 266 `<entity>_istio-request-total` + 39 `<node>_node-*`；1442 行 / 1 s 采样 | `time` / `s` | 命名规则一致 |
| `logs.csv` | 7 列：`time,timestamp,container_name,message,level,req_path,error` | `timestamp` / `ns` | 完全一致 |
| `traces.csv` | 11 列：`time,traceID,spanID,serviceName,methodName,operationName,startTimeMillis,startTime,duration,statusCode,parentSpanID` | `startTime` / `us` | 完全一致 |
| `inject_time.txt` | 单个 Unix 秒时间戳 | — | 一致 |

三种单位均由"换算后偏移量落在 ±720 s"独立验证，不采信文档声明。

`scripts/run_p1_metric_change.py::_service_columns` 的 `startswith(service + "_")`
前缀规则在 RE2-TT 上无碰撞：所有候选名不含下划线，因此 `Y_...` 只在 `Y == X` 时
以 `X_` 开头。`ts-order-service` / `ts-order-other-service`、
`ts-travel-service` / `ts-travel-plan-service` / `ts-travel2-service` 等相似名不会互相吞并。

### 3.4 窗口可行性

`inject_time` 在全部 90 case 中精确居中：`metrics.csv` 覆盖
`[t0 − 720 s, t0 + 720 s]`，logs/traces 同样为 ±720 s。冻结的
`T_pre = T_post = 300 s` 半开窗完整落在数据内，两侧各留 420 s 余量。
90 case 中 `pre < 300` 与 `post < 300` 的数量均为 0。

### 3.5 实体映射与三模态 coverage（±300 s 窗内，全 90 case 扫描）

- **未映射实体 0 个**：全部 `container_name` 与全部 `serviceName` 都已落在候选集
  内。RE2-TT **不需要** 任何 alias 表（对比：RE2-OB 需要
  `RE2_TRACE_SERVICE_ALIASES = {"frontendservice": "frontend"}`）。
- metrics：68/68 候选恒有列。
- logs：窗内有行的候选数 min 39 / median 45 / max 48 / mean 44.52（占 68 的 65.5%）。
- traces：窗内有 span 的候选数 min 20 / median 27 / max 27 / mean 26.07（占 38.3%）。
- **注入 root 的自身可观测性**：traces 在 pre 与 post 段 90/90 全覆盖；
  metrics 90/90；**logs 有 18/90 case 缺失，且全部 18 个都是
  `ts-train-service`（即它自己的全部 18 个 case）**。

另有两项 per-case 质量事实：

- 12/90 case 的窗内 coverage 明显偏低（log 39–40、trace 20），分布为
  root `ts-order-service` 7 / `ts-travel-service` 3 / `ts-route-service` 2，
  fault 分散，replicate 2/3 占 11 个 —— 更像采集批次的部署差异，不集中于单一 root
  分组；
- `ts-train-service_socket/1` 是唯一离群 case：pre 段 log 实体数为 **0**、trace 实体数
  仅 14，post 段正常（45 / 27）。metrics 仍为完整 ±720 s。按冻结的事件提取规则
  （"full [-300s,+300s) coverage; missing entities fully masked"）它会被整段掩码而不
  被剔除，无需新规则，但必须登记。

注入 root 的日志缺口是必须在 H1 replication 之前登记的实质限制：日志模态对
`ts-train-service` 在 ±300 s 内基本不可见，掩码会把它整段置 0.0。这是按 root 系统
性缺失（5 个 root-macro 分组中的 1 个），不是随机缺失，会直接反映在 root-macro
读数上。M1-S 的 staged 通道只用 metric + trace（见 §6.2），因此该缺口不影响 H1
的单因素本身，但会限制 C1-I 中 log whole 通道的贡献上限。

### 3.6 成本外推

| 项 | RE2-OB 实测 | RE2-TT 外推 | 倍数 |
|---|---|---|---|
| metric 候选列（90 case 合计） | 33,020 | ≈135,000 | ≈4.1× |
| log 扫描行数 | 15,053,223 | ≈25,166,700 | ≈1.67× |
| trace 扫描行数 | 34,461,235 | ≈102,909,150 | ≈3.0× |
| 原始体积 | 8.1 GB | 22 GB | 2.7× |
| service row 数 | 990 | 6,120 | 6.2× |

只读全扫（logs + traces 两列）实测 173 s / 90 case，因此完整 extraction 属于分钟
到十几分钟量级，不需要 nohup 长作业。

## 4. 扩展架构

### 4.1 独立产物根

```text
artifacts/ext/re2tt/
  manifests/            inputs.jsonl labels.jsonl sources.jsonl manifest.json
  source_snapshot/
  telemetry_diagnostics.json
  splits/               assignments.jsonl split_manifest.json
  baselines/{random,root_frequency,metric_change}/
  features/<extractor>/
  event_features/{p2_log_l0_v1,p2_trace_t0_v1}/
  runs/{c1_i,m1_s}/
  gate_audit.json  linear_ablation_*.json  m1_s_*.json
```

只有一个数据集，因此不再嵌套 `<dataset>/` 层。

### 4.2 数据集身份

| 项 | 取值 |
|---|---|
| `dataset` | `RCAEval-RE2-TT` |
| case ID 命名空间 | `RCAEval:RE2-TT:{relative_directory}` |
| case ID 前缀 | `re2tt-` + sha256 前 16 位 |
| URI base | `rcaeval://re2-tt/{case_id}` |

命名空间与 RE2-OB 不同，因此 case ID 不可能碰撞，两套 manifest 可并存。

### 4.3 schema version 复用而非新造

所有写入器原样复用，因此产物格式**就是**既有版本：`p1_rca_manifest_v1`、
`p1_split_manifest_v1`、`p1_source_snapshot_v1`、`p1_telemetry_diagnostics_v1`、
`p2_feature_bundle_v2`、`p2_nested_oof_v1`。为同一格式另造版本标签只会让现有
verifier 失效。仅新增的 **audit 文件**使用扩展专属标签：
`ext_re2tt_gate_audit_v1`、`ext_re2tt_m1_s_audit_v1`、
`ext_re2tt_paired_bootstrap_v1`。

### 4.4 adapter 参数化，不 fork

`src/data/rcaeval.py` 现有 6 处 RE2-OB 专用硬编码（辅助实体集、case ID 载荷、
`dataset` 串、URI base、replicate 元组、必需文件集）。做法是抽出冻结的
dataset profile：

```python
@dataclass(frozen=True)
class RCAEvalDatasetProfile:
    key: str                 # "re2ob" | "re2tt"
    dataset: str
    id_namespace: str
    id_prefix: str
    uri_namespace: str
    auxiliary_entities: frozenset
    replicates: Tuple[str, ...]
```

`load_re2ob_cases(raw_path)` 保留为绑定 `RE2OB_PROFILE` 的薄封装，签名不变；新增
`load_re2tt_cases(raw_path)`。

**强制回归**：重构后必须在临时目录用 adapter 重建 RE2-OB 的 `inputs.jsonl` /
`labels.jsonl`，与 `artifacts/p1/manifests/re2ob/` 的已冻结文件做**逐字节**比对。
这在不重跑 `prepare_p1_manifests.py` 的前提下证明既有 case ID 与 digest 未被改动。

### 4.5 复用的数据集无关库

导入、不复制：`write_manifest_bundle`、`verify_manifest_bundle`、
`read_manifest_cases`、`build_singleton_groups`、`assign_balanced_group_folds`、
`validate_split_integrity`、`write_source_snapshot`、`verify_source_snapshot`、
`src/data/telemetry_diagnostics.py` 的 `_scan_re2_metrics` /
`_scan_re2_event_stream`、`scripts/run_p1_sanity_baselines.py::_run_dataset`
（已完全参数化）、`scripts/run_p2_c0_metric.py` 的
`_fit_and_rank` / `_case_rows` / `_root_macro_avg5` / `_sha256` / `_write_jsonl` /
`RUN_SCHEMA_VERSION`。

冻结常量原样 import，不重新声明：`SPLIT_SCHEMA_VERSION`、`SELECTION_THRESHOLDS`、
`AXIS_WEIGHTS`、`DEFAULT_ONSET_SECONDS`、`DEFAULT_EVENT_ONSET_SECONDS`。

## 5. 阶段阶梯与门禁

| 阶段 | 内容 | 产物 | 门禁 | 实际结果（§9） |
|---|---|---|---|---|
| E0 | 只读侦察 | 本文件 §3 | 已完成 | — |
| E1 | adapter profile 化 + manifests + source snapshot | `manifests/`、`source_snapshot/` | E-G1 | ✅ 通过 |
| E2 | telemetry diagnostics（三模态 coverage / missingness / onset 支持性） | `telemetry_diagnostics.json` | E-G2 | ✅ 通过 |
| E3 | 5-fold singleton-stratified split | `splits/` | E-G3 | ✅ 通过 |
| E4 | B0/B1/B2 sanity baselines = **headroom 审计** | `baselines/` | E-G4 | ❌ **H-1 不通过** |
| E5 | 扩展 gate audit（E1–E4 独立复核） | `gate_audit.json` | E-G5 | ✅ 通过（并复判 E4 为 fail） |
| E6 | metric + L0/T0 特征提取 | `features/`、`event_features/` | E-G6 | ⛔ 被阻塞，未执行 |
| E7 | C1-I → M1-S 单因素 H1 replication + 配对 bootstrap | `runs/`、`m1_s_*.json` | E-G7 | ⛔ 被阻塞，未执行 |

**E1–E5 全部通过前不进入 E6/E7。** 这是用户指定的 audit-first 顺序。

## 6. 预登记判定规则（在看到任何 E4/E7 数值之前冻结）

### 6.1 E-G1 … E-G3：数据质量与协议合规

- E-G1：90/90 case 收录，exclusion = 0；90/90 root ∈ candidates；候选集恒定；
  RE2-OB manifest 逐字节回归通过；`verify_manifest_bundle` 通过；
  Label Firewall 全部 flag 为 false。
- E-G2：三模态 ±300 s coverage 与 30/60/120 s onset 支持性按与 P1/P2 相同的口径
  产出；`ts-train-service` 日志缺口写入产物而非仅写文档。
  **落点更正（在看到 E4/E7 数值之前作出）**：该缺口是 *root-conditioned* 事实，
  必须读 `labels.jsonl` 才能算出，而 `telemetry_diagnostics.json` 与 P1 同口径、
  刻意不接触标签（它只报告 label-free 的候选整体 presence ratio）。因此本条的
  产物落点是 **E5 的 `gate_audit.json` → `root_conditioned_coverage`**，
  而不是 `telemetry_diagnostics.json`。E5 是本扩展中唯一被允许在扫描原始遥测时
  读取标签的阶段（它是审计，不产出任何进入模型的特征），这一豁免只适用于 E5。
- E-G3：`assign_balanced_group_folds(seed=20260819, n_folds=5)` 在冻结
  `SELECTION_THRESHOLDS` / `AXIS_WEIGHTS` 下 `selection_eligible = True`；
  `validate_split_integrity` 通过；case/group 重叠 = 0。

### 6.2 E-G6/E-G7：单因素协议完全冻结

C1-I = whole M/L/T（17+5+8 = 30 值，60 列）。
M1-S = C1-I + metric/trace staged（105 值，210 列）。

**即使 RE2-TT 的 log staged 通过 0.80 content-complete 门槛，也不加入 M1-S。**
M1-S 的通道集合是跨数据集统一冻结的，改动它就毁掉 H1 的单因素可比性。RE2-TT 的
log staged coverage 只作为诊断报告。

其余一切照搬：inner 4-fold 轮换、`C ∈ {0.01,0.1,1,10}`、`onset ∈ {60,120}`、
per-case 权重 root 0.5 / 非 root 合计 0.5、tie-break（更简单模型 → 更强正则 →
更短且 coverage-supported 的 onset → 配置字典序）、主端点 root-service macro
Avg@5、关键次端点 root-macro AC@1、10,000 次配对 bootstrap（seed `20260819`，
RE2-TT 按 case 重采样）。每 dataset 160 inner + 5 outer fits。

### 6.3 E-G4 headroom 门禁（本扩展存在的理由）

在 root-service macro、冻结 5 折 OOF 上评估：

| 编号 | 判据 | 性质 | RE2-OB 对照值 |
|---|---|---|---|
| H-1 | B2 `metric_change` Avg@5 ≤ 0.90 | **门禁** | 0.933333（不通过） |
| H-2 | B2 `metric_change` AC@1 ≤ 0.90 | **门禁** | 0.855556（通过） |
| H-3 | B2 在主、次端点上均严格优于 B1 `root_frequency` | **门禁** | 通过 |
| H-4 | B1 `root_frequency` 的 AC@5 | 仅报告 | 1.000000 |

H-1 取 0.90 的理由：单 case 量子在 90-case / 5-root-macro 设定下为 AC@1 0.011111、
Avg@5 0.002222；0.10 的空间约等于 45 个 Avg@5 量子，足以让真实效应超出粒度。
RE2-OB 的 0.933333 只留下约 30 个量子给 B2 之上的全部方法，而 C1-I 实测已到
0.995556，仅剩 2 个量子。

H-4 是必须公开的**继承性退化**：RE2-TT 的 root 仍只有 5 个，B1 只要把这 5 个排在
最前就能拿到 AC@5 = 1.0。RE2-TT 修复的是"干扰项从 6 个变 63 个"，**没有**修复
5-root 先验，也**没有**改善 90-case 的统计粒度。这两条必须写进 limitations，不得
以"新数据集"为名淡化。

若 H-1/H-2/H-3 任一不通过：登记 RE2-TT 同样不具备判别力，**不执行 E6/E7**，
route ② 以"第三 benchmark 也饱和"收口。

### 6.4 E-G7 的读数规则与 claim 边界

- **主读数 = 三数据集 {GAIA main, RE2-OB, RE2-TT}**，按原冻结 Go/No-Go 评估。
  由于 RE2-OB 已判负，主读数下 H1 不可能转为"获得支持"；这一点在跑之前就登记，
  避免事后被当成翻案。
- **次读数 = {GAIA main, RE2-TT}**，明确标注为 replication reading，只用于归因。
- 允许得出"RE2-OB 的 no-go 可归因于天花板饱和"的结论，**当且仅当**同时满足：
  E-G4 全部通过；RE2-TT 主端点 95% 配对 bootstrap CI 下界 > 0；RE2-TT 次端点点
  估计 ≥ −0.01。此时两张表必须同时呈现，措辞上限为"H1 在 3 个 benchmark 中的 2 个
  上获得支持，RE2-OB 的失败可归因于饱和"，不得写成"H1 成立"。
- 若 RE2-TT 也未通过：这是比 P2-G4 更强的负面证据（有 headroom 仍无效），如实登记
  并停止，不再为 H1 寻找第四个数据集。
- 不论结果如何，Event-stage 的去留与是否进入 H2 由用户在看到 E7 结果后决定；本扩展
  不实现 M2-R / M2-D / M3-G。

## 7. 已知风险

1. **候选空间不可比**：GAIA 10 候选、RE2-OB 11、RE2-TT 68。跨数据集的绝对数值本
   来就不可比，这不是新问题（协议一直只做同数据集内的配对对照），但 RE2-TT 会把
   差距放大，报告时必须始终并列候选数。
2. **数据库实体进入候选集**：68 个候选里有大量 `ts-*-mongo` / `ts-*-mysql`，它们永
   远不是 root。不把它们剔除是刻意的——剔除需要用到 root 标签，属于泄漏。代价是
   模型可以学到"数据库实体不是 root"这一先验，其收益应被视为候选空间结构而非事件
   动态；解读 RE2-TT 上的模态贡献时必须考虑这一点。
3. **trace 掩码本身携带信息**：68 候选中仅约 27 个有 span，"是否有 trace"几乎是一
   个静态实体类型标签。M1-S 的 trace staged 通道可能主要在利用这一静态结构。E7 报
   告需附 coverage slice，并把该混杂写入 limitations。
4. **`ts-train-service` 日志不可见**（18/90 case），系统性而非随机。
5. **log 掩码与 root 类别完全混杂**（E5 实测后新增，见 §9.5）：缺失的 18 个 case
   恰好且仅是 `ts-train-service` 的全部 18 个 case，无一例外；反向也成立——其余 4 个
   root 的 72 个 case 全部双侧可见。因此"该 case 的 root 的 log 通道被掩码"这一位
   在 RE2-TT 上是"root **不是** `ts-train-service`"的近确定性指示器。任何用到 log
   observed mask 的模型都可能靠这条捷径抬高 root-macro 读数，而该捷径不迁移到别的
   数据集。三条约束：C1-I 的 log whole 通道读数必须附 coverage slice；M1-S 的 staged
   通道按 §6.2 只用 metric + trace，因此 H1 单因素本身不吃这条捷径；若 E7 出现
   `ts-train-service` 分组异常高/低，须先排除该混杂再解释。
6. **90-case 粒度未改善**，与 RE2-OB 相同。
7. **单一应用、每场景仅 3 replicate、无级联故障**，与 RE2-OB 同类局限。
8. **RE2-TT 上三个报告层同值**（E4 实测）：root 5×18、fault 6×15 均为完全均衡设计，
   故 overall / fault-type macro / root-service macro 恒等。这让 RE2-TT 无法暴露
   "总体好但某个分组塌陷"的失效模式——恰好是 GAIA 上最有信息量的那种失效模式。

## 8. 治理要求

每个 E-Gate 关闭时：向 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) 追加记录（append-only，
用其 §17 模板），同步 [RESEARCH_STATUS.md](RESEARCH_STATUS.md) 版本，更新
[README.md](README.md) 阅读顺序与状态行，并跑三道闸门：

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
git diff --check
```

任何数值必须附数据版本、split、seed、命令、提交哈希与产物路径。RE2-TT 结果一律
单列，不与 P2-G4 的表格合并。

## 9. E1–E5 执行结果（2026-08-20）

**状态标签**：`仓库已验证`。**证据等级**：`artifact-verified`（E5 独立复算，见 §9.6）。

### 9.0 物料护照

| 项 | 值 |
|---|---|
| 数据版本 | `RCAEval-RE2-TT`，`content_identity_sha256 = ad1396d0713fe21343a9db9233a60e51aa043026f8f39cd30fa1ba0fd5f15fc0` |
| 源根 | `/home/zhangll24/RCA_project/datasets/RCAEval/RE2`（`RE2-TT/` + `RE2-TT.zip`） |
| split | `artifacts/ext/re2tt/splits/`，`grouped_stratified_5fold`，`p1_split_manifest_v1` |
| seed | `20260819` |
| 提交哈希 | `c2af48f2be4066de7363d7e5f0871052e8564301` |
| 代码状态 | ⚠ 扩展 driver 当时**未提交**（untracked）。逐文件 SHA-256 见下表，用于在提交后核对是否为同一份代码 |
| 产物根 | `artifacts/ext/re2tt/`（gitignored） |

驱动代码 SHA-256（记录时刻的工作树内容）：

| 文件 | SHA-256 |
|---|---|
| `scripts/prepare_ext_re2tt_manifests.py` | `5bee68eca34a566a5f92553567a15e194c6fc7409c893a352ea82fcbce5e29b6` |
| `scripts/diagnose_ext_re2tt_telemetry.py` | `8b7b1a2d1e48ad03db85e4d2906029cac94bdc10ceefed5fdedac52b9f15790a` |
| `scripts/prepare_ext_re2tt_splits.py` | `47c56980c7a292a697e4badb6b7c6eee767da86db67f45e3cc36ce9a140cec0d` |
| `scripts/run_ext_re2tt_baselines.py` | `eb00fed7549e8a6de0f3dc94d4e78332cd8f890892dab478f77f74609df3cb8d` |
| `scripts/audit_ext_re2tt_gates.py` | `698bacd8094b6557fa9016c8423f1ef62cb2d0a0b195db6eea77fc1b6fc24f5d` |
| `src/data/rcaeval.py` | `abda9055d4ddb7675d295b3c03f27e7f46776147fe3a7dc5b2f8fe93545864dd` |
| `src/data/telemetry_diagnostics.py` | `6112aecf18df47faaae902acfd441ac3a9691af3a183f0b76004b253b82819e1` |
| `src/data/__init__.py` | `02bb2eb1acb9bc147c1a3d8c4029e43ac4afdec43c91378529a394413f2695fe` |
| `scripts/run_p1_metric_change.py` | `1148e04ad3ec8e439fc8848d2b5057ee9057a06494841221298a150e794f6594` |

复现命令（全部走默认值；P1/P2 产物根从不被写入）：

```bash
python scripts/prepare_ext_re2tt_manifests.py    # E1
python scripts/diagnose_ext_re2tt_telemetry.py   # E2
python scripts/prepare_ext_re2tt_splits.py       # E3  (--folds 5 --seed 20260819)
python scripts/run_ext_re2tt_baselines.py        # E4
python scripts/audit_ext_re2tt_gates.py          # E5
```

### 9.1 E1 — manifests + source snapshot（E-G1 通过）

| 判据 | 实测 |
|---|---|
| case 收录 / 排除 | 90 / 0 |
| 候选服务数、候选元组变体数 | 68、**1**（全 case 同一候选集） |
| case group | 90 singleton，`largest_group = 1` |
| root 分布 | `ts-auth-service` / `ts-order-service` / `ts-route-service` / `ts-train-service` / `ts-travel-service` 各 **18** |
| fault 分布 | `cpu` / `delay` / `disk` / `loss` / `mem` / `socket` 各 **15** |
| root ∉ 候选集 | `[]`（0 个） |
| `assert_label_free` 通过的 input | 90 / 90 |
| 原始路径泄漏扫描 | `raw_home_paths` / `condition_directory_names` / `release_directory` 全为 `false` |
| 与冻结 RE2-OB 的 case_id 冲突 | 0 |
| **RE2-OB 逐字节回归** | **passed**，比较 `inputs.jsonl` / `labels.jsonl` / `sources.jsonl` / `groups.jsonl` 四个文件（raw root `…/RCAEval/RE2-OB`，90 case） |

`manifests/manifest.json`（`p1_rca_manifest_v1`）digest `e7c0bb4c8fd3866f7af11db3c08078133a3f8557517ff8886bdd62b7743a7aa2`；
`inputs.jsonl` `4f88c44456b39ed2488ec7a22cb17b3d0691906029f173b41143e09f93368b87`（90 行，
`prediction_input`）、`labels.jsonl` `950885c2d89872f670d8cb06121a0d8d4c3e9fbd6be5e97a29af89a6013481ca`
（90 行，`training_evaluation_label`）、`sources.jsonl`
`c9e42d7e21cb920d68e516b4e0cee477d74047e9bfaaa867908652ef61243ac6`、`groups.jsonl`
`a427c3897d15e3068c715900ba76db00ed68a529bcd00d6516e2abd701d73db6`、`excluded.jsonl` 空文件。

source snapshot（`p1_source_snapshot_v1`，`source_root_name = "RE2"`）：1 个归档
`RE2-TT.zip` 2,801,345,134 B（`6706311d2c00d9f5a335f73f6a11f4ad4417522abed7e7ee8c88b8e898088805`）
+ 360 个被消费文件 22,752,639,186 B（logs 10,063,063,616 / traces 10,291,805,680 /
metrics 2,397,768,990 / inject_time 900；每种角色 90 个文件）。

**RE2-OB 逐字节回归是本阶段唯一真正的风险控制点**：adapter 被 profile 化以容纳
RE2-TT，若这次改动扰动了 RE2-OB 的任何一个字节，所有已记录的 P1/P2 数字都会失效。
它通过，说明参数化是无损的。

### 9.2 E2 — telemetry diagnostics（E-G2 通过，落点按 §6.1 更正）

`artifacts/ext/re2tt/telemetry_diagnostics.json`（`p1_telemetry_diagnostics_v1`，
`scan_scope = "full"`，digest `eb4cba13ca8b8d6c8f93143d8dd78ae739adc7ddf8ce08c0a5e369b9b2a19522`）：

| 模态 | 行数 | 无效时间戳 | 必填字段缺失 | t0 落在观测区间的 case |
|---|---|---|---|---|
| metrics | 129,690 | 0 | 无 | 90 / 90 |
| logs | 21,291,651 | 0 | `timestamp` 0、`container_name` 0 | **89 / 90** |
| traces | 67,345,051 | 0 | `parentSpanID` 563,677（root span，预期）；其余 0 | 90 / 90 |

- metrics 采样节律**精确 1000 ms**：`expected_timestamps = observed_unique_timestamps
  = 129,690`，`duplicate_timestamp_rows = 0`，`missing_timestamps = 0`；每 case 恒
  1441 行、恒 `[t0 − 720 s, t0 + 720 s]`。`missing_value_ratio = 0.003631987690551998`
  （736,632 / 202,817,868 值单元）。
- 那 1 个 t0 不在 log 观测区间内的 case，其 log 起点为 **+123.289 s**（即 t0 之前完全
  无日志）——与 §3.5 记录的 `ts-train-service_socket/1` 一致。
- 候选整体（label-free）窗内 presence ratio，按冻结的 ±300 s 与 onset 候选：

| 模态 | 30 s | 60 s | 120 s | 300 s |
|---|---|---|---|---|
| metrics | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| logs | 0.636601 | 0.637092 | 0.643791 | 0.654739 |
| traces | 0.376961 | 0.378595 | 0.382680 | 0.383333 |

  metrics 在 68/68 候选上恒有列，故 60/120 s onset 在 metric staged 通道上完全被支持
  （这正是 M1-S 的 staged 通道所需）。logs 在 30/60 s 上有 case 掉到 0，traces 掉到
  7/68——再次印证 30 s 只能作为不受支持的对照。
- 拓扑：无显式静态服务图（`explicit_static_service_graph = false`），
  90/90 case 可由 trace 推导动态图，`cluster_info` 被明确归类为 log-template 元数据而
  非服务拓扑。这与 GAIA/RE2-OB 的处理一致，不改变任何协议。

### 9.3 E3 — 冻结 5 折（E-G3 通过）

`assign_balanced_group_folds(seed=20260819, n_folds=5)`，`AXIS_WEIGHTS` 与
`SELECTION_THRESHOLDS` 均直接 import 自 `scripts/prepare_p1_splits.py`：

| 判据 | 阈值 | 实测 |
|---|---|---|
| `selection_eligible` | — | **true** |
| fold case 数 | — | 18 / 18 / 18 / 18 / 18 |
| `max_case_count_relative_deviation` | ≤ 0.15 | **0.0** |
| `max_fault_type_total_variation` | ≤ 0.10 | **0.0** |
| `max_root_service_total_variation` | ≤ 0.10 | **0.06666666666666665** |
| 每折含全部 5 root / 6 fault | — | true / true |
| 折间 case 重叠 / group 重叠 | = 0 | 0 / 0 |
| `validate_split_integrity` | — | 通过 |

`assignments.jsonl` `6a1f165dbdbb3b651f7def513ea3447873da7d5276f685bb1aec8dc8a40f1662`、
`split_manifest.json` `7e590bb408e163ab0b8474fba26162abbdd6d05fdee861b2adcad04e27aa684c`；
重跑逐字节一致。与冻结 RE2-OB split 的 case_id 冲突 0。

### 9.4 E4 — B0/B1/B2 与 headroom 门禁（E-G4 **不通过**）

root-service macro，冻结 5 折 OOF。**RE2-TT 单列**，右侧两列仅作参照，不合并统计：

| baseline | dataset | 候选数 | AC@1 | AC@3 | AC@5 | **Avg@5** | MRR |
|---|---|---|---|---|---|---|---|
| B0 `random` | **RE2-TT** | 68 | 0.000000 | 0.033333 | 0.077778 | **0.037778** | 0.063523 |
| B0 `random` | RE2-OB | 11 | 0.055556 | 0.300000 | 0.400000 | 0.255556 | 0.254665 |
| B0 `random` | GAIA main | 10 | 0.099289 | 0.321772 | 0.509778 | 0.313242 | 0.298441 |
| B1 `root_frequency` | **RE2-TT** | 68 | 0.166667 | 0.555556 | 1.000000 | **0.566667** | 0.424074 |
| B1 `root_frequency` | RE2-OB | 11 | 0.166667 | 0.555556 | 1.000000 | 0.566667 | 0.424074 |
| B1 `root_frequency` | GAIA main | 10 | 0.099398 | 0.300000 | 0.492101 | 0.296640 | 0.291806 |
| B2 `metric_change` | **RE2-TT** | 68 | 0.822222 | 0.922222 | 0.944444 | **0.904444** | 0.878432 |
| B2 `metric_change` | RE2-OB | 11 | 0.855556 | 0.933333 | 0.988889 | 0.933333 | 0.904630 |
| B2 `metric_change` | GAIA main | 10 | 0.531066 | 0.745685 | 0.821936 | 0.708573 | 0.664663 |

run manifest digest：B0 `f5bb15b29f33e3a08304fca0d3c83efb0dc1e2b40f5437a2d8ada266c1e92c0c`、
B1 `2ba40b4f94fae1b87694821a84cd71e9b90040fb2a47240fa3ac2d8d4aa32330`、
B2 `7294f3c841d7b4dcec94c71204d9a44f37522f1d8f5bbed3246f8990f0ba0a8b`。
B2 使用与 RE2-OB 完全相同的冻结配置（300 s 半开窗、每侧 ≥ 2 样本、top-5 特征、
cap 20.0），该配置等值性由 `tests/test_ext_re2tt_extension.py::FrozenReuseTest` 钉住。

**预登记门禁判定（`headroom_gate.json`，`ext_re2tt_headroom_gate_v1`）：**

| 编号 | 判据 | 性质 | RE2-TT 实测 | 结果 |
|---|---|---|---|---|
| H-1 | B2 Avg@5 ≤ 0.90 | 门禁 | **0.9044444444444444** | ❌ **不通过** |
| H-2 | B2 AC@1 ≤ 0.90 | 门禁 | 0.8222222222222222 | ✅ 通过 |
| H-3 | B2 在主、次端点均严格优于 B1 | 门禁 | Avg@5 +0.33777777777777773、AC@1 +0.6555555555555556 | ✅ 通过 |
| H-4 | B1 AC@5 | 仅报告 | **1.000000** | 与 RE2-OB 同值 |

`decision = "fail"`，`failed_gates = ["H-1"]`。超出天花板 **0.004444**，恰为
**2 个 Avg@5 case 量子**（1 量子 = 0.002222）。

三条必须写进 limitations 的读数：

1. **B0 确实大幅下降**（Avg@5 0.255556 → 0.037778）：68 候选的干扰项扩张是真实的，
   RE2-TT 在"随机基线难度"这一维上确实比 RE2-OB 难得多。
2. **B1 与 RE2-OB 逐位同值**（0.166667 / 0.555556 / 1.000000 / 0.566667 / 0.424074）。
   已排除接线错误：run manifest 绑定的是 `artifacts/ext/re2tt/{manifests,splits}`
   且 digest 相符，预测把全部 68 个 RE2-TT 服务排序（训练折可见的 5 个 root 在前、
   其余 63 个按字典序在后）。同值是**设计的结构性后果**：两个数据集都是 5 root × 18
   case 的均衡设计，而 B1 只依赖训练折的 root 频率先验，与候选总数无关。
   H-4 因此成立：**RE2-TT 没有修复 5-root 先验退化**，只修复了干扰项数量。
3. **B2 几乎没有下降**（0.933333 → 0.904444，仅 −0.028889）。这是门禁失败的实质：
   干扰项从 6 个变成 63 个之后，一个纯 within-case metric shift 的无训练基线仍然拿到
   0.904 的主端点。RE2-TT 的天花板问题**不是**候选空间造成的，而是"注入式单点故障 +
   完整 metric 覆盖"这一实验设计本身造成的——换 RCAEval 的另一个 release 无法解决。

### 9.5 E5 — 独立门禁审计（E-G5 通过；同时给出 root-conditioned coverage）

`artifacts/ext/re2tt/gate_audit.json`（`ext_re2tt_gate_audit_v1`，digest
`8895138ab617d49f8928072c571bf0bdb5a8cfadb0600ac29a6c528fda197cf4`）。
审计不 import 任何 E1/E3/E4 driver 的计算 helper，因此 driver 里的 bug 无法自我隐藏。

| 审计节 | 结果 |
|---|---|
| `e1_manifests` | passed（5 个文件 digest 全部重算一致；`verify_manifest_bundle` 通过） |
| `e1_source_snapshot` | passed（`archives.jsonl` / `consumed_files.jsonl` digest 重算一致） |
| `e3_splits` | passed（重建 `SplitAssignment`/`CaseGroup` 后 `validate_split_integrity` 通过；`fold_label_consistent = true`） |
| `e4_baselines` | passed（每条预测都是 68 服务的无重复全排列；三个报告层的每个指标与记录值差 ≤ `1e-12`；OOF 覆盖完整） |
| `integrity_gates_passed` | **true** |
| headroom 独立复判 | `decision = "fail"`、`failed_gates = ["H-1"]`、`matches_recorded_gate = true` |

**root-conditioned coverage（本扩展中唯一读标签的原始遥测扫描；`§6.1` 指定的产物落点）**
定义：被标注的 root 服务在 `[t0 − 300 s, t0)` 与 `[t0, t0 + 300 s)` 两侧各至少有一行
（metrics 额外要求该行有有限数值单元）：

| 模态 | 双侧覆盖 | 窗内有任何活动 | 双侧未覆盖 | 未覆盖 case 的 root 分布 |
|---|---|---|---|---|
| metrics | **90 / 90** | 90 | 0 | — |
| traces | **90 / 90** | 90 | 0 | — |
| logs | **72 / 90** | 79 | 18 | `ts-train-service` **18**（该 root 全部 case） |

18 个未覆盖 case 中：11 个在 ±300 s 内**完全没有** root 日志行、6 个仅 post 侧、
1 个仅 pre 侧。其余 4 个 root 的 72 个 case **全部**双侧可见。这构成 §7 第 5 条登记
的完全混杂风险。

### 9.6 阶段结论与阻塞

- E1、E2、E3、E5 全部通过；**E4 的预登记 headroom 门禁 H-1 不通过**。
- 按 §5「E1–E5 全部通过前不进入 E6/E7」与 §6.3「若 H-1/H-2/H-3 任一不通过：登记
  RE2-TT 同样不具备判别力，**不执行 E6/E7**」，`gate_audit.json` 记录
  `blocked_stages = ["E6", "E7"]`。**E6/E7 未执行。**
- 用户指令「不放宽 gate」在此优先于任何"差一点就过"的论证：0.904444 与 0.90 的差距
  只有 2 个 case 量子，但门禁是在看到数值之前冻结的，事后调整阈值会使整条 route ②
  失去证据价值。
- **route ② 的收口结论**：第三个 benchmark 也饱和。P2-G4 的原始 `no-go` 与
  H1「冻结门禁下未获支持」保持不变，且现在多了一条更强的证据——RE2-OB 的天花板不是
  它自己的偶然缺陷，而是 RCAEval RE2 注入式设计的共性；扩展到 RE2-TT（候选空间 6.2×、
  原始体积 2.7×）后，无训练 metric 基线的主端点只降了 0.028889。
- **不做的事**：不修改 inner selection objective、不放宽门禁、不删除 RE2-OB、
  不为 H1 寻找第四个数据集、不实现 M2-R / M2-D / M3-G、不预判 H2/H3。
- 是否在此基础上继续（例如接受"RE2-TT 也饱和"作为终局并转向别的证据策略），
  是用户决策，不由本文件决定。
