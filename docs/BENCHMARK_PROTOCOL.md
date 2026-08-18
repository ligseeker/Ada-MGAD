# P1 Unified Event-level / Service-level RCA Benchmark Protocol V0.5

> 决策状态：**V0.1 用户已确认；V0.2–V0.5 为数据诊断与收尾审计驱动的执行修订**
> 实现状态：**G1–G8 与 P1 reproducibility closeout completed**
> 最近核验：2026-08-19

## 1. P1 目标

P1 不设计复杂网络。目标是把 GAIA fault injection 与 RCAEval RE2-OB failure case 映射为同一种 RCA 任务，并建立能够阻止标签泄漏的统一 schema、评价器、sanity baselines 和数据诊断。

统一任务单位为一个独立的 **Fault Case / Fault Event**，不再使用“一个重叠滑动窗口 = 一个 RCA 样本”。

## 2. 统一上下文

每个 case 使用固定锚点窗口：

\[
[t_i^0-T_{\mathrm{pre}},\ t_i^0+T_{\mathrm{post}}].
\]

协议不依赖“真实故障结束时间”。真实结束时间若存在，只能作为诊断元数据或敏感性分析依据，不能成为所有数据集必须提供的输入。

G8 诊断后，主协议冻结 `T_pre=T_post=300 s`。切片统一使用毫秒时间和半开区间
`[t0-300 s, t0+300 s)`；不合成边界填充值。定时 metric 缺失使用显式 mask，
log/trace 无事件只表示该窗口无活动。GAIA split 原子组必须使用实际注入区间与
该上下文区间之并集的重叠连通分量。

## 3. 逻辑 schema

```python
RCACase:
    case_id: str
    dataset: str
    anchor_time: int | float
    services: tuple[str, ...]
    metrics: TelemetryRef | None
    logs: TelemetryRef | None
    traces: TelemetryRef | None
    topology: TopologyRef | None
    root_service: str       # label only
    fault_type: str | None  # analysis metadata only
    metadata: dict
```

为了落实 Label Firewall，代码层建议物理拆分为：

```python
RCACaseInput:
    case_id, dataset, anchor_time, services,
    metrics, logs, traces, topology, metadata_without_labels

RCACaseLabel:
    case_id, root_service, fault_type
```

模型的 `fit/transform/predict` 接口只接收 `RCACaseInput`；label 只由 loss、splitter、evaluator 和 error analysis 持有。

### 必须满足的 schema invariants

- `case_id` 在全数据集中唯一且稳定；
- `services` 非空、去重、顺序确定；
- `root_service in services`，否则 adapter 必须显式失败或记录 invalid case，不能静默补入；
- ranking 必须覆盖且仅覆盖 `services`，不得重复；
- 所有 timestamp 统一时区与单位；
- 缺失模态显式表示，不能用全零张量冒充真实观测而不带 mask；
- topology 的来源和时间范围可追溯；
- `fault_type` 不进入测试期 feature、normalization、candidate filtering 或 ranking heuristic。

## 4. 数据集映射

### 4.1 GAIA adapter

```text
run_table_2021-07.csv
        |
        v
parse_anomaly_event / label_events.csv
        |
        v
one valid injection event = one RCACase
        |
        +--> t0 = parsed injection start
        +--> root_service = injected service
        +--> fault_type = parsed event type
        +--> telemetry around t0
```

当前 `util/GAIA/pre_GAIA.py` 已能生成 `label_events.csv`，这是可复用的事件解析资产；同一文件随后把事件映射为 30 s `label.csv`，这一窗口映射不能直接作为新 RCA sample 定义。

GAIA adapter 在实现前必须审计：

- 哪些 event 类型有效，哪些为 normal/error/unknown；
- 单根因、多根因和重叠 event 如何定义；
- event 的 service 名称如何映射到 telemetry entity；
- 相邻/重叠事件的上下文污染如何处理；
- 可用拓扑是静态图还是由事件窗口 traces 构建；
- 独立 injection 数量与各 fault/root 分布。

当前 inventory 为 16,200 条 supported、time-bounded operational anomaly
events。主结果使用 `anchor_unique_root_service` cohort：若另一 root service 的
事件在目标 `t0` 仍活跃，则该 case 从主表移入 multi-root sensitivity；同 root
并发仍保留。主 cohort 为 13,470 cases，multi-root sensitivity 为 2,730 cases。
label 表示目标运维事件指定的 service，不声称它是整个 ±300 s context 中唯一
因果故障。

### 4.2 RCAEval RE2-OB adapter

当前本地数据组织以独立 case 为单位，每个正式 case 使用 `metrics.csv`、
`inject_time.txt`、`logs.csv`、`traces.csv`。

```text
one RE2-OB case directory/index row = one RCACase
        |
        +--> t0 = inject_time
        +--> root_service = official annotation
        +--> fault_type = official fault metadata
        +--> metrics/logs/traces around t0
```

当前正式 manifest 覆盖 RE2-OB 90/90 cases。由于本地资产没有保存上游
commit/tag/DOI/version 文件，执行版本以 content-addressed snapshot 固定：原始
`RE2-OB.zip` 加上实际消费的 360 个文件均有逐文件 size/SHA-256，组合内容身份为
`cec0030da8b499914af7979283eb1a2cd4f2cad2430742c70130829cd28133c9`。该身份
能验证实验字节一致性，但不能替代上游 provenance 声明。

### 4.3 候选服务集合

主实验使用所有可观测服务：

\[
V_i=\{v_1,\ldots,v_N\}.
\]

候选集合只能由 service registry、telemetry entities 或测试时可得 topology 推导。禁止依据真实根因选择邻域、删去困难服务或补入 label-derived candidate。

## 5. Label Firewall

### 测试期允许访问

\[
\{t_0,X^m,X^l,X^t,G,V\}.
\]

### 测试期禁止访问

\[
\{v^*,c,\text{root indicator},\text{future label statistics}\}.
\]

以下对象只能由训练 loss、splitter、evaluator 或 error analysis 访问：

- `root_service`；
- `fault_type`；
- root indicator；
- 测试集 root frequency；
- 由 root 决定的图边、候选过滤或 normalization；
- 使用全部数据拟合的词表、统计量或 embedding。

建议为 feature pipeline 写负向测试：把 label 字段放入会在访问时抛错的 proxy，确认 `transform/predict` 无读取行为。

## 6. Split 与学习型预处理

固定顺序：

```text
Raw Cases
   -> Case-level Split
   -> Train / Validation / Test
   -> Fit preprocessing on TRAIN only
   -> Transform Train / Validation / Test
   -> Train / Predict / Evaluate
```

必须在 case split 之后拟合：

- normalization statistics；
- log vocabulary/template/embedding statistics；
- learned missing-value parameters；
- root frequency prior；
- 任何跨 case 聚合阈值。

### RE2-OB

对需要跨 case 学习的方法，主协议是 seed `20260819` 的 5-fold case-level
cross-validation，最终拼接 90 个 out-of-fold predictions 后统一评价。90 个
singleton case 按 root service 与 fault type 平衡，joint stratum 仅作为权重
0.25 的软目标；五折各 18 cases，每种 fault 每折 3 条，每个 root 每折 3–4 条。

不需要跨 case 训练的 baseline 在相同 90 cases 上直接运行，但不得利用其他 case 的测试标签统计。

### GAIA

±300 s 上下文与实际注入区间的并集形成 322 个不可拆分组，最大组 471。完整
inventory assignment 使用 seed `20260819` 的 grouped-stratified 5-fold，fold
sizes 为 3239/3239/3239/3241/3242；root/fault/joint 轴权重为 1/1/0.25。
主 cohort 继承这些 group-safe folds，过滤后 sizes 为
2693/2684/2711/2705/2677，五折仍覆盖全部 5 个 fault types 和 10 个 roots。

作为候选的 context-purged temporal block 虽然大小和总体 TV 均通过软门槛，
但 12 条 CPU fault 全部落在 fold-4，前四折均无 CPU，且 fold-0 无 access-
permission fault，因此不满足“每个由至少 5 个原子组支持的 root/fault 类别
必须出现在每折”的覆盖约束。该 assignment 只保留为 temporal-shift
sensitivity，不作为主 OOF split。

无论采用哪种 split，同一次 injection 及其任何 telemetry 片段都只能属于一个 partition。

## 7. 排名与评价指标

对 single-root service-level RCA：

\[
AC@k=\frac{1}{M}\sum_{i=1}^{M}
\mathbb I[v_i^*\in R_i^{1:k}].
\]

报告：

- AC@1；
- AC@3；
- AC@5；
- Avg@5；
- MRR。

其中：

\[
Avg@5=\frac{1}{5}\sum_{j=1}^{5}AC@j,
\]

\[
MRR=\frac{1}{M}\sum_{i=1}^{M}\frac{1}{rank_i(v_i^*)}.
\]

RCAEval 当前 evaluator 同时实现 fine-grained 与 service-level coarse-grained accuracy/average；本项目直接输出唯一服务列表，避免从 `(entity, metric)` 压缩到 service 时出现重复 entity。

主结果至少分三层：

1. Overall；
2. Fault-type macro；
3. Root-service macro。

其中 AC@1 与 Avg@5 是论文主指标，MRR 是补充排名质量指标。运行时间应记录，但在 P1 不作为 Gate。

## 8. P1 sanity baselines

### B0 Random

对全部候选服务按 `SHA256(seed, case_id, service)` 生成可复现随机完整排名，
seed 为 `20260819`，用于验证 evaluator 和候选集合。

### B1 Root Frequency Prior

仅从当前四个训练 folds 统计根因频率；同频服务按名称升序。用于检测数据集
root 分布捷径。每折保存 train/test ID digest 与 overlap=0 证据。

### B2 Metric Change Score

比较半开窗口 `[t0-300s,t0)` 与 `[t0,t0+300s)`。每 feature 使用 pooled-std
标准化的绝对均值位移，score cap=20，单侧至少 2 个样本；每服务聚合 Top-5
feature scores。无有效 feature 的服务在有观测服务之后按名称回退，并报告
observed/fallback coverage slices。B2 不做跨 case 拟合。

P1 不实现新神经网络，也不把 Ada-MGAD score 当作跨数据集主 baseline。

## 9. 数据诊断报告

每个数据集至少输出：

- 有效/无效 case 数及排除原因；
- 候选服务数与分布；
- fault-type 分布；
- root-service 分布；
- Metrics/Logs/Traces 覆盖率；
- telemetry 相对 (t_0) 的时间跨度；
- timestamp 粒度与重复情况；
- missingness；
- topology 覆盖率与来源；
- case 重叠/潜在污染；
- `root_service not in services` 等 schema violation。

报告必须在选择 `T_pre/T_post`、GAIA split 和 RE2-OB fold 策略之前生成。

## 10. P1 Gate

- **G1**：GAIA 中“一次有效 fault injection = 一个 RCACase”。
- **G2**：RE2-OB 90 个官方 failure cases 全部被读取或逐一给出排除原因；目标为 90/90 有效。
- **G3**：两个数据集都能输出无重复的完整 service ranking。
- **G4**：AC@1/3/5、Avg@5、MRR 通过人工 toy case 单元测试。
- **G5**：Random、Frequency、Metric-change 在两个数据集上运行并保存结果。
- **G6**：case/event 不跨 split；有自动测试或可审计 manifest。
- **G7**：preprocessing 与预测路径不读取 test root/fault label；有 Label Firewall 测试。
- **G8**：生成完整 dataset diagnostic report。

G1–G8 与 reproducibility closeout 已完成。最终机器审计重新验证了 9 组
baseline outputs、split/source/cohort checksum bindings、完整 rankings、指标重算、
Label Firewall、349,120,558 行全量诊断以及 RE2 原始字节；详见
[P1_REPRODUCIBILITY_AUDIT.md](P1_REPRODUCIBILITY_AUDIT.md)。P2 可在不修改本
协议冻结项的前提下开始。

## 11. 建议工程结构

```text
docs/
src/
  data/
    schema.py
    gaia.py
    rcaeval.py
    source_snapshot.py
  evaluation/
    metrics.py
    evaluator.py
  baselines/
    random.py
    frequency.py
    metric_change.py
scripts/
  prepare_p1_manifests.py
  prepare_p1_splits.py
  pin_p1_rcaeval_source.py
  audit_p1_gates.py
tests/
  test_schema.py
  test_metrics.py
  test_label_firewall.py
  test_split_integrity.py
```

P1 暂不新增 Standalone RCA 模型目录或模型文件；现有 Ada-MGAD `src/model.py` 不属于 P1 实现范围。

## 12. 官方依据

- [RCAEval 官方仓库与数据说明](https://github.com/phamquiluan/RCAEval)
- [RCAEval evaluator：service-level AC@k / Avg@k](https://github.com/phamquiluan/RCAEval/blob/main/RCAEval/benchmark/evaluation.py)
- [RCAEval 官方论文](https://arxiv.org/abs/2412.17015)
