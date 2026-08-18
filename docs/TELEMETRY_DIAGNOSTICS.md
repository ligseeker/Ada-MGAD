# P1 全量遥测诊断报告

> 诊断日期：2026-08-18 至 2026-08-19
> 状态：G8 completed；主窗口已冻结；后续 split assignment 已完成
> 诊断 schema：`p1_telemetry_diagnostics_v1`
> 主产物：`artifacts/p1/telemetry_diagnostics.json`
> SHA-256：`957ef5f94f50e096b53e7f0107920361cdb70ad5b0610a87640bbacbc327c6b6`
> Context 产物：`artifacts/p1/context_group_diagnostics.json`
> Context SHA-256：`6dca047930d342bbc7902ca686dd55e8e1cb53cb43a889e9d8f67bb597efec12`

## 1. 范围与口径

本报告基于 GAIA MicroSS 2021-07 与 RCAEval RE2-OB 的本地原始遥测，采用
10 万行分块流式读取。全量扫描成功退出，六个“数据集 × 模态”组合均无
未完成文件。

- metric missingness：分别统计空/NaN value cell、按文件推断的规则采样时间缺口和重复 timestamp；
- log/trace：统计必填字段和时间解析失败，并用 30 s bin 表示活动覆盖；无事件不等同于 scheduled-sample missingness；
- 候选对称窗口：±30、±60、±300、±600、±1800 s；
- layout digest 绑定相对路径与文件大小；没有计算全部 telemetry 字节的内容哈希；
- topology 只审计来源和可构建性，没有在本轮物化 trace-derived graph。

## 2. 全量读取结果

| 数据集 | 模态 | 扫描范围 | 行数 | 原始大小 | 未完成文件 |
|---|---|---:|---:|---:|---:|
| GAIA | metrics | 5,724 个候选服务文件 | 182,820,535 | 4.02 GB | 0 |
| GAIA | logs | 10/10 service files | 87,974,871 | 18.45 GB | 0 |
| GAIA | traces | 10/10 service files | 28,681,438 | 8.34 GB | 0 |
| RE2-OB | metrics | 90/90 cases | 129,256 | — | 0 |
| RE2-OB | logs | 90/90 cases | 15,053,223 | — | 0 |
| RE2-OB | traces | 90/90 cases | 34,461,235 | — | 0 |

合计读取 349,120,558 行。GAIA metric 目录共有 6,640 个 CSV；未作为 RCA
候选服务扫描的 916 个文件全部属于 Redis（544）或 ZooKeeper（372）基础设施
实体，共 771,831,301 bytes。当前协议冻结的 GAIA 候选集合是 10 个 application
service instances，因此这 916 个文件属于显式的非候选范围，不是读取失败。

## 3. GAIA 诊断

### 3.1 Case、候选与重叠

- 有效候选注入 16,200；排除 952 条非 case 记录；`root_service not in services` 为 0；
- 每个 case 有 10 个候选应用服务；
- 4,037 个 case 参与原始注入区间重叠，形成 914 个非单例连通组，最大组 39；
- fault/root 分布和排除原因见 [DATASET_AUDIT.md](DATASET_AUDIT.md)。

### 3.2 时间与质量

| 项目 | metrics | logs | traces |
|---|---:|---:|---:|
| 可解析时间行 | 182,820,535 | 87,974,577 | 28,681,438 |
| 无效时间行 | 0 | 294 | 0 |
| 全局时间起点（Asia/Shanghai） | 2021-07-01 18:00:01 | 2021-07-01 09:57:04.258 | 2021-07-01 09:57:04.255 |
| 全局时间终点（Asia/Shanghai） | 2021-07-31 23:59:57 | 2021-07-31 23:59:59.920 | 2021-07-31 23:59:59.904 |
| 正时间差中位数 | 30 s | 59 ms | 508 ms |
| source-order 零时间差占比 | 11.84% | 18.98% | 0.013% |
| source-order 负时间差占比 | 0.048% | 0.0046% | 0.816% |

GAIA business log 的 `datetime` 列 87,974,871 行均不含时分秒；294 行 message
为空或无法提供前缀时间。所有有效 log 时间均来自 `message` 的毫秒前缀。
source-order 的负时间差说明原文件并非严格全局排序，不代表时间解析失败；切片
实现不得依赖输入行顺序。

候选 metric 文件没有空 value cell 或非数值 value。按每个文件的 lower-median
正时间差推断采样网格后：

- 5,714 个文件可形成规则网格；名义间隔中位数 30 s、P95 60 s；
- 预期 timestamp 192,166,727，缺口 31,098,659，推断缺口率 16.18%；
- 重复 timestamp 行 21,752,467；
- 这些是时间网格质量统计，不能与 value-cell missingness 混为一谈。

### 3.3 Case 窗口活动覆盖

下表给出 case 的平均 30 s bin 占用率，以及候选服务活动覆盖均值：

| 窗口 | metrics bins / services | logs bins / services | traces bins / services |
|---|---:|---:|---:|
| ±30 s | 88.35% / 84.43% | 99.97% / 96.48% | 99.97% / 92.81% |
| ±60 s | 88.35% / 84.48% | 99.97% / 96.48% | 99.97% / 92.81% |
| ±300 s | 88.33% / 84.66% | 99.97% / 96.50% | 99.97% / 92.81% |
| ±600 s | 88.32% / 84.78% | 99.96% / 96.53% | 99.96% / 92.83% |
| ±1800 s | 88.28% / 85.24% | 99.92% / 96.63% | 99.92% / 92.92% |

±300 s 内 14,358/16,200 个 case 有任一候选 metric 活动；logs/traces 均为
16,196/16,200。扩大窗口不能实质修复 metric 的内部时间缺口：±300 s 无 metric
活动的 case 中，仅 82 个完全落在全局 metric 起点之前，其余主要对应原始时序
内部无候选 metric 活动。主协议若保留这些 case，必须显式携带 modality/activity
mask；Metric-change baseline 对无观测服务需要预注册的确定性 fallback。

### 3.4 Topology

GAIA 没有已登记的显式静态 service graph。10/10 trace 文件都具有完整的
`trace_id/span_id/parent_id/service_name` 字段，因此可按 case 窗口构建动态有向图；
本轮只确认可构建性，不物化或验证图连通性。

## 4. RE2-OB 诊断

### 4.1 Case、候选与相对时间

- 官方布局中的 90/90 case 全部读取，排除 0；每 case 11 个应用服务；
- 90/90 case 的三种模态都覆盖 anchor `t0`；
- metrics 的起点全部为 `t0-720 s`，终点中位数为 `t0+720 s`；
- logs 起点中位数 `t0-719.994 s`，终点中位数 `t0+719.984 s`；
- traces 起点中位数 `t0-719.997 s`，终点中位数 `t0+719.990 s`；
- 少数 case 提前结束：最短 metrics/logs/traces 终点分别为 +209.0、+172.483、+38.561 s。

### 4.2 质量与覆盖

- metrics：1 s 规则采样；128,742 个有效 timestamp 无网格缺口或重复；另有
  514 行无效 timestamp；55,227,875 个 value cells 中 353,885 个缺失，missingness 0.6408%；
- logs：15,053,223 行时间全部有效；必填 `container_name/timestamp` 无缺失；
- traces：34,461,235 行时间全部有效；`traceID/spanID/serviceName/startTime` 无缺失；
  2,112,216 行 `parentSpanID` 为空，占 6.13%，应在构图时区分合法 root span 与异常缺失；
- ±300 s 平均 30 s bin 占用率：metrics 99.84%、logs 99.74%、traces 99.47%；
- ±600 s 对应 99.38%、99.05%、99.30%；±1800 s 超出原始约 ±720 s 采集范围，
  三模态占用率约 40%，不适合作为主窗口。

RE2-OB 的 `cluster_info.json` 是日志模板元数据，不是服务调用拓扑；90/90 case
存在可读 pod-node 部署映射，11 个候选服务的部署覆盖率为 100%，但这仍不是
service call graph。90/90 trace 文件允许按 span parent 关系构建动态图。

## 5. 上下文窗口与 split 可行性

为避免两个 case 共享同一段 telemetry 却跨 split，GAIA 的分组必须基于
“实际注入区间 ∪ 候选上下文窗口”的连通分量，而不能只使用原始注入区间。

| 对称窗口 | 连通组 | 非单例组 | 非单例组中的 cases | 最大组 |
|---:|---:|---:|---:|---:|
| ±30 s | 9,375 | 3,155 | 9,980 | 39 |
| ±60 s | 6,239 | 3,523 | 13,484 | 39 |
| ±300 s | 322 | 306 | 16,184 | 471 |
| ±600 s | 23 | 18 | 16,195 | 3,206 |
| ±1800 s | 5 | 5 | 16,200 | 8,783 |

当前证据据此冻结 **`T_pre=T_post=300 s` 作为主窗口**：它在 GAIA 的 30 s
metric 粒度下每侧约有 10 个预期采样点；RE2-OB 三模态平均活动覆盖均高于
99.47%；同时保留 322 个可分组单元，最大组只占 2.91%。±600 s 的覆盖收益很小，
却把分组单元降至 23、最大组升至 19.79%，不利于稳定的 grouped evaluation。

半开区间与 context-aware `groups.jsonl` 已写入 manifest；切片器、无观测
mask/fallback、实际 split assignment 与 GAIA inclusion 已由后续实验完成，
因此当前 G1/G6/G7 均 completed。

## 6. Gate 结论与限制

G8 判为 **completed**。证据包括本报告、全量 JSON、确定性 manifests 和现有
adapter/evaluator 测试。G8 完成不代表 P1 完成；当前仍有以下限制：

1. RCAEval 的发布 tag/commit 与完整原始内容 checksum 尚未固定；
2. 全量 JSON 使用 layout digest 与关键 annotation SHA-256，不是 telemetry 内容哈希；
3. 动态 topology 尚未物化；
4. 主窗口与 context-aware groups 已冻结，实际 5-fold assignment 见
   [SPLIT_DIAGNOSTICS.md](SPLIT_DIAGNOSTICS.md)；
5. 后续已运行 Random、Frequency、Metric-change sanity baselines，见
   [BASELINE_RESULTS.md](BASELINE_RESULTS.md)。
