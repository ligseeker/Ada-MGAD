# Nezha — OnlineBoutique 数据集深度分析报告

> 分析对象：Nezha 仓库（`/home/zhangll24/RCA_project/Nezha`）中的 **OnlineBoutique** 子集
> 预处理产物：`/home/zhangll24/RCA_project/Ada-MGAD-exp/data/Nezha-pre/`
> 预处理/加载代码：`util/Nezha/pre_Nezha.py`、`util/Nezha/data_Nezha.py`、`util/Nezha/constant.py`
> 分析日期：2026-08-05

---

## 1. 概述

Nezha 原始仓库包含**两个微服务应用**：

| 应用 | 原始目录日期 | 服务数 | 是否在当前预处理中 |
|---|---|---|---|
| **OnlineBoutique**（hipster 前缀） | 2022-08-22, 2022-08-23 | **10** | ✅ 已处理（`NEZHA_DAYS=['2022-08-22','2022-08-23']`） |
| TrainTicket（ts- 前缀） | 2023-01-29, 2023-01-30 | ~27 | ❌ 未处理（缺 `dependency.csv`，`NEZHA_DAYS` 未包含） |

本报告只分析 **OnlineBoutique** 这一子集。它是目前 `data/Nezha-pre/` 中实际存在、且经核验三模态齐全的数据集。

Nezha 为每个数据集同时提供两类数据：
- **`rca_data/<day>/`**：故障期（fault-suffering），含 `*-fault_list.json` 故障注入标签。
- **`construct_data/<day>/`**：无故障期（fault-free）基线数据。

预处理（`day_window_files`）把**同一天的 `rca_data` 与 `construct_data` 合并**处理，因此最终时间轴同时包含故障段与无故障段（这与 Nezha 原论文"用无故障期模式对比故障期"的思路一致）。

---

## 2. 原始数据结构（OnlineBoutique）

### 2.1 节点与服务
10 个微服务（即图的节点）：

```
adservice, cartservice, checkoutservice, currencyservice, emailservice,
frontend, paymentservice, productcatalogservice, recommendationservice, shippingservice
```

### 2.2 故障标签（`rca_data/<day>/<day>-fault_list.json`）
- 按小时分组的字典：`{ "03": [fault, ...], "04": [...], ... }`。
- 单条故障字段：`inject_time`（字符串）、`inject_timestamp`（epoch 秒字符串）、`inject_pod`（如 `frontend-579b9bff58-t2dbm`）、`inject_type`（故障类型）。
- **两天故障统计**：

| 日期 | 故障总数 | 类型分布 |
|---|---|---|
| 2022-08-22 | 24 | cpu_contention×8, network_delay×7, cpu_consumed×3, return×3, exception×3 |
| 2022-08-23 | 32 | network_delay×9, cpu_consumed×7, cpu_contention×8, return×4, exception×4 |
| **合计** | **56** | 覆盖全部 10 个服务 |

- 注入的服务覆盖全部 10 个节点（无"永不注入"的节点）。

### 2.3 调用图（`rca_data/<day>/metric/dependency.csv`）
- 格式：`Source,Target,Weight`。
- 注意：文件中混入了 **IP 地址（如 `33.33.33.x`）和 `loadgenerator`** 这类非服务实体。
- 预处理 `load_adjacency()` 用 `NEZHA_SERVICE2NID` 过滤，自动丢弃这些噪声边，最终得到 **14 条有效有向边**（见 §4.3）。

### 2.4 各模态原始文件 schema（两应用完全一致）
- **Metric**：`<pod>_metric.csv`，列含 `Time, TimeStamp, PodName` + 22 个 KPI（其中 `NEZHA_KPIS` 取 8 个核心指标）。
- **Log**：`<HH>_log.csv`，列 `Timestamp, TimeUnixNano, Node, PodName, Container, TraceID, SpanID, Log`；`Log` 字段是 JSON 字符串，内嵌 `message` / `severity` / `timestamp`。
- **Trace**：`<HH>_trace.csv`，列 `TraceID, SpanID, ParentID, PodName, OperationName, StartTimeUnixNano, EndTimeUnixNano, Duration`。
- **TraceID 映射**：`<HH>_traceid.csv`（用于关联 trace 与更上层调用）。

### 2.5 时间戳核对（关键正确性点）
- **Metric `TimeStamp` 列损坏**：实测 `TimeStamp` 比 `Time` 列大 **200,000,000 秒（~6.3 年）**。例如 `adservice` 首行 `TimeStamp=1861140279` 而 `Time` 解析为 `1661140279`。
- 预处理 `repair_timestamps()` 检测到该偏移（offset 阈值 >30s），**回退使用 `Time` 列**重建时间戳。预处理日志中有明确 warning：`pod adservice ... TimeStamp corrupted (offset 200000000, residual 22100 s), rebuilt from Time column`。✅ 处理正确。
- **Trace `EndTimeUnixNano`**：纳秒级 epoch，与 metric 时间轴同源，无需跨时钟估计（这是 Nezha 比 SN/TT 稳的关键）。

---

## 3. 预处理流水线（per 模态）

时间分辨率：`BUCKET_SEC = 60`（逐分钟）；`FAULT_DURATION_SEC = 180`（故障标签持续 3 分钟，与论文一致；作为回退默认值，实际优先读 `fault_list.json` 的 `duration` 字段）；`robust_zscore` 归一化。

### 3.1 Metric
1. `load_day_metric`：每 pod 读 `<pod>_metric.csv`，`repair_timestamps` 修时间戳，按 `pod_to_service` 映射到服务。
2. 按 `BUCKET_SEC` 聚合到分钟桶（`groupby(bucket).mean()`），`reindex(range(T)).ffill().fillna(0)`。
3. `robust_zscore`：中位数/MAD 归一化（公式 `(x-median)/(1.4826*MAD)`；MAD≈0 的常量特征置 0）。
4. 输出 `metric.csv`：`now` 列（0..T 连续索引）+ 每服务 8 列 = 80 特征列。

**核验**：随机抽 `adservice` 的 `CpuUsageRate(%)`，重算 robust-z 得 `2.066`，与 `metric.csv` 中保存值完全一致。✅ 归一化正确。

### 3.2 Log
1. `extract_log_message`：从 `Log` JSON 抽出 `message` 字段。
2. Drain3 模板挖掘（`nezha.ini`），得到 `cluster_id`。
3. 桶对齐：`bucket = (ts_ns//1e9 - t0) // BUCKET_SEC`。
4. 输出 `log.csv`：`templateid, Hostname, @timestamp`（仅存 id，**不存模板文本**）。
5. `data_Nezha._load_logs`：加载时堆成 `(T, node, log_len)` 计数矩阵，再做 min-max 归一化。

**说明**：日志表达力弱于 GAIA（GAIA 还有日志级别维度 + log_total），这里只是"模板词袋计数"。`log_len` 自动取最大 templateid = **84**。

### 3.3 Trace
1. `load_day_traces`：解析 span 的 `ParentID → SpanID` 映射出有向边 `(src_service, dst_service, OperationName)`，按 `EndTimeUnixNano` 桶对齐。
2. 输出 `trace.csv`：`cmbd_id, fatherpod, stats, end_time, duration`。
3. `data_Nezha._load_traces`：堆成 `(T, src, dst, num_stats)` 张量，对 duration 做 `log1p` 并对每段做 `/(mean*10)` 归一化。

**说明**：`stats`（即 `OperationName`，如 `hipstershop.ProductCatalogService/GetProduct`）经 `stats_vocab.pkl` 编码，共 **14** 种。

### 3.4 Label
1. `load_day_labels`：读 `*-fault_list.json`，对每条故障 `lo=(inject_timestamp - t0)//BUCKET_SEC`，`hi=(inject_timestamp + duration - 1 - t0)//BUCKET_SEC`，在 `[lo, hi]` 桶把对应服务节点置 1。
2. 输出 `label.pkl`：`(T_total, num_nodes)` 的 0/1 矩阵。
3. 加载时转成 `label`(one-hot 2) 与 `mask`(one-hot 3，全标注第 0 类恒为 0)。

> **✅ 故障持续时长（已写入文件）**：`fault_list.json` **现已包含 `duration` 字段**（每条故障 `duration=180`，单位秒，对应论文的 3 分钟）。标签区间的"结束时刻"优先由文件字段推出，无字段时回退到常量：
> - `util/Nezha/constant.py` 的 **`FAULT_DURATION_SEC = 180`**（3 分钟，与论文一致）作为回退默认值。
> - `pre_Nezha.py:load_day_labels` 用 `duration = int(fault.get('duration', FAULT_DURATION_SEC))`，再算 `hi = (start + duration - 1 - t0)//BUCKET_SEC` 生成标签窗口 → 每条 180s 故障约占 3~4 个分钟桶（因注入时刻与分钟桶边界不齐而向上取整，如 `buckets [2,3,4,5]`）。
> - 该 180s **来自论文（FSE'23 §3）原文**："We set each fault duration to 3 minutes to emulate the process between fault occurrence to fix." 非来自 Nezha 原仓库（其 `pattern_ranker.py` 不维护 duration，只用 `inject_time` + 2 min 作为分析点）。
> - 已在 `Nezha/rca_data/2022-08-2*/<day>-fault_list.json` 写入 56 条 `duration=180`，并重建 `data/Nezha-pre/label.pkl`（shape `(1930,10)`，56/56 故障全部正确对齐到 10 个节点）。原文件备份为 `*-fault_list.json.bak`。

---

## 4. 预处理产物与正确性核验（实测）

### 4.1 产物清单（`data/Nezha-pre/`）

| 文件 | 说明 |
|---|---|
| `metric.csv` | 1930 行 × 81 列（含 `now`），80 个 metric 特征 |
| `log.csv` | 3,912,853 行，84 个模板 |
| `trace.csv` | 868,424 行（有向边） |
| `label.pkl` | `(1930, 10)` 0/1 标签矩阵 |
| `trace_path.pkl` | `(10,10)` 邻接矩阵，14 条边 |
| `stats_vocab.pkl` | 14 个 trace operation 类型 |
| `bounds.pkl` | 2 天拼接边界：day0 `[0,1209)`、day1 `[1209,1930)`，均 `has_fault=True` |

### 4.2 时间轴拼接
- `T_total = 1930` 分钟 ≈ 32.2 小时。
- day0（2022-08-22）：1209 分钟；day1（2022-08-23）：721 分钟。
- 故障期与无故障期（construct_data）已合并进同一天时间轴。

### 4.3 邻接矩阵（14 条边）
```
checkoutservice  <-> cartservice, currencyservice, emailservice, frontend, paymentservice, productcatalogservice, shippingservice  (deg 7)
frontend         <-> adservice, cartservice, checkoutservice, currencyservice, productcatalogservice, recommendationservice, shippingservice (deg 7)
productcatalogservice <-> checkoutservice, frontend, recommendationservice (deg 3)
cartservice/currencyservice/emailservice/paymentservice/shippingservice/recommendationservice (deg 1~2)
```
无孤立节点，调用图连通。IP/`loadgenerator` 噪声边已剔除。✅

### 4.4 标签对齐核验（已全部通过）
逐条把 56 个原始故障映射到 `label.pkl`，**56/56 故障在"注入服务的对应桶"上标签=1，0 遗漏、0 错标**。✅
（首条 `frontend cpu_contention @1661140434` → bucket `[2,3,4]`，与预处理日志一致。）

### 4.5 每节点三模态齐全性（核心结论）
实测每个节点在 T=1930 上的覆盖率（"有数据的时间步占比"）：

| 节点 | Metric | Log% | Trace% | 标签# |
|---|---|---|---|---|
| adservice | ✅ | 11.4 | 11.3 | 21 |
| cartservice | ✅ | 11.7 | 11.6 | 12 |
| checkoutservice | ✅ | 9.7 | 9.7 | 24 |
| currencyservice | ✅ | 12.4 | 12.4 | 12 |
| emailservice | ✅ | 9.1 | 9.1 | 15 |
| frontend | ✅ | 12.7 | 12.6 | 24 |
| paymentservice | ✅ | 9.3 | 9.3 | 12 |
| productcatalogservice | ✅ | 12.4 | 12.3 | 14 |
| recommendationservice | ✅ | 11.4 | 11.4 | 18 |
| shippingservice | ✅ | 10.7 | 10.7 | 15 |

- **Metric**：10/10 节点齐全，无 NaN，全零时间步 = 0。✅
- **Log**：10/10 节点都有日志活动，无"永远无日志"节点。✅
- **Trace**：10/10 节点都参与调用（作为 src 或 dst），无孤立/无活动节点。✅
- **结论：OnlineBoutique 是三个新数据集中唯一"每节点三模态全齐全"的数据集**，契合 Ada-MGAD 多模态图方法"每节点数据齐全效果好"的前提。

> 注：Log/Trace 的"有数据时间步占比"仅 ~10–13%，这是微服务观测数据的固有稀疏性（并非缺失），属正常。Metric 则是逐分钟连续。

---

## 5. 训练/切分行为（实测）

用 `util/Nezha/data_Nezha.Process` 实测（`window=10, step=1, test_experiment=0`）：
- 总窗口数：**1912**。
- 切分：`_build_split` 按 `exp_id`（天）做 leave-one-day-out，`test_experiment` 指定的那天作为测试集。
- 上例：`test_experiment=0` → 测试集 = day0 = **1200** 窗口，训练集 = day1 = **712** 窗口。
  - ⚠️ **注意**：day0 时间更长，导致"被留出测试集反而比训练集大"（1200 > 712）。如需训练集更大，应设 `test_experiment=1`（留出 day1）。
- 图边：14。`data_edge` 形状 `(window, 10, 10, raw_edge=14)`，`data_log` 形状 `(window, 10, log_len=256)`（256 为 log_len 上限，含 84 模板 + padding）。

---

## 6. 已知问题与注意事项

1. **未包含 TrainTicket**：当前 `NEZHA_DAYS` 仅 OnlineBoutique 两天。`data/Nezha-pre/` 不含 2023-01 的 TrainTicket（~27 节点）。若需更大规模，须扩展（且 TrainTicket 缺 `dependency.csv`，需改 `load_adjacency` 从 trace 推断调用图）。
6. ~~**故障标签时长与论文不一致（建议修正）**~~ **（已修正）**：原 `fault_list.json` 无 `duration` 字段、`FAULT_DURATION_SEC=120` 与论文 180s 不一致。现已采用 (b)+(a) 组合方案落地：给 `Nezha/rca_data/2022-08-2*/<day>-fault_list.json` 每条故障写入 `duration=180`，`pre_Nezha.py` 优先读文件字段（无则回退常量），并把 `FAULT_DURATION_SEC` 改为 `180`（出处：论文 FSE'23 §3 "each fault duration to 3 minutes"），已重建 `label.pkl`（见 §3.4）。原文件备份为 `*-fault_list.json.bak`。
2. **故障期与无故障期合并**：最终时间轴同时含 `rca_data` 与 `construct_data`。这符合 Nezha 原方法，但若你的任务假设"全为故障数据"需留意。
3. **日志表达力弱**：仅有模板词袋计数，无日志级别/总数维度；且**未落盘模板文本文件**（无法人工核验模板语义，重跑时 templateid 可能漂移）。
4. **切分不平衡**：leave-one-day-out 下测试集可能大于训练集（取决于 `test_experiment`）。
5. **Metric 含常量化特征**：17/80 个 (节点×特征) 的 MAD≈0 → 归一化后恒为 0（robust-z 的 `flat` 分支）。这是正常行为，但意味着部分特征通道无信息量。

---

## 7. 使用建议

- **作为主实验/对比数据集**：OnlineBoutique 三模态齐全、trace 对齐稳（无跨时钟估计）、无损坏实验，**是三个新数据集里最干净、最推荐的一个**。
- **推荐配置**：`test_experiment=1`（留出较短的 day1 作测试，训练集更大）；`window`/`step` 按模型需要设。
- **如需补强**：
  - 给 `pre_Nezha.py` 增加 `save_template`/`save_state`（照 MSDS 实现），保证日志模板可复现。
  - 若想扩到 TrainTicket：在 `constant.py` 增一套 27 节点配置，`load_adjacency` 增加"无 `dependency.csv` 时由 trace 推断邻接"的兜底，并把 `NEZHA_DAYS` 扩为 4 天（或拆成独立 dataset 类）。

---

## 附：关键文件索引

| 用途 | 路径 |
|---|---|
| 原始数据（故障期） | `/home/zhangll24/RCA_project/Nezha/rca_data/2022-08-22` , `.../2022-08-23` |
| 原始数据（无故障期） | `/home/zhangll24/RCA_project/Nezha/construct_data/2022-08-22` , `.../2022-08-23` |
| 预处理脚本 | `util/Nezha/pre_Nezha.py` |
| 数据加载/滑窗 | `util/Nezha/data_Nezha.py` |
| 常量（服务/天数/KPI） | `util/Nezha/constant.py` |
| 预处理产物 | `data/Nezha-pre/` |
| 预处理日志 | `data/pre_nezha.log` |

---

## 附：故障持续时长（FAULT_DURATION_SEC）溯源

故障注入标签参考文件为 `rca_data/<day>/<day>-fault_list.json`（OnlineBoutique 共 2 个：2022-08-22、2022-08-23）。单条故障字段：

```json
{ "inject_time": "2022-08-22 03:53:54",
  "inject_timestamp": "1661140434",
  "inject_pod": "frontend-579b9bff58-t2dbm",
  "inject_type": "cpu_contention",
  "duration": 180 }
```

**文件现含 `duration` 字段**（每条 `duration=180`，单位秒），由本修正补充。标签区间的"结束时刻"优先由文件字段推算，无字段时回退常量。`duration` 取值的三个来源：

| 来源 | 故障时长取值 | 证据 |
|---|---|---|
| Nezha 原仓库代码 | **无 duration 概念** | `pattern_ranker.py:234` 只用 `inject_time`，算 `abnormal_time = inject_time + 2 min` 作分析点；全仓无 `duration` / `120` 常量 |
| Nezha 论文（FSE'23 §3） | **3 分钟（180s）** | 原文："We set each fault duration to 3 minutes to emulate the process between fault occurrence to fix." |
| 本项目 `util/Nezha/constant.py` | **3 分钟（180s）** | 已改为 `FAULT_DURATION_SEC = 180` 并标注出处（论文 FSE'23 §3）；同时 `fault_list.json` 写入 `duration=180`，`pre_Nezha.py` 优先读取 |

**结论**：`duration=180` 已写入 `Nezha/rca_data/2022-08-2*/<day>-fault_list.json`（56 条故障），`FAULT_DURATION_SEC` 已对齐论文为 180，`label.pkl` 已重建（按 180s 生成，比旧版 120s 多覆盖第 3 分钟桶，不再漏标）。原文件备份为 `*-fault_list.json.bak`。
