# GAIA 多模态预处理审计 V1

## 结论摘要

本审计的结论不是“尽量多保留特征”，也不是继续使用“十节点公共交集”。对当前 Ada-MGAD，合理目标是：在固定、全节点同构的 slot 空间内，用 Train-only 的语义、可用性、动态性和冗余审计，控制每种模态的有效维数。

本轮建议的目标表示是：

| 模态 | 建议 | 目标 raw 维数 | 本轮状态 |
|---|---|---:|---|
| Metric | `global container + compact host-of-node + scope/observability`；当前数据没有已证实的 candidate service-type 专属指标 | 约 48–64 | 候选方案；须经 P0 Train-only 全量统计冻结 |
| Log | 全局 Train-only Drain3；stable template + level-aware RARE + level-aware UNK + level counts；不把 `log_total` 作为模型输入 | 约 32–64 | 候选方案；须经 P0 完整 Train vocabulary 审计冻结 |
| Trace | 每个有向边、每个 status 保留 `log1p(count)` 与条件 `mean_latency`；删除 `duration_sum` | 8 | 推荐冻结候选 |

三个最高优先级问题是：

1. **MUST FIX：** Metric 当前公共交集把全部 384 个可映射 host logical features 排除在外，同时保留大量同义/累计型 container 指标；统一维度不等于原始指标必须处处相同。
2. **MUST FIX：** Log 每个 Drain3 cluster 都占一维，且 reconstruction 对 raw slots 直接求和；template 数一旦达到数百，Log 会获得隐式模态权重。
3. **MUST FIX：** Trace 的 `duration_sum=count×mean_latency` 混合流量与时延；parent lookup 又用整月、仅 `span_id` 的索引，Train graph 的 source resolution 尚未严格 split-local/trace-local。

本轮没有执行正式全量 preprocessing、训练、Test 评估或 RCA 训练。本文中的方案不是正式实验结果，P0 尚未完成的统计不得由 Test F1 或标签驱动补选。

## 0. 审计边界、材料护照与证据标记

- Worktree：`/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2`
- 审计 HEAD：`f405fce1107a9ed0ed3ced83de53c953f7bd6ff3`
- V3 config SHA-256：`121bdaddc7d8d70087a5ecd41ebc2070a3534d52ec13b76f9945dc916df90449`
- 当前 run table SHA-256：`f6752f1bb46b63ebff05639093477963735370c6d70a1588ddebb304d952cb0b`
- Train：`[1625133600000, 1626963120000)`，即 Asia/Shanghai `2021-07-01 18:00:00` 至 `2021-07-22 22:12:00`，共 60,984 个 30 s bins。
- Test：`[1626963120000, 1627747200000)`；Test 仅允许在 schema 冻结后 transform。

证据标记：

- **【已验证】**：本轮直接读取当前 HEAD、当前文件名/metadata 或当前 artifact 得到。
- **【已验证（审计实现）】**：本轮执行了 Train-only 流式审计，但其实现与 production reader 尚有已声明差异；可支持方向判断，进入 schema freeze 前须用正式 audit-only 脚本复核。
- **【抽样结果】**：有确定性采样方法，但不是完整总体统计。
- **【历史全量证据】**：仓库已有全量扫描；本轮验证了路径、文件数、总字节数和 layout digest，但没有重新计算所有原始文件的逐文件内容 hash。它可用于风险判断，不能替代当前 P0 的 Train-only schema freeze。
- **【待全量统计】**：本轮不伪造数值；对应项目是进入 P1 freeze 的 hard gate。
- **【建议】**：由方法约束和数据事实推出的方案，不是测得的性能增益。

当前 raw layout 已验证为：

| 模态 | 文件数 | 字节数 | size-layout SHA-256 |
|---|---:|---:|---|
| Metric | 6,640 | 4,790,077,103 | `38cabf0c6582a900a5ae80489eb0af392dedc00b4383ab7c4fabe1463eefd556` |
| Log | 10 | 18,445,461,523 | `ee10c7afacf40bb9eddb00a3cdfa2d99ce73d531654066b27595e579663c409b` |
| Trace | 10 | 8,343,966,340 | `a12ae950f613c8f119742e7d1d149b56407cd76437020c4d31736e356e1fd08d` |

注意：较早的 G0R2 artifact 绑定的是另一个 run-table hash；Telemetry 分布事实与 GT/schema 决策分开使用，绝不把旧 GT provenance 当作当前 V3 GT provenance。

## 1. Ada-MGAD 输入约束

### 1.1 实际 shape

`TimestampedArrayDataset` 的单个窗口输入是：

```text
Metric  data_node: [B, W, N, R_node]
Log     data_log:  [B, W, N, R_log]
Trace   data_edge: [B, W, N, N, R_edge]
```

其中当前 GAIA `N=10`、`W=10`。来源为 `src/e2e/ad_data.py:112-147`。

三个共享线性层分别执行：

```text
R_node -> feature_node(16)
R_log  -> feature_log(8)
R_edge -> feature_edge(4)
```

见 `src/model.py:34-46` 与 `src/model_util.py:506-532`。因此一个 batch 内、所有时间、所有节点的 `R_node/R_log` 必须一致；所有边的 `R_edge` 必须一致。Log 的线性层同样在所有节点共享，故 vocabulary 必须全节点共享。

### 1.2 “统一输入维度”不等于“原始指标处处存在”

两者不是同一要求：

- **结构要求：** 所有节点得到相同长度、相同 index 语义的向量。
- **数据事实：** 某个节点是否真实观测到该指标。

模型只要求前者。可以定义统一 slot `host_of_node.cpu_iowait`：有 host telemetry 的节点写入实值，无对应 telemetry 的节点写入 Train-neutral value，并由 `host_applicable/host_observed_fraction` 区分。不能做的是让 index 17 在 DB 节点表示 DB connections、在 Redis 节点表示 key count。

原始 GAIA loader 仅按列名排序后对每个 service 取前 `raw_node` 列（`util/GAIA/data_GAIA.py:95-115`），它保证长度，不保证 slot 语义一致；因此不能把原始实现当作语义正确性的依据。

### 1.3 embedding、attention、fusion 和 reconstruction

1. Metric、Log、Trace 先独立线性 embedding 加位置编码。
2. Spatial attention 将 Metric 与 Log embedding 拼接为 node state，以 Trace embedding 作为 GAT edge attribute（`src/model_util.py:356-382`）。
3. Temporal attention 分别处理三模态；随后 gated cross-modal fusion 使用各模态的全局平均 attention 表示互相调制（`src/model_util.py:257-353`）。
4. Decoder 经过 masked temporal attention 和 encoder-decoder cross attention，投影回三个 raw spaces（`src/model_util.py:460-502`）。
5. 重构误差在每个 raw slot 上取平方，Trace edge residual 再分配到两个端点，最后拼接 Metric、Log、Trace residual（`src/model.py:134-142`）。
6. 评估时 `rec_score=sum(rec, dim=-1)`；训练 reconstruction 分支也先对所有 raw slots 求和（`src/model.py:144-176`，`util/train.py:202-215`）。

所以：

```text
实际隐式模态权重 ~= raw slot 数 × 每 slot 残差尺度 × 可激活频率
```

固定 embedding dimension、gated fusion 和三模态等权 contrastive loss都不会自动消除该偏置。分类头也直接读取长度为 `R_node+R_log+R_edge` 的 residual vector（`src/model.py:48-50`）。因此预处理必须同时控制尺度、冗余和维数。

Trace 还有一个结构效应：节点级 Trace score 随 incident edge 数增加；每条 edge 被两个端点各分配一半，但高 degree 节点仍有更多误差来源。

## 2. Current V3：当前到底做了什么

### 2.1 Metric

当前 effective V3 流程（`src/e2e/ad_preprocess.py:101-313`）：

1. 从 filename 解析 service、IP、feature 和日期范围；只纳入 filename 时间范围与 Train 相交的文件。
2. `docker_cpu_core_<id>_*` 名称折叠为 `docker_cpu_core_X_*`。
3. 取 10 个 candidate services 的 logical feature 严格交集。
4. 对每个 `service×feature` 的 Train aligned series 计算 coverage、unique、q05-q95 span、dynamic ratio。
5. 仅当该 feature 在 10 个节点都通过 `coverage>=0.20, unique>=2, span>=1e-10, dynamic_ratio>=1e-4` 时保留。
6. Train/Test 分块执行 `ffill(limit=10)`，即最多跨 300 s，但不考虑该 feature 的实际采样周期。
7. 每个 `service×feature` 用自己的 Train min/max 缩放；剩余 NaN/Inf 全变为 0。

### 2.2 Log

文件中前部还有旧的 level-only `build_log_arrays`，但 Python 名称被后部 V3 定义覆盖；正式入口实际使用 `src/e2e/ad_preprocess.py:899-986`：

```text
全体服务的 Train message payload
-> 一个全局 Drain3（canonical service/file/row order）
-> freeze state 和 cluster IDs
-> Train/Test 仅 match
-> unseen -> template_UNK
-> 每个30s节点bin统计 template counts + 5 level counts + log_total
-> 每列除以跨所有Train节点/bin的最大值
```

Drain3 只消费 message 的 payload；level 从完整 message 另行解析（`src/e2e/log_templates.py:19-41`）。V3 的 frozen transform 不创建 Test cluster（同文件 `:119-185`）。当前没有 rare bucket，也没有 template 上限；config 中的 `log_features` 列表并不约束 cluster 数。

### 2.3 Trace

当前 Tensor 语义是：

```text
[time, source_service, destination_service, status]
status = 200/300/400/500
time = child span end_time floor到30s
value = 同bin、同有向边、同status的 duration_sum
```

`duration=(end_time-start_time).total_seconds()`；parent service 由 child 的 `parent_id` 在各 service 的 `span_id` 索引中查找（`src/e2e/ad_preprocess.py:580-720`）。缩放为 `duration_sum/(10×Train mean+1e-6)`。

静态 graph 只把 child bin 落在 Train 的成功解析边加入，然后对称化；raw Trace tensor 保持有向。需要明确区分“有向观测”与“无向模型 adjacency”。

## 3. GAIA 数据事实

### 3.1 Metric 来源与 logical schema

**【已验证】** 6,640 个 raw CSV 不是 6,640 个模型 features：

| 层次 | 数量 | 解释 |
|---|---:|---|
| raw files | 6,640 | 大多数指标各有 `07-01..07-15` 与 `07-15..07-31` 两个 shard |
| unique `service_ip_feature` | 3,320 | 合并两个日期 shard 后的物理 series |
| candidate `service×logical_feature` | 3,972 | multi-core collapse 且 system→host service 映射后 |
| 10-node 公共 logical names | 90 | 全部为 candidate container 的 Docker metrics |
| 可映射 host logical names | 384 | system metrics；只覆盖有 system 文件的 3 台 host |

按 raw filename prefix 计数：

| 来源 | raw files |
|---|---:|
| dbservice1 / dbservice2 | 318 / 366 |
| logservice1 / logservice2 | 366 / 366 |
| mobservice1 / mobservice2 | 318 / 318 |
| redisservice1 / redisservice2 | 318 / 366 |
| webservice1 / webservice2 | 318 / 366 |
| system | 2,304 |
| unmapped `redis` / `zookeeper` | 544 / 372 |

文件名时间 shard 分为 `2021-07-01..07-15` 与 `2021-07-15..07-31`，各 3,320 个；它们都与 Train 至少部分相交，但并不意味着每条 series 在 Train 内连续可用。**【历史全量证据】** Metric 最早有效采样约为 07-01 18:00，而 Log/Trace 最早记录约为 07-01 09:57；选择统一 Train 起点 18:00 可避免在 Metric 尚未开始时制造整段伪缺失。

十个 candidate-named container series 在 multi-core collapse 后都具有相同的 90 个 Docker logical names。当前 filename/mapping 证据没有显示 DB/Web/Mob/Log/Redis 五类各自额外的 candidate-owned exporter features。

按问题要求拆开计数：

| 口径 | DB | Log-service | Mob | Redis-service | Web |
|---|---:|---:|---:|---:|---:|
| 两实例共有的 candidate-owned logical features | 90 | 90 | 90 | 90 | 90 |
| 加上当前可映射 host names 后的两实例交集 | 474 | 90 | 474 | 474 | 90 |

所有十节点共有的仍是 90；已证实的“仅单个 candidate service 才有”的 container logical feature 数为 0。第二行的 384 个额外 names 是 host scope，不是 DB/Mob/Redis type-specific 指标；Log/Web 各有一个实例位于无 system telemetry 的 `.3`，所以 pair intersection 回到 90。

额外 raw namespaces 是：

- `redis@0.0.0.3`：272 个 unique raw feature names，其中包含 `redis_info_*` 与 `redis_keyspace_*`；
- `zookeeper@0.0.0.1`：186 个 unique raw feature names，其中包含 `zookeeper_server_*`；
- 当前 candidate registry 和 `_target_services_for_metric()` 不把它们映射到 10 个节点。

因此它们不能被悄悄称为“Redis-specific candidate metrics”。`redis` 究竟是基础设施节点、第三个实例还是应归属某个 candidate，必须由外部部署拓扑证据确认；在此之前保持隔离。

Host mapping 的直接事实：system 文件只存在于 `0.0.0.1/.2/.4`；`logservice1` 和 `webservice2` 位于 `.3`，没有 system telemetry。其余 8 个 candidate services 分别继承其 host 的 384 个 logical slots。

### 3.2 严格交集删除了什么

**【已验证】** 严格交集从 474 个可映射 logical names 的 union 中保留 90、删除 384；按实际存在的 node-feature pairs 计，从 3,972 保留 900、删除 3,072（77.34%）。被删的 3,072 对全部是 host-derived，不是五类 candidate service-specific features。

这说明当前策略确实过严，但不是因为五种服务类型都有大量专属指标，而是因为它把“不在所有节点可用”误当成“不能进入统一 schema”。

### 3.3 Metric 质量、采样与缺口

**【历史全量证据】** 绑定相同 telemetry size-layout 的既有扫描记录：

- 182,820,535 个 metric rows；
- positive timestamp delta 中位数 30 s、p95 60 s；
- 以每文件 lower-median delta 作为 nominal interval 时，推断 gap ratio 为 16.18%；
- 21,752,467 个 duplicate-timestamp rows，说明必须先按指标语义做 duplicate reduction。

**【已验证（审计实现）】** 本轮对当前 Train 窗口作了流式统计：审计处理 137,095,611 行，覆盖 10 个节点各自的 90 个 container logical features；结果复现了 30 s 对齐与 V3 quality 阈值，但没有完整复用 production reader 的 exact `(timestamp,value)` 去重和按 feature keyword 聚合分支。因此下表是可靠的方向性实测，63/90 在 P1 前仍须 exact replay：

| Service | 通过 V3 quality / 90 | coverage 中位数 | constant (`unique<2`) |
|---|---:|---:|---:|
| dbservice1 | 65 | 0.9278 | 17 |
| dbservice2 | 65 | 0.8514 | 19 |
| logservice1 | 63 | 0.8757 | 19 |
| logservice2 | 65 | 0.8514 | 19 |
| mobservice1 | 63 | 0.8642 | 21 |
| mobservice2 | 65 | 0.9277 | 17 |
| redisservice1 | 63 | 0.8642 | 21 |
| redisservice2 | 65 | 0.8514 | 19 |
| webservice1 | 63 | 0.8642 | 21 |
| webservice2 | 63 | 0.8757 | 17 |

十节点共同通过的是 **63/90**；当前 V3 因任一节点失败而删除 27 个 logical features：CPU 2 个，Disk I/O 9 个，Memory 8 个，Network 8 个。这里的 `constant` 是明确的 `unique<2`；near-constant 的最终定义及数量仍需 P0 冻结，不能事后调整。

**【抽样结果】** 200 个 raw files、约 780 万行中，主采样间隔为 30 s；发现约 41,545 个重复 timestamp，主要来自 system 文件，并有少量 29/31 s 抖动。对 90 个 container features，各节点 `max_missing_gap_bins` 的中位数为：

| 节点组 | 中位最大缺口 |
|---|---:|
| dbservice1、mobservice2 | 1,831 bins（约 15.3 h） |
| dbservice2、logservice2、redisservice2 | 7,232 bins（约 60.3 h） |
| logservice1、webservice2 | 4,264 bins（约 35.5 h） |
| mobservice1、redisservice1、webservice1 | 3,289 bins（约 27.4 h） |

`ffill(limit=10)` 对每个 feature 实际填补的中位数只有约 20–46 bins。10 bins 在主采样间隔下等于 5 min，本身不是离谱的数量级，但它既不随 30/60 s 实际周期变化，也无法处理长缺口；更严重的是剩余 missing 最终落到归一化 0，与真实最小值混合。

**【待全量统计】** 当前 V3 Train 的每个 `service×logical_feature` 还须由正式 audit 输出：coverage、unique count、zero ratio、variance、q05/q95、dynamic span/ratio、median/p95 sampling interval，以及连续 missing-gap 的 30/60/90/120/300 s 和尾部统计。进入 P1 前还必须给出：

- constant 和 near-constant 的明确定义及数量；
- 90 个 global features 在当前 70% Train 下逐节点通过/失败原因；
- 384 个 host features 在三个有 telemetry 的物理 host 上的质量；
- `.3` host 缺失是 `not available`，不能计为 feature quality failure。

### 3.4 Metric 冗余

**【已验证：语义层】** 当前 90 个公共 names 中包含明显的重复族：

- `*_pct` 与 `*_norm_pct`；
- `rss`、`rss_total`、`usage_total`；
- `active_*` 与 `total_active_*` 等 container/cgroup 镜像；
- cumulative `ticks/bytes/packets/ops/pgfault` 与对应 rate/percentage；
- disk `read/write/summary` 派生量。

这些列即使数值不完全相同，也可能让同一物理信号在 reconstruction loss 中重复出现。累计 counter 还会把“月份进度/进程存活时长”变成主趋势；应先 reset-aware 差分成 rate，再做相关审计。

**【待全量统计】** Pearson/Spearman 必须仅用 Train，并分别报告 `|rho|>=0.90/0.95/0.98/0.995` 的 pair 数和 connected components。推荐的全局判据不是在一次拼接上武断聚类，而是：同一语义族内，至少 8/10 candidate nodes 都达到阈值；host signals 则每个物理 host 只计一次，避免因共宿节点复制而放大相关证据。

代表列按性能盲规则选：直接 gauge/rate 优先于累计 counter；coverage 高、gap 短优先；物理含义清楚优先。不得用 Test F1 决定代表列或相关阈值。

### 3.5 Multi-core

当前 V3 对所有 `docker_cpu_core_<id>_*` 在同一 30 s bin 求 mean。对于 core utilization，mean 代表总体负载，但会掩盖单核热点；对 cumulative core ticks，直接 mean 的物理意义更弱且与 total CPU counters 高度重叠。

**【抽样结果】** 已完成 5/10 节点的 Train-only mean-vs-max 检查：

| family | Pearson 中位数（范围） | Spearman 中位数（范围） |
|---|---:|---:|
| core utilization `pct/norm_pct` | 0.500（0.152–0.728） | 0.337（0.249–0.555） |
| core cumulative `ticks` | 0.999（0.999–1.000） | 1.000（1.000–1.000） |

这支持两个不同处理：core utilization 的 max 并非 mean 的近重复，可能保留单核热点；ticks 的 max 基本冗余。其余 5 个节点及更完整的 core family 仍待 P0。

**【建议】** 仅对 core utilization family 保留 `mean` 和 `max` 两个候选；不默认加入 `std`，也不为 ticks 复制 mean/max/std。P0 必须在每个节点报告 mean-vs-max 的 Pearson/Spearman；若完整结果推翻上述模式，再按预注册规则处理。该决定只看 Train dynamics/冗余，不看标签。

### 3.6 Log 分布

**【历史全量证据】** 全月 87,974,871 条 CSV records；294 条 message-prefix timestamp 无效。Level 分布为：

| Level | records |
|---|---:|
| INFO | 84,269,925 |
| WARNING | 2,897,917 |
| ERROR | 806,735 |
| DEBUG | 0 |

WARNING 的 99.998% 集中于 `redisservice1/redisservice2`。这说明“只在单一 service/type 出现”不能作为删除依据。Business ERROR 是 telemetry；run table 中 ERROR 是否属于 GT 与 Log feature 是否保留是两条独立语义链。

**【待全量统计】** 当前 Train-only Drain3 的 cluster 数、每 cluster record/bin/day/service/level 分布尚未正式扫描；Test unseen 率也尚未在冻结 vocabulary 后生成。本审计不以全月 level 分布替代这些 Train 决策。

### 3.7 Trace 分布与图

**【历史全量证据】** 全月 28,681,438 条 trace records；status 和 duration full scan 为：

| Status | records |
|---|---:|
| 200 | 24,453,030 |
| 300 | 3,040,760 |
| 400 | 30,980 |
| 500 | 1,156,668 |
| non-200 | 4,228,408（14.7427%） |

3xx/4xx 共 3,071,740 条，因此 `status>=500` 会漏掉它们，不能代替冻结的 `status!=200` 语义。300、400 主要/仅见于 Redis traces，也说明 status identity 含有 type-specific 行为，不能在没有证据时全部抹成一个 duration sum。

Duration 无 invalid、negative 或 zero；full-scan mean 0.242663 s、min 0.001934 s、max 8,889.437669 s。每 257 行取一条的确定性样本为 median 0.016062 s、p75 0.312332 s、p90 0.735759 s。分布极长尾，raw mean/max scaling 都会不稳。

**【历史全量证据】** 较早 Train graph 为 58 个对称 adjacency entries，即 29 个无向 pair；它不是当前 70/30 P1 的正式边数。

**【待全量统计】** 当前 Train 必须输出每条 directed edge 的 call count、active bins/days、status 分布、count/mean/duration_sum 相关、latency quantiles、无 parent/跨 split parent 数；并报告低频阈值曲线，而不是直接把“一次出现”冻结成 topology。

## 4. 当前问题分级

### MUST FIX

1. **Trace duration_sum 混合量。** 同一个值可来自高流量低延迟或低流量高延迟，异常含义不可辨识。
2. **Trace parent resolution 不闭合。** span index 来自整月、只按 `span_id`，没有强制同一 `trace_id` 或 split-local parent；虽然 graph edge inclusion 检查 Train bin，source identity 仍可能受 split 外索引影响。
3. **Log 维数无上限。** 所有 Train Drain3 clusters 各占一列，raw-dimension sum 会改变模态权重。
4. **Metric missing 与真实最小值混为 0。** min-max 后的真实 Train minimum 和 residual NaN 都是 0，host not-applicable 也无法表示。
5. **Metric counter/gauge 未分型。** 累计 counter 直接进入 min-max/reconstruction，重复且容易让长期趋势主导。
6. **Metric strict intersection 误用结构约束。** 它删除 77.34% 的可映射 node-feature pairs；统一 tensor slot 不要求每个节点原始可观测性相同。
7. **P1 freeze 仍缺关键 Train-only 全量审计。** 不得在这些表缺失时直接正式 preprocessing/training。

### SHOULD FIX

1. 用 time-aware fill 替代固定 300 s forward fill。
2. 用 Train percentile clipping + min-max、Train median imputation 和少量 observability slots 替代 raw min-max + NaN→0。
3. 用 `log1p + Train q99-positive` 替代 Log Train maximum normalization。
4. Log 的 RARE/UNK 都按 level 区分；`log_total` 仅作 audit 派生量，不作为模型 slot。
5. Train graph 加入最低重复性证据，并记录 directed observation 与 symmetric adjacency 的映射。

### OPTIONAL

1. Trace p95/max latency 作为冻结后的消融分支；不进入第一版 baseline。
2. 对已验证的外部 `redis/zookeeper` infrastructure 建独立节点或上下文；这会改变图/候选宇宙，不属于本轮最小方案。
3. 修改 Ada-MGAD loss 做显式 modality weighting；只有预处理维数/尺度控制仍不足时才考虑。

## 5. 推荐 preprocessing 方案

下表中的 leakage 均指 preprocessing decision 是否接触 Test。所有 `fit/choose/cluster/scale/graph` 操作只在 Train。

### 5.1 Metric 完整流程

| 步骤 | 做法 | 解决的问题 | 改核心模型？ | Leakage 风险 |
|---|---|---|---|---|
| 1 | filename + 部署映射生成带 scope 的 logical registry；未知 `redis/zookeeper` namespace 隔离 | 防止 raw filename 当 feature、错误归属 | 否 | 低；mapping 不读标签/Test |
| 2 | 标注 gauge/counter/rate；counter 做 reset-aware `delta/time`，负 delta 视为 reset/missing | 去掉长期累计趋势和派生重复 | 否 | 低；规则按语义，参数 Train-fit |
| 3 | multi-core 仅生成 utilization mean/max 候选 | 保留平均负载与单核热点 | 否 | 低 |
| 4 | 对 global、host 分 scope 做 Train quality 表 | 不让 host N/A 拖垮 global schema | 否 | 低 |
| 5 | time-aware fill：`max_age=clip(2×Train median positive interval,60s,120s)`，严格 split-local | 不跨过长 outage，也适配 30/60s 指标 | 否 | 低 |
| 6 | residual missing 用该 `service×feature` 的 Train median；另给 scope observability | 不再与真实 min 混淆 | 否 | 低 |
| 7 | 在同语义族内作 Pearson/Spearman cluster；按 coverage、gap、直接物理含义选代表 | 防止重复 reconstruction weight | 否 | 低 |
| 8 | Train q01/q99 clipping 后 min-max 到 `[0,1]`；常量在此前移除 | 抗 outlier 且与当前模型范围兼容 | 否 | 低 |
| 9 | 冻结 ordered schema、scope、transform、quantiles、代表列和 source hashes；Test 只 transform | 可复现和防回看 | 否 | Test 不参与选择 |

为什么不首选 median/MAD scaling：它对长尾更 robust，但输出无界、可为负，会较大改变当前 reconstruction residual 的尺度。`Train percentile clipping + min-max` 保留当前 `[0,1]` 习惯，是风险更小的修正。Median/MAD 可作为以后预注册消融，不能由 Test 结果救场。

缩放参数的粒度也要冻结：container slots 按 `service×logical_feature` 在 Train 内拟合 q01/q99，使共享线性层接收可比较的“本节点相对动态范围”；host slots 按 `physical_host×logical_feature` 拟合后复制给该 host 上的 candidate nodes，不能因复制到多个 service 而重复拟合出不同尺度。slot 名、变换类型和顺序全节点一致，参数允许绑定到真实数据源实体；这仍满足统一语义，不等于要求各节点使用同一数值上下界。

### 5.2 Log 完整流程

| 阶段 | 为什么/解决什么 | 改核心模型？ | Leakage 风险 |
|---|---|---|---|
| Train-only 全局 Drain3 | 让所有节点共享 slot 语义，阻止 Test 建新模板 | 否 | 低；fit 只读 Train |
| records/bin/day/service/level 审计 | 区分稳定模式、单次 burst 与稀有高严重度消息 | 否 | 低；不用 GT/Test |
| stable + level-aware RARE/UNK | 控制 raw 维数，同时不删除 rare ERROR | 否 | 低；规则和 K 在 Train 冻结 |
| `log1p + Train q99-positive` | 抑制 count 长尾，避免 Train max 被单个 burst 支配 | 否 | 低；Test 不更新 scale |
| ordered schema/state/hash freeze | 保证 Train/Test、所有节点使用同一 vocabulary | 否 | 低；Test 只 match/transform |

1. 只用 Train、全节点共享的 Drain3，固定 config、service/file/row 顺序。
2. 为每个 cluster 统计 records、active 30s bins、active days、services、level association；stable 判据先由 Train 分布的 elbow 和预注册维数 budget 冻结。
3. stable clusters 各占一列；其余 cluster **不删除**，进入 `RARE_INFO/WARNING/ERROR/DEBUG/UNKNOWN`。
4. Test 只 match。无匹配消息进入 `UNK_INFO/WARNING/ERROR/DEBUG/UNKNOWN`，不能创建 cluster。
5. 保留 5 个全量 level counts。`log_total=sum(level_counts)`，同时也等于 stable+RARE+UNK 总和，故只写 audit，不进入模型，避免第三次计算同一事件量。
6. 每列 `log1p(count)`，再除以 Train positive-bin `q99` 的 `log1p` 值并 clip 到 `[0,1]`；若 Train 无正值，该 slot 不应存在。Test 不更新 scale。
7. 冻结 `K_stable + 5 RARE + 5 UNK + 5 levels` 的 ordered schema。目标总维数 32–64，即 stable template budget 约 17–49；具体 K 必须由 P0 Train-only 表确定，不能先写死为“频次前 K”。

稀有 ERROR/WARNING 因此仍保留“严重度 + 稀有性 + 数量”，但不会让每个一次性文本各占一列。只在单个 service 出现不是删除条件；node index 已经提供 service identity。

### 5.3 Trace 完整流程

| 阶段 | 为什么/解决什么 | 改核心模型？ | Leakage 风险 |
|---|---|---|---|
| split-local `(trace_id,span_id)` parent map | 防止 span 碰撞和跨 split source resolution | 否 | 低；Train/Test index 分离 |
| Train-only edge registry/graph | Test 不得改变模型 topology | 否 | 低；不读 Test/GT 选边 |
| status×`count+mean_latency` | 分离流量和延迟，去掉乘积冗余 | 否，仅 `raw_edge=8` | 低；固定聚合规则 |
| `log1p + Train q99` scale | 控制 count/latency 长尾与 slot 尺度 | 否 | 低；Test 不更新 scale |
| directed evidence + symmetric adjacency 双记录 | 保留调用方向，同时兼容当前 GAT | 否 | 低；映射在 Train 冻结 |

1. 使用 `(trace_id, span_id)` 建立 parent map；child 只允许匹配同一 trace。Train/Test 分别建局部索引；边界 trace 默认 fail closed 为 unmatched，并计数。
2. 只由 Train 中成功解析的 directed cross-service calls 建图；Test 不得增边。
3. graph inclusion 不采用“一次出现即进入”。P0 报告 count/active-bin/active-day 曲线后，冻结最低重复性规则；建议起点是“至少 2 个 Train active bins”，并保留人工语义白名单入口，但不得看 Test/GT。
4. 模型 adjacency 为兼容当前结构继续对称化；artifact 同时保存 directed observed edge table，避免把无向 adjacency 误称为调用方向。
5. 每个 30s、每条 directed edge、每个 status `200/300/400/500` 生成两列：

```text
log1p(call_count)
log1p(mean_latency_seconds)  # count=0时置0，count slot明确其不可用
```

最终 `raw_edge=4 statuses×2=8`。不保留 `duration_sum`，因为它可由 count×mean 恢复且会重复加权。第一版不加 p95：edge/status/bin 稀疏时不稳、计算重且再加 4 维；将其保留为预注册消融。

6. Count 用 Train positive-bin q99 缩放；latency 先 `log1p`，再按对应 Train edge/status 的 q99 缩放。稀疏 slot 可回退到同 status 的全 Train pooled scale，但回退层级必须预先固定并写入 artifact。

此方案不改变 Ada-MGAD 核心模型，仅把 `raw_edge` 从 4 改为 8；`feature_edge=4` 可以保持不变。它比 `count+mean+duration_sum` 少冗余，也比第一版 `count+mean+p95` 稳健。

## 6. Metric schema proposal

### 6.1 分层 slot 设计

建议 ordered node vector：

```text
[global_container_representatives]
[host_of_node_representatives]
[global_observed_fraction]
[host_applicable]
[host_observed_fraction]
```

- `global_container_*`：来源于当前 90 个 candidate-shared Docker logical names，经 counter transform、quality 和 redundancy 后的代表列。
- `host_of_node_*`：从 384 个 system logical names中按物理 host（不是复制后的 service）做 quality/冗余，选择紧凑代表；在有 host telemetry 的 8 个节点复制 host-of-node 值。
- `.3` 上两个节点的 host slots 用 Train-neutral value；`host_applicable=0`。其他节点为 1。
- observability 是 scope aggregate，不为每个 feature 复制一套 missing mask，避免维数翻倍。

### 6.2 为什么本轮不建立五个 service-type blocks

当前 raw filename 事实没有证明 10 个 candidate services 拥有 DB/Redis/Web/Mob/Log 专属 exporter features。人为创建五个空/错配 blocks 没有意义；把 `redis@0.0.0.3` 强行映射到 `redisservice1/2` 更会污染语义。

若 P0 获得部署证据，未来可在相同 index 语义下添加 type block：例如 `redis.redis_keyspace_keys` 对 Redis nodes 有值，其他 nodes 为 Train-neutral，并有 `type_applicable`。但这必须是新的 schema revision，不是本轮推测。

### 6.3 维数预算

当前可证实的候选基数是 global 90、host 384，而不是 474 全量 union。建议 P1 采用：

| Scope | 质量过滤前 | 当前/待完成的质量过滤 | 冗余过滤后 |
|---|---:|---:|---:|
| global container | 90 | 63 个十节点共同通过（本轮审计实现；待 production-exact replay） | 目标 30–40，相关 cluster 尚待 P0 |
| candidate service-type | 0 个已证实 | 不适用 | 0；`redis/zookeeper` 在拓扑证明前隔离 |
| host-of-node | 384 names，覆盖 3 台 host/8 个 nodes | 待按物理 host 完成 Train quality | 目标 15–21，尚待 P0 |
| observability | 3 个设计 slots | 规则定义，不做数据驱动筛选 | 3 |

所以本轮能够用实际统计确定的是 `90 -> 63` 的 global quality 收缩，以及 host full union 必须从 384 大幅压缩；`63 -> 30–40` 和 `384 -> 15–21` 仍是模型规模约束下的预算，不是伪装成统计结果的 exact count。

| Block | 目标预算 | 依据 |
|---|---:|---|
| global container | 30–40 | 90 names 中存在 counter/rate、pct/norm_pct、total/non-total 等明显冗余 |
| host-of-node | 15–21 | 384 host names 仅选 CPU/load、memory/swap、disk I/O、network、process 的直接 gauge/rate 代表 |
| scope/observability | 3 | 区分 missing、not-applicable 和真实低值 |
| total | **48–64** | 与当前 embedding/reconstruction 规模相容，避免 full union |

这是 dimension budget，不是已经冻结的 exact feature list。P0 表未完成时不得把区间端点挑成正式维度。一个合理的 P1 freeze 目标是约 56 维；若按预注册规则超过 64，应优先继续语义/冗余去重，而不是扩大模型输入。

Host 值复制给共宿 service 会使同一物理信号在多个节点重构。由于 block 已被压到约 15–21 维，该风险可控；manifest 还应记录每 host 的 candidate multiplicity。不要用每节点独立 min-max 抹掉 host slot 的共同幅度语义；host scale 应按物理 feature 在 Train hosts 上 pooled fit。

## 7. Log schema proposal

推荐顺序：

```text
stable_template_<frozen_id>  × K
RARE_INFO/WARNING/ERROR/DEBUG/UNKNOWN × 5
UNK_INFO/WARNING/ERROR/DEBUG/UNKNOWN  × 5
level_INFO/WARNING/ERROR/DEBUG/UNKNOWN × 5
```

关键规则：

- vocabulary 必须全节点共享；single-service template 可以保留，slot 在其他节点为真实 count 0。
- RARE 是“Train 见过但未获独立 slot”；UNK 是“Train 从未见过/冻结 matcher 无匹配”，两者不得混合。
- rare 不等于无信息。ERROR/WARNING 的 rare buckets 保留异常稀有性。
- `log_total` 写到统计 artifact，但不作为 raw slot；它与五个 level 总和精确重复。
- stable 判据以 active bins/days 为主、record count 为辅，避免一次 burst 靠重复行占据独立 slot。
- 本轮不宣称 Drain3 cluster 数或 Test unseen 率；这两项是 P0 hard gate。

## 8. Trace schema proposal

最终推荐：

```text
raw_edge = 8

200_count_log, 200_mean_latency_log,
300_count_log, 300_mean_latency_log,
400_count_log, 400_mean_latency_log,
500_count_log, 500_mean_latency_log
```

为什么不是其他候选：

| 候选 | 判断 |
|---|---|
| duration_sum only | 拒绝；流量和时延不可辨识 |
| count + mean | **推荐**；信息正交度最好，8 维仍很小 |
| count + mean + duration_sum | 拒绝；第三项是前两项乘积，重复 reconstruction 权重 |
| count + mean + p95 | 仅消融；稀疏 edge/status/bin 下 p95 不稳且升到 12 维 |

Status 继续保留四类，因为当前数据中 3xx/4xx 规模和 service-type 分布不同；不能只保留 5xx。若未来要压到更低维，必须在独立、Train-only、性能盲的表示审计中比较 `200 vs non-200`，不能在本轮凭直觉合并。

## 9. 对 Ada-RCA adapter 的影响

Ada-RCA 的 68D Z2 representation 保持冻结，不随本文重新设计。它与 Ada-MGAD tensor 是独立 raw adapter。

必须单独修正/冻结的问题：

1. `src/e2e/gaia_rca_adapter.py:451` 复用 `_logical_metric_schema(metric_dir)`，因此也采用全时间 filename universe 的 10-service common intersection。
2. `_q_by_service()` 实际对每个 candidate 的可用 indicators 做 robust deviation 后取 Q90（`src/e2e/rca_features.py:136-176`）；68D 输出并不要求每个 candidate 拥有相同数量的 raw metric indicators。
3. 因此 common intersection 对 Ada-RCA 不是 tensor shape 必需条件，可能造成不必要信息损失。修正应只改变 performance-blind raw indicator registry：Train-only schema、语义 scope、可用性/质量规则；不得改变四 channel、Q90 aggregation、Z2 68D 或已冻结 scorer。
4. RCA 当前 Trace 已使用 `status_not_200_count + latency_mean`，不要把 Ada-MGAD 新的 8D edge tensor直接替换进 68D pipeline。
5. RCA raw trace 不是基于 parent edge，而是按 child service 聚合；Ada-MGAD 的 parent-map 修复和 RCA adapter 语义应分别验证，不能把两个 track 说成同一生成链。

## 10. P0 完整审计命令与必需输出

本轮已新增独立的 audit-only 脚本 `scripts/p5/audit_gaia_multimodal_preprocessing.py`。默认只绑定 source/config 并写 metadata；本轮只验证了 `--help` 与 metadata-only，**没有运行 `--full`**，也没有写正式 preprocessing 目录。完整 P0 命令为：

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/p5/audit_gaia_multimodal_preprocessing.py \
  --config configs/e2e/gaia_p5_v3.json \
  --raw-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS \
  --output-root artifacts/p5/v3_preprocessing_audit \
  --split train \
  --chunk-rows 150000 \
  --workers 1 \
  --start-method spawn \
  --full
```

当前实现为了确定性和资源上界采用串行扫描；`workers/start-method` 会进入 manifest，但 `workers` 暂不驱动并行。因此不要把 `--workers 24` 误写成已经实现的 24 进程加速。Trace parent index 使用 Train-only、磁盘有界的 `(trace_id,span_id)` 128-bit hash SQLite index；duration 的 count/sum/min/max 为流式精确统计，p50/p95 明确标记为 deterministic bounded-reservoir 近似。Log 的 Test pass 只做冻结 matcher 的 `[boundary, absolute_end)` UNK transform 统计，不参与任何 selection。

该脚本的完成条件不是一个 summary 数字，而是以下机器可读文件全部存在且 source/config hash 绑定一致：

```text
source_binding.json
metric_registry.csv
metric_quality_train.csv
metric_missing_gaps_train.csv
metric_correlation_clusters_train.json
metric_multicore_train.csv
log_templates_train.csv
log_template_distribution_train.json
log_frozen_transform_stats.json
trace_edges_train.csv
trace_parent_resolution_train.json
trace_feature_distribution_train.json
audit_manifest.json
```

`log_frozen_transform_stats.json` 可以在 vocabulary 完成后报告 Test UNK transform 统计，但任何 Test 字段都不得成为 stable/rare/K/scale 的输入。审计脚本还必须输出显式字段 `decision_inputs=["train"]`、`gt_labels_used=false`、`test_used_for_selection=false`。

在该脚本的 `--full` 尚未跑完并复核前，本文件中的 48–64/32–64 是预算，不是正式冻结维数；不得开始 P4/P5。

## 11. 最小风险实施顺序

### P0：数据审计（当前 gate：脚本已实现，full scan 未执行）

- 在隔离输出目录执行并复核上节 audit-only 脚本；输出 Train-only quality/correlation/template/edge 表。
- 对 `redis/zookeeper` namespace 和 `.3` 缺失 system telemetry 做部署语义核验。
- 只做数据审计，不写正式 preprocessing 目录，不训练。

**Gate：** 全部机器可读表、hash、复现命令、evidence ledger 齐全；否则 NO-GO。

### P1：preprocessing schema 冻结

- 冻结 Metric ordered slots、counter transform、fill、clip/scale、维数。
- 冻结 Log Drain config、stable 判据、K budget、RARE/UNK/level slots、scale。
- 冻结 Trace parent rule、graph edge rule、8D ordered slots、scale。
- 冻结所有阈值时不查看 Test F1/GT labels。

**Gate：** schema JSON + source/config digest + deterministic replay；post-freeze 只允许 presentation-only 修正。

### P2：代码实现

- 在 `ad_preprocess.py/log_templates.py` 边界层实现，不先改 Ada-MGAD 核心 loss。
- 为 split-local parent map、missing/not-applicable、counter reset、RARE/UNK、维数和 slot order 添加测试。
- 同步修正 RCA raw metric registry 的不必要 common intersection，但保持 68D representation/scorer 不变并单独版本化。

### P3：preprocessing smoke

- 小时间窗、所有服务、每模态至少一个正常/缺失/稀有/status/parent fixture。
- 验证 shape、finite、slot 语义、Train/Test firewall、cache binding 和 deterministic replay。
- Smoke 明确标记 `NOT FORMAL RESULT`。

### P4：full preprocessing

- 在隔离输出目录、单 writer、冻结 config 下执行。
- 先验证 manifest，再允许后续训练。

### P5：Ada-MGAD 正式训练

- 使用冻结 schema；Train-only checkpoint/calibration；Test 只作一次冻结推理与报告。
- 若结果不理想，报告 limitation；不得依据 Test 回头改变 schema。

## 12. 最终决策

当前状态是：

```text
SCIENTIFIC DIRECTION: GO
FORMAL PREPROCESSING/TRAINING: NO-GO UNTIL P0/P1
```

可立即接受的方向是：Metric 分层紧凑 schema、Log level-aware rare/UNK 控维、Trace `count+mean` 8D、split/trace-local parent resolution。不能立即接受的是任何 exact Metric/Log feature list、相关阈值、stable template 数或 Train graph 边阈值；这些必须由尚未完成的 Train-only 全量审计冻结。
