# GAIA Event Inclusion and Concurrency Audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-19
- Verification Status: UNVERIFIED
- Version Label: p1_gaia_inclusion_v1

> 状态：GAIA 主 cohort 已冻结为 13,470 个锚点唯一 root-service events；G1 completed

## 1. 标签语义边界

本地 GAIA 说明明确提示“GAIA 数据集本身没有很明确的 label”，当前标签来自
`run_table_2021-07.csv` 的运维事件解析。五类保留事件的 message 使用
`simulate`、`trigger` 或显式 anomaly marker，但这仍只支持以下保守表述：

- 每个 case 的 label 是该条目标运维事件指定的 service；
- 它不是整个 ±300 s 窗口中“唯一真实因果故障”的证明；
- `fault_type` 仍为分析元数据，不是 prediction feature。

因此协议把完整解析结果称为 **event inventory**，把满足 service-level
single-root 条件的子集称为 **main cohort**。

## 2. Inventory 与排除规则

原始 run table 共 17,152 条记录：

- 保留 16,200 条：五类 supported、time-bounded、root in candidate services 的
  operational anomaly events；
- 排除 952 条：861 normal、38 non-injection error、36 unsupported、17 recovery
  marker；
- 不因 telemetry missingness、模型分数或下游表现删除 case。

五类 inventory 分布为 login 15,478、memory 652、file-moving 43、access 15、
CPU 12。

## 3. 主 cohort 规则

对目标 event 的锚点 `t0`，用半开实际事件区间判断其他事件是否仍活跃：

```text
other.start <= t0 < other.end
```

若至少一个活跃事件来自不同 root service，则该 case 在 service-level 上不是
single-root，进入 concurrency sensitivity；否则进入主 cohort。同 root 的并发
事件仍保留，因为 root-service label 仍唯一，fault type 只作分析元数据。

该规则在查看 cohort-specific baseline 结果之前按任务语义固定，不使用模型表现
作纳入依据。

## 4. 并发计数

| Cohort/flag | Cases | 解释 |
|---|---:|---|
| Full event inventory | 16,200 | 所有受支持的目标事件 |
| Actual interval isolated | 12,163 | 与任何其他真实事件区间都不相交 |
| Anchor no concurrent event | 13,077 | `t0` 时没有其他事件活跃 |
| Anchor same-root concurrent | 393 | `t0` 有并发，但 service root 仍唯一 |
| Main: anchor unique root service | 13,470 | 13,077 + 393 |
| Sensitivity: anchor multi-root | 2,730 | `t0` 有不同 service root 活跃 |
| Full ±300 s context isolated | 16 | expanded context 与任何其他 case 不相交 |

不能把“整个 ±300 s context 完全隔离”作为主纳入条件：那只剩 16 cases，且会
系统性删除密集的 login/mobile 事件。跨 split 污染继续由 322 个 context groups
控制；case 纯度与 split 泄漏是两个不同问题。

## 5. Main cohort 分布与 folds

主 cohort 的 fault counts：login 12,890、memory 526、file-moving 33、access 12、
CPU 9。root counts 仍高度不平衡：mobservice1/2 为 6,476/6,525，其余服务为
38–76。

主 cohort 继承已经通过 group-integrity 的 assignment，同一 context group 不会
跨 fold。过滤后的 fold sizes 为：

| Fold | Cases |
|---|---:|
| 0 | 2,693 |
| 1 | 2,684 |
| 2 | 2,711 |
| 3 | 2,705 |
| 4 | 2,677 |

五折均含全部五种 fault types 与全部十个 root services。B1 Frequency 的每折
root counts 已重新只从 main-cohort training cases 拟合；没有沿用 full inventory
的 prior。

## 6. Main-cohort sanity baselines

| Baseline | Overall AC@1 | Overall Avg@5 | Root-macro AC@1 | Root-macro Avg@5 | Fault-macro AC@1 | Fault-macro Avg@5 |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.1019 | 0.3031 | 0.0993 | 0.3132 | 0.1202 | 0.3561 |
| Root Frequency | 0.4808 | 0.8745 | 0.0994 | 0.2966 | 0.1435 | 0.3889 |
| Metric Change | 0.1405 | 0.3577 | 0.5311 | 0.7086 | 0.4592 | 0.5981 |

过滤并没有消除 GAIA frequency shortcut；Frequency 的 overall AC@1 仍为
0.4808，而 root-macro 只有 0.0994。Metric Change 的 13,470 main cases 中，
11,843 有至少一个 service 的有效 metric score，1,627 为全服务 fallback。

完整 16,200 inventory 的结果仍保存在 `artifacts/p1/baselines/gaia/`，用于
concurrency sensitivity；主结果使用 `artifacts/p1/baselines/gaia_main/`。

## 7. 产物与哈希

| Artifact | SHA-256 |
|---|---|
| Inclusion diagnostics | `9b9ddf8a63973eeb8c0cee4fe07bc18a2465ff353d2482ae18498792f07f13be` |
| Inclusion flags | `9a616cd64939b33e5192b0517200fbd8f92891bdaeffefc42e4188a7b9ff9542` |
| Main cohort | `bb00bddaef6753a68c0651734003e0a75eed33a2f3fac03c40fb5dac4df450c8` |
| Inclusion manifest | `88cef66b3cff0633b1cbd40c70b42fb004b1db0674364b77a59c618c353b6cca` |
| GAIA-main Random run manifest | `5f8c5a1920ee8a72088d30b2a2a463ef5f46da280c24c2eccfdc087dbcddbe81` |
| GAIA-main Frequency run manifest | `7fc2181299a0afe232084c5795f5e1ef0c6a591f1854262a5faa48b8ee92cf44` |
| GAIA-main Metric Change run manifest | `a04af8ce6b34dfa3d1ed2202c5b3c653f79c7420b99e87e2364d7b8715b08cf4` |

## 8. 结论

G1 记为 completed：

- 16,200/16,200 supported events 有稳定 `RCACase` inventory；
- 952 个排除项逐条有原因；
- 13,470-case service-single-root 主 cohort 与 2,730-case multi-root sensitivity
  均有可审计 sidecar；
- 主 cohort baselines、fold-specific Frequency fit 和完整 rankings 已重新生成；
- 任务声明从“窗口内唯一故障”收紧为“锚点时唯一 root service 的目标事件”。
