# P1 数据集初步审计

> 审计日期：2026-08-18
> 状态：初步数据清单与 adapter 验证；不等同于完整 G8 diagnostics
> 代码基线：ab31282055b625838f41a1a3c91e51098175f254 + 当前未提交 P1 变更

## 1. 审计范围与证据边界

本轮只读检查了用户提供的两个本地数据目录：

- RCAEval：/home/zhangll24/RCA_project/datasets/RCAEval
- GAIA MicroSS：/home/zhangll24/project_2/MultimodalAD/MSTGAD-GAIA/data/GAIA/MicroSS

外部语义依据为 RCAEval 与 GAIA 官方仓库。官方 RCAEval 当前说明 RE2-OB 由
5 个注入服务、6 类故障、每类 3 次重复组成，共 90 cases；GAIA 官方说明
MicroSS 的 run 目录保存系统日志和全部异常注入记录。

本轮没有计算完整模态 missingness、case 级时间覆盖或 topology 覆盖，因此
G8 仍为 pending。RCAEval 本地数据尚未固定下载来源、版本标识与 checksum，
不能把本审计当作最终数据 manifest。

## 2. RCAEval RE2-OB

### 2.1 本地布局

本地目录包含 30 个 root_service_fault_type 条件目录，每个目录有 1、2、3
三个正式 case，共 90 cases。某个 case 下额外存在 multi-source-data 派生目录；
它不是第 91 个 case。

本地正式 case 使用：

- metrics.csv；
- simple_metrics.csv；
- logs.csv；
- traces.csv；
- inject_time.txt；
- 若干派生日志或辅助文件。

这与官方 README 当前概述的 metrics.json 布局不同，adapter 必须以本地
manifest 为准，同时在复现实验中固定数据发布版本。

### 2.2 Adapter 结果

| 项目 | 结果 |
|---|---:|
| 正式 case 数 | 90 |
| 成功映射为 RCACase | 90 |
| 排除 | 0 |
| 每 case 候选服务数 | 11 |
| root_service not in services | 0 |
| 故障类型 | cpu/delay/disk/loss/mem/socket，各 15 |
| 根服务 | checkout/currency/email/productcatalog/recommendation，各 18 |

候选服务由 CPU/Memory 服务指标实体推导，并排除 loadgenerator、
frontend-external、frontend-check、istio-init 等辅助实体。过滤前，
checkoutservice_loss/3 出现 frontend-check，checkoutservice_socket/1 出现
istio-init，导致两个 case 有 12 个候选；过滤后 90 个 case 均为 11 个应用服务。

为防止从目录名直接读取标签，预测可见 case_id 使用不可读哈希，
TelemetryRef 使用 rcaeval://re2-ob/<opaque-id>/...；真实
root_fault/replicate 路径仅保存在可信 source sidecar，root_service 与
fault_type 仅保存在 RCACaseLabel。

## 3. GAIA MicroSS

### 3.1 原始资产

| 资产 | 本地观察 |
|---|---:|
| run_table_2021-07.csv 记录数 | 17,152 |
| metric CSV shards | 6,640 |
| trace service files | 10 |
| business-log service files | 10 |
| RCA 候选服务实例 | 10 |

指标目录还包含 redis、system、zookeeper 等基础设施实体；当前 RCA 候选集合
沿用已冻结的 10 个 GAIA application service instances。是否把中间件实体加入
后续候选空间属于研究设计变化，不能在 P1 中静默修改。

### 3.2 注入记录映射

将官方 run 表中可支持、且具有可解析起止时间的注入记录映射为候选 RCACase：

| fault_type | 数量 |
|---|---:|
| login failure | 15,478 |
| memory_anomalies | 652 |
| file moving program | 43 |
| access permission denied exception | 15 |
| cpu_anomalies | 12 |
| 合计 | 16,200 |

排除 952 条非 case 记录：

| 原因 | 数量 |
|---|---:|
| normal_record | 861 |
| non_injection_error_record | 38 |
| unsupported_event_type | 36 |
| recovery_marker | 17 |

旧 parse_anomaly_event 对 CPU 时长只接受整数，但 12 条 CPU 注入使用浮点秒，
因此旧逻辑得到空结束时间。新 GAIA adapter 仅在 RCA 路径修复该正则，不修改
旧异常检测预处理器。

### 3.3 分布与污染风险

根服务分布高度不平衡：

| root_service | 数量 |
|---|---:|
| mobservice1 | 7,815 |
| mobservice2 | 7,799 |
| dbservice1 | 83 |
| dbservice2 | 96 |
| logservice1 | 66 |
| logservice2 | 63 |
| redisservice1 | 55 |
| redisservice2 | 57 |
| webservice1 | 82 |
| webservice2 | 84 |

共有 4,037 个 case 与至少一个其他注入区间重叠，涉及 3,429 对重叠事件。
因此不能简单随机拆分单条事件；本轮已构建 overlap-connected groups，后续
将比较 grouped split 与时间分块。最终是否排除污染 case、缩短上下文或作为
单独困难子集，需结合模态时间覆盖与 T_pre/T_post 诊断后冻结。

login failure 占绝大多数，且集中在两个 mobservice 实例。后续结果必须同时
报告 overall、fault-type macro、root-service macro，并保留 train-fold-only
Frequency baseline，以揭示分布捷径。

## 4. 可审计 manifest 与 overlap groups

两个数据集已生成 `p1_rca_manifest_v1` bundle，路径为
`artifacts/p1/manifests/{gaia,re2ob}/`。每个 bundle 至少包含：

- `inputs.jsonl`：prediction-visible 字段；
- `labels.jsonl`：仅训练/评价使用的 root_service 与 fault_type；
- `sources.jsonl`：可信侧原始路径解析；
- `groups.jsonl`：未来 split 的原子分组；
- `manifest.json`：逐文件行数、角色与 SHA-256。

GAIA 另含 16,200 行 `event_audit.jsonl` 与 952 行 `excluded.jsonl`；
RE2-OB 的 exclusion 文件为 0 行。生成后已重新计算全部 checksum，并扫描两个
`inputs.jsonl`：`root_service`、`fault_type`、原始绝对路径、
`relative_directory` 与 `source_index` 均为零命中。

GAIA 的 16,200 个 case 形成 13,077 个 overlap-connected groups。其中
12,163 个为 singleton；914 个非单例组覆盖 4,037 个 case，最大组 39。
这说明 grouped split 不会因单一巨型连通分量而立即失效，但最终采用 grouped
fold 还是时间分块仍需结合完整时间覆盖诊断决定。RE2-OB 的 90 个正式 case
暂按 90 个 singleton groups 记录。

## 5. 当前 Gate 解释

- G2：adapter 已在本地达到 90/90 可读、0 排除；仍需补数据版本 manifest。
- G4：AC@1/3/5、Avg@5、MRR 与非法 ranking toy tests 已通过。
- G1：GAIA 注入到 RCACase 的代码路径已实现；重叠清洗、上下文与 split
  策略尚未冻结，因此本轮只记为 partial。
- G6：overlap-connected groups、统一 split validator 与正/负向测试已实现；
  实际 split policy 和 assignment 尚未冻结，因此记为 partial。
- G7：输入/标签/来源物理拆分、metadata 敏感键拦截、opaque RE2 URI、
  checksum 与 input-only predictor 测试已实现；完整 preprocessing/baseline
  路径尚未建立，因此仍为 partial。
- G8：本文件只是初步审计，不含完整 missingness、时间覆盖和 topology 诊断，
  仍为 pending。

## 6. 下一步

1. 固定 RCAEval 数据来源/发布标识，并记录 raw data checksum。
2. 流式统计两数据集各模态相对 t0 的覆盖、采样粒度、重复与 missingness。
3. 基于诊断冻结 T_pre/T_post、GAIA split 与 topology 来源，生成实际 split assignment。
4. 实现三个 sanity baselines 并保存 per-case ranking。

## 7. 官方来源

- https://github.com/phamquiluan/RCAEval
- https://github.com/phamquiluan/RCAEval/blob/main/RCAEval/benchmark/evaluation.py
- https://github.com/CloudWise-OpenSource/GAIA-DataSet
