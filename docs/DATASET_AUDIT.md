# P1 数据集审计与全量诊断索引

> 审计日期：2026-08-18 至 2026-08-19
> 状态：数据清单、adapter 验证与 G8 全量 diagnostics 已完成
> 代码基线：ab31282055b625838f41a1a3c91e51098175f254 + 当前未提交 P1 变更

## 1. 审计范围与证据边界

本轮只读检查了用户提供的两个本地数据目录：

- RCAEval：/home/zhangll24/RCA_project/datasets/RCAEval
- GAIA MicroSS：/home/zhangll24/project_2/MultimodalAD/MSTGAD-GAIA/data/GAIA/MicroSS

外部语义依据为 RCAEval 与 GAIA 官方仓库。官方 RCAEval 当前说明 RE2-OB 由
5 个注入服务、6 类故障、每类 3 次重复组成，共 90 cases；GAIA 官方说明
MicroSS 的 run 目录保存系统日志和全部异常注入记录。

后续已完成约 38 GB 原始 telemetry 的流式全量扫描。missingness、case 级时间
覆盖、timestamp 质量与 topology 来源见
[TELEMETRY_DIAGNOSTICS.md](TELEMETRY_DIAGNOSTICS.md)；机器可读产物为
`artifacts/p1/telemetry_diagnostics.json`。RCAEval 本地数据仍未固定下载
来源、版本标识与完整内容 checksum。

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

将 run 表中可支持、且具有可解析起止时间的 operational anomaly events 映射为
RCACase inventory：

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
因此不能简单随机拆分单条事件；本轮已构建 context-overlap-connected groups，
并在候选比较后把 grouped-stratified 5-fold 冻结为主 assignment。最终是否
排除污染 case 或将其作为单独困难子集仍须在 G1 纳入规则中冻结。

login failure 占绝大多数，且集中在两个 mobservice 实例。主结果进一步排除
2,730 个锚点时存在其他 root service 的 case，保留 13,470 个锚点唯一
root-service cases；完整 16,200 inventory 作为 concurrency sensitivity。结果必须同时
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

仅按原始注入区间审计时，GAIA 的 16,200 个 case 形成 13,077 个连通组，其中
914 个非单例组覆盖 4,037 个 case，最大组 39。主窗口冻结为 ±300 s 后，
正式 `groups.jsonl` 改为“实际注入区间 ∪ 半开上下文”的连通分量：共 322 组，
306 个非单例组覆盖 16,184 个 case，最大组 471。RE2-OB 的 90 个正式 case
来自相互独立的实验目录，仍按 90 个 singleton groups 记录。实际 assignment
已冻结为 seed `20260819` 的 stratified 5-fold；每折 18 cases、每种 fault
每折 3 条、每个 root 每折 3–4 条。

GAIA 的候选比较也已完成：时间块将全部 12 条 CPU fault 放入最后一折，不能
满足每折可行 fault 覆盖约束；主方案因此冻结为 grouped-stratified 5-fold，
fold sizes 3239/3239/3239/3241/3242。实际 assignment 与完整比较见
[SPLIT_DIAGNOSTICS.md](SPLIT_DIAGNOSTICS.md)。

## 5. 全量遥测诊断摘要

- 共读取 349,120,558 行；GAIA/RE2-OB 的 metrics/logs/traces 均无未完成文件；
- GAIA 候选 metrics 5,724 文件、logs 10/10、traces 10/10；916 个未扫描
  metric 文件全部属于非候选 Redis/ZooKeeper 基础设施实体；
- GAIA 候选 metric value-cell missingness 为 0，但推断采样网格缺口率为
  16.18%，重复 timestamp 行为 21,752,467；
- GAIA 的 294 行无效 log 时间全部对应缺失/不可解析的 message 前缀；
- RE2-OB metrics value-cell missingness 为 0.6408%，有效 timestamp 网格无缺口；
- RE2-OB 90/90 case 的三模态均覆盖 `t0`；±300 s 的三模态平均 30 s bin
  占用率为 99.84%/99.74%/99.47%；
- 两数据集都没有显式静态 service call graph，但 trace 字段允许按 case
  窗口构建动态有向图。

全量产物 SHA-256 为
`957ef5f94f50e096b53e7f0107920361cdb70ad5b0610a87640bbacbc327c6b6`。

## 6. 当前 Gate 解释

context-group 诊断产物为 `artifacts/p1/context_group_diagnostics.json`，
SHA-256 `6dca047930d342bbc7902ca686dd55e8e1cb53cb43a889e9d8f67bb597efec12`。
- G2：adapter 已在本地达到 90/90 可读、0 排除；原始归档与实际消费的 360
  个文件已由 content-addressed snapshot 固定，记为 completed。
- G4：AC@1/3/5、Avg@5、MRR 与非法 ranking toy tests 已通过。
- G1：16,200 event inventory 已全部映射；主 cohort 为 13,470，multi-root
  sensitivity 为 2,730，纳入规则与 flags 已冻结，记为 completed。
- G6：context-aware groups、统一 split validator、实际 5-fold assignment、
  source checksum binding 与正/负向测试均已实现，记为 completed。
- G7：输入/标签/来源物理拆分、metadata 敏感键拦截、opaque RE2 URI、
  checksum、input-only predictor、B1 train-fold-only audit、B2 within-case
  fit scope 与 prediction 敏感字段扫描均已实现，记为 completed。
- G8：全量机器可读产物与诊断报告均已生成，记为 completed。

## 7. 下一步

P1 全产物一致性 closeout 已通过，详见
[P1_REPRODUCIBILITY_AUDIT.md](P1_REPRODUCIBILITY_AUDIT.md)。下一阶段进入
P2 表征研究；本数据审计中的 case、窗口、split 与候选集合保持冻结。

## 8. 官方来源

- https://github.com/phamquiluan/RCAEval
- https://github.com/phamquiluan/RCAEval/blob/main/RCAEval/benchmark/evaluation.py
- https://github.com/CloudWise-OpenSource/GAIA-DataSet
