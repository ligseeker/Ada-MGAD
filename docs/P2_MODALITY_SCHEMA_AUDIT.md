# P2 Log and Trace Schema Audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-19
- Verification Status: UNVERIFIED
- Version Label: p2_modality_schema_audit_v0.1

> 状态：raw schema、L0/T0 纯函数与真实双数据集 smoke completed；全量 pending

## 1. 审计范围

本轮复用 P1 对 349,120,558 行 telemetry 的全量诊断，并只读检查两数据集实际 CSV
headers 与样例值。目标是先区分 raw、预派生和不可比字段，再冻结 Track C 的最低
共同表征。

## 2. Logs

### GAIA

- 10 个 service files，87,974,871 rows，约 18.45 GB；
- header：`datetime, service, message`；
- `datetime` 只有日期，真实毫秒时间必须从 message 的
  `YYYY-MM-DD HH:MM:SS,mmm` 前缀解析；
- severity 位于 pipe-delimited message 内，例如 `INFO`；
- 没有 raw template ID 或已验证的通用 template 字段；
- ±300 s 下 99.975% cases 有任一 log activity，候选服务 presence 均值约 0.965。

### RE2-OB

- 90 个 case-local files，15,053,223 rows；
- header：`time, timestamp, container_name, message, level, req_path, error,
  cluster_id, log_template`；
- `timestamp` 为 ns，`container_name` 直接对应候选服务；
- `cluster_id/log_template` 是数据发布方预派生字段，GAIA 没有对称 raw 字段。

### L0 决策

统一 L0 只使用 raw、无需训练词表的字段：event rate、active-bin fraction、severity
error fraction、message-length mean/scale 及相应 whole/stage contrasts。空事件段是合法
零活动，不等价于 source missing；依赖事件内容的统计在无事件时 mask。

`cluster_id/log_template` 不进入统一 L0。template cardinality/entropy 与 vocabulary
进入 L1；GAIA parser、RE2 vocabulary 和任何 normalization 必须在 outer-train 内拟合，
不能在全数据上先生成再交叉验证。

## 3. Traces

### GAIA

- 10 个 service files，28,681,438 rows，约 8.34 GB；
- header：`timestamp, host_ip, service_name, trace_id, span_id, parent_id,
  start_time, end_time, url, status_code, message`；
- `start_time/end_time` 可得到秒级 duration，P1 全量扫描未发现负 duration；
- status 为 HTTP 风格（样例 200/500）；parent/trace/span/service 字段在 10 个文件
  均具备，可构建动态 parent graph；
- ±300 s 下 99.975% cases 有任一 trace activity，候选服务 presence 均值约 0.928。

### RE2-OB

- 90 个 case-local files，34,461,235 rows；
- header：`time, traceID, spanID, serviceName, methodName, operationName,
  startTimeMillis, startTime, duration, statusCode, parentSpanID`；
- `startTime` 为 µs，`duration` 为 µs；status 为 RPC 风格，0 表示 success；
- `parentSpanID` 缺失 2,112,216 rows，缺失必须作为 root/unmatched 状态保留；
- raw trace entity 只直接出现 checkout/currency/email/frontendservice/payment/
  productcatalog/recommendation 七类，不能把未出现的四个候选服务伪造成有观测。

### T0 与实体映射决策

- 唯一允许的名称 alias 为静态、版本化的 `frontendservice → frontend`；
- alias 来自 raw deployment naming 与合法 candidate set，不使用 root label；
- T0 每服务报告 span rate、error/status fraction、duration median/p90/p99、unique
  trace/operation count、parent coverage，并保存 modality-observed mask；
- RE2 中 ad/cart/redis/shipping 等无 raw trace entity 的候选保留完整 ranking row，
  trace values 为 0、mask=false；
- incoming/outgoing parent edge 与 observed/randomized/identity graph 不混入 T0，留到
  H3 结构实验，避免单模态 trace baseline 已隐式使用图传播。

## 4. 时间与状态统一

| Field | GAIA | RE2-OB | 统一内部单位 |
|---|---|---|---|
| Log time | message prefix，本地 Asia/Shanghai | integer ns | epoch ms |
| Trace time | local datetime，Asia/Shanghai | integer µs / millis helper | epoch ms |
| Duration | end-start | integer µs | seconds float64 |
| Error | parsed severity | non-success level/explicit error | bool + coverage |
| Trace error | HTTP status >=500 | finite RPC status !=0 | bool + status-present mask |

所有区间沿用 P1 半开窗口；禁止同时使用 `timestamp` 文本显示列代替高精度整数时间。

## 5. 真实 smoke 结果

`scripts/extract_p2_event_features_smoke.py` 不读取 labels：GAIA 由全量 telemetry
diagnostics 的 log/trace 共同覆盖区间中点选择一个 main case，并完整流式读取
`dbservice1` 单服务原始文件；RE2 选 1 case，保留全部 11 candidates。

| Bundle | Raw scope | Rows | Features | Observed cells | Manifest SHA-256 |
|---|---|---:|---:|---:|---|
| GAIA L0 | 6,018,741 raw rows / 1,510 window events | 1 | 50 | 50/50 | `cb3e065208224ff476d8b7b876afddd1499ee4ef38a21a77e7633b649cc3c6c7` |
| GAIA T0 | 1,437,954 raw rows / 382 window spans | 1 | 80 | 80/80 | `fabaeea9192b21951ce9b794e130205964b5f9410d75c8d1a66585abbdcea2a2` |
| RE2 L0 | 165,696 case rows / 10 observed entities | 11 | 50 | 468/550 | `e3268ce4758f67878b7beb97141fbeeaaf3f8d062dd77e3714b7dcb3f65712b2` |
| RE2 T0 | 379,084 case rows / 7 observed entities | 11 | 80 | 560/880 | `7a0cdef0b4a68773bceba7d2d27a857198dda751de54d704030e585f654047cf` |

四个 bundles 共 16 个 index/value/mask/manifest 文件连续运行两次，SHA-256
全部一致。RE2 T0 的 560 observed cells 恰为 7 entities × 80 features；其余四个
候选 values=0/mask=false，证明缺失实体未被当成零异常活动。

## 6. 仍待验证

- 30 s log/trace onset 是否具备足够事件覆盖；该结论不能从 metric coverage 外推；
- RE2 trace 的候选缺失对 C0-T macro 指标影响；
- L1 template parser/vocabulary 的 train-fold-only 实现成本；
- parent graph 的 operation/entity alias 是否还需新增，新增必须通过全量未知实体审计。

下一步把 reader 改为全量流式/可恢复实现并扫描全部 logs/traces；不得复用 RE2 已
派生 `log_template` 作为 GAIA raw L0 的替代，也不能仅凭单 case smoke 宣称总体
coverage 足够。
