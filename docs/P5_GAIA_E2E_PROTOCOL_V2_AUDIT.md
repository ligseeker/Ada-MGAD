# P5 GAIA 两阶段端到端协议 V2 审计报告

- 审计轮次:P5-V2 protocol & raw-data audit(只审计,不训练,不改模型)
- 审计日期:2026-09-13
- 审计依据:`docs/GAIA两阶段端到端原始数据与标签处理方案V2.md`(下称 V2 方案)
- 审计 worktree:`/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2`(branch `e2e-v2`,base `origin/e2e` @ `0ee8b060cab0e79c44997ff6b1fbaf0432ef4046`,建立时干净)
- 旧 worktree `/home/zhangll24/RCA_project/Ada-MGAD-e2e`(branch `e2e` @ 0ee8b06)仅作只读参考,未修改;其未提交的 `artifacts/p5/i1/` formal 结果被用作现状性能证据。

---

## 0. 结论速览

**总判定:GO(有条件)** —— 数据与协议层的审计全部完成,GT taxonomy、时间语义、70/30 boundary、标签规则、无 Validation 训练协议均已给出可冻结结论;在按第 6 节最小修改清单完成 MUST FIX 项之前,不得启动 formal 重跑。当前 E2E F1 低(Diagnosis F1@1 = 0.2847)的主因**不是 GT 定义错误**,而是:事件检测 Test Recall 崩塌(0.1903,θ 在 Validation 上校准后迁移失效)+ 训练协议漂移(实际 epochs=50/patience=5、12 epoch 早停,与冻结 config 120/7 不一致)+ 日志表示降维(6 维 level 统计,丢失 main 的 Drain3 模板信息,而 96% 的 GT 是 11s Login Failure 且其中 68% 在故障区间内没有任何 metric 观测)。

### 11 个问题一句话结论

| # | 问题 | 结论 | 标记 |
|---|------|------|------|
| 1 | 16200 是否正确 | 数值正确:从原始 message 独立重推导得 N_GT=16200(五类),与预解析列 0 不一致;但 `expected_events: 16200` 硬编码断言必须删除,N_GT 只能是输出 | KEEP(taxonomy)/ MUST FIX(去硬编码) |
| 2 | normal memory freed 是否进 GT | 应纳入:"前次注入结束标记"假说被证据否定(最小间隔 29.4 分钟,中位 7.6 小时);Ada-MGAD main 与 DiagFusion 均将其作为独立故障(start=message ts, duration=600s);纳入后 N_GT=16217 | SHOULD FIX(纳入为第 6 类)/ OPEN QUESTION(物理语义命名) |
| 3 | 38 条 ERROR 如何处理 | 全部为 `log_upload_failure`(pymysql 1040 连接数超限的运维上传失败),另有 36 行是其 traceback 续行碎片;无注入语义,全部排除 GT;当前 e2e 的排除处理正确 | KEEP(排除) |
| 4 | 最终 taxonomy 与数量 | 6 类 16217(推荐):login 15478 / memory 652 / file_moving 43 / freed 17 / access 15 / cpu 12;排除 952 行(INFO 861 + ERROR 38 + traceback 碎片 36 + injection-failed 10 已含于 INFO 861 内,见 §3.4) | 冻结于本报告 |
| 5 | 70/30 boundary | 必须由 telemetry timeline 决定(与 GT 无关):主口径 metric_only(与 main 的 metric-ts 网格定义一致),N=87120,K=60984,**T_split = 1626963120000 = 2021-07-22 22:12:00 CST**,比例精确 0.700/0.300;废除 60/20/20 与 Validation | MUST FIX |
| 6 | 63 metric features 是否取消固定 | 是:63 是"90 共享 schema → Train-only 质量过滤"的输出;四阈值与 ffill(limit=10) 均为 main 原生规则(非 e2e 新增 heuristic),规则本身 KEEP,数字与 schema 必须重新在新 Train 段 fit 并输出 `metric_feature_schema.json` | SHOULD FIX |
| 7 | Logs 是否恢复 Drain/template | 是:main 的 GAIA 日志表示 = Drain3 模板计数 + 5 level 计数 + log_total(e2e 的 6 维只是其真子块);应恢复模板特征并改为 Train-only fit 词表 + Test 冻结 transform + 新模板 UNK | SHOULD FIX(强烈) |
| 8 | AD interval labels 是否正确 | 规则错误、影响极小:当前 floor 双闭区间在 end 恰落 bin 边界时多标 1 bin(全库仅 1 个事件、37862 vs 37861 个 positive label 差 1);必须改为半开 overlap 规则 `b<end ∧ b+30s>start` | MUST FIX(规则)/影响已量化 |
| 9 | t_hat 用 bin start 还是 bin end | bin end(prediction_available_time):当前用 bin start,导致 Test 匹配 delay 中位数 −3.87s(预测"早于"故障发生,物理不可能);同时匹配必须改为因果约束 `0 ≤ t_hat − gt_start ≤ 60s` | MUST FIX |
| 10 | 无 Validation 后 checkpoint/threshold | 固定 epochs=120、final checkpoint(禁止 main 的 best-F1-on-Test 选模);θ 在 Train predictions + Train Event GT 上精确枚举后冻结;rec-score median/MAD 校准必须 Train-fit(当前按被评估集合各自计算,违反 V2 §34);Test 只评估一次 | MUST FIX |
| 11 | 低 F1 最可能的解释 | 排序:① θ Val→Test 迁移失效 + 分数分布漂移(Test P=0.9973/R=0.1903,操作点极端保守);② 训练协议漂移(50/5 vs 冻结 120/7,best@6,共 12 epoch,config sha 不一致);③ 6 维日志表示对 11s login(68% 无 metric 观测)信息不足;④ t_hat/匹配语义失真(次要);⑤ RCA 阶段不是瓶颈(matched AC@1: oracle 0.931 / detected 0.891,但 detector-only 0.965 反超,为负增益,另案处理) | 见 §5 |

---

## 1. 审计范围与数据溯源

### 1.1 仓库与 commit

| 仓库 | 路径 | commit | 用途 |
|---|---|---|---|
| Ada-MGAD e2e(审计对象) | Ada-MGAD-e2e-v2 | `0ee8b06`(= origin/e2e) | 当前 E2E pipeline |
| Ada-MGAD main(原方法) | Ada-MGAD | `ab31282`(= origin/main) | 原始数据语义/训练协议对照 |
| Ada-RCA | Ada-RCA | `6c3abaa`(exp/p6-baselines) | 表示冻结对照(见 §3.8 与附录 D) |
| DiagFusion | DiagFusion | `9ec3f73` | GAIA case taxonomy 对照 |
| GAIA 原始数据 | datasets/GAIA/MicroSS | 2021-08-23 落盘 | metric 4.5G/6640 文件、business ~18G/10 文件、trace ~8G/10 文件、run table |

### 1.2 run table 溯源(重要)

- 原始 `run/run/run/run_table_2021-07.csv` **已不在磁盘上**,其 sha256(`ca8d44dd…2820`)仅存于 `artifacts/p5/i1/protocol_manifest.json`,现已不可验证。
- 本轮审计对象为数据所有者解析产生的 `run_table_2021-07_fixed.csv`(sha256 `ba5848fa…234c`,17152 数据行,列 `datetime,service,message,timestamp,level,service_node,anomaly_type,duration_seconds`)。所有者确认(2026-09-13):**前三列为原始字段,`message` 列保存完整原始日志行**(`<ts>,<ms> | LEVEL | node_ip | container_ip | service | payload`);后五列为预解析断言。
- 本审计因此**全部从 message 原文独立重推导**(类型、level、时间语义、duration),再与预解析列交叉验证:**17152 行中可比较行 17116 行,0 不一致**(36 行 traceback 碎片无可比列;`preparse_crosscheck.json`)。GT 结论的证据基础是原始文本,不是预解析列。
- 遗留风险(OPEN QUESTION,provenance):原始文件缺失导致"预解析是否丢行/改行"无法与官方 GAIA-DataSet 逐字节比对;建议从官方仓库重新下载一份原始 run table 存档比对(`run_table_2021-07.csv` 行数应 ≥17152−36 合并的多行记录)。

### 1.3 审计方法

1. 六个专项审计(GT taxonomy、telemetry 覆盖/timeline、main 分支、e2e 分支、DiagFusion、Ada-RCA)并行执行,全部结论带 file:line 或 artifact 证据。
2. 所有机器可读 artifact 写入 `data/p5/v2/audit/`(脚本在 `scripts/p5_v2/`),汇总镜像至 `artifacts/p5/v2_audit/` 提交。
3. telemetry 全量计数(trace status、business level、行数)与仓库内已提交的 G0R2 历史全量扫描 `artifacts/p5/g0r2/raw_telemetry_timing.json` 逐项核对一致(24453030/3040760/1156668/30980/28681438/87974871/4228408 全部命中),metric 端点另经 head/tail 抽查(附录 E)。

---

## 2. V2 §44 审计清单完成度

| 要求 | 状态 | 所在 artifact |
|---|---|---|
| raw run-table 行数 / 每种 message type 数量 | ✅ 17152 行,11 种 raw_type | `taxonomy_summary.json`(pattern inventory) |
| GT included/excluded taxonomy | ✅ | `run_record_registry.csv`(semantic_role/gt_included/decision_reason 逐行) |
| 17 条 normal memory freed 审计 | ✅ 逐条 + 双口径间隔 | `normal_memory_freed_audit.json` |
| 38 条 ERROR 逐类审计 | ✅ 逐条(+36 碎片溯源) | `error_records_audit.json` |
| 12 条 CPU 解析验证 | ✅ 逐条,0 失败 | `cpu_anomaly_audit.json` |
| 最终 N_GT / fault 分布 / service 分布 | ✅ 16200 / 16217 两口径 | `taxonomy_summary.json` |
| raw telemetry coverage | ✅(抽样项已标注) | `telemetry_coverage.json` |
| 30s detector timeline | ✅ 两口径 | `detector_timeline.json` |
| 70/30 split timestamp | ✅ 1626963120000 | 同上 |
| Train/Test 事件数 / boundary crossing / same-bin 冲突 | ✅ | `split_audit.json` |
| metric candidate schema / Train-selected schema / F_metric | ⚠️ 部分:现行 90→63(i1 manifest);**新 split 下必须重新 fit,数字允许变化**(V2 §16) | 跟进项(§6 清单第 7 条) |
| Train/Test metric missingness | ⚠️ 部分:覆盖级红旗已量化;逐特征 missingness artifact 随预处理重跑生成 | `telemetry_coverage.json` + 跟进项 |
| log representation audit | ✅ main=Drain3 模板+6 统计;e2e=仅 6 统计 | 本报告 §3.5 |
| trace status distribution | ✅ 全量 | `telemetry_coverage.json` |
| RCA context contamination | ✅ oracle 口径全量逐 case;detected 口径需重训后生成(N/A 本轮) | `rca_contamination.json` + 逐 case csv |

---

## 3. 逐问题审计结论

### 3.1 Q1:16200 是否正确 —— 数值正确;硬编码必须删除

**独立重推导结果**(`scripts/p5_v2/audit_gt/audit_gt_taxonomy.py`,无任何期望值硬编码):

| raw_type | 行数 | 处置 |
|---|---|---|
| login_failure | 15478 | GT |
| memory_anomalies | 652 | GT |
| file_moving | 43 | GT |
| access_permission_denied | 15 | GT |
| cpu_anomalies | 12 | GT |
| **N_GT(主口径)** | **16200** | |
| normal_memory_freed | 17 | 候选(§3.2) |
| log_upload_failure(level=ERROR) | 38 | 排除(§3.3) |
| traceback_continuation | 36 | 排除(§3.3) |
| exception_injection_failed(level=INFO) | 10 | 排除:注入尝试自身报错,未形成故障 |
| upload_success / unknown(INFO 运维记录) | 832 / 19 | 排除:非故障系统记录(V2 §8) |
| 合计 | 17152 | level 分布:WARNING 16217 = 16200+17;INFO 861;ERROR 38;碎片 36 |

五类计数与旧假设(15478/652/43/15/12)**逐项一致**,但这是解析输出而非输入。与所有者预解析列交叉验证 **0 不一致**。

**MUST FIX**:`configs/e2e/gaia_p5_v1.yaml:12` `"expected_events": 16200` 与 `src/e2e/protocol.py:188` 的行数断言违反 V2 §5(taxonomy 变更后断言会把正确数据判为错误)。删除断言,改为把 N_GT 写入 artifact。

**标记:KEEP(taxonomy 数值)/ MUST FIX(去硬编码)。**

### 3.2 Q2:normal memory freed 应纳入 GT(第 6 类,N_GT=16217)

17 条记录逐条审计(`normal_memory_freed_audit.json`):

1. **payload 完全同构**:`[normal memory freed label] lasts ten minutes`,WARNING 级,10 服务均现;duration 词 "ten minutes"→600s,与预解析列一致。
2. **"前次高内存注入的结束标记"假说被否定**:每条与同 service 最近一次在前的 memory_anomalies 的间隔(以注入真实 end 计)min=1761.8s(29.4 分钟)、median=27469.6s(7.6 小时)、max=149909s(41.6 小时)。若是结束标记,间隔应≈0。
3. **Ada-MGAD main 原生将其标为 positive**:`util/GAIA/pre_GAIA.py:190-196` 解析为 start=message ts、duration=600s;标签过滤(:302-306)只排除 `[normal]`,`[normal memory freed label] ≠ [normal]`,故 main 的 GAIA 标签网格包含这 17 个事件。**当前 e2e 将其排除(recovery_marker_candidate)反而偏离了原方法。**
4. **DiagFusion 将其作为独立故障类**(产物级证据,DiagFusion@9ec3f73):`gaia_resplit.csv`(1099 case)含 16/17 条(缺的 1 条只因全局丢弃 07-04 之前记录);类别索引由 `sorted(set(anomaly_type))` 动态生成,`[normal memory freed label]`=类 4,分类头 `N_A=5`(gaia_config.yaml:50);已提交的训练/预测产物中该类真实出现(train 3 例、test 13 例、预测 CSV GT=4 共 13 次)。其时间语义与本审计一致:st=message ts,duration=600s,ed=st+600s。
5. **"Free using memory" 名称在 DiagFusion 仓库中零命中**——该名称只存在于论文转述;V2 §6 的候选解释(start=message ts, duration=600s)与 main、DiagFusion 数据完全一致,予以确认。
6. 为什么 DiagFusion 数量与 17 不同:它使用的是 07-04 之后、大幅降采样的 1099-case demo 子集(login 527/15478),并非全量 run table。

**结论:SHOULD FIX —— 纳入为第 6 类 `normal_memory_freed`(1 raw injection = 1 event,start=message ts,duration=600s),N_GT=16217;纳入/排除两口径的 split 统计均已产出(§3.5 表)。** 残余 OPEN QUESTION:该记录的物理语义(注入工具触发的"内存释放"程序本身是否算服务异常)无法从数据内部完全证实;建议正式实验中按 fault type 分层报告,使结论对该口径不敏感。

### 3.3 Q3:38 条 ERROR 全部排除;另发现 36 行 traceback 碎片

- 38 条 ERROR **全部**是同一模式:`upload business/trace/run_logs logs on <date> failed: (pymysql.err.OperationalError) (1040, 'ny connections')` + SQLAlchemy 错误链接,集中于 logservice1/2、dbservice1/2 的凌晨上传窗口(如 03:50)。点事件、无注入措辞、无 duration、无 start-at → `gt_included=false, reason=operational log-upload failure`(V2 §7:semantics_not_established)。逐条明细在 `error_records_audit.json`;level 列与 message 内 LEVEL 0 不一致。
- **新发现**:run table 中 36 行 `(Background on this error at: http://sqlalche.me/e/14/e3q8)` 是上述 ERROR 的多行 traceback 续行,在生成 CSV 时被拆成独立行(父行相邻、service 匹配 36/36、预解析列全空)。旧 G0R2 将其计入 "unsupported 36"。它们不是事件,不入 GT;`run_record_registry.csv` 中以 `traceback_continuation/parse_error` 保留。
- 对照:main 的 `pre_GAIA.py:159-164` 会把 ERROR 变成 duration=0 的点事件并写入正标签(st=ed floor 后闭区间仍标 1 个 bin)——这正是 V2 §7 明令禁止的行为;**当前 e2e 的排除处理正确,KEEP**。同时注意:main 的历史 GAIA 结果因此包含 38 个 ERROR 正 bin + 17 个 freed 事件,与 e2e/V2 口径不可直接比较。
- 10 条 `exception_injection_failed`(INFO,`get a error: 'NoneType'/'Redis' object is not callable`):注入尝试自身失败,未形成故障区间,排除并保留 pattern(V2 §8)。

**标记:KEEP(排除),碎片行处理已文档化。**

### 3.4 Q4:最终 GT taxonomy(冻结文本)

**per-type 时间语义**(全部从原文验证,`taxonomy_summary.json.per_type_semantics`):

| fault_type | start 来源 | duration | message 打出时刻 | 证据 |
|---|---|---|---|---|
| login_failure | message ts | 11s(`wait for 11 seconds`,15478 条全部数值型) | 故障开始 | offset 无内嵌 start |
| memory_anomalies | payload `start at <µs ts>` | 600s | **故障结束**(msg−start median 600.9s) | offset 分布 |
| file_moving | payload `start with <ts>` | 600s(`last for`) | **故障结束**(offset ~600.1s) | offset 分布 |
| cpu_anomalies | payload `start at <µs ts>` | 小数秒 2.892~3.24s(正则 `\d+(?:\.\d+)?`,12/12 解析成功,0 精度丢失) | 故障开始(offset median 0.0s) | `cpu_anomaly_audit.json` |
| access_permission_denied | message ts | 3600s(`will lasts an hour`) | 故障开始 | 15/15 |
| normal_memory_freed | message ts | 600s(`lasts ten minutes`) | 故障开始(采纳口径) | §3.2 |

- 1 injection = 1 event;不按 duration 拆分;3s CPU 与 3600s access 同为 1 个 Event GT(V2 §10/§26)✅ 当前 e2e registry 已遵守,KEEP。
- labelled service 分布(主口径):mobservice1 7815 / mobservice2 7799(login 占 96%,集中于 mob 两服务)/ dbservice2 96 / webservice2 84 / dbservice1 83 / webservice1 82 / logservice1 66 / logservice2 63 / redisservice2 57 / redisservice1 55。
- GT service 统一表述为 **labelled injected service**(V2 §11),不得宣称 independently validated causal root cause。
- 排除侧完整账目:17152 = 16200 GT + 17 freed(候选)+ 38 ERROR + 36 碎片 + 10 injection-failed + 832 upload_success + 19 INFO 运维记录。

### 3.5 Q5:70/30 boundary 必须由 telemetry timeline 决定

**当前实现违规**(证据:`artifacts/p5/g0r2/split_feasibility.json` start_min_ms/end_max_ms 与 GT registry min/max 完全相等;`configs/e2e/gaia_p5_v1.yaml` split 节;`src/e2e/protocol.py:162-173`):60/20/20 三段、含正式 Validation、boundary=GT 时间跨度的 3/5 与 4/5(b1=07-19 18:50:40、b2=07-25 21:25:16 CST)——split 依赖 GT taxonomy,违反 V2 §12/§13。

**V2 口径**(GT 无关,`detector_timeline.json`):

| 口径 | timeline | N bins | K | T_split | Train/Test 比例 |
|---|---|---|---|---|---|
| **metric_only(推荐主口径)** | 07-01 18:00:00 → 08-01 00:00:00 CST | 87120 | 60984 | **1626963120000 = 07-22 22:12:00 CST** | 0.700000 / 0.300000 |
| union(敏感性) | 07-01 09:57:00 → 08-01 00:00:00 | 88086 | 61660 | 1626954420000 = 07-22 19:47:00 | 0.699998 / 0.300002 |

- 主口径与 **Ada-MGAD main 的真实 timeline 定义一致**(main 的 30s 网格由 metric 时间戳 floor 对齐生成,`pre_GAIA.py:283-295`),满足 V2 §13"优先保持 main 的 timeline 定义"。
- K 的计算必须用精确整数算术 `(7*N)//10`:IEEE754 下 `int(0.7*87120)=60983`(0.7×87120=60983.999…),会把 boundary 错移 30s(附录 B)。N±1 端点变体下 split 不变(`split_unchanged=true`)。
- **事件分配**(`split_audit.json`,metric_only):

| taxonomy | Train | Test | 跨界 purge | timeline 外 |
|---|---|---|---|---|
| 主口径 5 类 | 10332(login 9857/memory 438/file 27/access 10) | 5783(login 5546/memory 204/file 16/access 5/**cpu 12**) | 1(memory,webservice1,07-22 22:05:46–22:15:46) | 84(login 75+memory 9,均在 07-01 11:06–18:00,早于 metric 起点) |
| 含 freed 6 类 | 10344(+12) | 5787(+4) | 1 | 85(+1 freed) |

- 跨界事件按 V2 §28 purge(Event/RCA 层)并生成 `boundary_exclusion_mask`(17 个 bin);union 口径下跨界=0、timeline 外=0,但 Train 头部 826 bins(~6h53m)无任何 metric 观测(会被 NaN→0 填充伪特征)。**推荐 metric_only + 显式报告 84 个 timeline-excluded 事件**(占 GT 0.52%,全部在 7 月 1 日白天;logs/traces 存在,metric 不存在——属真实 observability 边界,不得为纳入它们而人为前移 timeline 起点)。
- 漂移事实(不得据此移动 split,V2 §13):12 个 cpu_anomalies 全部位于 07-27 23:46 → 07-31 20:00,无论新旧 split 都整体落在 Test;Train 校准永远看不到 cpu 类。必须在结果中按 fault type 分层报告,并说明 macro 平均的该结构限制。
- Test 段覆盖完整性:business/trace 全 service 每日有数据,无日级空洞;metric 除 8 个 half-2 文件在 split 前早停造成的特征级空洞外完整(红旗清单见 `telemetry_coverage.json`)。

**MUST FIX 汇总**:改 split 派生(telemetry timeline)、改 70/30、删 Validation、跨界/timeline 外 purge 自动化、boundary_exclusion_mask 落地。

### 3.6 Q6:取消"63 特征"固定;质量过滤规则确认为 main 原生

- **四阈值是 main 原生规则,不是 e2e 新增 heuristic**:`util/GAIA/pre_GAIA.py:853-855`(coverage≥0.20、dynamic_ratio≥1e-4、unique≥2)+ `:874`(dynamic_span≥1e-10)。V2 §17 的疑问就此关闭:**KEEP 规则**。
- ffill(limit=10) 同为 main 原生(`pre_GAIA.py:920-922`);e2e 的按-split 填充(`ad_preprocess.py:151-155`)额外保证了不跨 Train/Test 边界填充,符合 V2 §19,**KEEP**。
- main 无 "63" 硬编码(全仓 git grep 零命中;唯一历史数字是死代码注释里的 76)。63 = e2e 当前"90 个 10-service 共享 schema 特征 → Train-only 四阈值过滤(须 10 服务全过)"的输出(`ad_data_manifest.json:3326→:7`)。**协议应冻结 selection rule + `metric_feature_schema.json`(feature_names/count/规则/Train 统计/checksum),数字是输出**(V2 §16)。新 split 下 Train 段变化,63 允许变为 62/64…,不是协议错误。
- 与 main 的合法偏离:main 的 min-max 在整月全量上 fit(transductive,`pre_GAIA.py:951-956`);e2e 改为 Train-only(`ad_preprocess.py:242-252`)——这是 V2 §15 防火墙的要求,**KEEP e2e 做法**,并在论文中说明与 historical main 结果的可比性限制。
- 新红旗(来自覆盖审计,影响 schema 冻结):system 组件 metric 07-05 14:37 才开始(Train 头部 ~11114 bins 无 host 特征);10 个 header-only 空文件;socket_port/cgroup-limit 家族 ~20 个文件早停/晚启/重复 ts/全 0,其中 8 个 half-2 文件在 split 前终止 → Test 段特征级空洞,冻结 schema 后按 missing 规则(ffill→0)处理并写入 missingness artifact;全局采样缺口率 16.18%、重复时间戳行 21.75M(聚合规则=同 bin mean,与 main `data_GAIA.py:133-152` 一致)。

**标记:SHOULD FIX(schema artifact 化 + 文字表述去"63")/ KEEP(规则本身)。**

### 3.7 Q7:Logs 应恢复 main 的 Drain3 模板表示(Train-only fit)

- **main 的真实日志语义**(`pre_GAIA.py:372-498`):Drain3 TemplateMiner(gaia.ini:sim_th=0.3, depth=5, masking NUM/IP/HEX/SEQ/URL/UUID),30s bin 特征 = `template_<id>` 计数 + `level_INFO/WARNING/ERROR/DEBUG/UNKNOWN` 计数 + `log_total`,另存 `log_templates.tsv` 与 `log_state.pkl`;数据源同为 business log。词表在整月全量上单次 fit(transductive)。
- **e2e 现状**:仅 6 维 level 统计(`ad_preprocess.py:49-52`),manifest 自认 `adapter_compromise … no full-month Drain3 template learning`(:396)。6 维是 main 布局的真子块,模板主特征全部丢失。
- **信息损失评估**:GT 的 96% 是 11s login failure,其故障区间内 metric 可观测率仅 31.7%(`temporal_observability.csv`:login 4913/15478;cpu 0/12;memory 571/652;file 34/43;freed 14/17;总体 5546/16217=34.2%)——短故障的 AD 证据主要在 **logs/trace 通道**。level 计数把"哪条模板出现"抹平为 5 个数字,对 11s 级故障的判别信息显著弱于模板计数;business 日志中 ERROR 806,735 条(db/web 各 ~18 万)是可用信号(V2 §21:必须保留,当前 e2e 已保留 ERROR 计数 ✅)。
- **V2 合规改造方向**(§20):Train 日志 fit Drain 词表 → 冻结 parser/词表 → Train/Test 统一 transform → Test 新模板 UNK,不更新词表。这是相对 main 的合法偏离(main 全月 fit 是 transductive,V2 §15 禁止),须在论文说明。
- 实施风险提示:Train 段 business 日志 ~6000 万条,Drain3 全量 fit 计算量大(main 曾整月跑通一次,说明可行但慢);允许分块/并行,但词表 fit 只准见 Train。

**标记:SHOULD FIX(强烈;这是 recall 侧最可能的信息增益点,但属表示恢复而非模型重设计)。** 是否逐字节恢复 main 的 feature layout(模板 id 排序等)由实现阶段兼容性审计定,协议只冻结"模板计数 + 5 level + log_total,Train-only 词表"。

### 3.8 Q8/Q9:AD 标签规则、窗口、t_hat 与匹配

**AD 标签(§23-27)**:
- 当前实现 `ad_data.py:48-53`:`first_bin=floor(start)`,`last_bin=floor(end)`,searchsorted 右开取到 floor(end) —— 即 **floor 双闭区间**,非半开 overlap。差异仅在 `end_ms % 30000 == 0` 时多标 1 bin:全库仅 1 个事件(memory,gaia-v2-12447),positive (service,bin) 标签总数 37862(双闭)vs 37861(半开)。**MUST FIX 为 `b_k < end ∧ b_k+30s > start`**(规则正确性;数值影响已量化为 1 bin)。同时消除现存语义分裂:注册表 overlap 分析用半开(`revalidate_gaia_events.py:270`)而标注用双闭,G0R2 自己已记录该不一致(`label_reconciliation.json:62,118`)。
- 短故障:11s login 落单 bin → 1 个 positive bin;跨 bin 短故障 → 2 个(§25)✅;3s cpu 即使区间内 0 metric 观测仍是 positive(§24)——观测缺口已量化(cpu 0/12 可观测),标签不看观测,KEEP。
- 同 bin 多注入:按 service 列布尔 OR,事件不合并;start-bin 口径冲突统计:1003 bins 多事件(max 3)、483 bins 同 service 多事件(max 3)、536 bins 多 service(max 3);Test 段 404/202。旧口径(positive-bin 级、旧 split)在 `g0r2/label_reconciliation.json`。detector 时间分辨率上限按 V2 §27 用这两组数字报告。KEEP。
- 窗口:10×30s、stride=1、只预测最后一个 bin(`ad_data.py:118-133` target=end−1)✅ 与 main 一致(`data_GAIA.py:433-435,460-463`);split 内独立构窗,无 9+1 跨 split 窗口(`ad_data.py:200-206`),每段头部 9 个 target bin burn-in ✅(V2 §29/30 KEEP;main 的窗口列表按索引 70/30 切分在边界两侧存在 9/10 重叠窗口的共享,`runtime.py:55-57`——e2e 的隔离做法比 main 更严格,是 V2 要求的方向,KEEP)。

**t_hat 与匹配(§31/35)——MUST FIX**:
- 当前 `t_hat = prediction_timestamp = target bin start`(`event_detection.py:168`、`ad_data.py:142-144`)。模型观测完 `[b,b+30s)` 才能出预测,V2 §31 要求 `t_hat = prediction_available_time = b+30s`。
- 当前匹配 ±60s **对称、非因果**(`event_detection.py:235-236,261-272`),允许预测早于 GT;实测 Test delay mean −4.10s / median −3.87s(event_detection_metrics.json),`event_matching.csv` 存在 delay=−2.092s 等——bin-start t_hat + 非因果匹配共同制造了"预测早于故障"的假象。V2 §35:改为 `0 ≤ t_hat − gt_start ≤ 60s` + 确定性一对一匹配(现贪心一对一机制可保留,候选窗改因果)。
- 修正后典型 login 的 delay 将变为 +19~30s(GT 落 bin b → 可用时刻 b+30s),仍在 60s 容差内;tolerance=60s 本身与 30s 观测延迟相容,KEEP 60s(已按 V2 §35 检查:episode 定义=连续 positive bin 段取首个,DiagFusion 不做事件级 delay 匹配,历史实现为 ±60s)。
- oracle/detected anchor 语义必须统一说明:oracle=GT 精确 start(ms),detected=t_hat(bin end)。当前 i1 的 `anchor_degradation`(detected−oracle:AC@1 −0.0404)混入了 floor 栅格化偏移(−4s 级),修正后才可解释为纯检测质量差。

**标记:标签规则 MUST FIX(1 bin 影响)/ 窗口与冲突处理 KEEP / t_hat+因果匹配 MUST FIX。**

### 3.9 Q10:无 Validation 的训练协议(最简、无 Test 泄漏)

现状与违规点:
1. checkpoint:`validation_f1_only` best-F1 + validation 早停(`util/train.py:122-138,267-287`)——Validation 取消后不可用;**main 的 f1_stage checkpoint 直接在 Test 上选 best-F1(`train.py:224-241`),是 V2 §32 明令禁止的泄漏,不得回退到它**。
2. 协议漂移(可复现性断裂,MUST FIX 流程问题):formal i1 运行 `config_sha256=0a192c…`、`model_args epochs=50/patience=5`,实际 12 epoch 早停、best@6(val node F1 0.5536);而仓库冻结的 `gaia_p5_v1.yaml`/`protocol_manifest.json` 是 `9191c3…`、`epochs=120/patience=7`。**报告数字所用协议 ≠ 仓库冻结协议。**
3. rec-score 校准:`train.py:284-288` 的 median/MAD 在**被评估集合自身**上计算(评估 Test 时用 Test 分布),违反 V2 §34;且节点分数实际为 `0.7*cls+0.3*rec_prob` 融合(alpha=0.7,继承 main),而 `docs/P5_GAIA_E2E_IMPLEMENTATION_V1.md:113-114` 声称触发"no score fusion"——文档与代码冲突,SHOULD FIX(改文档,或按文档改代码;建议保留融合、改文档,因为融合是 main 原生语义)。

**冻结的 V2 合规协议**(继承 main 超参,不用 Test/Val 做任何选择):
- 超参:main 的 GAIA 默认(`runtime_config.py:66-79`:epochs=120, lr=1e-3 AdaBelief, StepLR(30,0.5), weight_decay=5e-4, abnormal_weight=96, label_percent=0.5, label_weight=1e-2, seed=42)。
- checkpoint:**固定 120 epochs,取 final checkpoint**;不 early stop(main 的 train-loss 早停 patience=7 不消费任何 held-out 数据,可作为备选保留,但 formal 协议选"固定 epoch+final"最简且无可争议)。
- 分数校准:rec-score 的 median/MAD 与融合所需的一切统计**只在 Train 预测分数上 fit 一次**,冻结后 Test transform-only(V2 §34)。
- event threshold θ:在 **Train predictions + Train Event GT** 上精确枚举唯一分数、最大化 Train Event F1(平局取最高 θ,沿用现 `select_validation_threshold` 的确定性机制改数据源),冻结;Test 只用该 θ 评估一次(V2 §33)。可选补充:Train 内部 blocked/OOF 校准敏感性实验(不构成第三个 split)。
- 流程:run_manifest 的 config_sha256 必须与仓库冻结 config 一致(单一事实源),epochs/patience 等 model_args 从冻结 config 读取,禁止运行期覆盖。

**标记:MUST FIX(checkpoint/θ/校准/流程四项)。**

### 3.10 Q8(RCA 侧):Ada-RCA 接口审计

- **canonical Ada-RCA 仓库与 GAIA 无关**(附录 D):其冻结表示是 RCAEval RE2 上的 **W600-B15(80 bins)**,oracle-only anchor(inject_time.txt,epoch 秒),detector 集成/GAIA 明文 out of scope(`docs/README.md:18`、`LEGACY_ASSET_MAP.md:14` DO_NOT_PORT)。E2E 使用的 W300-B15 68D 是 **e2e 仓库内的拷贝再参数化**(`src/e2e/rca_features.py`,SOURCE_COMMIT a2c6209,与当前 Ada-RCA HEAD 6c3abaa 存在版本漂移)。V2 §38 冻结的 W300-B15/68D Z2/Conditional Logit 以该拷贝为准,**KEEP**,但必须在 protocol manifest 里钉死 SOURCE_COMMIT 并说明与 canonical 仓库的关系(OPEN QUESTION:是否将 GAIA adapter 回合进 Ada-RCA 仓库治理)。
- 68D=4 通道×(8 base+9 morphology),Z2 三层归一化全部 case-local/pre-event-only 或 Train-only fit(e2e:`rca_model.py:261` StandardScaler 只在 Train oracle candidate 行 fit ✅;Ada-RCA 同源 `p4.py:146-148`)。**KEEP**。
- Trace Error = `status != 200`(`gaia_rca_adapter.py:166`)✅ V2 §40。全量分布:200=24,453,030 / **300=3,040,760(仅 redis)** / 400=30,980(仅 redis)/ 500=1,156,668;非 200 合计 4,228,408(14.74%)。redis 的 300 占其 trace ~12.5%,`!=200` 把它们计为 error——规则已冻结(KEEP),但 300 的业务语义未验证,列为 OPEN QUESTION(建议后续做 !=200 vs >=500 的敏感性分层报告,不改冻结规则)。
- **表示层已知限制(必须写进论文,不属本轮修改)**:W300-B15 的 onset 特征要求相邻两个 15s bin 均有观测且 ≥3.0(`rca_features.py:202-211` 继承 Ada-RCA `features.py:142-147`);GAIA metric 原生 30s 采样 → metric 通道相邻 15s bin 几乎不可能同时有观测,`onset_seconds` 基本恒为 sentinel(300)、`onset_missing` 恒 1,68D 中 2 个字段在 metric 通道结构性失效(log/trace 通道 15s 派生序列不受影响)。15s bin 无观测 = missing(不复制、不插值)✅ V2 §39——e2e adapter 未做 30s→15s 复制/插值,KEEP。
- context 防火墙:W300 越界 purge 已实现(`protocol.py:224-253`;i1 实测 purge 2 例、detected crossing 0 例);新 split 下 oracle_boundary_purge = Train 2 / Test 5,oracle context 超出 timeline = Train 87 / Test 2(`rca_contamination.json`;新 boundary 下需重跑 pipeline 复核 detected 口径)。
- **contamination(V2 §42)首次全量量化**(oracle 口径,主 taxonomy):Test 5783 case 中 clean-context 仅 46(0.8%),contaminated 5737;foreign_event_count p50=4 / p90=7 / p99=9 / max=13;Train clean 234/10332。GAIA 注入密度决定了污染是常态——**不得删污染 case 提性能**,正式结果按 clean/contaminated 两个 strata 分层报告;逐 case 计数已产出(`rca_contamination_cases_primary.csv`),pipeline 侧应把该统计内置为输出(SHOULD FIX)。
- RCA 不消费 Ada-MGAD npy,只接 anchor 后从原始 telemetry 重建特征(`gaia_rca_adapter.py` 直读 business/trace/metric 原始 csv)✅ V2 §36,KEEP。

### 3.11 Q11 见 §5。

---

## 4. 本轮可冻结 / 仍开放(V2 §46 更新)

**现在可以冻结(数值即结果)**:
- GT taxonomy v2-audit-20260913:6 类 / N_GT=16217(含 freed;主口径敏感性 5 类 / 16200 并存报告);1 injection = 1 event;per-type 时间语义表(§3.4)。
- detector timeline:metric_only,`[1625133600000, 1627747200000)`,N=87120,K=60984,**T_split=1626963120000(2021-07-22 22:12:00 CST)**,精确 70/30;union 口径仅作敏感性。
- 事件账目:Train 10332 / Test 5783(主口径);跨界 purge 1 + mask 17 bins;timeline 外 84;burn-in 每段 9 target bins。
- AD 规则:半开 overlap 标注;10×30s 窗、预测最后 bin、split 内构窗;同 bin OR 不合并事件。
- t_hat = target bin end;因果匹配 `0 ≤ t_hat − gt_start ≤ 60s`,确定性一对一。
- 训练协议:main 超参 + epochs=120 + final checkpoint;θ/校准/一切统计 Train-only fit;Test 单次评估。
- metric 质量过滤四阈值 + ffill(limit=10) + 同 bin mean 聚合(main 原生规则,Train-only fit)。
- RCA:W300-B15/68D Z2/Conditional Logit/`status!=200`/15s missing 不插值/oracle 与 detected 双口径 + 污染分层报告。

**仍然开放**:
- freed 的物理语义命名与是否列为主口径(建议:纳入 + 分层报告);
- 原始 run table 缺失的字节级溯源;
- trace status 300 的业务语义(仅影响 !=200 vs >=500 敏感性分析);
- Ada-RCA GAIA adapter 的仓库治理与 SOURCE_COMMIT 对齐;
- metric 通道 onset 字段失效是否需要表示层补丁(本轮明确不动 68D);
- detector-only 基线反超 Ada-RCA(§5.3)的建模解释。

---

## 5. 当前 E2E F1 低的归因(证据排序)

现状数字(i1 formal,旧 60/20/20 split,`artifacts/p5/i1/`):

| 层 | 指标 |
|---|---|
| Node(Test) | P 0.9806 / R 0.3858 / F1 0.5538 / AUC 0.8244 / AP 0.5036 |
| Event(Val) | P 0.9657 / R 0.5132 / F1 0.6703 |
| **Event(Test)** | **P 0.9973 / R 0.1903 / F1 0.3196**(θ=0.2575,744 episodes vs 3900 GT) |
| RCA matched(oracle/detected) | AC@1 0.9313 / 0.8908;MRR 0.9636 / 0.9428 |
| **E2E Diagnosis F1@1** | **0.2847**(TP@1 661 / GT 3899) |

1. **Recall 崩塌是唯一主因,且发生在事件层**:Diagnosis recall 0.1695 ≈ Event recall 0.1903 × detected AC@1 0.8908(乘法分解成立)——RCA 排序几乎不损失,瓶颈全在 detector 触发太少。Node 层 Test P=0.98/R=0.39 说明分数可分(AUC 0.82)但操作点极端保守:θ 在 Val 上按 F1 最优选取,迁移到 Test 后分数分布漂移(Val node AUC 0.899 → Test 0.824),同一 θ 只触发 744 episodes。**改为 Train 校准 + 报告 Train P/R 曲线全貌,是预期收益最大的单点修复。**
2. **训练协议漂移**:实际只训练 12 epochs(best@6,config 50/5)而冻结协议是 120/7;欠训练 + config sha 不一致使当前数字甚至不代表冻结协议的结果。修复流程后重跑,数字才可与协议对照。
3. **日志表示降维**:96% 的 GT 是 11s login,其区间内 metric 可观测率仅 31.7%(cpu 0%),logs 是短故障主证据通道;6 维 level 统计丢失模板信息。恢复 Drain3(Train-only)直接作用于 login recall。
4. **时间语义失真(次要但必须修)**:t_hat=bin start + 非因果匹配 → delay 中位 −3.87s;修正后 delay 变为物理可信的 +19~30s,对 60s 容差下的 recall 影响很小,但影响 Detection Delay 指标与 anchor_degradation 解释。
5. **特征质量红旗**:旧 grid 头部 826 bins 零 metric(11:07–18:00 段 NaN→0 伪造)、system 特征 07-05 才启动、8 个 half-2 文件 Test 段早停——新 timeline(metric_only)自动消除第 1 项,其余按冻结 missing 规则处理并报告。
6. **RCA 负增益(另案)**:matched 集上 detector-only(按 t_hat 处节点分数排序)AC@1 0.9650 > oracle Ada-RCA-G 0.9313 > detected 0.8908。第二阶段当前没有带来排序增益;root-frequency AC@1 仅 0.4906 说明任务本身可学。本轮不重设计(V2 §38),但该事实必须随结果公布,并作为后续建模议题。

---

## 6. GO / NO-GO 与最小修改清单

**判定:GO** —— 审计问题全部有证据结论,数据处理与协议层的修复路径明确,不需要重设计任何模型。条件:以下 MUST FIX 完成并复跑一次审计级 sanity check(标签 bin 数、split 账目、θ 来源)后,才允许 formal 训练与单次 Test 评估。

### MUST FIX(正式重跑前必须完成)

| # | 修改 | 文件 |
|---|---|---|
| 1 | 删 `expected_events: 16200` 与行数断言;N_GT 写入 artifact | `configs/e2e/gaia_p5_v1.yaml:12`、`src/e2e/protocol.py:188` |
| 2 | GT 生成纳入 `normal_memory_freed`(第 6 类,start=msg ts,dur=600s);taxonomy_version=v2-audit-20260913;排除侧保持(ERROR/碎片/injection-failed/INFO) | `src/e2e/protocol.py:24-30`(SUPPORTED_FAULT_TYPES)、事件 registry 重建脚本(以 `data/p5/v2/audit/gt_event_registry.csv` 为准) |
| 3 | split 改为 telemetry timeline 派生:metric_only,70/30,T_split=1626963120000;删 Validation 段;跨界 purge + boundary_exclusion_mask + timeline 外 84 事件显式报告;K 用 `(7*N)//10` 整数算术 | `configs/e2e/gaia_p5_v1.yaml`(split 节)、`src/e2e/protocol.py:162-253`、`scripts/p5/build_i1_protocol.py` |
| 4 | AD 标签改半开 overlap:`bin_start < end_ms ∧ bin_start+30000 > start_ms` | `src/e2e/ad_data.py:48-53` |
| 5 | t_hat = target bin end(prediction_available_time);匹配改因果 `0 ≤ t_hat−gt_start ≤ 60s`,保持确定性一对一 | `src/e2e/ad_data.py:142-144`、`src/e2e/event_detection.py:168,201-315` |
| 6 | θ 在 Train predictions+Train Event GT 上选定后冻结;rec-score median/MAD 校准 Train-fit、Test transform-only;checkpoint=固定 120 epochs final;run_manifest 与冻结 config 单一事实源(sha 一致) | `src/e2e/event_detection.py:512-566`、`util/train.py:251-288`、`scripts/p5/run_i1_ad.py`、manifest 生成脚本 |

### SHOULD FIX(强烈建议随同完成)

| # | 修改 | 文件 |
|---|---|---|
| 7 | 恢复 Drain3 模板日志特征(模板计数+5 level+log_total),词表 Train-only fit、Test 冻结 transform、新模板 UNK;产出 `log_schema.json` | `src/e2e/ad_preprocess.py:49-52,367-430`、`util/GAIA/gaia.ini`(复用) |
| 8 | 产出 `metric_feature_schema.json`(names/count/rule/Train 统计/checksum)与 Train/Test missingness artifact;文字协议删除"63 特征"表述 | `src/e2e/ad_preprocess.py`、docs |
| 9 | RCA contamination 统计内置到 pipeline 输出(foreign/diff-service/diff-fault per case + clean/contaminated 分层);oracle/detected purge 计数分别上报 | `src/e2e/gaia_rca_adapter.py`、`scripts/p5/run_i1_e2e.py`(参考 `data/p5/v2/audit/rca_contamination.json` 的口径) |
| 10 | 文档-代码对齐:score fusion(alpha=0.7)如实写入 IMPLEMENTATION 文档;Ada-RCA 拷贝的 SOURCE_COMMIT 钉死并说明与 canonical(W600-B15/RCAEval)的关系;metric 通道 onset 字段结构性失效写为已知限制 | `docs/P5_GAIA_E2E_IMPLEMENTATION_V1.md:113-114`、`src/e2e/rca_features.py` 头注 |

### 明确不做(本轮边界)

- 不重设计 Ada-MGAD / Ada-RCA 模型,不联合训练,不引入 learned eventizer;
- 不为提高 recall 修改/合并 GT、不移动 split、不删除污染 case;
- 不使用 Test 做任何选择;detector-only 反超问题仅记录,不在本轮处理。

---

## 7. Artifact 索引与复现

```text
data/p5/v2/audit/                        (大文件,gitignore;镜像 JSON 见 artifacts/p5/v2_audit/)
├── provenance.json                      worktree/commit/数据集/checksum/所有者确认记录
├── run_record_registry.csv              17152 行逐行语义判定(7.3MB)
├── gt_event_registry.csv                16217 事件(16200 verified + 17 candidate)
├── taxonomy_summary.json                pattern 清单/per-type 语义/两口径分布/冲突统计
├── normal_memory_freed_audit.json       17 条逐条 + 间隔证据
├── error_records_audit.json             38 条逐条 + level 校验
├── cpu_anomaly_audit.json               12 条逐条 + 小数解析验证
├── preparse_crosscheck.json             与预解析列交叉验证(0 不一致)
├── telemetry_coverage.json              三 modality 覆盖/质量红旗(全量计数,与 G0R2 交叉验证)
├── detector_timeline.json               两口径 timeline + 70/30 boundary + N±1 敏感性
├── timeline_vs_current_code.json        现行 60/20/20 GT 派生 split 与 V2 口径的数值差异
├── split_audit.json                     Train/Test 事件账目/purge/mask/冲突/burn-in(4 变体)
├── temporal_observability.csv           16217 事件 × metric 区间可观测性
├── rca_contamination.json               W300 oracle 污染汇总 + purge
└── rca_contamination_cases_{primary,with_freed}.csv  逐 case 污染计数

scripts/p5_v2/
├── audit_gt/audit_gt_taxonomy.py        GT 重推导(纯 stdlib,单脚本)
├── audit_telemetry/build_timeline.py    timeline/boundary 确定性重建
└── audit_merge/merge_split_audit.py     split/observability/contamination 合并
```

复现:`python3 scripts/p5_v2/audit_gt/audit_gt_taxonomy.py && python3 scripts/p5_v2/audit_telemetry/build_timeline.py && python3 scripts/p5_v2/audit_merge/merge_split_audit.py`(解释器 `/home/zhangll24/miniconda3/envs/DAG/bin/python3`,pandas 1.5.3;前两个脚本仅依赖 stdlib)。

---

## 附录

### A. 36 行 traceback 碎片
`(Background on this error at: http://sqlalche.me/e/14/e3q8)` 行是 38 条 ERROR(pymysql 1040)的多行 traceback 在 CSV 化时被拆出的续行:父行相邻、service 匹配 36/36、预解析列全空、无时间戳。旧 G0R2 的 "unsupported_event_type 36" 即此。任何 parser 不得将其计为事件或丢弃不报。

### B. K=floor(0.7N) 的浮点陷阱
`int(0.7*87120) = 60983`(0.7×87120 在 IEEE754 下为 60983.99999999999),会把 T_split 错移 30s 并使 Train 少 1 bin。一律使用 `(7*N)//10`。本审计的 `build_timeline.py` 已按此实现并在 artifact 中记录敏感性(N+1 变体 split 不变)。

### C. 为什么 DiagFusion 的 GAIA 数字与 16200/16217 不可比
DiagFusion 使用离线固化的 1099-case demo 子集(`gaia_resplit.csv`,train 160 / test 939,按时间前缀划分):丢弃 07-04 之前全部记录(因此 freed 只有 16/17)、丢弃 cpu_anomalies(12)/log_upload_failure(38)/exception_injection_failed(10)、login 降采样至 527/15478、memory 505/652、file 36/43、access 15/15。它证明的是 taxonomy 处理(freed 作为独立类、per-type 时间语义),不是事件总量。

### D. canonical Ada-RCA 与 E2E adapter 的关系
Ada-RCA@6c3abaa 的冻结表示是 RCAEval RE2 上的 W600-B15(80×15s)、oracle-only anchor(epoch 秒)、无 detector 集成,GAIA 被明文列为 out of scope/DO_NOT_PORT。E2E 的 W300-B15 68D 是 `src/e2e/rca_features.py` 对 Ada-RCA@a2c6209 `features.py` 的拷贝再参数化(run_manifest source binding 记录)。风险:anchor 语义单点耦合(pre-event median/MAD 归一化依赖 t0=真实故障起点;anchor 前移会把故障后数据混入 pre 段)、单位耦合(秒 vs 毫秒无断言)、支撑缺失时静默退化(全 missing 通道→平局→按服务名字典序)。E2E 侧 detected anchor 改为 bin-end 后,以上耦合点在实现阶段必须加断言(单位/支撑/anchor 语义)。

### E. 子代理工具故障与 telemetry 数字的验证链
本轮两个数据扫描子任务遭遇 auto-mode 分类器临时故障(写入通道中断):GT 审计脚本改由主会话直接落盘运行(结果即本报告);telemetry 全量计数由只读通道完成,并与仓库内已提交的 G0R2 历史全量扫描逐项核对一致(trace 28,681,438 行、status 200/300/400/500=24,453,030/3,040,760/30,980/1,156,668、非 200=4,228,408;business 87,974,871 条),metric 端点另经 head/tail 独立抽查(首样本 1625133601000、末段文件尾 1627747170000/1627747200000)。`telemetry_coverage.json` 中抽样项均已标注 estimated/method。

### F. 漂移与分层报告义务
cpu_anomalies(12)全部位于 Test(07-27→07-31);login 占 GT 95.5% 且集中于 mobservice1/2;business WARNING 99.998% 集中于 redisservice。正式结果必须按 fault type / duration / same-bin collision / same-service collision 分层(§3.4/3.5 的统计即分层基础),并明示 Train 校准对 cpu 类零可见。
