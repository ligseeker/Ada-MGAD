# 研究状态与决策日志

> 状态版本：`research_state_v9`
> 最近更新：2026-08-19
> 状态所有者：本文件；完成 Gate 后必须同步更新

## 1. 当前状态摘要

| 项目 | 状态 | 证据 |
|---|---|---|
| Ada-MGAD 异常检测代码 | 已存在 | 当前 `src/`、`util/`、`README.md` |
| 异常检测论文/完整实验产物 | 当前工作区不可用 | 仅参考对话提及附件与结果 |
| 旧 Ada-MGAD-dependent RCA Pilot | 对话报告已完成，当前 clone 不可核验 | [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) |
| Standalone RCA 研究定位 | 已冻结 | [RCA_RESEARCH_DESIGN.md](RCA_RESEARCH_DESIGN.md) V0.2 |
| P1 Benchmark Protocol | 已完成并冻结 | [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) V0.5 |
| `rca-standalone` 分支 | 已建立 | Git 当前分支与远端 tracking branch |
| `RCACase` / adapters / evaluator | 已实现首版 | `src/data/`、`src/evaluation/`、`tests/` |
| Label-separated manifests | GAIA 16,200 + RE2-OB 90 已按修正后的 log 时间引用重生并校验 | `artifacts/p1/manifests/` |
| Telemetry diagnostics scanner | 约 38 GB raw telemetry 全量扫描完成 | `artifacts/p1/telemetry_diagnostics.json`、[TELEMETRY_DIAGNOSTICS.md](TELEMETRY_DIAGNOSTICS.md) |
| Actual 5-fold assignments | GAIA grouped-stratified + RE2-OB singleton stratified 已生成并校验 | `artifacts/p1/splits/`、[SPLIT_DIAGNOSTICS.md](SPLIT_DIAGNOSTICS.md) |
| P1 sanity baselines | B0/B1/B2 已保存完整 rankings、macro metrics 与 run manifests | `artifacts/p1/baselines/`、[BASELINE_RESULTS.md](BASELINE_RESULTS.md) |
| GAIA main cohort | 13,470 anchor-unique-root cases；2,730 multi-root sensitivity | `artifacts/p1/inclusion/gaia/`、[GAIA_INCLUSION_AUDIT.md](GAIA_INCLUSION_AUDIT.md) |
| RE2-OB 数据 | 90/90 可读；归档与 360 个实际输入文件已 content-pinned | `artifacts/p1/source_snapshots/re2ob/`、[P1_REPRODUCIBILITY_AUDIT.md](P1_REPRODUCIBILITY_AUDIT.md) |
| P1 G1–G8 | 8/8 completed；最终自动门禁通过 | `artifacts/p1/gate_audit.json`、见下方 Gate 表 |
| P2 Experiment Plan | V0.2；metric onset 由 coverage 收缩为 60/120 s | [P2_EXPERIMENT_PLAN.md](P2_EXPERIMENT_PLAN.md) |
| P2-G1 metric | completed；170 维全量 bundle 与独立 coverage audit | [P2_METRIC_FEATURES.md](P2_METRIC_FEATURES.md)、`artifacts/p2/features/` |
| P2-G2 C0-M | completed；nested OOF、泄漏审计、确定性复跑 | [P2_C0_M_RESULTS.md](P2_C0_M_RESULTS.md)、`artifacts/p2/runs/c0_m/` |
| P2 logs/traces | L0/T0 schema、synthetic tests 与真实 smoke completed；全量 pending | [P2_MODALITY_SCHEMA_AUDIT.md](P2_MODALITY_SCHEMA_AUDIT.md)、`artifacts/p2/event_features_smoke/` |

**当前研究阶段：P2 — Multimodal RCA Representation and Attribution。**

## 2. 仓库快照

```text
branch:       rca-standalone
remote:       origin/rca-standalone
commit:       ab31282055b625838f41a1a3c91e51098175f254
subject:      first commit
commit date:  2026-07-15T22:27:48+08:00
working tree: 当前 P1 代码与文档未提交
```

当前事实：

- Git 仅显示一个本地分支与对应远端分支；
- `docs/` 在本轮之前为空；
- 旧 Ada-MGAD 的 GAIA/MSDS 异常检测实现保持不动；
- `src/data/schema.py` 已物理拆分 prediction input 与 label；
- `src/evaluation/` 已实现完整 ranking 校验与 AC@1/3/5、Avg@5、MRR；
- `src/data/gaia.py` 已解析 16,200 条候选注入并保留重叠审计；
- `src/data/rcaeval.py` 已在本地 RE2-OB 达到 90/90 可读；
- `src/data/manifest.py` 已输出分离的 input/label/source bundle 与文件 SHA-256；
- GAIA 原始注入区间有 4,037 个重叠 case；±300 s context-aware manifest 为 322 组、最大组 471；
- 原始资产体积约为 GAIA 30.3 GB、RE2-OB 8.1 GB，scanner 采用 chunked streaming；
- GAIA business-log 的真实毫秒时间位于 `message` 前缀，adapter 与 manifest 已修正；
- RE2-OB 的 `cluster_info.json` 是日志模板元数据；pod-node 表是部署映射，不是服务调用图；
- 全量 scanner 共读取 349,120,558 行，GAIA/RE2-OB 三模态均无未完成文件；
- G8 已完成；主窗口已冻结为 ±300 s，context-aware groups 已写入 manifest；
- GAIA 主 split 已冻结为 grouped-stratified 5-fold，RE2-OB 为 singleton
  stratified 5-fold，统一 seed `20260819`；
- Random/Frequency/Metric-change 已在两个数据集完成并字节级复现；
- P1 closeout 时 `tests/` 51/51 通过；P2 新增 metric/event feature 与
  linear-ranker tests 后当前全套 66/66 通过；
- GAIA 16,200 event inventory 已分为 13,470-case main 与 2,730-case
  multi-root sensitivity；
- RE2-OB 原始归档与 8,441,465,341 字节实际消费文件已逐字节固定，content
  identity 为 `cec0030da8b499914af7979283eb1a2cd4f2cad2430742c70130829cd28133c9`；
- 最终 `p1_gate_audit_v1` 已验证 9 组 baseline、全部 bindings、指标重算与
  Label Firewall，`raw_re2_source_bytes_verified=true`；
- G1–G8 与 reproducibility closeout 均 completed，P1 已关闭；
- P2 metric 全量表征为 GAIA 134,700 rows、RE2 990 rows、每行 170 features；
- GAIA 30 s metric onset block 仅约 0.09% service rows 可观察，故调参候选只保留
  60/120 s；
- C0-M GAIA/RE2 root-macro Avg@5 为 0.5820/0.9756；相对 P1 B2 为
  −0.1266/+0.0422，不形成统一优越性结论；
- P2-G2 的 predictions/metrics/training-audit 核心文件重跑 SHA-256 一致。

## 3. ARS 工作流位置

本轮从既有 research state 接续，使用 `academic-research-suite` 的 `experiment-agent` 路由实施 P1；没有启动完整论文十阶段 pipeline，也没有声称通过论文完整性 Gate。

当前实质工作属于研究设计到实验规划的过渡：

```text
研究问题与路线收敛       completed
Standalone RCA Design    completed / frozen V0.2
P1 Benchmark Protocol    completed / frozen V0.5
P1 implementation        G1-G8 + reproducibility closeout completed
P2 representation        metric completed / L0-T0 smoke completed
P2 controlled experiments P2-G2 completed / P2-G3 inputs pending
paper full drafting      not started in this workspace
```

## 4. P1 Gate 状态

| Gate | 内容 | 状态 | 证据路径 |
|---|---|---|---|
| G1 | GAIA event → RCACase | completed | 16,200 inventory 全部映射；13,470 main + 2,730 multi-root sensitivity；`artifacts/p1/inclusion/gaia/`、[GAIA_INCLUSION_AUDIT.md](GAIA_INCLUSION_AUDIT.md) |
| G2 | RE2-OB 90 cases → RCACase | completed | `src/data/rcaeval.py`、`artifacts/p1/manifests/re2ob/`、`artifacts/p1/source_snapshots/re2ob/`；90/90、排除 0、实际输入逐字节固定 |
| G3 | 两数据集完整 service ranking | completed | `artifacts/p1/baselines/{gaia,re2ob}/*/predictions.jsonl`；三种 baseline 均通过完整排列校验 |
| G4 | AC/Avg/MRR toy tests | completed | `src/evaluation/`、`tests/test_metrics.py`；最终审计又从 9 组完整 rankings 重算 metrics |
| G5 | 三个 sanity baselines | completed | `artifacts/p1/baseline_summary.json`、[BASELINE_RESULTS.md](BASELINE_RESULTS.md)；B0/B1/B2 两数据集均完成 |
| G6 | event/case split 无泄漏 | completed | `artifacts/p1/splits/`、`artifacts/p1/split_diagnostics.json`、`src/data/split.py`、`tests/test_split_integrity.py`；实际 assignment 与 source checksum binding 已校验 |
| G7 | Label Firewall | completed | 分离 manifests、input-only predictors、B1 train-fold-only ID audit、B2 within-case fit scope、prediction 敏感字段零命中与测试 |
| G8 | dataset diagnostic report | completed | `artifacts/p1/telemetry_diagnostics.json`、[TELEMETRY_DIAGNOSTICS.md](TELEMETRY_DIAGNOSTICS.md)；全量 349,120,558 行、未完成文件 0 |

G1–G8 已全部 `completed` 且具备代码/测试/产物路径。RCAEval 原始数据
content snapshot 与全产物一致性 closeout 也已完成；最终审计 SHA-256 为
`a02625a0a37136f0d763c8c7fa83d39b003c15eeac033310f6990ab55ce1bf45`。

## 5. 已冻结决策

1. 毕设包含独立的异常检测与 RCA 两项研究，最终在事件层集成。
2. Standalone RCA 不使用 Ada-MGAD latent representation。
3. 旧 `rca` 只作为 Pilot，不作为新方法开发基线。
4. 新 RCA 任务是 event-level、service-level、single-root ranking 主协议。
5. 主数据为 GAIA + RE2-OB；RE2-TT 为扩展。
6. 主实验为 Oracle RCA；Trigger Robustness 与 Ada-MGAD-triggered RCA 后置。
7. 候选集合由测试时可得信息构造，主实验包含全部可观测服务。
8. 先 case/event split，再拟合任何学习型 preprocessing。
9. P1 实现 Label Firewall，并只做 Random/Frequency/Metric-change。
10. H1/H2/H3 是可检验假设，不是预设创新点。
11. P1 不把 Event-stage 三段切分固化到数据协议，也不实现新神经网络。
12. 主上下文冻结为 `T_pre=T_post=300 s`，采用毫秒半开区间且不合成边界填充值。
13. 只有 P1 G1–G8 与 reproducibility closeout 完成后才进入 P2；该条件现已满足。
14. GAIA 主 split 为 322 个 context groups 上的 grouped-stratified 5-fold；
    RE2-OB 为 singleton stratified 5-fold；seed 均为 `20260819`。时间块只作
    GAIA temporal-shift sensitivity。
15. GAIA 主结果使用 13,470 个锚点唯一 root-service cases；2,730 个锚点
    multi-root cases 只作 concurrency sensitivity；完整 inventory 不删除。

## 6. 尚未决定的问题

这些问题必须由 P2 实验回答，不能在新对话中凭经验直接填值：

- RE2-OB 指标候选已稳定为 11 个应用服务；trace 静态 alias 已冻结，但缺失候选的
  mask/coverage 对模型的影响仍待实验；
- topology 的统一表示与缺失/动态策略；
- logs/traces 的 30/60/120 s stage coverage；metric 已冻结为 60/120 s；
- H2 的比较模块与 H3 的结构机制；
- 最终论文题目是否加入“根因分析”。

## 7. 下一步执行队列

### P1.1：schema + evaluator（completed）

- 建立 `RCACaseInput` / `RCACaseLabel`；
- 实现 schema invariants；
- 实现 AC@1/3/5、Avg@5、MRR；
- 用手工 toy cases 覆盖正确首位、Top-k 命中、缺失根因、重复/不完整 ranking；
- 建立 Label Firewall 的最小接口测试。

### P1.2：GAIA adapter（completed）

- 已复用并审计 `parse_anomaly_event`，修复 RCA 路径的 CPU 浮点时长；
- 已生成 16,200 条独立 injection manifest 与 952 条排除记录；
- 已生成注入区间与 ±300 s 半开上下文并集的 context-aware group manifest；
- 流式 scanner 与全量 diagnostics 已完成；13,470-case 主 cohort 与
  2,730-case multi-root sensitivity 已冻结。

### P1.3：RE2-OB adapter（completed）

- 已完成 1-case smoke test 与 90/90 case 覆盖；
- 已输出 input/label/source/group/exclusion manifests；
- scanner 已记录 layout digest 与 inject_time 内容摘要；原始 `RE2-OB.zip` 和
  90 cases 实际消费的 360 个文件已有逐文件 SHA-256 content snapshot。

### P1.3.5：telemetry diagnostics（completed）

- 已实现 Metrics/Logs/Traces 的 chunked streaming scanner；
- 定时 metric 分开报告 value-cell missingness、采样时间缺口与重复 timestamp；
- event streams 分开报告必填字段/时间解析失败和 30 s activity coverage；
- 已识别 GAIA log timestamp 来源与 RE2 topology metadata 语义；
- 真实 smoke、manifest regeneration 与全量扫描均 exit 0；G8 报告已生成。

### P1.4：split + sanity baselines（completed）

- 322 组 context-aware manifest 与 split-integrity validator 已实现；
- grouped/time-aware 候选已比较，GAIA/RE2-OB 实际 5-fold assignment 已生成；
- train-only Root Frequency 与 within-case Random/Metric-change 已实现；
- per-case ranking、overall/fault-macro/root-macro 与 run manifest 已保存。

### P1.5：Gate audit（completed）

- G1–G8 状态与证据路径已审核为 completed；
- `p1_gate_audit_v1` 已交叉验证 9 组 baseline、split/cohort/source bindings、
  指标重算、Label Firewall、全量 diagnostics 与 RE2 原始字节；
- 51/51 tests、全量 compile 与 `git diff --check` 通过。

### P2.0：representation experiment design（completed）

- 在 P1 冻结边界内定义 metric/log/trace 的测试时可得表征；
- 固定 learned preprocessing 的 inner-validation 与 train-fold-only fit 协议；
- 将 H1/H2/H3 拆成可独立否证的最小实验与 ablation；
- 先实现单模态/简单融合对照，再决定是否进入结构传播模块。

执行协议见 [P2_EXPERIMENT_PLAN.md](P2_EXPERIMENT_PLAN.md) V0.2。外部
RCAEval 参考实现固定审计 commit 为
`4695aa69f4f1f57b9094ca04ff235908b73a8e24`；官方输入/窗口与 Track C 不等价，
必须分表。

### P2-G1：feature schema + modality extractors（partial）

- 通用 label-free feature bundle 与 metric extractor 已完成；
- GAIA main/RE2-OB metric 全量产物、coverage audit 与确定性 smoke 已完成；
- L0/T0 raw schema、`frontendservice→frontend` alias、纯函数 tests 与真实 smoke
  已完成；
- 下一步把 readers 改为可恢复的全量流式 extraction，并审计总体 coverage。

### P2-G2：C0-M nested OOF（completed）

- outer 5-fold、outer-train 内四折轮换、四个 C 候选全部运行；
- 两数据集各 80 inner fits 与 5 outer fits，case/group overlap 均为 0；
- 完整 rankings、service scores、metrics、fit-state hashes 与 source bindings 已保存；
- 独立审计精确重算 metrics，第二次全量运行核心文件 hashes 一致；
- GAIA 不优于 P1 B2 的 root-macro 主端点，负结果保留。

## 8. 主要风险

| 风险 | 当前控制 |
|---|---|
| 把异常检测标签语义直接当 RCA 证据 | GAIA + RE2-OB 双数据集；Frequency baseline；宏平均 |
| 同一/重叠事件或共享上下文跨 split | ±300 s context-aware groups + 实际 assignment + split-integrity/checksum test |
| test label 进入特征/候选/normalization | 物理拆分 Input/Label + Label Firewall test |
| 旧 Pilot 数字被误当新方法结果 | 在实验日志标记 `conversation-reported` |
| 先设计模型后发现任务不统一 | P1 benchmark 与最终门禁已完成；P2 必须复用冻结协议 |
| 结构创新被预设 | H3 Go/No-Go |
| RCAEval 主分支持续变化 | 本地归档与实际消费文件已 content-pinned；上游 provenance 缺失需披露 |
| 官方 service-level 口径与自定义 evaluator 漂移 | toy tests + 对照官方 evaluator |

## 9. 决策日志

| 日期 | 旧理解 | 新决策 | 原因/证据 |
|---|---|---|---|
| 2026-08-18 | RCA 可作为 Ada-MGAD 下游 head | RCA 应独立成立，只在事件层集成 | 用户希望形成两项独立研究；AD 表征目标与 root attribution 不同 |
| 2026-08-18 | 继续旧 `rca` 分支 | 旧分支冻结为 Pilot；新分支从 Ada-MGAD 基线重新开发 | 避免继承 A/P/T 和检测器强耦合设计 |
| 2026-08-18 | GAIA 窗口可作为 RCA 样本 | 一次 fault injection / failure case 为一个样本 | 避免 pseudo-replication 与 event leakage |
| 2026-08-18 | 重叠注入可逐条随机分配 | 直接或传递重叠的 injection 构成不可拆分 group | 防止相关事件及其 telemetry 上下文跨 split；不预设 fold 类型 |
| 2026-08-18 | RCA 粒度可能含 indicator | 主任务固定为 service-level | GAIA 与 RCAEval 的公共可比较粒度 |
| 2026-08-19 | G8 只有 scanner 实现、原始分组只看注入区间 | G8 以全量产物完成；主窗口冻结为 ±300 s，并以注入区间与上下文并集生成 split 原子组 | 全量扫描 349,120,558 行；±300 s 为 322 组/最大 471，±600 s 仅 23 组/最大 3,206 |
| 2026-08-19 | GAIA grouped fold 与 purged time block 尚未选择，RE2-OB 无实际 fold | 主协议冻结为 seed `20260819` 的 grouped-stratified 5-fold；GAIA 时间块只作敏感性分析 | GAIA 时间块把 12 条 CPU fault 全放入 fold-4；grouped folds 大小 3239/3239/3239/3241/3242 且全类别覆盖；RE2 每折 18 cases |
| 2026-08-19 | P1 只有 evaluator，没有实际 sanity rankings | B0 Random、B1 train-fold Root Frequency、B2 within-case Metric Change 全部运行并保存 | GAIA Frequency 暴露 overall shortcut；B2 GAIA/RE2 AC@1 为 0.1414/0.8556；预测与 run manifests 字节级复现 |
| 2026-08-19 | GAIA 16,200 cases 的 single-root 纳入语义未冻结 | 主结果排除锚点时存在其他 root service 的 2,730 cases；13,470 cases 进入 main，同-root 并发保留 | 规则直接来自 service-level label cardinality，不使用模型表现；主 folds 仍完整覆盖 root/fault |
| 2026-08-19 | RE2-OB 只有 90/90 可读和目录布局摘要，P1 仍待收口 | 对原始归档与 360 个实际消费文件做 content-addressed snapshot；最终自动门禁通过后关闭 P1、进入 P2 | 本地无上游 version 元数据；内容身份可逐字节固定实际实验输入，gate audit 同时验证 9 组 baseline 与全部绑定 |
| 2026-08-19 | metric onset 候选为 30/60/120 s | 统一 metric 调参只允许 60/120 s，30 s 保留 unsupported control | GAIA 30 s onset blocks 仅约 0.09% service rows 可观察；决策发生在 stage 模型运行前 |
| 2026-08-19 | C0-M 只是计划项 | C0-M nested OOF 完成并作为 learned metric reference；不宣称统一优于 B2 | GAIA/RE2 root-macro Avg@5 为 0.5820/0.9756，相对 B2 为 −0.1266/+0.0422；完整泄漏与确定性审计通过 |
| 2026-08-19 | log/trace entity 与派生字段边界未定 | L0 只用 raw 无词表统计；T0 固定 `frontendservice→frontend`，缺失 trace entities 保留 mask | GAIA 无 raw template ID；RE2 template 字段已派生且 trace 仅直接覆盖 7 类 entity |
| 2026-08-18 | HR@k 为主 | 对齐 AC@1/3/5 + Avg@5，MRR 补充 | 对齐 RCAEval 官方 evaluator |
| 2026-08-18 | 直接进入方法开发 | 先完成 P1 Benchmark Layer | 当前任务协议、候选集合、split 和泄漏风险尚未落地 |
| 2026-08-18 | P1 可能预切 event stages | P1 只提供完整锚点上下文 | 阶段划分属于 H1，不应固化成数据协议 |
| 2026-08-18 | 参考对话即项目状态 | 仓库证据与对话报告分级 | 当前 clone 缺少旧 Pilot 和 RCA 实现 |

## 10. 新对话恢复检查

继续工作前只需确认：

1. 当前分支与提交是否仍与本文件一致；
2. 是否已有未登记的 RCA 文件、数据或实验产物；
3. P1 冻结产物的哈希是否仍通过 `scripts/audit_p1_gates.py`；
4. 下一项是否仍为 P2-G1 L0/T0 full streaming extraction + coverage audit。

如仓库事实变化，以代码/测试/产物为准更新本文件，不沿用过期状态。
