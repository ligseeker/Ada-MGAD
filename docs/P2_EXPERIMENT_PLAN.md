# P2 Multimodal RCA Representation Experiment Plan V0.2

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-19
- Verification Status: UNVERIFIED
- Version Label: p2_experiment_plan_v0.2

> 状态：P2-G1/G2、全量 M/L/T、C0-M/L/T 与 C1-I completed；M1-S in progress

## 1. 研究目标与冻结边界

P2 研究以下问题：在已确认故障事件中，能否通过多模态演化、case 内相对关系和
合法结构信息，把根因服务与传播症状区分开。

P1 的以下部分不因模型结果改变：

- GAIA 13,470-case main 与 2,730-case concurrency sensitivity；
- RE2-OB 90 cases；
- ±300 s 半开上下文与 seed `20260819` 的 5-fold assignments；
- 测试时完整候选服务集合；
- service-level 完整 ranking；
- AC@1/3/5、Avg@5、MRR 及 overall/fault-macro/root-macro；
- Label Firewall 和所有 train-fold-only fit 规则。

P2 不消费 Ada-MGAD latent representation，也不把 fault type 当输入。

## 2. 两条实验轨道

### Track C：统一协议内受控实验（主证据）

GAIA 与 RE2-OB 使用同一逻辑 extractor、同一 outer-fold 协议和同一 evaluator。
模型只接收由原始 telemetry 与合法 candidate/topology 生成的服务级特征。H1/H2/H3
的结论只来自该轨道，因为只有它能保持双数据集和单因素对照。

### Track E：RCAEval 官方方法（外部参考）

只在 RE2-OB 上复现固定 commit 的官方 BARO/多源方法，结果单列为 external
reference。官方方法使用 indicator-level ranking、`simple_metrics.csv` 及预聚合
`logts.csv`/`tracets_*.csv`，窗口和 preprocessing 也不同，不能替代 Track C 的
H1/H2/H3 ablation。

## 3. Track C 表征定义

所有 extractor 输出每个 `(case_id, service)` 一行固定维度向量、coverage/missing
mask 和版本化 feature names；不输出 root/fault。特征文件在训练前一次生成，模型
runner 才通过 case ID 在受控边界合并训练 labels。

### 3.1 Metrics

对每个原始 metric series，在每个比较区间计算：

- 有限样本数与 coverage；
- mean、population std、least-squares slope；
- signed standardized mean shift；
- log scale ratio；
- standardized slope shift。

然后在 service 内跨 metric dimensions 聚合为可精确流式计算的固定统计量：
valid count/ratio、maximum、Top-3 mean、Top-5 mean、signed mean 和 positive
fraction。这使 GAIA 与 RE2 不需要共享 metric 名称或数量，也不需要为 GAIA
缓存数亿个 series-case 中间值。该集合在查看任何 P2 模型结果前固定。

### 3.2 Logs

分两级执行：

- L0：不需要词表且可流式复现的 event-rate ratio、active-second-fraction shift、
  severity/error-fraction shift、message-length mean/scale shift，共 10 blocks × 5 =
  50 features；
- L1：fold-specific template/vocabulary features。若 GAIA 需要 Drain parsing，parser
  与 vocabulary 只能在 outer-train cases 上拟合，再转换 validation/test。

L0 先于 L1，避免文本模板工程掩盖多模态问题。

### 3.3 Traces

每服务提取 span rate、error/status ratio、duration median/p90/p99、unique trace 与
operation rate、parent-presence coverage，共 10 blocks × 8 = 80 features。实体映射必须
版本化，例如 RE2 的 `frontendservice` 只能通过显式 alias 映射到合法候选
`frontend`，不能通过 root label 修正。

incoming/outgoing parent edge 不进入独立 T0，留到 H3 observed/randomized/identity
结构对照，避免 C0-T 已隐式消费图传播。

### 3.4 Whole 与 Event-stage

- Whole：pre=`[-300,0)`，post=`[0,300)`；
- Staged：pre=`[-300,0)`，onset=`[0,T_o)`，impact=`[T_o,300)`；
- metric `T_o` 候选为 60/120 s，由 outer-train 内部验证选择；30 s 特征因
  GAIA 采样 coverage 不足仅保留为 unsupported control；
- stage features 同时包括 pre→onset、pre→impact、onset→impact，缺失段保留 mask。

不得根据 outer-test 或完整数据结果固定 `T_o`。

全量 coverage audit 后、查看任何 M1-S 模型结果前，统一 M1-S 的可用 staged
通道冻结如下：

- metric 仅使用 coverage-supported 的 60/120 s 候选，30 s 继续作为 unsupported
  control，不进入主模型；
- trace 的 60/120 s 在 GAIA 与 RE2 均达到预先冻结的 content-complete 绝对覆盖
  0.80、相对 whole 覆盖 0.90 门槛，进入 M1-S；
- RE2 log staged 的 content-complete 比率为 0.794900，虽与 whole 相同但低于
  0.80 绝对门槛，因此 unified M1-S 不加入 log staged；C1-I 已有的 log whole
  保留；
- M1-S 因而是 C1-I whole M/L/T 加 metric+trace staged，`T_o ∈ {60,120}` 与
  `C` 在每个 outer-train 的四折 inner validation 内联合选择。并列时先更强正则，
  再更短 onset。

## 4. 受控模型阶梯

| ID | 输入/改变因素 | 目的 |
|---|---|---|
| C0-M | Metrics whole-context，独立线性 scorer | 可学习 metric 基线；对照 P1 B2 |
| C0-L | Logs whole-context，独立线性 scorer | 单模态可辨识性 |
| C0-T | Traces whole-context，独立线性 scorer | 单模态可辨识性 |
| C1-I | M/L/T concatenate + missing masks，独立 scorer | 多模态是否优于最佳单模态 |
| M1-S | C1-I + event-stage representation | 检验 H1 |
| M2-R | M1-S + case-relative normalization/context | 检验 H2 |
| M2-D | 仅当 M2-R 有信号时加入 DeepSets/listwise scorer | 排除线性相对特征容量不足 |
| M3-G | 最强非结构模型 + trace-derived structure | 检验 H3 |

第一轮统一使用可解释线性 scorer，不以神经网络容量作为改进来源。M2-D/M3-G 只有
前一 Gate 通过才实现。

## 5. 学习与模型选择

### 5.1 Outer evaluation

复用 P1 五折。每个 outer test fold 只产生一次最终 prediction；五折拼接后统一
评价。GAIA main 继承 inventory group-safe assignment，不能重新随机拆 case。

### 5.2 Inner validation

对当前 outer-train 的四个既有 folds 做轮换验证：每次三折训练、一折验证，四次
validation 的 root-service macro Avg@5 均值作为唯一调参目标。并列时按以下顺序：

1. 更简单模型；
2. 更强正则；
3. 更短且 coverage-supported 的 onset；
4. 配置字典序。

初始线性 scorer 使用 L2 logistic regression，`C ∈ {0.01, 0.1, 1, 10}`。每个 case
总训练权重相同：root row 权重 0.5，所有 non-root rows 合计 0.5。imputer、scaler、
feature selector 与任何词表只在当前训练折拟合。

### 5.3 H2 的最小实现

M2-R 不先引入 attention。对每个 service feature 添加 case 内 median-centered
value、MAD-scaled value 和 percentile rank，并保留原值。这样相对模型与独立
模型共享 extractor/scorer，仅改变“是否看见其他候选服务的相对位置”。

## 6. 结构实验边界

两数据集没有可直接共用的显式静态 service call graph。H3 的第一合法结构来源是
当前 case 原始 traces 中通过 `trace_id/span_id/parent_id/service` 构建的动态有向图。

M3-G 必须包含：

- no-graph 最强基线；
- observed graph；
- degree-preserving randomized graph；
- identity/self-only graph；
- parent-link missingness/coverage 报告。

使用真实 root 选择 graph-near hard negatives 只允许发生在 outer training folds，
测试时禁止 root-conditioned 邻域。若结构只在一个数据集改善，H3 记为 dataset-
specific，不形成双数据集“结构感知”主 claim。

## 7. 统计与 Go/No-Go

主比较端点为 root-service macro Avg@5，root-macro AC@1 为关键次端点；同时报告
fault-macro 与 overall。每个方法还需报告 coverage slices 和每折结果。

配对不确定性使用 seed `20260819` 的 10,000 次 bootstrap：GAIA 按不可拆分
context group 重采样，RE2 按 case 重采样。任何选择、早停或超参搜索均不能使用
bootstrap/test 结果。

- exploratory signal：两个数据集的主端点点估计均为正；
- claim-ready：两个数据集的主端点 95% paired bootstrap CI 下界均大于 0，且关键
  次端点没有超过 0.01 的负向变化；
- H1/H2 不满足 exploratory signal：记录 no-go，不继续放大相应模块；
- H3 不满足 claim-ready：不使用“结构感知 RCA”作为主创新 claim。

## 8. External baseline 适配审计

本轮核验的官方仓库 main commit 为
`4695aa69f4f1f57b9094ca04ff235908b73a8e24`（2026-08-19 查询）。执行前必须
固定该 commit 或另建明确版本，不使用移动的 `main`。

| 方法 | 固定代码实际使用 | P2 角色 |
|---|---|---|
| BARO | raw/simple metrics；normal 上 RobustScaler，按 anomalous max z-score 排 indicator | metric external reference |
| Multi-source BARO | metrics + `logts` + `tracets_err/lat`，各 indicator 独立 robust z-score 后混排 | naive multi-source external reference |
| Multi-source CIRCA | 当前实现组合 metrics + `logts`，不读取 trace series | metric+log structural reference |
| Multi-source RCD | 当前入口接收多源字典，但固定代码的实际 RCD 计算只使用 metrics | 不作为“三模态”主对照；可列 metric reference |

官方 runner 默认读取预聚合 time series，并把 indicator ranking 去重为 service
ranking。Track E 在运行前还需：

1. 把其辅助派生文件加入独立 source snapshot；
2. 保存官方原始 indicator ranking；
3. 以本项目固定候选集合做确定性 service projection；
4. 分开报告 exact-official window 与 adapted-±300 s window；
5. 不把 RE2-only external result 用作 GAIA/RE2 双数据集机制结论。

TORAI 使用另行处理的数据与环境；EventADL 面向 CloudTrail-style event logs，均不
进入 P2 第一轮主表。

## 9. P2 Gates

| Gate | 完成条件 |
|---|---|
| P2-G1 | metric 子门 completed；L0/T0 schema、synthetic tests 与真实 smoke completed，full extraction pending |
| P2-G2 | completed；C0-M OOF、80-inner-fit audit、完整 ranking、macro metrics、P1 B2 对照与确定性复跑齐全 |
| P2-G3 | C0-L/C0-T/C1-I 完成；每模态 coverage 与 best-single-modal 对照齐全 |
| P2-G4 | M1-S 对 C1-I 的 H1 配对检验完成，给出 go/no-go |
| P2-G5 | M2-R（必要时 M2-D）对 M1-S 的 H2 配对检验完成 |
| P2-G6 | 官方外部基线以固定 commit、独立输入审计和分表完成 |
| P2-G7 | M3-G 的 observed/randomized/identity 对照完成，给出 H3 claim/no-go |

## 10. 产物规范与立即执行项

建议路径：

```text
artifacts/p2/features/<dataset>/<extractor>/
artifacts/p2/runs/<method>/<dataset>/
artifacts/p2/external/rcaeval-<commit>/
```

每个 feature manifest 必须绑定 P1 dataset/cohort/split/source hashes、extractor
config、feature names、row/case counts、output checksums。每个 run 保存 fold-specific
selected config、train/validation/test ID digests、fit statistics、per-case rankings、
metrics 和 runtime。

下一项按顺序执行：

1. 将已通过 smoke 的 L0/T0 reader 改为全量可恢复/流式实现；
2. 先运行全量 L0/T0 coverage extraction，不读取 labels；
3. 独立验证 feature manifests、entity masks 与 30/60/120 s coverage；
4. 运行 C0-L/C0-T，再运行 C1-I；
5. 不并行开发 template L1、stage 模型或图模型。

## 11. 本轮官方依据

- [RCAEval 官方仓库固定 commit](https://github.com/phamquiluan/RCAEval/tree/4695aa69f4f1f57b9094ca04ff235908b73a8e24)
- [RCAEval 官方 README](https://github.com/phamquiluan/RCAEval/blob/4695aa69f4f1f57b9094ca04ff235908b73a8e24/README.md)
- [官方 benchmark runner](https://github.com/phamquiluan/RCAEval/blob/4695aa69f4f1f57b9094ca04ff235908b73a8e24/main.py)
- [BARO / Multi-source BARO 实现](https://github.com/phamquiluan/RCAEval/blob/4695aa69f4f1f57b9094ca04ff235908b73a8e24/RCAEval/e2e/baro.py)
- [CIRCA / Multi-source CIRCA 实现](https://github.com/phamquiluan/RCAEval/blob/4695aa69f4f1f57b9094ca04ff235908b73a8e24/RCAEval/e2e/circa.py)
- [Multi-source RCD 实现](https://github.com/phamquiluan/RCAEval/blob/4695aa69f4f1f57b9094ca04ff235908b73a8e24/RCAEval/e2e/mmrcd.py)
- [RCAEval evaluator](https://github.com/phamquiluan/RCAEval/blob/4695aa69f4f1f57b9094ca04ff235908b73a8e24/RCAEval/benchmark/evaluation.py)
- [RCAEval paper, arXiv v5](https://arxiv.org/abs/2412.17015v5)
- [RCAEval dataset DOI](https://doi.org/10.5281/zenodo.14590730)
