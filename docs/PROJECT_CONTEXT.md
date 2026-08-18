# 项目背景与总体研究框架

> 文档版本：`project_context_v2`
> 状态：背景基线  
> 最近更新：2026-08-18

## 1. 项目定位

当前毕设题目暂为：

> **面向微服务系统的结构感知多模态异常检测方法研究**

实际研究已形成两项相互关联但应独立验证的任务：

1. **研究内容一：连续运行场景中的节点级多模态异常检测。** 已有方法为 Ada-MGAD，主要使用 GAIA 与 MSDS。
2. **研究内容二：已确认故障事件条件下的服务级根因排序。** 计划开发不依赖 Ada-MGAD 隐表示的 Standalone RCA，主要使用 GAIA 与 RCAEval RE2-OB。

两项方法最终通过标准化的故障事件接口做系统级集成：

```text
连续 Metrics / Logs / Traces
             |
             v
         Ada-MGAD
      节点异常检测/事件触发
             |
             v
      标准化 Fault Event
             |
             v
       Standalone RCA
      根因服务完整排序
```

核心原则是：**任务级解耦，事件级集成**。Standalone RCA 不直接消费 Ada-MGAD latent features；Ada-MGAD 只在最终系统实验中提供检测触发或事件边界。

## 2. 研究问题

### RQ1：异常检测

> 在连续运行的微服务系统中，如何联合建模多模态可观测数据与服务依赖结构，实现复杂异构环境下可靠的节点级异常检测？

对应 Ada-MGAD。当前代码可验证的核心机制包括：

- `DynamicGraphLearner`：在静态图约束下学习动态边权；
- `GatedCrossModalFusion`：对 Metric、Log、Trace 的时序注意信息进行门控融合；
- 编码器—解码器一致性相关的对比损失；
- 分类分数与重构分数融合的节点异常评分。

当前仓库的 `src/` 与 `util/` 仍是这条异常检测代码线。投稿版、Camera Ready 和开题综述未存放在本工作区，因此论文数值和审稿后口径只能暂按参考对话记录，不能由本仓库独立复核。

### RQ2：根因分析

> 在故障事件已被识别的条件下，如何联合利用多模态故障演化信息与微服务间关联关系，对候选服务进行相对根因建模，从而区分故障源服务与故障传播产生的受影响服务，实现准确的服务级根因排序？

任务鸿沟是：

- 异常检测回答“哪些服务表现异常”；
- RCA 回答“多个异常或受影响服务中，哪个服务是故障源”。

故障传播会使根因服务与下游受影响服务同时出现异常，因此异常程度与根因可能性并不存在简单的一一对应关系。

## 3. 已冻结的任务边界

### Ada-MGAD

- 任务：连续时间轴、滑动窗口、节点级异常检测。
- 数据：GAIA + MSDS。
- 输出：每个时刻/窗口中各服务节点的异常分数或标签。
- 研究状态：异常检测方法和代码已存在；论文材料和完整实验产物不在当前仓库。

### Standalone RCA

- 任务：独立、故障事件级、服务级根因排序。
- 主协议：给定故障分析锚点的 Oracle RCA。
- 输入：事件上下文中的 Metrics、Logs、Traces、合法拓扑和候选服务集合。
- 输出：不重复、覆盖全部候选服务的完整 ranking。
- 主数据：GAIA + RCAEval RE2-OB。
- 扩展数据：RE2-TT，仅在主方法稳定后用于复杂场景验证。
- 禁止：把 indicator-level RCA、连续异常检测或端到端诊断结果混入主任务 claim。

## 4. 数据集角色

| 数据集 | 角色 | 当前结论 |
|---|---|---|
| GAIA | Bridge Dataset | 同时支持 Ada-MGAD 异常检测、fault injection 事件和最终 AD→RCA 集成 |
| MSDS | 异常检测验证 | 用于 Ada-MGAD，不要求承担 RCA |
| RCAEval RE2-OB | 主外部 RCA benchmark | 官方列出 90 个 failure cases，含 Metrics、Logs、Traces 和根因标注 |
| RCAEval RE2-TT | 扩展验证 | 候选服务空间与指标规模更大，P1/P2 不优先 |
| RCAEval RE2-SS | 非主线 | 官方说明缺少 Trace，不适合当前三模态主协议 |
| Nezha OnlineBoutique | 非主线 | 若以后使用，不能与 RE2-OB 夸张表述为两个完全独立的应用家族 |

RCAEval 官方仓库当前说明其 benchmark 包含 9 个数据集、735 个 failure cases，并支持 coarse-grained 与 fine-grained RCA；RE2-OB 为 90 cases，带 Logs 与 Traces。外部事实核验日期为 2026-08-18，参见文末官方来源。

## 5. 代码线与资产边界

参考对话曾规划三条代码线：

```text
main             # Ada-MGAD 异常检测基线
rca              # 旧 Ada-MGAD-dependent RCA Pilot，冻结
rca-standalone   # 新的独立 RCA 研究
pipeline         # 未来按需建立，负责事件级系统集成
```

当前工作区能直接验证的情况：

- 本地与远端可见分支只有 `rca-standalone` / `origin/rca-standalone`；
- 当前提交为 `ab31282`（`first commit`）；
- 当前树已有 label-separated `RCACaseInput`/`RCACaseLabel`、RCA evaluator、
  P1 sanity baselines、数据诊断、split/cohort/source manifests 与 51 个测试；
- 当前 GAIA 预处理器会解析 `run_table_2021-07.csv` 并保存 `label_events.csv`，随后又把事件映射到 30 s `label.csv`；P1 可以复用前者的事件解析，但不能把后者的窗口标签直接当成新的 RCA case 协议；
- P1 G1–G8 与 reproducibility closeout 已通过，当前进入 P2；
- 参考对话提到的旧 `rca` Pilot 分支及其实验产物仍不在当前 clone 中。

因此，文档继续对旧 Pilot 使用“对话报告”标签；任何要写入论文的 Pilot 结论都
必须先恢复原分支/产物或在当前协议下重新复现。当前可复核的 P1 结果只作为
benchmark/data sanity evidence，不代表 H1/H2/H3 已成立。

## 6. 毕业论文叙事基线

建议的 8 章结构仍为暂定框架：

1. 绪论；
2. 微服务多模态故障诊断相关理论与研究现状；
3. 多模态故障诊断问题建模与总体框架；
4. Ada-MGAD 异常检测方法；
5. Ada-MGAD 实验与分析；
6. 面向故障事件的 Standalone RCA 方法；
7. RCA 与异常检测—根因分析全流程实验；
8. 总结与展望。

第 6、7 章的方法细节不能先于实验被“写死”。当前题目是否增加“根因分析”应在 RCA 形成独立方法、双数据集证据和完整消融后再与导师确认。

## 7. Claim 边界

当前可以说：

- Ada-MGAD 是节点级多模态异常检测方法；
- Standalone RCA 的任务、数据角色和 benchmark 协议已冻结，P1 已完成；
- Random、Root Frequency、Metric Change sanity baselines 已在 GAIA 与 RE2-OB
  上复现；
- 旧 Pilot 提供了“跨节点相对比较值得研究”的前期线索。

当前不能说：

- P2 的多模态、阶段、相对比较或结构方法已经取得结果；
- H1/H2/H3 已被验证；
- 结构信息必然提升 RCA；
- 当前方法解决 indicator-level RCA；
- Oracle RCA 主表代表端到端故障诊断性能；
- 旧窗口级 Pilot 数字已由当前分支复现。

## 8. 官方来源

- [RCAEval 官方仓库](https://github.com/phamquiluan/RCAEval)
- [RCAEval 官方论文（arXiv）](https://arxiv.org/abs/2412.17015)
- [RCAEval 当前 evaluator 实现](https://github.com/phamquiluan/RCAEval/blob/main/RCAEval/benchmark/evaluation.py)
