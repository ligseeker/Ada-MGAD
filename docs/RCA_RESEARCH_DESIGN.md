# Standalone RCA Research Design V0.2

> 决策状态：**用户已确认，作为当前研究基线冻结**  
> 冻结来源：参考对话 `6a82610b-1ca8-83ea-8662-6c1d7a91cb72`  
> 当前执行阶段：P1 Benchmark Layer  
> 最近整理：2026-08-18

## 1. 核心定位

> RCA 作为独立的故障事件级根因排序任务进行研究，以 GAIA 和 RCAEval RE2-OB 为主要验证数据集；Ada-MGAD 不参与 RCA 表征学习，仅在最终系统实验中负责从连续运行数据中提供故障触发信息。

Standalone RCA 的主任务不是异常节点分类，而是：在一次已确认故障事件中，对所有合法候选服务进行相对比较，输出根因服务的完整排名。

## 2. 形式化定义

对故障 case (E_i)，定义：

\[
E_i=(t_i^0,\mathcal X_i,V_i,v_i^*,c_i),
\]

其中：

- (t_i^0)：故障分析锚点；
- \(\mathcal X_i=\{X_i^m,X_i^l,X_i^t\}\)：Metrics、Logs、Traces 事件上下文；
- (V_i)：测试时可合法确定的候选服务集合；
- (v_i^*)：真实根因服务，只用于训练、评价或错误分析；
- (c_i)：故障类型，只用于训练标签或分组分析，不作为测试输入。

模型学习：

\[
f_{\mathrm{RCA}}(t_i^0,\mathcal X_i,G_i,V_i)
\rightarrow
R_i=[v_{(1)},v_{(2)},\ldots,v_{(|V_i|)}].
\]

输出必须是 service-level ranking；不声称定位某个具体 KPI 或 indicator。

## 3. 核心科学问题

> 受影响服务可能与根因服务同时表现异常，如何利用故障事件中的多模态演化差异、跨服务相对关系以及合法的结构信息，区分故障源与传播症状？

这一定义把“异常识别”与“因果归因/根因归属”区分开，也解释了为什么单纯按异常分数排序只能作为 baseline，而不是最终 RCA 任务定义。

## 4. 三个可检验研究假设

### H1：Event-stage Representation

整段上下文的单一 pooling 可能丢失故障发生与传播过程。候选机制是围绕 (t_0) 构造：

- Baseline：\([t_0-T_b,t_0)\)；
- Onset：\([t_0,t_0+T_o)\)；
- Impact：\([t_0+T_o,t_0+T_i)\)。

假设：根因服务和受影响服务的异常演化模式不同，阶段差分比全窗口 pooling 更能区分二者。

**当前状态：待验证。** P1 只保存完整 raw context，不在数据层强制三段切分。

### H2：Cross-node Relative Ranking

逐服务独立打分：

\[
r_v=g(h_v)
\]

可能不如在同一 case 内联合比较：

\[
\tilde h_1,\ldots,\tilde h_N
=\operatorname{CrossNode}(h_1,\ldots,h_N),
\quad r_v=g(\tilde h_v).
\]

假设：RCA 的关键是解释“为何 A 比 B/C/D 更像根因”，而不是独立判断“A 是否异常”。

**当前状态：有旧 Pilot 线索，但尚无 event-level / standalone 证据。**

### H3：Structural Interaction / Structure-aware Distinction

与根因结构相近且具有强异常表现的服务是更困难的 negatives。结构可能用于事件特异的交互、传播特征或训练期 hard-negative 策略。

**当前状态：待验证。** 只有当结构模块在 GAIA 与 RE2-OB 上相对强的非结构 baseline 都产生稳定增量时，才能形成“结构感知 RCA”的正式 claim。

## 5. 三层评价协议

### Protocol A：Oracle RCA（主协议）

给定真实 fault case 与分析锚点 (t_0)，评价 Standalone RCA 的根因排序能力。Oracle 只提供“何时/对哪个 case 做 RCA”，不提供根因。

主表必须标为 **Oracle RCA**，不能称为端到端故障诊断。

### Protocol B：Trigger Robustness（方法定型后）

令：

\[
\hat t_0=t_0+\Delta t,
\]

观察分析锚点提前或延迟时的性能变化。候选延迟为 30 s、60 s、120 s，但具体取值需由两数据集的采样粒度与有效上下文诊断后确定，不能现在写死。

### Protocol C：Ada-MGAD-triggered RCA（最终系统实验）

Ada-MGAD 从连续数据中产生 detected event，再由 Standalone RCA 排序。比较：

\[
\Delta P=P_{\mathrm{Oracle}}-P_{\mathrm{Detected}}.
\]

分析差异来自 trigger delay、event boundary error、早期证据缺失、误报或漏报。该协议是系统级贡献，不是 Standalone RCA 成立的前提。

## 6. 数据角色

- **GAIA**：Bridge Dataset；同时承担 Oracle RCA、Ada-MGAD-triggered RCA 和两协议差值分析。
- **RE2-OB**：主外部 Standalone RCA benchmark；官方当前列出 90 个 cases，含三模态数据。
- **RE2-TT**：方法成熟后的复杂度扩展，不是 P1 最低完成线。
- **MSDS**：保持为异常检测数据集，不强制构造 RCA 标签。

## 7. 方法应由实验逐层长出

```text
B0  Random / Root-frequency / Metric-change sanity baselines
 |
B1  Single-modal RCA
 |
B2  Multimodal independent scorer
 |
M1  + Event-stage representation       (检验 H1)
 |
M2  + Cross-node relative ranking      (检验 H2)
 |
M3  + Structural interaction/negatives (检验 H3)
```

对应问题：

1. 多模态是否稳定优于单模态？
2. Event-stage 是否优于 whole-context pooling？
3. Cross-node 是否优于 independent scoring？
4. Structure 是否在强 baseline 上产生稳定增量？

在 P1 完成前不新增任何 Standalone RCA 模型实现；现有 Ada-MGAD `src/model.py` 保持不动，也不提前命名一个“大而全”的 RCA 模型。

## 8. Baseline 选择原则

P1 只实现：

- Random；
- Root Frequency Prior（只从训练 fold 统计）；
- Metric Change Score。

P1.5 再接代表性官方方法：

- 经典/指标类：BARO、RCD、CIRCA；
- 多源类：Multi-source BARO、Multi-source RCD、Multi-source CIRCA。

TORAI、EventADL 或其他后来加入 RCAEval 的方法是否进入主表，必须根据输入模态、数据版本、任务粒度、环境和公平性另行审查，不能因为官方仓库存在就自动纳入。

## 9. Go / No-Go 规则

- H1/H2/H3 均是实验假设，不是预设创新点。
- 旧窗口级 Pilot 只能解释研究动机，不能作为 standalone/event-level 结论。
- H3 若不能在 GAIA 与 RE2-OB 上稳定超过强非结构 baseline，则不使用“结构感知 RCA”作为主 claim。
- 即使 H3 不成立，RCA 仍可如实定位为事件级多模态跨服务比较式根因排序。
- Ada-MGAD 与 RCA 始终独立训练、独立评价，只通过 fault-event interface 集成。

## 10. 版本变更规则

V0.2 是当前冻结基线。只有出现以下证据之一才升级设计版本：

- P1 数据诊断表明两个数据集无法使用同一 service-level schema；
- label/candidate 语义无法无泄漏对齐；
- 官方 benchmark 版本变化导致评价口径实质改变；
- P2 预实验否定核心任务设定而非仅否定某个模型模块；
- 用户明确批准新的研究方向。

任何升级必须在 [RESEARCH_STATUS.md](RESEARCH_STATUS.md) 的决策日志记录“旧决策—新证据—新决策”。
