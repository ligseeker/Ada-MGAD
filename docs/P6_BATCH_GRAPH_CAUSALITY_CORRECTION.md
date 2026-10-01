# 动态图批内前视：历史解释更正

2026-10-01，独立修复分支 `experiment/p6-c0-window-causal`。

## 确认的缺陷

`src/model_util.py` 中 `DynamicGraphLearner` 的 `last` 和 `mean` 都沿 batch 聚合输入；得到 `[N,N]` 的共享动态图，随后对各窗口广播。顺序 batch32 每条相隔30s，因此最早窗口最多依赖其后930s的输入。首窗口保持不变、仅改变同行的合成反例能改变其 graph 和 logit。时间匹配要求 `0≤delay≤60s` 不能消除这条输入前视路径。

它是未来 telemetry 耦合，不是 GT/root-service 标签泄漏。没有据此证明指标全部被抬高；具体影响需要审计和一致重训。

## 影响范围及结果使用

- P5 声明执行 commit `7dea779fc5196218a43d86a9777b591d290f047b` 和 C0 `cedc4a2bd7492933a8295067c8075e631cbf3df9` 具有此实现。
- C1-v2 detector 沿用此 learner；其 anchors、B/C 配对和下游 E2E 应解释为批条件下的回顾性结果。不能继续称为在线因果 detector 或因果 OOS anchor。
- 旧固定 scores 的 matching、RCA 排序和指标算术仍可核验；这些数值没有自动获得逐窗因果性。
- 旧 C0F Fit/Validation inference source 本来仍为 `UNVERIFIED`；本次发现不消除该独立限制。
- 既有报告、checkpoint、锁和原始产物保持不变；本文件单独记录更正。原候选 NO-GO 是历史开发决定，不是逐窗口因果基线的候选对照。

## 修复边界

新 `WindowDynamicGraphLearner` 保留每个窗口的 `[B,N,N]` 图。新 C0 显式设 `graph_batch_scope=window`，原 default=batch 仅用于历史复现。必须使用该机制重新训练；对旧 checkpoint 切换模式只是诊断。此次不重新运行 Test，也不自动重建 P5/C1 全链路。

另一个独立限制：原共享 preprocessing 在70% Train拟合，包含 C0 Validation和 C1后续fold的无标签输入；修复batch耦合不会把该设计变成 strict preprocessing-OOS。
