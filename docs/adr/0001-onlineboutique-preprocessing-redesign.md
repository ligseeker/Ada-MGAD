# OnlineBoutique (Nezha) 预处理重构

Ada-MGAD 在 OnlineBoutique 上效果不佳，排查发现日志模态预处理有真 bug（Go 服务 adservice/cartservice 的消息抽取回退到整段 JSON 包装串）、Drain 配置弱（sim_th=0.5、`/ : .` 额外分隔符切碎模板）、且产物缺绝对时间无法人工核对。2026-08-06 决定按 Nezha 原仓库的解析方式重构日志模态，并同步审查 metric/trace。

核心决策：

1. **按服务抽取日志消息**：adservice/cartservice 取 `json['log']` 原文，其余 8 服务双层解析取 `message`（照搬 Nezha `log_parsing.py`）。
2. **Drain 配置原样照搬** `drain3_hipster.ini`（sim_th=0.9, depth=4, max_clusters=1024, extra_delimiters=["_"]，20 条掩码），不自创修改。
3. **模板词表在两天（训练天 08-22 + 测试天 08-23）全部日志上合并挖掘**（Nezha 原做法）。曾考虑仅用训练天挖掘并冻结词表以防泄漏，被明确否决：接受词表含测试天信息这一事实，解读结果时须知情。
4. **日志特征 = K 个模板计数 + ERROR/WARNING/INFO 计数 + log_total**（`log_len = K+4`）。数据实证：两天 396 万行日志中 ERROR 6413、WARNING 1190，只存在这三个级别。adservice 用文本正则提取级别；cartservice 无级别标记，级别列恒为 0，不伪造。
5. **Metric 扩至 10 KPI**：原 8 个 + PodWorkload(Ops)、PodClientLatencyP99(s)。不引入绝对量（与 Rate 冗余）和 Node 级指标（跨服务伪相关）。按天 robust z-score、60s 桶、ffill 保持不变。
6. **Trace 原样保留**：跨服务边、duration 求和、按天 log1p÷(均值×10)、14 类 OperationName。
7. **双轨时间戳**：内部保留 0 基桶序号（loader/模型不动），metric/log/trace CSV 与 bounds.pkl 增加绝对时间（epoch + UTC）列；另产出 `label.csv`、`fault_alignment.csv`（56 条故障逐条可核对）、`log_templates.csv`、`miner_state.bin`。
8. 切分固定 `test_experiment=1`（08-22 训练 / 08-23 测试），construct_data 无故障段继续合并进时间轴。故障时长 180s（fault_list.json 已含 duration 字段）。
