# Ada-MGAD

微服务多模态图异常检测方法（Ada-MGAD）的实验仓库。在 GAIA、MSDS 上已完成验证，正在扩展到 SN、TT、Nezha 数据集。

## Language

### 评测

**逐点异常检测 (Point-level Anomaly Detection)**:
Ada-MGAD 的原生任务与评测协议：对每一秒 × 每个微服务节点做二分类（正常/异常），报告 binary precision/recall/F1/AUC/AP。
_Avoid_: 窗口级检测（那是 Eadro 的协议，两者不可直接对比数字）

**窗口级检测 (Window-level Detection)**:
Eadro 论文的评测协议：每个故障注入窗口整体判断"有无故障"并按 culprit 排序，F1 按窗口统计 TP/FP/FN。论文中 0.98 F1 属于此协议。
_Avoid_: 与逐点 F1 混报

### 数据

**实验 (Experiment)**:
一次独立的故障注入运行（如 SN.2022-04-17T181245D2022-04-17T183616），含 metric/log/trace 与 fault JSON。是 SN/TT 数据的基本组织单元。
_Avoid_: 会话、run

**无故障实验 (No-fault Experiment)**:
不含故障注入的对照实验，用作正常样本来源。
_Avoid_: 基线（Nezha 中 construct_data 才是基线）

**故障窗口 (Fault Window)**:
fault JSON 给出的精确注入区间 [start, start+duration]。异常标签只覆盖该区间，不做前后缓冲扩展。
_Avoid_: 异常区间、故障段

**实验级划分 (Experiment-level Split)**:
SN/TT/Nezha 的训练/测试划分以整个实验为单位（如留一交叉验证或按实验 7/3），而非时间轴 70/30 切分。
_Avoid_: 时序切分（仅适用于 GAIA/MSDS 这类连续数据）

**离散时间窗口 (Discrete Time Windows)**:
Nezha 数据的组织方式：log/trace 按约 1 分钟的窗口文件离散存储，窗口间有间隔，不能跨窗口滑动。
_Avoid_: 连续时间序列
