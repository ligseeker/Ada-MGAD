# 新数据集验证总计划（SN / TT / Nezha-OnlineBoutique）

> 创建日期: 2026-08-05
> 分支: exp
> 状态图例: [ ] 未开始 | [~] 进行中 | [x] 完成 | [!] 受阻

## 1. 背景与目标

Ada-MGAD 已在 GAIA、MSDS 上完成验证。本计划将其扩展到三个新数据集：

| 数据集 | 服务数 | 数据形态 | 备注 |
|--------|:-----:|----------|------|
| Eadro SN | 12 | 7 个离散实验（4 故障 + 3 无故障），~23.5 分钟/个 | 路径 `/home/zhangll24/RCA_project/Eadro/SN_Dataset` |
| Eadro TT | 27 | 11 个离散实验，~107 分钟/个 | 全量读取会爆内存，先做部分实验 |
| Nezha OnlineBoutique | 10 | 离散 1 分钟窗口（rca_data + construct_data 基线） | 跳过 Nezha-Trainticket（模态不一致） |

**目标**：在保持 Ada-MGAD 逐点评测协议的前提下，在新数据集上得到合理水平的 F1（不追求复现 Eadro 的 0.98）。

## 2. 关键事实与决策记录

### 2.1 关键事实

1. **Eadro 0.98 F1 是窗口级检测指标**（`Eadro/codes/base.py:25-62`）：每个故障注入窗口整体判断"有无故障"并按 culprit 排序，F1 按窗口统计 TP/FP/FN。与 Ada-MGAD 的逐点（秒×节点）binary F1（`util/util.py:13-33`）不同协议，**不可直接对标**。
2. MSTGAD 版 SN 适配即使用随机分层划分，测试 F1 仍为 0（train F1=0.49），说明问题不止在划分方式；嫌疑还包括：trace/metric 时间对齐（SN 存在 ~30000s 时钟偏移）、标签密度、日志模态近乎无效（仅 4–13 个模板）。
3. Ada-MGAD-exp 目前无 SN/TT/Nezha 适配代码；MSTGAD 仓库（`util/SN`、`util/TT`、`util/OnlineBoutique`）有可参考的预处理实现。
4. Ada-MGAD 原划分为时序前 70%/后 30%（`util/runtime.py:55`），适用于 GAIA/MSDS 连续数据，不适用于离散实验型数据集。

### 2.2 决策（2026-08-05 拷问会话确认）

| # | 决策 | 结论 |
|---|------|------|
| D1 | 成功标准 | 保持 Ada-MGAD 逐点评测协议，达到合理水平即可，不强追 0.98 |
| D2 | 数据集范围 | SN + TT + Nezha-OnlineBoutique；跳过 Nezha-Trainticket |
| D3 | 工作流 | 先在 MSTGAD 现有 SN 适配上诊断，定位根因后再移植到 Ada-MGAD-exp |
| D4 | 划分策略 | 实验级划分（SN 留一交叉验证；TT/Nezha 按实验约 7/3） |
| D5 | 标签语义 | 精确故障窗口 [start, start+duration]，不加缓冲 |
| D6 | TT 内存 | 先在部分实验上验证；后续内存到位再扩全量 |
| D7 | 基线 | 只跑 Ada-MGAD，基线数字引用论文 |
| D8 | 工程 | 本文档为总计划，进度同步更新；每任务完成后 commit |
| D9 | subagent 分工 | 主对话设计方案；关键任务 subagent 用 Qwen3.8-Max，其余 Qwen3.7-Plus（effort=auto）。注：Agent 工具无模型参数，尽量经 Workflow 的 model 选项实现，不可行时主对话执行并记录 |

## 3. 阶段计划与进度

### Phase 1: SN 诊断（在 MSTGAD 上） [~]

| 任务 | 状态 | 产出 |
|------|:----:|------|
| 1.1 验证预处理产物正确性：故障窗口内 metric 是否可见异常形态（如 cpu_load 时 CPU 上升）；trace/metric 时间对齐是否正确 | [x] | 见 4.1 |
| 1.2 划分方式对照：时序 70/30 vs 随机分层 vs 实验级留一 | [~] | 见 4.1 结论 F2；拟改在 Ada-MGAD 侧做对照 |
| 1.3 模态消融：metric-only / metric+trace / 全模态 | [ ] | 拟改在 Ada-MGAD 侧做 |
| 1.4 标签密度与类别不平衡分析 | [x] | 每实验 9 故障×120s，逐点异常率 ~5.8–6.4% |
| 1.5 诊断报告与根因结论，更新本文档 | [~] | 初步结论见 4.1 |

### Phase 2: SN 移植到 Ada-MGAD-exp [ ]

| 任务 | 状态 | 产出 |
|------|:----:|------|
| 2.1 新建 `util/SN/`（constant/parser/pre/data），复用 MSTGAD 修正后的预处理逻辑 | [ ] | 代码 |
| 2.2 注册 `DATASET_PROFILES`，支持实验级划分 | [ ] | 代码 |
| 2.3 训练与评估，记录指标 | [ ] | 结果 |

### Phase 3: Nezha-OnlineBoutique 适配 [ ]

| 任务 | 状态 | 产出 |
|------|:----:|------|
| 3.1 新建 `util/Nezha/`：日志-Trace 解耦、离散窗口对齐、dependency.csv 构图、fault_list 标签 | [ ] | 代码 |
| 3.2 训练与评估 | [ ] | 结果 |

### Phase 4: TT 适配（部分实验） [ ]

| 任务 | 状态 | 产出 |
|------|:----:|------|
| 4.1 新建 `util/TT/`：逐实验流式预处理，先选部分实验 | [ ] | 代码 |
| 4.2 训练与评估 | [ ] | 结果 |

### Phase 5: 汇总 [ ]

| 任务 | 状态 | 产出 |
|------|:----:|------|
| 5.1 汇总三数据集结果，更新本文档与 README | [ ] | 最终报告 |

## 4. 诊断记录

### 4.1 SN 首轮诊断（2026-08-05）

脚本：`scripts/diag/sn_diag_data.py`、`scripts/diag/sn_trivial_detector.py`

**F1 信号存在且标签对齐**：cpu_load 故障窗口内故障服务 CPU 上升 2.6–1000+ 倍；network_delay/loss 窗口内 rx/tx 降至 0.09–0.87 倍。metric 时间轴（epoch）与 fault JSON、日志（UTC+8 本地时间经 mktime）一致。

**F2 逐点检测存在天然上限**：不做任何学习，仅用逐节点鲁棒 z 分数（median/MAD，|z|>3–6）的平凡检测器，4 个实验汇总逐点 F1 仅 0.24–0.33（召回 ~0.4）。即 SN 上逐点 F1 的合理预期是中等水平，与 Eadro 的窗口级 0.98 无对比意义。MSTGAD 单实验结果（F1=0.26, AUC=0.88）已接近该信号上限，单实验流程本身不算坏。

**F3 多实验拼接是主要崩坏源**：历史训练日志显示多实验拼接后测试 F1≈0、AUC<0.5（系统性问题）。嫌疑机制：
- 全局 z-score 跨实验归一化（`preprocess_all_experiments`），各实验正常基线不同，故障统计量污染全局均值/方差；
- 滑窗跨越实验拼接边界（`data.py:_transform` 在拼接时间轴上连续滑动）；
- trace 对齐用"最早 span 对齐 fault JSON start"的启发式（`preprocess.py:430-449`），校正后仍有 13–36% span 落在时间轴外，各实验偏移不一致。

**F4 数据集本身的问题**：
- 第 2 个实验 spans.json 损坏（无 trace）；
- 第 3 个实验注入了 `nginx-thrift` 容器故障，不在 12 个监控服务内，标签丢失（该实验只有 6 个可标注故障）；
- no-fault 实验 3 个均为未解压的 tar.xz，现有流程只找到 1 个已解压目录。

**结论**：移植到 Ada-MGAD 时必须采用——逐实验预处理与归一化（统计量只来自训练实验）、不跨实验边界滑窗、实验级划分、逐实验 trace 对齐校验。划分方式与模态消融对照改在 Ada-MGAD 侧进行（MSTGAD 单次训练 25–50 分钟，其流水线即将被替换，继续在其上做大规模对照性价比低）。

## 5. 进度日志

- 2026-08-05: 完成拷问会话，敲定 D1–D9 决策；建立 CONTEXT.md；创建本计划文档。
- 2026-08-05: SN 首轮数据级诊断完成（信号存在、逐点检测天然上限 F1≈0.24–0.33、多实验拼接为崩坏主因）；详见 4.1。
