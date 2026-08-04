# Ada-MGAD RCA 扩展：开发统筹与实验日志

> 本文档是 RCA 研究项目的**持久化开发计划 + 实验日志**。每完成一个计划步骤或一轮实验，必须同步更新本文档（状态、结果、下一步）。新对话接手时：**先读本文件，再读 `CONTEXT.md`（术语与决策共识），然后按"下一步"继续推进**。

最后更新：2026-07-28（setattn + E1/E2 实验完成后）

---

## 1. 研究目标

在已发表（IEEE Cloud 2026）的 Ada-MGAD 多模态异常**检测**模型之上，扩展**根因分析（RCA）**能力：对 GAIA 数据集每个异常观测窗口，输出 10 个服务实例节点的 root-cause 排序。完整决策共识见 `CONTEXT.md`（Q1–Q10，含任务定义、指标、标签语义、baseline 阵容、工程组织）。

核心设定（速查）：
- 任务：10 节点 root-cause ranking，纯窗口级样本，Oracle 评估（检测不 gate RCA）
- Ground truth：4 类故障（login failure / memory_anomalies / file moving program / access permission denied），排除 cpu（3s 低于可观测下限）与非故障标签
- 指标：HR@1/3/5、MRR（per-type + **macro 平均为头条**）、NDCG@3/5
- 方法：冻结检测器 + 训练定位模块（两阶段）；主表 50% 标签预算，附录标签效率曲线
- 范围：**仅 GAIA**（MSDS 根因标签系伪造且根因退化，已剔除——见 §6 陷阱）
- 工程：`Ada-MGAD` 仓库 `rca` 分支；5 种子 mean±std

## 2. 总体开发计划与状态

| 步骤 | 内容 | 状态 |
|---|---|---|
| 1 | 切 `rca` 分支；`util/GAIA/build_rca_labels.py` 生成 `label_rca.csv` + 统计报告 | ✅ 完成 |
| 2 | `util/eval_rca.py` 评估框架 + 零成本基线（检测分数直接排序） | ✅ 完成 |
| 3 | 三证据定位模块（`src/rca/`）+ 消融 | ✅ 首轮完成（结论见 §4） |
| 4 | 标签效率曲线实验（兼检验"P/T 证据在低标签机制下有价值"假设） | ⬜ **下一步** |
| 5 | Baseline：启发式（已有）、DiagFusion（已有 GAIA 适配）、Eadro（需写 GAIA 适配器） | ⬜ 待做 |
| 6 | 主表 + 消融 + 标签效率曲线出图；论文分析小节（难类型拆解、孪生混淆分析） | ⬜ 待做 |

## 3. 关键资产与路径

| 资产 | 路径 |
|---|---|
| 代码（rca 分支） | `/home/zhangll24/RCA_project/Ada-MGAD` |
| RCA 标签 | `.../MSTGAD-dynamic/data/GAIA-pre/label_rca.csv` |
| 冻结检测器 checkpoint | `/home/zhangll24/project_2/MultimodalAD/MSTGAD-dynamic/result_old/MSTGAD-GAIA-save-c4b9704e-1773605775`（`my_f1_stage.ckpt`，与 Ada-MGAD `MyModel` state dict 完全兼容） |
| 特征缓存 | `result/rca-APT-seed42/features_{train,test}.npz`（复用：`--reuse_features`） |
| 实验结果 | `result/rca-eval-*.json`（仓库 `result/` 目录） |
| 训练好的 localizer | `result/rca-{config}-seed{42..46}/`（config.json + localizer.pt） |

数据规模：`label_rca.csv` 单根因窗口 23184 + multi-root 4314（主实验排除）；测试段单根因 7569（login 4773 / memory 2387 / access denied 286 / file moving 123）。

环境：conda 环境 `DAG`（python 3.8），GPU Tesla V100-32GB（空闲），系统 RAM 60GB。特征提取（train+test ~3.9 万窗口）约 2 分钟；单次 localizer 训练（复用特征）约 50 秒。

## 4. 实验日志

### 第 1 步：RCA 标签构建（commit `6527ef2`）
- `label_rca.csv`：timestamp × root_cause_id × fault_type × multi_root。floor 对齐 30s 窗口，与检测标签逻辑一致。
- 1878 个事件未匹配到网格（84 在网格起点前、1794 在 metric 数据空洞段），与原预处理行为一致。
- 发现：现有检测 `label.csv` 本就只标注入节点（根因导向），但含 ~1.4% 噪声（normal-freed/cpu/error/unknown）——**检测标签冻结不动**（Q9），RCA 标签独立。

### 第 2 步：零成本基线（commit `08af602`, `4564a94`）
检测分数直接排序的 per-type HR@1（**所有后续工作的基准线**）：

| fault type | n | detector HR@1 | MRR |
|---|---|---|---|
| access permission denied | 286 | 1.000 | 1.000 |
| file moving program | 123 | 1.000 | 1.000 |
| memory_anomalies | 2387 | 0.9535 | 0.965 |
| **login failure** | 4773 | **0.776** | 0.835 |
| **macro** | | **0.9324** | 0.9500 |

启示：两个类型已饱和；真正战场是 login failure（11s 短事件 + mob1/mob2 孪生服务难消歧）。trace_dev 启发式在 login 上 0.653（其他类型 <0.09），证明传播/trace 证据有互补价值。

### 第 3 步：定位模块迭代（commit `1cff0c5`, `a67a2e2`, `4b6cd8b`）

**v1：逐节点 MLP（4 配置 × 5 种子）**：macro HR@1 — A 0.9364 / AP 0.9397 / AT 0.9361 / APT 0.9400。提升小。
- **错误分析关键发现**：login 错误中 **44.3% 是 mob1↔mob2 孪生混淆** → 逐节点独立打分无法表达窗口内相对比较。

**v2：集合注意力（setattn，跨节点比较，5 种子）**：

| config | macro HR@1 | macro MRR | login HR@1 | memory HR@1 |
|---|---|---|---|---|
| detector | 0.9324 | 0.9500 | 0.776 | 0.9535 |
| MLP APT | 0.9400±.006 | 0.9571±.006 | 0.812 | 0.951 |
| **saA** | **0.9540±.001** | 0.9730±.001 | 0.873 | 0.9465 |
| saAPT | 0.9534±.000 | 0.9726±.000 | 0.872 | 0.9455 |

- login failure **+9.7 个点**（0.776→0.873），方差降一个数量级；macro HR@1 +2.2、MRR +2.3。
- **问题 1**：P/T 证据在所有配置下贡献为零（saA≈saAP≈saAT≈saAPT）。
- **问题 2**：memory 退化 -0.8（login 数量优势下的容量权衡）。

**v2 补验：E1 残差排序 / E2 类型平衡（各 5 种子）**：均为中性（±0.001），memory 退化非训练分布问题。**放弃这两个方向**。

### 当前结论与最优候选
- 最优：saA（0.9540）≈ saAPT（0.9534），差异不显著。
- 论文叙事倾向："冻结多模态检测器 + 跨节点比较重排模块"；P/T 证据的价值待第 4 步低标签实验裁决。

## 5. 下一步行动（第 4 步，已获用户确认方向）

**标签效率曲线实验**：saA vs saAP vs saAPT × 标签预算 {0.1, 0.25, 0.5, 1.0} × 3 种子（42/43/44）。
- 命令模板：`python util/train_rca.py --model_path $CK --data_path $DP --dataset_path $DS --out_dir result/rca-<tag>-b<budget>-seed<s> --groups <...> --arch setattn --label_budget <budget> --seed <s> --reuse_features result/rca-APT-seed42`
- 评估：`python util/eval_rca.py ... --localizer_path <每个 out_dir> --out result/rca-eval-budget.json`，再按预算聚合 macro HR@1/MRR 画曲线。
- 假设：P/T 先验结构在低标签（10%）时提供归纳偏置（saAP/saAPT > saA）；若仍中性，则贡献框定为跨节点比较重排，P/T 消融如实报告。

后续（第 5、6 步）：Eadro GAIA 适配器（`project_2/baselines/Eadro-GAIA/` 或仓库 `baselines/`）、DiagFusion 复跑（`baselines.zip` 内已有 GAIA 适配版）、主表出图。

## 6. 已知陷阱（不要重蹈）

1. **MSDS 根因标签是伪造的**：`MSDS-pre/groundtruth_*.pkl`/`fault_type_labels.*` 与真实注入记录（Rally 报告 31 次容器重启，全部命中同一节点）零对应，无任何生成脚本。RCA 不做 MSDS。`label.pkl` 是干净的，已发表检测结果不受影响。
2. **`util/` 缺 `__init__.py`**（已在 rca 分支补上）：否则从 `util/` 内跑脚本时 `util/util.py` 遮蔽整个包（'util' is not a package）。
3. **模型按固定 batch_size 预建边索引**：评估/特征提取必须补齐末批（`_pad_batch`），否则 GATv2 维度不匹配。
4. **zsh 不做未加引号的变量分词**：循环拼参数必须用数组（`arr+=(--flag x)` + `"${arr[@]}"`）。
5. **`trace_path.pkl` 被对称化**（pre_GAIA.py:570-571），有向调用图需从 `trace.csv` 的 src/dst_service 重建（`src/rca/evidence.py: build_call_graph`）。
6. **eval 表格行命名**：`--localizer_path` 行名来自目录名（`rca-<variant>-seed<N>`），同名会互相覆盖。
7. 特征缓存复用断言：`features_*.npz` 的 windows 顺序必须与当前数据划分一致（`--reuse_features` 会校验）。
