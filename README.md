# Ada-MGAD + Ada-RCA：GAIA 双阶段异常检测与根因定位

本仓库当前不是单纯的 Ada-MGAD 异常检测复现目录，而是一个面向 GAIA
MicroSS 的两阶段研究实现：

~~~
GAIA Metric / Log / Trace
        │
        ▼
Train-only 多模态预处理（V2 schema）
        │
        ▼
Ada-MGAD-G：节点异常分数
        │
        ▼
事件触发、时间匹配与 detected onset
        │
        ▼
Ada-RCA-G：W300-B15 / 68D 表示与服务排序
        │
        ▼
检测、RCA、E2E diagnosis 指标
~~~

Ada-MGAD 和 Ada-RCA 的核心模型保持冻结；本仓库实现 GAIA 适配、数据预处理、
事件协议、双阶段编排、RCA 特征生成、基线和结果 provenance。

## 新 code-agent 的第一入口

开始任何代码探索或实验操作前，先读取：

1. [AGENTS.md](AGENTS.md)；
2. [docs/GAIA_P5_CURRENT_CONTEXT.md](docs/GAIA_P5_CURRENT_CONTEXT.md)；
3. 当前实验的 [final_report.md](experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440/final_report.md)。

GAIA_P5_CURRENT_CONTEXT.md 是当前状态、路径、指标、约束和已知问题的单一
上下文入口。详细审计文档保留完整决策依据和历史证据，不在 README 中重复。

## 当前研究状态

当前代码状态：

~~~
repository: /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
branch:     e2e-v2
HEAD:       7dea779fc5196218a43d86a9777b591d290f047b
~~~

最近一次完整 GAIA run：

~~~
experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440/
status: FORMAL_FULL_DATA_COMPLETE
~~~

该 run 在前一次进程被杀后续跑完成，当前没有活动训练进程。核心结果如下：

| 层级 | 主要结果 |
|---|---:|
| Ada-MGAD Test node F1 / AUC | 0.8693 / 0.9328 |
| Test event F1 / Precision / Recall | 0.7751 / 0.9827 / 0.6399 |
| Detected-anchor RCA AC@1 / AC@3 / AC@5 | 0.4657 / 0.9214 / 0.9830 |
| Full diagnosis F1@1 / F1@3 / F1@5 | 0.3611 / 0.7145 / 0.7623 |

当前结论：Ada-MGAD 节点检测较强；事件检测偏保守、漏检较多；Ada-RCA 在
detected anchor 下的 Top-1 定位是主要瓶颈。GT-anchor Ada-RCA 的 AC@1 为
0.9125，说明不能把问题简单归因于 RCA 优化器未收敛。

上面的状态和数字是当前 run 的快照。新的正式 run 完成后，先更新
GAIA_P5_CURRENT_CONTEXT.md 并保留旧 run，再按需更新本节；不要用新结果覆盖
历史实验目录或历史报告。

## 方法与数据契约

### Ada-MGAD 输入

GAIA 具有固定的 10 个 canonical service。当前 V2 预处理输出：

~~~
Metric node tensor: [T, 10, 48]
Log node tensor:    [T, 10, 32]
Trace edge tensor:  [T, 10, 10, 8]
~~~

所有节点的 tensor feature dimension 必须一致，但这不表示每个服务真实拥有
相同的原始指标。相同 slot 的语义必须固定；缺失或不适用指标由冻结 schema
和 mask 语义表示。

### 事件阶段

- 30 秒 detector grid、10-bin detector window；
- t_hat = prediction_available_time = target_bin_end；
- 系统分数为各节点异常分数的最大值；
- 事件阈值只在 Train 上进行 exact unique-score sweep；
- 采用 causal、one-to-one、maximum-cardinality/minimum-delay 匹配；
- Test 只用于冻结方案完成后的最终评估。

### Ada-RCA 阶段

- 使用独立的 GAIA raw telemetry adapter，不直接把 Ada-MGAD tensor 当作 RCA 输入；
- RCA context 为 W300-B15，candidate service 数为 10；
- 每个 candidate 的冻结 Z2 表示为 68D；
- StandardScaler 只在 Train candidate rows 上拟合；
- oracle anchor 使用 GT 起点，detected anchor 使用检测得到的时间点；
- 结果含 weighted overall、root macro、fault macro 和 full diagnosis；
- 当前标签语义是 labelled injected/fault service，不代表独立的因果证明。

### Train/Test 防泄漏边界

以下内容只能由 Train 决定：

- Metric schema、质量过滤、冗余过滤和 normalization；
- Log vocabulary、stable/rare/UNK 分组和 scaling；
- Trace graph/statistics；
- event threshold；
- RCA scaler 和模型参数；
- checkpoint 选择。

Test 只能执行 frozen transform、推理和最终指标计算。不得根据 Test F1、Test
label 或 Test ranking 反向修改 preprocessing、阈值、schema 或模型。

## 代码与产物布局

~~~
src/model.py                         Ada-MGAD 核心模型
src/model_util.py                    Ada-MGAD 训练/推理工具
src/e2e/gaia_preprocessing/          V2 Metric/Log/Trace 预处理 primitives
src/e2e/ad_preprocess.py             Ada-MGAD 输入适配
src/e2e/event_detection.py           事件触发、匹配和指标
src/e2e/gaia_rca_adapter.py          GAIA raw RCA adapter
src/e2e/rca_features.py              W300-B15 / 68D 特征生成
src/e2e/rca_model.py                 Ada-RCA 训练与排序
src/e2e/e2e_evaluation.py            E2E diagnosis 指标
scripts/p5/run_i1_pipeline.py       唯一编排入口
configs/e2e/gaia_p5_v3_preprocessing_v2.json
                                      当前执行配置
~~~

共享输入目录只读：

~~~
data/p5/v3_preprocessing_v2/ad/
artifacts/p5/v3_preprocessing_v2/ad/
artifacts/p5/v3_preprocessing_v2/protocol/
data/p5/v3/rca_raw_index/
data/p5/v3/rca_features_gt/
~~~

每次实验的可变输出必须进入独立目录：

~~~
experiments/p5/gaia_v2/<unique_run_id>/
├── checkpoint/
├── ad/
├── events/
├── rca/
├── logs/
├── run_state.json
├── run_manifest.json
└── final_report.md
~~~

不要复用已完成 run 的 checkpoint、calibration、prediction、RCA model 或日志，
也不要把结果写回共享输入根目录。

`experiments/p5/gaia_v2/<run_id>/` 下的每个 run 都只向远程仓库保留用于分析的
核心记录：训练摘要、事件指标、RCA/E2E 指标与分层报告、配置、状态和
feature-health manifest。逐时间窗/逐案例的预测 CSV（包括 AD Test 分数、事件匹配
表和 RCA 排名）保留在本地，不提交到远程仓库；checkpoint、日志、Train 全量预测、
RCA 模型权重、完整 68D `.npy` 特征和并行 shard 也不会提交。`.gitignore` 对未来
新建的 run 目录自动应用同一白名单。

## 环境

最近一次 GPU Ada-MGAD run 使用 DAG 环境：

~~~
Python 3.8
PyTorch 1.12.0
CUDA 11.3
GPU available: true
~~~

RCA 的 canonical source commit 为：

~~~
a2c620922e7c0ab3615d34654d4a3690d1b22c8e
~~~

推荐把 Ada-MGAD（GPU）和 RCA/post-ad（CPU/兼容环境）分阶段运行；只有当一个
环境同时具备两套依赖时，才使用 train-evaluate 组合入口。具体 worker、环境
检查和恢复规则见 [docs/GAIA_V3_IMPLEMENTATION.md](docs/GAIA_V3_IMPLEMENTATION.md)。

## 运行入口

### 1. 只读检查当前 run

~~~
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
git status --short
RUN=experiments/p5/gaia_v2/gaia-v2-seed42-20260915T181440
grep -E '"status"|"updated_at"' "$RUN/run_state.json"
sed -n '1,100p' "$RUN/final_report.md"
~~~

### 2. Smoke

~~~
python scripts/p5/run_i1_pipeline.py smoke --gpu false
~~~

Smoke 只验证代码、shape、有限值和最小流程，不能当作 GAIA 正式结果。

### 3. 共享 V2 preprocessing

只有在共享 V2 manifest 不存在或需要新 revision 时才运行。正式全量 preprocessing
不应与另一条 preprocessing 同时写同一输出根：

~~~
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
PY=/home/zhangll24/miniconda3/envs/DAG/bin/python
CONFIG=configs/e2e/gaia_p5_v3_preprocessing_v2.json

"$PY" scripts/p5/run_i1_pipeline.py preprocess \
  --config "$CONFIG" \
  --raw-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS \
  --raw-workers 8 \
  --feature-workers 24 \
  --chunk-rows 150000 \
  --case-chunk-size 128 \
  --start-method spawn \
  --gpu false
~~~

已有完整 manifest 时不要重复执行；需要新的输入或 schema 时，应使用新的 revision
和明确的输出根，而不是覆盖现有数据。

### 4. 新的正式双阶段 run

先创建唯一、尚不存在的 RUN_DIR。同一个 RUN_DIR 贯穿 AD 和 post-ad 两阶段：

~~~
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
source /home/zhangll24/miniconda3/etc/profile.d/conda.sh
conda activate DAG

PY=/home/zhangll24/miniconda3/envs/DAG/bin/python
CONFIG=configs/e2e/gaia_p5_v3_preprocessing_v2.json
RUN_ID="gaia-v2-seed42-$(date +%Y%m%dT%H%M%S)"
RUN_DIR="experiments/p5/gaia_v2/$RUN_ID"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

"$PY" scripts/p5/run_i1_pipeline.py train-evaluate \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --feature-workers 24 \
  --event-workers 8 \
  --case-chunk-size 128 \
  --start-method spawn \
  --gpu true
~~~

train-evaluate 是串行的 ad-train → post-ad 编排，不会创建第二个输出根。若
GPU 和 RCA 依赖不在同一环境，改用 ad-train 和 post-ad 两个阶段；不要把已
完成的当前 run 目录填回该命令。

完整的分阶段命令、CPU/GPU 分配和恢复说明见
[docs/GAIA_V3_IMPLEMENTATION.md](docs/GAIA_V3_IMPLEMENTATION.md)。

### 5. 测试

~~~
PYTHONDONTWRITEBYTECODE=1 pytest -q
~~~

当前正式 run 的 final_report.md 仍记录 Pytest: not recorded；在没有实际执行
并记录测试前，不要宣称该 run 已通过完整测试套件。

## 当前结果中的重要限制

本次 run 的数值已完整生成，但发布前仍需处理以下 provenance 问题：

- rca_train_manifest.json 中两个 E2E 重写文件的 hash 是旧值；
- e2e_diagnosis_metrics.json 没有填入 detected feature hash；
- shared reusable input 的历史 preprocessing config hash 与本次 execution config 不同；
- 服务/故障分布严重不均衡，overall 指标不能替代 macro 指标。

这些问题不应通过删除失败样本、改写 GT 或重新选择 Test 最优 epoch 来“修复”。
详细数值和证据链见
[docs/GAIA_P5_CURRENT_CONTEXT.md](docs/GAIA_P5_CURRENT_CONTEXT.md)。

## 历史兼容入口

仓库仍保留原始 Ada-MGAD/MSDS/GAIA 代码，例如 main.py、util/GAIA/pre_GAIA.py
和 util/GAIA/data_GAIA.py，用于历史复现或单元测试。它们不是当前 GAIA P5
双阶段正式实验入口；当前实验统一使用 scripts/p5/run_i1_pipeline.py 和独立
experiments/p5/gaia_v2/<run_id>。

## 上游资料

- [Ada-MGAD original repository](https://github.com/ligseeker/Ada-MGAD)
- [GAIA dataset](https://github.com/CloudWise-OpenSource/GAIA-DataSet)
- [GAIA multimodal preprocessing audit](docs/GAIA_MULTIMODAL_PREPROCESSING_AUDIT_V1.md)
- [P5 E2E protocol audit](docs/P5_GAIA_E2E_PROTOCOL_V2_AUDIT.md)
- [P5 implementation runbook](docs/GAIA_V3_IMPLEMENTATION.md)
