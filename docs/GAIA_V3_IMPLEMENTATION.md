# GAIA V3 implementation runbook

本文件记录 `/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2` 对《GAIA两阶段端到端数据处理与实验协议V3》的实现状态和正式执行命令。

## 当前状态

代码、协议检查、最小 fixture 和端到端 smoke 已完成。本轮没有执行 4.5G Metrics、18G Business Logs、8G Traces 的全量预处理，没有执行正式 100 epoch training，也没有执行正式 Test；所有 smoke 输出均标记为 `SMOKE_TEST / NOT FORMAL RESULT`。

正式 V3 入口只接受 `Train` / `Test` 两个 split。GT 由冻结 run table 的 message 原文解析，保留六类 fault taxonomy；`ERROR` 和 traceback continuation 只作为 telemetry，不进入 GT。所有时间区间采用 half-open 语义。Ada-MGAD 的 Train/Test 数据、metric quality/schema/normalization、Drain3 template state、trace graph/statistics 和 reconstruction calibration 均按 Train-only 规则生成。

事件阶段使用 `prediction_available_time=target_bin_end` 作为 `t_hat`，阈值只在 Train 上进行 exact unique-score 选择；事件匹配使用 `0 <= t_hat - gt_start_ms <= 60s`、maximum-cardinality、minimum-total-delay、one-to-one。Ada-RCA 使用独立 raw telemetry adapter：Oracle case 以 GT start 为 anchor，Detected case 以匹配到的 `prediction_available_time` 为 anchor。E2E 报告同时输出 Detector-only、Detected RCA、Oracle RCA 和 Root-Frequency Train-only 四层结果，并保留完整 event-level failure semantics。

## 独立实验目录与分阶段执行

正式实验使用唯一的 `run_dir`。共享预处理数据和冻结元数据只读；每次训练、事件检测、RCA 和 E2E 评估的可变输出都必须位于该实验目录内。这样可以并行提交 DAG 节点，也可以在一台机器上串行执行，而不会让不同实验写入同一个 checkpoint、prediction、模型或日志。

### 目录约定

```text
data/p5/v3_preprocessing_v2/ad/                 # 共享预处理数组，只读
artifacts/p5/v3_preprocessing_v2/ad/            # 共享 ad_data_manifest/schema，只读
artifacts/p5/v3_preprocessing_v2/protocol/      # 共享冻结 protocol/GT，只读

experiments/p5/gaia_v2/<run_id>/
├── run_state.json
├── run.lock
├── resolved_config.json
├── input_manifest.json
├── logs/
│   ├── pipeline.log
│   ├── 01_ad_train.log
│   ├── 02_event_detection.log
│   ├── 03_rca_features.log
│   ├── 04_rca_train.log
│   └── 05_e2e.log
├── checkpoint/{best_train_f1.pt,best_train_loss.pt,last.pt}
├── ad/{ad_training_summary.json,reconstruction_calibration.json,
│        ad_train_predictions.csv,ad_test_predictions.csv}
├── events/
├── rca/{detected/,detected_features/,conditional_logit.npz,...}
├── run_manifest.json
└── final_report.md
```

`run_state.json` 记录 run ID、git commit、配置 SHA、随机种子、启动命令和状态迁移；`resolved_config.json` 保存完整解析配置；`input_manifest.json` 锁定共享预处理 manifest/schema、protocol、event registry、RCA index/GT bundle 的路径和 SHA-256。`run_manifest.json` 只在所有 required artifact 完整时生成；smoke 结果不能冒充 formal result。

### 运行 ID、日志和不覆盖语义

先为每一次正式实验生成一次唯一 ID，并在所有 DAG 节点中复用同一个 `run_dir`：

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2
RUN_ID="gaia-v2-seed42-$(date +%Y%m%dT%H%M%S)"
RUN_DIR="experiments/p5/gaia_v2/${RUN_ID}"
CONFIG="configs/e2e/gaia_p5_v3_preprocessing_v2.json"
AD_PY="/home/zhangll24/miniconda3/envs/DAG/bin/python"
RCA_PY="/home/zhangll24/miniconda3/envs/rcaeval/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
```

不要预先 `mkdir "$RUN_DIR"`；`ad-train` 必须自己创建新目录，已存在就拒绝运行。每个子阶段的 stdout/stderr 会同时显示在终端，并由 pipeline 自动追加到阶段日志和 `logs/pipeline.log`，因此命令本身不需要再包一层 `tee`。失败会写入 `run_state.json` 并阻止后续阶段。

以下规则是 fail-closed 约束：

- `ad-train` 创建新 `run_dir`；目录已存在或 `run_id` 冲突时立即失败，不使用静默覆盖或默认 `--force`。
- 阶段只能消费正确的 `run_state.json` 前置状态；共享输入路径和 SHA 记录在 `input_manifest.json` 中，关键 loader 保留原有的 schema/manifest 校验。
- 训练阶段的 `last.pt` 在同一 run 内按 epoch 更新是正常行为；`best_train_f1.pt` 和 `best_train_loss.pt` 只代表该 run。不同 run 不共享 checkpoint、calibration、prediction 或 RCA model。
- 共享预处理根只提供输入；训练 summary、calibration、prediction、事件、RCA feature/model 和日志不得回写共享根。
- 已完成阶段不得通过再次运行覆盖 formal artifact；如需重新训练，创建新的 `run_id`，并在 manifest 中保留对同一冻结输入的引用。

### DAG：Ada-MGAD 与 RCA 分阶段运行

当 DAG 节点分别运行在 GPU/DAG 环境和 CPU/RCA 环境时，先执行 Ada-MGAD 节点：

```bash
"$AD_PY" -c 'import torch, torch_geometric; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()'

"$AD_PY" scripts/p5/run_i1_pipeline.py ad-train \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --gpu true
```

该节点读取共享 Train/Test 张量和 `ad_data_manifest.json`，将 checkpoint 写入 `$RUN_DIR/checkpoint/`，将 calibration 和 Train/Test predictions 写入 `$RUN_DIR/ad/`；完成后状态必须为 `AD_COMPLETE`。

随后在 RCA/CPU 节点执行：

```bash
"$RCA_PY" -c 'import numpy, pandas, scipy, sklearn; print(numpy.__version__, pandas.__version__, scipy.__version__, sklearn.__version__)'

"$RCA_PY" scripts/p5/run_i1_pipeline.py post-ad \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --feature-workers 2 \
  --event-workers 2 \
  --case-chunk-size 64 \
  --start-method spawn
```

`post-ad` 只消费已完成 run 的 AD predictions，依次完成事件检测、Detected-anchor RCA feature、RCA training/inference、E2E evaluation 和 final manifest。前置状态、required input 或各阶段自有校验失败时会立即停止，不切换到历史输出。

### 单环境组合运行

如果同一环境同时具备 GPU/DAG 和 RCA 依赖，可用组合入口串行执行两个阶段：

```bash
"$AD_PY" scripts/p5/run_i1_pipeline.py train-evaluate \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --feature-workers 2 \
  --event-workers 2 \
  --case-chunk-size 64 \
  --start-method spawn \
  --gpu true
```

该命令仍使用同一套状态、checksum、日志和不覆盖规则；`train-evaluate` 只是 `ad-train` 与 `post-ad` 的串行编排，不会创建第二个输出根。若 GPU 不可用，将 `--gpu false`，但不得把 CPU smoke 或部分结果报告为 formal full-data result。

## Worker 建议

先查看服务器资源：

```bash
nproc
```

V3 配置的 CPU budget 是 30；大容器的参考组合是 metric/log/Ada-MGAD `8`、raw RCA index `8`、trace `24`、RCA feature materialization `24`，并使用 `spawn`。当前 `3 CPU / 30 GB` 容器应显式覆盖为 `feature-workers=2`、`event-workers=2`、`case-chunk-size=64`，同时将 OMP/MKL/OpenBLAS 线程固定为 1，为主进程和 OS 保留余量。parse/map 阶段写独立 cache/shard，最终 merge/reduce 按固定顺序执行。

## 历史 V3 逐步命令（不用于新 V2 run）

以下保留为旧 `gaia_p5_v3.json` artifact 的历史复现记录。新 preprocessing V2 实验不应执行这些硬编码共享输出命令，应使用本文前部的 `ad-train` → `post-ad` 独立 run 入口。

### 0. Provenance / GT / protocol

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2

PYTHONDONTWRITEBYTECODE=1 python scripts/p5/build_v3_gt.py \
  --config configs/e2e/gaia_p5_v3.json \
  --raw-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS

PYTHONDONTWRITEBYTECODE=1 python scripts/p5/build_v3_protocol.py \
  --config configs/e2e/gaia_p5_v3.json
```

这一步验证 run table SHA-256，写出 `artifacts/p5/v3/protocol/gt_event_registry.csv`、`assigned_event_registry.csv`、`purged_event_registry.csv`、`provenance.json`、`split_manifest.json` 和 `protocol_manifest.json`。

### 1. Full preprocessing

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/p5/run_i1_ad.py preprocess \
  --config configs/e2e/gaia_p5_v3.json \
  --raw-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS \
  --data-root data/p5/v3/ad \
  --artifact-root artifacts/p5/v3/ad \
  --checkpoint-dir data/p5/v3/checkpoint \
  --chunk-rows 150000 \
  --start-method spawn \
  --gpu false

PYTHONDONTWRITEBYTECODE=1 python scripts/p5/run_i1_rca_features.py case-registry \
  --config configs/e2e/gaia_p5_v3.json \
  --anchor-mode gt \
  --case-registry artifacts/p5/v3/rca/rca_case_registry_gt.csv

PYTHONDONTWRITEBYTECODE=1 python scripts/p5/run_i1_rca_features.py index \
  --config configs/e2e/gaia_p5_v3.json \
  --raw-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS \
  --index-root data/p5/v3/rca_raw_index \
  --chunk-rows 150000 \
  --workers 8 \
  --start-method spawn

PYTHONDONTWRITEBYTECODE=1 python scripts/p5/run_i1_rca_features.py materialize \
  --config configs/e2e/gaia_p5_v3.json \
  --index-root data/p5/v3/rca_raw_index \
  --feature-root data/p5/v3/rca_features_gt \
  --artifact-root artifacts/p5/v3/rca_gt_features \
  --case-registry artifacts/p5/v3/rca/rca_case_registry_gt.csv \
  --chunk-rows 150000 \
  --workers 24 \
  --case-chunk-size 128 \
  --start-method spawn \
  --limit-cases 0
```

`run_i1_ad.py preprocess` 不提供 `--workers` 时使用配置的 metric/log=8、trace=24；RCA raw index 使用 8，RCA feature shard 使用 24。所有命令均支持已有 checksum-valid cache/shard 的安全跳过；发现 binding 或 checksum 不一致会重新生成或 fail closed，不会复用近似 artifact。

### 2. Ada-MGAD training

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/p5/run_i1_ad.py train \
  --config configs/e2e/gaia_p5_v3.json \
  --data-root data/p5/v3/ad \
  --artifact-root artifacts/p5/v3/ad \
  --checkpoint-dir data/p5/v3/checkpoint \
  --gpu false
```

配置固定 `max_epochs=100`、`patience=10`、`early_stopping_metric=train_f1`。每轮完整 Train epoch 后计算 Train/Test 指标；Test 指标仅用于观察趋势，不参与梯度、早停或 checkpoint 选择。只有 Train F1 的严格提升会重置 patience，并保存 `best_train_f1.pt`、`best_train_loss.pt`、`last.pt`。主 checkpoint 始终是 `best_train_f1.pt`；`best_train_loss.pt` 只用于诊断。

### 3. Train calibration

训练命令完成主 checkpoint 后，会在同一命令内加载 `best_train_f1.pt`，只用 Train reconstruction scores 拟合并写出 calibration。可用以下无拟合检查确认该 handoff：

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/p5/v3/ad/reconstruction_calibration.json")
data = json.loads(path.read_text(encoding="utf-8"))
assert data["fit_split"] == "train"
assert data["train_count"] > 0
print({"fit_split": data["fit_split"], "train_count": data["train_count"]})
PY
```

Test reconstruction scores 不参与 calibration；calibration JSON 是后续 Test inference 的 frozen input。

### 4. Test inference

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/p5/run_i1_ad.py infer \
  --config configs/e2e/gaia_p5_v3.json \
  --data-root data/p5/v3/ad \
  --artifact-root artifacts/p5/v3/ad \
  --checkpoint-dir data/p5/v3/checkpoint \
  --gpu false
```

该命令验证主 checkpoint 和 training-summary checksum，加载 frozen Train calibration，并写出 `ad_train_predictions.csv` 与 `ad_test_predictions.csv`。每个 prediction row 的 `prediction_available_time` 等于 `target_bin_end`。

### 5. Event evaluation

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/p5/run_i1_events.py evaluate \
  --config configs/e2e/gaia_p5_v3.json \
  --artifact-root artifacts/p5/v3/events \
  --train-predictions artifacts/p5/v3/ad/ad_train_predictions.csv \
  --test-predictions artifacts/p5/v3/ad/ad_test_predictions.csv \
  --registry artifacts/p5/v3/protocol/gt_event_registry.csv \
  --workers 8 \
  --start-method spawn
```

输出包括 Train-only threshold、Train/Test episodes、causal one-to-one matching、event precision/recall/F1 和 delay distribution。Test 只使用已冻结阈值。

### 6. Detected-anchor RCA feature building

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/p5/run_i1_rca_features.py case-registry \
  --config configs/e2e/gaia_p5_v3.json \
  --anchor-mode detected \
  --matching artifacts/p5/v3/events/event_matching.csv \
  --case-registry artifacts/p5/v3/rca/rca_case_registry_detected.csv

PYTHONDONTWRITEBYTECODE=1 python scripts/p5/run_i1_rca_features.py materialize \
  --config configs/e2e/gaia_p5_v3.json \
  --index-root data/p5/v3/rca_raw_index \
  --feature-root data/p5/v3/rca_features_detected \
  --artifact-root artifacts/p5/v3/rca_detected_features \
  --case-registry artifacts/p5/v3/rca/rca_case_registry_detected.csv \
  --chunk-rows 150000 \
  --workers 24 \
  --case-chunk-size 128 \
  --start-method spawn \
  --limit-cases 0
```

Detected Test feature construction consumes only matched event identity plus `prediction_available_time`; service/fault labels remain in the separate registry. RCA W300 context crossing a chronological boundary is purged and recorded.

### 7. RCA training and inference

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/p5/run_i1_rca.py train-v3 \
  --config configs/e2e/gaia_p5_v3.json \
  --feature-root data/p5/v3/rca_features_gt \
  --case-registry artifacts/p5/v3/rca/rca_case_registry_gt.csv \
  --detected-feature-root data/p5/v3/rca_features_detected \
  --detected-case-registry artifacts/p5/v3/rca/rca_case_registry_detected.csv \
  --model-path data/p5/v3/rca_model/conditional_logit.npz \
  --artifact-root artifacts/p5/v3/rca
```

Conditional Logit 和 Root-Frequency 都只在 Oracle Train cases 上 fit。输出分别保留 Oracle GT-anchor、Detected prediction-anchor 和 Root-Frequency ranking。

### 8. E2E evaluation

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/p5/run_i1_e2e.py evaluate \
  --config configs/e2e/gaia_p5_v3.json \
  --artifact-root artifacts/p5/v3/rca \
  --index-manifest data/p5/v3/rca_raw_index/index_manifest.json \
  --model-path data/p5/v3/rca_model/conditional_logit.npz \
  --case-registry artifacts/p5/v3/rca/rca_case_registry_gt.csv \
  --event-matching artifacts/p5/v3/events/event_matching.csv \
  --test-node-predictions artifacts/p5/v3/ad/ad_test_predictions.csv \
  --oracle-predictions artifacts/p5/v3/rca/rca_oracle_predictions.csv \
  --root-frequency-predictions artifacts/p5/v3/rca/root_frequency_predictions.csv \
  --detected-predictions artifacts/p5/v3/rca/rca_detected_predictions.csv
```

该步骤消费独立 raw-adapter 已物化的 Detected ranking，不从 Ada-MGAD tensor 重建 RCA feature。报告同时写出 `e2e_diagnosis_metrics.json` 和 `e2e_layered_report.json`，对 unmatched prediction/GT 和 Top-k ranking failure 保留惩罚语义。

### 9. Final report

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/p5/finalize_v3_manifest.py \
  --config configs/e2e/gaia_p5_v3.json \
  --artifact-root artifacts/p5/v3 \
  --pytest-result "PYTHONDONTWRITEBYTECODE=1 pytest -q"
```

Finalizer 只汇总并校验 formal artifact 是否齐全，不执行 preprocessing、training 或 Test；缺失任何 required artifact 时保持 `IMPLEMENTATION_READY_FORMAL_FULL_DATA_PENDING_MANUAL_RUN`，不会把 smoke artifact 提升为 formal result。

## 一键编排入口

若共享 V2 preprocessing 尚未生成，先在当前 3 核容器执行一次（已有完整 manifest 时不要重复执行）：

```bash
"$AD_PY" scripts/p5/run_i1_pipeline.py preprocess \
  --config "$CONFIG" \
  --raw-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS \
  --chunk-rows 150000 \
  --raw-workers 2 \
  --ad-workers 2 \
  --feature-workers 2 \
  --case-chunk-size 64 \
  --start-method spawn
```

共享输入完成后，优先使用前文的两阶段命令。只有同一 Python 环境同时通过 AD 和 RCA 依赖检查时，才使用：

```bash
"$AD_PY" scripts/p5/run_i1_pipeline.py train-evaluate \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --feature-workers 2 \
  --event-workers 2 \
  --case-chunk-size 64 \
  --start-method spawn \
  --gpu true
```

## 最小必要代码调整

Ada-MGAD 和 Ada-RCA 的核心网络/Conditional Logit 结构未修改。为接入冻结 V3 协议做了以下边界层调整：

1. 上游 Ada-MGAD 图索引要求固定 batch size，而 split window 数通常不能整除 batch size；增加 split-local `PaddedSequentialSampler`，训练端保证完整 batch，推理端按 `sample_index` 去重，不跨 Train/Test。该调整不改变模型结构；最后一个 Train batch 的 padding 会重复最后一个 Train window，比例由 batch 对齐决定并写入训练行为，属于已知实现边界。
2. 历史 V1 entry points 保留给旧单元测试和审计复现；所有 V3 formal entry points 直接拒绝 Validation、旧 config、错误 config SHA、错误 schema 或错误 checkpoint binding。
3. 原始 RCA index 采用独立 source-file/chunk shards 和确定性串行 merge；Drain3 采用串行 canonical Train fit，再对 Test 做 frozen transform，未知模板写入 `template_UNK`。
