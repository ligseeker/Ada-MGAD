# GAIA V3 implementation runbook

本文件记录 `/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2` 对《GAIA两阶段端到端数据处理与实验协议V3》的实现状态和正式执行命令。

## 当前状态

代码、协议检查、最小 fixture 和端到端 smoke 已完成。本轮没有执行 4.5G Metrics、18G Business Logs、8G Traces 的全量预处理，没有执行正式 100 epoch training，也没有执行正式 Test；所有 smoke 输出均标记为 `SMOKE_TEST / NOT FORMAL RESULT`。

正式 V3 入口只接受 `Train` / `Test` 两个 split。GT 由冻结 run table 的 message 原文解析，保留六类 fault taxonomy；`ERROR` 和 traceback continuation 只作为 telemetry，不进入 GT。所有时间区间采用 half-open 语义。Ada-MGAD 的 Train/Test 数据、metric quality/schema/normalization、Drain3 template state、trace graph/statistics 和 reconstruction calibration 均按 Train-only 规则生成。

事件阶段使用 `prediction_available_time=target_bin_end` 作为 `t_hat`，阈值只在 Train 上进行 exact unique-score 选择；事件匹配使用 `0 <= t_hat - gt_start_ms <= 60s`、maximum-cardinality、minimum-total-delay、one-to-one。Ada-RCA 使用独立 raw telemetry adapter：Oracle case 以 GT start 为 anchor，Detected case 以匹配到的 `prediction_available_time` 为 anchor。E2E 报告同时输出 Detector-only、Detected RCA、Oracle RCA 和 Root-Frequency Train-only 四层结果，并保留完整 event-level failure semantics。

## Worker 建议

先查看服务器资源：

```bash
nproc
```

V3 配置的 CPU budget 是 30；默认的安全组合是 metric/log/Ada-MGAD `8`、raw RCA index `8`、trace `24`、RCA feature materialization `24`，并使用 `spawn`。parse/map 阶段写独立 cache/shard，最终 merge/reduce 按固定顺序执行。正式命令显式给出 `--workers`；若不提供 `run_i1_ad.py preprocess` 的全局 override，它会使用配置中每个 modality 的独立 worker 数。

## 正式全量执行命令

以下命令只供之后的正式人工执行。本轮没有执行这些命令。每一步结束后应检查 JSON manifest 的 `config_sha256`、输入 checksum、split 和 status，再继续下一步。

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

配置固定 `max_epochs=100`、`patience=10`、`early_stopping_metric=train_total_loss`。该命令只允许完整 Train epoch 的平均 total loss 触发 early stopping，并保存 `best_train_loss.pt`、`best_train_f1.pt`、`last.pt`。主 checkpoint 始终是 `best_train_loss.pt`；`best_train_f1.pt` 只用于诊断。

### 3. Train calibration

训练命令完成主 checkpoint 后，会在同一命令内加载 `best_train_loss.pt`，只用 Train reconstruction scores 拟合并写出 calibration。可用以下无拟合检查确认该 handoff：

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

如果希望在上述同一目录和 lock 下串行执行，可使用以下两个命令；它们等价于第 0/1 步和第 2/5/6/7/8/9 步的组合：

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/p5/run_i1_pipeline.py preprocess \
  --config configs/e2e/gaia_p5_v3.json \
  --raw-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS \
  --chunk-rows 150000 \
  --start-method spawn

PYTHONDONTWRITEBYTECODE=1 python scripts/p5/run_i1_pipeline.py train-evaluate \
  --config configs/e2e/gaia_p5_v3.json \
  --feature-workers 24 \
  --event-workers 8 \
  --case-chunk-size 128 \
  --start-method spawn \
  --gpu false
```

## 最小必要代码调整

Ada-MGAD 和 Ada-RCA 的核心网络/Conditional Logit 结构未修改。为接入冻结 V3 协议做了以下边界层调整：

1. 上游 Ada-MGAD 图索引要求固定 batch size，而 split window 数通常不能整除 batch size；增加 split-local `PaddedSequentialSampler`，训练端保证完整 batch，推理端按 `sample_index` 去重，不跨 Train/Test。该调整不改变模型结构；最后一个 Train batch 的 padding 会重复最后一个 Train window，比例由 batch 对齐决定并写入训练行为，属于已知实现边界。
2. 历史 V1 entry points 保留给旧单元测试和审计复现；所有 V3 formal entry points 直接拒绝 Validation、旧 config、错误 config SHA、错误 schema 或错误 checkpoint binding。
3. 原始 RCA index 采用独立 source-file/chunk shards 和确定性串行 merge；Drain3 采用串行 canonical Train fit，再对 Test 做 frozen transform，未知模板写入 `template_UNK`。
