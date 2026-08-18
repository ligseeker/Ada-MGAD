# P1 Sanity Baseline Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-19
- Verification Status: UNVERIFIED
- Version Label: p1_sanity_baselines_v1

> 状态：B0/B1/B2 已在 GAIA 13,470-case 主 cohort、16,200-case inventory 与 RE2-OB 90 cases 上完成；G3/G5/G7 completed

## 1. 实验边界

所有 baseline 输出完整且无重复的 service ranking，并使用冻结的 5-fold
assignment：

- GAIA 主结果：13,470-case anchor-unique-root cohort，继承 context-grouped
  5-fold，seed `20260819`；完整 16,200 inventory 另作 sensitivity；
- RE2-OB：singleton stratified 5-fold，seed `20260819`；
- B0 Random：基于 `seed + case_id + service` 的 SHA-256 排序，不使用标签；
- B1 Root Frequency：对每个测试 fold，只统计其余四折的 root labels；
- B2 Metric Change：无跨 case 拟合，只读取各 case 自身 ±300 s metrics。

机器可读产物位于 `artifacts/p1/baselines/`，汇总为
`artifacts/p1/baseline_summary.json`。每个 baseline/dataset 保存：

- `predictions.jsonl`：per-case 完整 ranking；
- `metrics.json`：overall、fault-type macro/by-category、root-service
  macro/by-category；
- `training_audit.json`：fit scope、fold IDs hashes 或 metric coverage；
- `run_manifest.json`：config、source split/dataset hashes、输出文件 hashes。

## 2. B2 固定定义

B2 对每个 service/metric 比较半开窗口：

```text
pre  = [t0-300 s, t0)
post = [t0, t0+300 s)
```

实现规则：

1. 重复 timestamp 的有限值取均值；
2. 单侧至少 2 个有限样本，否则该 feature masked；
3. feature score 为 pre/post 均值差的绝对值，除以合并窗口总体标准差与
   relative epsilon 之和；
4. feature score 上限为 20；
5. 每服务取最高 5 个有效 feature scores 的均值；
6. 无有效 feature 的服务排在有观测服务之后，再按 service name 排序；
7. 全服务无观测时使用完全字母序 fallback，并显式记录 mask。

这是 sanity baseline，不是新方法，也不进行跨 case normalization。

## 3. 主结果

### GAIA

| Baseline | Overall AC@1 | Overall Avg@5 | Overall MRR | Fault-macro AC@1 | Fault-macro Avg@5 | Root-macro AC@1 | Root-macro Avg@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.1019 | 0.3031 | 0.2953 | 0.1202 | 0.3561 | 0.0993 | 0.3132 |
| Root Frequency | 0.4808 | 0.8745 | 0.7296 | 0.1435 | 0.3889 | 0.0994 | 0.2966 |
| Metric Change | 0.1405 | 0.3577 | 0.3402 | 0.4592 | 0.5981 | 0.5311 | 0.7086 |

Frequency 的 overall 很高，但 root-macro AC@1/Avg@5 仅 0.0994/0.2966，
几乎等于随机。这是预期的 sanity warning：GAIA 的 mobile/login class imbalance
可以制造强 overall shortcut，后续不得只报告 overall。

Metric Change 的 overall 提升有限，但其 fault/root macro 远高于 overall。
完整 16,200 inventory 的逐 fault 结果仍保留，不能与 13,470-case 主表混用。

### RE2-OB

| Baseline | AC@1 | AC@3 | AC@5 | Avg@5 | MRR |
|---|---:|---:|---:|---:|---:|
| Random | 0.0556 | 0.3000 | 0.4000 | 0.2556 | 0.2547 |
| Root Frequency | 0.1667 | 0.5556 | 1.0000 | 0.5667 | 0.4241 |
| Metric Change | 0.8556 | 0.9333 | 0.9889 | 0.9333 | 0.9046 |

RE2-OB 的 root/fault 设计平衡，因此 overall 与两个 macro 的结果相同。B2 在
cpu/disk/mem/socket 上 AC@1=1.0；delay 为 0.6667，loss 为 0.4667。该结果
说明原始 metrics 的锚点前后变化非常强，但仍只是数据集 sanity baseline，不能
直接写成结构/传播模型的贡献。

## 4. Missingness 与 coverage slices

| Dataset | 全服务 fallback cases | 任一服务 fallback cases | 有 metric 证据 cases |
|---|---:|---:|---:|
| GAIA main | 1,627 | 2,791 | 11,843 |
| GAIA inventory | 1,895 | 3,234 | 14,305 |
| RE2-OB | 0 | 0 | 90 |

GAIA main 有 metric 证据的 11,843 cases：

- overall AC@1 = 0.1595，Avg@5 = 0.3921；
- root-macro AC@1 = 0.5973，Avg@5 = 0.7656；
- fault-macro AC@1 = 0.5041，Avg@5 = 0.6363。

全服务 fallback 的 1,627 main cases 使用字母序，overall AC@1 仅 0.0018。正式
`metrics.json` 同时保存 observed/fallback slices，避免 fallback 被总体平均隐藏。

## 5. Label Firewall 与复现

- B0/B2 ranking 函数只接受 `RCACaseInput`；
- B1 每折只接收 4 个 training folds 的 `RCACaseLabel`；
- B1 的 train/test case count、ID digest 与 overlap=0 按 fold 落盘；
- 六个 prediction files 中 `root_service`/`fault_type`/`label` 零命中；
- source metric 路径只由可信 sidecar 解析，B2 不解析 RE2 目录名中的标签；
- evaluator 对每个 ranking 检查它恰为 candidate services 的完整排列；
- B0/B1 重跑时 summary、predictions、run manifests 字节级一致；
- B2 再次全量读取 5,724 GAIA metric shards 与 90 个 RE2 cases，结果与 hashes
  字节级一致。

## 6. 关键哈希

| Artifact | SHA-256 |
|---|---|
| Baseline summary | `968c1e4937afef4c0a263d6b4c844438df687ae984ff7931f322b2f39934cdbd` |
| GAIA-main Random predictions | `a35b750412c3e08a3ed7480f6b8fb196d2e6a74a828a4c398125868159152c88` |
| GAIA-main Frequency predictions | `f5f46b649bf9339a211af133cea5984053a6b8ee3b8890276546865e82642fac` |
| GAIA-main Metric Change predictions | `ef5e581b4acbc43e2eea9931e931c09ffa3658812fab300983f81484cd71cea2` |
| RE2 Random predictions | `950535d6943f57c5d7eb701812f97cf4cf752e967ed4f757b22b4251da82ae91` |
| RE2 Frequency predictions | `a23380d78b6105c6ab5631503371b0561800a4eb542672f5409a894b257c372b` |
| RE2 Metric Change predictions | `574d839bfb7482a222298e62c03e01e12f5edef5c7befb1420effab4155f0998` |

对应 run-manifest hashes：GAIA-main Random
`5f8c5a1920ee8a72088d30b2a2a463ef5f46da280c24c2eccfdc087dbcddbe81`，
Frequency `7fc2181299a0afe232084c5795f5e1ef0c6a591f1854262a5faa48b8ee92cf44`，
Metric Change `a04af8ce6b34dfa3d1ed2202c5b3c653f79c7420b99e87e2364d7b8715b08cf4`；
RE2 分别为 `d4703a94175036e618deec603b1c1cb42c3c220987ab45cfe2a19e066bcdbf7c`、
`8fc5cc7fed556265cbf6c4a907d1743634cffb002c55fa0b363eeeccb3074d8f`、
`04a5d16a22180b5e5d2c043f1cc0a2b98726d348fd2e3620b26104c697765cf5`。

## 7. 命令与运行状态

```text
python scripts/run_p1_sanity_baselines.py --manifest-root artifacts/p1/manifests --split-root artifacts/p1/splits --output-root artifacts/p1/baselines --summary-output artifacts/p1/baseline_summary.json --random-seed 20260819

python scripts/run_p1_metric_change.py --manifest-root artifacts/p1/manifests --split-root artifacts/p1/splits --output-root artifacts/p1/baselines --summary-output artifacts/p1/baseline_summary.json --window-seconds 300 --min-samples-per-side 2 --top-k-features 5 --score-cap 20 --progress-every 100
```

B0/B1 最终运行约 25 s；B2 每次全量运行约 3.5 min。正式命令均 exit 0；
B2 smoke 与两次正式全量运行也均 exit 0。

## 8. Gate 解释

- G3 completed：两个数据集的三种 baseline 均保存 per-case 完整 ranking；
- G5 completed：B0/B1/B2 已在两个数据集运行并保存 overall/macro 结果；
- G7 completed：train-only fit、input-only prediction boundary、prediction-file
  扫描、fold overlap audit 与测试均有证据；
- G1 completed：GAIA 13,470-case 主 cohort 与 2,730-case multi-root
  sensitivity 已冻结，见 [GAIA_INCLUSION_AUDIT.md](GAIA_INCLUSION_AUDIT.md)。
