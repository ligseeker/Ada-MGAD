# P2 Metric Feature Extraction and Coverage Audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-19
- Verification Status: UNVERIFIED
- Version Label: p2_metric_features_v1

> 状态：P2-G1 metric 子门 completed；logs/traces extractor 仍 pending

## 1. 固定表征

`p2_metric_summary_v1` 为每个 `(case_id, candidate service)` 输出 170 维、
label-free、固定宽度向量。窗口保持 P1 的 ±300 s 半开语义：

```text
pre    = [t0-300 s, t0)
post   = [t0, t0+300 s)
onset  = [t0, t0+To)
impact = [t0+To, t0+300 s)
```

10 个 comparison blocks 包含 whole pre→post，以及 30/60/120 s 下的
pre→onset、pre→impact、onset→impact。每个原始 metric series 计算：

- standardized signed mean shift；
- log population-scale ratio；
- standardized least-squares slope shift。

每服务跨指标维度精确流式聚合为 valid count/ratio，以及每个 contrast field 的
absolute maximum、Top-3 mean、Top-5 mean、signed mean 和 positive fraction。
重复时间戳的有限值先取均值；每段至少 2 个样本；contrast 截断到 ±20。

## 2. Feature bundle 与 Label Firewall

`p2_feature_bundle_v2` 使用：

- `index.jsonl`：排序后的 case/service/extractor 索引；
- `values.npy`：little-endian float32；
- `observed.npy`：bool mask；
- `manifest.json`：schema/config/names/shape/checksum 与 P1 source bindings。

writer 要求完整且唯一的 case-service coverage、统一 feature names/extractor、有限值、
masked value=0，并拒绝 feature name 中的 root/fault/label/target 语义。独立 verifier
重新检查所有 hashes、shape/dtype、索引排序、有限性、mask 和输入覆盖。

## 3. Smoke 与确定性

GAIA 3 cases × 10 services × 前 5 个逻辑指标，以及 RE2-OB 1 case × 11
services 运行两次，8 个 bundle 文件逐字节一致：

| Dataset | Cases | Service rows | Features | Manifest SHA-256 |
|---|---:|---:|---:|---|
| GAIA smoke | 3 | 30 | 170 | `bac4851a5e68ff100ea3290f7ab94a737dff80f1d15468a8cc159e3d522934d2` |
| RE2-OB smoke | 1 | 11 | 170 | `4f7bc6ef835bd748931d5425dbe5dfe485d8d998dcf055c02755ca026e623b3c` |

缓存固定 feature names 的性能优化后再次运行，所有 smoke hashes 保持不变。

## 4. 全量产物

| Dataset | Cases | Service rows | 原始扫描 | Values SHA-256 | Manifest SHA-256 |
|---|---:|---:|---|---|---|
| GAIA main | 13,470 | 134,700 | 2,862 logical series / 5,724 files | `48e409b900c03e992f6220d064cb0a002bdefced40ab96d74a6dc7a24a97b943` | `ded9f7ce90a9551b1f60b463b32159d8e48380fcacfb749a8640e85dea198f2c` |
| RE2-OB | 90 | 990 | 33,020 candidate metric columns | `cae6347f36ed2d56dea9cbc8af98a73f740868eae6f501cfc875e96ab785976e` | `8f38b6a90281ed9b6e4543b8688f76139941fb1f440c6f270dbee5d6661b238a` |

正式路径为 `artifacts/p2/features/<dataset>/p2_metric_summary_v1/`。汇总与独立
coverage audit 分别为：

- `artifacts/p2/metric_feature_summary.json`，SHA-256
  `b019f3390398027f443026f2bb2412b3abfed7d0995e9d49974f10835b3b5969`；
- `artifacts/p2/metric_feature_audit.json`，SHA-256
  `4b08650bf7675abb70f1b41f027b9311a13404d521ba5d7c91269256175ab9db`。

独立 audit 重新 mmap 两组矩阵并确认：finite=true、masked-zero=true、敏感字段
零命中、case-service coverage 完整。

## 5. Coverage 与 onset 决策

下表的 observed 表示某 service row 在对应 block 至少有一个有效 metric series：

| Block | GAIA observed | GAIA mean valid-series ratio | RE2 observed | RE2 mean valid-series ratio |
|---|---:|---:|---:|---:|
| whole | 83.744% | 0.7297 | 100% | 0.9982 |
| 30 s pre→onset | 0.092% | 0.0005 | 100% | 0.9982 |
| 30 s onset→impact | 0.091% | 0.0005 | 100% | 0.9990 |
| 60 s pre→onset | 83.725% | 0.3712 | 100% | 0.9982 |
| 60 s onset→impact | 83.729% | 0.3712 | 100% | 0.9990 |
| 120 s pre→onset | 83.734% | 0.7281 | 100% | 0.9982 |
| 120 s onset→impact | 83.733% | 0.7281 | 100% | 0.9990 |

GAIA 的常见 30/60 s metric 采样使 30 s onset 通常只有一个点，无法满足预先固定的
每段 2 样本条件。因此在查看任何 stage 模型结果前作如下调整：

- 统一可选 onset 从 30/60/120 s 收缩为 60/120 s；
- 已生成的 30 s 特征保留为 unsupported coverage control，不进入调参；
- 这是基于 telemetry coverage 的设计修正，不是基于模型性能选窗口。

## 6. 命令

```text
python scripts/extract_p2_metric_features_smoke.py
python scripts/extract_p2_metric_features.py
python -m unittest tests.test_feature_schema tests.test_metric_features -v
```

## 7. 局限

- GAIA whole/stage 可观察 service rows 约 83.7%，模型必须保留 mask 并报告
  coverage slices；
- service 内聚合允许两数据集指标名不同，但会丢失具体 indicator 身份；
- float32 只用于固定表征落盘，统计计算使用 float64；
- 本结果只关闭 metric extractor 子门，不代表 P2-G1 的 logs/traces 已完成。
