# P2 G3 单模态与 Naive 融合结果（C0-L / C0-T / C1-I）

## Material Passport

- Origin Skill: claude-code / experiment-governance backfill
- Origin Mode: run
- Origin Date: 2026-08-19
- Verification Status: ARTIFACT-VERIFIED
- Version Label: p2_g3_modality_results_v1

> 状态：P2-G1 全量 L0/T0 extraction completed；P2-G3 completed；C1-I 达到
> exploratory signal，未达 claim-ready；H1/H2/H3 仍未检验（M1-S 尚未运行）。

本文件记录的全部数值来自 `artifacts/p2/` 下的可复现产物，非对话报告。逐 fold 结果
由 `artifacts/p2/runs/<method>/<ds>/predictions.jsonl` 与冻结 split assignments 重算
补齐（原 summary 未包含该切片）。

- Data version：GAIA MicroSS 13,470-case main cohort / RCAEval RE2-OB 90 cases
- Split：`artifacts/p1/splits/{gaia,re2ob}`，`p1_split_manifest_v1`，5-fold OOF
- Seed：`20260819`
- Git branch / commit：`claudecode` / `64bb681328fa1793014615a20ba2bc1fdf33c3b7`
- Working tree：仅有未跟踪的新增文件，无已跟踪文件修改

## 1. 协议

三个方法共用 P2-G2 冻结的 nested OOF 装置，只改输入 design 矩阵：

- outer：冻结 5-fold OOF（GAIA 按 322 context group 分组分层，RE2-OB 单例分层）；
- inner：每个 outer-train 内四折轮换，唯一目标为 inner root-service macro Avg@5；
- L2 logistic regression，`C ∈ {0.01,0.1,1,10}`；并列优先较小 C；
- StandardScaler 只在当前 fit rows 拟合；
- 每 case 总权重 1：root row 0.5，全部 non-root rows 合计 0.5；
- 分数并列按 service name 升序，输出完整候选排列；
- 每 dataset 5 outer fits + 80 inner fits。

design 矩阵（均只用 `whole.*` 通道及其等宽 masks，不含 stage、service identity、
fault type、root frequency）：

| 方法 | 输入 | 值列 | 含 masks 总列 |
|---|---|---:|---:|
| C0-M | metric `whole.*` | 17 | 34 |
| C0-L | L0 log `whole.*` | 5 | 10 |
| C0-T | T0 trace `whole.*` | 8 | 16 |
| C1-I | metric + log + trace `whole.*` 拼接 | 30 | 60 |

C1-I 是 naive early fusion：只做列拼接，不含 stage、不含跨模态交互项，因此它是
M1-S 单因素对照的合法基线。

## 2. 全量 L0/T0 extraction 与 coverage slices

extractor：`p2_log_l0_v1`（50 列）/ `p2_trace_t0_v1`（80 列），bundle schema
`p2_feature_bundle_v2`，窗口 `T_pre = T_post = 300 s` 半开、毫秒。

| Bundle | GAIA 原始行 | GAIA service rows | GAIA entity-observed 比例 | RE2 原始行 | RE2 service rows | RE2 entity-observed 比例 |
|---|---:|---:|---:|---:|---:|---:|
| metric `p2_metric_summary_v1` | — | 134,700 | 1.000000 | — | 990 | 1.000000 |
| log `p2_log_l0_v1` | 87,974,871 | 134,700 | 0.999777 | 15,053,223 | 990 | 0.911111 |
| trace `p2_trace_t0_v1` | 28,681,438 | 134,700 | 0.999777 | 34,461,235 | 990 | 0.636364 |

GAIA 两个事件模态各缺 30/134,700 行（10 个 service × 3 case），RE2-OB log 缺 88/990、
trace 缺 360/990（对应 11 个 trace entity 中仅 7 个可观测）。缺失行按 Label Firewall
规则整行 mask，值恰为 `0.0`。

onset support（阈值：absolute ≥ 0.80，relative-to-whole ≥ 0.90；下表为
`absolute_min_complete_ratio`）：

| Dataset | Modality | 30 s | 60 s | 120 s | 结论 |
|---|---|---:|---:|---:|---|
| GAIA main | log | 0.964833 | 0.964706 | 0.964677 | 三档均 supported |
| GAIA main | trace | 0.927051 | 0.926962 | 0.926962 | 三档均 supported |
| RE2-OB | log | 0.794900 | 0.794900 | 0.794900 | 三档均 **unsupported** |
| RE2-OB | trace | 0.995238 | 0.988889 | 0.988889 | 三档均 supported |

同表的 `relative_to_whole` 全部 ≥ 0.9888（GAIA 两模态 ≥ 0.99975，RE2-OB log 为
1.0，RE2-OB trace 为 0.9952 / 0.9889 / 0.9889），因此 RE2-OB log 的 unsupported
判定完全由 absolute 阈值触发，不是相对衰减导致。

RE2-OB log 的 content-complete 比例 0.794900 低于冻结阈值 0.80，因此 M1-S 的
staged 通道排除 log，只用 metric + trace 的 `stage{60,120}`。该排除已写入
`scripts/run_p2_m1_stage.py` 的 run manifest config（`log_stage_excluded`）。

事件模态 extraction 审计（`artifacts/p2/event_feature_audit.json`）：
`all_bundles_verified=true`、`all_values_finite=true`、`masked_values_zero=true`、
`labels_read_by_extractor=false`。注意该 audit 由 `extract_p2_event_features.py`
在同一次运行内经 `verify_feature_bundle` 复验后写出，**不是**独立第二进程产物；
与 `run_*`/`audit_*` 成对的独立审计不同，这一点在引用时不得混称。

## 3. OOF 结果：B2 / C0-M / C0-L / C0-T / C1-I 对照

主端点为 root-service macro Avg@5，关键次端点为 root-macro AC@1。

GAIA main（13,470 cases）：

| 方法 | Overall AC@1 | Overall Avg@5 | Fault-macro AC@1 | Fault-macro Avg@5 | Root-macro AC@1 | Root-macro Avg@5 |
|---|---:|---:|---:|---:|---:|---:|
| B2 Metric Change（P1，未学习） | 0.1405 | 0.3577 | 0.4592 | 0.5981 | 0.5311 | 0.7086 |
| C0-M | 0.1633 | 0.3953 | 0.4096 | 0.5523 | 0.4567 | 0.5820 |
| C0-L | 0.0856 | 0.4141 | 0.4180 | 0.5731 | 0.1622 | 0.3567 |
| C0-T | 0.3566 | 0.5043 | 0.4971 | 0.6096 | 0.1722 | 0.3549 |
| C1-I | 0.3376 | 0.6691 | 0.6020 | 0.7541 | 0.3656 | 0.6072 |

RE2-OB（90 cases）：

| 方法 | Overall AC@1 | Overall Avg@5 | Fault-macro AC@1 | Fault-macro Avg@5 | Root-macro AC@1 | Root-macro Avg@5 |
|---|---:|---:|---:|---:|---:|---:|
| B2 Metric Change（P1，未学习） | 0.8556 | 0.9333 | 0.8556 | 0.9333 | 0.8556 | 0.9333 |
| C0-M | 0.9111 | 0.9756 | 0.9111 | 0.9756 | 0.9111 | 0.9756 |
| C0-L | 0.2889 | 0.4667 | 0.2889 | 0.4667 | 0.2889 | 0.4667 |
| C0-T | 0.6222 | 0.8156 | 0.6222 | 0.8156 | 0.6222 | 0.8156 |
| C1-I | 0.9778 | 0.9956 | 0.9778 | 0.9956 | 0.9778 | 0.9956 |

三点可复现的读数：

1. GAIA 上两个事件单模态在主端点几乎不含定位信息（root-macro Avg@5 0.3567 /
   0.3549，接近 10 root 的低信息水平），但在 overall 层面高于 metric-only，说明它们
   捕捉到的是与类别频率相关的信号而非 root 的区分度。
2. RE2-OB 的 root_service / fault_type 分层与 overall 完全同值，因为该 cohort 每个
   case 都是单例分层单位；这使 RE2-OB 上三层指标不提供额外信息。
3. C1-I 在两个数据集的主端点都优于任一单模态，但 GAIA 的次端点 AC@1 明显退化。

## 4. C1-I vs best-single-modal

best single modality 按 root-macro Avg@5 点估计选出，两数据集均为 **C0-M**
（`artifacts/p2/linear_ablation_audit.json` → `best_single_comparison`）。

| Dataset | Δ Overall AC@1 | Δ Overall Avg@5 | Δ Fault-macro AC@1 | Δ Fault-macro Avg@5 | Δ Root-macro AC@1 | Δ Root-macro Avg@5 |
|---|---:|---:|---:|---:|---:|---:|
| GAIA main | +0.1744 | +0.2738 | +0.1924 | +0.2018 | −0.0911 | +0.0253 |
| RE2-OB | +0.0667 | +0.0200 | +0.0667 | +0.0200 | +0.0667 | +0.0200 |

主端点在两个数据集都为正 → `c1_i_primary_improved=true`（两侧），
`c1_i_exploratory_signal_both_datasets=true`。

对全部单模态的主端点差值：

| Dataset | vs C0-M | vs C0-L | vs C0-T |
|---|---:|---:|---:|
| GAIA main | +0.0253 | +0.2505 | +0.2523 |
| RE2-OB | +0.0200 | +0.5289 | +0.1800 |

## 5. C1-I vs P1 B2 Metric Change

| Dataset | Δ Root-macro AC@1 | Δ Root-macro Avg@5 |
|---|---:|---:|
| GAIA main | −0.1655 | −0.1013 |
| RE2-OB | +0.1222 | +0.0622 |

结论边界：截至 P2-G3，**GAIA 上没有任何 learned 方法在主端点超过未学习的 P1 B2**。
因此不得宣称学习式融合优于 metric-change 基线；C1-I 只是 Track C 内部的合法融合
对照。RE2-OB 上 C1-I 已达 0.9956，接近上限，天花板效应记为 limitation。

## 6. 逐 fold 结果（root-service macro）

GAIA main，fold 规模 2693 / 2684 / 2711 / 2705 / 2677：

| 方法 | 指标 | fold_0 | fold_1 | fold_2 | fold_3 | fold_4 |
|---|---|---:|---:|---:|---:|---:|
| C0-M | AC@1 | 0.4501 | 0.4984 | 0.4731 | 0.4343 | 0.4214 |
| C0-M | Avg@5 | 0.5610 | 0.6225 | 0.5853 | 0.6033 | 0.5487 |
| C0-L | AC@1 | 0.2181 | 0.1278 | 0.1802 | 0.1314 | 0.1471 |
| C0-L | Avg@5 | 0.4034 | 0.3214 | 0.3616 | 0.3404 | 0.3400 |
| C0-T | AC@1 | 0.1971 | 0.1636 | 0.1620 | 0.1681 | 0.1622 |
| C0-T | Avg@5 | 0.3659 | 0.3555 | 0.3322 | 0.3480 | 0.3522 |
| C1-I | AC@1 | 0.3822 | 0.3560 | 0.3673 | 0.3300 | 0.3739 |
| C1-I | Avg@5 | 0.6276 | 0.6038 | 0.5986 | 0.5879 | 0.6135 |

RE2-OB，每 fold 18 cases：

| 方法 | 指标 | fold_0 | fold_1 | fold_2 | fold_3 | fold_4 |
|---|---|---:|---:|---:|---:|---:|
| C0-M | AC@1 | 0.8667 | 0.9500 | 0.8333 | 0.9333 | 0.9333 |
| C0-M | Avg@5 | 0.9733 | 0.9900 | 0.9667 | 0.9600 | 0.9733 |
| C0-L | AC@1 | 0.2500 | 0.3167 | 0.2667 | 0.3333 | 0.3167 |
| C0-L | Avg@5 | 0.4267 | 0.5300 | 0.4633 | 0.4500 | 0.4967 |
| C0-T | AC@1 | 0.6500 | 0.6333 | 0.5833 | 0.7000 | 0.5833 |
| C0-T | Avg@5 | 0.7700 | 0.8500 | 0.8400 | 0.8300 | 0.8133 |
| C1-I | AC@1 | 1.0000 | 0.8833 | 1.0000 | 1.0000 | 1.0000 |
| C1-I | Avg@5 | 1.0000 | 0.9767 | 1.0000 | 1.0000 | 1.0000 |

C1-I 在 GAIA 五折的主端点范围为 0.5879–0.6276，无单折反转；C0-M 在 GAIA fold_1
的 AC@1 高于 C1-I 全部折，与第 3 节的次端点退化一致。

每折所选 C：

| 方法 | GAIA outer folds | RE2-OB outer folds |
|---|---|---|
| C0-M | 10 / 10 / 10 / 10 / 10 | 0.1 / 10 / 0.1 / 1 / 10 |
| C0-L | 1 / 1 / 10 / 1 / 0.1 | 1 / 1 / 0.01 / 10 / 10 |
| C0-T | 0.01 / 0.01 / 0.1 / 0.01 / 10 | 1 / 10 / 1 / 1 / 10 |
| C1-I | 10 / 10 / 10 / 10 / 10 | 10 / 1 / 1 / 1 / 1 |

## 7. C1-I 配对 bootstrap 与 Go/No-Go

`scripts/bootstrap_p2_c1_i.py`，schema `p2_c1_i_paired_bootstrap_v1`，10,000 次，
seed `20260819`，`numpy.default_rng/PCG64`；GAIA 按 322 context group 重采样，
RE2-OB 按 90 case 重采样；比较对象为 best single modality（C0-M）。

| Dataset | 指标 | point Δ | 95% CI lower | 95% CI upper | P(Δ ≤ 0) |
|---|---|---:|---:|---:|---:|
| GAIA main | Root-macro Avg@5（主） | +0.025261 | −0.003557 | +0.056340 | 0.0445 |
| GAIA main | Root-macro AC@1（次） | −0.091132 | −0.132206 | −0.049165 | 1.0000 |
| RE2-OB | Root-macro Avg@5（主） | +0.020000 | +0.002514 | +0.040000 | 0.0118 |
| RE2-OB | Root-macro AC@1（次） | +0.066667 | +0.002050 | +0.133333 | 0.0223 |

按冻结的 Go/No-Go 规则：

- `exploratory_signal = true`（两数据集主端点点估计均为正）；
- `claim_ready = false`，两项检查都不通过：
  - `both_primary_ci_lower_bounds_positive = false`（GAIA CI 下界 −0.003557 ≤ 0）；
  - `both_secondary_point_deltas_at_least_minus_0_01 = false`（GAIA AC@1
    −0.091132 < −0.01）。

GAIA 的 AC@1 退化是本 gate 的确定性诊断结果，**不作为修改 inner selection
objective 的依据**（用户已确认：改动会破坏 H1 的单因素可解释性）。若后续需要调整
selection objective，须单独做 protocol version bump + sensitivity experiment。

## 8. 泄漏与复现审计

`scripts/audit_p2_linear_ablations.py`（schema `p2_linear_ablation_audit_v1`）对
C0-L / C0-T / C1-I 各自独立完成：

- 每 dataset 5 outer folds、80 inner fits；
- 全部 outer train/test case overlap = 0、group overlap = 0；
- 全部 inner fit/validation overlap = 0；
- predictions 的 root / fault / ground-truth token（归一化后）零命中；
- rankings 经 evaluator 独立重算，metrics 与记录逐字相等（`!=` 即抛错）；
- selected C 可由保存的 inner objective 独立重建；
- run manifest 内每个核心文件的 SHA-256 逐一校验通过。

| Artifact | SHA-256 |
|---|---|
| Linear ablation summary | `ef1d3aa46f961dbf77776882d3c2234e26f5e474090e047bee2711654b62a718` |
| Linear ablation independent audit | `003655fc3207d7831f92d2d4d1631ab7bdeefbde17275acd5d67a717cdba75e3` |
| C1-I paired bootstrap | `f7399673340d7af540af2336e997ef5fffb4ca7495247a72e36597a95ca77212` |
| Event feature summary | `a2ba98d4e52c566639701e1c9a16df99894c27cc4a509622c6a17b666aa2aa3d` |
| Event feature coverage audit | `27367fb5d14a46d74551a767507c119be7aeb11491bd2bc69de3aaf86e1d1ff3` |
| C0-L GAIA run manifest | `da17d4bd1662f65948c48a079aaaa012069afcdd244fd5538c91f884db936fec` |
| C0-L RE2 run manifest | `b93ed5ad6c71d36e998b89a2df8b32700f4a25a879a152bf33d08a8350006563` |
| C0-T GAIA run manifest | `84b552b0f748ea59ac6adb3cd91132dcb09f5794d720b036af88e93ee815b93e` |
| C0-T RE2 run manifest | `4e84961775a58667b2707c8267b4e5c6a3c929f07d20991fd8751afea50f9c15` |
| C1-I GAIA run manifest | `57ff60315a49b703a459d818c78bdb914b980f040cc3fe65f4a82b8baac2e65e` |
| C1-I RE2 run manifest | `ed5f6d02cfe6f9c5b1a5cffe15aa8b26c5810e263d8b635f26557401ecc3898d` |

## 9. 命令与产物

```text
python scripts/extract_p2_event_features_smoke.py
python scripts/extract_p2_event_features.py
python scripts/run_p2_linear_ablations.py
python scripts/audit_p2_linear_ablations.py
python scripts/bootstrap_p2_c1_i.py
```

产物：

- `artifacts/p2/event_features/{gaia_main,re2ob}/{p2_log_l0_v1,p2_trace_t0_v1}/`
- `artifacts/p2/runs/{c0_l,c0_t,c1_i}/{gaia_main,re2ob}/`
- `artifacts/p2/{event_feature_summary,event_feature_audit}.json`
- `artifacts/p2/{linear_ablation_summary,linear_ablation_audit}.json`
- `artifacts/p2/c1_i_bootstrap.json`

## 10. 限制

1. RE2-OB 天花板效应：C1-I 主端点已达 0.9956，该数据集对后续方法的区分度极低；
   是否扩展 RE2-TT 等 H1 结果出来后再决定，当前不重开 P1/P2 数据协议。
2. RE2-OB 单例分层使三层指标同值，fault-type / root-service macro 不提供独立信息。
3. RE2-OB trace entity-observed 比例仅 0.6364，融合增益部分依赖 mask 通道。
4. GAIA 上所有 learned 方法主端点仍低于未学习的 P1 B2。
5. 逐 fold 切片由 predictions 重算补齐，未写回 summary artifact；如需 artifact 级
   固化应在下一次 gate 时扩展 summary schema 并 bump 版本。

## 11. 下一步

运行 M1-S（stage-aware 单因素对照），随后独立 audit 与 M1-S vs C1-I 配对 bootstrap，
给出 P2-G4 go/no-go。M1-S 不改动 inner selection 规则、不缩减 `C` / onset 网格。
M2-R / M2-D / M3-G 在 P2-G4 结论之前不实现。
