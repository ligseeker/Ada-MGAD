# P2 G4 Event-stage 单因素对照结果（M1-S）

## Material Passport

- Origin Skill: claude-code / experiment-governance
- Origin Mode: run
- Origin Date: 2026-08-20
- Verification Status: ARTIFACT-VERIFIED
- Version Label: p2_g4_stage_results_v1

> 状态：M1-S run completed；独立 audit completed；配对 bootstrap completed。
> **P2-G4 判定为 `no-go`**（`exploratory_signal=false`、`claim_ready=false`）。
> H1 未通过冻结门禁；H2 / H3 仍未检验。

- Data version：GAIA MicroSS 13,470-case main cohort / RCAEval RE2-OB 90 cases
- Split：`artifacts/p1/splits/{gaia,re2ob}`，`p1_split_manifest_v1`，5-fold OOF
- Seed：`20260819`
- Git branch / commit：`claudecode` / `64bb681328fa1793014615a20ba2bc1fdf33c3b7`
- 命令：`python scripts/run_p2_m1_stage.py` →
  `python scripts/audit_p2_m1_stage.py` → `python scripts/bootstrap_p2_m1_s.py`
- 产物：`artifacts/p2/runs/m1_s/{gaia_main,re2ob}/`、
  `artifacts/p2/{m1_s_summary,m1_s_audit,m1_s_bootstrap}.json`

## 1. 协议与 design

M1-S 复用 P2-G2 / P2-G3 冻结的 nested OOF 装置，**唯一变化是输入 design 增加了
event-stage 通道**，因此它是 C1-I 的单因素对照：

| 通道 | 内容 | 值列 |
|---|---|---:|
| `whole.*` | metric 17 + log 5 + trace 8（= C1-I 的全部输入） | 30 |
| `stage{60\|120}.*` | metric 51 + trace 24 | 75 |
| 合计 | 105 值 + 105 等宽 masks | **210 列** |

- log 的 staged 通道被排除，run manifest 的 `log_stage_excluded` 记录理由：
  RE2 log content-complete 0.794900 低于冻结的 0.80 绝对阈值（决策发生在 run 之前）。
- 网格 `C ∈ {0.01,0.1,1,10}` × `onset ∈ {60,120}` = 8 个候选，未缩减。
- inner validation 唯一目标仍为 inner root-service macro Avg@5；tie-break 仍为
  较小 C → 较短 onset。**未按 C1-I 的 GAIA AC@1 退化调整选择目标**（用户已确认）。
- 每 outer fold 32 inner fits ⇒ **每数据集 160 inner + 5 outer fits**，为
  C0-M / C1-I 的两倍。
- Runtime（informational）：GAIA 12,326.37 s（≈3 h 25 min）、RE2-OB 24.01 s。

## 2. OOF 结果

GAIA main（13,470 cases）：

| 层 | AC@1 | AC@3 | AC@5 | Avg@5 | MRR |
|---|---:|---:|---:|---:|---:|
| Overall | 0.5185 | 0.8711 | 0.9623 | 0.8086 | 0.7025 |
| Fault-type macro | 0.6461 | 0.8138 | 0.8803 | 0.7936 | 0.7560 |
| **Root-service macro（主层）** | **0.4055** | 0.6588 | 0.7886 | **0.6319** | 0.5768 |

RE2-OB（90 cases，单例分层 ⇒ 三层同值）：

| 层 | AC@1 | AC@3 | AC@5 | Avg@5 | MRR |
|---|---:|---:|---:|---:|---:|
| 三层同值 | 0.9667 | 1.0000 | 1.0000 | 0.9933 | 0.9833 |

## 3. 相对各参照的主层差值

| 参照 | GAIA Δ AC@1 | GAIA Δ Avg@5 | RE2 Δ AC@1 | RE2 Δ Avg@5 |
|---|---:|---:|---:|---:|
| **C1-I（单因素对照对象）** | **+0.039918** | **+0.024646** | **−0.011111** | **−0.002222** |
| C0-M（best single） | −0.051214 | +0.049907 | +0.055556 | +0.017778 |
| C0-L | +0.243351 | +0.275133 | +0.677778 | +0.526667 |
| C0-T | +0.233320 | +0.276926 | +0.344444 | +0.177778 |
| P1 B2 Metric Change（未学习） | −0.125551 | −0.076702 | +0.111111 | +0.060000 |

两点必须保留的读数：

1. **GAIA 上 M1-S 相对 C1-I 在主端点和次端点同时为正**，且次端点 AC@1 由 C1-I 的
   −0.0911（相对 C0-M）转为 +0.0399（相对 C1-I）——即 stage 通道在 GAIA 上同时
   改善了两个端点。
2. **GAIA 上 M1-S 主端点仍低于未学习的 P1 B2**（−0.0767，C1-I 时为 −0.1013）。
   差距缩小但未反转，因此"learned 方法在 GAIA 主端点不优于 B2"这一 P2-G3 结论
   **在 P2-G4 依然成立**，不得改写。

## 4. 逐 fold 结果（root-service macro）

GAIA main，fold 规模 2693 / 2684 / 2711 / 2705 / 2677：

| 方法 | 指标 | fold_0 | fold_1 | fold_2 | fold_3 | fold_4 |
|---|---|---:|---:|---:|---:|---:|
| C1-I | AC@1 | 0.3822 | 0.3560 | 0.3673 | 0.3300 | 0.3739 |
| C1-I | Avg@5 | 0.6276 | 0.6038 | 0.5986 | 0.5879 | 0.6135 |
| M1-S | AC@1 | 0.3703 | 0.4308 | 0.4182 | 0.3838 | 0.4231 |
| M1-S | Avg@5 | 0.6144 | 0.6531 | 0.6559 | 0.6078 | 0.6310 |

RE2-OB，每 fold 18 cases：

| 方法 | 指标 | fold_0 | fold_1 | fold_2 | fold_3 | fold_4 |
|---|---|---:|---:|---:|---:|---:|
| C1-I | AC@1 | 1.0000 | 0.8833 | 1.0000 | 1.0000 | 1.0000 |
| C1-I | Avg@5 | 1.0000 | 0.9767 | 1.0000 | 1.0000 | 1.0000 |
| M1-S | AC@1 | 1.0000 | 0.8167 | 1.0000 | 1.0000 | 1.0000 |
| M1-S | Avg@5 | 1.0000 | 0.9633 | 1.0000 | 1.0000 | 1.0000 |

GAIA 上 M1-S 在 4/5 折的主端点高于 C1-I，仅 fold_0 反转（0.6144 vs 0.6276）。
RE2-OB 上两方法在 4/5 折完全相同，全部差异集中在 fold_1。

每折所选超参：

| Dataset | selected C | selected onset (s) |
|---|---|---|
| GAIA main | 10 / 10 / 10 / 10 / 10 | 60 / 60 / 60 / 60 / 60 |
| RE2-OB | 1 / 1 / 10 / 0.1 / 0.1 | 120 / 60 / 120 / 120 / 120 |

GAIA 五折一致选中 60 s onset 与 C=10；RE2-OB 在 4/5 折选中 120 s。两个数据集的
inner selection 落在不同 onset 上，这一分歧本身是 P2-G4 的确定性观察结果。

## 5. 配对 bootstrap（M1-S vs C1-I）

`scripts/bootstrap_p2_m1_s.py`，schema `p2_m1_s_paired_bootstrap_v1`，10,000 次，
seed `20260819`，`numpy.default_rng/PCG64`；GAIA 按 322 context group 重采样，
RE2-OB 按 90 case 重采样。

| Dataset | 指标 | point Δ | 95% CI lower | 95% CI upper | P(Δ ≤ 0) |
|---|---|---:|---:|---:|---:|
| GAIA main | Root-macro Avg@5（主） | +0.024646 | +0.005371 | +0.043954 | 0.0070 |
| GAIA main | Root-macro AC@1（次） | +0.039918 | +0.008128 | +0.070546 | 0.0066 |
| RE2-OB | Root-macro Avg@5（主） | −0.002222 | −0.007500 | 0.000000 | 1.0000 |
| RE2-OB | Root-macro AC@1（次） | −0.011111 | −0.037500 | 0.000000 | 1.0000 |

## 6. Go/No-Go 判定：`no-go`

按冻结规则逐条核对：

| 判据 | 要求 | 实际 | 结论 |
|---|---|---|---|
| `exploratory_signal` | 两数据集主端点点估计**均**为正 | GAIA +0.024646；RE2 **−0.002222** | **false** |
| `both_primary_ci_lower_bounds_positive` | 两数据集主端点 CI 下界 > 0 | GAIA +0.005371；RE2 **−0.007500** | **false** |
| `both_secondary_point_deltas_at_least_minus_0_01` | 两数据集次端点 ≥ −0.01 | GAIA +0.039918；RE2 **−0.011111** | **false** |
| `claim_ready` | 上两项同时成立 | — | **false** |

⇒ `p2_g4_decision = "no-go"`。**H1 在冻结门禁下未获支持。** 不得在任何论文级
文本中宣称 event-stage 切分带来一致收益。

M1-S 与 C1-I 的判定形态是**互补而非一致**：C1-I 在 RE2 通过、在 GAIA 被次端点
拖垮；M1-S 在 GAIA 双端点通过、在 RE2 被主端点符号与次端点 guardrail 拖垮。
两个 gate 都没有出现"两数据集同向"的证据。

## 7. RE2-OB 退化的确定性溯源

RE2-OB 的全部退化来自**一个 case**：

- 逐 case 比对 `runs/{c1_i,m1_s}/re2ob/predictions.jsonl`：89/90 个 case 的候选
  排列发生了尾部重排，但**只有 1/90 个 case 的 root 排名位置发生变化**；
- 该 case 为 `re2ob-c9c8f348d3f5974b`（fold_1，root = `recommendationservice`），
  root 由 rank 1 落到 rank 2；
- 该数据集 5 个 root service × 每 root 18 cases，因此 1 个 case 使该 root 的
  AC@1 下降 1/18，宏平均下降 (1/18)/5 = 0.011111，与记录的
  −0.011111111111111072 完全一致；
- root 仍在 top-3/top-5 内，故 AC@3 / AC@5 差值恰为 0.0，
  Avg@5 差值 = −0.011111/5 = −0.002222，同样与记录一致。

这是 P2-G3 已记录的 RE2-OB 天花板效应在门禁上的直接后果：C1-I 已达 0.9956，
剩余 headroom 不足一个 case，单个 case 翻转即可同时让主端点符号变负、并越过
−0.01 的次端点 guardrail。**该溯源只作诊断，不改变第 6 节的判定**，也不构成
放宽阈值或更换数据集的依据。

附带记录：本次 guardrail 差值为 −0.011111，与 −0.01 的距离远大于浮点表示误差，
因此 `audit_p2_m1_stage.py` 与 `bootstrap_p2_c1_i.py` 采用的无容差比较在本次
判定中不存在边界歧义。

## 8. 泄漏与复现审计

`scripts/audit_p2_m1_stage.py`（schema `p2_m1_stage_audit_v1`）对两个数据集独立
完成，全部通过：

| 检查项 | GAIA main | RE2-OB |
|---|---|---|
| `outer_fold_count` | 5 | 5 |
| `inner_fit_count` | **160** | **160** |
| `outer_fold_ids_match_frozen_split` | true | true |
| `prediction_folds_match_frozen_split` | true | true |
| `test_overlap_max` / `test_group_overlap_max` | 0 / 0 | 0 / 0 |
| `inner_fit_validation_overlap_max` / group | 0 / 0 | 0 / 0 |
| selected C + onset 由 inner scores 重建 | 一致 | 一致 |
| `metrics_exactly_recomputed` | true | true |
| `label_free_predictions` / `label_firewall_flags_all_false` | true / true | true / true |
| `core_files_verified` | true | true |
| `comparator_metrics_exactly_recomputed`（C1-I） | true | true |
| `design_column_count` | 210 | 210 |

第二次运行 `audit_p2_m1_stage.py` 与 `bootstrap_p2_m1_s.py`（输出至 scratch 路径，
不覆盖既有产物）得到**逐字节相同**的 SHA-256，确定性契约成立。

| Artifact | SHA-256 |
|---|---|
| M1-S summary | `d77b31436e1eb0c0ec6dac61ad917d31418d82988e1b819b61c84be17a5dc495` |
| M1-S independent audit | `930c3ae14269a3dc7ecb64187f18f2297efd37605bf6a0fd2d0b7fe247f82910` |
| M1-S paired bootstrap | `65788c2dd7f8c79d82c4494255cf2b2295e64ab8f983f9ea48b33ca84f0b323a` |
| M1-S GAIA run manifest | `b774070c77b4fa4ae9e01bcc055111992fc52208aba1ca859bffa0de41667687` |
| M1-S RE2 run manifest | `e96efb25ea73845ab01c368ca51ea692a141be1948d1eae1f33fddc3b785f0bb` |

## 9. 限制

1. **RE2-OB 天花板效应已实质阻断该数据集的判别力**：C1-I 0.9956 / M1-S 0.9933，
   1 个 case 即决定门禁方向。该数据集当前无法为 H1 提供有效的正/负证据。
2. GAIA 上所有 learned 方法（C0-M / C0-L / C0-T / C1-I / M1-S）主端点仍低于未学习
   的 P1 B2。
3. 两数据集的 inner selection 落在不同 onset（GAIA 全 60 s、RE2 多为 120 s），
   单一冻结 onset 网格下的"最优 stage 粒度"不具跨数据集一致性。
4. M1-S 同时引入了 stage 切分与 75 个新值列，因此它是"stage 通道整体"的单因素
   对照，不能分离"切分本身"与"特征维度增加"的贡献。
5. 逐 fold 切片与 RE2 单 case 溯源由 predictions 重算得到，未写回 summary artifact。
6. `scripts/run_p2_m1_stage.py` 无 outer-fold checkpoint/resume；本次为一次性
   完整运行。

## 10. 下一步（需用户决策，本轮不执行）

P2-G4 判定为 `no-go`，按既定停止点本轮工作在此结束。以下选项互斥，**任何一项都
需要用户明确决定后才实施**，且都不得回改已冻结协议或既有记录：

- 维持现状，把 H1 记为"未获支持"并据此重写方法叙述；
- 就 RE2-OB 天花板效应重新讨论是否扩展 RE2-TT（P2-G3 已把该决定推迟到 H1 结果之后，
  现结果已产生）；
- 对 inner selection objective 做单独的 protocol version bump + sensitivity
  experiment（P2-G3 已记录该动作必须独立于 M1-S 进行）。

M2-R / M2-D / M3-G 在上述决策产生前不实现。H2 / H3 未检验，不得在任何文本中
预判其结论。
