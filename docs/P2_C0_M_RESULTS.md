# P2 C0-M Nested OOF Results

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-19
- Verification Status: UNVERIFIED
- Version Label: p2_c0_m_results_v1

> 状态：P2-G2 completed；结果建立 metric-only learned baseline，不检验 H1/H2/H3

## 1. 协议

C0-M 只使用 `p2_metric_summary_v1` 的 `whole.*` 17 个值及其 17 个 masks，
不使用 stage、service identity、fault type 或 root frequency。scorer 为独立的 L2
logistic regression：

- outer：冻结的 5-fold OOF；
- inner：每个 outer-train 的四折轮换验证；
- `C ∈ {0.01,0.1,1,10}`；唯一目标为 inner root-service macro Avg@5；
- 并列优先较小 C；StandardScaler 只在当前 fit rows 拟合；
- 每 case 总权重 1：root row 0.5，全部 non-root rows 合计 0.5；
- 分数并列按 service name 升序，输出完整候选排列。

GAIA 使用 13,470-case main；RE2-OB 使用 90 cases。seed 为 `20260819`。

## 2. OOF 结果

| Dataset | Overall AC@1 | Overall Avg@5 | Fault-macro AC@1 | Fault-macro Avg@5 | Root-macro AC@1 | Root-macro Avg@5 |
|---|---:|---:|---:|---:|---:|---:|
| GAIA main | 0.1633 | 0.3953 | 0.4096 | 0.5523 | 0.4567 | 0.5820 |
| RE2-OB | 0.9111 | 0.9756 | 0.9111 | 0.9756 | 0.9111 | 0.9756 |

所选 C：

- GAIA outer folds：10/10/10/10/10；
- RE2-OB outer folds：0.1/10/0.1/1/10。

## 3. 与 P1 B2 Metric Change 的同口径差值

| Dataset | Δ Overall AC@1 | Δ Overall Avg@5 | Δ Fault-macro AC@1 | Δ Fault-macro Avg@5 | Δ Root-macro AC@1 | Δ Root-macro Avg@5 |
|---|---:|---:|---:|---:|---:|---:|
| GAIA main | +0.0228 | +0.0376 | −0.0496 | −0.0458 | −0.0743 | −0.1266 |
| RE2-OB | +0.0556 | +0.0422 | +0.0556 | +0.0422 | +0.0556 | +0.0422 |

结果存在数据集分化。C0-M 在 RE2-OB 超过 B2，但在 GAIA 的主端点 root-macro
Avg@5 明显低于 B2；它只在受类别频率影响更强的 GAIA overall 指标上提高。因此：

- C0-M 是后续 C0-L/C0-T/C1-I 的合法 learned reference；
- 不宣称其统一优于 P1 B2；
- 不根据该结果改动 P1 B2 或主 endpoint；
- H1/H2/H3 仍完全未检验。

## 4. 泄漏与复现审计

`scripts/audit_p2_c0_metric.py` 独立完成：

- 两数据集各 5 outer folds、80 inner fits；
- 所有 inner fit/validation 与 outer train/test case overlap=0；
- 所有 group overlap=0；
- predictions 的 root/fault/ground-truth token 零命中；
- 完整 rankings 经 evaluator 重算，metrics 与记录逐字相等；
- selected C 可从保存的 inner objective 独立重建。

完整实验复跑后，两个数据集的 predictions、metrics、training audit 六个核心文件
SHA-256 全部一致。runtime 单列为 informational，不进入核心 hash。

| Artifact | SHA-256 |
|---|---|
| C0-M summary | `74a39bfbc447c1c7711e52c2b1e297b146f4be4e06c860a545bb5951c175936b` |
| C0-M independent audit | `32db253baba2de3f00b9059caf2404f04469515971de28c95cf698cd961a6592` |
| GAIA run manifest | `a6d2d8dd5698a6971daf7ccd5722a2167a1e451704349abe5ccc05191ec187f5` |
| RE2 run manifest | `f0234a8f85a453255d6a64ca071591fcbf430ca7633a1119310f77d39c36b15a` |

## 5. 命令与产物

```text
python scripts/run_p2_c0_metric.py
python scripts/audit_p2_c0_metric.py
```

产物位于 `artifacts/p2/runs/c0_m/{gaia_main,re2ob}/`，包含 per-case scores 与完整
rankings、metrics、80-inner-fit/5-outer-fit audit 和 source-bound run manifest。

## 6. 下一步

先完成 raw L0/T0 feature schema、coverage 与单模态 C0-L/C0-T，再进行 C1-I。
不应直接把 stage features 加到 C0-M，因为那会跳过 best-single-modal 与独立融合
对照，也无法把 H1 增益与模态增益分开。
