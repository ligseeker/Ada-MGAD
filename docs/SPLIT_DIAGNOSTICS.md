# P1 Split Assignment Diagnostics

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-19
- Verification Status: UNVERIFIED
- Version Label: p1_split_diagnostics_v1

> 状态：16,200 inventory assignment 已通过 group-integrity/checksum 校验；13,470-case main cohort 继承该 assignment；G6 completed

## 1. 目标与输入

本诊断把已冻结的 case manifest 转化为可直接供 OOF evaluation 使用的实际
fold assignment：

- GAIA：16,200 个候选 injection，±300 s 上下文与实际注入区间的并集形成
  322 个不可拆分原子组；
- RE2-OB：90 个官方 case，90 个 singleton groups；
- fold 数：5；
- seed：`20260819`，仅用于确定性 tie-breaking；
- source binding：同时绑定 `manifest.json`、`inputs.jsonl`、`labels.jsonl` 和
  `groups.jsonl` 的 SHA-256。

机器可读结果位于：

- `artifacts/p1/split_diagnostics.json`；
- `artifacts/p1/splits/gaia/`；
- `artifacts/p1/splits/re2ob/`。

## 2. 候选方案与预先声明的选择门槛

GAIA 比较两个方案：

1. `grouped_stratified_5fold`：任何 context group 不跨 fold，同时近似平衡
   root service、fault type 和低权重 joint stratum；
2. `temporal_block_5fold`：按 anchor time 对非交错 context groups 做连续动态规划
   分块，因此是 context-purged temporal block。

RE2-OB 使用相同的 grouped-stratified 算法；由于每个 case 都是 singleton，实际
等价于确定性的 stratified case-level 5-fold。

硬约束为完整覆盖、case 唯一、group 不跨 fold，以及每个“至少由 5 个原子组
支持”的 root/fault 类别在每折都出现。软门槛为：

- 最大 fold case-count 相对偏差不超过 0.15；
- 最大 root-service total variation (TV) 不超过 0.10；
- 最大 fault-type TV 不超过 0.10。

joint stratum 只按权重 0.25 参与软平衡，不要求每折覆盖。RE2-OB 的每个
root×fault cell 只有 3 个 case，从定义上不可能覆盖 5 折。

## 3. GAIA 结果

| 候选 | Fold sizes | Max size deviation | Max root TV | Max fault TV | 可行标签覆盖 | Eligible |
|---|---|---:|---:|---:|---|---|
| Grouped-stratified | 3239/3239/3239/3241/3242 | 0.00062 | 0.00426 | 0.00031 | 全部通过 | 是 |
| Context-purged temporal block | 3247/3262/3263/3222/3206 | 0.01049 | 0.01024 | 0.01013 | 失败 | 否 |

时间块的总体分布距离看似很小，是因为 `login failure` 占 15,478/16,200，
会支配 TV。逐类别覆盖揭示了被总体指标掩盖的问题：

- 12 条 `cpu_anomalies` 全部位于 temporal fold-4，fold-0 至 fold-3 均为 0；
- `access permission denied exception` 在 temporal fold-0 为 0；
- 因此 fold-4 作为测试集时，其训练集完全没有 CPU fault，不能作为主学习型
  baseline 的稳定 OOF 方案。

主方案据此冻结为 `grouped_stratified_5fold`。各折 fault counts 为：

| Fold | access | cpu | file-moving | login | memory | Cases |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 3 | 9 | 3094 | 130 | 3239 |
| 1 | 3 | 2 | 8 | 3095 | 131 | 3239 |
| 2 | 3 | 3 | 9 | 3094 | 130 | 3239 |
| 3 | 3 | 2 | 8 | 3097 | 131 | 3241 |
| 4 | 3 | 2 | 9 | 3098 | 130 | 3242 |

时间块 assignment 保留为 temporal-shift sensitivity artifact，但不作为主
OOF split。反过来，主 grouped split 的时间范围会交错，因此不能把其性能
解释为跨时期泛化证据。

后续 G1 inclusion audit 从 inventory 排除 2,730 个 anchor multi-root cases 作为
主表外 sensitivity；过滤后的 main fold sizes 为 2693/2684/2711/2705/2677，
仍覆盖全部 root/fault 类别，且继承原 group assignment，因此不会引入新的
context group crossing。

## 4. RE2-OB 结果

RE2-OB 五折均为 18 cases：

- 每种 fault（cpu/delay/disk/loss/mem/socket）每折恰好 3 条，fault TV = 0；
- 每个 root service 每折为 3 或 4 条，最大 root TV = 0.06667；
- case/group 完整性与全部可行 root/fault 覆盖均通过。

因此 RE2-OB 主方案冻结为 seed `20260819` 的
`grouped_stratified_5fold`（singleton case groups）。

## 5. 产物与哈希

| 产物 | SHA-256 |
|---|---|
| Split diagnostics | `bd174e785a6e44eb3b7f366defc232b63f35d01f82ac96b35b7f7f06646fceb7` |
| GAIA selected assignments | `996a03698f77fa8128c737ffc6547fe767e3f5d9d8fd7b430d86341c6dac7c90` |
| GAIA temporal alternative | `658db9916d6041064074228ee24fa6cc44c6a28b0cd90e286bc87d9a7791563e` |
| GAIA split manifest | `adb80cc7540893a541dd6dec9de5ad7add55d04906d8bb9ffc5ddd020323aecd` |
| RE2-OB selected assignments | `c2205f39fa4742e7a22e474664cfccd70f3827e154534965760f0d16caf30c78` |
| RE2-OB split manifest | `a131a73b4bf94cfc3074bb98d0b060b28384f41d8b7651c06503faf3cbe9e41d` |

上述哈希对应本报告写入前的最终确定性生成结果。每个 split manifest 还绑定
其 source manifest 和三个核心 JSONL 文件的 SHA-256。

## 6. 执行异常与修正

首次生成在 RE2-OB 质量门处 exit 1：初版纯贪心分配的最大 root TV 为
0.12632，超过 0.10；没有将该结果写成正式 assignment。随后加入确定性的
single-group move 与小规模 pair-swap 局部改进，并增加 90-case 两轴重复网格
测试。修正后 RE2-OB 达到 18/18/18/18/18，最终命令 exit 0。

## 7. 结论与下一步

- G6 从 partial 更新为 completed：实际 assignment、group-integrity validator、
  单元测试和 checksum-bearing split manifest 均已存在；
- 主学习型 preprocessing 必须逐 fold 只在其余 4 folds 上拟合；
- B0/B1/B2 已由后续 P1-SANITY-BASELINES 完成，见
  [BASELINE_RESULTS.md](BASELINE_RESULTS.md)；
- GAIA 的后续纳入审计已冻结 13,470-case main 与 2,730-case multi-root
  sensitivity，G1 已完成；见 [GAIA_INCLUSION_AUDIT.md](GAIA_INCLUSION_AUDIT.md)。
