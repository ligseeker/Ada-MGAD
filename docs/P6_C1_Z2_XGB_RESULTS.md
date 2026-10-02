# P6-C1 68D + XGBRanker 单次移植结果

**状态：COMPLETE；证据等级：已复用 Test 的探索性配对对照。** 新工作树
`/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2-xgb`，分支
`experiment/p6-z2-xgb`，冻结执行源码 commit `e08f3a0`。执行协议见
[P6_C1_Z2_XGB_PROTOCOL_V1.md](P6_C1_Z2_XGB_PROTOCOL_V1.md)，唯一新 run
为 `experiments/p6/c1_z2_xgb/c1-z2-xgb-v1-seed20260826/`。

## 变量和有效性

沿用 P6-C1-v2 已封存的 3,225 个共同 Train case（折 935/1,168/1,122）
和 4,214 个 C0 Test detected episode。两模型使用相同的 10 个候选、
GAIA W300-B15/40-bin 68D 特征、detected Train/Test anchor、匹配和评价。
唯一方法改动是把 Arm C 的 Conditional Logit 程序换成来源方法的固定
XGBRanker 程序；前者按旧协议使用 GT-Train-fitted scaler，后者按来源
XGB 协议不使用 scaler。XGB 参数固定为 pairwise/200 trees/depth 3/0.05/
seed 20260826，XGBoost 2.1.4。检测器、threshold、offset、feature
schema 和 Train cohort 均未改动，没有 Test 调参。

新 run 的 `run_lock.json` 绑定执行 commit、源码、来源 run、引用 XGB
源码及环境。Train、label-free prediction、evaluation 三阶段 manifest 均为
`COMPLETE`，逐文件 SHA-256 封存。预测锁记录 4,214 episode、4,213 legal
ranking、1 invalid context、零 ranking failure，明确记录 Test matching/GT
未读取。评价阶段才读取绑定的 C0 Test matching 和 GT registry。旧 Arm C 的
AC@1 及 E2E @1/3/5 由新评价器逐项精确复算，`baseline_replay_gate=PASS`；
另外用独立的 CSV join 和整数 hit 计数复算了两臂 AC@1/3/5、MRR 和
E2E F1，结果相同。

## 同一 legal matched Test cohort 的 RCA

分母均为 **4,197**；列 `CL` 是原 P6-C1-v2 Arm C，`XGB` 是本次新模型。

| 指标 | CL | XGB | XGB − CL |
|---|---:|---:|---:|
| AC@1 | 2915/4197 = 0.6945 | 3093/4197 = 0.7370 | +0.04241 |
| AC@3 | 4171/4197 = 0.9938 | 4189/4197 = 0.9981 | +0.00429 |
| AC@5 | 4181/4197 = 0.9962 | 4194/4197 = 0.9993 | +0.00310 |
| MRR | 0.8444 | 0.8665 | +0.02205 |

AC@1 配对状态：两者均正确 2,631；仅 CL 正确 284；仅 XGB 正确 462；
两者均错误 820。净多正确 178 案。按 UTC onset day 的固定 10,000 次
cluster bootstrap，差值的 **描述性** 95% percentile 区间为
`[+0.02913, +0.05379]`，只有 10 个 day cluster，且 Test 曾被研究使用，
不能作为独立确认的置信证据。

## 完整 E2E Diagnosis

冻结 Stage-1 为 TP 4,198 / FP 16 / FN 1,589；GT 5,787，预测
episode 4,214。各 @k 的诊断 TP 为 legal matched 且 root 在 Top-k 的
event；错误/缺失排名与非法 context 对对应 episode 计 FP、对 GT 计 FN。

| k | 模型 | TP | P | R | F1 |
|---|---|---:|---:|---:|---:|
| @1 | CL | 2915 | 0.6917 | 0.5037 | 0.5829 |
| @1 | XGB | 3093 | 0.7340 | 0.5345 | 0.6185 |
| @3 | CL | 4171 | 0.9898 | 0.7208 | 0.8341 |
| @3 | XGB | 4189 | 0.9941 | 0.7239 | 0.8377 |
| @5 | CL | 4181 | 0.9922 | 0.7225 | 0.8361 |
| @5 | XGB | 4194 | 0.9953 | 0.7247 | 0.8387 |

F1@1 净增加 0.03560；@3 增加 0.00360；@5 增加 0.00260。无论
RCA 排序如何改进，1,589 个 Stage-1 missed event 都仍是诊断 FN。

| failure ledger 类别 | CL | XGB |
|---|---:|---:|
| EVENT_MISSED | 1589 | 1589 |
| EVENT_FALSE_ALARM | 16 | 16 |
| RCA_CONTEXT_INVALID | 1 | 1 |
| RCA_RANKING_MISSING | 0 | 0 |
| ROOT_OUTSIDE_TOP1（实际 rank 2–3） | 1256 | 1096 |
| ROOT_OUTSIDE_TOP3（实际 rank 4–5） | 10 | 5 |
| ROOT_OUTSIDE_TOP5（实际 rank >5） | 16 | 3 |
| SUCCESS_TOP1 | 2915 | 3093 |

## 改善的范围与限制

Train roots 中 `mobservice1/2` 共 **3,204/3,225 = 99.35%**；
legal matched Test 中这两个 root 共 **4,171/4,197 = 99.38%**，
`login_failure` 为 **4,164/4,197 = 99.21%**。全部 +178 个 AC@1 净收益
来自这两个 mobservice root（分别 +64、+114）。其余 26 个 root case
的 CL 与 XGB Top-1 都是 **0/26**；`memory_anomalies` 只有 30 案，
Top-1 从 5/30 降至 4/30。XGB 的 legal episode Top-1 预测中，
`mobservice1/2` 合计 4,185/4,213。因此该结果支持对主流两服务
login failure 的排序改善，不能声称解决了跨服务或少数故障 RCA。

GAIA 30s 网格、多事件同 bin、长时/内存事件的 Stage-1 漏报以及
supervision-OOS 共享预处理限制仍在。来源 RE2 方法是 W600，本文用
GAIA W300 68D 重新拟合，不能称为来源模型权重的直接复现。因 Test
多次复用，不再基于这次分数调整参数或继续跑 Test。若论文必须主张
XGB 的泛化优势，需在预先冻结的 Train-only 时间阻断验证或新的独立
数据集上复核；本次结果只可作为探索性 E2E 对照。
