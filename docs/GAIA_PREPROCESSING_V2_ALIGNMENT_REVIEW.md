# GAIA Preprocessing V2 全流程对齐审查

## Material passport

- Review base: `f405fce1107a9ed0ed3ced83de53c953f7bd6ff3`
- Reviewed implementation: `6bbf874f311030b7d6fbf7298f5dfd04159beec6`
- Review date: `2026-09-14`
- Spec: `GAIA_MULTIMODAL_PREPROCESSING_AUDIT_V1.md` and `GAIA_MULTIMODAL_PREPROCESSING_CHANGE_PLAN_V1.md`
- Formal preprocessing executed: `NO`
- Formal training/Test evaluation executed: `NO`
- Decision: `NO-GO FOR V2 FULL PREPROCESSING`

## Executive result

当前实现证明了 Ada-MGAD 核心模型可以接收 `raw_node=48, log_len=32,
raw_edge=8`，但 V2 仍只有 schema/transform primitives，没有把 raw GAIA 数据、
frozen schema、completion manifest 和下游 pipeline 接成一条可执行链。因此不能用
现有 `run_i1_ad.py preprocess` 执行新方案；该入口主动 fail closed 是正确行为。

一次 Train-only full audit 曾启动，但发现审计实现仍会把 Trace parent 的跨 service
冲突静默保留为 first match，随后立即中止。部分输出已移到
`/tmp/gaia_audit_full_6bbf874.incomplete`，不得用于 schema freeze。该问题已新增回归
测试并修正；修正后的 full audit 尚未重跑。

## End-to-end alignment matrix

| Stage | Status | Evidence / blocker |
|---|---|---|
| P0 audit source binding | PARTIAL | Train-only binding 已有；semantic counter/rate 后的最终 Metric audit 尚未完成 |
| Trace audit parent resolution | FIXED, NOT FULL-RERUN | `(trace_id, span_id)` 多 service 冲突现在标为 ambiguous；需重跑全量 |
| P0B policy | MISSING | exact quality/correlation policy、Log K、Trace edge support 未批准 |
| P1 frozen schema | MISSING | 无正式 policy/config/audit hash-bound schema |
| Frozen schema validation | PARTIAL | 已校验 Train/Test/GT firewall、维数与 slot order；尚未强制完整 scaler、canonical services、non-empty graph |
| Metric raw adapter | MISSING | primitives 已有；scope-aware registry 到统一 tensor 的 streaming adapter 未接入 |
| Log raw adapter | MISSING | stable/RARE/UNK primitive 已有；Drain frozen state artifact 与 streaming transform 未接入 |
| Trace raw adapter | MISSING | split-local 8D primitive 已有；7.8G trace 的 bounded-memory production adapter 未接入 |
| V2 completion manifest | MISSING | 尚无 schema/source/slot/graph/scaler hashes 和 atomic completion publication |
| `TimestampedArrayDataset` | CONDITIONALLY ALIGNED | 只要产物满足 `[T,10,Rm]`, `[T,10,L]`, `[T,10,10,8]` 即可读取 |
| `MyModel` dimensions | ALIGNED | synthetic `48/32/8` forward/backward PASS，loss finite |
| AD train/infer loader | PARTIAL | dimensions 动态读取；manifest loader 尚未验证 V2-specific bindings |
| Event stage | NOT REVISION-SAFE | 默认 prediction paths 仍硬编码 `artifacts/p5/v3/ad` |
| RCA adapter | SCIENTIFICALLY SEPARATE | 68D 不应重设计；pipeline paths 仍需 revision-safe 参数传递 |
| Pipeline lock/output roots | NOT ALIGNED | `run_i1_pipeline.py` lock、AD/Event/RCA/final paths 大量硬编码 `p5/v3` |
| Final manifest | NOT ALIGNED | `finalize_v3_manifest.py` checkpoint/data roots 仍硬编码旧 revision |

## Standards review

- **Blocking:** `materialize.py` 只有 shape/finite 校验；V2 config 会被 legacy
  `build_ad_data` 拒绝。当前没有唯一 production orchestration 或 completion manifest。
- **High:** schema 仍允许缺 scaler、空 graph 和未绑定 canonical service order；这类
  schema 不能驱动 production transform。
- **Judgement call — duplicated code / shotgun surgery:** 新 primitives 与
  `ad_preprocess.py` 旧 Metric/Log/Trace 实现并存。只有 V2 adapter 接入后，才能把旧
  函数降为明确 compatibility path。

## Spec review

- `freeze_preprocessing_schema(...)` 和 `materialize_ad_inputs(...)` 尚未实现。
- 尚无 `gaia_p5_v3_preprocessing_v2` policy/config/schema artifacts。
- Metric audit 尚未在 counter/rate 与 core aggregation 后冻结 quality/redundancy。
- Log audit 尚未保存可 hash-bind 的 Drain frozen state 和 exact stable cluster IDs。
- Trace 8D/scaler primitive 已实现，但真实 split-local streaming transform 未实现。
- P3 应使用目标 dimensions 做全链 smoke；现有 `run_i1_ad.py smoke` 仍使用 `4/6/4`。

## Required implementation order

```text
P0A-1 fix and test audit semantics
→ P0A-2 rerun full Train audit into a fresh revision directory
→ P0B approve label-free exact policy
→ P1 freeze and validate exact schema
→ P2 implement streaming Metric/Log/Trace adapters + atomic manifest
→ P3 revision-specific 48..64 / 32..64 / 8D smoke
→ P4 run V2 full preprocessing with an isolated lock/output root
→ P5 train/infer/events/RCA using only explicit revision paths
```

在上述步骤完成前，不应删除 V2 guard，也不应把旧 V3 preprocessing 输出重命名为
V2。模型维度兼容并不等价于数据生产链已经对齐。
