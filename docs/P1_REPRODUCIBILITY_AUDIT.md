# P1 Reproducibility and Gate Audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-19
- Verification Status: UNVERIFIED
- Version Label: p1_reproducibility_audit_v1

> 状态：RCAEval 本地内容已固定；P1 G1–G8 与 reproducibility closeout completed

## 1. 审计结论

P1 已完成。最终审计同时验证了：

- GAIA 16,200-case inventory、13,470-case main cohort 与 2,730-case
  multi-root sensitivity 的集合和哈希绑定；
- RE2-OB 90 个 case 的 case manifest、实际消费文件和原始归档内容身份；
- 两数据集选定 split 的全覆盖与 group integrity；
- GAIA inventory、GAIA main、RE2-OB 各三个 baseline，共 9 组完整 rankings；
- 每组 baseline 的 output checksum、case/split 对齐、指标重算与训练审计；
- prediction records 中不存在 `root_service` 或 `fault_type`；
- 全量 telemetry diagnostics 的 349,120,558 行与零未完成文件；
- 51/51 单元测试、全量 Python 编译和 `git diff --check`。

机器审计产物 `artifacts/p1/gate_audit.json` 的
`all_checks_passed=true`、`raw_re2_source_bytes_verified=true`。

## 2. RCAEval 内容身份

本地 RCAEval 目录没有可用于唯一辨认 RE2-OB 的 commit、tag、DOI、README 或
version 文件。因此不把文件修改时间或“最新版”当作版本标识，而采用两层
content-addressed snapshot：

1. 对原始 `RE2-OB.zip` 计算完整 SHA-256；
2. 对 90-case manifest 实际引用的 metrics、logs、traces、inject-time 文件逐个
   计算 size 与 SHA-256。

范围和结果：

| 范围 | 文件数 | 字节数 |
|---|---:|---:|
| 实际消费 metrics | 90 | 608,436,662 |
| 实际消费 logs | 90 | 1,536,192,544 |
| 实际消费 traces | 90 | 6,296,835,235 |
| 实际消费 inject time | 90 | 900 |
| 实际消费合计 | 360 | 8,441,465,341 |
| 原始 RE2-OB 归档 | 1 | 1,191,025,569 |

快照使用相对路径，不把机器绝对路径写入内容身份。它同时绑定当前 90-case
manifest SHA-256
`a28dba97ab089ccfbfe963b0054a84d0c02fb1124e196070f2df4cda74327343`
与 `sources.jsonl` SHA-256
`c8d8efcb1f9aec98d6dde34aeb7350d3c9d48b4097b93f5219af15e334fb0f1c`。

## 3. 关键哈希

| Artifact | SHA-256 |
|---|---|
| RE2 content identity | `cec0030da8b499914af7979283eb1a2cd4f2cad2430742c70130829cd28133c9` |
| Source snapshot manifest | `61f369b5e8b0cb3aaf540decbf585fb1643b7d4c153cb0dff4af1f445c07fab3` |
| Consumed-file index | `30ec786aaa26bf1126eea766d587f8812a6921f5856e77b20244f7aa01b9c5b7` |
| Archive index | `375d3d2e57c9f8ab1306eb1be1c4f2c324f52142c6de4a4158a3a20a6178ddd1` |
| Raw `RE2-OB.zip` | `0605a36cdcad8a6ae0107f2357c9c91ecee2c4ab5d72579bffea0372d9747513` |
| Final gate audit | `a02625a0a37136f0d763c8c7fa83d39b003c15eeac033310f6990ab55ce1bf45` |

## 4. 执行命令

生成快照：

```text
python scripts/pin_p1_rcaeval_source.py --case-manifest artifacts/p1/manifests/re2ob --source-root /home/zhangll24/RCA_project/datasets/RCAEval --archive /home/zhangll24/RCA_project/datasets/RCAEval/RE2-OB.zip --output artifacts/p1/source_snapshots/re2ob
```

逐字节复核：

```text
python scripts/pin_p1_rcaeval_source.py --case-manifest artifacts/p1/manifests/re2ob --source-root /home/zhangll24/RCA_project/datasets/RCAEval --archive /home/zhangll24/RCA_project/datasets/RCAEval/RE2-OB.zip --output artifacts/p1/source_snapshots/re2ob --verify-only
```

全门禁审计（包含再次读取原始 RE2 字节）：

```text
python scripts/audit_p1_gates.py --artifact-root artifacts/p1 --raw-source-root /home/zhangll24/RCA_project/datasets/RCAEval --output artifacts/p1/gate_audit.json
```

代码验证：

```text
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
git diff --check
```

## 5. 可复现性边界

该快照足以判定另一份本地数据是否与本次实验逐字节一致，也能发现解压数据被
局部修改而原始归档未变的情况。后续官方核验发现 RCAEval commit
`4695aa69f4f1f57b9094ca04ff235908b73a8e24` 的 downloader 将同名
`RE2-OB.zip` 指向 DOI `10.5281/zenodo.14590730`。但 Zenodo 在本轮返回 403/429，
未能取得远端 checksum 与本地 SHA-256 对照，因此 DOI 是明确的候选上游来源，
“本地文件与 DOI 发布逐字节相同”仍未验证。这个 provenance 限制需要披露，但
不阻塞 P2，因为 P1 的实际实验输入已经被完整固定。

## 6. 阶段转换

P1 的目标是建立统一、可审计、无标签泄漏的 benchmark layer，而不是提出新
模型。该目标已经达到。后续工作进入 P2：在不改变 P1 case、split、candidate、
metric 与 Label Firewall 的前提下，研究多模态事件表征和根因归因机制。
