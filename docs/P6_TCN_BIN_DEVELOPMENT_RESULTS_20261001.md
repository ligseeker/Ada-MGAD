# P6 TCN + 独立bin输出：开发结果与后续复制

## Material Passport

- Date: 2026-10-01
- Mode: completed controlled development evaluation
- Training source: `6dbc908468d68fec4b671dcb8062a3fd12ea3276`
- Analysis source: `0945667` in `experiment/p6-c0-normal-forecast`
- Evidence: reused Validation; no new Test, RCA or full E2E result

## 完整结果

Validation17414窗口、GT2901；时间、标签各自目标、全部GT身份/metadata、
Train输入/源与完成清单均核验。TCN训练10epoch，选epoch1（0-based），
24078参数；原NN2903354参数。真实窗口batch1/2/32最大误差7.15e-7，
同行扰动和顺序扰动误差0，gate通过。三个分析完成清单全部封存。

| 方案 | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| repaired recent60 NN |2103|19|798|.991046|.724922|.837348|
| E1 onset30 NN merged |2057|131|844|.940128|.709066|.808410|
| TCN onset30 merged |2127|6|774|.997187|.733195|.845054|
| TCN bins，固定merged阈值（主结果） |2577|14|324|.994597|.888314|.938456|
| TCN bins，预声明Val重选阈值（辅助） |2588|23|313|.991191|.892106|.939042|

TCN merged阈值`.9025284051895142`；独立bin sweep阈值`.8988370299339294`。
主结果不需要重选阈值。每个阳性完成bin生成一条候选，一对一匹配，所有多余
预测照常计FP；没有改GT、容差、预测时刻或分母。

## 归因与paired

- 同目标encoder包对比TCN vs E1：新命中126、丢失56、净70，R+.024130，
  F1+.036643；该对比只支持完整encoder包，不能归因给单独卷积/Trace聚合。
- TCN merged vs repaired baseline：新47、丢失23、净24，R+.008273、
  F1+.007705，仍NO-GO。
- TCN bins固定阈值 vs 同TCN merged：新450、丢失0。这个450是固定分数下
  输出容量/解码改善，不能称为增加了450个独立新事件识别能力。
- TCN bins固定阈值 vs repaired baseline：新486、丢失12、净474，
  R+.163392、F1+.101108，FP19→14；所有冻结开发门槛通过。
- 单起点1947→2313/2521；多起点156→264/380。单起点指同30s onset bin
  的起点数为1，不等于附近没有活动/相邻事件。
- memory14→16/110；长事件16→21/125。巨大整体增益没有解决稀有长故障。
  候选login_failure2556/2776=.920749，仍由短login事件主导。

主候选剩余324 FN：below101、matching competition222、no-legal1；
no-new-episode0。竞争位置不能单独证明唯一的网格容量原因或模型不可辨识。

## 当前决定及边界

`GO_TO_REPLICATION`，先复核候选的种子与时间稳定性，再进入Fit-OOS锚点和
固定68D+XGB的完整Validation E2E开发。主候选保持固定merged阈值的bin decoder；
不选择辅助阈值或另一个随机种子刷分。初步GO不等同于稳定性/完整E2E通过。

正常预测支线仅准备代码/完成Fit预检，不训练：正常窗3987/1077/1184，
完整event-holdout1536，clean-memory13不足30。当前TCN bin GO使其训练启动条件
不满足；先验证已改善的候选。

共享70%Train preprocessing包括Validation无标签输入，原filled mask和Trace
parent可用性局限继续保留；不宣称完整pipeline在线认证或独立确认。
复用Validation，必须用未触碰的新时间段作后续独立确认；本轮不重跑Test。

## 产物

- TCN：`../Ada-MGAD-e2e-v2-c0onsettcn/experiments/p6/c0_onset_tcn/tcn-v1-seed42/`
- 本树：`experiments/p6/c0_onset_tcn/encoder-vs-e1-v1/`
- 本树：`experiments/p6/c0_onset_tcn/total-vs-baseline-v1/`
- 本树：`experiments/p6/c0_onset_tcn/bin-decoder-v1/`
- 本树：`experiments/p6/normal_forecast_fit_screen/preflight-v1/`
