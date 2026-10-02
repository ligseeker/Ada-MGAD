# 正常预测 Fit screen V2：TCN 稳定性失败后的启动条款

## Material Passport

- Date: 2026-10-01
- State: protocol amendment before any normal-forecaster scores
- Trigger: sealed TCN initialization replication `STABILITY_NO_GO`
- Scope: original Fit internal screen; no Validation telemetry or Test inference

## 证据与适应性边界

TCN seed42 fixed-bin初步GO为2577/14/324；预先指定17与2026复制为
2605/414/296与2488/3388/413，两个新seed的预声明bin重选阈值也NO-GO。
全部三seed时间半段Recall增益仍通过，但误报/总体F1不稳定，完整E2E入口关闭。
监督模型训练和分析全部封存，实际window-independence gate通过。
源协议、完整成员及分数/GT/cohort绑定均须验证；不挑最佳seed继续。

V1正常预测启动guard只表达单seed初轮NO-GO，无法表达随后才注册的跨seed
稳定性失败。本V2是**在看到上述Validation稳定性结果后**补充的新启动协议。
不把这个分支触发条件称为最初预注册；预测器本身及全部Fit-screen决策仍
采用此前冻结且尚未训练的V1方案。新路径不改动旧源/配置/产物。

## 唯一假设、方法与失败退出

检验：仅从Fit内无登记事件的轨迹学习下一bin的45D数值预测，残差能否在
固定60秒事件容差内提供超出Persistence的事件检测信号。
该方法正常样本监督、故障标签只用于正常资格和最终事件评价；不用起点二分类
损失，也不使用故障/root标签拟合预测器。不假定预测误差一定对应因果故障。

所有规则完全保留[V1协议](P6_NORMAL_FORECAST_FIT_SCREEN_PROTOCOL.md)：
Fit内60/20/20、300秒history purge、9bin输入/1bin目标、45数值+3mask输入、
仅预测45数值、10服务共享TCN、seed42、固定20epoch最后checkpoint、
Huber/AdamW固定超参、正常cal99.5%分位、mean45→max10 residual、merged decoder、
Persistence参考、60秒因果一对一匹配。真实批次独立gate及源/HEAD/输入闭合必需。

GO同时要求 P≥.90、R≥.60、F1≥.72、F1相对Persistence≥+.02、
无登记事件holdout窗口阳性比例≤.01。正常样本支持各块≥512；clean-memory13
不足30，稀有干净起点解释仍受限，不据其选择参数。Fit完整holdout GT1536。
失败即停止这个固定候选，不更换分位阈值/offset/分数聚合/模型或找最佳seed。
进入本轮Scientific Freeze评估；通过才另行冻结原Validation开发实施方案。

## 实现和绑定

- V1分派保留原guard；V2另核已完成stability manifest、report、其config与全部
  model/decoder成员的source/artifact hashes，17/2026两个bin变体均NO-GO。
- config除protocol/start evidence之外与原V1精确相同；训练/预测/指标函数及
  main AST与原815b568源相同，模型与Fit state源码字节相同。
- 仅原Train的Fit行参与新训练/校准/评分。全局registry的interval/domain列
  扫描用于路由，Test service/fault字段不解析；启动读取已锁Validation决策JSON
  不等于新Validation遥测推理，适应性如上披露。
- 唯一新run，开始前scope/source/配置/数据锁，checkpoint、校准阈值、全部
  holdout预测先seal，再做score-to-GT metric join；cohort/normal资格使用GT
  的既有规则照常披露，不伪称完全不知道Fit holdout标签。
- frozen70%Train preprocessing含留出段协变量；filled-mask、正常GT不完备、
  单seed和当前复用开发数据局限保留。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
 /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_normal_forecast_fit_screen.py train \
 --config configs/e2e/gaia_p6_normal_forecast_stability_v2.json \
 --output-dir experiments/p6/normal_forecast_fit_screen/stability-fallback-v2-seed42 \
 --data-root ../Ada-MGAD-e2e-v2/data/p5/v3_preprocessing_v2/ad \
 --artifact-root ../Ada-MGAD-e2e-v2/artifacts/p5/v3_preprocessing_v2/ad \
 --registry ../Ada-MGAD-e2e-v2/artifacts/p5/v3/protocol/gt_event_registry.csv --gpu --threads 8
```
