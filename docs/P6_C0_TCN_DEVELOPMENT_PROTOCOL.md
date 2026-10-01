# P6 起点检测：条件性 TCN 编码器实验

## Material Passport

- Mode: run, conditional controlled development
- Source: onset30 protocol ac94227 and repaired C0 frozen Train inputs
- Human read status: not attested
- Evidence grade: development only; reused Validation; no Test inference
- Date: 2026-10-01

## 问题和启动条件

若 E1 起点监督及 E2 两个输出方式对比均未达到已冻结检测门槛，再比较一个小型
TCN编码器。编码器结构在E1结果完成前固定，不按Validation逐案错误设计救分规则。
工作树`Ada-MGAD-e2e-v2-c0onsettcn`，branch`experiment/p6-c0-onset-tcn`。
只有E1/E2完成并通过来源校验、均NO-GO后，入口才允许启动TCN训练。

## 唯一架构变量及归因边界

替换整个编码器计算包：原spatial/temporal attention+FFN改为预加权Trace的
入边/出边聚合，与Metric/Log embedding拼接后逐服务共享TCN。
这是编码器架构比较；不能进一步把收益单独归因为卷积或单独归因为Trace聚合。

保留原三模态Embed、逐窗口动态图、相同graph稀疏/KL项及系数.001、mean/max/std
service pooling和scalar head。superclass初始化后替换encoder，使其余模块初始
参数与相同seed的E1精确一致。无root label、reconstruction或node anomaly输入。

固定配置：joined32D、hidden32、kernel3、dilation1/2/4、每块2个卷积、dropout.2；
左侧padding、仅每时刻channel LayerNorm；每service独立序列，无BatchNorm或同行汇总。
卷积理论感受野29bin，但输入仍只有冻结10bin/300s，额外位置是左侧零填充。
Trace按冻结非零边数量归一化聚合；已预乘原动态图权重，不额外训练新的图。

## 控制和验证

- 起点30s目标、旧60s IGNORE、Fit派生pos_weight、同seed42、训练超参/预算/
  Validation checkpoint/threshold selector及merged-episode decoder与E1相同。
- 原training engine只加model factory参数，AST校验必须证明其余整个模块相同；
  输入hash、原embedding/graph/head source、窗口、时间和GT身份全部核对。
- 主要归因对比：TCN vs E1（同目标/decoder，仅encoder不同）。同时报告TCN vs
  repaired recent-onset60 NN作为总体方案对比，后者含目标和encoder两个累计变化。
- 使用已冻结效果门槛：对原修复后NN Recall+.05、F1+.02、P>=.98，
  single-onset R下降<=.03。报告fault/root macro、memory和多事件、完整逐案失败。
- 初步GO后才seed/时间复核和全GT E2E开发；不在本候选上追加hidden/depth/dropout搜索。
- 两项模型测试覆盖时间前视、同行扰动、batch-size、非encoder初始化与graph正则
  精确一致；正式run还须真实32窗order/batch1/2/32 gate，atol1e-5。

## 限制和当前状态

当前仅实现/合成验证，不训练。共享70%Train预处理包括Validation输入，原filled
observability mask与Trace parent可用性限制沿用；不宣称完整pipeline已获在线认证。
E1的训练文件、HEAD和run目录不会因本工作树开发而改变。

```bash
cd /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2-c0onsettcn
env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /home/zhangll24/miniconda3/envs/DAG/bin/python scripts/p6/run_c0_tcn_development.py \
  --config configs/e2e/gaia_p6_c0_onset_tcn_v1.json \
  --output-dir experiments/p6/c0_onset_tcn/tcn-v1-seed42 \
  --data-root /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/data/p5/v3_preprocessing_v2/ad \
  --artifact-root /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/artifacts/p5/v3_preprocessing_v2/ad \
  --registry /home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/artifacts/p5/v3/protocol/gt_event_registry.csv \
  --gpu --threads 8 --threshold-workers 8
```
