# 已有候选的冻结 Test 比较

## Material Passport

- 2026-10-01，用户明确要求所有已尝试、效果较好的方案在Test评估；文件探索由gpt-6-luna完成。
- 本树：experiment/p6-frozen-test-review，单独新输出目录，不覆盖任何旧run或共享输入。
- 候选清单只依据已完成Train/Validation，冻结后不依据Test换threshold、offset、seed、epoch或超参。
- Test此前已复用，全部新增结果是描述性探索，不称独立确认；旧NO-GO结论不追溯改为GO。

## 候选范围与排除

广义入选：已有完整模型/固定decoder且Validation F1>=.80，或预先评价的整体Recall/
故障组有收益；包括该family全部预定seed与主/辅助decoder。不能只选漂亮seed。
因此包括旧C0与C0R2、GPU control/long weight、Metric drift、修正逐窗口C0、
Flat XGB、onset30 merged及两个bin方案、TCN三seed各自merged/fixed-bin/aux-bin。
绝对F1合格但比较NO-GO的模型也进入本次观察，表中保留其原决定。
旧P5作为已有完整结果参考；其训练/检测协议不同，不作为受控差值归因。

RCA包括已有C1 B/C参考、68D XGB原锚点、Train锁定25.621s backdate，以及Train
前推两折均小幅正向的no_metric/no_all_magnitude。后二者按已有固定配方在锁定
3,225 Train case各拟合一次最终scorer；无搜索、不根据Test改变特征集合。
完整38组新/旧detector与原锚点/backdate后端的比较分别保留，后端不按Test选择。

不纳入：正常预测/Persistence Fit-screen低Recall失败；未执行的raw-mask、checkpoint
selector和native新detector-OOS RCA；没有冻结阈值的raw起点变化诊断；Train两折
都下降的mobpair分类器/GT混训候选。C0R3的64等预算方案是诊断对照，不是训练或
选定的部署方法，不扩充为64个Test模型。

## 顺序与不可变性

1. 提交新入口与完整JSON候选清单；绑定config、当前HEAD、来源selection/checkpoint/
   completion、数据/graph/manifest和源码。Test文件内容SHA只用于身份，不参与选择。
2. NN/TCN/Metric用已有checkpoint和原model_args；先重放三个实际Validation批次，
   包含尾部padding，概率误差<=1e-6、logit误差<=1e-4；不得因Test结果换实现。
   Flat XGB用原环境/权重和完全相同16,000D变换做96行Validation精确重放。
3. 输入型Test dataset只返回Metric/Log/Trace/sample_index。无label/registry API。
   保持10-bin窗口、完整冻结Test时间块、原32 batch及last-row padding；最后t=TestEnd
   的完整窗口保留。旧C0只读既有CSV的index/time/score/logit，标签列不解析。
4. 所有模型分数及19个固定decoder的episodes先形成global_prediction_lock；随后
   读取Test GT并一对一因果匹配0–60s，完整GT固定5,787。全部FP/FN、非法上下文、
   无排序/错误排序保留；报告P/R/F1、paired cases、fault/root/duration/multiplicity
   分组及failure ledger。该ledger定位失败位置，不证明根因。
5. RCA在独立工作树执行：先锁固定scorer/Train最终拟合/原始index/所有anchor规则。
   使用同一W300-B15、canonical10服务和68D Z2；无scaler。原XGB围绕t_hat，
   backdate模型围绕t_hat-25621ms。两种上下文分别检查完整落入Test块，失败不删除。
   新episode特征和完整ranking先锁，再与GT匹配结果连接做AC@1/3/5、MRR及
   full E2E P/R/F1@1/3/5。完整E2E的TP为匹配且root位于Top-k，FN=GT-TP，
   FP=全部predictions-TP；匹配错误root同时产生FP/FN。

## 解释边界

旧C0/weighted/Metric/P5的batch graph可消费同批较晚遥测，保留原batch构成用于
忠实观察；明确batch-conditioned retrospective，不当作在线因果方法。
修正C0/onset/TCN为window graph；Trace parent可用性尚未认证，RCA±300s本身也
需要后文，不声称零延迟在线E2E。共享预处理仍拟合原70%Train，包括detector Val。

新detector接旧RCA权重是另行注册的frozen-scorer transfer，来源训练锚点仍是旧
batch-conditioned监督，不等同于新detector-native Fit-OOS RCA，也不偷偷声称闭合
此前未运行的native协议。源接口维度/服务/通道/数值变换相同已由Luna只读核验。
旧RCA原有Test锁与更正evaluator结果只读复核，无需重复其已有推理。

模型/数据/源码不一致或重放失败即停止相关入口修复；保留INCOMPLETE目录，
用新目录重新开始故障部分，不替换已完成候选或修改冻结来源。全部比较完成后
按完整证据报告效果，不用Test选择“最终最优参数”。

Flat XGB最终产物在c0onset只有部分Git归档；本次引用c0rawmask中完整的同一
finalized-v1-seed42目录，其11项completion哈希均验证相同，trigger.ubj也与
c0flatxgb最初Fit权重SHA完全相同。原目录不补写、不改变模型或重做选择。

用户补充：三seed不稳定可能与训练不足有关。现有训练实际10/11/11epoch，
由merged-event patience早停。该解释是待验证假设；本次仍测试已有冻结权重。
固定训练预算/early-stopping对齐属于后续Train/Validation独立单变量方案，
不根据这次Test延长某个seed或替换checkpoint。
