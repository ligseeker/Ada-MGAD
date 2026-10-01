# P6 正常行为预测：固定 Fit 内筛查协议

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: controlled development plan
- Date: 2026-10-01
- Evidence: protocol only; prediction model has not run
- Worktree: `Ada-MGAD-e2e-v2-c0normal`, `experiment/p6-c0-normal-forecast`

## 假设与范围

只学习无登记事件的历史轨迹，预测残差是否能在冻结的60秒事件容差内发现异常，
包括现有分类器召回很低的memory/长事件。先检验一个固定候选，不搜索架构/分数/
阈值。这是正常样本监督的半监督方案，正常仅指无已登记故障，可能包含未标注异常。

该筛查只打开原Train数组与原Detector-Fit范围的服务/故障GT注释；与既有开发
loader一样，完整registry的区间字段仅用于冻结时间路由，并如实记录该扫描。
共享预处理仍由原70%
Train拟合，含本筛查留出段的无标签输入；因此不是独立确认或严格prefix-OOS。
不重建全量原始预处理；使用冻结填充序列，不能称为只对原始真实观测打分。

## 数据与时间

- 原Fit按时间60/20/20分为normal-training、normal-calibration和event-holdout。
  两个内部边界向下对齐冻结30秒网格，原Fit终点保持原定义。
- 每个窗口总足迹`[t-300s,t)`必须完整位于所属子块；不借用前一块的历史。
- 预测输入为前9个bin（270秒），预测目标为最后1个bin（30秒）；分数在t可用。
  模型从未消费目标bin，评分才将预测与该bin观测作差。
- 输入每服务45个数值槽及3个冻结aggregate mask，输出45个数值槽；10个服务
  共用预测器。保留数值槽的既有scaler和ffill，不预测mask，不重拟合scaler。
- normal窗口与所有原Fit已知合法事件的区间无交集；活动事件跨子块边界也排除。
  recent-onset NEG/IGNORE标签不是normal资格。点事件按1毫秒支持排除。
- event-holdout保留该块所有完整GT；purged窗口或因300秒历史不可用而漏检的
  事件照常进入FN。正常窗口支持不足不调整时间划分。

## 唯一候选与预算（分数生成前固定）

- 共享causal TCN，hidden32、kernel3、dilation1/2，每块两个卷积，左padding；
  每时刻channel LayerNorm、dropout0.2、最后一步linear输出45槽。
- Huber loss（delta=1）对正常training窗口所有45槽/10服务平均。
- AdamW，lr0.001、weight_decay0.0001；batch256、shuffle仅training、seed42。
- 固定20epoch，使用最后checkpoint；calibration不选择epoch。grad clip10，
  不加early stopping/scheduler/图/故障加权/额外监督。
- 首个筛查同时报告固定persistence参考（前一个bin预测最后bin），不训练或调参。
  参考检验学习预测是否提供超过简单延续的信号，不参与选择候选模型。

## 分数、阈值和事件协议

- 每服务45槽平均绝对残差，再对10服务取最大值。
- 模型和persistence各自取normal-calibration分数的99.5百分位，固定NumPy
  线性插值；相同`score >= threshold`和merged-episode decoder。
- 沿用30秒网格、60秒因果容差、最大匹配数/最小延迟的一对一匹配。
- 报告完整event-holdout P/R/F1、TP/FP/FN、单/多起点、fault/root样本数、
  memory/长事件、无登记故障窗口误报率、冻结aggregate observed fraction分层。
- normal-training/calibration/holdout各至少512窗才执行效果筛查；干净memory
  起点至少30才解释该稀有分组。memory支持不足不删除重叠事件或阻止整体指标。
  干净指事件首个已完成起点bin的300秒窗口，除自身外无其他已登记事件交集。

## 筛查GO与下一阶段

Fit内部筛查采用既有engineering门槛P>=0.90、R>=0.60、F1>=0.72；
并要求相对persistence F1提高至少0.02，normal-holdout阳性窗比例<=0.01。
这是预先固定的低成本screen，不等同于整个Validation的效果GO。

筛查通过后，才在原Fit正常窗口上拟合、以原Fit内部正常校准选择模型/阈值，
进入完整Validation比较；其细节必须另行锁定。Validation正式开发门槛仍是
相对修复后基线Recall+0.05、F1+0.02、P>=0.98、single-onset下降<=0.03。
进一步通过才seed/时间复核及68D+XGB完整E2E开发。不得依本筛查结果改变
阈值百分位、残差聚合、模型结构或事件容差。

TCN主线和其独立decoder实验完成且均NO-GO后才启动本预测器训练。
训练前可做本Fit-only计数/边界预检。若TCN已GO，先复核它，不并行启动新模型。
若本固定筛查NO-GO，则停止这一候选，记录局限并进入本轮Scientific Freeze评估。

## 绑定与完整性

输入数组/manifest/registry、协议/config、实现、参考运行和执行HEAD在开始前
锁定，运行结束后核验。新目录、失败保留、无Test入口。完整清单绑定checkpoint、
normal资格、子块cohort、各预测时间、完整GT与purged事件、分数、阈值和结果。
模型合成验证须覆盖目标bin不可见、时序因果性与batch同行独立；真实输入复验
须比较singleton/peer/batch logits或预测，绝对误差门槛1e-5。
