# 冻结候选 Test RCA 与完整 E2E 比较

## Material Passport

- 2026-10-01登记，2026-10-02继续执行；用户授权所有已有较好候选的冻结Test观察，Luna探索各树真实产物。
- 独立工作树experiment/p6-frozen-rca-test-review，旧run和共享输入只读。
- 范围：19 detector臂 × 原XGB/Train固定25.621s backdate = 38组transfer；
  no_metric/no_all_magnitude两个已在Train两次前推正向的消融各固定最终拟合/Test。
- 全部是复用Test的描述性观察，保留旧NO-GO；不据Test改seed、epoch、阈值或参数。

## 冻结顺序

1. detector先产生全部输入型scores与19臂episodes/global lock；本树配置在Git中
   独立捕获该lock和10个模型completion的SHA。此时只登记无标签预测receipt，
   不读取新的Test GT/matching；detector评价移到全部RCA排名封存后。
2. 提交本树源码/配置，绑定旧模型、旧Train/Test cache及全部provenance锁。
   raw manifest逐项校验3,788个引用数组；不重新生成预处理、schema、scaler。
3. 两消融只在旧共同Train 3,225case拟合一次，group=10，固定200trees/depth3/
   learning_rate.05/seed20260826/n_jobs1/hist。分别删除Metric17列（51D），
   四通道的magnitude/post_mean/delta_mean列（56D），不zero-fill或调参。
4. 为19臂两种anchor独立检查±300s完整落在Test块。按exact anchor、同raw-index/
   同numeric schema复用旧cache，六个真实anchor逐值重放；新anchor并行抽取
   W300-B15 canonical10×68D，无GT/root输入。非法context保留空ranking。
5. 原/backdate固定权重评分38组，两消融在旧C0episode评分；40组完整排名先锁。
   evaluator须提供独立捕获的prediction-lock SHA才可读取matching/GT标签。
6. 40组排名封存后，独立detector guard核验commit里的预测receipt和全部model
   completion，再运行detector evaluator并逐臂核验matching整数。RCA evaluator
   在读GT/matching前核验独立捕获的audit.json SHA及其中的evaluation manifest
   SHA。各臂ranking预测ID集合须与matching非miss集合完全相同。
   complete GT 5787；完整E2E TP为匹配且root在Top-k，FP=全部预测−TP，
   FN=5787−TP。错误root同时计FP/FN，边界/缺排名不删除。matched RCA单独报
   legal matched分母，backdate-vs-original另报共同合法配对交集。输出全部
   AC@1/3/5、MRR、full E2E@1/3/5、root/fault分层与逐案例failure ledger。
   两消融复核旧XGB完整ranking和3093/4189/4194整数后计算paired improvement。

## 来源与限制

transfer是既有RCA scorer迁移到新detector锚点，不是新detector-native Fit-OOS。
旧cohort/模型来源仍为batch-conditioned supervision-OOS，不能升级在线因果主张。
RCA±300s需要后文；共享70%Train preprocessing包括detector Val，Trace parent
可用性未认证，GAIA30s粒度/同格多事件/数据偏斜保留。两scorer的context mask
允许不同，matched指标不可直接忽视分母差异；全GT与全预测口径始终不变。
