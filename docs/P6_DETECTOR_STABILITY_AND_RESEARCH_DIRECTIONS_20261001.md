# 检测阶段稳定性结果与后续研究方向

## Material Passport

- Date: 2026-10-01, Asia/Shanghai.
- State: completed TCN initialization replication and normal-forecast Fit screen; bounded study Scientific Freeze.
- Scope: detector development and conditional next directions; no new Test, RCA or full E2E result.
- Sources: sealed run JSON/CSV, completion manifests, configs and actual evaluator code.
- User preference: supervised and semi-supervised approaches are both allowed; retain methods by reproducible event and diagnosis effect.

## 1. 最新监督候选不能被保留为稳定提升

同一Validation完整GT为2,901，因果匹配容差0–60s，输入/窗口/IGNORE规则相同。
各seed使用自己的原merged checkpoint/threshold选择规则，随后冻结该阈值做
独立bin主分析；没有把seed42数值阈值转移给其他模型，也没有选择最佳seed。

| 方法 | TP/FP/FN | P | R | F1 | 决定 |
|---|---|---:|---:|---:|---|
| 逐窗口修正C0，recent60/merged | 2103/19/798 | .991046 | .724922 | .837348 | 修正基准 |
| E1 onset30/merged | 2057/131/844 | .940128 | .709066 | .808410 | NO-GO |
| TCN onset30/bin，seed42 | 2577/14/324 | .994597 | .888314 | .938456 | 初步GO被复制门槛否决 |
| 同一候选seed17 | 2605/414/296 | .862868 | .897966 | .880068 | Precision门槛失败 |
| 同一候选seed2026 | 2488/3388/413 | .423417 | .857635 | .566936 | Precision/F1门槛失败 |

三个seed都通过前后半段Recall增益门槛，但总体稳定性为`STABILITY_NO_GO`。
17/2026的预声明辅助bin阈值也失败，不可以用辅助结果救回候选。
seed17与2026全部锁定source hash map相同，model_args唯一差异为random_seed；
seed42共同依赖一致但runner/protocol包装不同，不能声称三份完整源码树仅差seed。
真实输入batch/peer独立性检查均通过；没有发现cohort、窗口时间或代码漂移。

封存来源：

- `../Ada-MGAD-e2e-v2-c0causal/experiments/p6/c0_window_causal/retrained-v1-seed42/`
- `../Ada-MGAD-e2e-v2-c0onset/experiments/p6/c0_onset_development/onset30-v1-seed42/`
- `../Ada-MGAD-e2e-v2-c0replica/experiments/p6/c0_onset_tcn_replication/stability-v1/`
- stability completion SHA256 `b09e308a4086386d3a4bf6ea0b52d7dcc47121a63209c5240005202fddb72a41`
- stability report SHA256 `ef5bd2fef8301f58703ff17040ed699b9da647ad1f2b030907c9984d886cabfe`

这些都是重复使用开发集的结果，不是独立确认。旧batch-conditioned结果不能用作
新在线检测器的性能基准；固定分数指标与在线可用性声明必须分开。

## 2. 目前最具体的失败机制

已观察：三个seed在onset30 POS时间格的命中接近，但NEG高分格分别为812、1162、
4009。seed2026的3,388个bin误报中3,333来自NEG、55来自IGNORE；其中3,290
是原recent60目标也为NEG的格。不能把主要误报归因于IGNORE或30s/60s目标错位。
NEG是监督标签，不能等同于完全健康、没有任何真实异常。

seed2026有5,876个阳性bin、2,135个连续高分段。其中一个3,321-bin、27.675小时
高分段包含3,309个bin FP和12个TP；merged只输出该段首点，成为一个false alarm。
因此merged FP45与bin FP3,388的差异主要反映输出容量及持续误报，不是新增
独立异常证据。每格报警、连续段报警各自都必须报告误报持续时间/占比。

待验证假设：按merged事件F1选择checkpoint可能偏好这种持续高分模型，而最终
bin输出直接暴露其负样本区分不足。辅助bin阈值重选只评价原selected checkpoint，
不能证明在所有epoch中换checkpoint准则有效，也不能证明没有效果。
随机初始化差异是观察到的稳定性问题；优化器、mask或某个模态是根因尚未证实。

## 3. 半监督筛查已完成：当前固定候选停止

唯一假设：正常轨迹的下一格预测残差能否提供超出Persistence的事件信号。
原Fit内部按60/20/20时间切分，分别正常训练、正常校准、事件留出；窗口完整位于
一个块中，保持300s历史清洗。输入270s，预测下一30s的45个Metric数值，10服务
共享小TCN，20固定epoch，最后checkpoint。故障标签只用于正常资格与事件评价，
不用于异常起点二分类或root拟合。

正常calibration的99.5%分位阈值固定，正常模型与Persistence各自校准；同一merged
decoder、完整GT与因果一对一匹配。GO须同时P≥.90、R≥.60、F1≥.72、相对
Persistence F1≥+.02、无登记事件holdout格阳性比例≤.01。仅是进入原Validation
开发的筛查门槛，并不等于胜过修正C0或达到最终E2E要求。

V2仅补充看到复制失败后的启动条件；模型/训练/校准参数与未运行的V1一致。
这是适应性分支启动，不伪称最初预注册。新run为
`../Ada-MGAD-e2e-v2-c0normalfallback/experiments/p6/normal_forecast_fit_screen/stability-fallback-v2-seed42/`，
execution commit `dcdaeb9937dbfebe9e9f31ffa572c8c2705496e1`。20epoch全部完成，
`COMPLETE_DEVELOPMENT_ONLY`，决定`NO_GO_FIT_SCREEN`。

| Fit event holdout方法 | TP/FP/FN | P | R | F1 | 无登记事件格阳性 |
|---|---|---:|---:|---:|---:|
| 正常TCN预测残差 | 18/64/1518 | .219512 | .011719 | .022250 | 8/1184=.006757 |
| Persistence残差 | 52/99/1484 | .344371 | .033854 | .061648 | 16/1184=.013514 |

同一完整GT1,536，不与第1节Validation的2,901混为一个cohort。
预测器F1比Persistence低.039398；仅正常阳性比例门槛通过，其余四项均失败。
正常cal1,077格，forecast threshold=.19044798612594574，
Persistence=.08064676195383065；两者均精确复算为正常cal的线性99.5%分位。

forecast failure为MATCHED18、NO_LEGAL_PREDICTION2、BELOW_THRESHOLD1512、
NO_NEW_EPISODE2、MATCHING_COMPETITION2，完整分母闭合。
1,386案在固定响应观察范围内从未越阈值，115案首个越线在60–300s，10案在300s后，
22案在60s内、3案删失。这定位当前固定残差/阈值缺少及时响应；不能证明
分数反向、异常本来就不可观测或某个新阈值一定有效。解码器无法凭空生成
未越线事件，不把TCN监督候选的decoder机制直接套到本预测器。

只读复核：21个completion输出和84个source/input哈希全部一致；prediction lock
绑定scope、final checkpoint及两个score CSV；实际batch/peer误差均0；
round-trip CSV精确重放两种阈值、episodes、匹配TP/FP/FN及P/R/F1一致；
全部1,536个ledger case身份与GT一致且唯一，正常holdout资格和8/16阳性计数一致。
默认CSV浮点解析可改变约1e-17末位，复核使用`float_precision='round_trip'`。
未发现实现错误；不据此断言所有半监督模型都无效。

- completion SHA256 `62cde9f6d015e29a9d4409390e2b0d2294a3e9a5b15003626b4a7447febdec2f`
- result SHA256 `875add960e90c4aa4c06970b0668413748eedba672f53d5ca55a45973a8d87e6`

停止这个固定候选，不改分位、aggregation、offset或挑seed；不进入原Validation
推理或新的RCA/E2E。共享70%Train预处理、filled-mask、正常资格依赖登记GT、
单seed及clean-memory只有13的限制保留。正常格全落在filled-observation最高层，
当前产物不能支持低原始观测率分层结论。

## 4. 下一轮优先方向一：监督checkpoint选择与报警单位一致

这是下一个独立研究方案，尚未授权执行的新候选。若用户决定开启下一轮，
仅将原merged事件checkpoint/threshold选择准则改为最终bin事件准则；保持
onset30 target、IGNORE、TCN/input、训练超参和预算、GT/cohort、0–60s匹配不变。
高分持续时间、NEG高分率、每日报警数是必报诊断，不在同一实验加入新的loss、
class weight、mask或模型模块。训练只在Fit，选择只在Validation，Test保持关闭。

预先固定seed42/17/2026，不找最佳seed。开发GO可沿用原同一基准的全部门槛：
每个seed P≥.98、Recall增益≥.05、F1增益≥.02、single-onset Recall下降≤.03；
两个固定时间半段Recall增益分别≥.03。失败停止该候选。
这仍然是复用Validation开发验证；有效后需要新未用时间段/数据确认。

## 5. 下一轮方向二：输入契约修复；RCA保持固定

- raw-observability mask修复已准备，当前filled-mask不能证明原始观测存在。
  这是输入契约修复，不承诺提升分数；全量预处理仍只有手动方案。
  若未来重新构建输入，另开独立实验，仅改变mask/输入版本后重建公平基准；
  不与checkpoint准则实验同时改变。Trace parent在时间点可用性仍需认证。
  具体沿用[P6_RAWOBS_MASK_MANUAL_PLAN.md](P6_RAWOBS_MASK_MANUAL_PLAN.md)：
  Train-array gate须确认45数值槽/Log/Trace/host applicability精确不变，仅原始
  global/host observed fraction降低；identity或45槽支持失败即停止，不补槽。
  效果筛查沿用原逐窗口C0及相同选择规则，Validation F1≥基准+.005、Recall
  不下降、P≥.90；通过仅代表值得进一步复制，尚不能替代第4节的高精度稳定性
  或完整诊断门槛。单组少数case改善不判GO。语义修复是否正确与分数是否提高分开。
- 30s聚合、同格多注入及多事件重叠会限制起点分辨率；无任何模型可保证恢复
  聚合后不可识别的每个独立注入。保留完整GT与失败分母，不能放宽容差制造收益。
- 稀有memory/long故障存在可用变化的有限证据，但root/fault和持续时间混杂；
  当前正常预测Fit holdout的clean-memory仅13，低于30解释门槛。
  没有足够证据直接宣布解决稀有故障，也不再盲目加class weight或模块。
- RCA固定为用户当前论文的68D+XGB。检测候选稳定后，用新逐窗口Fit-OOS锚点
  拟合一个固定scorer，同时评分基准与候选的Validation锚点。分别报告Stage-1、
  matched RCA、完整E2E P/R/F1@1/3/5、matched与未matched failure分母。
  当前未执行该阶段，不能声称检测Recall增益已经成为诊断增益。

## 6. 本轮停止规则

当前TCN候选NO-GO，完整E2E入口关闭。正常预测是本轮最后一个固定候选，
其Fit筛查也NO-GO。本轮进入Scientific Freeze，保留修正基准与全部失败证据。
上面的监督selector方向及新的raw输入研究是下一轮可审批的假设方案，
不是本轮失败后自动扩展搜索；无需为了保留一个漂亮seed继续刷同一开发集。
