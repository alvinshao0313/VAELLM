# 显存与精度：分清驻留对象、数值存储和验证范围

## 问题与证据

单模块FP32参数存储对照中，两组仍为BF16计算；代表形状的allocated仅增加0.33MiB，reserved增加24MiB。该值不能推算整模型峰值，也不能据实现通过宣称精度提升。[精度记录](../records/docs/distill_fp32_validation.md)

历史mid-eval OOM计划指出两条风险：no-cache推理仍物化dense解码scratch；直接unwrap模型时连AMP/autocast一起剥离。[排查方案](../history/plans/2026-09-07-e2e-mid-eval-oom-memory-lifecycle-fix-cn.md)。这份来源是计划，不能由其标题认定所有修复当前已验收。

## 下次怎么做

- 分别记录参数、梯度、optimizer moments和计算dtype；加载时先恢复存储精度，避免先低精度舍入再升回FP32。
- 同时报告allocated、reserved和外部GPU观测，注明context、工作区、激活与同卡作业影响。
- 训练能跑不代表中间评测/导出能跑；审计teacher targets、解码cache、scratch及wrapper的完整生命周期。
- offload或sequence chunk优化需要数值与学生梯度对照，保持原损失归一化；“转到CPU”不自动代表GPU峰值降了。
- 多进程队列需要超时和子进程存活检查；最后汇总前所有rank应参与collective。
- 测试矩阵明确CPU、单GPU、单rank、真实多GPU；不得把单rank torchrun或同卡layer_mp当成多卡证明。

## 边界

现有单模块数值、历史实现审查和硬件冒烟各自只支持限定范围。设计新资源预算时检查当前可用GPU及共享任务，不继承旧日志中的卡号授权。[历史硬件范围](../history/audits/TRAINING_STACK_HARDWARE_SMOKE_AUDIT.md)
