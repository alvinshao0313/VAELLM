# 原C：同旧after_o的gate encoder宽度对照

本支闭合[原四臂筛选](../cat_structure_probe_20261029/README.md)中保留未决的C：gate D32/K32/S2、asymmetric E256/D128/R0；与原A的E128/D128相比只改gate encoder宽度。父为真实`b63bfe79-3889-40c6-bf66-1d002893f874`，保留其144个已压attention、108个live dense MLP、512×256校准缓存及全部统计。消费原native v6，不转换compact，不使用新C0或Rice，不借用三层probe端点或其它分支后缀。

原A完整控制直接复用：UUID`26a03a51-efb4-41ec-abc1-a329a01c9fea`，九项63.52216414161746，MMLU61.123771542515314，logical2.497367527173913／packed2.5107421875。真实来源为[原终态审计](../../result/qwen3_8b_mmlu65_20261004/overnight_goal_20261029/terminal_audit_20261030.json)及其A原始结果；A权重已依收尾规则删除，不重训或重评控制。

三类gate/up/down分别原M36 pooled BSQ拟合、两stage各10000更新，每层向量batch8192，每更新262144权重标量；各拟合本支当时live权重，再各500步原`remaining_lora_prefix_decoder`恢复。seed31，shuffle400031／500031／600031；保留原native连续RNG，不在类别边界额外重seed。encoder容量改变可影响后续随机数消费，此为完整配方比较，不宣称后缀optimizer初值逐张量匹配。恢复沿原B4/max1024、dynamic padding、FP32 decoder和仅未来dense rank12、LR1e-5／1e-4、KL top100/T1/prompt.3、原五源train混合、保存250/500与原500日程。encoder不部署；完整位宽由真实最终native ledger核算。

## 独立实现与真实证据

- `bindings.py`只读绑定实际迁入的父metadata/runtime/weights、真实`cat_structure_full_20261029/prepared.json`中的原A argv／runtime／父状态摘要及终态控制。旧绝对路径只在新运行对象中用既有映射解析，原件不改。
- `prepare.py`在CPU严格读取真实2127字段父模型，核原状态摘要，允许未来gate E256及与A相同的asymmetric/D128设置，保存新UUID独立fork并在新进程full-state strict重载。公共exact-resume仍拒绝未fork的改宽身份。
- `run.py`仅包装原N fitter／recover／saver，记录实际拟合、optimizer初始化、Trainer真实500终态和原生保存；不替换数学。新的E256必要短验只做全36gate两stage各2更新、真实编码／round_base保存及独立CPU strict。原D32/D128恢复、保存后回训证据复用，不重复B短验。正式从未更新的after_o fork fresh起步，不继承短验权重。
- `source.py`只读核正式来源、252投影／3423字段／504码库、rank0、真实logical≤2.6及独立strict，不重配全局评测dispatch。`evaluate.py`／`evaluation_bindings.py`由主线程实现，以原native消费者完成一份full8＋LongBench21/4750；质量not_met合法完成，summarize真实返回值传播。

模型根：`/data1/cmcc/VAELLM/artifacts/cat_structure_C_cmcc_20261101`。计划、核心日志、CPU／短验核证和评测根：[独立结果目录](../../result/qwen3_8b_mmlu65_20261004/cat_structure_C_cmcc_20261101/)。仅GPU0训练、GPU1本支后继完整九项；不使用GPU4/5/6/7，不等待资源、重试或占卡，不覆盖旧结果。

## 固定调用

调用者先激活`bitvae`；命令在项目根目录、tmux/nohup后台执行。

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -B -u -m experiments.cat_structure_C_cmcc_20261101.prepare

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -B -u -m experiments.cat_structure_C_cmcc_20261101.sequence \
  --support /data1/cmcc/VAELLM/artifacts/adaptive_class_cmcc_20261030/evaluation_support
```

sequence使用flock保护一次启动，短验→独立strict→fresh正式→独立strict/实际native审计→GPU1完整九项逐项真实exit0后才接依赖。日志`{short,formal}_train.log`、`{short,formal}_fresh.log`及评测日志分别落盘；`run.json`记录实际argv、GPU、PID/start、阶段exit，不把启动或短验当成完整效果。

CPU真实父／独立fork的2127字段完整摘要均为`6d086ea76fe2af6e16d55ea3267ed07621fd0a2ffca3162eb67a63822dab21d3`，新fork UUID为`e3c93f6a-84a8-4b53-9820-be878e67e5c6`。独立strict结果在`fork_fresh.json`，CUDA未初始化。最初prepare在全部模型检查后，因并行加入已审查的short-only观测导致末尾旧源码SHA检查退出1；原`prepare.log`／`prepare.exit`和`prepared_cpu_initial.json`保持。只补最终代码绑定的`pin_completion.json`／`.exit`实际为0，继承源／fork／fresh原SHA，不重读模型；补齐两份observer依赖后971项pins稳定，原MLP992项pins全部未变。正式生效`prepared.json` SHA为`c0db0808ac83009454727c789d39fe0239657cc5954813ebe2067498052e0729`。主线程`evaluation_cpu.json`实际核证五组件CLI、拒绝未完成源和pure reader后dispatch保持，CUDA未初始化。

固定链在tmux `CAT_C_E256_native_fixed_20261101`，controller1510948；GPU0新形状短验112.70秒exit0，全M36两stage各2次真实更新、有限非零梯度及连续Adam均过，实际编码／保存完成，峰allocated14,411,400,704B、reserved14,887,682,048B。独立2559字段strict重载exact也已exit0，正式从未更新fork fresh启动worker1515668。正式训练27067.58秒（7.52小时），同原10K×2／三类500恢复完成；旧native A的27146.50秒仅为旧机参考，不作严格同机速度比较。

## 完整九项结果与保留用途

2026-11-02 11:11固定链全部完成，`run.json`为`completed_C_native_and_full9`，short／formal train与fresh及evaluate均exit0；`evaluation/evaluation.json`为`complete_full8_full21_current_goal`。同一最终UUID`02f0b8c3-405a-4e0a-bf09-1fff9496bf1f`完成完整64叶／24742题（MMLU batch64，其余七项batch32）与LongBench v1 21任务／4750题，均0-shot、无limit。原始证据：[运行链](../../result/qwen3_8b_mmlu65_20261004/cat_structure_C_cmcc_20261101/run.json)、[评测绑定](../../result/qwen3_8b_mmlu65_20261004/cat_structure_C_cmcc_20261101/evaluation/evaluation.json)、[完整九项验收](../../result/qwen3_8b_mmlu65_20261004/cat_structure_C_cmcc_20261101/evaluation/longbench/merged/nine_task_acceptance.json)。

| 指标 | 原A E128/D128 | 原C E256/D128 | C相对A |
| --- | --- | --- | --- |
| 九项等权均分 | 63.52216414161746 | 64.10220094639317 | +0.58003680477571pp |
| MMLU % | 61.123771542515314 | 61.40150975644495 | +0.277738213929636pp |
| LongBench21宏均分 | 43.91590364990137 | 42.50035489854355 | -1.41554875135782pp |
| 投影logical bpw | 2.497367527173913 | 2.497367527173913 | 0 |
| 投影packed bpw | 2.5107421875 | 2.5107421875 | 0 |

logical包含主码、decoder、保护值／scale／索引及旋转，packed另计padding；不把投影账本当整模型实存。encoder未部署，最终位宽与A相同。C九项比同协议BF16低3.9449439715153103pp，质量`not_met`；不是技术失败，也未达最终验收。该完整结构配方有净收益，支持保留encoder256候选；它只改变gate encoder，并包含自身live后缀与RNG消费变化，不把收益单独归因于encoder容量机制，不把单seed结论推广到其它D/K或模型。LongBench下降亦须保留。经验归入[同码率结构的完整结果](../../docs/experience/lessons/compression_choices.md#当前先闭合同码率结构对照)。

本链结项，不重复训练或评测。最终模型保留作D256完整结果闭合后的有界配方取舍及恢复迁移来源；配置、核心日志、原始指标和源SHA保留。短验权重已按`short_cleanup.json`清理，本次不删除C最终权重或其具体后续依赖。
