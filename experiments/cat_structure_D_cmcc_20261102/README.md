# 原D：同源gate decoder256完整分支

本支补齐[原四臂筛选](../cat_structure_probe_20261029/README.md)的D：gate D32/K32/S2、asymmetric E128/D256/R0，相对原A只改gate decoder128→256。父为真实`b63bfe79-3889-40c6-bf66-1d002893f874`，保留其144个已压attention、108个live dense MLP、512×256校准缓存及252项统计；不消费三层probe端点。复用[原C独立包](../cat_structure_C_cmcc_20261101/README.md)的成熟迁机映射和闭合N namespace，不修改活动源码，不转compact、不借用donor或新C0/Rice。

原A完整控制复用UUID`26a03a51-efb4-41ec-abc1-a329a01c9fea`、九项63.52216414161746、MMLU61.123771542515314、logical2.497367527173913／packed2.5107421875，来源为[原终态审计](../../result/qwen3_8b_mmlu65_20261004/overnight_goal_20261029/terminal_audit_20261030.json)及其实际验收文件，不重训控制。

gate/up/down各自按本支当时live权重进行全M36 pooled BSQ拟合，B8192、S2各10000更新，再各500步原`remaining_lora_prefix_decoder`恢复；up/down仍原symmetric H128/R0。seed31、shuffle400031／500031／600031、原native连续RNG，不在类别边界额外reseed。恢复保持原B4/max1024 dynamic padding、FP32 decoder与未来dense rank12、LR1e-5／1e-4、KL top100/T1/prompt.3、原五源train混合和原500日程。不同decoder容量影响后续随机消费，不称后缀初值逐张量匹配；最终无外置LoRA。

## 最小验证与停止条件

旧D256证据只覆盖M3局部筛选和编码，缺原B4/max1024模型级恢复反向。本次一次真实短链覆盖全36gate、S2各2次拟合、真实编码、原500日程前2次恢复和category_boundary原生保存，随后独立CPU full-state strict重载。观测真实packed D256非零有限梯度、连续FP32 Adam及实际参数变化；保持180个已压prefix和72个未来dense rank12、真实batch与最大长度。短验仅证明实现，不用于比较质量；正式从未训练after_o fork fresh重新开始，不继承短验更新。

完整训练预算12小时，超时或技术失败停止依赖，不自动重试。A旧机7.54h和C本机7.52h仅成本参考，D256可能更慢；不据局部loss延长预算或判断收益。正式完成后独立strict、真实252投影／3423字段／504码库、rank0、logical≤2.6及packed／全存储审计，再在自身GPU0完成full8（MMLU64／seven32）及LongBench21／4750／32K。沿同BF16、九项严格掉点<0.9pp及≤2.6验收；质量`not_met`是有效结果，技术／协议失败停止后继。

模型根：`/data1/cmcc/VAELLM/artifacts/cat_structure_D_cmcc_20261102`。原始配置、命令、PID/start、阶段退出、日志及评测：[结果目录](../../result/qwen3_8b_mmlu65_20261004/cat_structure_D_cmcc_20261102/)。仅用GPU0训练与后继评测，不占其它GPU、不等卡、不建监控或自动重试。

## 固定调用与状态

调用者激活`bitvae`后，在项目根的tmux/nohup中运行：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -B -u -m experiments.cat_structure_D_cmcc_20261102.prepare

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -B -u -m experiments.cat_structure_D_cmcc_20261102.sequence \
  --support /data1/cmcc/VAELLM/artifacts/adaptive_class_cmcc_20261030/evaluation_support
```

CPU实际配方／原N namespace／评测组件parser与pure source dispatch验证已exit0，Astra/ultra核心复核通过；独立真实父fork及2127字段CPU full-state strict也已exit0，用时515.47秒，摘要为`6d086ea76fe2af6e16d55ea3267ed07621fd0a2ffca3162eb67a63822dab21d3`。实现／来源文件SHA绑定在`prepared.json`，不伪造commit ID。

固定链tmux为`CAT_D_D256_native_fixed_20261102`，controller1604872。GPU0必要短验已exit0，用时190.80秒，M36两stage各2次真实更新及编码完成；原500日程前2次恢复覆盖`[4,456]`和`[4,1024]`，1224个FP32训练叶Adam1/2连续，第二次更新实际改变1,237,090个gate decoder元素。峰allocated42,871,720,960B／reserved46,468,694,016B，原生category_boundary UUID`809ca19f-8bd3-4398-8c98-45550807a8de`。独立CPU strict已exit0，2559字段全state摘要`9cdcd08236d2ff591379440a5b85e16966886d412817bf5a97b85a7b36464925` exact，CUDA未初始化；实际命令、退出和完整观测见`short_validation.json`、`short/report.json`、`short/strict_reload.json`。

正式从未更新fork UUID`46746ad7-3a05-42ad-8286-5c0b44f74766` fresh执行，prepared SHA为`1bc24db44c610cb2db05d7212b51b46aee688ffc376fd3eebf7ed40b74537d89`。11月3日原件核证：[run.json](../../result/qwen3_8b_mmlu65_20261004/cat_structure_D_cmcc_20261102/run.json)为`completed_D_native_and_full9`，正式训练、独立strict和完整评测均exit0，controller及各阶段记录PID均已退出。训练30401.34秒（8.445h）、独立strict241.05秒、完整评测20264.89秒（5.629h），原10K／500日程未变。成品UUID为`badcd04f-028d-4428-a65c-6a18e7a092d5`，3423字段full-state strict exact，真实252投影／504码库／R0审计与完整九项均已闭合。

原父保留，独立fork已完成本固定链用途；正式生成权重与副本按下节收尾。已核短验独立退出、formal只读取未训练fork，清理`short/native`的22个无后续用途文件，实际释放21,556,146,176B（约20.076GiB）。短验报告、配置、完整state摘要、strict/proof、核心日志及正式依赖均保留，精确路径／原因／删除核验见`cleanup.json`。完整结果及取舍见下节；参考[容量与整模型证据边界](../../docs/experience/lessons/compression_choices.md)。

## 完整结果与证据边界

[九项验收原件](../../result/qwen3_8b_mmlu65_20261004/cat_structure_D_cmcc_20261102/evaluation/longbench/merged/nine_task_acceptance.json)为`not_met`：技术流程已完成，logical≤2.6预算通过，BF16九项掉点4.3205117010pp，未满足严格小于0.9pp的质量目标。评测外层exit0；内层`acceptance_exit_code=1`表示这个质量结果，不能写成程序失败或验收通过。

| 指标 | 原A完整控制 | 本D256 | 相对A差值 |
| --- | ---: | ---: | ---: |
| 九项 | 63.5221641416 | 63.7266332169 | +0.2044690753pp |
| MMLU | 61.1237715425 | 62.3913972369 | +1.2676256943pp |
| LongBench21 task macro | 43.9159036499 | 42.8690534561 | −1.0468501938pp |
| logical bpw | 2.4973675272 | 2.4987899117 | +0.0014223845 |
| packed bpw | 2.5107421875 | 2.5121645720 | +0.0014223845 |

完整九项相对A的其余代价为BoolQ +1.1009174312、RTE −0.7220216606、WinoGrande +1.1049723757、ARC-E +0.3367003367、ARC-C +1.1092150171、OBQA −1.8、PIQA +0.4896626768pp。MMLU实核8761/14042题；full8为64叶／24742条，LongBench为21任务／4750条／32K，全部原预测SHA、数量及scores/config SHA与验收记录相符。full8、LongBench、native存储及pipeline/strict均绑定上述同一成品UUID；九项验收SHA256为`f90a46e3ca96dd4961eaabbceadb55d5be24ac1e0ed7a98ed8584ea99f65eb6d`。

位宽为实际checkpoint tensor storage核算，含code、BF16 decoder、protected indices/scales/weights与rotation，packed另含码流padding；全模型tensor为4671049728B，embedding/head和norm在账本中另列，不能把linear bpw当全模型字节压缩率。该结果只覆盖本Qwen／原after_o父／本预算的完整native流程，不能据单次结果归因容量或宣布跨模型迁移有效；最终原始BF16整链及跨系列验收仍未完成。本次未新跑GPU或改变验收口径。

本批决定：D固定结构完整批次结束，不追加预算；相对原A的九项和M正差保留为本配方事实，但完整九项未超过原B／C，不选作默认后恢复基座。不能把该结果写成所有decoder增容无效，也不能因九项排序称C在每项能力上支配D。

收尾：确认本支无已安排后继、E256的1007源码pins及6／7线程实际plan/request不引用本支权重，选中文件无活动FD/mapping后，已删除本支独立fork、各category／round-base／final生成权重、runtime、optimizer/scheduler/RNG与重复tokenizer共72文件，释放54509019136B（allocated）。[精确清理receipt](../../result/qwen3_8b_mmlu65_20261004/c1_m_target_muon_cmcc_20261102/revision_group_fix_20261102/cleanup_terminal_20261103.json)逐路径记录原因、大小和核验。配置/meta、源码绑定、核心日志、原始指标／预测及strict摘要保留；原始after_o未动，历史run/acceptance不改，成品路径现仅保留证据，不再代表可加载权重。
