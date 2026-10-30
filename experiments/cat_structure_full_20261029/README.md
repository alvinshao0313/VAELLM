# 完整36gate结构对照与各自live MLP后继

10月30日已核完整终态：A九项63.52216414161746、B63.78399082894941，两支实际逻辑位宽均通过，严格质量目标均未达。按预设完整九项规则选择B继续自适应对照；完整分项、成本、身份和预算见[夜间完整结果](../overnight_goal_20261029/README.md#10月30日真实终态与完整结果)。下文10月29日在途描述是当时记录，不代表目前仍运行或缺少评测。

A最终成品不再安排续训／复评，已依持续清理授权移除其模型权重、runtime及tokenizer副本8个文件，实际释放4,709,863,424 B；config／metadata、原始预测、完整指标、核心日志和复现入口保留。精确路径／SHA及保留文件复核见[终态审计的cleanup字段](../../result/qwen3_8b_mmlu65_20261004/overnight_goal_20261029/terminal_audit_20261030.json)。A final_model现为证据目录，不可直接加载；B、共同after_o及当前活动依赖未动。

同一真实父模型`/data/star/VAELLM/qwen_recompression_20261026/C/initial_after_o`，UUID`b63bfe79-3889-40c6-bf66-1d002893f874`。原144个attention及108个live dense MLP、原CAT校准缓存／统计和恢复历史全部保留；不使用[三层筛选](../cat_structure_probe_20261029/README.md)端点或另一支up/down donor。

| 分支 | gate D/K/S | gate encoder/decoder | gate每层向量batch | 后继 |
| --- | --- | --- | --- | --- |
| A | 32/32/2 | asymmetric E128/D128，R0 | 8192 | 本支live up D32/K32/S2，再本支live down D32/K48/S2 |
| B | 64/64/2 | asymmetric E128/D128，R0 | 4096 | 同A后继结构，各自live权重，batch均8192 |

每类全36层一起原CAT拟合，保留M36 pooled BSQ entropy；各stage10000更新。类别batch通过共同标量曝光`D×batch=262144`实际传入原拟合函数，明确写入隔离runtime identity和运行记录。基础seed31，沿原C派生函数得到gate400031/up500031/down600031，A/B一致；三层probe显式shuffle31属于不同分组与映射条件，不能称复现其轨迹。

只有gate结构改变；up/down仍为原C symmetric H128/R0。每类结束均用原N快照的500步`remaining_lora_prefix_decoder`恢复，B4／1024、decoder LR1e-5、未来dense rank12 LR1e-4并原生融合，KL top100／T1／prompt.3、原五源train-only mix、保存250／500及其它条件保持。最终无外置持久LoRA。asymmetric不属于现whole-fused资格，不因尺寸合法就声称使用该核。

真实旧入口为`cat_recompression/fork_boundary.py`及`run_cat.py`，原生主体是`qwen_native_category_rotation/source_snapshot/train_utils/cat_train_pipeline.py`，配方源为`formal_k38_seq500/launch.json`。本目录只使用隔离进程中的明确调用适配：未来gate结构受控分叉、类别batch合同、实际保存记录和B必要短验；原公共源码／v6严格身份比较与旧实验pins不改。新fork不是旧未来结构的exact resume，也不恢复不存在的全局RNG。

先在调用者激活`bitvae`、项目根目录执行CPU准备：

```bash
CUDA_VISIBLE_DEVICES='' python -B experiments/cat_structure_full_20261029/prepare.py
```

唯一B短验从它自己的fresh after_o开始，全M36／D64/K64真实形状，两stage各2次VAE更新，再由原500-step日程进行真实2次恢复更新。短验保留全180个压缩prefix与72个未来dense rank12、B4/1024；callback观察真实输入、梯度、连续Adam及FP32更新，在step2停止。短验原生边界显式标为validation-only，不得作为正式父模型。原生CPU fresh strict核证通过后，两正式分支仍从各自未改的after_o、完整10K／500日程重新开始，无短验callback或更新残留。

```bash
cd /home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/qwen_native_category_rotation/source_snapshot
CUDA_VISIBLE_DEVICES=4 python -B -u /home/star/VAELLM/experiments/cat_structure_full_20261029/run.py --arm B --phase short2 --gpu 4
CUDA_VISIBLE_DEVICES='' python -B /home/star/VAELLM/experiments/cat_structure_full_20261029/verify_short.py
bash /home/star/VAELLM/experiments/cat_structure_full_20261029/run_A.sh
bash /home/star/VAELLM/experiments/cat_structure_full_20261029/run_B.sh
```

GPU执行使用上述原snapshot cwd，短验调用者沿`run_B.sh`相同环境变量；各次均由调用者tmux／nohup后台运行、stdout/stderr落盘，启动前检查指定GPU空闲。输出根为`/data/star/VAELLM/cat_structure_full_20261029`：`prepared.json`保存真实fork／配置／完整state／源码绑定，`B/short2.json`和`short2_validation.json`保存必要验证；`{A,B}/run.json`记录PID/start_ticks、实际native argv、每类生效batch及真实保存UUID。拒绝覆盖，不自动重试或等待其它任务。

完整模型logical≤2.6按最终实物核账，packed/padding另列。B约2.49873153仅为此前几何估算，当前没有完整新成品或九项结论。完整九项由root待两分支完成后统一调度，不按八项提前筛掉LongBench；本入口不运行额外评测。

## 10月29日：验证完成，两条正式已实际推进

真实配置与负例检查10项通过；CPU结构估算A logical2.497367527／packed2.510742188，B logical2.498731530／packed2.512106191，明确不是最终实物账本。CPU双fork耗时624.824秒，2127个tensor／13,884,036,096字节与父模型完整SHA `6d086ea76fe2af6e16d55ea3267ed07621fd0a2ffca3162eb67a63822dab21d3`一致，B独立fresh strict通过。A新UUID为`fd803862-95b6-4930-a0ba-3b9a7899e2cc`，B为`a9af7fab-1435-4391-8eb6-f155868c5e26`。

首次B短验真实完成全36gate两stage编码后，原恢复首个student forward被自有`shape==[4,1024]`验证hook误拒；未完成优化更新，原失败`B/short2.json`／`short2.log`保留。原正式配置启用dynamic padding，不要求每批恰好1024。CPU真实dataset／原collator回放和随后GPU实测都得到首两批`4×456`／`4×1024`，有效token分别`[454,27,152,317]`／`[70,524,296,1024]`，input／mask SHA也一致；没有强制padding、改变采样或降低batch／max长度。

缺失的反向补验只读原真实round_base UUID`dd46f947-5ca0-41ac-b4f1-29af4821f574`，使用`--short-round-base`显式新验证参数。原online CAT不接受直接round_base，因此这里直接调用同一个`run_after_category_distill`及原生saver，真实fresh0→2、不伪造Trainer step0／optimizer状态，不重做M36拟合／双fork。`prepared_initial.json`保留初始完整CPU证据；当前`prepared.json`记录验证逻辑修订及新源码SHA，差分保存在`B/short2_validation_fix.patch`。公共／旧实验源码不变。

补验exit0、75.604秒，保留原500日程的两更新、180个prefix codec和72个未来dense rank12。1080个decoder叶及144个LoRA叶均为FP32，真实梯度有限且Adam连续step1／2；warmup使第一步LR为0，第二步gate decoder有1,211,818个FP32元素实际变化。峰值CUDA allocated40,762,400,768B／reserved41,060,139,008B。原生保存UUID`5d8d62c5-5120-4f83-8648-fe8689e51f0a`，2559个tensor／10,737,410,304字节；独立CPU fresh strict全state SHA `88e9b843eb826de792883839a83208434d1ec1a2910dbf4456672556acc1093a`与保存前相等，CUDA未初始化。`short2_validation.json`已verified，明确validation-only、actual2／max500及不得作正式父模型。

随后在原N snapshot cwd、调用者已激活的bitvae环境，启动前实核GPU3／4均0MiB且无compute，两支各自fresh正式从上述after_o分叉重跑，未继承任何短验更新或callback：

| 正式分支 | GPU／tmux | worker PID／start_ticks | 最后一次有限确认 |
| --- | --- | --- | --- |
| A | 3／`cat_structure_A_full_GPU3_20261029` | 425649／9254904 | 13:44:19，全36gate stage1已1500／10000，D32／batch8192 |
| B | 4／`cat_structure_B_full_GPU4_20261029` | 425653／9254905 | 13:44:20，全36gate stage1已1400／10000，D64／batch4096 |

唯一当前结果根为`/data/star/VAELLM/cat_structure_full_20261029`。正式命令与源UUID见`{A,B}/run.json`，核心日志`{A,B}/run.log`，自然结束后外层才写`run.exit`；本次交付时运行中，不写成完成。验证及清理归`validation.json`，短验核心日志为`B/short2_recovery.log`。正式进程依赖的源码、配置、两个初始fork均保持；已结束且无后续消费者的两份短验生成权重／重复tokenizer等25个文件已收尾，按各目标实际allocated计共释放21,554,307,072B。原失败及成功指标／metadata／日志／配置／差分保留，短验端点不再作可加载交付模型；删除后受保护记录SHA及两个正式初态稳定stat均核对不变。

在确认上述真实推进后停止主动监控，不等后续低频保存／恢复或完整结束，不自动评测。当前没有A/B完整低比特成品或九项质量结论；同源结构比较只是总流程研究的一步，不能证明整个HF复现／跨模型迁移目标完成。

## 候选：在恢复上限内选择节点，再进入下一类别

2026-10-29用户提出按固定间隔评完整八项、用较好节点进入下一类。**可以不总用末步，但当前A/B代码没有自动八项回调或最优节点选择。** 现配置为`save_steps=250`、原500步日程、`eval_tasks=''`、`eval_every=0`；原CAT Trainer安装的是训练日志／token统计回调，保存钩子只写实际mutable D＋LoRA与训练状态。已读gate250的`trainer_state.json`只有训练日志，`global_step=250/max_steps=500`，`best_metric/best_model_checkpoint=None`。其它E2E路径有评测回调，不代表本CAT路径已接入。本候选不修改、暂停或替换在途A/B固定500与其它已排任务。

只读核查时，两臂gate的250／500原训练节点、round_base及已融合`after_gate_proj`均存在。原相对路径为`{A,B}/native/home_star_models_Qwen_Qwen3-8B_20261029_134231/training_rounds/0004_gate_proj/trainer_state/checkpoint-{250,500}`；原`save_total_limit=2`不仅影响单Trainer节点，还会在类别边界提交后删除最旧完整round，可能连gate250／500一起删掉。root已安排并完成最小保全：`/data/star/VAELLM/cat_structure_full_20261029/selection_candidates/{A,B}/gate{250,500}`，四份合计290121871B，6.3016秒／exit0；原mutable、meta、trainer_state逐字节保留，1224个mutable（1080 prefix decoder＋144未来dense LoRA）及独立fresh读取exact。唯一保全报告为`selection_candidates/report.json`；没有复制optimizer/RNG或大round_base，原活动产物未动。它只是保全，不等于已导出／融合或已评八项。

每个节点必须还原**全部180个已压prefix decoder和72个未来dense的LoRA**，不是仅替换36个gate decoder。重用已保全raw A/B gate＋共同after_o重建原round_base，再按该节点真实`mutable_state_manifest`及`immutable_resume_contract`重建PEFT／FP32参数并strict恢复，调用原N的decoder finalize、dense LoRA merge和BF16 export；只读导出到新目录，不构造Trainer、不作optimizer step。生成新UUID并保留父raw UUID、训练节点UUID、actual250/500、原planned500、runtime／数据／loss／LoRA范围。500优先复用已有真实`after_gate_proj`作为控制，必要导出一致性检查复用同一500节点即可，不重复评相同成品。CPU/GPU BF16 merge的正常差异单列，不用补偿伪造逐位相同。

已有成熟先例为`result/qwen3_8b_mmlu65_20261004/qwen_broad_stage_screen_20261017/export_cat_step.py`及后续gate／up读出；旧入口硬绑定旧v类别／108decoder／144LoRA／2000日程，不能原样调用。只复用其`load_v6_training_model_state`、`build_model_level_trainable_selection`、`restore_model_level_mutable_state`、原生融合及strict保存链，范围全部从当前节点meta派生。归档中原`round_base_ref`作为历史来源保持；新解析器显式从已核证归档＋after_o取得同一基底，不改活动文件，也不制造缺失父模型／伪续训。完整八项评的是这一已融合、可独立加载的模型；此时仍有72个dense投影，不能当成最终≤2.6bit成品。

**最小配对先限定一个类别、两个既存节点。** 在事前指定的一条结构分支上，只比较gate250与gate500，保持原完整八项协议（64叶／24742题、原0-shot／batch等），按未舍入八项等权均分选高者，完全同分固定选500，不加MMLU或其它单任务门槛。另一条结构的节点已保全，可按原A/B研究安排再用，不据观察到的250成绩临时换来源。这个首轮只增加两次完整八项、无需重训gate；每25步评满500则为20次完整八项，是当前两点的10倍评测次数，未有证据前不铺该密度。

若500胜或同分，直接保留已运行固定500控制，不重复其后缀；若250胜，才决定匹配后缀预算：**250与500两个完整融合边界都按同一预先固定的fresh后缀重seed规则**运行各自live up/down压缩与恢复，比较最终九项。原不停机500后缀只作既有参考，不能直接充当单变量公平控制；不能拼接其up/down来冒称250的后继。第一轮后继恢复仍用既定500，先隔离“gate选点”是否有最终价值，有收益后才逐类推广，不同时铺类别／频率网格。当前只读250／500完整八项不依赖后缀决定，不因此推迟导出或另起训练。

原恢复优化轨迹保持500日程，250必须来自该轨迹，不能把`max_steps`改250重跑不同cosine。选中后不是恢复该类step250的optimizer：先完成真实250的部署融合和新category-boundary，后缀另按匹配规则启动。N源码仅在`run_cat_train`入口调用`set_seed(cat_args.seed)`；up的500031只传给采样generator，`MultiLayerVAE`初始化仍消费全局RNG。因此新进程重seed31的250后缀不自动等价于原不停机500后缀。未来两源在训练前共同冻结初始化seed及重设位置，保持原up/down采样seed500031/600031、恢复stage seed36/37、数据／预算和新阶段optimizer，并记录实际初始化身份；这是匹配fresh后缀，不宣称原连续轨迹exact-resume。

checkpoint500的`rng_state.pth`位于Trainer保存点，尚未经过stable finalize/save；后面的decoder spec抽取会创建模块并消费随机流，所以不能把它直接当作next-up的初始化RNG。root已明确本候选不补存RNG或optimizer、不展开连续恢复路线；若250确实胜出，再安排上述双源同规则后缀。评测／导出在独立进程执行，不改变活跃master／RNG；“最好节点”仅指上限内预定候选的八项最好，不新增提前停止启发式。后续频率仍待两点实际结果后再定。

八项选点不保证最终九项或后继最佳。[已有经验](../../docs/experience/lessons/experiment_design.md)包括：原fresh CAT同轨迹down1000→2000八项增加0.731968pp，固定H50却下降1.208612pp；另一次H50下降0.251955而完整LongBench21增加0.240974，说明代理之间也会反转。旧0920搜索确实出现中途2500优于5000，支持保留中途候选，但不能据此预判当前250会优于500。最终仍完整八项＋LongBench21／4750的九项、同协议BF16严格掉点<0.9pp及全部存储验收，不用H50或八项独立门槛替换。

按正式八项反复选点属于明确的**开发反馈／测试驱动选点**，不是盲验；记录候选数、节点、实际全部分数和选择规则，不把测试逐题结果转成训练样本。以后可以研究固定train／auxiliary_train留出代理先提名以省成本，但需单独约定并校准，不能静默替代用户本次要求的完整八项。当前仅完成源码／来源核查、方案和小产物保全；没有新增八项、节点优胜或新后继训练结论。
