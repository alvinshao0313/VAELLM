# PUBLIC完整八项与离线结果接线

本包独立接通原BF16／PUBLIC native-v6模型的完整八项入口、原BF16分层receipt、实际worker→profile/offline结果适配。它不修改`cat_auto_protocol_cmcc_20261104`、PUBLIC核心、N snapshot、up／LR包或其活动pins，不实现S0–S13持久executor。初次接线只执行CPU核证，后续正式profile的真实GPU证据见下文。

`protocol.py`将共同任务／版本／metric／执行条件与候选UUID／权重SHA／文件路径分开。原BF16参考通过迁移receipt的repository／revision／全部五分片SHA，以及原官方byte receipt绑定模型／tokenizer metadata；仅同config不够。原两份完整结果的57科MMLU＋七项、64叶／24742题、B64／B32、0-shot／no_limit、实际seed与BF16／FA2条件复用既有验收；不伪造v6 UUID。历史完整BF16分数保持`quality_reference_verified=true`，原逐题未序列化，故`ordered_pairing_verified=false`及明确`unavailable`；不为补日志重评GPU。历史recorded Transformers／git／execution与当前installed／新serializer证据分栏保存。

`results.py`从实际completed、生产者绑定的registry及其SHA重算candidate／parent／model／data／consumer身份，核真实S2×10000、semantic stage seeds、冻结补集、512 pooled top100＋outside-mass KL／T1／prompt.3、独立完整state及BF16 logits fresh。profile保留完整N与future dense账本，offline费用取实际类别的`logical_bits/packed_bits`。逐投影费用从真实packed bank的storage／byte spans消除别名并归属，与整个类别及component费用精确闭合；真正跨投影重叠共享storage没有分摊合同则显式`impl_blocked`，不猜份额。formal透传并绑定原`BSQ_diagnostics`两个stage1 5000／10000实际batch；technical1／2明确无该诊断，不能静默改变commit／entropy条件队列。技术或CPU产物不能进入正式排名。

registry有两种真实绑定方式：新profile wrapper的`process.json.request/request_sha256`可直接使用；其它生产者显式传实际`--request`与`--request-sha256`。不猜上级目录、不伪造`request.json`副本；最终还核`completed.request_identity`。

`evaluate.py`将正式raw交给真实PUBLIC v6 strict loader与`tools.cat_eval`的group8预热，保留实际BF16参数／FP32 RoPE／FA2（Qwen）及原HFLM likelihood window。实际HFLM／`simple_evaluate`使用MMLU64、其余32、0-shot／no_limit、原seed；直接复用成熟sample writer和metric-row helper，保留全部真实samples及SHA。两个分片、任务configs／versions、实际HFLM window／BOS／seed及endpoint绑定全部有效才产生summary。BF16独立加载实际原模型，使用content reference ID，始终没有伪造checkpoint UUID。

`read_native_eight`重读实际summary／分片／逐题文件并调用既有`nodes.load_complete_eight_score`，返原8个0–1指标与百分制均分。`results.attach_full8`只接该真实reader的路径输出；`freeze_order`在原selector之前核跨类别及BF16共同协议／原模型tokenizer／共同512教师合同。S13保留原BF16summary级参考，不宣称与其逐题exact；新候选之间可依据真实有序输入身份配对。

此入口限profile/offline的raw。恢复selected-node完整导出可能另有`cat_auto_selected_node.source_node_id`，需要其真实接续绑定；本轮没有承诺或伪造该全路径。后续[up三档正式profile完整八项](../../artifacts/cat_profile_full8_cmcc_20261104/proposal_20261104_1837/README.md)已完成真实8B PUBLIC loader／FA2／group8／完整八项及canonical关联，补齐正式profile reader的GPU调用正例；这些结果的purpose仍为profile，不是offline调参槽全流程正例。

## 实际CPU证据

权威目录为`result/qwen3_8b_mmlu65_20261004/cat_offline_execution_cmcc_20261104/current/`：

- `bf16_quality_receipt.json`：实际历史BF16完整八项、来源／协议和证据层级；无新推理。
- `native_reader_cpu.json`：20.1385秒、CUDA未初始化。实际PUBLIC pilot的36个投影storage费用精确闭合、512 KL／fresh适配及technical资格拒绝；真实tiny Qwen3 v6重建／strict load／group8四个native预热、BF16重复forward／FP32 RoPE、实际HFLM window／BOS构造通过。
- `bindings_cpu.json`：23.82635秒、exit0／CUDA未初始化，包含当前BF16全部15份source SHA、真实现成完整64叶原始sample reader、缺片拒绝、真实profile外置registry path／SHA、同config错weights拒绝、缺formal BSQ诊断拒绝、实际child环境去除继承CPU模式及最新tiny的错UUID／RoPE拒绝；70份最终source pins复核一致。

CPU未调用完整八项evaluator计算新分数。完整sample reader核证用现成原up控制的真实文件，只证明sample读取边界，不冒称PUBLIC runtime或新的BF16逐题证据。两次临时CPU测试中的save返回对象字段误读、逻辑／真实路径比较错误均只影响测试器，修正后重验；失败核心日志保留。所有tiny模型位于`TemporaryDirectory`，结束自动清理，无复制8B模型。

已删除被`current/`完整替代的5份旧成功receipt／报告／日志，释放allocated155648B；7份当前／失败证据hash未变，精确路径见同根`cleanup_cpu.json`。该CPU批次未包含正式10K worker正向mapper及其两条真实BSQ诊断；后续[三份正式profile终态](../cat_batch_profiles_cmcc_20261104/README.md)已补齐实际读取与核证。技术2步证据没有被升级成该证明，原CPU历史范围保持。

## 使用实际收据

调用者先激活`bitvae`，在项目根目录用tmux/nohup后台执行；以下大写路径和ID须替换为实际已绑定对象。现有正式profile不依赖本包完成，也不因此获得新评测预算。

```bash
CUDA_VISIBLE_DEVICES='' python -B -m experiments.cat_offline_execution_cmcc_20261104.evaluate \
  --component baseline --inventory ACTUAL_INVENTORY_RECEIPT --output NEW_BF16_RECEIPT

CUDA_VISIBLE_DEVICES='' python -B -m experiments.cat_offline_execution_cmcc_20261104.results \
  --completed ACTUAL_FORMAL_COMPLETED --candidate-id ACTUAL_ID --output NEW_ADAPTED_RESULT

CUDA_VISIBLE_DEVICES='' python -B -m experiments.cat_offline_execution_cmcc_20261104.evaluate \
  --component prepare --completed ACTUAL_FORMAL_COMPLETED --candidate-id ACTUAL_ID \
  --baseline ACTUAL_BF16_RECEIPT --output NEW_EVALUATION_DIRECTORY
```

不带authorization可做CPU准备，`run/shard`会拒绝启动GPU。经后续讨论确认的授权文件须含真实`authorization_reference,physical_GPU,authorized_GPU_hours,authorized_start_epoch,absolute_deadline_epoch`；准备plan时用`--authorization ACTUAL_CONFIRMED_BUDGET`冻结。两分片共享同一绝对deadline、kill-after留20秒、独立session/PG及finally完整清理，无自动retry。child显式`ACCELERATE_USE_CPU=false`，allocator为`expandable_segments:True`。GPU执行入口为`evaluate --component run --plan ACTUAL_PLAN`；BF16新评测仅在另获实际用途／预算时用`prepare-bf16`准备，不自动替换或重评已认可参考。

实际完整八项结束后将原件交回结果adapter：

```bash
CUDA_VISIBLE_DEVICES='' python -B -m experiments.cat_offline_execution_cmcc_20261104.results \
  --completed ACTUAL_FORMAL_COMPLETED --candidate-id ACTUAL_ID \
  --summary ACTUAL_FULL8_SUMMARY --evaluation ACTUAL_FULL8_EVALUATION --output NEW_SCORED_RESULT
```

新运行必须新目录，保留原始指标及来源，完成／失败按[离线执行规范](../../docs/experience/records/w25_transfer/2026-11-04_cat_offline_execution.md)处理；不将worker已完成重命名为整个离线搜索已完成。

## v0.2完整八项选择层

最新SC-CAT协议要求**profile／offline／live／replay固定S2、原槽16空置、每类最多19个实际端点**，槽号仍为1–20。本次实际收紧的是本包可调用的profile／offline／joint来源入口；旧core的live／replay统一执行器尚未接通，本次未核其实际运行路径，不重新分配空槽。此前16.889秒及5.743秒报告属于当时20端点／S4尚在库的源码快照；其真实历史八项来源证据继续复用，不能据此宣称最新规则已通过。

`selection.py`独立消费已冻结的真实候选／joint提案，不调用原`offline.rank_joint/raw_full8_coverage`或`proposals.select_joint/category_caps`的旧KL／≤10逻辑，也不改这些仍被冻结的文件。它没有S0–S13持久槽表、候选生成、调度或新GPU授权；既有runner不会自动转用它。

`freeze_block`输入实际`inventory_receipt,baseline_path,run_identity,category_id,stage,class_cap_bits,expected`；每条expected明确给`slot,candidate_id,recipe,worker={completed,request:{path,sha256}},full8={summary,evaluation}`。候选必须出现在其真实formal offline request中，cap必须绑定同一model／原BF16父／inventory／共同协议及完整joint八项证书。1–19个不同S2 recipe／ID／槽按稳定槽号冻结，明确拒绝slot16；同一端点复用不额外占槽。各前段调用者须完整传入本次决策已宣布合法的累计集合，不能先完成者先选；本层只能证明显式集合闭合，不能证明外部执行器没有漏登记候选或已经完成整个20槽编号框架。

`select_block`逐端重读`worker_result`与`read_native_eight`，核原request→cap、真实fit／bank／endpoint、原始八项／全部样本、同父／数据／代码／consumer／semantic seed合同、共同512教师及有序八项输入。费用使用真实类别logical／packed值，并受原Bc约束；不接受调用方填入的full8均分或verified标志。缺fit／评测、失败依赖返回`pending`且没有赢家；错UUID／来源／协议、technical产物、旧KL-only cap或实际超cap明确拒绝。全部齐备后按未舍入`(-S8,class_logical_bits,class_packed_bits,candidate_id)`返回排名与原件path／SHA证书。

Python接口另提供`retained_geometries`、`capacity_decision`、`interaction_basis`、`close_category`与`freeze_order`。容量只在S8严格提高，或S8完全相同且(logical,packed)费用字典序降低时采纳；ID／KL不改变容量同分决定。容量／slot20接口缺显式冻结queue时返回`unconfigured_conditional_queue`。已明确配置的queue须有本base／父／protocol及实际确认来源，最多三条指令；每步用既有resolver重解完整S2 recipe及κ owner派生合同，核候选只将该指令作用于当前已接受配置，不能把独立同base点当叠加链。BSQ只应入段冻结规则，到槽再读取当前accepted基底诊断并生成方向；阈值／方向仍未审，故本轮**任何含BSQ的queue均返回`unconfigured_BSQ_generation`**，即使带opaque rule引用也不宣称已配置。当前accepted candidate ID／诊断digest的绑定纯函数已验，但未将其包装成已完成的BSQ生成器，不提前冻结未来诊断或系数。无容量采纳时可明确返回另一保留几何上的optimizer／倍率交互role，但新recipe仍由调用者显式提供并另做合法性／去重／cap检查。类别关闭保留anchor和最佳不同于anchor者；跨类冻结保留已确认的`(d_c,同512教师KL,稳定类别ID)`，不将单类下降相加。

`freeze_joint`输入同一实际context及全部显式提案；每份含`proposal_id,rate_vector,banks,full8`，每类bank含实际`candidate_id,recipe,bank_digest,worker`。现协议已确认profile完整八项及joint两轮各≤4／合≤8；本函数仍只接受外部显式rate向量，不生成4＋4提名或预算规则，持久executor及相应阶段前置约束仍须接通。`select_joint`对全部提案的实际整模八项全齐设屏障，逐bank核部署state的全部键／tensor digest及拓扑，核norm／embed／head等非目标补集等于同BF16原件，并重测完整N账本与所有类别费用。赢家按实际整模`(-S8,logical_bits,packed_bits,稳定rate向量)`选择，证书保留所有被比较提案的完整原件来源。未给显式已确认`cap_allocation={authorization_reference,categories:{category:cap_bits}}`时返回`budget_allocation_pending`，不自动套旧reserve规则；给定后仍须真实赢家各类费用≤Bc、ΣBc≤floor(2.6N)才形成可消费cap证书。

CLI只负责冻结或读取现成对象，JSON字段就是上述函数实参；输出路径必须尚不存在：

```bash
CUDA_VISIBLE_DEVICES='' python -B -m experiments.cat_offline_execution_cmcc_20261104.selection \
  --component freeze-block --input ACTUAL_BLOCK_INPUT --output NEW_FROZEN_BLOCK
CUDA_VISIBLE_DEVICES='' python -B -m experiments.cat_offline_execution_cmcc_20261104.selection \
  --component select-block --input ACTUAL_FROZEN_BLOCK --output NEW_SELECTION_CERTIFICATE
```

joint对应`freeze-joint/select-joint`。`freeze_order`等Python接口同样只读实际已冻结来源，不会自动拟合、评测或补下一槽。

本轮CPU证据位于`result/qwen3_8b_mmlu65_20261004/cat_offline_execution_cmcc_20261104/s8_selection_20261104_1717/`。完整入口为`CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cat_offline_execution_cmcc_20261104.validate_selection_cpu --output NEW_REPORT`，由已激活bitvae的tmux调用者执行。`selection_cpu.json`为16.889316秒／exit0，CUDA未初始化：复用既有真实mixed reader，重读同after-gate父的控制67.6733108398258、Muon67.50944587170605、D1666.59370953118375全部64叶／24742条、原样本与共同条件，拒绝缺片／错UUID／错来源hash；候选权重已清理，只证明保留证据与排序机制，不升级为新PUBLIC formal正例。原控制full8 wrapper仍是exit1，按已保留的真实CPU闭合记录核两分片成功及完整summary，没有改写历史退出状态。旧mixed/N reader在独立CPU子进程中运行，PUBLIC来源验证留在父进程，避免混用consumer family。

原快照纯CPU规则还覆盖未舍入排序、容量同分费用规则、anchor入围、真实resolver及owner派生字段、独立同base点不能伪称叠加、真实profile registry／technical资格拒绝及foreign parent／protocol来源反例。request→cap导入边界在`source_cpu.json`中单独补验。最新固定S2／≤19／slot16空置／BSQ当前accepted来源仅补受影响CPU规则，复用原历史报告，不重复64叶样本读取。初次测试器的旧服务器BASE、历史控制wrapper状态误判及同进程consumer污染均已修正，均未改变原指标、reader或正式配置。

最新[sc_cat_current_cpu.json](../../result/qwen3_8b_mmlu65_20261004/cat_offline_execution_cmcc_20261104/s8_selection_20261104_1717/sc_cat_current_cpu.json)为5.837507963秒／exit0、CUDAfalse，绑定原16.889秒完整历史报告及当前全部source pins；验证19端点允许、slot16单点／20端点／重复拒绝、真实S4候选拒绝、当前accepted诊断ID／digest绑定，以及opaque rule／预先field、value和诊断齐备时BSQ queue仍保持unconfigured。两组被此最终结果替代的SC临时报表／日志已归并清理，不保留过期可配置状态。

**已验证的PUBLIC正式profile reader：** [up三档正式profile完整八项](../../artifacts/cat_profile_full8_cmcc_20261104/proposal_20261104_1837/README.md)已通过真实formal raw、PUBLIC加载、两分片64叶／24742题及`attach_full8`关联；[七类K32来源汇总](../../artifacts/cat_night_profiles_cmcc_20261104/night_20261104_1918/k32_summary_20261104.json)进一步保留同BF16／DataPlan／source9、各自独立单类profile的正式拟合及完整八项证据。这些正例可直接复用，不再以“reader正例缺失”为由重复启动补验；它们不等于真实offline调参端点、offline槽选择全流程或七类组合成品。

**尚未闭合的范围：** [显式producer](../cat_joint_export_cmcc_20261104/README.md)已有必要真实8B安装／导出／fresh／实际账本的技术正例；真实offline调参槽全流程、联合完整八项／cap证书、原20槽编号的持久执行及跨模型整链仍缺。SC-CAT逐参数表正在另线程讨论，当前范围与确认状态以[主协议](../../docs/experience/records/w25_transfer/2026-11-04_cat_auto_protocol.md)为准；本节旧槽图／CPU快照不代表最新版逐参数表，也不构成新任务授权。现有profile结果不扩大原协议或替代这些未完成环节。
