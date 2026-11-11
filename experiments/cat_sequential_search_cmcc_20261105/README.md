# 七类别逐参数有限对照

本包只执行本次已确认的单BF16逐类别参数表：同类当前参数候选可并行，全部合法已声明候选完成完整八项后选优，再继承超参进入后项；不同类独立接续。每份新配置从共同原始BF16 fresh开始，不继续训练上轮压缩器。保留统一anchor及当前配方，按未舍入S8、较低logical bits、packed bits和candidate ID依次选取。没有KL筛除、旧7＋2槽图、分层转置、额外BSQ、回访、KD、重试或占卡。

真实cap须来自R1／R2完整联合比较的`actual_joint_full8_cap_certificate`，每类rate和cap锁定，不借额。anchor复用适用原21 profile中的实际同源bank／完整八项，保留`purpose=profile`和`role=reused_profile_anchor`。拟合身份仍为原source9；实际新增编排及现成night/batched/PUBLIC代码另由request SHA固定，不重拟合原bank。源码由活动任务pin后不得修改。

规则JSON由root绑定本次真实已确认顺序、预算和授权来源；未确认顺序及未就绪cap不能生成实际HPO集合。字段为`confirmed`、`authorization_reference`、`order`（八个参数名）、`maximum_distinct_per_category=19`、`extra_layerwise_or_BSQ_or_revisit=false`、`gpus`（只允许0–5）、`total_GPU_hours`、`formal_base_seconds`（D8:7200／D16:5400／其余3600）、`full8_seconds=900`、`technical_base_seconds=900`、`technical_large_seconds=1800`。确认顺序以文件为准，源码不凭旧表推定用户选择。

formal上限按D对应base乘`(encoder_width+decoder_width)/256`；必要新资源路径technical2上限通常900秒，D8或任一256宽度为1800秒；已有实际same_resource_path技术证据直接复用，倍率不重复短测。S2两stage各10000，`B×D=262144`、`K=rD/2`必须合法整数。保护继承请求比例，几何变化后重新取nearest-legal实际counts，因此旋转／D对照可能同时改变保护count，不能称纯单变量因果。保护1/32仅拟合前预测超cap才改全0；质量差不补第二臂。

调用者激活`bitvae`，隐藏CUDA，在tmux或nohup执行：

```bash
CUDA_VISIBLE_DEVICES='' python -B -m experiments.cat_sequential_search_cmcc_20261105.validate_cpu --output result/cat_sequential_search_cmcc_20261105/cpu_validation.json
CUDA_VISIBLE_DEVICES='' python -B -m experiments.cat_sequential_search_cmcc_20261105.run --component prepare --output artifacts/cat_sequential_search_cmcc_20261105/RUN --rules RULES.json --cap artifacts/cat_joint_round_two_cmcc_20261105/round_two_20261105/final_cap/cap_certificate.json
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.cat_sequential_search_cmcc_20261105.run --component run --output artifacts/cat_sequential_search_cmcc_20261105/RUN
```

这是本批固定候选的有限后台执行入口；无外部监控、定时或通用调度。每个候选目录有actual request／process／exit／worker log、completed、canonical result及full8原件。`search.json`为接续状态，`selected_settings.json`为总表；terminal后更新[同一结果Markdown](../../docs/experience/records/w25_transfer/2026-11-05_cat_selected_settings.md)，原21事实保留。只可在旧controller及其子任务都已结束后用`--resume`接续已完成边界；运行／半成品不重启，失败候选不重试。

实际失败封锁该类后项，同轮其它已声明候选及其它类可完成；非法／超cap保留未覆盖原因、不替补。完成formal后自动清理本候选已被替代的technical tensor／tokenizer冗余副本，先将精确路径／大小／原因记入`candidate.json`，保留技术配置／核心日志／fit／fresh receipts。正式产物与旧21 profiles不清理。总表逐stage列实际LR／κ、counts、D/K/S/B/T/rotation／宽度、完整S8、类logical／packed费用、cap及未覆盖；只称已评集合内入选，不能称全局最优或完整CAT／九项验收通过。

## 当前验证与接续边界

2026-11-05最终[CPU证据](../../result/cat_sequential_search_cmcc_20261105/cpu_validation_final.json)及[核心日志](../../result/cat_sequential_search_cmcc_20261105/cpu_validation_final.log)、[退出码](../../result/cat_sequential_search_cmcc_20261105/cpu_validation_final.exit)已落盘，exit0。证据内部计时2.183293581秒；`source_sha256`固定实际新编排／现成runner／coverage reader源码，`fit_code_identity.source_sha256`保持原9份拟合源码，不把编排纳入拟合身份重做原21份bank。验证涵盖真实profile小源重算S8／费用／UUID／有序输入／协议、实际两stage首步LR报告、五档D/K/B与几何合法count、LR同资源证据复用、保留当前点、失败封锁、NaN／Inf／bool／保护卡反例，以及逐槽292.25GPUh最坏预算。未重hash大权重或sample；CUDA未初始化。Astra已完成实现终审。

文档索引已生成；`docs.py check`仍有既有1654项缺链，新包及唯一设置表无报错，不能写全库通过。索引／检查全量临时日志已归并为本段摘要；初版CPU输出已由最终证据替代并清理。无新增算法经验。

**已确认顺序并后台启动：** 用户在`call_4b41f999c67645f28f9e833c9feef708`选择“按上述19点顺序执行（推荐）”，即`transpose→rotation→dimension→optimizer→multiplier→encoder→decoder→protection`。实际[rules JSON](../../artifacts/cat_sequential_search_cmcc_20261105/rules_confirmed_20261105.json) SHA为`cad34def5cc1de464074e8b09cb3b6c99b42b5c0dc444bd29bc6ce9d6337c4b4`；绑定GPU0–5与root经Astra评审收窄的292.25 GPUh上限，该数字不是用户逐字指定。另一项保护先提案不替换此次答复。

真实[cap certificate](../../artifacts/cat_joint_round_two_cmcc_20261105/round_two_20261105/final_cap/cap_certificate.json) SHA为`366bf5923d0e2bb5d5518105e33febdd58f22466a86f0facd9297fae065ccf4a`，赢家仍R1-B，S8=`64.32986503987618`，rate为q2／k3／v3／o3／gate2.5／up2／down2.5，sum cap=`18058995302`。没有重做cap的大state／full8审计。prepare实际exit0；[manifest](../../artifacts/cat_sequential_search_cmcc_20261105/confirmed_order_20261105_1049/manifest.json) SHA为`56195b0d45273ac782a5920eb05a052ffea810fd58f32a45b3617d5f9ddab30f`，identity为`a9bdd019f9bca0813f84a156bd2ad1231f18103ff8cffd3edd25d475c088e2f0`，保留source9及7份适用profile anchors。

2026-11-05 11:03已用tmux `Cat_seq19_formal_20261105`启动，CPU controller PID=`3252579`；[主状态](../../artifacts/cat_sequential_search_cmcc_20261105/confirmed_order_20261105_1049/search.json)及[核心日志](../../artifacts/cat_sequential_search_cmcc_20261105/confirmed_order_20261105_1049/controller.log)保留实际运行记录。首批T=True必要technical2占GPU0 q（worker3253255）／1 k（3253253）／2 v（3253254）／3 o（3253252）／4 gate（3253250）／5 up（3253251），down声明候选等待首卡释放。

**11:13有限启动核实已完成：** 上述六候选的实际`technical_validation/completed.json`全部completed／exit0，两stage各2次更新，独立GPU fresh的同输入BF16 logits exact=true；每个实际`formal_request.json`均为同原BF16 fresh、`technical_steps=null`、每stage10000，并绑定真实cap。正式worker PID为q3260594／k3259880／v3259471／o3261079／gate3262824／up3262314。k/v正式`formal/worker.log`已出现stage1 `steps=10000`及真实前两次更新：LR=.003／.0029997、optimizer实际step1／2及参数变化；这证明正式主循环推进，不表示完整拟合或八项完成。其它四个正式入口已启动，核实时仍在初始化／拟合准备，未将其PID当成主循环已推进的证据。

候选证据统一在本根`candidates/CANDIDATE_ID/`内；`search.json`绑定类别／ID、`candidate.json`绑定各phase request，`process.json`／`exit.json`／`completed.json`和原worker日志保留真实命令、PID、来源及状态。六份技术证据及k/v正式更新的核实结论已归并至本段；不重hash已通过cap的大state／full8、不追加GPU检查。新增关键geometry／optimizer资源路径仍按规则验证，正常完成后自动接本人fresh10K／full8及所属类后项；当前已停止主动监控，没有额外外部监控、定时或占卡，运行代码保持冻结。
