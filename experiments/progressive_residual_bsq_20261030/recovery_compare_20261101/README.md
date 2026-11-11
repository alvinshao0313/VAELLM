# 同一四阶bank的恢复位置比较

2026-11-01用户授权实施并启动两臂低成本比较；11月2日唤醒后实核两臂各56更新、fresh与完整八项全部完成，独立接续run／continuation均exit0。sequential MMLU70.79475858%、八项69.53589734%，terminal为70.73778664%／69.08149851%；本配方下逐类臂八项高0.45439883pp，MMLU仅多8题，不能据此认定一般恢复位置优劣或已恢复BF16性能。[完整两臂结果、真实UUID及边界](../../s4_recovery_resume_20261101/README.md#两臂56完成与本配置结论)为权威正文。原固定链首次评分前的`sample_artifacts`导入失败及1062源码pins保持不动；本批只使用GPU6／7，原S4基线 `b12014e7-f931-4502-86e3-bca93085eaf3` 完整保留，不重拟合bank。

两臂各56更新，七段各8更新，group顺序0..31,0..23。末端臂一直全S4；逐类臂从原CAT BF16 factory开始依次装q/k/v/o/gate/up/down的同一原S4 bank。两臂每段训练相同的累计prefix decoder集合，未来dense／未来decoder及所有码、保护、旋转、骨干均冻结；逐类安装不覆盖已恢复decoder。源recipe的`rot_llm=False`，只保留VAELinear内部two-sided旋转，工厂构造不会额外RMS融合或重复旋转；实际147非投影及future dense合同仍须实核。

AdamW LR1e-6、β=(.9,.95)、eps1e-8、WD0、foreach=False、joint clip1.5；使用既有500步线性前缀，56步不重warmup、不重置优化器。第j类首次激活时Adam状态为零，已有master／moments／step连续，最终各类更新数56/48/40/32/24/16/8。

原样复用成熟H96 `data.load()` 的train／auxiliary_train数据、顺序、输入和teacher缓存：M目标一次`.5F+.5gold`混合，QA为gold-prefix full-vocab教师正向KL，目标静态。此为答案级静态TF混合目标，不称纯教师或全tokenKD；学生父不同不使同一BF16教师缓存失效。按真实group计input与监督answer positions；每段8组不等于每段token相等。不加LoRA／OPD／改码／门控／其它目标臂。

必要真实短验每臂只做4更新：stage1的完整group2、stage2的group10，再装齐原bank并在全252 decoder训练范围下各完成group2／10，覆盖最长M958及QA10593。短验覆盖末点独立BF16导出、完整state fresh exact和真实前向；没有执行HFLM评分API，因此未发现本次新增调用者的裸模块导入缺口，不能把原基线HFLM数值证据等同新入口上下文已测。正式每臂从原parent新起，固定小train probe及终态完整MMLU B64／seven B32，不做LongBench或声称完整九项，不复制全池0/64/128读出。

实现分为来源／固定协议`bindings.py`、factory与激活`runtime.py`、连续Adam及损失`training.py`、独立CPU BF16导出`export.py`、实际consumer／评测`evaluate.py`；`run.py`与固定链入口保持薄。源码依赖只读复用，任一技术失败停止，不自动重试或研究；通过真实短验才从原parent后台启动正式，实核推进后停止主动监控。

## 当前实现与验证证据

原S4 source／1033pins及真实H96 `data.load()` 的输入／目标、官方split和原缓存身份已CPU核验，原目标digest `35f476e2f43c597e2bc1e1b85f603da36ebec3fd2e2f4469debf76c21c147b1b` 不变。每臂56组实际曝光为 **1,918,340非padding输入tokens／4,989监督位置**，M1792＋Qasper128＋Hotpot320次呈现；七段非padding输入分别339622／186517／339045／187972／339622／186517／339045，监督位置885／483／885／483／885／483／885。M非padding为636,909，M32 padded输入槽1,165,280；QA均B1合计1,281,431，因此含M padding的输入槽总数为 **2,446,711**，不将非padding曝光误称GPU实际输入槽或FLOPs。各组pad字段直接保留于原CPU proof；原始数据与许可绑定引用既有evidence，没有生成新数据或教师缓存，不把每段8组写成等token。

最初CPU数据证据位于 `/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/recovery_compare_20261101/pair_20261101_1158/CPU_validation.json`。随后只为合并显式bank冻结、short末态live数值观察及研究父标识更新四个本包源码；当前 `pair_20261101_1200/CPU_validation.json` 通过SHA继承原真实数据与曝光，仅重新compile／import四文件及核控制／来源差异，未重复全池校验或CPU全模型数值测试。1062个当前pins包含完整原1033个来源pins；两份CPU proof均 `cuda_initialized=false`，真实factory／冻结／反向／导出仍须GPU短验完成，不能由CPU准备推称已通过。

short末态live fixture保留真实`model.train()`、`torch.enable_grad()`及原packed trainable_decoder消费，核logits.requires_grad后仅detach最后token完整vocab，不追加backward。fresh复用同一fixture的BF16部署输出，记录relativeL2／maxAbs／top1差；这是FP32-master训练→BF16部署差量，不设无据exact门槛，不用no_grad路径冒充训练消费者。

## 实际后台入口

当前运行根目录 `R=/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/recovery_compare_20261101/pair_20261101_1200`，tmux **`S4_recovery_compare_20261101_1200`**，shell1228569，CPU driver **1228576/start24302844**，首个short terminal worker **1228910/start24303510**、物理GPU6／7；启动前两卡各1MiB且无其它compute进程。plan SHA `a9a9511142def827e3423b85dac3ac8275624c5a8c077103bbab054cdcd42923`。实际调用者已激活bitvae并核Python3.11.16：

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.progressive_residual_bsq_20261030.recovery_compare_20261101.sequence --plan /data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/recovery_compare_20261101/pair_20261101_1200/plan.json
```

`R/run.json`区分validating与formal_started，日志为`R/continuation.log`和每个阶段独立`*.log`，控制器／shell退出分别落`R/run.exit`／`R/continuation.exit`。每臂`updates/`记录真实loss／曝光／峰值；`segments/`只记录类别、global step、active数量、每类Adam有效step、耗时和输入／监督位置，不做逐段整模型保存或全MMLU。正式从原bank重新构造，不继承short四更新；只有两臂short训练及独立fresh均exit0才进入正式56。源码在启动后冻结，任一技术异常停止本固定链，不自动重试。

末端臂short训练／BF16导出已exit0，耗时670.936秒；独立fresh也exit0，3927持久字段exact，原码／保护／旋转／骨干与bank完全一致，位宽仍4.094031292459239bpw。stage7全252 decoder在group2／10各完成实际更新，耗时185.801／172.692秒；GPU6／7最大更新峰allocated24,174,365,184／23,922,742,784B、reserved25,901,924,352／25,650,266,112B，覆盖M958／QA10593。grad-enabled live→BF16部署单末tokenrelativeL2为0.0162719564673444、maxAbs0.375、top1差0、finite，作为不同数值路径的观测保留，不据此宣称逐位相等、质量无损或与单卡消费者完全相同。

逐类臂short训练／导出719.478秒、独立fresh均exit0：stage1／2为36S4＋216原factory dense、72S4＋180原dense，完整冻结补集／实际factory来源／非decoder字段exact，已有q master与Adam未被k bank覆盖；全252峰组2／10实际更新185.924／172.740秒，3927字段fresh exact，峰allocated与末端臂相同。其grad-enabled live→BF16单末tokenrelativeL2为0.014636887845404968、maxAbs0.375、top1差0、finite。固定小probe只读一个原train组，初末live解码状态随active集合变化，不将其差值单独解释为部署质量收益；终态质量只认正式56后的原HFLM MMLU／seven。

## 已用完短态载荷的最小收尾

删除前确认两臂short训练及fresh四阶段均exit0。`sequence.py`此后仅消费新formal目录；两臂formal构建、导出来源均是原b120 bank，未读取或继承short模型。无其它进程打开以下文件，且它们没有已安排的续训／复评用途，故仅删除这四个独立常规文件。短态是四更新机制证据，未选作交付成品；不保留重复整模型。精确路径以本页R为根：

| 本次已删除的精确文件 | 删除前字节 |
| --- | ---: |
| `short/terminal/final_model/pytorch_model.bin` | 6,046,045,569 |
| `short/terminal/final_model/cat_runtime_state.pt` | 22,729,058 |
| `short/sequential/final_model/pytorch_model.bin` | 6,046,045,569 |
| `short/sequential/final_model/cat_runtime_state.pt` | 22,729,058 |

共4文件，逻辑12,137,549,254B（11.303974GiB），删除前独占分配块12,137,570,304B。保留两臂config／checkpoint metadata、CPU及GPU证明、3927字段expected_state、真实fixture／live末token观测、更新／切换核证、核心日志与复现入口。没有生成optimizer checkpoint或新教师缓存，不存在相应删除。原S4完整父、共享目标缓存、当前formal及其后继读出模型全部保护；四文件已逐项确认删除，原b120完整目录全部inode／大小／mtime保持不变，保留证明与metadata可用；不用共享磁盘free变化估算本次释放。

## 正式比较已实际推进

四个short／fresh子阶段全部exit0后，正式terminal worker **1246046/start24450793** 已实际启动，`validation_only=false`、`planned_updates=56`、来源仍为原b120 bank，未继承short。实核第一更新 **stage1／global step1／group0**：LR1e-6，q类Adam有效step1，有限非零梯度，37.5166秒，weighted objective0.10382843902584682；GPU6／7更新峰allocated17,371,520,000／17,237,833,216B。当前1062源码pins收尾复核全部一致，原第一阶段1033pins未改；上述单组objective不与不同group的short loss直接排名。

原入口确认首更新后停止主动监控；14:28用户唤醒才核到下述真实终态。未安排LongBench、完整九项、额外恢复或门控，不根据旧PID存在或首更新推定整链完成。

## 末端56完成与首次评分导入失败

原 `R/run.json`／`run.exit`在13:43左右真实failed／1，保持原样。terminal已完成56更新、七段及预定各类Adam有效step56／48／40／32／24／16／8，训练／导出4625.804349秒，模型UUID **`b85c0689-0d16-40a5-84e4-3b4c6a595c81`**，`formal/terminal/fresh.json`为verified、3927字段exact。MMLU／seven两个消费者在全252预热后、实际评分前退出：`evaluate.py`用原文件加载`eval_utils_with_samples.py`，其裸导入`sample_artifacts`却缺SAMPLE目录上下文；原主S4 evaluation有对应路径设置，新增consumer遗漏。`mmlu.exit=1`／`seven.exit=1`、failure与log保留；尚未训练的sequential不能因terminal训练完成而称完成。

修复不改本目录任何Python或原1062pins，而是新包`experiments/s4_recovery_resume_20261101`显式设置worker `PYTHONPATH=SAMPLE:REPO:N_snapshot`。CPU以真实旧evaluate导入顺序成功导入原API及support，路径／SHA匹配原pins、CUDA未初始化；不重复数据池／teacher缓存／CPU模型或GPU短验。补评输出在新只读view，三项链接仅指原terminal report／fresh／模型，其评分结果与退出写新目录，不覆盖原失败。14:42两片均成功：完整57科／14042题MMLU和seven7叶／10700题、0shot B64／B32、同UUID及56更新。随后原b120→sequential56 worker1326290/start25271253首个实际更新已通过（37.542982秒），不是从terminal或short恢复。终态八项及下一臂实际入口、结果、限制统一见[接续记录](../../s4_recovery_resume_20261101/README.md)；原失败与末端56成品均保留，不以一臂小幅正差预判两臂胜负或九项收益。
