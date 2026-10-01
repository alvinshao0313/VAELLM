# 2026-09-24 E2E decoder 联合训练的数值发散排查

## 后续正式搜索证据（2026-10-01）

以下保留2026-09-24各次诊断的当时状态，不将早期“尚无完成对照”解释为当前状态。同一0920模型的[普通rank8 LoRA长程搜索](../../e2e_0920/2026-09-24_rank8_search.md)采用decoder冻结、residual none及head linear，已完成全部六个坐标，其中12组到5000步；最佳step2500模型独立严格重载后八任务均分68.024489%，未达到69。该搜索相对早期诊断还改变了head等条件，因此提供可运行配方和真实下游证据，不是仅冻结decoder的单变量因果试验；此前未定位的失稳根因仍保留不确定性。本次只整理已产生结果，没有改动或重启训练。

## 低 decoder LR 与重新启用 replace 的新证据（同日18:26）

`165835`（decoder_lora、residual none、decoder_lr1e-5）已记录至230步，跨过旧失稳区间：120/130/140/150步窗口loss分别为0.4443/0.4352/0.4545/0.4506，裁剪前grad_norm为14.74/14.22/16.22/18.29；230步loss0.4407、hidden0.03068、pre-MLP0.02696、grad_norm15.70。原日志关键行为448、450、452、454、470。它与`155415`的有效配置只差decoder_lr从3e-5降至1e-5，因此对“原decoder更新强度参与早期失稳”提供了直接对照支持；这是单seed、230步观察，不证明完整训练或其他协议稳定。未取得该运行的退出状态，不标记正常完成。

用户新启动的`181434`为 `train_mode=lora`，但同时把`residual_lora_mode`从none改回replace，FP32清单去掉decoder。相比`165835`，它同时冻结decoder并改变残差结构，不能作为“仅冻结decoder”的对照。`decoder_lr=1e-5`虽然仍在参数快照中，在decoder冻结时不控制任何decoder更新；残差模块由独立aux开关安装并训练，`train_mode=lora`不会将它禁用，其LR仍随learning_rate=1e-4。

`181434`第10步loss32.5001、grad_norm106007；第70步loss19.1444；第80步loss17.1924、hidden11.0989、pre-MLP1.0867、grad_norm309.33（原日志425/437/439行）。截至80步是高起点后下降，与之前replace组的初始劣化一致；不能据此说已经重现none组约130步的突发发散，也不能断言损失公式有误。该初始checkpoint不含残差模块，重新安装replace再次用rank8投影替代72处恒等skip。

若目的是隔离decoder训练，受控配置应为 `train_mode=lora` 且 `residual_lora_mode=none`，其余沿用对照；若有意研究replace，则应把它作为改变初始模型结构的独立实验。这里仅解释配置与建议，没有改动脚本、停止现有进程或启动新实验。

## 后续对照与结论修订（同日 17 时）

**新证据修订了上一轮首要嫌疑：replace 破坏初始函数，但不是约120–130步共同发散的必要条件。** 新的 additive 与 none 两组只有残差模式不同，前110步总 loss 均约0.41–0.49，随后都在约120步先出现梯度异常，130步 hidden/pre-MLP 误差爆涨。不能继续把残差 LoRA 或其 BF16 参数当作 none 组发散的原因。

五组来自同一0920 checkpoint，配置以各自 `normalized_e2e_runtime_args.json` 为准：

| run_id 后缀（均为 `Qwen_Qwen3-8B_20260924_`） | 残差模式 | loss_type | decoder 峰值 LR | 已观察结果 |
| --- | --- | --- | ---: | --- |
| `102602` | replace | kd_top_partial | 3e-5 | step10 loss31.91，已记录至550步，严重发散 |
| `132618` | replace | kl_top_partial | 3e-5 | step10 loss31.04，已记录至290步，严重发散 |
| `150311` | additive | kd_top_partial | 3e-5 | step10 loss0.4611，120步梯度113.62，130步发散；日志至150步 |
| `155415` | none | kd_top_partial | 3e-5 | step10 loss0.4620，120步梯度2167.82，130步发散；日志至190步 |
| `165835` | none | kd_top_partial | 1e-5 | 用户新启动；17:03:47记录step10 loss0.4628、grad_norm9.58，尚不能判断是否解决失稳 |

`102602→150311` 仅改 replace→additive；`150311→155415` 仅改 additive→none；`155415→165835` 仅改 decoder LR。这里的“仅改”指有效参数快照，不宣称已证明所有运行软件/随机状态逐位相同。前四组日志没有结束标志，不将其标记正常完成。

最新已失稳的 none 组关键时序（总 loss 是窗口均值，分项是 rank0 最后一批）：

| step | 总 loss | hidden loss | pre-MLP loss | 裁剪前 grad_norm | decoder 当步 LR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 110 | 0.4786 | 0.06381 | 0.05698 | 19.78 | 2.18e-5 |
| 120 | 0.4925 | 0.03566 | 0.02692 | 2167.82 | 2.38e-5 |
| 130 | 272.5015 | 17612.92 | 17015.49 | 8340005 | 2.58e-5 |
| 140 | 30938.6937 | 485815.63 | 446446.50 | 325953120 | 2.78e-5 |

warmup 为150步，LR连续增加，120–130步没有 scheduler 阶段切换。日志 `learning_rate` 来自第一个 LoRA 参数组；decoder LR为其0.3倍，不是 decoder 被误设成 `1e-4`。step120总loss仍小、梯度先异常，说明只看每10步总loss会晚发现风险。平均token数在该段没有突增；同seed仍可能重复遇到触发样本，现有日志未保存逐批样本标识，不能排除具体batch的作用。

### decoder 参数尺度的实测与解释边界

在当前 `bitvae` 环境用单线程 CPU mmap 读取初始 checkpoint 的小型 decoder 张量，无整模型/GPU加载。模型有252个压缩Linear、每个2个stage；均为hidden128的symmetric decoder。原CAT训练启用stage归一化，导出时将std/mean融合到 `linear_out.weight/bias`；`q_scale` 则融合在 `linear_in`，两者不要混淆。融合见 [cat_train_pipeline.py](../../../../../train_utils/cat_train_pipeline.py) 的 `_fuse_norm_into_decoder`，q_scale见 [autoencoder.py](../../../../../litebsq/autoencoder.py)。

| 参数 | stage1 张量 RMS 中位数 | stage2 张量 RMS 中位数 | 全部 stage 最小 RMS |
| --- | ---: | ---: | ---: |
| linear_out.weight | 0.003413 | 0.002069 | 0.001416 |
| linear_out.bias | 0.001350 | 0.000800 | 0.000397 |

E2E直接训练这些融合后的参数，output bias也在decoder组，全部采用同一decoder LR、零weight decay。若Adam归一化更新因子约为1，峰值`3e-5`对应典型output weight RMS约0.9%/1.45%，典型bias RMS约2.2%/3.75%；这是更新量级估算，不是已测得的真实参数位移。decoder参数在大量权重块中共享，改变它会同时改变整块重构权重，不能用其绝对LR小于LoRA LR推断更新更温和。

数学上 `W_fused=std*W_normalized` 的前向等价不保证同绝对LR下Adam优化等价，换算回原参数坐标的步长尺度为 `LR/std`。这给“decoder联合更新过强导致失稳”提供了具体机制依据；尚没有失稳前参数/梯度分组快照，不能证明哪层、哪组参数首先失稳或认定LR是唯一原因。

### 代码及历史证据的排除范围

- 正常decoder训练优先走packed uint8第一层，再用原生norm/activation/output linear，条件见 [vae_linear.py](../../../../../litebsq/vae_linear.py) 的 `_decode_weight`；不应直接归咎整段fused decoder的自定义backward。静态检查未发现packed第一层梯度漏除、重复归约或step120切换，decoder kernel的码块形状也不由动态token长度决定。
- FP32叶参数及Adam状态没有被前向永久降为BF16，临时计算仍为BF16。上一轮loss/hook/crop/clip检查仍适用；裁剪限制梯度范数，并不直接限制Adam参数更新或下一步激活。
- 0916–0919的历史E2E实际为 `train_mode=lora`，decoder冻结；即使快照包含decoder_lr也不能当作decoder参与训练的证据。核对原始日志，0916_093211开启hidden0.1/preMLP0.01，记录至6820步，末窗口loss0.1813、grad_norm0.3477；0918_111452至7160步，末窗口loss0.1805、grad_norm0.2261；0919_011304至10000步，末窗口loss0.1881、grad_norm0.2465。三组记录均未出现本次同类爆涨。后两组还与当前相同地开启norm all、lm_head LoRA，LoRA/norm/head LR均为1e-4，支持重点排查decoder联合更新；原日志在 `/root/data/ckpts/result/compressed_e2e_fintuning/only_lora/` 对应完整run_id目录下，关键行分别为1944、2048、2674。0919日志下一行train_loss0.20617是全程摘要，不能当最后窗口loss。
- 上述冻结组使用0910_180322 checkpoint，当前使用0920；以0919为例还存在SFT→LM、任务样本比0.3→0.5、KL→KD、hidden0/0→0.1/0.01、总步数10000→5000（同warmup_ratio使预热300→150步）等变化。因此历史稳定是有价值的定位证据，但不是“只开放decoder”的单变量因果证明；需要时应在当前checkpoint和协议下仅改变train_mode来隔离。
- 0907_181143的旧decoder_lora、decoder_lr1e-5曾稳定到5000步，但初始checkpoint、hidden权重及协议不同，不能证明当前1e-5必然有效。同一0920 checkpoint尚无decoder冻结或低LR已完成稳定训练的对照。

当前优先解释是共有的decoder联合优化与参数尺度/更新强度问题，仍保留norm/head、hidden梯度交互及特定batch触发的未验证可能。`165835` 的单变量降LR已经由用户启动，检查时torchrun PID为`659862`，无需重复启动同配置；稳定性判断至少须跨过已知120–150步失稳区间。若仍失稳，下一个有区分力的对照是只冻结decoder、保持其余条件，必要时采集失稳前分组梯度和逐层相对误差，而不是再次切换残差模式。此处是分析建议，不是代理启动新实验的授权。

以下保留14:51阶段的事实和判断来源；首要嫌疑及下一步建议以上述新证据为准。

## 状态与范围

本次只读分析现有训练、配置、checkpoint 元数据与源码，没有重启、停止或修改训练。2026-09-24 14:51（Asia/Shanghai）进程检查时，最新运行仍在计算，torchrun PID `609494`、四个 rank PID `609571`–`609574`；这是数值发散，不是已确认的进程崩溃。前次运行未取得退出状态，不标记正常完成。

检查环境是 `cs-6b0bf-a61ff-server:/root/VAELLM`，可见 4 张 A800-SXM4-80GB，训练使用 `/root/miniconda3/envs/bitvae/bin/python`、Python 3.11.13。默认 SSH 别名 `iaaccn74` 在此会话无法解析，以下证据来自当前训练主机，不宣称已同步远程文档主源。

## 实际配置与证据

两组目录均位于 `/root/data/ckpts/result/compressed_e2e_fintuning/only_lora/`：

| run_id | 唯一有效配置差异 | 观察 |
| --- | --- | --- |
| `Qwen_Qwen3-8B_20260924_102602` | `loss_type=kd_top_partial` | step120 hidden loss 100487.7，后续持续剧烈波动 |
| `Qwen_Qwen3-8B_20260924_132618` | `loss_type=kl_top_partial` | step130 hidden loss 14877941，裁剪前 grad_norm 28688318464 |

每个目录中的 `normalized_e2e_runtime_args.json` 是实际完整配置，`compressed_e2e_fintuning.log` 是核心日志。除输出路径、时间字段外，有效配置只有上表差异；不是同 checkpoint 的 only-LoRA 对照。共同条件：

- 初始模型 `/root/data/ckpts/result/catlora/remaining_lora_mass/Qwen_Qwen3-8B_20260920_095822/final_model`，checkpoint id `1a6fa98c-6685-4dab-a222-e03695919bfb`。
- `train_mode=decoder_lora`，`decoder_lr=3e-5`，projection/residual LoRA、norm、lm_head 的峰值 LR 均为 `1e-4`；warmup 为 5000 步的 3%，梯度裁剪 1.5。
- `residual_lora_mode=replace`，rank8、alpha16、dropout0；FP32 清单为 `decoder,lora,lm_head,norm`，没有 `residual_lora`。
- `hidden_loss_weight=0.1`，`pre_mlp_hidden_loss_weight=0.01`，`linear_depth`，K=100，temperature1，seed/data_seed 均为0。
- `dataset_task=lm`，长度1024，四卡 DP、每卡 batch8、accumulation1；数据混合权重为 `edgerazor_ii_7m=.341,edgerazor_ii_gen=.067,edgerazor_tulu=.028,edgerazor_am=.064,vaellm_eval_task=.5`。

源码基于 commit `82993cfdb589934e4dc771797343701b36890393`；已有未提交脚本修改保留。复现入口 [e2e_decoder.sh](../../../../../compressed_e2e_fintuning/scripts/e2e_decoder.sh) 的检查时 SHA256 为 `96bd7390bd78752e6182d41f07dc649dd3165b2b11b8025e74bf2bafaeae2027`。这只是入口与配置追溯，不是建议继续原样启动。

## 已确认与待验证原因

1. 初始 checkpoint 元数据没有残差拓扑；在 bitvae 中用 CPU FakeTensorMode/mmap 只读检查 2919 个 state_dict 条目，也没有 `residual_lora` 权重。本次是新装模块。[安装路径](../../../../../train_utils/model_level_trainables.py) 的 `build_model_level_trainable_selection` 调用 `install_residual_lora`。
2. [残差实现](../../../../../e2e_common/residual_lora.py) 对 replace 初始化正交行矩阵 A，令 B=A.T/scaling，前向只返回 D(x)，不保留 x。于是 36 层、attention/MLP 两处的 4096 维恒等 skip 全部变成 rank8 投影。初始函数发生结构性变化是确定事实；14:51将它列为首要机制假设，后续additive/none对照已排除它是共同发散的必要条件。
3. 新残差参数继承初始 norm 的 BF16 dtype。`configure_distill_precision` 按独立 inventory 升精度，`lora` 不包含 `residual_lora`，因此本配置没有把残差参数转为 FP32。其稳定性贡献尚未独立验证；decoder 同时训练也是混杂条件。
4. 当前纯 KL 路径不使用 alpha=0.95，也不含 CE；实际目标为 `KL_top_partial + 0.1*hidden + 0.01*pre_mlp`。两种输出损失下都发散，证据不支持“去掉 CE 即可修复”。
5. hidden/pre-MLP 实现为 FP32 masked MSE / (教师 masked mean-square + 1e-6)，跨层权重归一化后平均。静态检查未发现层数误求和、教师梯度串入、checkpoint 重算重复采集或学习率参数组串用。不能据此宣称所有底层算子均已运行验证。
6. 本地 Transformers 4.51.0 的 Trainer 调用梯度裁剪，并记录返回的裁剪前范数；大 grad_norm 不证明裁剪失效。[trainer.py](../../../../../compressed_e2e_fintuning/trainer.py) 的 `_store_loss_parts/log` 则记录最后一次本 rank 分项，与总 loss 的窗口/多卡平均不同。不能用同一行分项之和核对总 loss；日志口径不解释真实分项爆涨。

历史 [残差对照](residual_lora_ab_20260923.md) 使用0910 checkpoint、additive、decoder冻结、残差FP32，不能当作本轮 replace 配置的稳定性证明。没有保存本次中间模型或逐层观测，现有材料不能判断最先失稳的层/参数组，也没有本轮下游结果。

## 建议与收尾

14:51曾建议从同一原checkpoint仅改变残差模式做对照；用户随后运行的additive/none已补充这项证据，下一步不重复该建议。本次两轮分析均未修改算法、损失、学习率或运行中的脚本，也未启动或停止训练。

经验合并到[残差 LoRA 经验](../../../lessons/residual_lora.md)。当前实验仍运行，日志/配置和初始模型保留；本次未生成测试模型、临时脚本或评测缓存，无清理对象、释放空间为0。不删除前次证据，也不因诊断自动淘汰整个方法分支。

在当前 bitvae 环境执行了 `python tools/docs.py index` 和 `python tools/docs.py check`。当前目录缺少部分历史/结果文件，检查仍有既有失效链接，本次修改文档的链接未报错；索引重建会删除大量远程历史条目，因此已撤回本次生成的索引内容，保留原目录，待远程主源可连接后刷新。不扩大本次诊断去修复缺失的历史资料。
