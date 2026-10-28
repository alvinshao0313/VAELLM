# CAT 旋转后双向归一化与保护通道六臂对照

2026-10-22，Qwen3-8B 压缩地基试验。用户已授权在保留 VAELLM/BSQ 主体的条件下比较旋转后的归一化与 INT8 保护。**10月23日终态：六臂正式训练与完整局部读出全部正常完成。保留INT8时，learned相对current的完整MLP误差降低约1.41%；去保护仍明显恶化。没有完整模型质量结论，不据此扩大为全模型重压。**

唯一实现模块为 [`experiments/cat_biaxial_scale`](../../../../experiments/cat_biaxial_scale/)，唯一结果根目录为 [`result/qwen_cat_biaxial_20261022`](../../../../result/qwen_cat_biaxial_20261022/)。本记录持续维护授权设计、验证、正式运行与最终取舍；模块 README 仅维护使用接口。

## 问题、对照和解释边界

检验假设：在相同 BSQ 码率与 decoder 结构下，先对旋转后的权重做行、列尺度分解，是否改善原坐标权重和真实输入上的重建；允许尺度学习后，收益与存储代价如何变化。同时比较有无 INT8 保护，**不能在结果出来前预判保护可以取消**。

三个归一化条件与两个保护条件完全交叉，共六臂：

| 归一化 | `int8` | `none` |
| --- | --- | --- |
| `current`：现有逐 stage 标量均值/标准差归一化 | 当前归一化＋保护 | 当前归一化＋无保护 |
| fixed biaxial：旋转后固定行列尺度，再保留逐 stage 标量归一化 | 固定双向＋保护 | 固定双向＋无保护 |
| learned biaxial：同初始化，训练中学习行列尺度 | 可学习双向＋保护 | 可学习双向＋无保护 |

三种归一化路径均已完成必要验证；六臂固定预算比较已完成，不据末尾训练 loss 宣称收敛。

所有臂从同一原始 BF16 HF 权重重新压缩。仅选择第 0、18、35 层的 `gate_proj`、`up_proj`、`down_proj`，共九个**完整矩阵**，不缩成局部 tile。每类三个独立自编码器以 M3 并行组训练，不共享 decoder。三种归一化在同一保护条件下使用相同 M3 的 pooled BSQ 辅助损失口径；去保护会改变向量总体、每轮边界及随机数消费，不能声称两保护条件逐 batch 内容完全相同。

历史纯 CAT `ab0280cf` 的修核后完整八项为 61.33014121734129%、投影位宽 2.4569887907608696bpw，来源见[历史验收](../../../../result/qwen3_8b_w25_20260930/reconstruction_block32_o3_down3/eval_fixed_decoder/acceptance.json)。它按类别 M36 并行，本次为 M3，BSQ 的 pooled population 已变；历史完整八项仅作背景，**不能直接与本次局部诊断组成归一化单变量对照**。已恢复的 f927、a0、C1、B32 也不是本次 fresh 权重基线。

参考经验：[压缩配置与组合边界](../../lessons/compression_choices.md)、[输出损失及局部代理边界](../../lessons/output_losses.md)、[保存重载与数值路径](../../lessons/checkpoint_lifecycle.md)。已有局部输出误差下降而完整任务退化的反例，因此本次不按局部 NMSE 宣称全模型质量提升。

## 固定训练条件与保护来源

| 项目 | 本次条件 |
| --- | --- |
| 原模型 | `/home/star/models/Qwen/Qwen3-8B`，官方 revision `b968826d9c46dd6066d109eabc6255188de91218`；九矩阵直接读取原始 safetensors |
| 范围 | L0/L18/L35，各完整 gate/up/down；gate/up 为 12288×4096，down 为 4096×12288 |
| 编码结构 | D32、S2、H128、R0，LayerNorm＋Swish、symmetric decoder；gate/up K32，down K48 |
| 主码名义码率 | gate/up 为 2bit/weight，down 为 3bit/weight；不是含 decoder/scale/保护的实际位宽 |
| 旋转与布局 | 双侧 Hadamard block32；gate/up 不转置，down 转置；每个矩阵独立符号与 decoder |
| 重构预算 | 每类别 M3，每个独立矩阵每批 8192 个 D32 向量，每 stage 10000 步 |
| 优化器 | AdamW，LR0.003，betas=(0.9,0.95)，weight decay0，linear，无 warmup，seed31 |
| BSQ 辅助项 | 沿原 pooled BSQ：l1_weight1、lfq_weight2.5、commitment0.25、entropy0.01、inv_temperature100；不以归一化改变其人口或系数来冒充单变量 |
| 文本训练、模型恢复 | 无模型级 KD、无 LoRA、无 norm/head 更新；权重拟合与尺度初始化不使用文本 |

`int8` 臂仅从以下纯 CAT 成品读取各 Linear 的 `protected_input_indices`：

```text
result/qwen3_8b_w25_20260930/reconstruction_block32_o3_down3/
  home_star_models_Qwen_Qwen3-8B_20261001_112401/final_model
UUID ab0280cf-3637-47e3-b771-074b74532a54
```

gate/up 各保护 64 个输入通道，down 为 256；索引、排序和数量完全复用。其当时排名来源是 Alpaca official train、seed31、512 个连续 packed×256 token 块，`channel_weight_actmean_abs`、input/layer scope、refresh=false，见[原始来源记录](2026-09-30_goal.md)。**不读取该成品的恢复后投影权重或 INT8 payload**；保护列来自本次原始 HF BF16 权重，重新 INT8 encode。`none` 臂不保护任何列，K/D/S 保持相同，不把释放的存储位重新分配给主码。

## 归一化数学与必须保持的目标

用实数矩阵 `U[m,n]` 表示抽出保护列后、按原两侧旋转得到的完整未保护矩阵，按规范的 `[output,input]` 坐标定义尺度；转置只影响向量布局，不互换行列尺度的语义。令全矩阵 RMS 为 `rho`，固定初始化为：

```text
r_i = RMS(U[i,:]) / rho
c_j = RMS((U / r[:,None])[:,j]) / rho
S_ij = r_i * c_j
V = U / S
```

零 RMS 行/列采用单位尺度；尺度须为有限正值。先顺序计算 FP32 行、列统计，再将两向量落为 BF16，并以实际 BF16 存储值做 normalize/restore。`current` 沿原各 stage 的标量均值/标准差；双向臂在 V 空间仍保留相同的逐 stage 标量归一化与其 decoder 融合。

双向变换不能仅优化均衡后空间的未加权 MSE。重构误差必须乘回实际 S，回到物理旋转坐标；逐 stage 标量归一化的 std 也必须还原，再按对应原物理残差 std 的尺度比较。现有 fixed 实现的加权项为：

```text
MSE(((decoded - normalized_target) * S_block * normalized_std / physical_std))
```

输入目标沿原生 BF16 训练边界，加权误差在 FP32 计算；不把它写成拟合未舍入 FP32 目标的证据。原尺度权重 NMSE 与部署输出诊断另行重算，防止只在均衡后空间获得表面改善。

learned 臂保持原物理矩阵 U 不变。若尺度 S 在第二 stage 继续变化、第一 stage 的归一化坐标输出为冻结 F1，第二 stage 的动态残差是 **`U / S - F1`**；不沿用旧 S 的静态 residual。实际损失为 `mean(((S*(F1+mu+std*decoded)-U)/physical_std)^2)`，stage 起点的统计量和分母固定，动态输入和外侧 S 均保留梯度。每矩阵每轴使用去均值的 FP32 log-delta，前向取实际 BF16 scale；Adam 在 stage 边界重建，保留尺度值，学习率/调度与 codec 相同。CPU 梯度验证及真实两 stage 更新均通过。

须保留两项解释边界：learned 拟合 exact FP32 U，fixed 沿原生 BF16 target；learned 的 F1 来自真实 packed decoder，fixed/native 使用未融合 forward 再以 FP32 mu/std 恢复的残差。因而 learned−fixed 检验完整新方法，不能严格全部归因为“只解冻 scale”。各 stage 记录 `input_cast_physical_sse`，最终统一比较实际部署矩阵；小于相关舍入影响的收益不作强结论。

部署先将各 stage 归一化空间解码结果相加，再乘回行列尺度，之后逆旋转、恢复保护列并转为 BF16 完整原坐标矩阵。行列 scale 不当作免费参数：真实 BF16 scale、decoder、packed codes、保护 INT8/scale/indices、旋转 signs 全部计入独立 endpoint 的 logical/storage 位宽；FP32 训练 master、encoder 和优化器不计部署库存。保存后须 strict 重载，核实际持久状态与 BF16 部署矩阵一致；本次是局部 packed endpoint，尚非完整模型 v6 成品或全模型≤2.5bpw验收。

## 已完成的输入准备与诊断口径

[prepared/manifest.json](../../../../result/qwen_cat_biaxial_20261022/prepared/manifest.json)已为 `captured`：九份原始 BF16 权重、保护索引、输入 token 与实际原 HF 输入激活已经准备完成。`inputs.pt` 905983626B；`tokens.pt` 525586B；`activations.pt` 3015478320B。SHA、逐矩阵身份、每文档行号/内容 hash 和输入来源由该 manifest 唯一保存。

诊断文本为 `HuggingFaceFW/fineweb-edu` revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`、`sample-10BT`、**official train**、ODC-BY v1。来自已核完整首分片的本地 `data/fineweb_edu_10bt_20261005/train.jsonl`，固定物理行 `[700000,700128)`、128 文档、max256，实际 **30675 个有效 token**。它不是 128 个全部满长序列；连续行在候选结果前指定，无文本/长度/模型表现筛选。源版本与许可见[原数据 manifest](../../../../data/fineweb_edu_10bt_20261005/manifest.json)。

冻结原 BF16 Qwen、FlashAttention2、FP32 RoPE、TF32关闭，一次捕获 L0/L18/L35 的真实原输入；gate/up 共用输入，down 输入来自原教师 MLP。已完成 capture 耗时12.9454秒，峰 allocated16526343680B；这只说明数据读出已完成，不是正式六臂训练或质量验证。

每臂完成 packed 保存重载后报告三种诊断：

1. 九矩阵原坐标权重 NMSE。
2. 九矩阵在固定**原教师输入**上的 Linear 输出 NMSE；单独 down 使用教师 down 输入，以隔离该投影误差。
3. 三层完整 MLP 输出 NMSE：固定教师 gate 输入，**candidate gate＋candidate up → SiLU乘积 → candidate down**，不在最后一步换回教师 down。

诊断从实际 packed endpoint 解码出 BF16 部署矩阵，再转 FP32 计算 Linear/SiLU/完整 MLP；误差与参考能量以 FP64 累加，TF32关闭。这不是原生 BF16 全模型执行，也不是完整下游任务/PPL/LongBench质量。分别保留每矩阵、每层结果与宏均，不用一个平均掩盖不同层/类别的相反变化。文本与这些读出均不进入纯权重拟合或尺度初始化。

## 必要验证、执行边界与收尾

### 10月23日：六臂完整终态

两条原链均已自然结束，`int8_chain.exit`／`none_chain.exit`均为0，原会话已退出。18份类别报告均为completed／exit0，36个stage的实际最后更新均为10000；54个packed端点均报告strict持久状态及BF16部署矩阵重载exact，逐项逻辑／存储bit账本相加一致。六份观察均为completed，沿同一128篇official train／30675 token和原教师输入。这是局部诊断完成，不能当作完整模型九项验收。

| 归一化／保护 | 权重 NMSE 宏均 | Linear 输出 NMSE 宏均 | 完整 MLP 输出 NMSE 宏均 |
| --- | ---: | ---: | ---: |
| current／INT8 | 0.112574228 | 0.043542521 | 0.115361616 |
| fixed biaxial／INT8 | 0.112072632 | 0.043148564 | 0.114118483 |
| learned biaxial／INT8 | 0.112069535 | 0.043204341 | 0.113732695 |
| current／none | 0.114052407 | 0.072745073 | 0.230050725 |
| fixed biaxial／none | 0.113559175 | 0.071975244 | 0.244905008 |
| learned biaxial／none | 0.113714418 | 0.071951226 | 0.241468943 |

新增原件：[learned INT8观察](../../../../result/qwen_cat_biaxial_20261022/formal/learned_int8/observation.json)及其[gate](../../../../result/qwen_cat_biaxial_20261022/formal/learned_int8/gate_proj/report.json)／[up](../../../../result/qwen_cat_biaxial_20261022/formal/learned_int8/up_proj/report.json)／[down](../../../../result/qwen_cat_biaxial_20261022/formal/learned_int8/down_proj/report.json)；[learned none观察](../../../../result/qwen_cat_biaxial_20261022/formal/learned_none/observation.json)及其[gate](../../../../result/qwen_cat_biaxial_20261022/formal/learned_none/gate_proj/report.json)／[up](../../../../result/qwen_cat_biaxial_20261022/formal/learned_none/up_proj/report.json)／[down](../../../../result/qwen_cat_biaxial_20261022/formal/learned_none/down_proj/report.json)。其余四臂原件见下方阶段记录。

learned相对current，在INT8条件下权重／Linear／完整MLP误差分别降低0.4483%／0.7767%／1.4120%；三层MLP NMSE分别为0.108582832／0.188285004／0.044330249，均低于current，但L0仅微小变化。相对fixed，其Linear宏均略差、MLP宏均略好，不能称各项一致提升。无保护时，learned的权重／Linear误差分别降低0.2963%／1.0913%，完整MLP却增加4.9633%；L35从current的0.328808187升至0.371390106，继续体现非线性组合与单Linear排序分离。learned与fixed的训练边界差异仍按前述数学部分解释，不能将差值全部归因于解冻scale。

本终态不支持取消保护，也未提供足以直接扩大为全模型重压的局部增益。保留结果供后续设计取舍，不再按同一配方追加种子／步数网格；没有新下游精度或全模型位宽结论。此段替代下方历史记录中的“learned尚未完成”。经验增量仍属既有局部代理边界，没有新增通用算法结论。

### 18:55 四臂正式阶段结果与夜间取舍

本次只读核对四份 `observation.json` 和十二份类别 `report.json`：均为 `completed`、类别 `exit_code=0`，每 stage 实际 10000 步，三个矩阵的持久状态及 BF16 部署权重 strict 重载检查均通过。完整局部读出仍使用上述固定 128 篇 train 文档／30675 token 和原教师输入，不是原生 BF16 完整模型推理。learned 两臂尚未完成全部类别，不能用其已完成 gate 代替整体结果。

| 已完成臂 | 权重 NMSE 宏均 | Linear 输出 NMSE 宏均 | 完整 MLP 输出 NMSE 宏均 | 原始读出 |
| --- | ---: | ---: | ---: | --- |
| current＋INT8 | 0.112574228 | 0.043542521 | 0.115361616 | [observation](../../../../result/qwen_cat_biaxial_20261022/formal/current_int8/observation.json) |
| fixed biaxial＋INT8 | 0.112072632 | 0.043148564 | 0.114118483 | [observation](../../../../result/qwen_cat_biaxial_20261022/formal/biaxial_int8/observation.json) |
| current＋none | 0.114052407 | 0.072745073 | 0.230050725 | [observation](../../../../result/qwen_cat_biaxial_20261022/formal/current_none/observation.json) |
| fixed biaxial＋none | 0.113559175 | 0.071975244 | 0.244905008 | [observation](../../../../result/qwen_cat_biaxial_20261022/formal/biaxial_none/observation.json) |

固定双向相对同保护条件的 current，INT8 下权重／Linear／完整 MLP 宏均误差相对变化为 −0.446%／−0.905%／−1.078%；但完整 MLP 的 L0／L18／L35 分别为 −0.754%／+0.521%／−8.519%，不是逐层一致改善。无保护下权重／Linear 分别降低 0.432%／1.058%，完整 MLP 却增加 6.457%；三层分别为 −1.845%／−2.948%／+16.362%。两种保护条件下均有 8/9 个 Linear 输出误差降低，仍不能推出完整 MLP 误差同向。

在本次保持主码 K/D/S、不重新分配存储的条件下，去保护使 current／fixed 的完整 MLP 宏均误差分别增加 99.417%／114.606%；L35 的 NMSE 分别由 0.045875433→0.328808187、0.041967094→0.382606529。这个结果不支持取消保护，也不能外推为所有无保护或重分配码率方案无效。fixed 真实 BF16 双向尺度增加每矩阵约 0.00513–0.00521bpw，已计入端点账本；这些仍是九矩阵局部账本，不是完整模型 ≤2.5bpw 验收。

正式训练、配置、源 SHA、账本和重载证据：current＋INT8 的 [gate](../../../../result/qwen_cat_biaxial_20261022/formal/current_int8/gate_proj/report.json)／[up](../../../../result/qwen_cat_biaxial_20261022/formal/current_int8/up_proj/report.json)／[down](../../../../result/qwen_cat_biaxial_20261022/formal/current_int8/down_proj/report.json)；fixed＋INT8 的 [gate](../../../../result/qwen_cat_biaxial_20261022/formal/biaxial_int8/gate_proj/report.json)／[up](../../../../result/qwen_cat_biaxial_20261022/formal/biaxial_int8/up_proj/report.json)／[down](../../../../result/qwen_cat_biaxial_20261022/formal/biaxial_int8/down_proj/report.json)；current＋none 的 [gate](../../../../result/qwen_cat_biaxial_20261022/formal/current_none/gate_proj/report.json)／[up](../../../../result/qwen_cat_biaxial_20261022/formal/current_none/up_proj/report.json)／[down](../../../../result/qwen_cat_biaxial_20261022/formal/current_none/down_proj/report.json)；fixed＋none 的 [gate](../../../../result/qwen_cat_biaxial_20261022/formal/biaxial_none/gate_proj/report.json)／[up](../../../../result/qwen_cat_biaxial_20261022/formal/biaxial_none/up_proj/report.json)／[down](../../../../result/qwen_cat_biaxial_20261022/formal/biaxial_none/down_proj/report.json)。

**本轮决定：不新增夜间 GPU 确认或训练任务。** 保留原 CAT learned 两臂及链中既定完整局部读出，保留 H256 完整评测和 DKD128／32、128 两端点完整九项；GPU2／3／7 留给 DKD 后续评测。固定双向的小幅、跨层不一致局部改善，尚不足以单独支持新的恢复训练或完整框架整合；不新增 seed、比例或步数网格，不凭局部 NMSE 宣称九项收益。

经验增量检查：上述结果补充了[输出损失及局部代理边界](../../lessons/output_losses.md)与[压缩配置和存储口径](../../lessons/compression_choices.md)已有规则的同源证据，尤其是单 Linear 改善与完整非线性 MLP 排序分离；本阶段没有需要另建 lesson 的新通用规则。六臂仍未收尾，本次仅更新这一份记录，不清理活跃产物、不修改活动源码或配置、不追加监控。

### 已有必要验证与启动记录

CPU 检查覆盖原/转置向量尺度索引、正值及零轴处理、加权损失及梯度、S=1 时原生 M3 BSQ 损失/参数梯度、标量融合、动态输入和尺度 gauge。真实 GPU 验证沿正式 M3/B8192/D32/S2/H128/K32或48 路径，仅将每 stage 更新缩至2步；共18个真实端点的 persistent state 与同一内核 BF16 部署权重在 strict 重载后完全一致：

| 验证 | 证据 | 峰 allocated |
| --- | --- | --- |
| fixed 三类、INT8，9矩阵 | [gate](../../../../result/qwen_cat_biaxial_20261022/validate_fixed_r2/gate_int8/report.json)、[up](../../../../result/qwen_cat_biaxial_20261022/validate_fixed_r2/up_proj/report.json)、[down](../../../../result/qwen_cat_biaxial_20261022/validate_fixed_r2/down_proj/report.json) | 最大1435921920B |
| learned gate-none、down-int8，6矩阵 | [gate](../../../../result/qwen_cat_biaxial_20261022/validate_learned/gate_none/report.json)、[down](../../../../result/qwen_cat_biaxial_20261022/validate_learned/down_int8/report.json) | 最大3504441344B |
| current up-none，3矩阵 | [原生控制包装](../../../../result/qwen_cat_biaxial_20261022/validate_current/up_none/report.json) | 1280906752B |

learned 两 stage 的 scale 梯度均有限非零，BF16 scale 实际发生变化，第一 stage 的码 SHA 在第二 stage 保持不变。fixed 的9端点已完成[整套读出](../../../../result/qwen_cat_biaxial_20261022/validate_fixed_r2/observation.json)，三类误差均为有限值，耗时12.7488秒；仅2步短测的误差不能用于质量排名。代码及真实路径验证已充分，不另设重复启动前审计。

首次独立加载器遗漏 native grouped decode plan 刷新，被重载检查发现；已复用 `refresh_vae_linear_runtime_after_state_load` 修复并通过上述复验。公共源码未改。该问题属于已有[缓存刷新经验](../../lessons/checkpoint_lifecycle.md)覆盖范围，无新增压缩收益经验。核心日志统一保留在 [validation.log](../../../../result/qwen_cat_biaxial_20261022/validation.log)，每个成功报告保留实际参数、源码 SHA、梯度/更新、位宽和 exit0。

原拟使用 GPU4/5。最终启动前核查发现15:59新增 `qwen_L9_H256_short2_20261022`（GPU4短验，后续GPU4训练/full8、4/5 LongBench），16:01新增 `llama_DS32_MLP_layers_GPU5_20261022`（GPU5层扫描）；GPU0/2/3与1/6/7仍由旧R512/J32-anchor完整评测占用。**用户明确选择“保留现有安排，CAT等空卡”。** 没有停止任何既有任务，也没有启动正式 CAT、占卡或监控自动化；下一次用户唤起后重新核实空卡及预约，再运行。

17时恢复推进后，原 J32-anchor 的 GPU6／1 分片依次自然完成，原 Python1528684／1528681 已退出，分别实查 GPU 无 compute 进程后执行以下既定入口。调用者 shell 使用项目根目录和已激活的 `bitvae`，实验脚本和全部训练条件未改：

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/cat_biaxial_scale/run_int8.sh
CUDA_VISIBLE_DEVICES=6 bash experiments/cat_biaxial_scale/run_none.sh
```

每条流程顺序执行 current、biaxial、learned 的三类训练，共9个任务，每个完整归一化条件训练后读出。结果写入 `result/qwen_cat_biaxial_20261022/formal/<normalization>_<protection>/<category>/`，无自动重试。所有正式训练均从原权重/随机 codec 初始化，不继承短测权重。

真实启动与命令归 [launch.json](../../../../result/qwen_cat_biaxial_20261022/launch.json)：无保护链 tmux `qwen_CAT_none_20261022`，pane1609607／首训练1609620；INT8链 tmux `qwen_CAT_int8_20261022`，pane1611896／首训练1611910。两链的 stdout/stderr 与退出分别落在结果根目录 `none_chain.log/.exit`、`int8_chain.log/.exit`，各类别核心日志在 `formal/`。17:30:25无保护首阶段已完成100／10000步；17:33:23 INT8同样完成100／10000步，M3／B8192／两stage／正式10K参数保持。这里只确认实际推进，完整六臂及局部读出尚未完成，不代表九项验收通过；不追加自动重试或监控调度。

启动前仅核入口版本差异：当前 `run.py` SHA `6562580c952379153e73787993a61f951de5964611457538934015de127dac02` 不同于短测记录；最新三份短测的其余7个模块核心 SHA 与当前一致。旧本地字节码的静态对比确认 `fit`／`save_endpoints`、入口条件、设备、随机初始化及错误处理未变，仅增报告依赖 SHA 和精度／初始化说明，新增15个依赖路径均存在。因此复用已通过的真实短测，没有重跑 GPU 验证，也没有修改核心源码。当前入口和两脚本 SHA 随实际命令保存在 `launch.json`。

按持续清理授权，已删除失败的 `validate_fixed/` 和成功短测的6个 `endpoints/` 目录，删除归并后的散日志/exit文件；保留6份成功训练报告、完整读出JSON和单一核心日志。实际删除文件286619636B，对应allocated文件空间286773248B（约273.5MiB）；未计共享磁盘变化。短测报告中的旧endpoint路径是历史证据，权重已清理。`prepared/`约3.65GiB保留供本次已安排的正式比较使用；原HF、原数据、保护来源及所有公共源码均保留。

收尾检查：两份启动脚本 `bash -n` 通过，`python tools/docs.py index` 成功；全库 `check` 仍报告148处既有失效链接，本次新增记录/模块说明没有失效链接。检查原始输出已追加至 `validation.log`，不能将全库链接检查写成通过。


## 2026-10-23 生成权重收尾

六个正式局部对照及其观察已完成，54个正式端点不再用于后续训练/复评；仅保留结果证据，不将局部误差当作完整九项结论。 清理前已核正式退出、所录PID已退出、当前进程与打开文件、新QA-RKL源码及可用manifest引用。下表在删除前登记；每条路径均为普通文件、nlink=1、解析后位于本批准目录。原始数据、QA/M/R教师缓存、源J32/HF、全部源代码/日志/指标/报告/配置/metadata保持。

CAT prepared输入/激活和全部短验端点本轮保留，正式18份类别report、六份observation、核心日志和账本原件保持。

精确路径公共前缀：`/home/star/VAELLM/result/qwen_cat_biaxial_20261022/formal/`。下表每行是该前缀下的唯一相对路径；大小为删除前`st_blocks*512`，不使用共享磁盘变化估算。

| 精确相对路径 | allocated bytes | 原因 |
| --- | ---: | --- |
| `biaxial_int8/down_proj/endpoints/model_layers_0_mlp_down_proj.pt` | 19632128 | 已结束生成权重 |
| `biaxial_int8/down_proj/endpoints/model_layers_18_mlp_down_proj.pt` | 19632128 | 已结束生成权重 |
| `biaxial_int8/down_proj/endpoints/model_layers_35_mlp_down_proj.pt` | 19632128 | 已结束生成权重 |
| `biaxial_int8/gate_proj/endpoints/model_layers_0_mlp_gate_proj.pt` | 13266944 | 已结束生成权重 |
| `biaxial_int8/gate_proj/endpoints/model_layers_18_mlp_gate_proj.pt` | 13266944 | 已结束生成权重 |
| `biaxial_int8/gate_proj/endpoints/model_layers_35_mlp_gate_proj.pt` | 13266944 | 已结束生成权重 |
| `biaxial_int8/up_proj/endpoints/model_layers_0_mlp_up_proj.pt` | 13266944 | 已结束生成权重 |
| `biaxial_int8/up_proj/endpoints/model_layers_18_mlp_up_proj.pt` | 13266944 | 已结束生成权重 |
| `biaxial_int8/up_proj/endpoints/model_layers_35_mlp_up_proj.pt` | 13266944 | 已结束生成权重 |
| `biaxial_none/down_proj/endpoints/model_layers_0_mlp_down_proj.pt` | 18976768 | 已结束生成权重 |
| `biaxial_none/down_proj/endpoints/model_layers_18_mlp_down_proj.pt` | 18976768 | 已结束生成权重 |
| `biaxial_none/down_proj/endpoints/model_layers_35_mlp_down_proj.pt` | 18976768 | 已结束生成权重 |
| `biaxial_none/gate_proj/endpoints/model_layers_0_mlp_gate_proj.pt` | 12677120 | 已结束生成权重 |
| `biaxial_none/gate_proj/endpoints/model_layers_18_mlp_gate_proj.pt` | 12677120 | 已结束生成权重 |
| `biaxial_none/gate_proj/endpoints/model_layers_35_mlp_gate_proj.pt` | 12677120 | 已结束生成权重 |
| `biaxial_none/up_proj/endpoints/model_layers_0_mlp_up_proj.pt` | 12677120 | 已结束生成权重 |
| `biaxial_none/up_proj/endpoints/model_layers_18_mlp_up_proj.pt` | 12677120 | 已结束生成权重 |
| `biaxial_none/up_proj/endpoints/model_layers_35_mlp_up_proj.pt` | 12677120 | 已结束生成权重 |
| `current_int8/down_proj/endpoints/model_layers_0_mlp_down_proj.pt` | 19599360 | 已结束生成权重 |
| `current_int8/down_proj/endpoints/model_layers_18_mlp_down_proj.pt` | 19599360 | 已结束生成权重 |
| `current_int8/down_proj/endpoints/model_layers_35_mlp_down_proj.pt` | 19599360 | 已结束生成权重 |
| `current_int8/gate_proj/endpoints/model_layers_0_mlp_gate_proj.pt` | 13234176 | 已结束生成权重 |
| `current_int8/gate_proj/endpoints/model_layers_18_mlp_gate_proj.pt` | 13234176 | 已结束生成权重 |
| `current_int8/gate_proj/endpoints/model_layers_35_mlp_gate_proj.pt` | 13234176 | 已结束生成权重 |
| `current_int8/up_proj/endpoints/model_layers_0_mlp_up_proj.pt` | 13234176 | 已结束生成权重 |
| `current_int8/up_proj/endpoints/model_layers_18_mlp_up_proj.pt` | 13234176 | 已结束生成权重 |
| `current_int8/up_proj/endpoints/model_layers_35_mlp_up_proj.pt` | 13234176 | 已结束生成权重 |
| `current_none/down_proj/endpoints/model_layers_0_mlp_down_proj.pt` | 18944000 | 已结束生成权重 |
| `current_none/down_proj/endpoints/model_layers_18_mlp_down_proj.pt` | 18944000 | 已结束生成权重 |
| `current_none/down_proj/endpoints/model_layers_35_mlp_down_proj.pt` | 18944000 | 已结束生成权重 |
| `current_none/gate_proj/endpoints/model_layers_0_mlp_gate_proj.pt` | 12644352 | 已结束生成权重 |
| `current_none/gate_proj/endpoints/model_layers_18_mlp_gate_proj.pt` | 12644352 | 已结束生成权重 |
| `current_none/gate_proj/endpoints/model_layers_35_mlp_gate_proj.pt` | 12644352 | 已结束生成权重 |
| `current_none/up_proj/endpoints/model_layers_0_mlp_up_proj.pt` | 12644352 | 已结束生成权重 |
| `current_none/up_proj/endpoints/model_layers_18_mlp_up_proj.pt` | 12644352 | 已结束生成权重 |
| `current_none/up_proj/endpoints/model_layers_35_mlp_up_proj.pt` | 12644352 | 已结束生成权重 |
| `learned_int8/down_proj/endpoints/model_layers_0_mlp_down_proj.pt` | 19632128 | 已结束生成权重 |
| `learned_int8/down_proj/endpoints/model_layers_18_mlp_down_proj.pt` | 19632128 | 已结束生成权重 |
| `learned_int8/down_proj/endpoints/model_layers_35_mlp_down_proj.pt` | 19632128 | 已结束生成权重 |
| `learned_int8/gate_proj/endpoints/model_layers_0_mlp_gate_proj.pt` | 13266944 | 已结束生成权重 |
| `learned_int8/gate_proj/endpoints/model_layers_18_mlp_gate_proj.pt` | 13266944 | 已结束生成权重 |
| `learned_int8/gate_proj/endpoints/model_layers_35_mlp_gate_proj.pt` | 13266944 | 已结束生成权重 |
| `learned_int8/up_proj/endpoints/model_layers_0_mlp_up_proj.pt` | 13266944 | 已结束生成权重 |
| `learned_int8/up_proj/endpoints/model_layers_18_mlp_up_proj.pt` | 13266944 | 已结束生成权重 |
| `learned_int8/up_proj/endpoints/model_layers_35_mlp_up_proj.pt` | 13266944 | 已结束生成权重 |
| `learned_none/down_proj/endpoints/model_layers_0_mlp_down_proj.pt` | 18976768 | 已结束生成权重 |
| `learned_none/down_proj/endpoints/model_layers_18_mlp_down_proj.pt` | 18976768 | 已结束生成权重 |
| `learned_none/down_proj/endpoints/model_layers_35_mlp_down_proj.pt` | 18976768 | 已结束生成权重 |
| `learned_none/gate_proj/endpoints/model_layers_0_mlp_gate_proj.pt` | 12677120 | 已结束生成权重 |
| `learned_none/gate_proj/endpoints/model_layers_18_mlp_gate_proj.pt` | 12677120 | 已结束生成权重 |
| `learned_none/gate_proj/endpoints/model_layers_35_mlp_gate_proj.pt` | 12677120 | 已结束生成权重 |
| `learned_none/up_proj/endpoints/model_layers_0_mlp_up_proj.pt` | 12677120 | 已结束生成权重 |
| `learned_none/up_proj/endpoints/model_layers_18_mlp_up_proj.pt` | 12677120 | 已结束生成权重 |
| `learned_none/up_proj/endpoints/model_layers_35_mlp_up_proj.pt` | 12677120 | 已结束生成权重 |

待删除54个文件，allocated合计813883392B。

删除后核验通过：54个正式endpoint文件均已移除，实际释放allocated 813883392B（0.757987976GiB）；全部保留文件逐SHA256不变，prepared输入/激活、短验端点及全部正式报告/观察/日志/配置仍在。原始HF/J32、下一QA-RKL链与夜间归并证据未受影响。
