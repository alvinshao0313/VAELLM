# S4 QK／VO fresh64 隔离执行

本包只执行11月6日已批准的两条范围细分：GPU4为QK-only，GPU5为VO-only，均从原down/pre `fa50bf40-c682-4910-8e72-d024f6b678e4` fresh初始化AdamW，训练groups0…63后独立导出、strict fresh及完整八项。研究目标、匹配控制和后续讨论边界见[原范围批后续安排](../s4_recovery_screen_20261105/README.md#后续候选与当前交付边界)，唯一当前排程见[当前计划](../../docs/experience/records/w25_transfer/current_plan.md)。不自动续128、改变目标／数据或执行LongBench。

新输出根：

```text
/data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/skip_up_20261103/scope_refine_20261106/trial_20261106_qk_vo64
```

## 完整64步结果与本轮结论

两条64步、独立导出／fresh、MMLU与其余七项及controller／driver均已结束，全部exit0，GPU4／5释放。QK终态UUID为`40f76c27-8f58-4bf8-b448-08af2ea439c9`，VO为`e49cff6a-07cd-48a5-8b79-ba83c4837949`。本表所有恢复候选均为同父、相同64步和数据，128步接续另作匹配比较。

| 指标（%） | 共同父0 | all64 | attention64 | QK64 | VO64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMLU | 71.26477710 | 71.24341262 | 71.24341262 | 71.24341262 | 71.23629113 |
| ARC-C | 54.94880546 | 54.52218430 | 54.77815700 | 54.77815700 | 54.86348123 |
| ARC-E | 78.70370370 | 79.58754209 | 79.33501684 | 78.87205387 | 78.82996633 |
| BoolQ | 86.33027523 | 86.17737003 | 86.54434251 | 86.33027523 | 86.48318043 |
| OpenBookQA | 41.00000000 | 41.40000000 | 41.80000000 | 42.60000000 | 41.60000000 |
| PIQA | 77.80195865 | 77.52992383 | 78.12840044 | 78.01958651 | 77.80195865 |
| RTE | 77.97833935 | 77.25631769 | 77.97833935 | 78.70036101 | 77.61732852 |
| Winogrande | 69.37647987 | 70.48145225 | 69.85003946 | 69.92896606 | 69.45540647 |
| 八项均值 | 69.67554242 | 69.77477535 | 69.95721353 | 70.05910154 | 69.73595159 |
| MMLU正确数／14042 | 10007 | 10004 | 10004 | 10004 | 10003 |

QK比attention64八项高.1018880112pp，M持平；3项升、3项降、2项平。OBQA多4题、RTE多2题、Wino多1题，ARC-E少11题、BoolQ少7题、PIQA少2题，M与ARC-C持平；八项等权均值不能解读为所有任务占优。VO比attention64八项低.2212619314pp，M少1题。相对父，QK／VO八项分别+.3835591164／+.0604091738pp，M分别−3／−4题。

| MMLU逐题配对 | 修对 | 改错 | 净变化 |
| --- | ---: | ---: | ---: |
| 父→QK64 | 162 | 165 | −3 |
| 父→VO64 | 164 | 168 | −4 |
| attention64→QK64 | 118 | 118 | 0 |
| attention64→VO64 | 81 | 82 | −1 |
| all64→QK64 | 203 | 203 | 0 |
| QK64→VO64 | 128 | 129 | −1 |

父及四个64步模型共320个原始样本文件SHA、每份64叶／24742条、逐题doc_id及doc／prompt／target hash均核一致；评测版本、0-shot、B64／B32、limit和acc／acc_norm口径相同，逐题重算结果与原始指标一致。原件在各臂`result.json`、`post/evaluation/{mmlu,seven}.json`及对应samples；请求父、data_digest、学习合同、groups／stop64、heldout一致。单父、单seed和已记录的BF16核路径差异仍限制因果解释，不由此认定MLP或某一投影为知识瓶颈。

| 固定面板（各128题；自身0→64） | QK | VO | 64步曝光 |
| --- | --- | --- | --- |
| early | 70→77（+7） | 70→79（+9） | 已训练前半 |
| late | 81→83（+2） | 83→81（−2） | 尚未消费后半 |
| heldout | 60→60（0） | 62→59（−3） | 独立未参训 |

各臂三个面板gold NLL及early／late实际混合目标loss均下降，VO更强拟合未换来更好下游；0步已有1–2题差异，必须使用自身基准，不以loss或小面板替代M／完整八项选优。实际各训练Web1024＋Med1024各一次、QA128各4次；输入1,536,932 tokens／答案监督5472位置，未消费后半不能称训练题。

| 成本与技术完成 | QK64 | VO64 |
| --- | ---: | ---: |
| 训练进程总秒数 | 2240.0662 | 1946.7278 |
| 实际更新平均秒数 | 33.4674 | 28.9405 |
| 训练峰allocated GiB | 35.393353 | 35.629925 |
| MMLU阶段墙钟包络秒数 | 257.0280 | 256.5840 |
| 七项阶段墙钟包络秒数 | 189.3240 | 189.3280 |
| MMLU／七项各自峰allocated GiB | 59.903460／44.797969 | 59.903460／44.797969 |
| 含导出／fresh／八项的正式链秒数 | 2802.7302 | 2506.9600 |
| 正式链GPUh | .77853615 | .69637777 |
| 模型文件bytes | 6,046,045,569 | 6,046,045,569 |
| 可恢复训练态bytes | 30,554,930 | 30,554,930 |

两个正式链合1.47491392GPUh，另计已记录的GPU6短验.150996706GPUh。QK／VO的run终态分别为北京时间11月6日23:48:17／23:48:27。分项墙钟包络按stage日志birth至exit文件mtime重建，包含进程启动退出；与`training.json`内部计时不同，不能再次叠加到`run.elapsed_seconds`。导出包络分别64.8520／64.6480秒，fresh39.9200／38.2040秒。完整时间依据及技术核证已归并产物根`completion_verification.json`。

1164个源码pins当前均匹配；冻结补集、独立读回FP32 masters／完整Adam／RNG／游标、fresh3927字段／有限前向／4.094031292459239bpw均通过。每臂432个FP32 master及Adam状态均有限、Adam step64，`next_cursor=next_group_index=64`。11月7日用户已选择进入QK目标比例批次；QK .5的一份模型／可恢复态保留作本批控制，VO64已结束且无后继，生成模型／runtime载荷／训练态已清理。表中VO文件大小为清理前历史值，现有模型目录仅metadata等证据、不可加载。与attention128及旧all64／attention64共四臂12文件实际释放24,547,909,632 B（22.8620224GiB），未重复计短验11.36GiB；精确清单、状态schema／拓扑／SHA及退出／依赖证据归产物根completion_verification.json的terminal_cleanup_20261107。641份保留证据SHA、1342份保护文件stat无变化；QK .5／原downpre／WebMed128／数据教师QA／源码保护，未运行GPU或修改旧checks。

本次两条细分均未改善M净收益，按预定条件结束进一步单投影／逐层范围搜索。QK可作64步综合分／成本暂选候选，尚非稳定优势或最终范围定稿；距M门槛10111仍少107题，VO少108题，均无LongBench，不能宣称最终九项达标。attention128也已完整结束（M10001、八项69.94632025），其自身64→128无净改善；三臂合并后建议下一批仅讨论QK64的教师alpha=.25／.75对照既有.5，约2GPUh参考，具体见研究正文。用户现已批准QK alpha两臂，现已完成隔离实现／验证并在GPU4／5正式推进，实际状态见当前计划；数据配比等后继未执行。

## 实现与复用

`core.selected_projections`按完整类别名和36层精确集合选择，QK为`q_proj/k_proj`，VO为`v_proj/o_proj`，均72投影／432个decoder叶。选择不依赖原runtime的类别前缀；所有252投影保持原生S4，另180投影decoder冻结。只有选中decoder转换FP32 master，调用原packed trainable decoder路径和原BF16 precision runtime。冻结投影沿原`warm_frozen`在普通`no_grad`内物化权重，实际投影前向保留输入梯度。

保存／恢复、独立进程导出、fresh和完整八项消费者从已验证scope64包隔离复用。训练保存FP32 masters、Adam moments／step、Python／NumPy／CPU／CUDA RNG、稳定参数名、数据游标、来源及源码绑定，正式末点`next_group_index=64`。导出在新进程从原父和保存态重建，检查完整冻结补集，保存一份native模型。最终全252投影账本仍须为logical／packed `4.094031292459239 bpw`，包括decoder／保护／旋转，embedding/head不计入投影账本。

输入、目标和顺序只读复用原WebMed固定缓存，不抽样或重算教师。每步Web16＋Med16＋QA8，每源1024题各一次，QA128各四遍；知识`.5教师＋.5gold`只混合一次，知识／QA各`.5`。seed137、LR5e-7、AdamW betas(.9,.95)、eps1e-8、weight_decay0、foreach=False、joint clip1.5；沿原500步linear日程的前64步。完整八项保持MMLU B64／14042，其余七项B32／10700、0-shot及原acc协议；匹配控制为旧all64／attention64。

## 最小必要验证与入口

CPU准备核真实父state的精确参数集合／数量、第一64组实际曝光、原数据与教师绑定、组2/9覆盖真实长度峰值，并引用旧attention的真实2＋1跨进程resume证据。GPU6每臂顺序执行组2/9两次真实正式batch／长度更新，核72投影有限非零梯度、早／中／晚FP32及BF16更新、全部冻结补集exact，然后独立导出重建named state、strict fresh及finite／位宽核验。新拓扑的masters／Adam／RNG／顺序／游标读取由独立导出实际覆盖；不重复旧resume后再更新的通用矩阵。

正常BF16核造成的0步输出差值只记录，不要求逐位输出一致。持久冻结字段、packed codes及保存重载状态仍保持精确合同。组2的QA长度10593、组9的知识长度969，均来自原固定shape表并由实际消费记录复核。

调用者先激活`bitvae`，每个命令在独立tmux／nohup后台执行：

```bash
source /home/cmcc/miniconda3/etc/profile.d/conda.sh
conda activate bitvae
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.s4_scope_refine_20261106.prepare
bash experiments/s4_scope_refine_20261106/validate_qk.sh
# QK验证通过后可独立启动，不等待VO：
bash experiments/s4_scope_refine_20261106/run_qk.sh
# GPU6空闲时验证VO，通过后独立启动GPU5：
bash experiments/s4_scope_refine_20261106/validate_vo.sh
bash experiments/s4_scope_refine_20261106/run_vo.sh
```

`prepare`和各arm入口均拒绝覆盖既有attempt；无自动重试／轮询／占卡。正式controller每臂依次完成64训练→退出→新进程export→fresh→MMLU→seven→结果汇总，使用自身GPU完成后退出。预计各正式链1.5–2GPUh、两臂新增验证合.5–1GPUh，非硬中断上限；代码准备另列，不根据参数量承诺加速。

实际启动、验证和结果事实在各臂`run.json`、`training/training.json`、`training/updates/`、`post/export.json`、`post/evaluation/`及`result.json`。本包不把部分训练或检查通过称为完整八项已完成。原批次每臂末点曾保留供讨论；评审后QK作为alpha控制继续保留，VO载荷已按上方收尾记录清理，父／数据／已有必要控制保护。

11月6日启动时核证：CPU准备29.8445秒、CUDA未初始化，两范围各432叶／2,479,104个FP32元素。新包文件初建22:52:35至CPU plan完成22:55:37共约182秒（含CPU核证；此前文档／源码阅读未单独精准计时），代码准备不计GPUh。Astra审选择／冻结路径及两臂实测证据通过，无未决阻断。

| 启动核证 | QK | VO |
| --- | --- | --- |
| GPU6短验全链 | 277.771秒／.077159GPUh | 265.817秒／.073838GPUh |
| 短验峰allocated | 35.4026GiB | 35.6392GiB |
| fresh UUID | `5d7014d4-2ed0-4279-99c7-f09ca45a7afe` | `d470924f-1d1c-4006-9502-658cf618a7df` |
| 0步对all末token relative L2／maxAbs | .01171816／.3125 | .01080862／.3125 |
| tmux | `S4_QK64_g4_20261106` | `S4_VO64_g5_20261106` |
| wrapper／controller／worker PID | 4035117／4035139／4035762 | 4039346／4039360／4039689 |
| 正式step1／2秒数 | 39.9531／41.8986 | 35.6879／37.7625 |

两次正式更新均为groups0/1、LR5e-7／4.99e-7，72投影梯度全部有限非零；首步早／中／晚FP32及BF16均有真实变化。详细进程start_ticks、实际命令、数据曝光及峰值保存在各臂`startup_verification.json`。每臂正式日志为根下`qk_driver.log`／`vo_driver.log`与各自`training.log`；已停止主动轮询，后台controller继续约定64→独立导出→fresh→完整八项。本段为历史启动证据；完整终态以本文结果章节及各臂result.json为准。

两次短验共543.588秒／.150996706GPUh。两个初态末tokentop1均未变，BF16核差值只作边界记录。短验各自真实2/9峰组、完整冻结补集、独立named-state读取／导出、fresh3927字段／finite／位宽均通过，各stage及controller退出0，原件在`validation_qk/result.json`和`validation_vo/result.json`。审核后已删除6份无后继短验权重／运行态／训练态，实际释放12,198,682,624B allocated；两个`cleanup.json`保精确路径与用途核查，原指标／状态hash／配置／日志／元数据、正式／父／旧模型保护。
