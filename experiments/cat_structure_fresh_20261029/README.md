# 从原始BF16重新生成CAT A/B完整基座

本入口独立于正在运行的[同历史after_o结构对照](../cat_structure_full_20261029/README.md)。运行输入只有指定的原始HF、选定A/B、全新输出目录和单张GPU；不读取历史模型权重、历史`cat_runtime_state.pt`、旧UUID或旧`launch.json`。固定配方直接写在`recipe.py`，唯一原生运行主体为`result/qwen3_8b_mmlu65_20261004/qwen_native_category_rotation/source_snapshot`（N）。公共根目录缺少本配方的类别旋转与H252支持，不能替换它。

当前实现针对Qwen3-8B：36层、hidden4096、intermediate12288，每类完整36个投影按原CAT M36 pooled BSQ entropy拟合。A gate为D32/K32/S2，B为D64/K64/S2；均asymmetric encoder128/decoder128/R0。其余六类保持原S2 symmetric H128/R0：q/k K38、v/up K32、o/down K48，D均32。保护、two-sided旋转、归一化及损失沿原配方；q/k使用full rotation（类别block0），其余block32。asymmetric路径没有声称使用whole-fused。

每stage10000更新；所有类别实际执行`D×batch=262144`，仅B gate用4096，其余8192。base seed31，原派生shuffle为q31/k100031/v200031/o300031/gate400031/up500031/down600031。每类后原生500步`remaining_lora_prefix_decoder`，B4/max1024、dynamic padding、decoder LR1e-5、未来dense rank12/alpha24/dropout.1及LR1e-4，原生融合后保存，没有外置持久LoRA。动态padding允许实际batch长度小于1024。

运行始终声明全部七类别：

1. prefix进程只从CLI给定HF初始化。原生重新生成Alpaca train512×256/seed31 token及全部252投影激活统计；不加载旧校准缓存。依次压缩q/k/v/o，各S2×10K及500恢复，未来dense LoRA的七类范围保持。原生完整`after_o_proj`类别边界保存返回后，记录真实UUID/metadata/runtime SHA和权重stat，再受控结束进程。
2. 父入口等待prefix exit0，核其真实保存记录，只接受当前输出目录`prefix/native/.../after_o_proj`。随后创建独立suffix Python进程，原生再次seed31，按完整相同identity严格加载本次新after_o与本次新runtime。按所选gate结构、该分支live up、live down继续各S2×10K与500恢复；不会复制另一分支或旧成品的后继权重。不是不中断的RNG/optimizer exact resume。

`run.py`只管理这两个原生进程和落盘状态；`worker.py`适配原生fit batch/runtime identity/保存事件，不另写训练循环，不缩短正式日程，不使用short callback。`fresh_support.py`核HF真实配置、tokenizer与safetensors头、限制后缀来源；不加载HF tensor做CPU验证。旧A/B和N源码、运行pins均不修改。

调用者先激活`bitvae`。CPU检查也须由tmux/nohup后台运行，例如：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -B -u experiments/cat_structure_fresh_20261029/validate_cpu.py \
  --base_model_path /home/star/models/Qwen/Qwen3-8B \
  --output_dir /data/star/VAELLM/cat_structure_fresh_20261029/cpu_validation
```

选定结构、复用已有GPU证据并补齐新路径的必要验证后，按当前目标已有授权后台执行下面的唯一全链入口；`GPU_INDEX`须替换为当时可用的已授权GPU编号：

```bash
python -B -u experiments/cat_structure_fresh_20261029/run.py \
  --base_model_path /home/star/models/Qwen/Qwen3-8B \
  --output_dir /data/star/VAELLM/cat_structure_fresh_20261029/fresh_A \
  --arm A --gpu GPU_INDEX
```

B只改`--arm B`和全新输出目录。入口启动前核单卡空闲，拒绝已有输出，不等待占卡、不自动重试。caller PID、实际命令、环境生效条件、源码SHA、HF身份、两子进程PID/exit、原生argv、逐类生效batch与保存UUID归入`request.json`、`run.json`及`prefix/suffix.json`；stdout/stderr为`prefix/suffix.log`，退出结果为对应`.exit`。原生checkpoint仍保存到各进程的`native`目录。

CPU对照只读原`formal_k38_gate_s4_seq500/run.sh`及一个真实历史after_o的metadata，独立列为离线证据。原模板gate S4/K16改为本次S2及A/B几何；其它六类resolved spec和七类恢复/数据配置必须一致。CPU检查从原生loader实际生成一次新Alpaca token，记录真实train来源、fingerprint及token SHA；恢复五源为原train-only mix，复用既有来源记录，不重复全22GB数据审计。测试只核解析、类别batch/seed、重启identity与保存边界判定；没有构造假模型、假after_o或空权重文件来声称原生save成功。

本轮只做实现和CPU验证。新prefix真实模型前向、激活统计、完整保存与新两进程续接，以及最终成品保存重载尚未经过本入口GPU验证。没有新压缩模型、实测logical/packed位宽、完整模型存储或完整九项质量结果；不能据CPU通过宣称达到logical≤2.6、九项下降严格<0.9pp或总迁移目标完成。其它模型层数/投影几何、分组选择及相同流程迁移均未验证。

实际CPU证据在`/data/star/VAELLM/cat_structure_fresh_20261029/cpu_validation/validation.json`和同结果根的`cpu_validation.log`。16个合同用例均有通过证据：14项未受影响检查通过，另两臂实际native builder入参对照复核通过。初次真实HF/数据检查用时8.474秒，HF共399个tensor／16,381,470,720 tensor bytes，仅查头/stat；Alpaca train真实52,002行、fingerprint `c8befa91042914e8`，新512×256 token SHA为`6f6f73ca44838dbd61e01bcd240d5a03c2a45266b63102fc582045f174d55475`。CUDA未初始化，历史权重/runtime读取审计为空。

窄审过程中曾将checkpoint metadata的`target_layers=None`误当作运行时parser结果，导致新增测试的错误断言失败；实际N `parse_target_layers('all')`返回`'all'`，两臂原生builder等式本身通过。已纠正该测试，保留直接复用原生parse函数及有效对照，原失败和受影响复核日志一起归入核心日志。原HF和Alpaca证据未因此重复生成。Astra核心窄审未发现其它实质阻断；此结论仍不替代上述待做GPU验证。本轮无新增算法或质量经验，不修改中央文档和既有实验。
