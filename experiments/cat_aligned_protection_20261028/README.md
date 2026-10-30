# C/G 对齐恢复：保护参数分支与 C1 组装

本目录接续[对齐恢复链](../cat_aligned_recovery_20261028/README.md)的真实第02阶段，独立实现03a、03b及04。每个 profile 只接受自己的 `02_online_opd32/native_final/final_model` 和 `native_audit.json` verified；缺少实际父模型时立即拒绝，不用09、旧e3或另一profile替代。

## 固定配方与数据

| 分支 | 原配方与实际新预算 | M 输入 | QA 输入与顺序 |
| --- | --- | --- | --- |
| `scale32`／03a | 旧 `qwen_open_mcqa_scale_coverage64_20261014/B_med64` 的实际前32组；新计划确为32更新 | 每组原aux前16＋既定Med连续16；512 auxiliary_train＋512 Med train各一次 | 原N32的32组各一次：Qasper32、NarrativeQA32、Hotpot192 |
| `q64`／03b | 旧 `qwen_n32_protected_qat_s64_20261014/B`；连续64更新 | 原auxiliary_train1024重复两遍 | 原N32组0..31连续两遍；第33步不重置Adam |

复用原 `nq32_bindings` 所绑定的 NarrativeQA train story、prompt、hash命名和顺序，仅原QA组0/1/2/3/16/17/18/19的槽36..39被替换，前四个Qasper槽保留。各组共M32＋QA8。Qasper／NarrativeQA生成上限128，Hotpot32；每步由当前本臂学生原生greedy生成，保留真实EOS和完整prefix，再由恢复全部252原HF投影的full-F教师给出全151936词表KL。旧QA teacher-forcing目标不加载到loss。

M只从 `cov64_targets` 核验过的真实raw-F缓存复用，`.5F+.5gold`仅混合一次。实际学生绑定时还须核HF、tokenizer、有效config及147个非投影张量与full-F载体一致；full-F载体不充当学生或R教师。

R只复用 `shared_retention/retention_dataset.pt` 的原train七任务各128文档、全部候选分支及32组每任务4文档。**C/G各自由自己的真实新02冻结初态生成一次概率缓存，供该profile的两臂共用。** BF16 logits→FP32全151936词表log_softmax，共15462个continuation token，raw概率载荷9,396,937,728 bytes/profile；记录完整初末state、input、payload SHA、逐张量SHA、归一化回读及零HF恢复。源码不调用会加载旧R概率的顶层旧 `load_inputs`。

## 数值、状态与原生出口

两臂均seed137；全部1512个D叶和252个外置保护叶共同clip1.5，AdamW(.9,.95)、eps1e-8、WD0。D LR1e-6，scale LR1e-3／q LR.03，均乘原 `1-(step-1)/500`，不按32／64重新缩放日程。M／QA／R三域各权重.5且不再归一化；R按task、doc、branch、实际continuation token依次取均值。1659个其余原生字段冻结，D和保护参数元素数从实际新02读取。

scale训练FP32 logdelta，从0开始；先 `s_eff=BF16(s0.float()*exp(delta))`，再 `rows=BF16(qsigned.float()*s_eff.float())`。q训练FP32有符号坐标，初始化为 `uint8.view(int8).float()`，clamp[-128,127]、ties-even round STE、固定原BF16 scale，无trust或±1限制。短两步没有硬q翻转仍是合法结果。

学生使用现有full-RHT snapshot、FA2、FP32 RoPE及group8原生缓存。生成时临时安装原生BF16 D和保护值，生成后清投影／KV缓存，精确恢复FP32参数、梯度和非空Adam的对象及数值。末态只保存实际BF16 D＋BF16 scale／uint8 q，不持久化master、Adam或RNG；再独立strict重载3423字段、核504个bank、实际几何与logical≤2.6、rank0。不能调用旧硬编码2.5的底层组装保存函数。

04以本profile的03a为base，03b为donor，完整替换全部144个Q/K/V/O投影；108个MLP来自03a。每个投影的D、bank、q、scale、indices、rotation一体迁移，147个非投影张量exact。`assemble.py`复用成熟 `validate_sources/_expected_state/_replace_modules`，经本目录 `deployment.save_and_verify` 保存新UUID及真实两父账本，C/G不交叉。

## 入口与后续复用

调用者先激活 `bitvae`，核对Python，cwd使用：

```text
/home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01
```

以下为执行入口说明。各次执行用tmux／nohup、独立落盘日志及已分配GPU；当前没有启动这些GPU步骤。先设置 `VAELLM_ALIGNED_PROFILE=G` 或 `C`，GPU步骤另设置 `VAELLM_ALIGNED_GPU`。

```bash
# 无学生绑定的真实材料检查，CUDA隐藏；不写正式prepared。
CUDA_VISIBLE_DEVICES='' python -B /home/star/VAELLM/experiments/cat_aligned_protection_20261028/prepare.py --materials-only

# 真实02原生审计完成后，生成该profile唯一初态R缓存。
CUDA_VISIBLE_DEVICES="$VAELLM_ALIGNED_GPU" python -B -u /home/star/VAELLM/experiments/cat_aligned_protection_20261028/cache.py --gpu "$VAELLM_ALIGNED_GPU"
CUDA_VISIBLE_DEVICES='' python -B /home/star/VAELLM/experiments/cat_aligned_protection_20261028/prepare.py

# G优先：每臂各一次真正两更新，原QA组[0,19]、R组[17,28]。
CUDA_VISIBLE_DEVICES="$VAELLM_ALIGNED_GPU" python -B -u /home/star/VAELLM/experiments/cat_aligned_protection_20261028/run.py \
  --prepared "/data/star/VAELLM/cat_aligned_recovery_20261028/$VAELLM_ALIGNED_PROFILE/03_protection/prepared.json" \
  --arm scale32 --phase short2 --gpu "$VAELLM_ALIGNED_GPU" --timeout-seconds 7200
```

`q64`短验使用同入口，仅把 `--arm` 改为 `q64`。短验包括非空1764叶Adam往返、冻结、cache清理和独立原生导出；正式 `run_scale32.sh`／`run_q64.sh`检查同臂短验证据。C可显式传 `--short-evidence <G同臂short2/report.json>` 复用G的接线证据，但必须以两份**实际新02 prepared**证明C各叶shape被G覆盖、方法及输入协议相同；旧09几何记录不能替代此检查。

正式末态完成、`formal/run.exit=0`后分别执行 `export.py --report <03_protection>/<arm>/formal/report.json --output-dir <03_protection>/<arm>/native_final/final_model`。两臂原生核证齐全才执行 `assemble.py`，04输出在 `<profile>/04_C1/final_model` 和 `assembly_report.json`。没有等待前驱、轮询、重试或自动下一阶段调度器。

后续05/06可只读复用以下接口，不修改本目录或借用03的实际02缓存：

- `data.load_materials()` → `(recovery, retention_archive, materials)`，提供真实输入与一次混合M目标。
- `native.load_student(config)`、`deployment.save_and_verify(...)`：显式2.6/rank0原生加载／保存核证。
- `retention_cache.build_native(...)`／`load_native(...)`：调用者先核真实父模型，再显式给 `directory, meta, source_audit` binding、`cache_dir, profile, schema`。03专用 `build/load` wrapper仍固定验证02；通用核可绑定真实04的独立初态，缓存记录不混源。

## 2026-10-28 CPU证据与边界

真实来源独立检查的唯一材料为 `/data/star/VAELLM/cat_aligned_recovery_20261028/protection_cpu_materials.json`，核心运行日志为同目录 `protection_cpu_materials.log`。当前已完成真实材料与一次M目标、N32槽／组序、256个QA的768次EOS／上限／完整prefix重放、scale先舍入再乘、q clamp／ties-even STE／uint8序列化、实际入口导入及全部Python AST／shell语法检查。C/G09仅用于动态1512D＋252保护／1659冻结及147非投影的CPU可行性核查，明确不是新02父。

原失败 `KeyError: case_ids` 来自覆盖manifest使用 `M_case_ids` 而训练group使用完整 `case_ids`，已按真实schema修复。随后源码绑定排除了torch动态模块的伪相对 `__file__`；旧09审计没有新阶段的weight-stat字段，改为核其已被真实01 prepared绑定的权重与元数据。失败原因及实际退出保留在同一核心日志；只删除已被当前验证替代的CPU JSON。

**新02尚未就绪，正式prepared、新R概率、GPU短验、03训练／原生导出与04组装均未执行。** CPU材料和只读代码审查不证明GPU往返、资源峰值、收敛或九项精度；GPU验证留待真实02，优先G两臂的两更新，再根据实际shape和路径证据决定C复用范围。参考已有[数值与checkpoint经验](../../docs/experience/lessons/checkpoint_lifecycle.md)、[编码STE经验](../../docs/experience/lessons/liftquant_recovery.md)及[KD与数据边界](../../docs/experience/lessons/kd_and_data.md)，本次不据CPU检查新增训练收益结论，暂无新增技术经验。

## 10月29日：G新02初态R已生成，CPU准备因配置比较失败

本节替代上述“新02尚未就绪”状态。[C／G02已完整32步并独立原生核证](../cat_aligned_opd_20261028/README.md#10月29日正式32与独立原生导出完成)。本次仅对G UUID `38a7461e-a2f1-4586-80e2-b99883ef2628`实际生成新的初态R896，未使用旧R；同日用户已调整研究优先方向，后续安排以[恢复主入口的新指示](../cat_aligned_recovery_20261028/README.md#10月29日五个成品读出与可迁移流程的下一步)为准，不自动启动03短验／正式训练或整批评测。

在调用者激活的`bitvae`、原snapshot cwd、实核空闲的GPU0上，tmux `cat_aligned_G_R896_GPU0_20261029`执行`cache.py --gpu 0`，worker340768／start7883804；完成后同shell隐藏CUDA执行`prepare.py`。准确命令、source审计SHA、进程身份与退出位置见 `/data/star/VAELLM/cat_aligned_recovery_20261028/G/03_protection/R896_launch_20261029.json`。核心日志为同目录`cache.log`／`prepare.log`，实际退出为`cache.exit=0`、`prepare.exit=1`、`cache_prepare.exit=1`；没有重试、算法修改或新训练。

`R896/cache.json`已verified：真实七任务train共896篇、全部候选分支、15462 continuation token、原生BF16 logits转FP32全151936词表log_softmax，零HF投影恢复，完整初末state一致、教师释放、保存及全部target-record SHA／归一化读回通过。实际`targets.pt`为9,397,767,115 B（raw概率9,396,937,728 B），SHA `a79c82b98a7aa35ef155ca1f37f92929cdf0c54cfa4399976ecf6d660452a80b`；manifest SHA为`1681626fc4e152ecb54293098b1ec69d38fc9836a61e5e9f85c7ba93c278fec0`。本次缓存峰值allocated72,534,277,120 B／reserved76,183,240,704 B，GPU任务结束后GPU0已释放。此缓存保留供该实际来源的后续准备，不因CPU失败重新生成。

CPU准备失败在`prepare.py:175`调用`cat_aligned_opd_20261028/bindings.py:126::teacher_identity`，报`Student and full-F carrier have different effective model configurations.`。只读比较发现，移除该函数原已排除的`_name_or_path`／`transformers_version`后，JSON仅两字段不同：旧full-F载体`pad_token_id=151643`／`use_cache=False`，新G02缺少`pad_token_id`且`use_cache=True`。这是当前实现的原始JSON相等检查触发原因；是否属于实际执行语义差异须进一步核实，不能直接忽略字段或改模型配置绕过。当前没有生成`prepared.json`，03短验、正式训练和04仍未执行；保留失败日志和有效R缓存，交实现诊断处理。无新增精度结论。

## 10月29日：运行配置语义修复，G CPU准备已闭合

本节替代上节的CPU准备失败状态。[03 bindings](bindings.py)现在在本模块实现`teacher_identity`，复用既有`assemble_runtime_categories._validate_runtime_configs`：只接受bool的`use_cache`及None／原tokenizer pad ID，双方其余保存配置严格相同，同HF路径／metadata／权重身份和147个非投影张量exact检查保持。没有改旧02/opd bindings、原[prepare.py](prepare.py)、模型配置或缓存；最终03 bindings SHA为`5647f9a602b8c1a51ad8f3fa23e6812c11c09358473677c7931505e61977102a`。

实际canonical v6加载器先从共同原HF配置构造模型，再装checkpoint tensor，不读取成品`config.json`作为构造配置。共同HF的有效pad为None，因此M/R手工batch以0填充，显式mask及position_ids保留真实token位置；M/R及QA前向均显式`use_cache=False`。greedy生成使用独立显式GenerationConfig、`pad_token_id=151643`／`use_cache=True`及`use_model_defaults=False`。差异属于保存的运行状态，原生FA2／FP32RoPE仍由既有加载路径核验。诊断中额外加入的“原HF对saved全字典相等”会被Transformers的`_attn_implementation_autoset`构造标记触发，因不属于实际合同已移除，没有把它加入忽略白名单。

12项既有runtime配置接受／拒绝测试通过；随后原入口在真实G02及现有R896上CPU完成，exit0、74.7588秒。`/data/star/VAELLM/cat_aligned_recovery_20261028/G/03_protection/prepared.json`实际产生，SHA为`07d526facc2aba6189b387c6b778658d84f2d1760a731d55ed31bfe0afd64c8d`，原`read_prepared`再次核完整来源及全部代码／材料绑定通过。核对147个非投影张量、256个QA／768次完整prefix重放、source完整state及R payload与全部target records；R manifest保持`1681626fc4e152ecb54293098b1ec69d38fc9836a61e5e9f85c7ba93c278fec0`，没有重建或改写概率。完整差分、退出及验证见同目录`prepare_runtime_validation.json`和唯一核心`prepare_runtime.log`；原失败日志／launch历史仍保留。

当前状态为`CPU_prepared_GPU_unvalidated`，CUDA未初始化；03短验、正式32／64训练与04均未启动。本次仅闭合cache＋prepare，复用既有配置／checkpoint经验，无新增精度或算法结论。三个已归并的自有退出散文件已删除。

## 10月29日：G两臂short2完成更新，独立导出因bank计数失败

用户随后明确要求G09后恢复并行，已重新授权从真实G02独立执行03a／03b；本节替代上面的“短验未启动”和此前暂缓运行声明。此次立即范围为G02完整九项锚、G02的03a／03b及另一路G09直接OPD；不自动启动G04、C或另开评测。实际学生仍为G02 `38a7461e-a2f1-4586-80e2-b99883ef2628`，prepared SHA仍为`07d526facc2aba6189b387c6b778658d84f2d1760a731d55ed31bfe0afd64c8d`，原R896没有重建或修改。

调用者激活`bitvae`并核Python后，在上述原snapshot cwd、实核空闲的GPU0／1分别运行完整short2，原组`[0,19]`、R组`[17,28]`、M32及最长32K QA条件均保留。准确命令、环境、GPU UUID、PID／start_ticks及终态在各臂`short2_launch.json`，stdout／stderr在各臂`short2.log`，外层`short2.exit`和内部`short2/run.exit`均为1。

| 臂 | GPU／tmux | shell PID／start_ticks | worker PID／start_ticks | 失败退出前实际秒 |
| --- | --- | --- | --- | ---: |
| scale32 | 0／`cat_aligned_G_scale32_short2_GPU0_20261029` | 363106／8747785 | 363109／8747785 | 815.5436440499907 |
| q64 | 1／`cat_aligned_G_q64_short2_GPU1_20261029` | 363113／8747800 | 363116／8747800 | 835.6524388830003 |

两臂均真实完成Adam1→2，第二更新完成时间分别751.3067389929929／761.653273746997秒，日志保留完整M、8QA及7R反向、原生生成状态往返和连续参数身份；最长实际QA反向输入32565 token。失败发生在`run.py:131`的独立export调用：`export.py:70 → deployment.py:25`报`Incomplete original252/504bank/3423 inventory.`，因此**短验最终失败，正式32／64尚未启动**，不能把两更新完成写成原生出口通过。

只读源码定位到`persistent_inventory`返回的`banks`是252个投影的字典，每个value内含两个bank，而`deployment.py`误用`len(banks)==504`；同函数audit中的`code_bank_count=len(banks)`也存在计数口径问题。root已接手实际bank计数修复及最小导出补验；本运行代理没有修改代码、放宽检查、重启或启动正式，GPU0／1已自然释放。

`run.py`的异常处理已将各`short2/report.json`覆盖为`failed`：仍保留source UUID、prepared路径／SHA、161份原implementation绑定、contract、组序、error／traceback，但**不再包含`training`、`endpoint`或`native_roundtrip`字段**。旧`deployment.py`绑定SHA为`18743add273af8efe35363630a6722b64340e6d298cf8f4ef12ebeb55b64ad1c`，prepared及报告中的旧绑定保持，不能冒称修复后代码已经验证。详细实际训练证据在原核心日志，保存的两个原生compact也均保留：

| 保留的实际compact（相对`/data/star/VAELLM/cat_aligned_recovery_20261028/G/03_protection/`） | bytes | SHA256 |
| --- | ---: | --- |
| `scale32/short2/protected_endpoint.pt` | 10206238 | `447269402a54d2bd8707df44a5137f6deff5d3b80136be2585d2dda39cf13c56` |
| `q64/short2/protected_endpoint.pt` | 128128222 | `a07aab8374d54cf8e0bfa967ccdce930d700c1fa56a0123d64411a00df1e1d6c` |

现无可用short2 native audit，不满足正式入口。后续只在实际短验补验及绑定闭合后从真实G02 fresh启动原`run_scale32.sh`／`run_q64.sh`；正式终态exit0后才可用原`export.py --report <arm>/formal/report.json --output-dir <arm>/native_final/final_model`独立导出。本次没有清理任何失败证据、旧绑定或compact，也没有新增精度结论。

## 10月29日：修正嵌套bank计数后的独立复验

root核实`persistent_inventory`已逐投影检查六个decoder字段／两个bank后，将部署层的总数改为`sum(len(values) for values in banks.values())`，仍要求252投影、504bank、3423状态；审计报告采用同一真实总数。未降低原生几何、冻结、存储或fresh strict条件。同时修正`run.py`异常分支：导出失败时保留已经生成的训练和原生往返报告，整体仍标失败，不再丢失完成部分的证据。

旧失败两臂的目录、核心日志、退出及launch和原prepared已归到 `/data/star/VAELLM/cat_aligned_recovery_20261028/G/03_protection/export_count_failure/`，内部保持原相对路径，另保留原`deployment.py`／`run.py`。精确旧路径→归档路径及SHA见该目录`archive.json`；上节两份compact当前路径均需加这个目录前缀。没有改写旧失败状态或补造被覆盖的成功报告。

因所需完整短验报告已丢失，修复后明确重做一次两更新短验；不追加完整训练对照、不更换数据或降低长度，不重新生成R。原CPU准备重新执行74.70秒、exit0，新prepared SHA `67e1a907f46c3c1e0097999635b5d0eb018b5e8c998df95e9f29858cba82ad30`；核心 `prepare_bank_count.log`。当前G02评测与G09直接OPD只引用未改的03 bindings，不依赖这两处修复源码。

GPU0／1实核空闲后已分别后台启动：`G03_scale32_bankfix_GPU0_20261029`（shell380506）、`G03_q64_bankfix_GPU1_20261029`（shell380517）。各新`short2_launch.json`保存准确调用者环境、命令、PID和prepared SHA；新核心仍为各臂`short2.log`，旧文件已归档。该有限shell只在short2真实exit0后fresh调用对应正式32／64脚本，正式日志为`formal.log`、外退出`formal.exit`；没有重试循环或自动评测。此节记录修复及复验启动，尚不声称原生导出成功或正式已启动。

随后两臂修复短验均完整exit0，实际两更新及原生往返分别800.05／812.02秒；独立导出均verified，3423字段fresh strict exact、真实504bank。短验UUID分别为`f898ba43-06e5-4e53-afd8-161b1bc9df6b`和`df07c6dd-76d1-4786-b820-dcfd28c2de49`，仅作实现验证，不用于正式初态。原有限shell已从相同G02 fresh启动scale32／q64正式训练；首个真实Adam更新分别在431.12／434.07秒完成，原M32、QA8和七项R全部参与。实际formal PID／start_ticks／命令与有限进度确认已写回各唯一`short2_launch.json.formal_progress_confirmation`，未另建监控。当前正式训练继续，未完成端点或九项，不自动接G04。

正式两臂各确认连续两次更新后，已按收尾授权清理短验权重：两个成功短验的完整 `.bin`、重复tokenizer、成功及失败的四份compact，共18文件、实际释放9873977344B。正式只读取保留的短验报告／audit并从G02加载，不消费这些权重；删除前核无活动FD，原G02、R896、两条正式运行及全部原始指标／核心日志／metadata／源码归档保持。精确路径及大小见`short2_cleanup.json`；前文compact和短验模型路径现是历史身份，不再可加载。无需因清理重做验证。

15:20前，scale32正式32更新已完成，`formal/report.json`为`aligned_protection_formal_complete`，实际7906.2757秒，内外退出均0。随后用原`export.py`独立导出，exit0；新成品UUID `6a446110-a963-45a5-9847-8458f5d88c40`，3423字段保存／fresh strict exact、504bank、1659冻结字段及原G02父源均核齐。实际logical2.5295091711956523／packed2.6391813858695654，整模tensor4781328384B，rank0。成品位于`scale32/native_final/final_model`，对应`scale32/native_audit.json` SHA为`29b7c6381aec24a045fb8e8ee643dc5a8a2251739cb773b7ade232121bc2df44`；精确命令与退出归`scale32/export_launch.json`，核心`export.log`。该模型保留用于既定完整九项，评测接续和最终质量统一归[并行恢复正文](../g09_recovery_20261029/README.md)。q64仍为原PID392770／GPU1、已完成34次更新，尚无正式末点；本次未启动04或修改任何活动训练依赖。
