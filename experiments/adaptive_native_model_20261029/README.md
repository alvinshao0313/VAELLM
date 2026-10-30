# 自适应短码流的独立完整原生模型格式

本目录是10月29日的隔离接口原型，当前代码采用N runtime与`vaellm_adaptive_native_model_v2`，支持native／compact／pending_dense三份互斥库存。早期v1/S曾将真实L18 `weighted_dynamic3`接入G09骨干；**该历史G09骨干与compact拟合目标不同，仅是格式／训练链验证**，旧报告保留原运行身份。后文另有真实A/B raw归档，以及B的144native＋36个D64 compact＋72dense恢复／保存／next-up短链。它们不代表B结构胜出、第三阶／动态路由有效或完整恢复／九项通过。公共v6、N/public源码及其它活动实验不改。

依据是[分组拟合独立复核](../adaptive_grouped_fit_20261029/README.md#10月29日独立研究复核并行语义规格扩展与下一项完整对照)、[HF桥接](../adaptive_native_bridge_20261029/README.md)、[checkpoint经验](../../docs/experience/lessons/checkpoint_lifecycle.md)及[存储口径经验](../../docs/experience/lessons/compression_choices.md)。本轮不改变路由分配、输入二阶矩代理或Rice控制预算；这些属于选定CAT结构后的同父算法对照。

## 文件职责和持久契约

- `spec.py`：按实际decoder `in_dim/out_dim/hidden_dim`、保护后矩阵几何和transpose解析每阶K、D、H、N、活动行数与`ceil(K/8)`；校验码padding、路由、保护q/scale/index与sign，拒绝不支持的保护拓扑，不补零或静默改输入。不同规格／库存的证据范围按后文各次真实运行区分。
- `codec.py`：读取真实保留的旧compact二进制与完整zlib长度frame；不沿用旧loader写死的末维6。旧输入文件只是导入来源，完整成品不依赖该文件。
- `consumer.py`：可训练`CompactRVQLinear`。只解码活动行，按stage顺序BF16加，不活动stage连bias/融合mean一起跳过。FP32 master只在显式启用decoder训练时创建，BF16计算可导；参数版本、load和train/eval切换清缓存。rotation sign只持久一份，路由、保护解码值及dense缓存可推导且不落盘。
- `checkpoint.py`：v2独立构造Qwen3，按完整互斥native／compact／pending_dense清单重建并strict load；单份`pytorch_model.bin`包含所有有效权重及真实merged dense，另有自身config／generation／metadata。没有历史source副本、被替换旧码／decoder或额外dense投影副本。中间dense状态如实计费，只有pending_dense为空才施加final≤2.6检查；FP32训练master与Adam另由专门训练checkpoint保存。
- `validate_cpu.py`：真实G09来源核证、原3423字段、单覆盖、3410非替换字段不变、实际保存、单独Python进程fresh全字段exact及有效权重读取路径检查。
- `validate_gpu.py`：唯一获分配GPU7的真实完整压缩模型短验入口；复用原辅助train M32第0/1组、一次`.5F+.5gold`目标，只更新compact的18个decoder叶，核两次Adam步、冻结补集、缓存、BF16部署、fresh真实M32分布。保留251 native的group8缓存、FA2与FP32 RoPE。

## 运行依赖与真实存储口径

早期G09/v1验证使用`result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01`（S），包含full4032需要的固定252阶Hadamard族；公共rotation不支持该因子。初次独立fresh实际暴露入口依赖，当时隔离包显式选择S并在manifest绑定SHA。后续原包无活动依赖后，当前`__init__.py`显式改绑`qwen_native_category_rotation/source_snapshot`（N），保留导入冲突检查与新manifest实际路径/SHA，不替换`sys.modules`，不设S回退。N fitter按类别`runtime_cfg.weight_rotation_block_size`执行，保留现行类别旋转规则；旧S/v1报告不冒称覆盖新N链。独立模型载荷仍只读自身文件；N/public与snapshot均未修改。

账本区分logical Linear bits（主码按真实K×活动行，其它持久字段按实物）、packed Linear bits（按实际uint8行宽）、全部非投影、序列化开销和完整目录字节。长度frame含其真实header及压缩载荷；decoder/norm、保护q/scale/index、sign和全部格式/config字节分别有账。`budget_passed`沿当前协议只指logical Linear≤2.6；packed与完整目录口径不混写。非投影embeddings/head/norm真实保存并独立报告，不能只按compact的约20MB收费。

当前真实输入是`/data/star/VAELLM/adaptive_grouped_fit_20261029/grouped/endpoint_18/weighted_dynamic3.avrvqz`，SHA `d32286371be510ea87e86ed167f4c291a0083f19bea25675334044f14635e2aa`。完整父为G09 UUID `e8e5efd4-655e-462c-b6e0-fedf4e2e0346`，由其现有`bindings.source()`核验，不凭名称选择新父。

## 验证状态与复现入口

调用者先激活`bitvae`；所有验证通过tmux后台运行，独立输出不能覆盖。CPU关闭CUDA可见性，避免第三方import探测初始化任意GPU。GPU仅在主线程协调后运行，脚本本身不占卡等待。

```bash
CUDA_VISIBLE_DEVICES= python -B -u experiments/adaptive_native_model_20261029/validate_cpu.py \
  --output /data/star/VAELLM/adaptive_native_model_20261029/cpu03
CUDA_VISIBLE_DEVICES=7 python -B -u experiments/adaptive_native_model_20261029/validate_gpu.py \
  --cpu-result /data/star/VAELLM/adaptive_native_model_20261029/cpu03 \
  --output /data/star/VAELLM/adaptive_native_model_20261029/gpu01
```

CPU03已完成，真实PID426902、tmux`adaptive_native_CPU03_20261029`，49.8703秒/exit0。原3423字段加载exact，替换掉旧L18的13字段，其余3410字段（含全部147个非投影）不变；保存后3437字段在独立Python进程逐项strict exact，只请求读取新成品自身`pytorch_model.bin`，CUDA未初始化。CPU02已保存真实完整状态，但其独立fresh因遗漏snapshot入口失败，不能记作通过。

真实完整账本：Linear分母6,945,767,424权重；logical **2.5303907739943354 bpw**，packed **2.640062988668248 bpw**，主码padding **761,757,696 bits**；全模型tensor **4,782,093,810 B**，单一权重文件 **4,783,290,191 B**，CPU03完整目录 **4,785,015,272 B**（含1,725,081 B的manifest/config/generation文件）。logical≤2.6成立，不把packed也写成≤2.6；完整目录按同一Linear分母是5.511287643 bit/weight，包含实际embedding/head/norm及格式开销。

其中compact有效持久载荷20,356,466 B：实际码19,093,524、长度frame131,742、三个BF16 decoder63,936、保护q1,048,576/scale512/index2048、旋转sign16,128 B；原16KiB层文件header不嵌套复制，其必要规格进入完整manifest并按实际字节计费。N=1,540,096，长度1/2/3数量分别24,712/1,388,610/126,774。

GPU短验已完整结束，GPU7的tmux`adaptive_native_GPU7_M2_20261029`，shellPID427578/worker427583，**94.0418秒/exit0**。输出`/data/star/VAELLM/adaptive_native_model_20261029/gpu01`，唯一完整原始证据为`report.json`、核心日志`gpu01.log`。18个decoder叶全部得到有限FP32梯度，31968个master元素实际变化；两组真实M32的loss分别0.1516767740、0.0617251359，样本组不同，不能当作同批次质量提升。两次Adam步和1e-6/9.98e-7日程正确，缓存按更新重建，全部3419个冻结持久字段exact；BF16重构权重实际变化27,095,632项。

初态完整compact权重与已验证bridge逐项相同；18叶master导出BF16逐项相同。fresh模型重新预热251 native后，compact BF16权重和原M32组0完整四选项log分布与保存前exact，maxabs=0，完整预算不增。最终checkpoint UUID`f429ca34-9adf-47d8-aebf-fea3cdf1a7dd`，3437字段digest`b73ea81280407499a2da3cf311d26fed7ce9a433d655c71fba01a0d0dfcd8388`；完整目录实测4,785,023,227 B，manifest/config/generation共1,733,036 B。

随后另开CUDA隐藏的独立Python进程PID428156，更新后的3437部署字段再次strict exact，唯一有效权重读取是`gpu01/model/pytorch_model.bin`，FP32 RoPE保持；该报告已归并到`report.json.independent_process_fresh`。13:50:27核worker退出且GPU7显存归零，已通知主线程接后续评测，没有占卡进程。

整段累计峰值allocated **72,333,825,536 B**，reserved **76,594,282,496 B**，包含两轮完整251 native的group8预热，**不是训练阶段单独峰值**。预热各约2.35/1.88秒，不能把单层HF bridge的约23.59GB峰值替代这里的完整原生条件。

对真实251 native规格再核：q/k两stage各K38→packed40，gate各K34→packed40；v/up各K32→32，o及其余35个down各K48→48。本compact三stage K48→48。padding只按这些真实规格统计，既不写成其它实验的K24，也不把行存储宽40说成逻辑K40。

本轮证明了新格式、完整载荷、真实原生训练与独立消费者闭合；没有新增质量或速度优胜结论。下一次同父完整gate配对仍需等待CAT选结构并实际生成该结构的同源码/保护/旋转和公平Rice预算，不能沿用本历史拼接成品作算法对照。

## 完成后的最小产物收尾

全部验证完成、无活动消费者/打开FD后，按持续授权删除CPU02、CPU03和GPU01三份无后续用途的完整临时权重及重复配置；同时归并独立fresh和退出散文件。GPU01的有效config/generation/完整manifest、CPU03来源及原字段证据、失败CPU02报告/核心日志、最终GPU原始报告/核心日志、代码和原先保留的20MB D32 compact输入仍在。**`gpu01/model`现为已清理权重的格式证据目录，不能直接加载；没有把该历史混合原型保留为交付模型。**

精确路径、用途核查、删除前大小/SHA、实际allocated释放和删除后核验都在`gpu01/report.json.cleanup`。主要收尾实际释放 **14,353,383,424 B**，原始G09、HF、数据以及其它任务产物未触碰。新增经验仅为独立进程必须显式闭合既有native runtime依赖，已更新[checkpoint经验](../../docs/experience/lessons/checkpoint_lifecycle.md)。

## 同父 native 两阶前缀 importer：纯CPU接口闭合

这项接续[已完成的接口设计](../adaptive_grouped_fit_20261029/README.md#后续接口复核从真实native两阶导入而非再次读取旧compact)，新增`native_prefix.py`与`validate_native_prefix.py`。旧`spec.py/consumer.py/checkpoint.py/codec.py`及其已验证pins不改，不新增第三阶、训练或路由择优；不绑定G09层号作为算法选择规则。

`from_native(native_spec, linear_state, provenance)`在真实形状的单Linear宿主上调用原native builder、严格加载相对字段，用stage getters取得实际BF16 decoder/码规格/码，返回`bundle,lengths,resolved`。首版仅覆盖当前实际单part、两stage、同D、INT8输入保护，无bias/dense副本/LoRA/稀疏/输出保护/保护残差。码`[N,1,ceil(K/8)]`只去中轴并clone，K取真实decoder输入及码logical shape，D取decoder输出及`stage_codebook_dims`；保护三字段与两侧sign原样复制。不得从D32码重排出D64码。

当前decoder已经融合BSQ符号尺度与原stage mean/std，并经过恢复；直接复制其现值，不再次融合。原始统计未持久化，bundle normalization只记录`already_fused_in_current_native_decoder`、`not_persisted_and_not_recoverable`等来源标志，**没有伪填mean0/scale1**。实际getter的非持久`_q_scale_fused`均为False，不能由此推断参数未融合。

`native_prefix_v1`是明确的新单层`.pt` schema，保存bundle与resolved spec；consumer可直接消费bundle。原始N长度张量不落盘，只保存合法全2frame（29B header＋zlib6），加载由frame重建长度，不冒充旧L0-three-stage格式。`save_prefix/load_prefix/ledger`包含全部真实codes/decoder/fixed/frame，单列torch容器及metadata开销；该**单层文件**账本不替代上面的完整模型账本。未来fixed2控制可以隐含length2，本次测试frame费用不能擅自加入Rice公平cap。

实际运行由调用者激活`bitvae`，tmux`adaptive_native_prefix_CPU01_20261029`，shellPID431048，独立fresh PID431356；8.1723秒/exit0，CUDA始终未初始化。复现时换全新输出目录：

```bash
CUDA_VISIBLE_DEVICES= python -B -u experiments/adaptive_native_model_20261029/validate_native_prefix.py \
  --output /data/star/VAELLM/adaptive_native_model_20261029/native_prefix_cpu01
```

唯一原始报告为`/data/star/VAELLM/adaptive_native_model_20261029/native_prefix_cpu01/report.json`，核心日志为同级`native_prefix_cpu01.log`。来源仍为已核证G09 UUID`e8e5efd4-655e-462c-b6e0-fedf4e2e0346`；仅mmap读取其第一份up/gate/q/down实际投影的54,606,272 B字段，不读取C09，不构造完整8B模型。四种投影各13个native字段strict exact、抽取后重新pack的6个packed decoder字段exact；完整双bank及padding、所有BF16 decoder叶、保护/sign、全2frame与几何/transpose exact。

| 实际投影 | K / D | N | 码行B | 双stage padding bits | 数值范围 |
| --- | --- | ---: | ---: | ---: | --- |
| G09 L0 up | 32 / 32 | 1,548,288 | 4 | 0 | 真实首/中/末3行 |
| G09 L0 gate | 34 / 32 | 1,548,288 | 5 | 18,579,456 | 真实连续0–8192行，chunk8192＋尾1 |
| G09 L0 q | 38 / 32 | 516,096 | 5 | 2,064,384 | 真实首/中/末3行 |
| G09 L0 down | 48 / 32 | 1,540,096 | 6 | 0 | 真实首/中/末3行 |

数值范围均在逆旋转/保护回填前；**不是完整Linear原坐标函数、GPU路径或下游等价证据**。在这些真实行上，当前compact与原module getter抽出的相同独立stage操作，在FP32/BF16的两stage及和均逐项exact。gate原native packed与独立decoder之和的FP32 maxabs1.490116119e-8、relative-L2 6.578432097e-8，满足既定逐元素rtol2e-5/atol1e-6及relative-L2≤1e-4；BF16 maxabs0.0009765625、relative-L2 **0.005656476208（约0.566%）**、MSE2.378582077e-8，262,176个输出元素中170,682个改变（约65.10%）。这些百分比仅描述**8193行前缀运算范围**，不能称为lossless function。每stage差量及其余K的范围原值都在报告，没有用BF16微差加补偿、设置任意通过阈值或扩大GPU验证。

gate临时文件实际 **16,330,778 B**，其中tensor16,322,156 B、torch容器/metadata8622 B；码15,482,880 B，decoder35,456 B，保护/旋转等fixed803,392 B，全2frame428 B（原始两位流387,072 B）。主码logical105,283,584 bits、packed123,863,040 bits，padding18,579,456 bits。小文件独立fresh只读取其自身，全部tensor digest`fc955fc475ab4da8ec249599e8f737e9d9142072925db4c932e70f8dd6414bef`及所测compact输出digest均exact；原始长度不重复持久化。

该G09验证闭合了native→compact前缀和保存接口，不证明D64、全矩阵/GPU等价或质量收益。正式两臂必须共享这个新compact fixed2消费者，在其**实际部署前缀**重算stage3 residual、前缀误差和稀疏残差，单列native→独立decoder的初态差量；不能把旧native residual加偏移补偿，或将转换差量记成动态路由收益。新live父、gate逐坐标h和Rice公平cap仍等待CAT选定结构后建立。

本次fresh消费者已完成，主线程确认没有安排旧G09 K34前缀的后续消费者。已删除该16MB临时prefix，fresh及退出散文件归并到原始`report.json`；3项精确路径、删除前大小/SHA、无活动FD及删除后核验写在`report.json.cleanup`，实际allocated释放 **16,347,136 B**。只留源码、来源、完整原始指标与核心日志；报告中的文件大小/SHA是清理前实测，当前无可直接加载的生成prefix权重。没有因此新增GPU或额外后续任务。

## A/B 恢复前 raw36gate 保全（10月29日）

已安排后续同父adaptive／Rice公平对照，需要保存A/B各自CAT两阶刚完成、原500恢复尚未开始的真实gate前缀。原CAT采用`save_total_limit=2`，后续正常推进会自然prune早期gate `round_base`；本归档及时保全不可从500恢复后反推的原始状态，**不阻止原清理、不改活动配置，也不删除来源**。新增唯一执行文件`archive_cat_gate_prefixes.py`，原importer／consumer及其pins保持不变。

两个真实来源为`/data/star/VAELLM/cat_structure_full_20261029/{A,B}/native/home_star_models_Qwen_Qwen3-8B_20261029_134231/training_rounds/0004_gate_proj/round_base`，A UUID`94c8c25b-19dd-4c6d-852c-19f5b12c4d6b`，B UUID`7c0b1199-b9ff-46af-a1b7-c2f5b7dabf7f`。完整原metadata核得`round_base`、180压缩＋72dense，completed仅q/k/v/o，`extra_meta.distill_stage=null`，确为gate恢复前边界。

部分归档位于`/data/star/VAELLM/adaptive_native_model_20261029/cat_gate_prefixes/{A,B}/`，schema明确为`cat_gate_raw_prefix_archive_v1`，`independently_loadable_model=false`。每支仅持久以下必要实物：

- `gate_state.pt`：36个gate的全部468原native持久字段，每模块13项；逐字段clone为自有连续storage，保留原码／decoder／保护／旋转，不意外保留源mmap的整模型storage。
- `raw_checkpoint_meta.json`、`raw_config.json`：原文件字节副本，完整保留真实每模块spec、原UUID及恢复历史；不伪造新的native模型metadata。
- `cat_runtime_state.pt`：本支原raw runtime的逐字节副本，**每份22,729,698 B**，SHA逐支写入manifest，供已安排的原500恢复／继续live类别使用，不能用after_o runtime冒充raw实际进度／材料。
- `manifest.json`：来源路径／UUID／meta SHA／weight stat、字段shape/dtype/bytes、源gate与摘取digest、完整raw状态digest、文件SHA和明确重建范围。

| 归档 | 实际gate结构 | 468 tensor B | `gate_state.pt`实测B | 归档目录总B |
| --- | --- | ---: | ---: | ---: |
| A | asymmetric D32K32/H128R0，双stage | 476,068,608 | 476,226,186 | 499,900,190 |
| B | asymmetric D64K64/H128R0，双stage | 477,252,864 | 477,410,442 | 501,084,230 |

两份gate文件各含157,578 B序列化开销；表中目录总量另含完整原metadata/config、manifest和必要runtime。它们是部分原生状态归档，**不是独立整模、完整低比特预算或新算法成品**，不以这些小文件估算全模型位宽。

共同dense目标和其它骨干继续保留原`/data/star/VAELLM/qwen_recompression_20261026/C/initial_after_o`，UUID`b63bfe79-3889-40c6-bf66-1d002893f874`。其36个真实dense gate目标共3,623,878,656 B，digest`e3fb94c71e8c2a53a39f105a6af3ce250ac60cfe7d4770196d04cbb4feb04cbd`；未另复制该目标或两份约10.74GB原raw整模型。after_o的具体保留用途是未来同父adaptive／Rice目标与raw非gate重建补集，不能按一般旧基线自动清理。

两支各2091个非gate字段与after_o补集逐项dtype／shape／value exact。去掉after_o的36个dense gate weight，换入各自468字段，独立CPU进程只实际读取after_o＋两份小gate归档，分别重建2559张量状态；完整digest与原raw exact：A为`8254a2084fe57b6726705cb52def09dc1747df028fd8efc01a98ad05a2f6acbe`，B为`eb349a176ca4ba5c9c9f9a249e603ec465d3d6b4f27af5b5b813c3379bc51f1e`。子进程没有再读取两份原raw大权重，也没有构造完整模型。

raw config缺`pad_token_id`，after_o为151643；两份raw config原样保留。上述结论是**完整tensor state可重建**，不扩成全部运行／生成语义或optimizer/RNG exact-resume相等；原runtime保全也没有替代未来恢复入口的必要协议。

附带复用已有CPU接口，在每支L0代表gate核真实完整bank／13字段／spec／getter提取repack，原样消费其asymmetric D32K32或D64K64，并以真实首／中／末3行核FP32数学对应、BF16跨算子差量及相同独立getter→compact exact。此证据仅限这些向量行和规格，不是fullLinear／整模／GPU或第三阶验证；临时compact bundle未额外落盘。

bitvae调用者、CUDA隐藏、tmux`adaptive_cat_raw_gate_CPU_20261029`，shell484089／worker484101，独立fresh PID485199；总96.2027秒／exit0。唯一原始报告`/data/star/VAELLM/adaptive_native_model_20261029/cat_gate_prefixes/report.json`，核心日志为外侧同主题根下的`cat_gate_prefixes.log`；原件与必要归档全部保留。fresh及退出散文件已归并报告后清理，未改原CAT自然prune或其它活动任务。复现须使用新的输出目录：

```bash
CUDA_VISIBLE_DEVICES= python -B -u experiments/adaptive_native_model_20261029/archive_cat_gate_prefixes.py \
  --output /data/star/VAELLM/adaptive_native_model_20261029/cat_gate_prefixes
```

## A/B gate250／500 可变端点保全

“每类恢复周期评测、在预算内选择下一类入口”是新增候选，当前A/B固定500流程继续。本次仅为已安排的250／500只读配对保留会被`save_total_limit=2`自然prune的真实端点；没有训练、评测、选择优胜节点或修改活动代码。新增独立`archive_gate_selection.py`，输出`/data/star/VAELLM/cat_structure_full_20261029/selection_candidates/{A,B}/gate{250,500}`；不改原metadata中的`round_base_ref`。

| 节点 | 原training_step UUID | mutable文件B | 节点目录总B |
| --- | --- | ---: | ---: |
| A250 | `6b99b752-2ac9-49ac-8a56-cdae212890d8` | 70,518,914 | 71,344,155 |
| A500 | `ffef68e6-14af-4afd-ab8c-c69cb23ac85c` | 70,518,914 | 71,348,125 |
| B250 | `c2d2ca9e-d3a2-4a3d-941c-2f98607de6df` | 72,887,426 | 73,712,811 |
| B500 | `e8a08b0f-9f76-43a8-a870-408212cf5ca6` | 72,887,426 | 73,716,780 |

每节点只保留原`training_model_state.pt`、完整`checkpoint_meta.json`、`trainer_state.json`、既有归档的原`raw_config.json`字节及用途manifest，四节点合计 **290,121,871 B**，不含根原始报告和核心日志。manifest记录源stat／SHA、真实原raw parent、保留的raw归档／after_o绑定，以及1224个真实mutable字段的shape／dtype／逐张量SHA和整体digest；其中1080项decoder、144项LoRA，A tensor70,050,816 B，B tensor72,419,328 B。四份global_step确为250／500、max_steps500，source UUID、各自raw parent和180压缩＋72dense库存均核对。源文件逐字节copy并验证SHA，独立fresh只读新小包，全部文件、字段和digest exact；没有构造模型、读取整模权重或初始化CUDA。

本schema明确为`cat_gate_selection_partial_archive_v1`、`independently_loadable_model=false`。实际部署重建仍须用上节**同支raw36gate＋共同after_o**还原base，再沿原native／PEFT导出合并路径恢复mutable；本次没有验证merged导出、GPU／下游等价或optimizer续训。不复制10.7GB raw、dense目标、optimizer／scheduler／RNG。原raw runtime继续按原500恢复／live类别用途保护，不能拿after_o runtime替代；四节点小包另有明确的250／500只读配对消费者，不影响任何原CAT自然清理或G03／Muon主线。

调用者`bitvae`、CUDA隐藏，tmux`cat_gate_selection_archive_CPU_20261029`，worker549398／独立fresh549515，**6.3016秒／exit0**。唯一原始报告`/data/star/VAELLM/cat_structure_full_20261029/selection_candidates/report.json`，核心日志为同级`selection_candidates.log`；源码SHA绑定在报告。fresh散文件已归并报告后删除，没有另留PID／退出散文件。实际命令如下，复现必须使用全新输出目录：

```bash
CUDA_VISIBLE_DEVICES= python -B -u experiments/adaptive_native_model_20261029/archive_gate_selection.py \
  --output /data/star/VAELLM/cat_structure_full_20261029/selection_candidates
```

## N/v2 混合恢复、保存回训与真实 next-up 交接

本节是接线短验，使用真实raw B作为输入，**不表示B结构胜出**。新增`recovery.py`承接原`remaining_lora_prefix_decoder`下层stage／五源train数据／collator／独立HF teacher／`CanonicalCatSFTTrainer` loss和原优化器；先让原下层启用144native D＋72dense rank12 PEFT，再在PEFT之后把36compact的真实Parameter加入同一个decoder inventory。没有硬用会拒compact的旧remaining runner，没有修改N/public或任何其它活动依赖。

`checkpoint.py`当前v2按实物保存互斥native／compact／pending_dense，后者包含真实merge后的dense字段；完整文件没有历史source权重副本。`training_checkpoint.py::MixedCatTrainer`用明确的`vaellm_mixed_training_step_v1`保存全部FP32 native/compact D和dense LoRA、必要decoder buffers及实际参数manifest，仍由原HF Trainer保存Adam／scheduler／RNG／Trainer state。它不调用v6 step serializer、不只保存PEFT，也未实现或声称exact-resume。常规配置仍为250／500保存；本次只将触发间隔改1以覆盖step1保存后回训。

`prepare_handoff.py`的CPU阶段已于16:29完成，74.1500秒／exit0；完整144native＋36compact＋72dense的**2811部署字段**独立fresh exact，仅有效读取自身完整模型。PEFT后的训练视图有2955字段；真实1296 D叶＝864native＋432compact、144 LoRA叶，共1440个FP32可导叶，无遗漏／重复，D参数3,949,056、LoRA参数14,155,776。实际optimizer覆盖两组，base D LR1e-5、LoRA LR1e-4、WD0。经Astra核心窄审无实质阻断后才启动GPU。

CPU／恢复后中间边界均含72dense，Linear denominator6,945,767,424，logical **9.485916137695312**／packed **9.4992907980214 bpw**，全模型tensor **10,737,418,944 B**。`final_budget_applicable=false`，不能把含dense的中间状态写成≤2.6成品或因此拒绝正常交接。恢复后完整目录实测10,739,684,163 B；N runtime实际路径/SHA与三库存均保存在其v2 manifest。只有无pending_dense的最终成品才执行final≤2.6检查。

GPU0在16:52:14核空后启动，tmux`adaptive_mixed_handoff_GPU0_20261029`，shell523881／driver523903，核心`/data/star/VAELLM/adaptive_native_model_20261029/mixed_handoff_gpu01.log`，完整原始证据`mixed_handoff_gpu01/report.json`。总 **393.5239秒／exit0**，17:01后已核GPU0回到0MiB；只使用0，没有暂停或重启任何主线。已复用此前通过的CPU成品，没有因对话中断重复启动或重做CPU。

真实恢复保留**原500步日程**、B4/max1024、dynamic padding、gradient checkpointing、原teacher、top100/T1/prompt.3及原LR/warmup；callback只在第2次真实更新后停，不把max_steps改2。实际两批为4×456和4×1024，mask有效tokens分别`[454,27,152,317]`、`[70,524,296,1024]`。step1原warmup LR0使master无变化是正常现象；step2有11,025,428个master元素变化，其中compact1,211,585、dense-LoRA7,077,888。1440叶均有有限FP32梯度和连续Adam step1→2，冻结码／路由／保护／旋转／骨干／dense base字段exact。

step1实际保存后回到step2；两次保存都核master、非空Adam对象及其值、scheduler和RNG在保存前后不变。每个实际训练checkpoint的FP32 master文件72,958,206 B、Adam文件146,047,346 B，另有真实scheduler／RNG／Trainer state；无resume通过声明。父进程Torch累计峰allocated **40,906,334,208 B**／reserved **50,644,123,648 B**。

训练结束先按原helper释放Trainer训练引用、offload teacher，再按原规则merge72个dense PEFT并移除adapter，全部native/compact导出BF16、清cache、冻结、恢复use_cache。实际merge规则逐项exact，随后保存完整mixed UUID`33e4d18d-4594-4fc5-8e1c-6d69dcb6c70a`。独立子进程只读新文件，2811字段及冻结状态exact；两侧采用相同144native group8预热，使用真实首批相同输入比较最后token BF16 logits，**逐项exact／maxabs0**。对应digest`9319232f49f184adce3a0057cd0dc2617a42d5c14d02f077870c64bbe7c12343`。

从**该fresh模型**的36份实际live up调用未改N `train_group_vae_payload`＋`apply_group_vae_payload`，D32/K32、M36、B8192、input64、rotation32、shuffle500031及原gpu_resident_data、保护／类别统计规则保持。真实dense目标digest`91fcc82bd453aaf583333d6903ac5c683ab450b37d91808a5bf2078fad252dfe`，没有读旧after_o up或donor。两个stage都实际执行Adam step1→2并完成原生替换，gate全部参数／码／长度／保护／sign exact，库存成为**180native＋36compact＋36dense**。这里VAE只做2步机制检查，实际LR **.003／.0015是两步调度**，不是正式10K拟合的前两步轨迹。独立fresh/next-up子阶段156.3457秒，Torch峰allocated27,075,492,864 B／reserved27,833,401,344 B。

这只证明“混合prefix恢复2更新→中途保存回训→future LoRA融合→保存fresh→真实next-up拟合／替换”闭合；没有启动第三阶、稀疏consumer或动态路由，没有完成原500恢复、up/down500、正式10K拟合、最终216native＋36compact导出或完整九项。两条研究主线继续：更好压缩基座与现有基座上的可迁移恢复，本短链只提供两者衔接工程。

本次实际命令如下；`--check-only`只是新入口合同检查并复用原CPU证据，不重建／重存CPU模型。以下路径记录历史运行，收尾已删除其中验证用载荷；后续复现须从保留的raw归档及after_o准备全新CPU输出，再使用全新GPU输出目录。

```bash
CUDA_VISIBLE_DEVICES=0 python -B -u experiments/adaptive_native_model_20261029/run_handoff.py \
  --prepared /data/star/VAELLM/adaptive_native_model_20261029/mixed_handoff_cpu01 \
  --output /data/star/VAELLM/adaptive_native_model_20261029/mixed_handoff_gpu01
```

一次性CPU／两步部署中点、FP32 master、Adam/RNG、临时probe与重复散文件没有后续消费者，已按持续授权清理，实际移除文件合计 **21,949,678,478 B**，释放allocated **21,949,804,544 B**。精确删除路径、用途、删除前大小／SHA、无活动FD与删除后保护核验归并GPU原始报告的`cleanup`。`after_recovery/`现仅为无权重的格式证据目录，保留v2 metadata/config/generation；另留一份真实step1训练manifest，不可当作可加载成品或续训checkpoint。CPU报告增加清理注记，历史验证结果不改；运行时记录的CPU报告SHA仍是清理注记前的实际版本。

原A/B必要小归档、各自raw runtime、共同after_o、原始CPU/GPU报告、源码及核心日志全部保留，验证过保护路径仍存在。没有为“以后可能formal”保留validation-only中点，也没有新增监控、占卡或后续验证。

## 向量误差与规格化费用核心（10月29日）

新增`objective.py`与`allocation.py`，供既定整类动态对Rice使用；没有修改现有消费者、活动训练或码本。前者按实际保护／旋转／transpose映射构造live目标、输入二阶矩和BF16前缀误差，gate逐坐标加权；后者按每阶`K_s`分别记录逻辑位与`ceil(K_s/8)`物理码行，使用真实长度frame和调用方完整serializer测量费用，最多四轮取最后可行。decoder和额外持久字段按实际保存内容收费，不因零活动而免账，不假定旧16KiB头，也不宣称压缩成本的全局最优。异K只覆盖费用数学；现bundle规格不支持时明确拒绝，不冒称异K部署已通过。待闭环问题见[向量位宽经验](../../docs/experience/lessons/compression_choices.md#向量级位宽分配的待闭环问题2026-10-29)。

调用者已激活`bitvae`，CPU后台tmux `adaptive_objective_allocation_CPU_20261029`实际执行：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -B -m pytest -q -s experiments/adaptive_native_model_20261029/test_objective_allocation.py
```

实际`12 passed in 6.05s`、`pytest_exit=0`，进程与tmux已退出。核心日志`/data/star/VAELLM/adaptive_native_model_20261029/objective_allocation_cpu01.log`保留。测试复用真实A/B完整L0码bank计费，只解码各17行；另核两布局加权数学、异K/padding费用、不可行拒绝、真实`torch.save` metadata／额外字段费用和最后可行选轮。费用回归采用明确数学scores，不是质量读出。未执行GPU、第三阶训练、异K部署或完整live矩阵旋转验证，没有新增精度收益。

本次SHA256：`objective.py=db1f2ddba74c23e53fd5f85626adc8f79ab1fd96983846019f38eed670b20ff4`；`allocation.py=a78c2e6887c3ff657635f469b132f2a600f407b6cb7dcc7954fa34307c2b69a5`；`test_objective_allocation.py=b565d49b5195356f896753468f85beced64af46352ccf4abbae55cb6c97731aa`。仅新增两个核心模块和对应回归测试，无临时模型或生成权重。
