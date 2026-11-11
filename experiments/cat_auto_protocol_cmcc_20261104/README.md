# CAT 自动协议的隔离实现

本包实现[已确认 v0 协议](../../docs/experience/records/w25_transfer/2026-11-04_cat_auto_protocol.md)的配置、真实消费者适配、候选生成、费用与恢复组件。当前已完成 CPU 组件及**本 Qwen3 全M36、正式形状B8192的 PUBLIC 两步技术路径 pilot**；未完成full10K、HPO、完整20槽、full8、跨类别链或跨模型验证，不能称完整自动链或质量通过。当前成熟的独立 raw 三点卡另有来源与预算，不能计入本包 BF16 离线20槽。

## 文件与实际入口

| 范围 | 入口 | 当前范围 |
| --- | --- | --- |
| 模型与身份 | `adapter.load_inventory`、`contracts.inventory_id`、`identity` | 实际 Qwen3/Llama HF 模块与 safetensors header、完整投影 N；内容身份不依目录；未知结构受阻 |
| 几何与费用 | `config.resolve_recipe`、`geometry.component_plan`、`ledger.measure_checkpoint` | 真实保护后两轴、Hadamard、Decoder、bitpack；预测与实际存储分开；其它类别 dense 照实收费 |
| 单类全 M 拟合 | `fit_adapter.fit_category` | 实际 PUBLIC `train_group_vae_payload` 与 `apply_group_vae_payload`，完整 pooled BSQ，不拆 M；逐 ref T/R、stage 初始化及 shuffle 已接真实调用 |
| 原生单类 bank | `extract_category_bank`、`install_category_bank` | 真 packed codes/decoder/保护/sign 保存与原生重建，可在共同 BF16 父上同时安装多个类别；不拼历史 live 后缀 |
| 优化器 | `vae_muon.VAEMuon` | 实际 R0 四个 ParallelLinear owner；矩阵 momentum FP32、NS5 BF16、辅助 Adam moment BF16；各 stage 固定 Adam .003/1× 校准 κ 后再乘倍率 |
| 类别预算 | `proposals.round_one/select_joint/round_two/category_caps` | Σ profile KL 仅提案；真实 joint 证据才可锁正式 cap；CPU fixture 向下始终非正式 |
| 离线20槽 | `offline` | 确定性集合覆盖、邻点、优化器、倍率、条件队列、slot20、raw8覆盖和排序；依赖实际结果才生成后续槽 |
| 预算与有限编排 | `budget`、`orchestration`、`search` | 完整 DAG 与未知费用；有限入口只执行显式1–4个 BF16 profile/offline候选，不自动执行175 fits |
| 数据与读出 | `prepare_data`、`data`、`proposal_readout` | root 持有；实际冻结校准、五源更新/选择集、Top100+outside KL、前后半逐层功能读出 |
| 恢复、节点与自身接续 | `native_builder`、`live`、`mutable`、`nodes`、`recovery`、`replay` | 独立 owner 的实际 native KD/完整 mutable 保存、导出、LoRA融合、受控新 UUID/live fit 桥；模型 GPU 整链与质量待验 |

`fit_category(model, *, inventory, recipe, run_identity, calibration_bundle, activation_bundle=None, device, output_dir, technical_steps=None)` 返回实际 `model/fit_report/activation_bundle/category_bank`。校准 bundle 必须提供冻结的 CPU `input_ids`、`manifest`、`data_identity`、`model_path`、`dataset_spec`。原生 cache 使用同一 canonical dataset 字符串并核对象身份，禁止静默重抽。激活复用同时绑定父 UUID/权重、当前目标 digest、校准 token digest 和数据身份。

PUBLIC 函数缺少的 factory/seed/per-ref rotation/诊断调用点由本包私有 AST 适配；源文件 SHA 与精确改点数不匹配立即拒绝。PUBLIC 共享 globals 和源码不修改。技术模式仅1/2次真实更新，保留原 linear10000 前缀，永不计完整离线槽。正式 CUDA 候选必须绑定实际 code/consumer、正式 registry 中的 exact recipe、真实 data plan 与校准张量；CPU fixture 不能因设备标签升级。

## CPU 证据与边界

证据目录为 [result/cat_auto_protocol_cmcc_20261104](../../result/qwen3_8b_mmlu65_20261004/cat_auto_protocol_cmcc_20261104)。所有 Python 验证由已激活 `bitvae` 的调用者以 tmux 后台运行，显式隐藏 CUDA。

| 证据 | 已实际验证 |
| --- | --- |
| `model_inventory_cpu.json` | 官方本地 Qwen3-8B 来源文件实际 SHA、meta-device 模型252投影、真实 tensor header；20.219秒，CUDA false；未分配 dense 模型/forward |
| `fit_cpu_final.json` | 真实 tiny Qwen3 的 gate/up 各全 M2、S2×2步；同 M 逐层 T/none-full rotation、独立 stage RNG、首两实步及 Adam state、双 packed bank 联合安装/逆序 fresh state+logits exact、冻结与完整 partial 账本；Muon M2/M3、E128/D256、κ、状态往返、observer 无扰动；仅机制 fixture |
| `fit_lr_cpu.json` | 本次受影响的真实 tiny PUBLIC up D32K32/S2两步 LR hook；两个stage均实测 .003/.0029997、原linear10000、实际位移与Adam step1/2；3.94147秒，CUDA false；未重跑成熟全套 |
| `planning_cpu_final.json` | 真实 Qwen up60几何与实际 factory 费用；7槽最小覆盖、后续生成与重复槽；纯数学 fixture 的 joint/cap、2.5/2.6、层组 regret 和预算状态；无实际训练收益 |
| `entry_cpu.json`、`entry_cpu_reuse.json` | 实际 root `full_data_v0/data_plan.pt` 身份与512×256张量绑定、有限 request/保护卡/预算/来源反例、全包语法；后者补同路径 LR/注册BSQ系数复用、新资源/算法及CPU/1步证据拒绝；未执行 GPU worker |
| `full_data_v0/report.json` | root 实际五源 SHA/行数、原 Alpaca train512×256 exact、7类别 T1000 数据前缀、512选择记录与原 ID 排除；107.693秒，CUDA false；数据身份 `e675f901e68f0f56bc3ad1a641db52e15b4c14ba4f62bb15255f0db1f5d828bb` |
| `data_cpu.json`、`data_contract_cpu_v2.json`、`local_readout_cpu.json` | root 单源/真实 tokenizer 机制、原更新批次兼容、真实 vocab KL/mask、tiny BF16 streaming512/B4逐层输入及 padding/mode/RNG/hook恢复；非正式 LLM GPU 质量 |
| `recovery_cpu_v3.json`、`recovery_loader_cpu.json`、`native_builder_cpu.json` | native mutable/保存后回训/选点及旧真实 full8 reader；真实 tiny teacher KL、B4/16workers、原 T500 前两步、native decoder/future LoRA、保存/导出；非模型 GPU/replay整链 |

早期 fit CPU 失败保留在 `fit_cpu.log`：fixture config 的整数键身份已改为 canonical JSON；原 cache dataset 未 canonical 导致原生 collector 想重抽，已在真实消费者入口修正。没有用网络回退或伪校准替代。最终报告记录当次实际源码 SHA；后续仅文档维护不进入源码 pins。

## 固定规则与覆盖状态

| 协议项 | 已接能力／待验 |
| --- | --- |
| §1–2 原 BF16/模型适配/数据/离线与 live 区分 | 实际 inventory/content identity、root完整 DataPlan、真实教师 KL/原生 bank 入口已接；此Qwen固定形状GPU技术路径完成，跨系列未验 |
| §3 D/K/S/B、T/R、保护、网络、loss、优化器 | 非8倍 K 合法，logical按 K、packed按 ceil(K/8)；不padding/降B。通用 Muon CPU 通过；本次Qwen M36/B8192/D32K32S2/Adam 的两步GPU路径已完成，其它新K/M/B/网络/算法仍须适用证据；常规注册LR/系数可复用同路径 |
| §4 joint budget | R1先≤2.5工作集合；R2全部合法邻点≤2.6，可高于2.5，记录实际偏差/headroom；旧赢家一起比较。完整整模 bank 安装与 root joint forward 都有实际 callable，尚无本包正式 profile/joint GPU结果 |
| §5 离线20槽 | 7+2+2+4+4+1；κ拟合前计划身份固定，实测 κ 另入 artifact；requested p仅生成 provenance，真实 counts 相同即复用。条件段顺序接受且仅此段叠加；全8联合覆盖≤10。当前缺全20结果驱动自动 executor/full8调度连接，不冒称一键完整搜索 |
| §6–7 live2/有界恢复/选节点 | recovery owner 提供真实 native bridge 与 CPU证据；当前只支持 native v6，mixed compact 明确受阻；完整模型 GPU、完整保存回训/自身下一类及结果选择待 pilot |
| §8 层组 | L=3q+r，前r组q+1；同父/数据/M/其它真实配置的 T/R cohort；每组 NMSE=ΣSSE/Σtarget²，full合并两半原始和；前后半最优集合交集选组几何，至少两组不同且两半 regret>0；按 N_g/N_c、N_c/N_all 加权提名前两类，只占原slot20。root实际 streaming读出已接；数学 fixture 不能触发正式提名 |
| §9 模型预算/缓存/调度 | 完整有依赖 DAG、相同语义 seed/cache/artifact两层身份、剩余anchor/replay/nine保留函数；未测单价保持unknown。有限入口共用绝对deadline/真实退出/无自动重试；模型新批须具体预算 |
| §10–11 新表示/重放/迁移 | INT4/双轴/异D/native异K/LFQ不冒充支持；full-model search/replay/nine/跨模型尚未执行。CPU属性通过不等于统一模型GPU链就绪 |

覆盖记录区分 `tested/matched_reuse/math_invalid/impl_blocked/budget_deferred/v0_fixed`。数学非法、实现未接、预算未测都不是质量淘汰；仅 level 出现不宣称交互穷尽。

## 有限入口与 GPUh

有限候选入口为 `python -m experiments.cat_auto_protocol_cmcc_20261104.search --request <frozen-request.json>`。由已激活 `bitvae` 的调用者在 tmux/nohup 执行；不在脚本内激活 conda。request 必需绑定实际 inventory/data plan 及 SHA、完整 `run_identity`、BF16 parent、显式候选/用途/各依赖费用、实际源码 pins、授权卡/共同绝对 deadline/总 GPUh。该入口保护0/2/6/7，只接受当次具体批准的1/3/4/5；不会自行挑卡或扩大批次。

`technical_validation` 仅原10K日程前1/2步，正式模型条件不缩 B/M。1步只作诊断，记录实际 `technical1` scope；正式证据须核实际 S 个 stage 各完成2步，不靠 scope 名称判通过。`formal` 保持同模型/DataPlan/实际 fit 代码/consumer 身份绑定：exact candidate 是充分证据，同路径只改注册 LR 倍率或单项 commitment/entropy 系数也可复用，不重复常规短测。资源签名仍完整绑定实际 M/group、B/D/K/S、网络/精度、T/R、保护及优化器/κ/BSQ关键算法合同；新资源、结构或算法路径无适用实证即拒绝。CPU/technical产物不因此升级为正式结果。离线正式候选另需真实 joint cap证书。入口执行 fit、真实 packed bank、冻结512教师KL与逐层读出、完整 native raw/实际 ledger、独立 fresh 同输入 BF16 logits；完整八项是另外的明确依赖，当前不自动追加。live 必须走 `live.fit_next_live_category` 保存其自身完整 prefix/runtime，不能借此 BF16 bank runner 冒充 live接续。

完整 v0 的已知数量上限（C=7）：21个额外 profile、140个离线 fit、14个 live fit、≤70次离线 raw8、14次 live raw8、≤23条恢复、≤100次恢复节点8；两条1000替换另算原500差额，参考链/BF16固定recipe重放/至少3次九项及共享开销另列。**模型总 GPUh 当前未知**：各类别/结构/范围缺实测上界，不能拿 down 单价填其它类或宽度/S4。`budget.build_task_graph` 返回 `total_upper_gpu_hours=None` 与明确 unknown_nodes；纯数学的每未知任务1h情景只验证计数，不是预算估计或执行授权。

唯一 GPU5≤0.5GPUh 的 PUBLIC 技术／成本 pilot 已完整结束：原 BF16 Qwen3-8B、up全M36、D32/K32/S2/B8192、E/D128/R0、Adam .003、TFalse/block32/input64，逐stage仅原linear10000前2步；实际校准512×256及proposal512未缩。`prepare_public_pilot.py`冻结一个新request，共同30min绝对deadline覆盖teacher/捕获/fit+编码/proxy/local/save/fresh。它不占HPO槽、不含full8/full10K/KD，没有重试、续期或自动后继；其它模型搜索预算仍须按实际费用讨论。

实际运行：tmux `CAT_PublicUp_pilot_g5_20261104_1455`，controller2646900/start51268596、worker2646901/start51268602，14:54:57启动，15:02:09 controller实际exit0，未到15:24:57固定截止。两进程已退出、GPU5释放。[唯一request](../../result/qwen3_8b_mmlu65_20261004/cat_auto_protocol_cmcc_20261104/public_up_pilot_20261104_1455/request.json) SHA `27daf04be51d878f556150dbd8fc9da27e16efe53f21301a74a73ae153f03121`；[原始完成报告](../../artifacts/cat_auto_protocol_cmcc_20261104/public_up_pilot_20261104_1455/completed.json)、[只读闭合核证](../../artifacts/cat_auto_protocol_cmcc_20261104/public_up_pilot_20261104_1455/closure.json)、worker日志、progress和实际exit均留原根目录。

| 实际阶段 | 秒 | Torch allocated峰B | Torch reserved峰B |
| --- | ---: | ---: | ---: |
| 来源／DataPlan绑定 | 22.059832 | 未采GPU峰 | 未采GPU峰 |
| proposal打包 | .439180 | 0 | 0 |
| 教师512 | 37.449882 | 22,613,389,312 | 22,741,516,288 |
| fit＋编码／bank | 218.068933 | 16,547,691,008 | 16,592,666,624 |
| 整模proxy512 | 37.605716 | 23,564,461,568 | 24,366,809,088 |
| 同父输入逐层512 | 61.246862 | 22,241,585,664 | 24,389,877,760 |
| 原生保存／实际账本 | 26.217199 | 35,913,728 | 83,886,080 |
| 独立fresh／同路径logits | 25.700605 | 20,391,247,360 | 21,005,074,432 |

worker全链429.829568秒（.119397102 GPUh）；从授权request起算到controller退出为432.367952秒（严格占用.120102209 GPUh），低于.5h硬限。整体Torch峰allocated23,564,461,568B（21.946115GiB）、reserved24,389,877,760B（22.714844GiB）。fit阶段包含父加载、完整512激活捕获、保护/旋转、两步拟合、编码及bank，不能称纯训练峰／纯训练耗时，也不能按218秒除4外推20K更新单价。CPU峰未采集；14:58:58单次RSS采样118,854,760KiB（113.348732GiB）仅为瞬时值。

两个stage各2次真实更新，LR均.003/.0029997，12叶Adam moment为BF16、实际state1/2及非零参数位移；非目标字段冻结digest exact。真packed单类bank476,068,608B，metadata174,493B。原生完整raw UUID `b0e27803-962d-48c4-8a7d-608591a3e887`，831字段独立fresh全部exact，同输入同BF16路径logits exact；实际文件SHA与原request全部source pins在闭合核证中再次核对。

账本共同N=6,945,767,424，36up实际payload3,808,548,864bit（2.101918538本类bpw），其中含72份真实stage codes/decoder、INT8保护、index/scale及rotation；其余216投影dense82,141,249,536bit全部计入。完整partial logical=packed85,949,798,400bit（12.374413532bpw），整模tensor13,233,660,672B、raw目录13,253,676,219B；无code padding/storage alias。此partial不适用最终2.6门槛，不冒称全七类压缩完成。固定512读出KL11.223188722与36层前后半实际功能统计仅为两步技术产物，不作收敛／质量排名。

当前只认可此Qwen/M36/B8192/D32K32/S2原Adam技术路径完成，`formal_eligible=false`保持。pilot没有encoder/optimizer/RNG供续训，完整profile必须从BF16 fresh拟合；原bank/raw现暂留至root/Astra明确后续用途，尚未清理。下一批由root/Astra讨论，不自动追加full10K、HPO、full20/full8、KD、跨类链或跨模型实验。
