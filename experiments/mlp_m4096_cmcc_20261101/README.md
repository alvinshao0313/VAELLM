# 同 C0 的 M 唯一覆盖对照

状态：固定128训练、64/128独立fresh、64 MMLU观察及128完整九项全部完成，外层exit0；质量`not_met`。末点九项63.88939213837067，较同C0 D128控制低.0654306123pp，MMLU仅多6题、LongBench低.1211742808分。按完整收益／成本决定结束本固定同源支持数配置，不继续扩量或延长。唯一执行计划由[当前计划](../../docs/experience/records/w25_transfer/current_plan.md)维护，本包保存固定接口与实际证据。

本轮对照只把已完成 C0 固定码全108 MLP decoder128 的 M1024 四遍，改为同源固定 hash 前缀 M4096 各一次。每组仍 M32/QA8、共128更新；QA沿控制原32组后缀顺序循环四遍，Qasper32/Hotpot96 的输入、目标、重复和权重完全相同。M仍为原 full-HF BF16 教师与 gold 各半、FP32 四选项目标，静态 TF，无 OPD；旧1024已混目标直接复用，新增3072仅混一次。

共同来源 UUID `cbb15c9f-1c1f-4394-817c-65c226d76ce7`，真实目录在 `bindings.C0`。冻结 attention、全部 codes、保护和非MLP。AdamW(.9,.95)、eps1e-8、WD0、foreach=False、clip1.5、FP32 masters/BF16 compute、D LR1e-6、原500线性日程前128、seed137。固定64/128独立BF16快照和fresh；0/64/128完整训练池拟合读出；64完整MMLU仅观察，128完整八项＋21项LongBench。固定128，无 LR 网格、质量选点、早停或恢复。

旧HOT128的同源M4096已是完整负例：九项65.938001，相对原HOT128＋.127586pp，但M−.519869pp、完整LB−.341636，微涨主要来自RTE。参见[覆盖经验](../../docs/experience/lessons/kd_and_data.md#auxiliary_train来源与覆盖预算)。本次新依据是不同真实fresh C0和全108MLP D：控制训练池M正确811→893、dense→966，而完整MMLU分别62.27745/61.50121，较C0少17/126题，拟合与迁移有差距。该证据支持一次配对收束，不预设同源扩池有效；若完整M和九项均未优于原D128，结束本同源扩池，不追加dense4096/目标/QA网格。

## 数据来源与最小验证

本机未迁移旧M4096 dataset/teacher和官方auxiliary_train Arrow。迁移manifest只包含相关源码/指标，且声明开放数据需按记录恢复。现按 `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe` 的 `auxiliary_train/train` 精确恢复原源，MIT许可；Parquet及README以固定revision API的LFS SHA256/size或git blob校验。官方CDN受本机网络限制，实际传输使用既有HF-mirror，数据身份仍是固定官方revision。新capture明确区别于原Arrow SHA，不冒称旧字节相同。

固定SHA256(`causal_output_fit/137/`＋原case_id)、按case_id破同hash，99842行取前4096，不按test或教师正确率筛选，不补subject。复用旧M builder；先把新构建前1024全字段、输入、原生选项支持和顺序与当前控制严格比较，再直接复用控制1024和全部QA case/目标。新增3072保留新capture物理来源。只使用开源train/auxiliary_train。

`prepare.py`列出128组真实M max/padded/nonpadding/target与QA长度；`cache.py`只在GPU1首次生成新增3072的真实full-HF BF16 teacher、保存重载并释放，原M1024/QA128缓存不变。`validate_cpu.py`核所有旧目标逐tensor相等、新M仅一次mix、QA128组后缀精确、完整曝光、源身份、CLI与评测派发。实际最大M不超过1222时复用已有同C0 D128路径证据；原958峰19.835/19.583GB按长度整峰线性估计约25.3GB，属估计，不称新GPU实测。FA2、last-query-head、检查点和长QA均不变，无新增训练short。

实测CPU prepare exit0：前1024完整字段exact，QA128组后缀exact；新峰group45的M max1222、padded39104/nonpadding13957、M target32，QA max2221/总12774。完整128输入4,223,364 tokens，相对控制4,212,624增加10,740（约0.255%），answer positions仍10,944；这是等M/QA呈现和更新对照，输入tokens不同。

GPU1 `teacher_v2` exit0，新增3072前向219.50秒、全缓存及核证265.04秒，保存重载exact、冻结全状态exact、教师完全释放；首尝试在任何target前因CPU身份核证函数接收GPU state触发设备契约错误，失败日志保留。修复为先CPU核HF身份，再沿成熟 `apply_layer_device_map` 把全部36层映GPU1，保留FP32 RoPE alias；没有改变目标数学或重算旧1024/QA。

## 隔离入口与输出

成熟数学/状态/快照/评测从只读 `mlp_learnability_formal_cmcc_20261101` 复用；本包拥有独立bindings/data_contract/source，严格要求本M4096数据身份及真实完成C0、fresh、保存字段和低比特预算，未放宽旧validator。旧包及活动992 source pins不改。

Python均为已激活 `bitvae` 的调用者shell，正式运行由 `sequence --prepare` 固定计划后 `sequence --run` 后台一次执行，无重试/资源轮询。GPU2/3训练和fresh，完成后GPU3固定评测；GPU1只补teacher后释放。训练12h上限；按控制7.65h实测及新增训练池读出，预计约9–11h，非完成承诺。所有模型产物在 `/data1/cmcc/VAELLM/artifacts/mlp_m4096_cmcc_20261101`，原始日志/计划/运行状态/CPU/评测在 `result/qwen3_8b_mmlu65_20261004/mlp_m4096_cmcc_20261101`。

```bash
python -B -u -m experiments.mlp_m4096_cmcc_20261101.prepare
CUDA_VISIBLE_DEVICES=1 python -B -u -m experiments.mlp_m4096_cmcc_20261101.cache
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.mlp_m4096_cmcc_20261101.validate_cpu
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.mlp_m4096_cmcc_20261101.sequence --prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.mlp_m4096_cmcc_20261101.sequence --run
```

这些命令由tmux后台运行、stdout/stderr落盘；实际进程、exit和命令以本包结果目录的 `single_20261101/run.json`及各阶段日志为准。已经完成的目录不可重复启动。

11月1日21:46正式启动，tmux `M4096_C0_D128_20261101`；controller `1511850/start27816285`，worker `1512143/start27818081`。1013项源码/数据pins，plan SHA `6bf4eba7a27c058556972e4d383d45a6eccc0f407ef5c3d5ca51e0eb4e76a136`。实际命令在[运行记录](../../result/qwen3_8b_mmlu65_20261004/mlp_m4096_cmcc_20261101/single_20261101/run.json)，[CPU核证](../../result/qwen3_8b_mmlu65_20261004/mlp_m4096_cmcc_20261101/cpu/validation.json)的旧目标/QA顺序/真实源/CLI/派发全部通过且CUDA未初始化。worker新 `execution.json` 已核真实C0、decoder/formal、128、seed137、仅GPU2/3，以及本数据contract SHA `8659d955f6514677945def2cde081b254493912c10c6ec79ba5f5d9a15b7a063`。

21:49有限实核同worker在GPU2/3各30,348MiB，GPU3瞬时99%计算，代码/源/新输出/数据绑定与计划一致，无更新文件或失败；这是初始化阶段观测，不是新M训练峰或完成128。零点完整池读出无需等完，确认真实CUDA推进后停止主动监控；保留固定后台链，随后依次fresh64/fresh128、GPU3完整MMLU64与128完整九项，不自动扩预算。

数据保留供来源复核及其它已安排train消费者；64/128评测已完成，生成权重按实际后继用途收尾，元数据中的历史路径不等于文件仍可加载。未提交代码以source SHA记录，不生成虚构commit。

先前收尾已清理本次官方source目录下可再生且已无引用的 `arrow_cache`（161,068,159B）和失败Hub `.cache`（12,289B），约154MiB；新Arrow的SHA/size及来源已归CPU证据，不再要求物理缓存存续。原始Parquet/README/capture、训练dataset/teacher/data contract、CPU证据、成功核心日志及唯一teacher设备契约失败证据均保留；已替代的三次下载失败诊断14,684B删除，不计入下面的本轮释放量。

11月2日完整配对结束后，[本轮清理记录](../../result/qwen3_8b_mmlu65_20261004/mlp_m4096_cmcc_20261101/cleanup_20261102.json)逐项核对活跃命令、打开文件／内存映射、后续脚本及活动计划用途，删除本臂64／128、原D128配对末点及旧两臂short各两份共7个无后继权重和10份已用完fresh测试文件。实际文件大小51,137,073,859B、分配占用51,137,142,784B（47.625175GiB）；没有落盘的master／optimizer／RNG可清，旧D64／dense64／dense128三权重未重复计入。全部配置、metadata、expected-state／fresh审计、源码SHA、退出与核心日志、原评估样本和汇总保留；逐步JSON含唯一训练指标，也保留。M4096 dataset／教师缓存及原始官方数据仍供S4等消费者使用，未改动；C0／C1、Muon控制、共同after_o／raw2及其它任务依赖均不在删除范围。

## 完整结果与研究取舍

已核[固定链终态](../../result/qwen3_8b_mmlu65_20261004/mlp_m4096_cmcc_20261101/single_20261101/run.json)为`completed_same_C0_M4096_fixed128/exit0`，train、fresh64、fresh128、midpoint及evaluate均exit0；[评测终态](../../result/qwen3_8b_mmlu65_20261004/mlp_m4096_cmcc_20261101/evaluate/evaluation.json)为`complete_full8_full21_current_goal`。固定64仅观察、不择优，128端点完成同UUID八项64叶／24742样本和LongBench21／4750。

| 配方／指定比较 | MMLU % | LongBench21 | 九项均分 |
| --- | --- | --- | --- |
| C0父参考 | 62.39851872952571 | 41.74183992400309 | 63.58416595020284 |
| 同C0 D128控制，M1024×4 | 62.27745335422304 | 43.864596592189265 | 63.9548227506918 |
| 本臂M4096×1 | 62.320182310212225 | 43.743422311417795 | 63.88939213837067 |

本臂对匹配D128控制，正式MMLU多6／14042题（+.0427289560pp），九项低.0654306123pp，LongBench低.1211742808分；M仍比原C0父少11题。训练阶段进程墙钟32900.39秒（9.139小时、双卡，含预定训练池读出），不把它当纯GPU kernel耗时。实测logical2.4987414816151494、packed2.5121161419412363bpw及4671007680 tensor bytes保持，与[完整验收原件](../../result/qwen3_8b_mmlu65_20261004/mlp_m4096_cmcc_20261101/evaluate/longbench/merged/nine_task_acceptance.json)同源；预算通过、相对BF16九项下降4.1577527795pp，质量未达。

Astra依完整收益、代价和既有同源扩池边界确认结束此固定支持数配置。这里M确有小正差，因此不声称触发原“M与九项均不优”的字面AND规则；也不由本结果否定不同知识来源、其它目标或所有覆盖研究。保留原控制证据、配方／曝光／源码SHA、核心日志及原始指标作为配对证据，不追加dense4096、同源扩量或更长预算。可迁移动作是分开训练拟合、支持数、不同来源与训练外净收益，模型更换须重算token化及真实监督；本次1024／4096与128仅为已测范围。[数据经验](../../docs/experience/lessons/kd_and_data.md#域损失clip和opd保留的真实贡献)补充本父证据，后续资源已释放。
