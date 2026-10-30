# Llama 原保护 INT8 q 与 decoder 的同源64步配对

本模块把 Qwen 已验证的固定 scale 保护整数 QAT 迁移到 Llama，比较同一 sparse005 初态的 fresh D-only64 与 D+q64。Qwen 同预算 QAT64 相对匹配 D64 的九项正差仅约 .020044pp，只提供一次迁移依据，不代表 Llama 收益或跨 seed 稳定性。Llama 旧 Adam/Muon32 仅更新 D，两臂完整九项均低于源，已经关闭；本次不重跑该32步比较，也不从其端点续训。

当前 CPU 准备与真实 GPU2 两步验证均已完成；GPU6/7正式fresh64两臂已启动，尚无正式质量结果。权威运行记录和完整结果在 [本轮结果目录](../../result/qwen3_8b_mmlu65_20261004/llama_protected_qat_pair_20261028/)。

共同源为 `/data/star/VAELLM/qwen3_8b_mmlu65_20261004/llama_best_sparse_mlp005_probe_20261024/native_export/final_model`，UUID `a1f09f8f-d9b2-4c41-91f9-e136d4ea986e`。保留全部224投影、512个实际 S2/S4 packed banks、3619持久张量，logical＝packed **2.548370287968562bit**、外置 rank0；存储账本包括主码、原保护、scale/index、decoder及已有稀疏残差等实际组件，不是理论码率。

| 臂 | 实际优化叶 / 元素 | 精确冻结补集 |
| --- | --- | --- |
| D | 1344 D / 4,078,592 | 2275 张量 |
| DQ | 同1344 D + 224 q / 总117,324,800 | 2051 张量 |

224个原保护 q 共113,246,208个元素，逐模块使用实际 shape；gate 为 S4/K16/D32，up 为 S2/K32/D32，down 为 S2/K24/D16且转置。两臂都冻结主码、原保护 scale/index、norm/head及全部576个已有 sparse codec 字段；不新挑 sparse 支持、不加 LoRA。

两臂 seed137，fresh AdamW `(.9,.95)/eps1e-8/WD0`，所有可训练参数统一 clip1.5。D LR `1e-6`，DQ 的 q LR `.03`，都乘原 `1-(step-1)/500`；groups0–31按序重复两遍，64步连续，32→33不重置优化器或 master。q master 在模型注册之外，初值为 `source_uint8.view(int8).float()`；前向使用 `clipped=q.clamp(-128,127)`、`q_eff=clipped+(clipped.round()-clipped).detach()`，再与固定原BF16 scale相乘并舍入BF16。遵循 PyTorch ties-to-even，不加信赖半径、不重标定scale。

M1024 auxiliary_train、Qasper32 train、Hotpot96 train、R七任务各128 train、分组/原生token编码及 `.5M+.5QA+.5R` 目标复用 [10月25日实际 Llama 配对](../../result/qwen3_8b_mmlu65_20261004/llama_sparse005_adam_muon32_pair_20261025/README.md) 的数据与成熟实现。M的HF full-vocabulary目标与gold只混一次；QA每步由各臂当前原生学生greedy生成，HF full-vocabulary教师在同prefix给目标，Qasper/Hotpot cap128/32，保留实际EOS、不补EOS。保留 Llama SDPA、BF16计算、FP32 RoPE、128256词表和原EOS集合，不使用Qwen tokenizer、tokens或概率缓存。各臂学生轨迹可内生分化，本实验不宣称固定trajectory的纯参数几何比较。

R概率复用同 a1 阶段初态已真实生成的896文档／15,400 continuation tokens、七文件7,901,433,680B；本次CPU实读全3619状态SHA `504009544f0f9c57d03d9b5c8b586ef4d27fb3b6b71c14a3b58c7e101643d92d`，与缓存源完全相同，并核七payload SHA、retention-prepared SHA及全部branch/input_signature。没有复用旧DS32概率，也没有重复GPU生成。数据来源、split、许可及原始身份沿用所链接配对和其已绑定数据记录。

`qat_controller.py` 为224个原保护位置提供真实可微行解码，沿原 `_materialize_full_weight` scatter进入原生权重和 F.linear。每次QA生成先临时安装当前uint8硬q并撤可微行adapter，再临时使用BF16 D和group8原生cache；返回依次清cache、恢复FP32 D、恢复source q和live adapter，完整证明全部1568叶的Parameter、梯度和非空Adam状态连续。末点仅保存BF16 D和原shape uint8 q，移除adapter；原生v6保存、序列化3619逐值比较及独立fresh strict3619必须通过，不导出FP32 q master或Adam状态，不作exact-resume声明。

CPU准备于2026-10-28完成，exit0，49.55秒，CUDA未初始化；224个q初值原生行、clamp/round/signed字节/STE梯度和真实原生2.6bit接口导入均通过。`prepared.json` SHA为 `b64cc95a748794270f51a63207380e0a204b833337bfd29c65717470da506cfb`；核心独立审查通过，仅证明代码可进入真实短验，不是GPU或质量结论。

短2只运行新增DQ路径的原groups2/14，包含最长10,398-token QA提示；两更新覆盖1344 D+224 q的有限非零梯度和真实FP32更新、第二次非空Adam原生往返、native/live的96个既有sparse消费者、冻结补集、原生保存/fresh strict，并加原R峰 `(17,boolq)/(28,piqa)` 仅FW/BW不再更新。两步硬q零变化合法，不造梯度强迫翻码。复用既有D/S4逐stage证据，不重复整套旧验证。

调用者先激活 `bitvae`；脚本只设固定环境和执行Python。以下入口都只允许在各自独立、未使用的输出上运行，不可重跑覆盖现有结果：

```bash
bash /home/star/VAELLM/experiments/llama_protected_qat_pair_20261028/prepare.sh
bash /home/star/VAELLM/experiments/llama_protected_qat_pair_20261028/short2.sh
bash /home/star/VAELLM/experiments/llama_protected_qat_pair_20261028/formal_D.sh
bash /home/star/VAELLM/experiments/llama_protected_qat_pair_20261028/formal_DQ.sh
```

上述四条均已实际执行，不要重复运行。短2在root授权且现场GPU2为0MiB/无compute后，通过tmux `llama_QAT_short2_GPU2_20261028` 启动，pane297521/start2848679、worker297535；精确命令、日志和退出位置见结果目录 `launch_short2.json`、`short2.log`、`short2.exit`。

短2真实exit0，806.3424秒。全部1568叶产生有限非零梯度与实际FP32更新，Adam连续到step2；第二次OPD原生往返的1568个非空step1状态、masters/gradients均exact，source q/live adapter与投影/KV cache恢复通过。native/live全部96个已有sparse消费者实际执行，两个R峰均FW/BW通过且没有额外更新。末q master的113,231,824个元素变化，最大绝对位移.0599212050；硬q仍零翻转，这是本两步预算的事实，不是方向阴性或收益证明。训练峰allocated75,475,690,496B；GPU模型释放后allocated0。

短2端点UUID `399fa940-ba12-4053-bdb3-02b27b1239ed`，末点native q行与hard-STE行exact，2051冻结补集及576sparse exact，source状态恢复通过。原生3619序列化及独立fresh strict3619逐值exact，仍为2.548370287968562bit/rank0。原始完整证据为结果目录 `DQ/short2/report.json` 和生成目录的 `native_model/assembly_report.json`；这只验证新增路径，不评质量。

短验权重的唯一用途完成且正式两臂已经从原a1加载后，核原worker退休、314个同UID进程cmd/maps/fd没有精确候选引用，按持续清理授权删除短2的compact、native bin和三个重复tokenizer文件，共五个精确文件，实际释放4,465,803,264个allocated bytes（4.159103GiB）。计划、路径、大小、原因、进程边界和删除前后保护SHA归并 `launch_short2.json.cleanup`；原报告及端点metadata/config/assembly audit保留。历史报告内compact/bin路径现仅为已完成验证定位，不能继续加载。正式入口只核短验报告，未绑定或读取这些已删文件；原a1、共享R缓存和全部正式输入未动。

短2完整通过后再次实测GPU6/7为0MiB、无compute，于2026-10-28约20:09通过下表单次启动既定有限链，均从原a1和fresh Adam开始，不读取短2权重或optimizer。精确命令、初态资源、prepared/short报告SHA和运行标识见结果目录 `launch_formal.json`。

| 臂 | GPU | tmux | pane/start ticks | 核心日志 |
| --- | --- | --- | --- | --- |
| D | 6 | `llama_protected_D64_full9_GPU6_20261028` | 307052/2938265 | `D/pipeline.log`、`D/train_export.log` |
| DQ | 7 | `llama_protected_DQ64_full9_GPU7_20261028` | 307059/2938267 | `DQ/pipeline.log`、`DQ/train_export.log` |

正式train worker分别为D `307251/start2938926`、DQ `307253/start2938932`，其controller/timeout进程也已记录在 `launch_formal.json`。两臂真实首个group0更新均已确认，首组八条原生QA的case ID与生成tokens逐项相同，首组加权目标同为 `.13654238046819955`。D实际322.4483秒、1344个FP32 finite Adam状态均step1；DQ实际312.0715秒、1568个状态均step1，D/q LR为 `1e-6/.03`。首步峰allocated分别73,744,262,144／74,536,985,600B。两臂均已进入第二组反向；这些事实只确认正式主循环推进，不证明64完成或质量。按启动交付规则停止主动监控，原有限链保留运行，尚无正式端点或八项/LB/九项结果。

正式每臂训练内部预算12h、有限完整链24h，实际预计训练约5h、全21约7h，仅为旧运行估计。末64各生成新UUID，依次完成固定0-shot/no-limit的完整八项（MMLU B64、seven B32，64叶／24742题）及LongBench21／4750，再按原BF16基线计算九项等权综合分；不以八项筛掉LB。验收仍为BF16绝对下降≤.8pp、实际≤2.6bit/rank0，MMLU无独立门槛。LongBench显式复用旧恢复链已经验证的2.6bit原生strict加载器，旧2.5bit入口不修改，也不绕过预算审计。没有守护、自动重试、额外续训或自动32点评测。
