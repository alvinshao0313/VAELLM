# 已确认联合R2与Nc预算闭合

本包位于所有R1活动pins之外，不改原producer／PUBLIC evaluator／selection／bank源码。只落实主协议§4：首轮全部实际完整八项齐备后，以真实joint赢家为父；在原共同费用集合中生成两个不同类别一升0.5、一下0.5的未测向量，按有符号 `Σ(BF16 S8−单类profile S8),logical bits,packed bits,稳定rate向量`取前4。单类下降不裁负、不乘Nc/N，仅作提名启发。少于4则少做，不扩大改变类型／工作集合，不生成第三轮。

原始21份完整profile、R1机器提名及实际源均绑定SHA；当前真实枚举为2187总向量、hard可行1062、两轮共用≤2.5集合667，`W=17,364,418,560`、`H=18,058,995,302`、`N=6,945,767,424`。本轮4个R1实际全齐后，bridge先核真实controller／run／worker exit／实际授权／proposal／来源pins／新UUID／完整账本／fresh／补集，再复用原 `read_native_eight`读完整八项；不能按先完成者或调用者传入分数选父。历史授权按实际request开始时刻核当时窗口，不用当前时间否定已完成实验。

CPU验证只做真实21源／费用池的纯规则、四个假设父向量的邻点机制、明确数学整数Nc例子与当时真实pending屏障，未虚构联合分数／cap。最终`source_cpu.json/log`原件在 [结果目录](../../result/qwen3_8b_mmlu65_20261004/cat_joint_round_two_cmcc_20261105/)；0.888138047秒、exit0、CUDAfalse，Astra窄审无阻断。早期同项CPU报告由此终版覆盖，重复文件已清。真实R1读取与prepare后，另由实际新运行验证本轮正例。

由调用者激活bitvae，以CUDA隐藏的tmux／nohup执行一次性CPU入口；没有daemon、自动wait或retry：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -u -m experiments.cat_joint_round_two_cmcc_20261105.driver --component prepare --r1-batch artifacts/cat_joint_round_one_cmcc_20261104/proposal_20261104_2147 --output artifacts/cat_joint_round_two_cmcc_20261105/round_two_20261105
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -u -m experiments.cat_joint_round_two_cmcc_20261105.driver --component finalize --r1-batch artifacts/cat_joint_round_one_cmcc_20261104/proposal_20261104_2147 --output artifacts/cat_joint_round_two_cmcc_20261105/round_two_20261105
```

`prepare`在R1未全终态exit0时仅返回pending，不读半写的模型结果、输出分数或创建R2。全齐时固定 `R1_actual_selection.json`，通过已GPU验证的 `producer.prepare`写每个R2显式spec／proposal和 `R2_prepared.json`。`proposal_UNAPPROVED`名称沿原生产接口的准备合同；用户11月5日“完成后续所有对照选取实验”已授权本轮，不需要重复询问，只须按当前真实卡号／时间窗另写实际auth后用原producer.run独立启动。每链export900／full8900／total1800秒、最多4链／2GPUh；资源集合0–5、建议0–3，6／7保护；无训练、LongBench、新512教师计算或新HPO。

`finalize`等待R1+R2所有已宣布端点全齐，核同一原集合、真实R1父和原邻点列表，不能漏掉失败端点。先对全部实际提案调用原 `freeze_joint→select_joint`，完成真实完整bank/state并集、非目标补集、原件八项和actual ledger核验；无allocation时只保存真实赢家事实。再令 `R=H−Σa_c`，各类 `q_c=floor(R*N_c/N)`，剩余bits按稳定类别ID升序各加1（不是最大余数法），得到 `B_c=a_c+q_c`且 `ΣB_c=H`。显式allocation绑定第一次实际选择和两轮全集，再对同全集调用原 `select_joint`核完整cap；没有新GPU评测或费用借额。

对HPO公开固定输出：`artifacts/cat_joint_round_two_cmcc_20261105/round_two_20261105/final_cap/cap_certificate.json`，schema沿原 `actual_joint_full8_cap_certificate`。同目录保留无allocation的真实选择、Nc整数分配、原及带allocation的冻结全集。cap文件只有全部真实验证完成才生成，缺项／失败保持pending，不以0步、CPU/tiny或不完整分数替代。原全部R1/R2联合state与formal bank/raw在闭合前必须保留。

这只完成原profile码率联合基底及类别费用分配，不能称逐参数19点搜索、正式CAT／恢复／九项或跨模型目标已经完成。已确认逐参数顺序与实际运行见[独立搜索包](../cat_sequential_search_cmcc_20261105/README.md)，可复用的取舍与迁移边界归入[码率profile与真实费用经验](../../docs/experience/lessons/compression_choices.md#固定d的码率profile与真实费用)。

## 11月5日实际终态

R1四份及R2四份均已真实完整八项／controller exit0，R2没有超过首轮B；两轮GPU拥有窗分别.993390264与1.062231202GPUh，各低于2上限，内部子窗不另加。[实际两轮运行／结果／分配记录](../../artifacts/cat_joint_round_two_cmcc_20261105/round_two_20261105/README.md)汇总全部源UUID、S8、费用及运行标识。

[正式cap证书](../../artifacts/cat_joint_round_two_cmcc_20261105/round_two_20261105/final_cap/cap_certificate.json) SHA `366bf5923d0e2bb5d5518105e33febdd58f22466a86f0facd9297fae065ccf4a`，`certificate_identity=6cf3392dc8a7e114e313a9929287a583b35cdf61aa946113d3f5fafbd2be6b38`；原`select_joint`对8实际模型先无allocation、再带Nc allocation均完整强核通过，status complete／formal_eligible true。最终赢家R1-B、S8 64.32986503987618、logical/packed2.493649690047554bpw，ΣBc=H=18,058,995,302bits，禁止跨类借额。CPU finalization 2045.801871942秒、exit0，不新增GPU；没有重评或第三轮。

后继可按同一原BF16/source9/data/consumer/seed与该文件SHA消费各类`cap_bits`。8份联合payload与21份formal bank/raw及证明继续保留，未在本次提前清理；这份有效预算证书也不表示逐参数所有类最优、九项目标或完整迁移已经完成。

文档收尾仅执行一次`python tools/docs.py index`与一次`check`：index exit0，340文档＋18 aliases；check exit1，canonical341／aliases18／relative路径4930，仍为既有1654项缺链，本包README没有新增missing。未声称全库通过；本次未重查模型或运行GPU。临时全量检查日志已归并为此摘要后清理。
