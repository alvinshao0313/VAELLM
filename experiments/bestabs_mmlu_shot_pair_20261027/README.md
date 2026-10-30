# 当前 Qwen bestabs 的 MMLU 0／5-shot 配对

2026-10-27，评测设置诊断，不训练或修改模型。用户已于10月9日授权比较合理的0／5-shot设置，按同设置BF16与压缩模型的掉点判断，并公开观察后选择。原B64／0-shot九项验收保留，不覆盖或交叉相减不同设置的基线。

## 范围与身份

- 压缩源：`qwen_b32_sparse_mlp005_probe_20261024/native_export/final_model`，UUID `e01b09b7-a85d-4534-a5d2-5ab3b91312e6`；logical 2.59261881786844、packed 2.605993478194527 bit，rank0。沿已确认的logical≤2.6口径，padding单列。
- 两臂仅比较0／5-shot，均B16、原生likelihood、无chat、57科／14,042题、完整无limit。5-shot使用既有MMLU协议的`dev/first_n`，并非将dev用于训练。
- 复用10月9日同B16的BF16完整0／5-shot：72.9810568295／75.0035607463。准备已核旧源码pins、HF权重／数据缓存身份和实际软件版本；没有为占卡重复推理。
- 复用已经真实支持当前稀疏成品的`result/qwen3_8b_mmlu65_20261004/lm_eval_sample_artifacts_20261010/cat_eval.py`。不改变loader、模型或其余七项与LongBench21协议。

## 入口与状态

调用者激活`bitvae`，从仓库根目录使用`prepare.py`准备；精确CLI、环境、源与数据绑定在结果目录的`prepared.json`，不在本文另抄一份可能过时的长命令。

结果根：`result/qwen3_8b_mmlu65_20261004/qwen_bestabs_mmlu_shot_pair_20261027/`。

- GPU2／0-shot：tmux `bestabs_mmlu_shot0_20261027`，worker `316339/242065441`；已完整完成，run与外层exit0。实测8,818／14,042，MMLU 62.7973223188%。原始样本身份、SHA与57科计数已核。
- GPU7／5-shot：tmux `bestabs_mmlu_shot5_20261027`，driver `316330/242065415`、worker `316340/242065448`；已完整完成，exit0。实测8,981／14,042，MMLU 63.9581256231%。两臂均在服务器22:02重启前完成，不需要恢复或重跑。
- 各臂实际命令、进程身份、日志与退出记录以各`run.json`及原日志为准，运行上限7200秒＋20秒退出宽限，无自动重试。

## 配对分析

`analyze.py` 已通过真实历史结果验证：核当前bestabs原B64全科样本，重现10月9日两组配对九项公式至1e−12。验证原件为结果根`analysis_verification.json`，不是新两臂的最终结论。

两臂完成后已执行一次以下CPU分析，`paired_report.json` 为 `complete_cpu_verified`：

```bash
CUDA_VISIBLE_DEVICES='' python -m experiments.bestabs_mmlu_shot_pair_20261027.analyze \
  --run-dir /home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/qwen_bestabs_mmlu_shot_pair_20261027
```

分析核对同shot配置、版本、逐科整数计数与原始预测，再分别替换两侧九项中的MMLU；各自另外七项及完整LongBench沿原件固定，不调用旧硬编码0-shot的通过器。2026-10-28按实际报告复核：

| 同B16协议 | BF16 MMLU正确数 | bestabs正确数 | BF16九项 | bestabs九项 | 绝对下降 |
| --- | --- | --- | --- | --- | --- |
| 0-shot | 10,248 | 8,818 | 68.0447710870 | 67.0148080478 | 1.0299630392 pp |
| 5-shot | 10,532 | 8,981 | 68.2694937444 | 67.1437861928 | 1.1257075517 pp |

每臂MMLU分母均14,042。两种设置都未达到0.8个百分点要求；5-shot使学生绝对分数提高，但教师增幅更大。按预定“较小的同协议下降”准则，观察后选择0-shot；两套完整结果均保留。原B64／0-shot验收67.0235120944及其基线68.0471449179不改，B16差异不作为模型恢复收益。

经验依据：[0／5-shot公平对照](../../docs/experience/lessons/experiment_design.md)、[10月9日配对与当前目标记录](../../docs/experience/records/w25_transfer/2026-10-04_mmlu65_llama31.md)。旧a0e5的5-shot掉点更大，本次成品和恢复状态已变，因此只做这组明确的新端点对照，不扩大prompt／shot搜索。
