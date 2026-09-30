# CatLoRA 严格 2-bit 多保真搜索

`tools/run_catlora_hparam_search.py` 复用 `scripts/catlora_simple2.sh`，但每个 trial 使用独立输出目录，并把所有类别固定为

```text
codebook_bits=32, codebook_dim=32, residual_stages=2
```

这对应名义 BSQ payload 的参数加权平均 2 bit；channel protection、decoder、索引和其他保存开销需要另行统计，不能混入该名义数字。

先运行短筛选，再提升前两名：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bitvae

python tools/run_catlora_hparam_search.py \
  --stage screen \
  --search_root /root/data/ckpts/result/catlora/tuning_2bit

python tools/run_catlora_hparam_search.py \
  --stage full \
  --promote 2 \
  --search_root /root/data/ckpts/result/catlora/tuning_2bit
```

筛选阶段只压缩目前最敏感的 `gate_proj,up_proj,down_proj`，使用 1000 个 VAE steps、500 个蒸馏 steps、每个任务 64 个样本；完整阶段恢复到全部七类和 10000/5000。搜索目标是八个下游任务均值减去任务分数跨度的一半，结果和命令保存在 `manifest.jsonl` 及各 trial 的 `tuner_result.json`。

当前 `catlora_simple2.sh` 固定使用 0,1,2,3 四张卡，且已有正式任务运行时不要启动该搜索；搜索器不会停止或覆盖已有进程和结果。
