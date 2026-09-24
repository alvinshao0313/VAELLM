# EdgeRazor 数据集接入

CAT 与 compressed E2E 统一通过 common data config 使用 EdgeRazor 数据：

```bash
--dataset_mix "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009"
--dataset_task sft
--model_max_length 1024
--dynamic_padding true
```

先运行 `bash scripts/prepare_vaellm_edgerazor_data.sh` 准备本地数据。alias、采样权重、tokenization 与 cache key 都由 shared data layer 处理；CAT 和 E2E 不各自维护第二套数据语义。

模型级 loss 仅支持 `sft`、`kl`、`kl_top`、`kd`、`kd_top`。推荐脚本使用 `kl_top` 和独立 `--top_k 100`，teacher 输出通过 `--teacher_output_offload cpu` 暂存。

正式入口：

- CAT：`scripts/catlora_simple2.sh`
- checkpoint-distill：`scripts/catlora_distill_from_checkpoint.sh`
- E2E：`compressed_e2e_fintuning/scripts/e2e_decoder.sh`

独立 block 蒸馏以及 E2E stage1/stage2 脚本已删除。
