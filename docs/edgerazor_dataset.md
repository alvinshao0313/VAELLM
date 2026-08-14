# EdgeRazor + VAELLM Eval 数据配方

## 配方说明

通用 instruction 语料（与 EdgeRazor Qwen3 一致）：

- `ii_7M_instruct.jsonl`
- `ii_gen_1.4M_instruct.jsonl`
- `tulu_0.6M_instruct.jsonl`
- `am_1.4M_instruct.jsonl`

下游任务切片（`task_vaellm_eval_instruct.jsonl`）：

- 保留：boolq, winogrande, arc_easy, arc_challenge, openbookqa, piqa
- 删去：hellaswag, social_iqa, hendrycks_ethics
- 新增：rte, mmlu, longbench

三条训练链路统一 mix：

```text
edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009
```

Python 常量见 `e2e_common/data.py` 中的 `VAELLM_EDGERAZOR_DATASET_MIX`。

## Qwen / Llama 切换

各训练脚本在 `python` 命令里直接写 `--model_path`；改对应脚本中的该行即可切换模型家族，chat template 与 response mask 由 tokenizer 自动推断：

```bash
# 例如在 scripts/catlora_simple.sh 里把
#   --model_path "Qwen/Qwen3-8B"
# 改成
#   --model_path "meta-llama/Llama-3.1-8B-Instruct"
```

常用 CLI（默认值见各 `.sh` 里的 python 命令）：

| CLI | 典型默认 | 用途 |
|---|---|---|
| `--model_path` | `Qwen/Qwen3-8B` | Cat / Block 蒸馏基座模型 |
| `--distill_model_max_length` | `8192` | Cat 蒸馏序列长度 |
| `--model_max_length` | `8192` | E2E 序列长度 |
| `--distill_dataset` / `--dataset_mix` | 见上 mix 串 | 蒸馏 / E2E 数据配方 |

Llama gated 模型需自行配置 `HF_TOKEN` 或训练 CLI 的 `--access_token`。

Block 蒸馏现已支持 Qwen3/Qwen2 与 Llama decoder（`train_utils/block_distill.py` 的 `validate_block_distill_model` / `run_decoder_block`）。

## 序列长度

EdgeRazor 论文（Appendix D.1）对所有已验证模型统一 **truncate 到 1024**。Qwen3-8B / Llama-3.1-8B 属于未在论文中验证的外推；当前脚本默认 **8192**（Cat 用 `--distill_model_max_length`，E2E 用 `--model_max_length`）。

覆盖示例：直接改脚本里对应 CLI 行，例如 `--distill_model_max_length "1024"` 或 `--model_max_length "2048"`。

## Cat 多卡蒸馏（DDP）

LoRA 蒸馏子阶段支持 `torchrun` DDP；**不新增蒸馏 CLI**，只改 shell 启动方式：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nproc_per_node=4 python tools/cat_distill_from_vae_checkpoint.py \
  --distill_batch_size "default=1" \
  ... # 其余参数与单卡相同
```

- `--distill_batch_size` 为**每卡** batch；global batch = `batch_size × GPU 数 × distill_gradient_accumulation_steps`
- VAE 训练 / activation calib 仍单卡；推荐仅 checkpoint 蒸馏脚本使用 `torchrun`
- **数据加载**：EdgeRazor 式 lazy Dataset（`load_dataset` 读 jsonl，`__getitem__` 内 tokenize）；`interleave_datasets` 按 mix 权重采样；并行靠 `dataloader_num_workers`（默认 16），无 rank-0 预处理 cache
- **训练量**：由 `--distill_steps` 控制
- **类别后 eval**：多卡 checkpoint 蒸馏下若设置 `--eval_tasks`，lm_eval 会按 rank 切分任务并行评测，rank0 汇总指标；所有 rank 在 eval 阶段同步，避免 NCCL 超时

## 蒸馏损失（EAKLD）

Cat 默认纯 EAKLD（`eakld`）；E2E / Block 仍用 `eakld_kd` 或 attention EAKLD：

| 链路 | 默认 loss | confidence_k CLI |
|---|---|---|
| Cat LoRA | `--distill_loss_type default=eakld` | `--distill_eakld_confidence_k 16` |
| E2E stage1/decoder | `--loss_type eakld_kd` | `--eakld_confidence_k 16` |
| Block attention KL | `--block_distill_entropy_aware_kl true` | `--block_distill_eakld_confidence_k 16` |

`eakld` / `eakld_kd` / `eakld_top[_K]` 对齐 `loss.py`：reverse KL 手写 `Σ p_s(log p_s - log p_t)`（避免 `F.kl_div` 不对 target 反传），γ 用全词表教师熵的全局 token 均值 / `log(confidence_k)`，KL 内乘 `T²`。`eakld` 为全词表 CAKLD；`eakld_top_K` / `eakld_topk[_K]` 的 FKL/RKL 分别取教师/学生 top-k；`eakld_kd` 为 `(1-α)·CE + α·CAKLD`，外侧不再额外乘 `T²`。

8K 全词表 EAKLD 建议开启 `--distill_teacher_logits_cpu_staging true`（默认已开）：teacher logits 暂存 CPU，算 loss 前整段搬回 GPU，数学等价、降低双 forward 峰值显存。

`confidence_k=16` 是**熵归一化常数 K**（`γ = 1 - H/log(K)`，再 clamp），**不是** vocab top-k。`eakld` 的 KL 在全词表；`eakld_top_K` 的 KL 在 top-k。

旧 loss（`kd_top_1000` 等）仍可用，例如：

```bash
--distill_loss_type default=kd_top_1000 bash scripts/catlora_simple.sh
```

LAFD 式 hidden 对齐：已有 `--distill_hidden_alignment_layer_weighting adaptive_top_3`，可与 EAKLD 叠加。

## 数据准备

在已激活 `bitvae` 的 shell 中执行（**无需**克隆 EdgeRazor 仓库）：

```bash
bash scripts/download_distill_dataset.sh
```

等价于 `bash scripts/prepare_vaellm_edgerazor_data.sh`。

流程：

1. `tools/prepare_edgerazor_distill_jsonl.py` 从 HuggingFace 下载并生成四条通用 instruct jsonl；
2. `tools/prepare_vaellm_task_mix.py` 生成 `task_vaellm_eval_instruct.jsonl`。

环境变量：

- `OUTPUT_DIR`：输出目录，默认 `data/edgerazor_qwen3`
- `HF_TOKEN`：`BAAI/Infinity-Instruct`（ii_7M / ii_gen）为 **gated 数据集**，需先在 [HF 数据集页](https://huggingface.co/datasets/BAAI/Infinity-Instruct) 申请访问，再配置有效 token（或 `huggingface-cli login`）
- `MAX_SAMPLES`：设为正整数时每个 general jsonl 只取 N 条（smoke）
- `MAX_SAMPLES_PER_TASK`：设为正整数时 task jsonl 每任务只取 N 条（smoke）

下载脚本默认直连官方 HuggingFace Hub（`https://huggingface.co`），不使用镜像站。

仅生成通用语料（调试用）：

```bash
python tools/prepare_edgerazor_distill_jsonl.py --output_dir data/edgerazor_qwen3 --skip_existing
python tools/prepare_edgerazor_distill_jsonl.py --output_dir data/edgerazor_qwen3 --datasets am_1.4M_instruct.jsonl --force
```

仅生成下游任务切片：

```bash
python tools/prepare_vaellm_task_mix.py --output_dir data/edgerazor_qwen3
```

小规模 smoke：

```bash
MAX_SAMPLES=100 MAX_SAMPLES_PER_TASK=10 bash scripts/download_distill_dataset.sh
```

## 训练脚本

- Cat 蒸馏：`scripts/catlora_simple.sh`、`scripts/catlora_distill_from_checkpoint.sh`
- Block 蒸馏：`scripts/block_vae_lora_simple.sh`
- E2E：`compressed_e2e_fintuning/scripts/e2e_stage1_pretrain.sh`、`e2e_stage2_instruct.sh`、`e2e_decoder.sh`

Cat 蒸馏序列长度由 `--distill_model_max_length` 控制（脚本中常见 4096 或 8192）。

### Truncation vs padding

- **Per-sample truncation**：`encode_edgerazor_messages_record()` / `encode_text_lm_record()` 在 collator 之前把单样本有效 token 截到配置的 max length（Cat：`--distill_model_max_length`；E2E：`--model_max_length`）。
- **Batch collation padding**：由共享 `build_edgerazor_data_collator()` 完成。
  - 固定模式（默认）：pad 到配置的 max length。
  - 动态模式（`--distill_dynamic_padding true` / `--dynamic_padding true`）：pad 到本 micro-batch 最长样本，再向上对齐到 8 的倍数。
- 新增 padding 位置固定满足：`input_ids=pad_token_id`、`attention_mask=0`、`labels=-100`，因此不进入 CE/KD/hidden alignment/token telemetry 的有效区域。

## LongBench 说明

`longbench` 使用 `THUDM/LongBench` 各子任务 test split 转成 `messages`。这部分与 LongBench eval 存在分布重叠，文档中仅作 eval-aware 微调用途。
