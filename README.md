# VAELLM

这个仓库当前主要用于大语言模型 `nn.Linear` 的类别压缩、`VAELinear` 替换、e2e LoRA 微调和评估。

## 环境

所有 Python 命令、测试和训练脚本都应该在 `bitvae` conda 环境中运行。

```bash
conda activate bitvae
export PYTHONPATH=.
```

## 主要入口

- `scripts/catlora_simple.sh`
  - 入口：`tools/cat_train.py`
  - 用途：按类别训练权重 VAE、把 `nn.Linear` 替换成 `VAELinear`、保存压缩模型
  - residual / low-rank 离群保护逻辑在 `train_utils/cat_train_residual_protection.py`
  - joint decoder 联合微调代码已关闭，旧实现仅以注释形式保留在 `train_utils/cat_joint_decoder.py`
- `scripts/catlora_distill_from_checkpoint.sh`
  - 入口：`tools/cat_distill_from_vae_checkpoint.py`
  - 用途：从已经完成 VAE 压缩的 cat checkpoint 继续逐类别蒸馏，不重新训练 VAE
  - 支持从 `after_<category>/` 续跑未完成类（`--distill_reset_completed false`）
  - 支持在已蒸馏 ckpt 上再蒸一轮，含用已有 `low_rank_a/b` 初始化 LoRA 并覆盖写回（`--distill_reset_completed true`）
  - 用法见 `docs/catlora_distill_from_checkpoint.md`
  - 多卡示例：`scripts/catlora_distill_4gpu_res0.sh`
- `scripts/eval.sh`
  - 入口：`tools/cat_eval.py`
  - 用途：对保存好的 checkpoint 做 PPL 和 lm-eval
- `compressed_e2e_fintuning/scripts/e2e_decoder.sh`
  - 压缩模型 checkpoint 直接训练 VAELinear 内部参数，最终仍保存压缩 `final_model/`
  - 入口参数是 `--student_checkpoint_dir`，不接受 `--student_model_path`
  - 支持 `--finetune_mode decoder|lora|both`
  - `lora` 模式训练 checkpoint 内已有 `low_rank_a/b` 并写回压缩模型，不保存 PEFT adapter
  - 三种模式说明见 `compressed_e2e_fintuning/README.md`
- `compressed_e2e_fintuning/scripts/e2e_choice_kd.sh`
  - 压缩模型 MCQA choice-KD 端到端微调脚本
- `tools/convert_legacy_checkpoint.py`
  - 把旧格式 e2e checkpoint 转成当前紧凑格式
- `tools/convert_cat_checkpoint_to_bitpack.py`
  - 把旧版 cat checkpoint（`vq_weight` 仍是 `torch.bool` 落盘）转换成当前 packed 格式
- `scripts/cat_train_args.md`
  - `cat_train.py` 主要参数说明

## 推荐流程

1. 先做类别压缩，产出 student checkpoint：

```bash
bash scripts/catlora_simple.sh
```

2. 如果需要继续端到端微调压缩 checkpoint，使用 compressed e2e：

```bash
bash compressed_e2e_fintuning/scripts/e2e_decoder.sh
```

3. 最后评估：

```bash
bash scripts/eval.sh
```

如果你手头是旧版 cat checkpoint，需要先转成当前 packed 格式再评估：

```bash
python tools/convert_cat_checkpoint_to_bitpack.py \
  --input .result/your_cat_run/final_model \
  --output .result/your_cat_run/final_model_packed
```

## 说明

- 当前仓库不再使用旧 README 里那些不存在的脚本：
  - `scripts/prepare.sh`
  - `scripts/release/*`
  - `scripts/lbl_train_tools.sh`
  - `scripts/train_linear_by_category.sh`
- `cat_train -> compressed_e2e_fintuning -> cat_eval` 是压缩模型的推荐链路。
- 当前 `cat_train` 保存的 `final_model/` 是 packed cat checkpoint：
  - `checkpoint_meta.json` 版本为 `5`
  - `vq_weight*` 现在按 `uint8 bit-pack` 落盘，不再按 `torch.bool` 一字节存
  - checkpoint meta 会记录 `low_rank_a/b` 和 sparse residual payload 的 shape / dtype
  - 这里只压缩 VQ bit payload；`embed_tokens`、`lm_head` 等未压缩 dense 权重仍会保留
- `cat_train` 当前支持 4 种离群保护模式：
  - `none`：不做离群保护
  - `channel`：压缩前保护 top-N channel
  - `channel_residual_vae`：主 VAE 压缩完整权重，训练后对选中 channel 的 residual 额外做多阶 VAE patch
  - `residual_sparse`：训练后保存稀疏残差补丁
- 推理时 `VAELinear` 的权重重建顺序固定为：
  - `VAE reconstruction -> protected_channel_residual_vae patch -> low_rank patch -> sparse_residual patch`
- `compressed_e2e_fintuning` 是唯一的压缩 checkpoint 端到端训练入口：
  - `decoder`：训练 VAELinear decoder
  - `lora`：训练已有低秩分支并写回 `low_rank_a/b`
  - `both`：同时训练 decoder 和低秩分支
- `compressed_e2e_fintuning` 支持 `--dataset_num_proc`：
  - 只影响数据预处理阶段的 `datasets.map(num_proc=...)`
  - 不影响 DataLoader worker，不影响训练并行
  - 多卡时用 `main_process_first` 让主进程先构建 datasets cache，其余 rank 复用
  - `--eval_strategy no` 时不会再预处理 eval 数据
- 新版 cat checkpoint 加载器只接受 packed 格式：
  - 旧版 `version=4` cat checkpoint 不再直接加载
  - 需要先执行 `python tools/convert_cat_checkpoint_to_bitpack.py ...`
  - 这个转换脚本是独立工具，不参与运行时 import，删掉它不影响新格式 checkpoint 的训练、保存、加载和评估
- `train_utils` 当前按扁平模块组织：
  - `cat_train_args.py`：`tools/cat_train.py` 参数解析和类别 override
  - `cat_data_prep.py` / `cat_train_data.py`：权重切分、block 数据构造和恢复；排序代码已关闭
  - `cat_train_residual_protection.py`：sparse residual protection
  - `cat_joint_decoder.py`：joint decoder 旧实现，已整体注释关闭
  - `model_checkpoint_io.py`：压缩模型 checkpoint 保存和加载
  - `lora_*`：after-category LoRA 数据、训练和融合
  - `eval_utils.py` / `cat_train_eval.py`：PPL、lm_eval、MSE 评估
- `tools/cat_eval.py` 支持 `--adapter_dir`：
  - 不传 `--adapter_dir`：评估压缩模型原始结果
  - 传 `--adapter_dir`：先从压缩模型重建 dense，再挂载并 merge adapter，评估端到端微调结果
- 历史 `e2e_fintuning` 已移除；旧 checkpoint 需要先执行：
  - `python -m tools.convert_legacy_checkpoint ...`
- `tools/cat_eval.py`、`tools/collect_activation_absmax.py`、激活统计相关工具现在要求设备配置和实际硬件一致；请求 CUDA 但机器没有 CUDA 时会直接报错，不再静默回退到 CPU。

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
