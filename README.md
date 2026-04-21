# VAELLM

这个仓库当前主要用于大语言模型 `nn.Linear` 的类别压缩、`VAELinear` 替换、e2e LoRA 微调和评估。

## 环境

所有 Python 命令、测试和训练脚本都应该在 `bitvae` conda 环境中运行。

```bash
conda activate bitvae
export PYTHONPATH=.
```

如果你不想切环境，也可以直接用：

```bash
/home/shaoyuantian/anaconda3/bin/conda run -n bitvae python ...
```

## 主要入口

- `scripts/catlora_simple.sh`
  - 入口：`tools/cat_train.py`
  - 用途：按类别训练权重 VAE、把 `nn.Linear` 替换成 `VAELinear`、保存压缩模型
- `scripts/eval.sh`
  - 入口：`tools/cat_eval.py`
  - 用途：对保存好的 checkpoint 做 PPL 和 lm-eval
- `e2e_fintuning/scripts/e2e_lora.sh`
  - 单卡 Alpaca-GPT4 蒸馏基线
  - `teacher_model_path` 不显式传，默认从 student checkpoint meta 推断
  - `save_strategy no`，默认只保存最终导出
  - `unload_vae_original_weights_on_save false`
- `e2e_fintuning/scripts/e2e_lora_openorca.sh`
  - 双卡 OpenOrca 蒸馏
  - 显式传 `teacher_model_path`
  - `model_max_length 4096`
  - `unload_vae_original_weights_on_save true`
  - 当前脚本保持 HF 默认，不额外启用梯度检查点
- `e2e_fintuning/scripts/e2e_lora_fineweb_edu.sh`
  - 单卡 FineWeb-Edu 蒸馏
  - `dataset_config_name sample-10BT`
  - `teacher_model_path` 不显式传，默认从 student checkpoint meta 推断
  - `save_strategy no`
  - `unload_vae_original_weights_on_save false`
- `e2e_fintuning/scripts/e2e_lora_redpajama.sh`
  - 四卡 RedPajama SFT
  - `save_strategy steps`
  - `unload_vae_original_weights_on_save false`
- `e2e_fintuning/scripts/e2e_lora_mix.sh`
  - 单卡 mixed-dataset 蒸馏基线
  - 默认混合池：`openorca=0.45,fineweb_edu=0.30,race=0.15,sciq=0.07,alpaca=0.03`
  - 通过 `--dataset_mix` 一次训练完成单次混合采样
- `raw_e2e_fintuning/scripts/e2e_raw_lora.sh`
  - 原模型（非 VAE）LoRA 基线
  - 入口参数是 `--student_model_path`，不接受 `--student_checkpoint_dir`
  - LoRA/AdaLoRA 参数名用 `--lora_*`、`--adalora_*`，不再使用 `--vae_*`
  - 最终保存为 HF/PEFT 目录：`final_adapter/`、`run_meta.json`
- `raw_e2e_fintuning/scripts/e2e_raw_lora_mix.sh`
  - 原模型（非 VAE）mixed-dataset LoRA 基线
  - 默认混合池：`openorca=0.45,fineweb_edu=0.30,race=0.15,sciq=0.07,alpaca=0.03`
  - LoRA/AdaLoRA 参数名用 `--lora_*`、`--adalora_*`，不再使用 `--vae_*`
  - 同样保存为 HF/PEFT 格式
- `e2e_fintuning/scripts/export_final_model.sh`
  - 从中间 `checkpoint-*` 目录重新导出最终模型
- `e2e_fintuning/scripts/convert_legacy_checkpoint.sh`
  - 把旧格式 e2e checkpoint 转成当前紧凑格式
- `scripts/cat_train_args.md`
  - `cat_train.py` 主要参数说明

## 推荐流程

1. 先做类别压缩，产出 student checkpoint：

```bash
bash scripts/catlora_simple.sh
```

2. 再选一个 e2e 脚本继续训练：

```bash
bash e2e_fintuning/scripts/e2e_lora.sh
```

或：

```bash
bash e2e_fintuning/scripts/e2e_lora_openorca.sh
bash e2e_fintuning/scripts/e2e_lora_fineweb_edu.sh
bash e2e_fintuning/scripts/e2e_lora_redpajama.sh
bash e2e_fintuning/scripts/e2e_lora_mix.sh
```

如果你要直接训练原始模型（不经过 VAE 压缩）：

```bash
bash raw_e2e_fintuning/scripts/e2e_raw_lora.sh
# 或
bash raw_e2e_fintuning/scripts/e2e_raw_lora_mix.sh
```

3. 最后评估：

```bash
bash scripts/eval.sh
```

## 说明

- 当前仓库不再使用旧 README 里那些不存在的脚本：
  - `scripts/prepare.sh`
  - `scripts/release/*`
  - `scripts/lbl_train_tools.sh`
  - `scripts/train_linear_by_category.sh`
- `cat_train -> e2e_fintuning -> cat_eval` 是当前推荐链路。
- 单源 e2e 脚本继续保留，适合做对照或单数据集实验。
- `e2e_lora_mix.sh` 走新的 `--dataset_mix` 入口，会对每个 source 分别做 pack，再按权重 interleave 成一个训练集。
- 旧轨（`e2e_fintuning/`）和新轨（`raw_e2e_fintuning/`）完全分开：
  - 旧轨输入：`--student_checkpoint_dir`（VAE checkpoint）
  - 旧轨输出：紧凑 checkpoint（含 `checkpoint_meta.json`）
  - 新轨输入：`--student_model_path`（原始 HF 模型）
  - 新轨输出：HF/PEFT（`final_adapter/` + `run_meta.json`，可选 `final_merged_model/`）
  - 两条轨的 checkpoint 格式不兼容，不能互相 resume。
- `tools/cat_eval.py`、`tools/collect_activation_absmax.py`、激活统计相关工具现在要求设备配置和实际硬件一致；请求 CUDA 但机器没有 CUDA 时会直接报错，不再静默回退到 CPU。

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
