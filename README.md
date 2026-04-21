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
- `dense_e2e_fintuning/scripts/e2e_dense_lora_mix.sh`
  - 压缩模型 checkpoint（`S`）重建 dense 模型（`C`）后做标准 PEFT 蒸馏
  - 入口参数是 `--student_checkpoint_dir`，不接受 `--student_model_path`
  - 默认混合池：`openorca=0.45,fineweb_edu=0.30,race=0.15,sciq=0.07,alpaca=0.03`
  - 导出两份产物：`final_adapter/` 与回写后的紧凑 `final_model/`
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
- `tools/convert_legacy_checkpoint.py`
  - 把旧格式 e2e checkpoint 转成当前紧凑格式
- `scripts/cat_train_args.md`
  - `cat_train.py` 主要参数说明

## 推荐流程

1. 先做类别压缩，产出 student checkpoint：

```bash
bash scripts/catlora_simple.sh
```

2. 如果是压缩 checkpoint（`S`），使用 dense e2e 继续训练：

```bash
bash dense_e2e_fintuning/scripts/e2e_dense_lora_mix.sh
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
- `cat_train -> dense_e2e_fintuning -> cat_eval` 是压缩模型的推荐链路。
- `raw_e2e_fintuning` 是原始模型独立训练链路，输入输出保持 HF/PEFT 格式。
- 两条训练轨完全隔离，不互相 import：
  - `dense_e2e_fintuning`：输入 `--student_checkpoint_dir`，输出 `final_adapter/` + 回写后的 `final_model/`
  - `raw_e2e_fintuning`：输入 `--student_model_path`，输出 `final_adapter/` + `run_meta.json`（可选 `final_merged_model/`）
- 历史 `e2e_fintuning` 已移除；旧 checkpoint 需要先执行：
  - `python -m tools.convert_legacy_checkpoint ...`
- `tools/cat_eval.py`、`tools/collect_activation_absmax.py`、激活统计相关工具现在要求设备配置和实际硬件一致；请求 CUDA 但机器没有 CUDA 时会直接报错，不再静默回退到 CPU。

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
