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
- `dense_e2e_fintuning/scripts/e2e_dense_lora.sh`
  - 压缩模型 checkpoint（`S`）重建 dense 模型（`C`）后做标准 PEFT 蒸馏
  - 入口参数是 `--student_checkpoint_dir`，不接受 `--student_model_path`
  - `--decode_device auto` 会按当前进程可见设备解析：单卡可见时用 `cuda:0`，多卡可见时按 `LOCAL_RANK` 选卡
  - `--dataset_num_proc` 控制数据预处理阶段的 `datasets.map(num_proc=...)`
  - 多卡下只有主进程先做数据预处理并写 cache，其他 rank 等待后复用
  - `--eval_strategy no` 会直接跳过 eval 数据的 prepare/tokenize/pack
  - 默认是单数据源脚本（可按需改 `--dataset_mix`）
  - 默认导出 `final_adapter/` + `run_meta.json` + `final_adapter/dense_adapter_meta.json`
- `raw_e2e_fintuning/scripts/e2e_raw_lora.sh`
  - 原模型（非 VAE）LoRA 基线
  - 入口参数是 `--student_model_path`，不接受 `--student_checkpoint_dir`
  - LoRA/AdaLoRA 参数名用 `--lora_*`、`--adalora_*`，不再使用 `--vae_*`
  - 最终保存为 HF/PEFT 目录：`final_adapter/`、`run_meta.json`
- `raw_e2e_fintuning/scripts/e2e_raw_lora_mix.sh`
  - 原模型（非 VAE）mixed-dataset LoRA 基线
  - 默认混合池：`openorca=0.45,fineweb_edu=0.30,race=0.15,sciq=0.07,alpaca=0.03`
  - LoRA/AdaLoRA 参数名用 `--lora_*`、`--adalora_*`，不再使用 `--vae_*`
  - 同样支持 `--dataset_num_proc`、主进程优先预处理、以及 `--eval_strategy no` 跳过 eval 预处理
  - 同样保存为 HF/PEFT 格式
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

2. 如果是压缩 checkpoint（`S`），使用 dense e2e 继续训练：

```bash
bash dense_e2e_fintuning/scripts/e2e_dense_lora.sh
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
- `cat_train -> dense_e2e_fintuning -> cat_eval` 是压缩模型的推荐链路。
- 当前 `cat_train` 保存的 `final_model/` 是 packed cat checkpoint：
  - `checkpoint_meta.json` 版本为 `5`
  - `vq_weight*` 现在按 `uint8 bit-pack` 落盘，不再按 `torch.bool` 一字节存
  - 这里只压缩 VQ bit payload；`embed_tokens`、`lm_head` 等未压缩 dense 权重仍会保留
- `raw_e2e_fintuning` 是原始模型独立训练链路，输入输出保持 HF/PEFT 格式。
- 两条训练轨完全隔离，不互相 import：
  - `dense_e2e_fintuning`：输入 `--student_checkpoint_dir`，输出 `final_adapter/` + `run_meta.json`
  - `raw_e2e_fintuning`：输入 `--student_model_path`，输出 `final_adapter/` + `run_meta.json`（可选 `final_merged_model/`）
- `dense_e2e_fintuning` 可以直接加载当前 `tools/cat_train.py` 产出的 packed cat checkpoint；部分历史 decoder key 布局在加载时仍会自动 remap。
- `dense_e2e_fintuning` 的 `decode_device=auto` 不再等价于“统一 0 卡”：
  - 当前进程只看见 1 张卡：解析到 `cuda:0`
  - 当前进程看见多张卡：解析到 `cuda:{LOCAL_RANK}`
  - 如果想强制固定设备，请显式传 `cuda:N`
- `dense_e2e_fintuning` 和 `raw_e2e_fintuning` 现在都支持 `--dataset_num_proc`：
  - 只影响数据预处理阶段的 `datasets.map(num_proc=...)`
  - 不影响 DataLoader worker，不影响训练并行
  - 多卡时用 `main_process_first` 让主进程先构建 datasets cache，其余 rank 复用
  - `--eval_strategy no` 时不会再预处理 eval 数据
- 新版 cat checkpoint 加载器只接受 packed 格式：
  - 旧版 `version=4` cat checkpoint 不再直接加载
  - 需要先执行 `python tools/convert_cat_checkpoint_to_bitpack.py ...`
  - 这个转换脚本是独立工具，不参与运行时 import，删掉它不影响新格式 checkpoint 的训练、保存、加载和评估
- `tools/cat_eval.py` 支持 `--adapter_dir`：
  - 不传 `--adapter_dir`：评估压缩模型原始结果
  - 传 `--adapter_dir`：先从压缩模型重建 dense，再挂载并 merge adapter，评估端到端微调结果
- 历史 `e2e_fintuning` 已移除；旧 checkpoint 需要先执行：
  - `python -m tools.convert_legacy_checkpoint ...`
- `tools/cat_eval.py`、`tools/collect_activation_absmax.py`、激活统计相关工具现在要求设备配置和实际硬件一致；请求 CUDA 但机器没有 CUDA 时会直接报错，不再静默回退到 CPU。

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
