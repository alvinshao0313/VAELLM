# VAELLM

## SC-CAT 项目级压缩流程标准

新模型的SC-CAT定制搜索与一阶段压缩，从[流程标准](docs/sc_cat_protocol.md)开始，按[执行与结果记录规范](docs/sc_cat_execution.md)完成逐类搜索、结果表、压缩恢复与验收。规则已确认，执行对齐和整链验证仍须按文档完成；具体任务及资源见[当前计划](docs/experience/records/w25_transfer/current_plan.md)。实验脚本的默认配置不代替标准搜索流程。

当前训练栈只有三条正式链路：

- CAT 类别压缩：`tools/cat_train.py`
- 已压缩模型的端到端恢复训练：`python -m compressed_e2e_fintuning.main`
- 已完成 VAE 压缩后的逐类别恢复：`tools/cat_distill_from_vae_checkpoint.py`

三条链路共享 `train_utils.config` 的数据、loss、目标范围和运行时配置。模型级 loss 支持 `sft`、`kl`、`kl_top`、`kl_top_partial`、`kl_top_mass`、`kl_top_mse`、`kd`、`kd_top`、`kd_top_partial`、`kd_top_mass`；top-k 通过独立的 `--top_k` 指定。完整校验以 `train_utils/config/configs.py` 的 `LOSS_TYPES` 为准。

## 运行环境

当前服务器用户为 `cmcc`，项目目录为 `/home/cmcc/VAELLM`，使用独立 conda 环境 `bitvae`。旧服务器用户为 `star`；历史命令中的模型、数据和输出路径必须按当前实际位置核对。

```bash
cd /home/cmcc/VAELLM
source /home/cmcc/miniconda3/etc/profile.d/conda.sh
conda activate bitvae
which python
python -V
```

新机器在安装 Miniconda 后，用以下命令创建环境并安装依赖。当前链路的版本清单是 [`requirements-runtime.txt`](requirements-runtime.txt)，包含原有 `requirements.txt`。`environment.yml` 是带旧用户绝对路径和额外工具包的历史完整快照，不直接用于当前机器重建。

```bash
conda create -y -n bitvae --override-channels -c conda-forge python=3.11.16 pip=25.1 setuptools=80.9.0 wheel=0.45.1
conda activate bitvae
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
DS_BUILD_OPS=0 python -m pip install -r requirements-runtime.txt
```

另需安装 FlashAttention 2.7.4.post1 和 fast-hadamard-transform 1.0.4.post1。下面的官方 FlashAttention wheel 对应 Linux x86_64、Python 3.11、PyTorch 2.6、CUDA 12、CXX11 ABI=false；先检查 `torch._C._GLIBCXX_USE_CXX11_ABI`。Hadamard 扩展使用本机 `nvcc` 编译，需 GCC/G++、CUDA Toolkit 和上述 PyTorch 已就绪；PyPI 上该版本源码包缺少 `csrc`，应使用上游 GitHub tag 源码。

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)'
python -m pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE MAX_JOBS=4 python -m pip install 'https://github.com/Dao-AILab/fast-hadamard-transform/archive/refs/tags/v1.0.4.post1.tar.gz' --no-build-isolation
python -m pip check
```

本机的 `decord==0.6.0` 安装包文件名声明 `py3`，内部平台标签却为 `cp36`，`pip check` 会报告该包不支持当前平台；本次在 Python 3.11 下已实际解码四帧视频。这一上游问题见 [decord issue #366](https://github.com/dmlc/decord/issues/366)，不表示其他依赖检查也可忽略。

新服务器可使用 [`tools/check_new_server.py`](tools/check_new_server.py) 检查训练栈、CUDA 扩展和多卡通信，具体命令与适用边界见[新服务器自检](docs/guides/research_handoff.md#新服务器自检)。本机 2026-10-30 的自检结果在 `result/cmcc_env_check_fixed_20261030/report.json`，不代表正式 8B 训练或完整评测通过。

PyTorch 安装命令见[官方旧版本说明](https://pytorch.org/get-started/previous-versions/)，扩展来自 [FlashAttention 官方发布](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1)和 [Hadamard 官方发布](https://github.com/Dao-AILab/fast-hadamard-transform/releases/tag/v1.0.4.post1)。环境搭建不包含模型权重、训练数据及 HF 评测缓存迁移；运行复现前仍需按[输入要求](docs/guides/qwen3_8b_69_reproduction.md)准备实际文件。

## 推荐流程

```bash
bash scripts/catlora_simple2.sh
bash compressed_e2e_fintuning/scripts/e2e_decoder.sh
python tools/cat_eval.py --checkpoint_dir .result/example/final_model \
  --eval_lm_eval --tasks boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu \
  --num_fewshot 0 --output_json .result/example/eval.json
```

CAT online 支持六种 `--after_category_mode`：

- `current_decoder`
- `current_lora`
- `current_lora_decoder`
- `remaining_lora`
- `remaining_lora_current_decoder`
- `remaining_lora_prefix_decoder`

checkpoint-distill 只支持前三种 current 模式。CAT/E2E 的模型级 LoRA 只支持 plain full-space LoRA。独立 block-VAE/block-distill 路径和 E2E stage1/stage2 脚本已经删除。

## 目标范围

- CAT 用 `--compression_categories`、`--target_layers` 和 `--skip_layers` 定义压缩集合。
- E2E 用 `--target_layers` 和 `--target_modules` 定义训练集合。
- `skip_layers` 只允许引用已发现的 CAT projection inventory；不会把普通 dense Linear 当作压缩目标。

## Checkpoint v6

所有正式读写统一使用 `vaellm_model_checkpoint_v6`，schema version 为 6：

- `training_step`：用于精确续训，引用不可变的 `round_base`，并保存 optimizer/scheduler/RNG 等可变状态。
- `round_base`：一轮训练的独立完整基座。
- `category_boundary`：CAT 完成一个精确类别前缀后的稳定完整模型。
- `final_model`：完成 finalize 后可独立加载的稳定模型。

稳定 checkpoint 已把 LoRA 写回每个目标的 `low_rank_a/b`；拓扑由各目标 payload shape 定义，metadata 中 `lora_config` 必须为 `null`。旧 checkpoint 不再被正式训练/评估入口直接加载，先执行：

```bash
python tools/migrate_checkpoint_v6.py \
  --source /path/to/legacy/checkpoint \
  --output_dir /path/to/new/v6/checkpoint \
  --dry_run false
```

`--dry_run true` 只检查可迁移性，不写文件。specialized bitpack/candidate 工具保留自己的紧凑 artifact schema，但完整模型输入输出仍走 v6。

详细参数见 [CAT 参数](docs/guides/cat_training.md)、[checkpoint-distill](docs/guides/catlora_distill_from_checkpoint.md) 和 [E2E](compressed_e2e_fintuning/README.md)。

## 项目文档

使用指南、模块说明、经验、实验结果和历史材料统一从 [文档入口](docs/README.md) 查找。维护方式见 [文档管理规范](docs/MANAGEMENT.md)。

## 经验与实验记录

实验设计前先检索 [经验库](docs/experience/README.md)，阅读相关经验及来源，避免重复踩坑。经验、原始结果、历史方案分别维护；完整来源见 [目录](docs/experience/CATALOG.md)。
