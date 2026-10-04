# 从 BF16 Qwen3 8B 复现八任务均分 69 以上的压缩模型

入口是 [`scripts/reproduce_qwen3_8b_69_from_bf16.sh`](../../scripts/reproduce_qwen3_8b_69_from_bf16.sh)。它从原始 BF16 `Qwen3-8B` 开始，自动完成逐类别压缩、必要的恢复训练、两次类别组装、最终训练及完整八任务评测。**不需要任何历史压缩 checkpoint，也不读取旧实验目录作为训练输入。**

复现目标是已经测得的 **八任务均分 69.3129801840%、投影权重 2.4569887908 bit/weight、无外置 LoRA** 的方案。历史最佳成品位于最后一轮的第 1200 步；新训练会生成新的模型身份，是否再次达到 69% 以本次生成的 `acceptance.json` 为准。此脚本准备工作没有重新执行完整压缩和训练，不能把历史成绩当作新运行的实测结果。

## 最快启动方法

在当前服务器中，激活环境后运行下面的后台命令。`--gpus 0,1` 是示例，请换成两张空闲的 A800 80GB；两张卡都参与计算。

```bash
cd /home/star/VAELLM
source /home/star/miniconda3/etc/profile.d/conda.sh
conda activate bitvae
which python
python -V
nvidia-smi

RUN_DIR="result/qwen3_8b_69_from_bf16_$(date +%Y%m%d_%H%M%S)"
mkdir -p result
(
  set -o noclobber
  OUTPUT_DIR="$RUN_DIR" nohup bash scripts/reproduce_qwen3_8b_69_from_bf16.sh \
    --gpus 0,1 > "$RUN_DIR.log" 2>&1 < /dev/null
) &
```

已有 tmux 会话时，也可以在其中直接运行：

```bash
bash scripts/reproduce_qwen3_8b_69_from_bf16.sh --gpus 0,1
```

脚本默认从 `/home/star/models/Qwen/Qwen3-8B` 读取 BF16 模型，并自动使用带时间戳的新输出目录。需要改变路径时，只修改输入与输出位置：

```bash
MODEL_PATH=/path/to/Qwen3-8B \
OUTPUT_DIR=result/my_qwen3_69_reproduction \
bash scripts/reproduce_qwen3_8b_69_from_bf16.sh --gpus 4,5
```

`OUTPUT_DIR` 必须不存在。脚本不覆盖已有结果，不自动重试，也不从旧任务断点恢复。源码和数据准备好后即可启动，不要求先重复整套短测。

## 运行前需要哪些东西

| 输入 | 要求 |
| --- | --- |
| 原始模型 | 完整 BF16 Qwen3-8B 的 config、tokenizer、safetensors index 及全部权重分片；不能传压缩后的 `final_model`。历史模型 revision 为 `b968826d9c46dd6066d109eabc6255188de91218`。 |
| 代码 | 当前工作区的完整实现，包括本次复现入口、计算核修复、组装和导出模块；只检出早期 commit 再复制 `.sh` 文件不够。 |
| 环境 | 已配置的 `bitvae`。核查版本：Python 3.11.16、PyTorch 2.6.0+cu124、Transformers 4.51.0、Triton 3.2.0、FlashAttention 2.7.4.post1、lm-eval 0.4.8。 |
| 训练数据 | 下表中的五份 JSONL。路径由项目的 dataset alias 解析，必须从仓库根目录运行。 |
| 校准与评测数据 | 本地 HF 缓存中的 `vicgalle/alpaca-gpt4`，以及 BoolQ、RTE、Winogrande、ARC、OpenBookQA、PIQA、MMLU 对应评测数据。 |
| GPU | 两张空闲 A800 80GB。每个训练子进程只看见一张卡，两个独立分支会同时运行；不要用 `torchrun` 包装此入口。 |
| 磁盘 | 除已有原模型和数据外，还需保存本次 CAT 产物、多阶段模型与训练状态；每份部署模型约 4.31 GiB，完整工作目录远大于一份成品。不要按最终模型大小估算全流程空间。 |

五份训练文件如下。末轮虽然只用任务数据，前面的 CAT 和混合数据恢复仍需要另外四份。

```text
data/edgerazor_qwen3/ii_7M_instruct.jsonl
data/edgerazor_qwen3/ii_gen_1.4M_instruct.jsonl
data/edgerazor_qwen3/tulu_0.6M_instruct.jsonl
data/edgerazor_qwen3/am_1.4M_instruct.jsonl
data/edgerazor_qwen3/task_vaellm_eval_instruct.jsonl
```

任务文件必须是已修复的 **176,597 行、191,870,906 bytes** 版本，SHA256 为：

```text
894a04aea907efebd8b0f52f040c5635750bb30dfb64d9c34e30a97421a155a1
```

脚本使用离线模式。缺少模型、JSONL 或缓存时，需要先按[数据接入说明](edgerazor_dataset.md)准备输入；脚本不会联网下载、安装依赖或改用另一批数据。跨服务器迁移需要同时准备原模型、训练数据和评测缓存，Git 仓库本身不包含这些大文件。

如果只想检查本地输入，不启动 GPU 训练，可以执行：

```bash
bash scripts/reproduce_qwen3_8b_69_from_bf16.sh --check_inputs
```

这项检查验证原模型目录、五份训练文件和任务文件身份，并报告环境版本；不会加载整模型做前向，也不等于校准/评测缓存或完整训练已通过运行验证。

## 脚本实际执行的路线

这是已知达标成品的必要来源链。表内分数是历史参考点，用来解释为什么保留这些步骤；新流程不要求中间分数逐位相同，也不会用它们代替最终验收。

| 顺序与目录 | 执行内容 | 来源与产物 |
| --- | --- | --- |
| `02_format_data` | 用原任务 JSONL 和原 tokenizer 准备固定格式数据 | 1024 条训练序列、32 条留出序列；选择规则固定，不使用模型分数筛样本。编号表示所属阶段，不表示这一准备步骤晚于 CAT。 |
| `01_cat` | 按 q/k/v/o/gate/up/down 七类压缩；每类完成后恢复已压缩 decoder，并训练、融合未压缩层的临时 rank12 LoRA | 原 BF16 → 完整 252 个投影的压缩底座；历史 66.5762%。成品不留外置 LoRA。 |
| `03_ce10` 与 `04_ce50` | 从同一 CAT 底座，分别运行两条 2000 步 decoder 恢复；两卡并行 | CE10 使用 task50 混合数据；CE50 使用 task100 数据。两条分支都冻结硬码。 |
| `05_ce50_attention_ce10_mlp` | 以 CE50 为 base，只用 CE10 的 gate/up/down 替换 MLP 类别 | 历史 67.1303% 的组装方式；attention 与 MLP 来源不同。 |
| `06_decoder_kl` 与 `07_chatml_joint256` | 从上一行的同一组装成品分别训练；两卡并行 | 一条是全 decoder 纯 KL 2000 步；另一条是固定 ChatML 数据上的第一层 down 联合训练 256 步。 |
| `08_joint_mlp_decoder_attention` | 以 ChatML 256 步成品为 base，换入纯 KL 2000 步成品的 q/k/v/o | 历史末轮起点 67.8045% 的组装方式。 |
| `09_joint1200` | 从本次组装成品重新初始化优化器，联合训练第一层 down 的 decoder 和全部码位 | 总调度仍是 2000 步，训练到 1200 步后保存并停止，再独立导出和审计。 |
| `evaluation` | 对导出模型进行完整 MMLU 与其余七任务评测；两卡并行 | 生成根目录的 `acceptance.json`，核对八任务均分、实际位宽与无 LoRA 约束。 |

两条组装规则已经写死在脚本中，不需要手工挑 donor。固定 ChatML 256 步必须从 `05_ce50_attention_ce10_mlp` 开始；它与 `06_decoder_kl` 是同源分支，不能改成先执行后者、再把其成品送给前者。

## 哪些参数已经固定

CAT 参数直接复用 [`cat_w25_remaining.sh`](../../scripts/cat_w25_remaining.sh)：q/k/v/gate/up 主码 2 bit，o/down 主码 3 bit；分组维度 D32、两 stage、decoder H128、无残差块；双侧 block32 旋转；INT8 输入通道保护为普通类别 64、down 256。VAE 每个类别配置为 10000 步，类别间恢复为 500 步。

后续训练的完整参数集中在 [`reproduction_recipe.py`](../../experiments/qwen3_w25/reproduction_recipe.py)，末轮关键设置如下：

| 参数 | 值 |
| --- | --- |
| 更新范围 | `model.layers.0.mlp.down_proj` 的 6 个 decoder 张量和两个码库；其余权重冻结 |
| 训练方式 | `decoder_sparse_bit`，FP32 decoder 参数、BF16 计算和成品 |
| 数据 | `vaellm_eval_task=1`，原生 SFT 数据流；最大长度 1024、动态 padding。CLI 保留 `group_by_length=true`，但原生 Iterable 数据路径会将其关闭，不能改成另一种长度采样方式。 |
| 损失 | `kd_top_mass`、alpha=0.5，即 0.5 CE + 0.5 mass-KL；top-k=100、temperature=1、prompt 权重 0 |
| 批量 | 单卡 batch4，梯度累积 1 |
| decoder 学习率 | 1e-5，cosine，按 2000 总步数计算，warmup20 |
| 码位更新 | Adam，LR=4e-5，active ratio=1，`decoder_sensitivity`，round steps=512 |
| 其他 | seed/data_seed=0，weight decay=0，max grad norm=1.5；norm/head 冻结、无 residual LoRA |
| 停止与保存 | 仍每 400 步保存，在完成第 1200 次真实优化器更新后停止 |

固定 ChatML 256 步阶段使用它原有的训练循环和留出检查，保留中途提交码位、FP32 参数提升及冻结缓存处理。该阶段的 `FLASH_ATTENTION_DETERMINISTIC=1` 只设置在对应子进程中，末轮不继承这个设置。

## 为什么这个版本更快

脚本只执行达标成品真正依赖的分支，不重做失败方案搜索。CE10/CE50 和后续两个同源分支分别并行；中间阶段保留真实训练保存、独立导出、冻结状态和位宽审计，但省去每个保存点重复跑完整八任务。最终成品仍进行完整评测。

最后一轮只走原 2000 步学习率曲线的前 1200 步，省去历史上没有提高成绩的最后 800 步。**`--finish_step 1200` 与把 `--steps` 改为 1200 不同。** 后者会改变从开头开始的学习率轨迹，不能用于这次复现。

这些调整缩短的是等待、无须重复的评测和已知无收益的尾段；没有减少 CAT 训练预算、两条必要分支的 2000 步、训练批量或最终评测题数。本次未完整测量新编排的墙钟耗时，因此不承诺“几分钟得到模型”。先前约 79 分钟的记录只包含最后一轮，不能当作从 BF16 开始的总耗时。

## 在哪里拿模型与判断成功

主要输出如下：

```text
<OUTPUT_DIR>/
  final_model -> 09_joint1200/step_1200/final_model
  acceptance.json
  pipeline.json
  run.exit
  logs/                         # 各阶段命令、stdout/stderr 和退出码
  evaluation/mmlu.json
  evaluation/seven.json
  09_joint1200/stage_result.json
  09_joint1200/step_1200/audit.json
```

`final_model` 是指向本次成品的相对符号链接。需要移动或交付模型时，应复制其实际目录 `09_joint1200/step_1200/final_model` 的完整内容，不能只复制链接。它是项目的 v6 压缩格式，需通过本项目 loader 加载；当前 loader 仍依赖原 HF 模型目录。

运行中可按需查看日志：

```bash
tail -n 40 "$RUN_DIR.log"
cat "$RUN_DIR/pipeline.json"
```

完成后，检查根目录 `run.exit` 和 `acceptance.json`：

- `run.exit=0` 且 `acceptance.json` 中 `passed=true`：本次完整八任务均分达到 69%，且位宽和无 LoRA 约束通过。
- `run.exit=1` 且 `status=not_met`：训练、导出和评测已完成，但本次指标未达目标。应读取实际 `mean_percent` 和 `failure_reasons`，不能报告历史 69.313% 为本次成绩。
- `run.exit=2`：输入或某个阶段失败。查看 `pipeline.json` 的 `error` 及 `logs/` 中对应阶段日志；不自动改参数或覆盖目录重试。
- 尚无 `run.exit`：尚未正常记录结束；不能仅凭进程存在就认定训练成功。异常断电或强制结束也可能没有该文件。

验收固定为 0-shot、无样本上限、fast tokenizer、v6 strict 加载、BF16 解码缓存；MMLU batch64，其余七任务 batch32。BoolQ/RTE/Winogrande/MMLU 使用 `acc`，其余四项使用 `acc_norm`。MMLU 先按 57 学科的 14042 题加权，再与另外七项等权平均，总评测题数为 24742。

位宽从成品实际张量计算：

```text
投影平均位宽 = 码位、decoder、保护权重、scale、索引及旋转符号的总位数
             / 252 个原始投影的权重总数
历史结果     = 17,065,672,704 / 6,945,767,424
             = 2.4569887908 bit/weight
```

embedding、lm_head 和模型原有 norm 不在此分母中，完整权重文件约 4.625 GB。69.313% 是固定八任务均分，历史 MMLU 为 61.0241%；本指南没有把它解释成每个任务都达到 69%。

## 复现边界与实现核验

原方案没有启用全流程完全确定性，新运行的 UUID、精确末位乃至最佳保存点可能不同。脚本使用已知最佳的 1200 步路径；如果本次没有达到 69%，会如实返回未达标，不会自动挑另一套指标或补做未经说明的搜索。

为允许从 BF16 重新生成各阶段，固定格式恢复和导出已改为核对本次真实来源 UUID；report、payload、源 metadata 和重载后的模型必须一致。原有源码/data hash、冻结状态及保存重载检查保留，历史记录不被修改，也不会伪造旧 checkpoint 身份。

本次验证范围：训练配方与历史实际命令逐项核对；21 项相关 CPU 回归通过，包括真实 Trainer 的 2000 步调度下在 1200 步停止并保存、v6 张量保存重载、来源身份不一致拒绝和文件 hash/stat 检查；真实本地输入的脚本检查退出 0，脚本语法与本文命令/链接检查通过。**尚未运行从 BF16 到最终八任务的完整新流水线，也未以新流水线的 GPU 实测证明再次达到 69%。**

方案原理、各模块配置、正反实验和原始证据见[详细汇报](../experience/records/qwen3_w25/2026-10-04_qwen3_8b_69_report.md)。只需复评已有成品或重做已保留起点之后的末轮训练时，见[原有最快复评说明](../experience/records/qwen3_w25/2026-10-04_qwen3_8b_69_quickstart.md)。
