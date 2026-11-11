# Qwen/Llama 最小研究接续包

2026-10-28，按用户确认的“从当前 Qwen/Llama 成品继续研究”制作，供本地保存并迁移到新服务器。本包在成品上开启新的实验，不提供从旧中断步骤精确续训的保证。

## 内容与范围

归档、外层 `SHA256SUMS` 和本文一起下载。解压后的 `manifest.json` 记录全部包内文件的路径、字节数、SHA256、模型身份和来源。实际大小以交付清单为准。

| 内容 | 包内位置 | 用途 |
| --- | --- | --- |
| Qwen bestabs 完整成品 | `models/qwen/` | UUID `e01b09b7-a85d-4534-a5d2-5ab3b91312e6`，4071 持久张量 |
| Llama sparse005 完整成品 | `models/llama/` | UUID `a1f09f8f-d9b2-4c41-91f9-e136d4ea986e`，3619 持久张量 |
| 实际完整旋转运行快照 | `runtime/` | 原生 v6、sparse 及 252／220／188 Hadamard 支持 |
| 仓库代码和 Git 历史 | `repo/`、`repository.bundle` | 核心经验、协议、代码；此次未提交接续工具和本文作为工作区增量保留 |
| 仓库外说明与关键证据 | `repo/result/` | 原位 Markdown、隔离源码、评分摘要；两个成品另保留原始预测和核心日志 |
| 环境实测 | `environment/` | Python／PyTorch／Transformers 和完整依赖版本、平台信息 |
| 验证入口与结果 | `handoff.py`、`validation/` | 文件校验、迁移目录下的 CPU strict 重载和逐张量核对 |

Qwen 完整九项约 **67.0235120944**，logical **2.5926188179**／packed **2.6059934782 bpw**；Llama 完整九项约 **60.9288983786**，logical＝packed **2.5483702880 bpw**。两者 rank0，均未达到当前相对各自 BF16 九项下降不超过 0.8pp 的目标。选择它们是为了保留研究接续点，不代表新完成达标。

本包不包含全部探索权重、原始 BF16 大权重、全量数据、教师概率／激活缓存、所有历史预测和日志，也不打包 conda 安装目录、驱动、账号凭据或 Codex 聊天。未装入的历史原件不能仅凭文档链接假定存在；两个选定成品通过下述入口加载时不依赖祖先 checkpoint。

## 下载、解压与文件校验

在新服务器将归档和 `SHA256SUMS` 放在同一目录，执行：

```bash
sha256sum -c SHA256SUMS
mkdir -p /data/star/migration
tar -xzf vaellm_research_handoff_20261028.tar.gz -C /data/star/migration
cd /data/star/migration/vaellm_research_handoff_20261028
python3 handoff.py verify
```

解压目标可换成任意有写权限的目录，不要求重建旧服务器全部绝对路径。磁盘准备为归档和解压内容之和。`verify` 只用 Python 标准库，逐文件检查 SHA256 和大小；通过表示传输一致，不代表目标机器 CUDA 已可用。Windows 本地可用 PowerShell `Get-FileHash -Algorithm SHA256` 核对归档，上传后仍执行完整校验。

## 新服务器环境

源环境实测：Linux x86_64、Python 3.11.16、PyTorch 2.6.0+cu124、Transformers 4.51.0、FlashAttention 2.7.4.post1、fast-hadamard-transform 1.0.4.post1。完整记录在 `environment/`。仓库旧 `environment.yml`／`requirements.txt` 有历史版本，优先对照本包实测；它们不是已验证的新机器安装方案。

由目标服务器操作者准备 `bitvae` 环境，并按该机器的驱动／CUDA 条件安装依赖。本包的 `environment/pip-versions.txt` 是源环境版本清单，CUDA 扩展可能需要重新构建；不保证直接批量交给 pip 就能完成。此次没有安装或修改原服务器依赖。

```bash
conda activate bitvae
which python
python -V
python -c "import torch, transformers, fast_hadamard_transform; print(torch.__version__, transformers.__version__, torch.version.cuda)"
```

后续模型检查在已确认的 `bitvae` 中执行。CPU strict 重载不需要 GPU，建议可用内存至少 40 GiB；正式生成／训练仍须根据协议准备 GPU 和必要资源。

## 搬大数据前检查新机器与SCP速度

2026-10-28新增两个独立入口：

- [`tools/check_new_server.py`](../../tools/check_new_server.py)：在**新服务器**已激活的`bitvae`中运行；无需下载大模型，检查项目依赖和三条正式链路导入、真实BSQ/VAE三步训练、Qwen/Llama小模型的两stage压缩层及v6严格保存重载。默认另测可见GPU0上的BF16前反向、FlashAttention、Hadamard CUDA扩展、Triton packed/fused decoder前反向。
- [`tools/scp_benchmark.py`](../../tools/scp_benchmark.py)：在**旧服务器**运行，只用Python标准库和已有OpenSSH；测实际旧机→新机的大文件吞吐。普通SSH模式在远端核SHA256；堡垒机模式显式指定资源目录，实际用`scp -s`上传，再SFTP回读核SHA256。关闭SSH压缩并复用专用连接；造文件、认证及校验不计入上传吞吐。

脚本不安装依赖、修改驱动或启动正式训练。输出目录必须尚不存在；重复运行换目录。基础检查是开发环境自检，不能代替真实8B模型的正式运行条件验证，CPU模式更不能表示GPU通过。

### 新服务器自检

目前`cmcc@202.194.69.66:60022`是**堡垒机**，交互菜单001映射到`feng / 172.16.227.196:22`。该入口拒绝远程`exec`，不能将其当成普通计算节点直接SSH执行命令。已通过平台支持的SFTP通道，将自检脚本送到节点`/home/cmcc/vaellm_tools_20261028/check_new_server.py`。先交互登录，再在菜单选001：

```bash
ssh -p 60022 cmcc@202.194.69.66
```

2026-10-28现场核对：8×A800 80GB均有既有任务；`/`约剩155GiB、占用98%；现有conda列表没有`bitvae`，`/home/cmcc`下未找到VAELLM仓库。本轮没有安装依赖、停止任务或清理他人数据。需要先准备环境和源码；现有可见磁盘不足以接收数百GiB全量数据，接收盘应另行确认。

随后按用户要求补做[原生硬件功能检查](../../result/new_server_hardware_20261028/report.json)，使用新机已有CUDA13工具链，不依赖Python环境：8张GPU逐卡通过32MiB显存三模式校验、1048576个FP32结果校验及BF16 Tensor Core矩阵乘法的CPU参考对照；128MiB主存三模式校验、64MiB磁盘`O_DIRECT`写入／fsync／读回也通过。日志和复现源码在同目录`evidence/`，原始归档为`hardware_results.tar.gz`；测试临时数据和编译二进制已清理。

0、4号GPU各有1次累计DRAM可纠正ECC，测试前后均未增长；全部GPU的易失ECC和累计不可纠正ECC计数为0，行重映射计数为0，无pending或failure。累计与易失计数含义参见[NVIDIA说明](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。这支持当前基本功能正常，不能证明全显存／全主存、长期压力、NCCL或磁盘SMART健康；当前账号也不能读取系统内核日志。VAELLM的Python／FlashAttention等软件链路仍须在`bitvae`准备好后单独验收。

环境与源码准备好后，在新机选择**空闲GPU**，并在tmux中执行。以下`/path/to/VAELLM`换成实际仓库目录；脚本可放在仓库外，通过`--repo`指定源码。

```bash
conda activate bitvae
which python
python -V
tmux new -s vaellm_check
CUDA_VISIBLE_DEVICES=0 python -u ~/vaellm_tools_20261028/check_new_server.py \
  --repo /path/to/VAELLM --output ~/vaellm_check_20261028_01 \
  > ~/vaellm_check_20261028_01.log 2>&1
```

按`Ctrl+B`再`D`可脱离会话，`tmux attach -t vaellm_check`返回；结果为输出目录的`report.json`。`status=passed`表示本次选定检查通过；首个失败会记录`failed_check`和堆栈并退出1。原始输出在上述`.log`中。

按需选以下一种命令，仍在已经激活环境的tmux中执行：

```bash
# 仅CPU路径；状态为cpu_passed_gpu_not_tested，不运行GPU检查。
python ~/vaellm_tools_20261028/check_new_server.py --repo /path/to/VAELLM \
  --cpu-only --output ~/vaellm_cpu_check_20261028_01

# 多卡开发：只测试显式选择的两张空闲卡，增加真实NCCL all_reduce。
CUDA_VISIBLE_DEVICES=0,1 python ~/vaellm_tools_20261028/check_new_server.py --repo /path/to/VAELLM \
  --nccl --output ~/vaellm_nccl_check_20261028_01

# 数据迁入后增加接续包逐文件校验、Qwen/Llama成品CPU独立重载。
CUDA_VISIBLE_DEVICES=0 python ~/vaellm_tools_20261028/check_new_server.py --repo /path/to/VAELLM \
  --handoff /path/to/vaellm_research_handoff_20261028 \
  --output ~/vaellm_handoff_check_20261028_01
```

`--nccl`会使用**全部可见GPU**，不指定时只做可见GPU0的计算。`--handoff`调用包中既有入口，在单独隐藏CUDA的进程逐个重载成品，不测试成品GPU生成；可用内存建议至少40GiB。旧接续归档早于这两个新脚本，需单独发送脚本，不会自动出现在旧归档中。

### 本机向新机器的SCP测速

在旧机当前仓库的已激活环境中执行。若使用密码认证，先进入附着的tmux终端；首次SSH通过终端读取密码，脚本不接收、保存或记录密码。

```bash
tmux new -s vaellm_scp_speed
python -u tools/scp_benchmark.py \
  --host cmcc@202.194.69.66 --port 60022 \
  --sftp-root '/172.16.227.196:22_ssh_cmcc@172.16.227.196-A800服务器' \
  --remote-encoding gb18030 \
  --local-dir /data/star --size-mib 64 --rounds 3 \
  --estimate-gib 7.621717 --output result/scp_speed_new_server_20261028_01
```

`7.621717GiB`对应本次接续归档量级，可换成实际迁移量或省略`--estimate-gib`。64MiB×3是本次堡垒机的实际测试规模；脚本未指定大小时默认1GiB×3。`--sftp-root`选择平台资源，`--remote-encoding gb18030`匹配该入口默认目录编码，后续通过同一SSH连接在实际计算节点路径中操作，不请求被禁用的exec通道。

堡垒机模式默认在节点账户家目录中新建独立测试目录，也可加`--remote-dir /data/cmcc`选择**已存在且可写**的接收盘，目录不含空格或shell特殊字符。应选实际接收数据的磁盘。每轮SCP上传后SFTP回读校验，**额外下载同等数据量**；下载和SHA耗时不计入上传吞吐。测试结束只删除其独立目录的`payload.bin`和空目录，不删除原有数据。

若以后有普通计算节点直连或标准SSH别名，省略`--sftp-root`即可使用普通模式，由远端`sha256sum`校验，无需回读。下例需替换为管理员提供的**真实节点通道**，不能把堡垒机端口直接当作节点端口：

```bash
python tools/scp_benchmark.py --host new-vaellm-node \
  --size-mib 1024 --rounds 3 --output result/scp_speed_direct_20261028_01
```

本机的`report.json`记录实际传输协议、逐轮秒数、SHA256、MiB/s及Mbps，`commands.log`记录命令与错误；端口、SSH私钥或跳板机配置分别可用`--port`、`--identity`、`--ssh-config`指定。每次操作默认1800秒超时，可调`--timeout`；不会自动重试、猜测替代通道或关闭主机身份检查。`--file /path/to/stable-file`可替代`--size-mib`使用现有文件，但仍会删除测试副本，不应当作正式迁移入口。

吞吐包含SSH加密、SCP协议、网络和文件IO，远端可能使用页缓存，不能视作裸网络带宽或强制落盘速度；大量小文件、断点重传及额外校验另有开销。随机文件与专用SSH master在结束或异常后清理；网络故障若阻止清理，会在报告中留下精确远端路径，不能把该次标为全部完成。

## 验证两个成品可独立重载

在解压根目录逐个执行，每个模型使用一个新 Python 进程：

```bash
CUDA_VISIBLE_DEVICES='' python handoff.py check-load --model qwen
CUDA_VISIBLE_DEVICES='' python handoff.py check-load --model llama
```

入口优先导入包内 `runtime/`，用成品自己的 config 在 CPU 构造模型，再用原生 v6 strict loader 重建全部压缩拓扑和持久状态，不需要原 HF 大权重。逐一核键、dtype、张量值、UUID、logical／packed 账本及 tokenizer，CUDA 不初始化。源 metadata／UUID／权重均保持原字节，旧绝对路径只作来源记录。

Qwen 仅构造阶段使用 CPU eager，加载后恢复正式 FA2 配置；Llama 恢复正式 SDPA。本检查没有 forward、生成、训练或完整九项复评，不能代替新硬件上正式运行所需的针对性验证。

## 恢复代码和开展新实验

先读 `repo/docs/README.md` 及研究主记录 `repo/docs/experience/records/w25_transfer/2026-10-04_mmlu65_llama31.md`。历史 PID、运行状态和后继安排不是新服务器已启动任务的证明。

需要 Git 历史时，可离线克隆到一个尚不存在的目录：

```bash
git clone -b master repository.bundle /data/star/VAELLM
```

bundle 对应 `manifest.json` 中的已有提交；`repo/` 同时包含本次未提交的接续工具、本文及导航更新，需将这些明确增量复制到新工作区，再由用户决定提交。不要覆盖新服务器已有修改。源服务器没有因打包而暂存、提交或推送。

包内加载函数可用于新实验入口：

```python
from pathlib import Path
from handoff import load_model

package = Path('/data/star/migration/vaellm_research_handoff_20261028')
model, tokenizer, metadata = load_model(package, 'qwen')  # 或 'llama'
print(metadata['checkpoint_id'])
```

从解压根目录启动，或将该目录加入 Python 路径。应在没有提前导入另一版 `train_utils`／`rotation`／`litebsq` 的新进程中调用。返回模型在 CPU、eval、参数冻结；新实验按设计设置训练范围、设备、原生预热和精度策略。`runtime/` 保持原样，改动放在新的工作区与输出目录。

历史脚本经常绑定旧绝对路径、祖先权重、数据缓存及逐文件 hash。它们作为证据保留，不保证解压后可原命令重放。以选定成品开启新实验时，复制所需配置并明确新的 source UUID、实际导入快照、数据和输出路径；不批量替换冻结历史证据或 checkpoint metadata。

## 还需准备什么

- **重算 HF 教师、选址或从头压缩：** 按 manifest 中的原 HF 仓库和 revision 获取 Qwen3-8B／Meta-Llama-3.1-8B-Instruct，遵守原许可和访问条件。本包 tokenizer/config 足够独立重载成品，但不含原始 BF16 大权重。
- **新训练：** 按研究记录恢复开源 `train`／`auxiliary_train` 数据及版本；固定教师概率缓存需要按原模型和归约协议重建或另行迁移，不以正式测试集代替训练数据。
- **正式复评：** 另行准备八任务／LongBench 数据及评测工具。本包原始预测和评分用于对照，不包含全部下载缓存。
- **精确恢复旧中断步：** 需另外取得对应 optimizer、scheduler、RNG、round-base、数据状态和不可变恢复配置，本包不提供该保证。

## 验证边界

交付时记录真实目录迁移后的两模型 CPU strict 重载、全部持久张量相等、实际存储账本、tokenizer、归档内容与 SHA256 的检查结果，见 `validation/` 和交付校验记录。上述步骤没有在未知的新服务器运行过；没有重新演练从 BF16 开始的完整压缩链，也没有启动新的 GPU 训练或全量评测。

## cmcc 新机的共用接线（2026-10-30）

当前仓库为 `/home/cmcc/VAELLM`，数据和实验实体分别为 `/data1/cmcc/VAELLM/data`、`/data1/cmcc/VAELLM/artifacts`。仓库 `data/`、`artifacts/` 是到这些实体的链接。N/S 冻结运行快照已从接续归档按原 SHA256 恢复到 `result/qwen3_8b_mmlu65_20261004/` 下原有相对入口；各自 `data/` 也链接到数据实体，因为冻结 preset loader 以快照文件位置解析相对数据路径。

共用常量和只读旧路径解析位于 [`experiments/cmcc_runtime_20261030/paths.py`](../../experiments/cmcc_runtime_20261030/paths.py)。它只映射已确认的旧仓库、artifact、模型根及原 `bitvae` Python，不建立 `/home/star` 或 `/data/star` 兼容目录，不修改历史 JSON、mtime、inode 或 hash。新运行须建立自己的 prepared 和实际代码／数据绑定，不能直接复用旧机 stat。

恢复选集与逐文件 SHA256 见 [快照和直接 helper 清单](../../result/migration_audit_20261030/dependency_restore.json)、[helper 引用闭包清单](../../result/migration_audit_20261030/helper_restore.json)。[共用 CPU 检查](../../result/migration_audit_20261030/shared_wiring_validation.json)确认两个快照的实际导入根及 4032 维的 252 阶 Hadamard 精确正交；[数据路径检查](../../result/migration_audit_20261030/snapshot_data_validation.json)确认 N preset 的五份 EdgeRazor `train` 源可用。这些检查没有运行 8B 前向、训练或评测。

[`baseline.py`](../../experiments/cmcc_runtime_20261030/baseline.py)为两分支准备共用 BF16 对照。输入阶段核新旧 LongBench token/index SHA、协议及 4750 条旧预测的输入身份和顺序；最终生成阶段复用已完成的输入及原模型官方字节验证，写出比较所需的路径、当前 stat 和对应身份 hash。任务配置、指标、生成文本和 token 等字段保持逐值相同。原始推理代码 hash 仍保留为历史来源，不把迁移绑定描述成重新执行基线。

```bash
# 已激活 bitvae，在仓库根目录；仅核输入，不需要 BF16 大权重。
CUDA_VISIBLE_DEVICES='' python -B -m experiments.cmcc_runtime_20261030.baseline --check-inputs-only

# 原 Qwen 五份 BF16 shard 全部到齐且 qwen_verified.json 通过后执行；复用固定 revision 的官方字节核证。
CUDA_VISIBLE_DEVICES='' python -B -m experiments.cmcc_runtime_20261030.baseline
```

最终位置固定为 `/data1/cmcc/VAELLM/artifacts/cmcc_baselines_20261030/qwen3_8b/`，包含 `mmlu.json`、`seven.json`、`longbench/{config.json,scores.json,predictions/}` 与 `migration_binding.json`。只可使用 `schema=cmcc_qwen3_bf16_baseline_migration_v1`、`status=verified` 的绑定；缺完整 BF16 权重时不会生成终态，已有目录不会被覆盖。上述迁移检查不重新推理或重算评分，GPU 分支仍按各自真实运行证据验收。
