# 动态三阶与INT8＋abs005的L18／L35同上限对照

2026-10-28已完成Qwen3-8B两个真实down层的直接比较：沿用[L0同cap对照](../adaptive_sparse_pair_20261028/README.md)的算法和计费规则，比较`INT8＋abs005稀疏`、`未加权dynamic3`和`激活加权dynamic3`。两层的校准／train留出／全128篇输出误差均为加权动态最低、未加权动态次之。本页是这组跨层共同cap结果的唯一正文；原始指标和完整文件在结果目录。

这与[基础预算复现](../adaptive_layer_transfer_20261028/README.md)回答不同问题：基础预算以fixed2的19,607,424 B为上限，动态文件还加21,312 B未物化第四decoder预留；本次以fixed2加实际稀疏payload确定共同上限，只计实际三个decoder。两组复用同一码本，但分配和对照对象不同，不能混用数值或结论。

## 固定条件与来源

- 每层严格绑定原fixed2、真实HF BF16目标、全部1,540,096个32维向量、source variance、保护值／索引／scale及旋转sign。实际矩阵4096×12288，D32／K48／H128／R0、transpose、双侧block32和256个INT8保护输入列均保持。
- 固定两阶源为`/data/star/VAELLM/adaptive_vector_rvq_20261027/repair_20261027/formal_l18_l35/layer_{18,35}/fixed2.avrvq`。第三阶来自已完成6000步的`/data/star/VAELLM/adaptive_layer_transfer_20261028/formal_l{18,35}/stage3.pt`，沿用seed2031、batch8192、LR0.003、linear6000的训练历史。**本次没有训练或更新任何BSQ码值／decoder参数**；动态臂复用了额外训练的第三阶，不能称与稀疏臂等训练算力。
- 输入和层身份由crosslayer `prepared/prepared.json`绑定，SHA256为`30d7e4ee4e77028bfb6dc9f6db5c7e1cc0c7f4eb6985e1d583e693e03b437470`。HF模型为`/home/star/models/Qwen/Qwen3-8B`；每个report保存源码／输入SHA、HF shard stat、目标digest、bank digest和实际命令，没有伪造commit ID。
- 数据为`HuggingFaceFW/fineweb-edu`、`sample-10BT`、revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`、`train` split、ODC-By v1.0。既有连续行700000–700127共128篇：前64篇15,183有效token只用于估计激活权重，后64篇15,492 token仅作本轮读出。**全部128篇此前已见，后半不是盲测，也不是正式测试集。**
- 加权分数保持输入sign→forward Hadamard→输入行二阶矩→均值归一为1→每行repeat128的布局，不裁剪、不加epsilon。两种动态分配共用完全相同的冻结bank。

## 真实共同存储上限

稀疏臂完整保留fixed2，以HF减实际部署BF16基底为残差，先排除保护列，再执行原生abs0.5%选择。每层均实际选251,659个元素、768个256²块；原native codec为两个UINT8局部坐标、UINT8值、UINT16块号、INT32指针和FP32 scale／zero。先将稀疏值解码为BF16，再按原生顺序加回BF16基底。

每层实际稀疏payload均为765,733 B，因此共同cap为 **19,607,424＋765,733＝20,373,157 B**。每份完整文件均计16 KiB头部和全部持久字段，包括码、decoder、保护、旋转、标签或稀疏索引／scale。动态只计实际三个decoder，不列不存在的第四decoder预留；最多四轮原λ分配和真实zlib6长度计费，取最后可行轮，不按输出质量挑选。

| 层／方案 | 实际bytes | 未用预算bytes | 实测bit／该Linear原权重 |
| --- | ---: | ---: | ---: |
| L18 INT8＋abs005稀疏 | 20,373,157 | 0 | 3.238226096 |
| L18 未加权dynamic3 | 20,372,657 | 500 | 3.238146623 |
| L18 激活加权dynamic3 | 20,373,154 | 3 | 3.238225619 |
| L35 INT8＋abs005稀疏 | 20,373,157 | 0 | 3.238226096 |
| L35 未加权dynamic3 | 20,372,723 | 434 | 3.238157113 |
| L35 激活加权dynamic3 | 20,373,154 | 3 | 3.238225619 |

这是相同**上限**比较，各臂实际文件并非逐字节相等。上述位宽仅针对该Linear，不是全模型平均位宽。两层未加权臂均取第3轮，第4轮分别超cap 15／19 B；两层加权臂均取第4轮，并在码数预算不变时结束。不存在质量网格或事后扩码。

## 完整局部读出

三臂及HF teacher在同一GPU2上执行原生BF16 `F.linear`，批量256，每批只计算一次共享HF输出。每层均完成121批、30,675 token；每组累计SSE与目标能量，全128篇以两组分子／分母相加，既不平均两半NMSE，也不平均不同层NMSE作为整模型指标。

| 层／输出NMSE，越低越好 | INT8＋abs005稀疏 | 未加权dynamic3 | 激活加权dynamic3 |
| --- | ---: | ---: | ---: |
| L18 前64篇校准 | 0.052049995743 | 0.049627476930 | 0.049439623254 |
| L18 后64篇train留出 | 0.052359482377 | 0.049899314825 | 0.049761267200 |
| L18 全128篇 | 0.052206070097 | 0.049764564996 | 0.049601828550 |
| L35 前64篇校准 | 0.008397864553 | 0.008340982511 | 0.007306444508 |
| L35 后64篇train留出 | 0.008230415741 | 0.008179736698 | 0.007208364949 |
| L35 全128篇 | 0.008312188700 | 0.008258480446 | 0.007256261710 |

相对各层稀疏臂，train留出输出误差下降分别为：L18未加权 **4.6986%**／加权 **4.9623%**，L35未加权 **0.6158%**／加权 **12.4180%**。全128篇分别下降4.6767%／4.9884%和0.6461%／12.7034%。这补齐了L0局部正例之外的两层同向证据，改善幅度并不一致；这些百分比是相对误差下降，不是任务准确率百分点。

| 原坐标权重NMSE | INT8＋abs005稀疏 | 未加权dynamic3 | 激活加权dynamic3 |
| --- | ---: | ---: | ---: |
| L18 | 0.069748836034 | 0.066452224778 | 0.066810015152 |
| L35 | 0.070408272539 | 0.067067249626 | 0.076961463597 |

L35加权臂的权重误差较稀疏臂更大，实际输出误差却最低，继续支持按真实输出目标判断；本次没有识别这种差异的完整因果。加权对角近似仍忽略输入协方差、保护误差交叉项和逆旋转BF16舍入。

## 运行、验证和复现

两层均在`bitvae`、单物理GPU2上顺序完成，源码和真实输入未变，退出状态均为0。L18完整三臂首先验证新GPU读出路径，完成后才运行L35；没有额外训练短测或新模型替代。

| 层 | tmux会话 | Python PID | 完整耗时 |
| --- | --- | ---: | ---: |
| L18 | `sparse_layers_l18_20261028` | 292240 | 22.85秒 |
| L35 | `sparse_layers_l35_20261028` | 293068 | 20.94秒 |

输出根为`/data/star/VAELLM/adaptive_sparse_layers_20261028/`：每层`layer_{18,35}/report.json`保存全部原始指标、实际配置和SHA，核心日志为`core_l18.log`／`core_l35.log`，均含`outer_exit=0`。六份完整端点为各层目录下的`sparse_abs005.avrqsp`、`unweighted_dynamic3.avrvqz`和`weighted_dynamic3.avrvqz`。

每层均通过实际完整保存／fresh重载、全部持久字段／metadata／路由、同路径BF16向量、dense与selected前缀误差、保护列和冻结bank检查；四个动态端点即使存在ragged末批也全部exact，没有放宽合同。GPU与原CPU准备的归一化h最大相对差为8.94e−9／6.10e−8，属于已量化的正常设备归约差异。两层峰值CUDA allocated均为1,719,795,712 B，完成后GPU2已释放。

代码入口为[run.py](run.py)，层来源适配在[sources.py](sources.py)。复用原`allocate`、`readout`、`sparse_file`及crosslayer helpers，不修改旧L0-only入口或公共CAT／VAELinear。旧紧凑文件的历史`l0` schema文本仅是格式名，实际层身份仍严格取geometry.name／source_key，不把L18／L35标成L0。

调用者先在项目根目录激活`bitvae`，再用tmux／nohup运行；输出必须为新目录，GPU2必须实际空闲：

```bash
bash experiments/adaptive_sparse_layers_20261028/run.sh \
  --layer 18 --output-dir /data/star/VAELLM/adaptive_sparse_layers_20261028/<新目录>
```

L35使用`--layer 35`和另一新目录。脚本固定`CUDA_VISIBLE_DEVICES=2`；现有输出已经完成，不能原位重跑或覆盖。代码仅支持这两层既有真实bank，不自行训练或回退到其它源。

清理前已将所选长度流与完整文件中的嵌入字段逐字节核对；随后删除16份冗余round长度流，共释放2,216,253 B，精确路径／大小／SHA／原因在各report的`cleanup`中。六份完整端点保留用于独立配对重读审阅，原始bank和输入未动。

## 结论边界与经验增量

新增证据是：在这两层fresh-HF基底、当前codec和同一实际存储上限下，两种动态分配均优于INT8＋原生abs005的局部输出误差，加权臂更低。基础预算下L18尚未优于fixed2的结果仍然成立；扩大预算和改变对照后得到的正例不能覆盖它。

当前没有完整模型组装、MMLU、LongBench或九项评估，不能据此判断≤0.8个百分点目标，不能直接与已恢复bestabs成品排名。完整模型验证还须保持共同起点、恢复数据／范围／预算及全局位宽口径。相关可迁移经验增量归入[压缩配置与存储经验](../../docs/experience/lessons/compression_choices.md)，本页保留本次结果和原始证据索引。
