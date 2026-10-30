# 自适应残差码本的独立目标分组拟合

本隔离实验验证[已说明的并行实现方案](../adaptive_vector_rvq/README.md#10月29日高效并行实现方案与讨论边界)：把同形层的encoder／decoder GEMM合批，同时分别调用原BSQ，保持每层熵目标、参数和优化轨迹定义独立。公共CAT／BSQ／v6和旧实验未修改。当前不是完整模型压缩或九项质量实验。

## 方法与公平对照

两份真实输入沿用[已完成的L18／L35准备](../adaptive_layer_transfer_20261028/README.md)，各为1,540,096个D32向量及其冻结fixed2 BF16前缀。第三阶为K48／H128／R0／LayerNorm，每层seed2031、batch8192、AdamW LR.003、原linear6000。仍对全部向量拟合，再按同预算分配动态长度，不改成active-only或周期重路由。

- `serial`直接调用原`training.fit_stage`依次拟合两层，各6000步。
- `grouped`从同样的两个独立初始化打包encoder／decoder；每层原BSQ看到`[8192,1,48]`，各自MSE和原辅助损失求和；不额外乘除模型数，不加全局clip。每层保留原shuffle和优化步数。当前两池均恰有188个完整batch，不填充或重采样尾批。
- 两者都在3000／6000按原路径编码完整向量、融合实际decoder及读出BF16残差。分组LayerNorm的affine与batched GEMM可产生正常BF16舍入差，因此除FP32数学检查外还比较最终实际输出；不要求训练码位逐位相同。
- 末点使用相同激活权重、四轮上限的λ／zlib分配和完整文件计费，生成真实weighted_dynamic3文件，再在同128篇既见FineWeb train输入上做完整局部BF16输出比较。共同cap取每层旧fixed2＋原生abs005文件20,373,157B，不冒称已与Rice7省存储版本配平。

时间主指标为`report.json.elapsed_seconds`，包含真实输入准备、拟合、编码、分配、保存／重载及完整局部读出。两模式顺序运行在同一GPU3。原串行实现有重复MSE统计，而分组实现复用一次结果，所以时间差属于整个实现流程，不全部归因于GEMM，也不外推为36层／整模型恢复提速。各子计时的范围在报告中注明。

## 已完成的最小验证

`/data/star/VAELLM/adaptive_grouped_fit_20261029/check/report.json`已完成，exit0、7.1207秒。CPU FP32使用各层真实残差前128行：两层总loss逐层相同，全部12叶梯度relative-L2分别4.98e-8／4.40e-8。真实GPU BF16按每层8192样本做两次连续更新，loss／梯度有限、参数确实更新、Adam计数和6000步日程一致。

BF16首步梯度relative-L2约0.004，第二步约0.0118／0.0086；记录了分组LN／GEMM及随后优化轨迹的差，未据此宣布最终误差相同。FP32误差容限针对正常线性／LN／归约舍入，不能用其容限掩盖跨层熵耦合。

## 运行与状态

调用者使用`bitvae`，当前固定成对实验由tmux `adaptive_grouped_pair_GPU3_20261029`运行，shell PID349077，串行worker349079。两个独立调用顺序执行，仅在串行成功后执行分组；不重试或改参数。开始时GPU3为空，无其它训练／模型加载与其争用计时条件。

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/home/star/VAELLM python -B -u experiments/adaptive_grouped_fit_20261029/run.py \
  --mode serial --gpu 3 --output /data/star/VAELLM/adaptive_grouped_fit_20261029/serial
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/home/star/VAELLM python -B -u experiments/adaptive_grouped_fit_20261029/run.py \
  --mode grouped --gpu 3 --output /data/star/VAELLM/adaptive_grouped_fit_20261029/grouped
```

当前配对已完整结束，serial／grouped／pair退出均为0，GPU3已释放。上述输出已被本次任务使用，不能覆盖重跑。核心日志为输出根下`serial.log`／`grouped.log`；每臂`report.json`保存真实PID、命令、输入／实现身份、实际训练记录、最终文件和局部结果。串行worker为349079，分组worker为349448。

## 10月29日完整配对结果

唯一汇总为`/data/star/VAELLM/adaptive_grouped_fit_20261029/comparison.json`，原始数据仍在两臂报告。完整流程串行234.97529077秒、分组199.53393471秒，速度比1.17762069，用时减少15.0830%。两层各实际消费49,152,000个向量、6000更新；训练阶段峰值allocated由147,204,096B到294,147,072B，这不是整个读出阶段的峰值。

| 层 | 串行后64篇NMSE | 分组后64篇NMSE | 相对误差变化 | 串行／分组实际文件B |
| --- | ---: | ---: | ---: | ---: |
| L18 | 0.049761267200 | 0.049758810918 | −0.00494% | 20,373,154／20,372,850 |
| L35 | 0.007208364949 | 0.007219688054 | ＋0.15708% | 20,373,154／20,373,154 |

四份文件均通过完整保存／fresh重载和实际cap核验，结束后再次核大小及SHA。串行局部读出复现旧结果；L18基本保持，L35有小幅误差增加，不能写质量完全相同。两层相对原稀疏对照的局部优势仍保持，但全部128篇属于既见train，不是盲测或九项精度结论。

决定：首版实际提速足以进入[整模型恢复桥接](../adaptive_native_bridge_20261029/README.md)，暂不继续2／4／8组数网格。bridge的动态臂消费本次grouped L18成品，验证并行拟合产物的真实模型反传与保存。只有后续整类运行表明拟合仍是主要耗时，才补一个目标组规模的决定性性能对照。该结果不证明36层、完整恢复或整条压缩流程也有同样加速。

## 10月29日：完成后最小产物收尾

配对及桥接均已完整结束，无本组活动进程或候选文件打开FD；本次仅清理已确认无后继用途的生成产物，不改源码、配置或原始指标。唯一清理记录追加在既有`comparison.json.cleanup`，包含每个精确路径、原因、原大小／SHA／inode、实际allocated释放量及退出值；原serial／grouped／check报告和核心非空日志保留。

保留`grouped/endpoint_18/weighted_dynamic3.avrvqz`作为后续扩展compact consumer到其它D／H规格时，已经验证的D32／H128真实加载输入；大小20,372,850 B，SHA `d32286371be510ea87e86ed167f4c291a0083f19bea25675334044f14635e2aa`。桥接更新后的另一份回归输入由[桥接正文](../adaptive_native_bridge_20261029/README.md#10月29日完成后最小产物收尾)维护，不将初态和更新后端点混用。

已删除serial的L18／L35成品及grouped的L35成品共3份；这些端点现不可再加载，原完整指标／SHA和比较结果仍在。4个端点allocation报告逐一绑定的16份round长度流，在核SHA及所选frame与完整文件嵌入`length_frame`逐字节相同后删除；实际chosen为serial两层第4轮、grouped L18第3轮／L35第4轮。全部路由载荷已在保留或清理前核证的完整文件中，不需要另留round副本。

另将两份serial `observations.json`与`serial/report.json.training.{18,35}`逐字段核对相同后归并删除；serial／grouped／pair／check的4份退出散文件均为0，已归入cleanup；空`check.log`一并移除。合计26文件、逻辑63,560,329 B、实际allocated释放 **63,619,072 B**。删除后保留成品及全部必要证据SHA、7份原大输入stat保持，原始旧adaptive输入／bank／HF模型均未触碰；没有新增质量或速度结论。

## 10月29日独立研究复核：并行语义、规格扩展与下一项完整对照

本节仅为源码及既有原始报告复核；没有新训练、性能测量或九项结果，不更改已绑定的Python源码。依据包括原型`training.py`／`codec.py`、本目录`grouped.py`／`run.py`、`adaptive_sparse_pair_20261028/allocate.py`、bridge消费者、公共`parallel_layers.py`／`bsq.py`／`checkpoint_v6.py`／`checkpoint_bpw.py`，以及[压缩与存储经验](../../docs/experience/lessons/compression_choices.md)、[输出损失经验](../../docs/experience/lessons/output_losses.md)。旧几何配额与共同阈值失败的条件仍成立；下面讨论的是已有正例的dense拟合后预算分配，不复活旧失败配置。

### 当前正例的确切顺序及迁移边界

1. 从同一真实目标矩阵取得保护集合、重新编码的保护值和旋转；移除保护列、旋转后按转置布局形成D维向量。旧原型的目标确为fresh HF BF16，但保护索引和旋转身份仍取历史G；它不是从任意HF独立生成保护计划的完整入口。
2. 第一阶训练2000步并按部署BF16融合规范化／BSQ尺度、编码和解码；冻结第一阶后，对**全部**向量的真实BF16前缀残差训练第二阶6000步。再冻结前两阶，对**全部**向量残差独立训练第三阶6000步；stage3 seed2031、B8192、linear LR0.003→0，逐层规范化和BSQ目标独立。后阶不能与本层尚未完成的前阶并发。首轮1–4阶配额训练是另一个实验，不能混写成当前dynamic3算法。
3. 保存各个真实前缀的误差`e[v,s]`；输入二阶矩`h`仅来自train校准，按真实向量布局广播，形成`h[v] * e[v,s] / source_variance`。在同一Linear内求`argmin_s(error[v,s] + lambda*s)`，以码行数约束求λ；不要求每阶边际收益递减，也不是一个共同剩余误差阈值。
4. 对真实两位长度流做zlib6实包，扣除实际三个decoder和所有固定字段，至多四轮重算可容纳码行数，取最后可行轮。它是有限预算迭代，不是压缩长度非可分成本的全局最优解。不同Linear目前各有自己的cap／λ，并未实现全模型统一λ；不能未经解释把不同层方差归一误差放进一个全局排序。
5. 最终路由只确定一次。按stage-major和原向量升序保存实际活动码，恢复阶段只训练decoder，码／长度／保护／旋转冻结；缺席stage必须连bias及融合mean一并跳过。解码保持stage顺序BF16相加，再逆旋转、回填保护及可选稀疏；decoder更新或状态加载必须使缓存失效。

可迁移的是这些矩阵几何、统计、固定成本和预算规则，尚不是已验证跨模型算法。当前入口仍硬绑定L0／18／35、Qwen4096×12288、G donor UUID、D32／K48／H128、128个向量／输入行及旧激活缓存。完整实现应按模型投影清单遍历整类，从当前父状态和train数据重算保护统计、旋转／布局及h；不能用旧层号、旧donor或旧h作最终选择规则。新模型需独立验收，已有三层局部结果不承担迁移证明。

### 为什么M2只减少15%用时，及M36／多卡的可行边界

原始报告显示：串行两个完整stage耗时111.3003＋107.6527＝218.9531秒，分组183.6736秒，stage范围约1.192倍；阶段外余量分别16.0222／15.8604秒。故已测增益主要发生在报告定义的拟合范围，阶段外的预算／保存／读出时间近似不变。该范围仍含3000／6000全量编码、导出和残差读出，串行又重复计算三次相同全量MSE；现有计时**不能**进一步断言GEMM、CPU同步或熵计算各占多少。

源码显示的具体未合并工作是：每层BSQ仍在Python循环中独立发起归一化、sigmoid／log／归约；每步有`isfinite(...).item()`和`loss.tolist()`导致主机观察；各层完整编码／观测仍顺序执行，grouped又要逐层抽出独立模型。encoder／decoder的矩阵乘才使用batched GEMM，LayerNorm的分组affine还增加独立算子。因此“只把Linear装进group”并未把一次更新合成一次GPU工作；这些是待测的瓶颈候选，不是已有profiler结论。

推广的最小等价方案如下，避免另外铺2／4／8／36网格：

- 以`D,K,Henc,Hdec,R,norm,activation,dtype,N,B,stage,更新数/LR日程`分固定训练组；同一类各层一般有相同N，D32的B8192或D64的B4096按预定标量曝光选择。36层可按实际分配GPU拆成独立层组，各卡无需DDP或跨卡平均梯度，只返回各层码／decoder及计账结果。组成员与每层输入身份保持，不能把层数变化解释成decoder共享。
- 在隔离拟合器内将BSQ逐层目标向量化：`[B,M,K]`仅沿B求各层边际，保留M；每层各自MSE、commit、sample entropy及K位边际熵，最终**求和**。保持原先球面归一化在BF16／强制FP32计算的先后顺序和全部系数；不直接调用会合并B/M的现`soft_entropy_loss`，不改公共CAT原有pooled语义。可用已有独立原函数作数值参照，不另造损失定义。
- 继续为每层保存独立shuffle generator、游标和已预定seed。当前stage3各层同为2031；“独立RNG”不等于擅自改成不同随机种子。打包Adam只在所有片段同更新数、同超参数、同LR且没有跨层clip时等价；矩和步数必须对应各自参数片段。不同日程的层进入不同组，不以零梯度假装冻结已完成的层，因为Adam动量仍会更新。
- 尾批按真实有效B分桶，同一外层更新中把各桶loss相加后只执行一次该组optimizer step；不填充／重采样，也不在熵分母计入无效行。固定N／同日程的36同类层无需为预期不存在的尾批异步建立调度系统。
- 最终短码流的活动数量不同：按`(stage,decoder spec,有效chunk行数)`分桶，完整chunk可以批量解码，尾chunk只处理真实行；各层结果仍按stage顺序加入原向量位置。不得把空码解成零码后加bias，不把跳过尾阶扩展成active-only训练。当前动态L18比dense三阶少约31.1%解码行，但比固定两阶多约3.3%，不能据此宣布部署更快。

只在即将采用的一个实际层组规模上补一次相同残差、相同日程的决定性性能检查，并分开记录更新／编码／分配／导出时间；无需先完整做组数搜索。已有FP32数学证据支持打包目标等价，BF16轨迹允许有已量化的正常差异，但必须检查最终真实输出与存储，不能宣称码位逐位相同。整模恢复的前向层依赖仍存在，offline拟合的并行收益不自动等于整模恢复收益。

### D／K／H应怎样成为真实规格

应在新的隔离模块中使用一份resolved spec贯穿训练、文件、消费者和计账，旧已绑定脚本保持原样。`D`取decoder `out_dim`、`K`取`in_dim`，encoder／decoder宽度分别显式保存；使用`decoder_type=asymmetric`时两宽度才独立。每层的N从保护后实际几何与flatten／transpose规则派生，不从Qwen维度推断。

root补充核查了`adaptive_layer_transfer_20261028/helpers.py::input_energy`：旧down转置布局的一条向量只跨输出坐标，所以每个向量可乘一个对应输入列的二阶矩；gate非转置的一条向量跨多个输入坐标，不能仅替换`repeat_interleave`因子或乘该向量的平均h。若沿用同一对角输入二阶矩代理，须按真实旋转／保护后布局将每个坐标的`h_j/mean(h)`映射到该向量，对每个stage前缀计算`sum_j(h_j_normalized * error_j²)/(D * source_variance)`。在旧down布局中这才退化为原`h_v * mean(error²)/variance`；在gate中一般不相等。新实现应核其向量分数和完整矩阵`sum_ij(h_j_normalized * error_ij²)`一致，不静默换成新的平均h近似；该对角代理仍不等于完整激活相关矩阵或真实MLP误差。

- 码行物理大小为`ceil(K/8)`，逻辑主码为`K*sum(lengths)`；A的K32、B的K64分别是4／8字节，旧K48才是6字节。复用`litebsq.bitpack`的logical shape和padding检查，所有分配器／loader／consumer都读该字段，禁止仅改训练CLI而保留`//6`／固定末维6。K不是8倍数时单列码padding，不能把其物理开销藏进logical口径。
- 长度原流是`ceil(N/4)`字节，再计真实zlib frame；成本由N决定，D64改变N，不能沿用D32旧长度文件或压缩率。`repeat_interleave`的因子由实际转置布局的`compressed_out_features/D`导出；非转置布局必须用其对应映射，不能照搬repeat128。必要向量padding须显式记录与计费，不能静默截断权重。
- R0／LayerNorm的单个BF16 decoder参数为`Hdec*(K+D+3)+D`；每阶A＝17216B、B＝33664B，旧D32K48H128＝21312B。实际state仍是最终账本权威来源，保护q／scale／index、旋转sign、稀疏坐标与值、长度frame、所有decoder及格式元数据都计入对应口径。encoder不部署，但其训练时间／峰值要报告。
- 当前正例采用每层同K的1–3阶前缀，故λ的成本为行数乘该层row bytes；本轮不引入每阶不同K／D。A/B只挑其中已提名的一种结构再比较固定与动态，不做结构×策略全网格。D64K64可在现有融合核规格内；3bit down改D64需要K96，不能机械转成K64或继续宣称旧whole-fused路径适用。

### 只有一个建议的下一实验：单一选定结构的整类两分支完整对照

待CAT方向确定一个结构后，在它已研究的**完整gate类别**实施`fixed2＋INT8＋abs005/Rice7`对`weighted_dynamic3`，其余类别和后续恢复规则保持共同设计，最终得到两份完整模型。选择gate是为了沿用CAT对A/B的实际同源结构问题；不是把down的局部收益外推为gate已有效。下一实验不同时研究多种D/H、周期路由、共享decoder或active-only训练。

共同前缀应由同一次从原始BF16按既定顺序生成，不能以旧after_o作为最终复现依赖；完整gate清单由模型结构取得。先训练并冻结双方共享的两阶base，保留该base实际采用的CAT熵／分组规则，不因接动态原型暗换为独立熵。动态臂在该共享base真实BF16残差上拟合第三阶，使用上文独立逐层目标；新gate第三阶预算预定为原CAT完整10000更新、同标量曝光、原linear日程和seed2031，不用旧down6000步推断新表示已收敛。控制臂不凭空加入无用训练，额外第三阶样本数、GPU秒和显存单独报告；这是同存储／同后恢复预算，不是等总训练算力。

控制的稀疏支持和值保持已有abs005选址／量化语义，但坐标使用已验证无损Rice7；每个新层重新实包。两臂本次的重构目标必须同为共享类别边界的实际live dense矩阵，稀疏残差和动态第三阶都从该矩阵减同一BF16前缀得到；成熟稀疏helper若硬取HF权重，须先使来源显式，不能把HF回填与live目标动态路由混作同目标比较。这与旧最终恢复成品上的`HF−W`稀疏实验有来源差异，应明确记录。动态每层cap取该层真实fixed2＋Rice7全部载荷，扣自己的三个decoder／固定状态／实际长度后分配；两臂共享同一个全模型上限并核实logical≤2.6。旧20,373,157B cap来自未压缩坐标，只能作历史对照，不能当对当前Rice7路线的公平预算。若固定密度实际超全局上限，应记录over_budget并停止该成品依赖步骤，不暗改密度或其它类别位宽。

最小原生整模实现范围建议放在新的`experiments/adaptive_native_model_20261029/`：`spec.py`解析结构／几何和数据绑定；`fit.py`完成上述独立分组拟合；`codec.py`负责规格驱动的动态文件、路由和全账本；`consumer.py`沿bridge扩展真实紧凑Linear；`checkpoint.py`保存／重建完整模型；`run_pair.py`作为薄入口复用现有恢复和完整九项下层流程。需要实现时交给sol/xhigh，当前没有创建这些文件。

这里最大的接口缺口是公共v6只识别`VAELinear`且要求每阶覆盖完整矩阵，`CompactRVQLinear`既不被它枚举，ragged码也不能通过现`checkpoint_bpw._code_specs`。不能将dense解码权重保存成HF成品后继续按短码大小报位宽；也不能在base＋override包保留已被替代的旧码／decoder却不计物理副本。隔离checkpoint须保存完整非替代状态及真实紧凑投影，按manifest重建普通VAELinear与紧凑模块，再严格load；每个投影恰好一个生效载荷，所有非投影权重及配置也实际可加载。新schema如实命名，不能伪称现v6原生支持；评测接线复用原generation／tokenizer／评分协议，只适配loader与库存，并保留64叶24742题和LongBench21／4750的完整身份核证。

恢复范围、train／aux样本、M／QA／R目标、更新数、LR和末点选择规则在两臂预先固定；同一新父需要的教师／输入缓存按真实来源重建，不能复用另一个父的旧缓存。只保留当前有依据的一段共同恢复，不自动附加全部历史尾链。原始压缩态与恢复后分别记录局部／固定train整模读出；最终两个成品均做完整九项，不以八项提前淘汰。完整九项目标仍为同协议BF16绝对下降严格<0.9pp，不能把误差下降百分比换成任务百分点。

新增关键路径的最小验证覆盖所选D/K/H的真实形状、活动长度含1／2／3和真实尾chunk、每层独立熵／梯度、两次更新及Adam时序、实际保存／fresh整模加载、冻结字段、缓存失效和完整预算。bridge单层23.59GB累计峰值不能证明整类峰值，正式恢复应以真实整类、正式batch／长度做必要少量更新；不以缩模型或缩并行规模替代。保存／重载等明确合同严格验证；新的grouped BF16对独立调用使用有依据的数值容限，不扩为训练全链逐位一致。

结果的下一步只分三种：动态完整九项更好且预算合法，则将这一结构／范围／恢复顺序纳入从HF复现候选；局部更好但恢复后九项不优，则用已保存压缩态／恢复态读出判断损失发生在哪一段，仅补有新证据的一个对照，不扩类别／加阶；完整对照的压缩态／恢复态和最终九项均不支持动态收益，则保留较好的固定分支，结束本结构的动态扩展，不能只凭局部负例省掉本次完整九项。两臂均未到<0.9pp时如实报告差距，不把两者排序当达标；是否跨模型迁移要等Qwen完整候选确定后单独验证。

### 后续接口复核：从真实native两阶导入，而非再次读取旧compact

上面的整模格式缺口随后已由[独立完整格式](../adaptive_native_model_20261029/README.md)闭合：251 native＋1真实D32K48 compact的3437字段、自包含加载、18叶两次更新与fresh M32已验证。完整临时权重按记录清理，旧小compact输入仍在。当前待补的是**同父native两阶→规格化compact前缀→未来第三阶／预算路由**的入口；不是重做旧20MB文件或8B格式短验。本节只读实物metadata／小字段和源码，没有新数值运行或GPU操作。

**融合状态。** `cat_train_pipeline.py`在输出原生payload前先把符号量化`±1/sqrt(K)`（旧quant则±1）融合到decoder第一层：`W'=2*q_scale*W`、`b'=b−q_scale*sum(W)`；再把stage标准差乘最后一层权重／bias、均值加最后bias。原生codes已经是bool01。G09经过后恢复后的decoder就是当前有效BF16参数；导入只复制，不再调用`_fuse_q_scale_into_decoder`／`_fuse_norm_into_decoder`，也不试图从已更新参数反推出原mean/std。v6 metadata没有这两项原始统计，且`_q_scale_fused`不是持久字段，不能因加载后这个Python标记为False便再次融合。新bundle的`normalization`应写“已融合、原统计不可恢复”的真实来源标记；当前consumer只把它作metadata，不能伪填mean0/std1作为原统计。

**实际源和直接导入接口。** 只读核证了C09 UUID`726b7b36-bb95-4418-b626-cc029b4f0aec`和G09 UUID`e8e5efd4-655e-462c-b6e0-fedf4e2e0346`的真实meta、原文件存在性；G09使用mmap仅观察L0 gate/up/q/down各自字段及每bank首／中／末三条实际码，CUDA未初始化。它们只是格式样例，不是新算法选层／donor规则。

推荐立即给sol/xhigh一个CPU任务：新增隔离`native_prefix.py::from_native(native_spec, linear_state, provenance)`，输入单个实际投影的metadata和原state，返回当前`CompactRVQLinear`可消费的`bundle,lengths,resolved`；不重训、不增加第三阶、不选择路由。首版限制当前已有的单part／两stage／同D、INT8输入保护、无LoRA／稀疏／输出保护拓扑，不支持就明确拒绝。用原`_rebuild_converted_modules`在一个真实形状Linear宿主上strict加载相对字段，随后以`get_stage_part_vq_storage/spec`和`get_stage_part_decoder`提取；不构造完整HF模型，不复制4.8GB父权重。原packed三维`[N,1,ceil(K/8)]`只移除已核为1的中轴，转为独立compact storage；所有decoder叶、保护q／scale／index和rotation sign原样复制，禁止解码后重新编码或重选保护。

对当前R0／LayerNorm，getter的实际映射可独立核证：`linear_in.conv.weight[s*H:(s+1)*H,:,0]`→`linear_in.weight`、同区间bias→`linear_in.bias`；`norm_out.weight[s]`／`bias[s]`→`norm_out.norm.weight`／`bias`；`linear_out.conv.weight[s*D:(s+1)*D,:,0]`及对应bias→`linear_out.weight`／`bias`。stage0／1分别取`vq_weight`／`vq_weight_s1`，中轴移除后clone为自有连续storage。decoder spec只取当前构造需要的`in_dim/out_dim/hidden_dim/num_res_blocks/norm_type/activation_type/decoder_type`，保留原BF16值；`use_checkpoint`属运行设置，不重新融合。原保护三字段名称保持，`weight_rotation.{input,output}_signs`映射为bundle的`rotation_{input,output}_signs`。geometry保存真实原形状、保护后形状、transpose、rotation spec和模块身份；没有live目标就不生成`source_variance`、目标digest或stage3 residual。

全2长度仅用于本次无训练导入，按现29B header＋zlib6生成合法frame，真实费用照计；它不是已完成的动态路由。可保存一份来源绑定的`native_prefix_v1`小产物供独立CPU进程fresh读取，不伪装旧L0-three-stage二进制。`codec.py`现仅能读历史输入，必要新增接口应接bundle或新的明确prefix文件，而不是伪造旧MAGIC/schema。`spec.py/consumer.py/checkpoint.py`当前被其他任务使用，修改须由root协调；已有格式无需为这个任务再造完整8B副本。未来固定两阶控制本可隐含length2，不能把本次测试用的全2frame额外费用偷偷加入Rice公平cap。

| K | 每行真实bytes | 每行padding bits | 当前可用实物 |
| --- | ---: | ---: | --- |
| 32 | 4 | 0 | C09 gate、G09 up |
| 34 | 5 | 6 | G09 gate |
| 38 | 5 | 2 | G09 q/k |
| 48 | 6 | 0 | G09 down/o |
| 64 | 8 | 0 | 等待A/B真实新端点，不复活已删probe／短验权重 |

码是little-endian字节内bit0对应latent坐标0，K取`stage_decoders[s].in_dim`或`stage_vq_weights[s].logical_shape[-1]`并要求一致；`stage_codebook_dims`保存的是**D，不是K**。K34末字节仅低2bit有效，K38仅低6bit有效，高padding必须为0。逻辑码费按`N*K`，物理按`N*ceil(K/8)*8`，不能把K34／38扩大成40再送decoder。当前读取的三条样例padding均为0，这不是全bank验证；实现验收再对所转换完整bank检查。

布局令实际保护后矩阵为`Wrot[O_c,I_c]`、`t=v*D+d`：非转置`vectors[v,d]=Wrot.flatten()[t]`，输入h坐标为`t % I_c`；转置取`Wrot.T.flatten()[t]`，h坐标为`t // O_c`。本次G09 gate为`[12288,4032]`、D32／N1548288，down为`[4096,12032]`、转置D32／N1540096。D64时必须从新实物重算N，不通过重排D32码得到D64码；现codec要求总元素整除D，无需预造padding数据。gate逐坐标h公式与上节root补充保持，导入任务本身不生成新的h、目标矩阵或误差分数。

**数值合同不能混淆。** 码／padding／参数叶／保护／sign／schema及新文件fresh重载应exact；同一新consumer、相同CPU dtype和chunk配置保存前后应exact。原native与新compact跨算子输出不承诺逐位相同：原`_parallel_stage_decoder`是M2，`Normalize`先将无affine LayerNorm输出舍入BF16，再分别乘／加；抽出的M1 decoder把affine集成进LayerNorm。原M2最后一层为BF16 matmul输出再加BF16 bias，M1 `F.linear`可在输出舍入前融合bias；GPU packed first-linear还保留FP32累加加bias后再cast，CPU原生fallback不等于该GPU路径。原native多stage reduction通常先FP32累加再cast，而compact是顺序BF16相加：两项和本身相同，三阶及以上一般不同。它们是明确的算子／舍入边界，不能仅凭state exact宣称旧native部署exact，也不能把所有差异都归给最后stage加法。

当前公共代码、A/B所用N snapshot与G09所用S snapshot的`autoencoder.py`、`parallel_layers.py`、`vae_linear.py`、`fused_multistage_decoder.py`、`packed_bit_linear.py`五文件SHA逐项相同；这支持复用上述算子分析，不把它扩大为两个runtime所有配置／旋转依赖相同。A/B gate是asymmetric，不能用whole-fused GPU证据代替其实际packed路径验证。

`validate_native_prefix.py`的有界验收建议为：G09一份完整K34 gate的字段／双bank padding／提取后重新pack参数exact、全2frame往返和小文件独立fresh；用真实bank预定连续8193行（覆盖8192＋尾1）比较两stage及和，CPU FP32核数学对应、BF16报告native-packed／native抽出独立stage／新compact三路差值，分开定位抽取改变与codec改变。new compact须与**相同独立stage运算**的原模块oracle exact；FP32跨打包重排沿既有2e-5/1e-6逐元素及1e-4 relative-L2上限检查，不以此容限放过字段／码损坏。BF16跨native/compact先报告maxabs、relative-L2、均方差及变更元素数，不能随意设“1%均可接受”或由CPU结果宣布GPU等价；正常舍入差不据此做补偿或阻断。另对同一G09的K32 up、K38 q、K48 down做真实小切片的schema／bitpack／布局核证，避免为了K32再引入另一份C09大来源；C09 gate的已读metadata仅保留作来源说明。不再训练、不全模型前向、不引入D64合成权重。可按需要只对一份完整gate做流式完整权重比对，不能把小切片结论写成全矩阵数值结果。

这项现在值得做，完成条件是原native→新compact的字段／几何／保存接口闭合、数值边界被实际量化，而非质量提升或D32胜出。可预先实现通用D/K/H解析、两阶提取和frame生成；必须等待选中结构的真实新live边界才能绑定其D64实物、第三阶目标／规范化、gate输入h、Rice实际cap和最终路由。正式公平配对要让固定与动态共享同一**实际部署**的两阶前缀；两臂使用共享的新compact fixed2消费者，并在该路径重算stage3 residual、各stage误差及稀疏残差，单列native→compact初态变化；不复用旧native residual后再加偏移补偿，也不把跨算子差算作动态路由收益。无需为正常BF16舍入差扩展一套补偿或恢复原native逐位算术的实现。

### 再次决策：不加旧G09 gate训练，保存真实A/B raw gate并补齐公用核心

native importer随后已完成[实际CPU验证](../adaptive_native_model_20261029/README.md)：同G09的K32／34／38／48完整字段、bank／padding、抽取重pack和fresh均通过；gate8193行的native-packed对独立BF16和relative-L2为0.566%，不是完整Linear测量。其临时prefix已按无后继用途清理，不为新任务恢复旧产物。本轮仅只读源码、metadata和mmap字段，**不启动G09 gate第三阶**；root已采纳这一取舍。

G09 gate的非转置h、K34 padding与Rice计费确有尚未执行的机制组合，但其恢复后两阶没有对应的未压live目标；若改用HF拟合，问题就是“恢复后的G09向HF回填”。其结果还要跨K34→K32／64、symmetric→asymmetric及原始CAT边界才能迁移，正负都不足以决定选中A/B后的动态路径。再做一个局部正例会增加训练、捕获和读出，却不直接改变当前决策；GPU空闲本身不是实验理由。既定CAT、G09恢复和完整评测继续，候选研究不替换它们。

已实际找到正式、恢复前的gate `round_base`。公共相对根为`/data/star/VAELLM/cat_structure_full_20261029/{arm}/native/home_star_models_Qwen_Qwen3-8B_20261029_134231/training_rounds/0004_gate_proj/round_base`：

| arm | raw gate UUID | D/K，N | 36gate实际字段／tensor bytes |
| --- | --- | --- | ---: |
| A | `94c8c25b-19dd-4c6d-852c-19f5b12c4d6b` | 32/32，1548288 | 468／476068608 |
| B | `7c0b1199-b9ff-46af-a1b7-c2f5b7dabf7f` | 64/64，774144 | 468／477252864 |

两者metadata均为`round_base`、180compressed＋72dense、36gate；completed只有q/k/v/o，`distill_stage_meta=None`。原snapshot源码在`run_after_category_distill`前实际保存，不能误取随后`after_gate_proj`的500恢复端点。两份原完整bin分别10737298477／10738482733B，当前活动源保持，不删除；每份gate只有双码、packed decoder6、保护3和sign2共13字段。应立即只读抽取两份必要36gate状态、对应36份spec和原完整metadata（约852KB），绑定UUID、原stat、字段digest、源配方／runtime身份。`cat_runtime_state.pt`各22729698B，若后续接原生类别流程需要则作为该归档的明确运行上下文保留，不复制无用tokenizer或两个完整10.74GB模型。

唯一共同dense目标是`/data/star/VAELLM/qwen_recompression_20261026/C/initial_after_o/pytorch_model.bin`中的36个`model.layers.*.mlp.gate_proj.weight`，父UUID`b63bfe79-3889-40c6-bf66-1d002893f874`，144compressed＋108dense。该父继续保护，也提供未替换骨干和真实输入上下文；无须再复制约3.62GB gate目标。归档应核raw gate以外状态与此父的实际一致性，证明二者可重建恢复前状态；若发现差异则记录保留其真实字段，不能先假定一致或掩盖差异。新B已存在真实D64数据，因此可以在归档中验证D64 importer，无需合成数据、复活已删probe或重跑拟合。

**可立即交给实现代理、但不启动quality训练的最小范围：** 在独立目录保存上述归档；补`objective.py`的真实几何／旋转输入二阶矩映射及前缀误差；补`allocation.py`的规格驱动实际费用／原四轮λ分配；补`fit.py`的resolved spec和独立逐层BSQ分组拟合。复用现有native importer／consumer／bitpack／Rice codec，不改活动源码或建立新调度器。只有实际新路径的必要CPU数值检查；不另扫组数、D/H、路由规则或优化器。

选中结构后，唯一下一训练对照仍是整类36gate fixed2＋INT8＋abs005/Rice7与weighted dynamic3。两臂共享该raw2的码／decoder／保护／旋转及新compact消费者；以共同after_o的live目标旋转向量减**该消费者当前BF16前缀**得到FP32残差，重新计算stage3全体mean/std，不取HF目标、不借旧native误差也不补转换偏移。第三阶沿已定10000更新、`B=262144/D`、seed2031、AdamW(.9,.95)/wd0、LR.003线性降零，无warmup；全向量dense训练，前两阶冻结。每层MSE／commit／两熵独立，求和更新，保持原BSQ球面／精度顺序；不能把原CAT前两阶M36 pooled熵改掉，也不能让第三阶跨M平均边际。每层shuffle／尾批／Adam步数独立；同规格同日程分组和分卡可并行，额外第三阶计算量单列。

h使用同一after_o真实前向、既定FineWeb train前64篇重算：去保护输入、input sign→forward Hadamard、逐输入坐标二阶矩再均值归一。非转置gate的每个前缀分数为`sum_d(h[(v*D+d)%I_c] * (target[v,d]-prefix_s[v,d])²)/(D*source_variance)`，与矩阵逐坐标归约核对；不能乘平均h。只保存h和tokens／来源身份即可，不必为后续每次读取持久化全部36gate激活。后64篇仍是既见train观察，不称盲测，不用其排序选择路由。

Rice控制从同一live目标减实际compact fixed2部署W生成abs005，保护列先相同，原INT8值／scale／zero和BF16相加顺序保持；Rice7只无损压坐标。动态的实际cap扣自身三decoder、保护／sign、长度frame、码padding及必要metadata，沿原四轮实包／取最后可行，不看输出择优；不加不存在的第四decoder，不把控制隐含length2改收费流，不复用旧20.37MB cap。当前新prefix的torch容器不是旧固定16KiB头，必须由实际serializer测量其开销，不能继续拿旧`HEADER_BYTES`公式宣称完整文件同cap。两臂最终完整模型logical≤2.6另行审计，packed与完整目录bytes单列；发现超预算不暗改密度或其它类别位宽。

原始压缩替换与后恢复的功能效果分开：前者用同父实际Linear／MLP输出和固定auxiliary_train小面板作诊断，不以局部／少量正确题数选点；两臂随后从自己的实际状态执行相同gate恢复、各自live up/down压缩与对应恢复，最终两份完整九项决定优劣。不得把分支gate拼回已走完另一路恢复的最终模型来假装完整顺序效果。新compact checkpoint目前仍硬要求251native＋1compact；恢复前的144native＋36compact＋72dense与最终216native＋36compact尚未覆盖，须在新流程实际需要时扩充库存／saver／恢复trainable选择，不把单层M32通过外推为整类已可运行。

停止条件是该同源两分支的预算、完整保存／重载、既定恢复和两份完整九项结束；不据先出的八项取消LongBench。局部有利但最终无收益时，只按已保存原始／恢复后读出定位一次具体衔接问题，不重复旧G09或down局部搜索；只有新证据才补一个对照。当前立即执行的工作是原件提取和上述公用核心实现，第三阶训练等选定结构，既定主线继续。

### 下一块实现交接：原500恢复的两更新、保存回训和next-up真实交接

root确认A/B raw gate归档96.2秒／exit0完成，旧包无活动依赖。优先实现一个新隔离恢复入口，输入真实144native＋36compact固定两阶＋72dense，不启动第三阶／路由或质量训练。以下是有界实现规格；当前尚未执行这条新链。

**显式改绑N runtime。** `adaptive_native_model_20261029/__init__.py`在无活动依赖后改绑`qwen_native_category_rotation/source_snapshot`，保留实际导入路径冲突检查，新manifest记录N路径／SHA；不隐式替换`sys.modules`、复制fitter或加入S兼容回退。已逐项核对consumer/importer/checkpoint直接核心14文件（bitpack、protected_channel_quant、autoencoder、parallel_layers、vae_linear、fused_multistage_decoder、packed_bit_linear、weight_preprocess、hadamard_utils、hadamard_extended、rotation.model_utils、checkpoint_v6、checkpoint_bpw、distill_precision）N/S SHA均相同；恢复common／distill、trainables／optimizer、distill_decoder、full_lora及data_prep／train_data亦相同。`train_group_vae_payload`的实际差异为N取`runtime_cfg.weight_rotation_block_size`，S取全局`vae_args.weight_rotation_block_size`；改绑N保留当前类别旋转规则。旧S报告仍是历史验证，不回写成N通过，新入口及本次真实短链覆盖N。

**不能直接调用旧remaining runner。** `run_after_category_distill`会进入`_run_canonical_remaining_family`；其`select_compressed_decoder_targets_from_inventory`拒绝compact，后续`enable_decoder_trainables`／`resolve_decoder_execution_plan`及`finalize_main_decoder_targets`也只懂VAELinear。新`recovery.py`只承接当前`remaining_lora_prefix_decoder`这一条：用原name inventory分出native、compact和精确未来dense全名，明确调用原下层功能，不更改原类型判断或伪装compact为VAELinear。

- 复用`resolve_cat_after_category_stage`、原`_prepare_after_category_dataset`／`_build_training_arguments`、collator、`DistillTeacherRuntime`和`CanonicalCatSFTTrainer`的loss计算；原五源train、top100/T1/prompt.3、dynamic padding、教师、seed、500日程均不改。新入口薄，scope／生命周期在`recovery.py`，保存hook在独立Trainer小子类；不重写训练损失、数据加载或优化器。
- 先让原`build_model_level_trainable_selection`处理144个native decoder与72个精确dense LoRA目标（rank12），`compressed_modules=()`，避免在compact上创建LoRA。PEFT会冻结base，故**在PEFT构造之后**调用36个compact的`enable_decoder_training()`，把实际decoder Parameter按身份去重加入同一个`selection.decoder_parameters`；再复用`configure_distill_precision`、`attach_model_level_optimizer_contract`。原优化器会核全部requires_grad与inventory完全相同，不能仅打开grad漏进optimizer。当前raw2预计native864＋compact432个decoder叶、dense LoRA144叶；按真实模块派生／核对，不用固定叶数替代字段完整性。decoder LR1e-5、LoRA LR1e-4与原weight decay／日程保持。
- 码、长度、保护、sign均为冻结持久字段；dense主权重在merge前冻结，允许更新的是原prefix decoder与未来dense LoRA。复用native的trainable-decode／finalize；compact按自己的FP32 master／BF16计算、版本失效和load hook运行，不走native group8打包；原native缓存路径保持。保存前后比较持久补集，不能把允许重建的非持久缓存身份当冻结合同。
- 原dense PEFT只作用于72个普通Linear，`finalize_model_level_lora(..., compressed_proxy_names=None)`的Path B可复用：它融合实际dense LoRA并移除adapter，不接触compact为compressed代理。随后原`prepare_model_export`转部署精度，分别清native和compact缓存、冻结全部参数并恢复`use_cache`。核72个未来投影已成为普通Linear、无PEFT残留、实际merged权重遵循原BF16 merge语义；正常舍入不能被误判为LoRA无更新。

**250／500常规保存必须同时接通。** 不能继续调用v6 `configure_v6_step_checkpoint/save_v6_training_step_payload`并假称mixed round_base受支持，也不能退回PEFT默认只存adapter而漏掉全部decoder。新Trainer小子类保留原训练/优化器/调度，保存hook按实际selection记录native＋compact FP32 decoder、dense LoRA及必要decoder buffers、身份／精确参数名／形状；原Trainer继续保存optimizer、scheduler、RNG及trainer state，manifest绑定独立mixed round-base和新格式。正式触发仍为原250／500；本次短验只把触发间隔改为1，在真实step1保存后继续step2，核save未改master／Adam／scheduler／随机状态。未启用resume，因此不实现或宣称exact-resume；记录可读的真实训练状态即可，不伪造step250/500或恢复成功。

独立stable checkpoint扩展为互斥且完整的native／compact／pending_dense三清单，支持当前144＋36＋72及next-up后的180＋36＋36；保存真实merged dense字段，每个目标只存一份有效载荷，不套旧251＋1硬限制。中间边界包含dense，位宽如实计费但不拿final≤2.6条件拒绝它；最终无pending dense时才执行完整成品验收。不要把这个修改单独交作任务完成，必须进入下面的真实恢复／next-up验证。

**GPU0最小实际验证顺序。** 使用root绑定的真实A/B raw2归档和共同after_o构造混合模型；可用真实B作为D64接线输入，不代表它已赢得结构选择。保持全36compact、全144native和72未来dense、真实teacher、B4／max1024及原500日程，只在Trainer callback于第二次真实更新后停止；合法dynamic padding不是强制4×1024，复用原首两批4×456／4×1024证据与实际tokens／mask核对。第一步warmup LR=0正常，第二步须有实际decoder／LoRA master更新，所有必要梯度和Adam计数有限且连续；记录真实峰值，不能引用单compact峰值替代。

step1按新常规保存路径落盘并回到step2；结束后按原dense PEFT规则融合、导出并独立fresh加载完整mixed边界，核冻结字段、当前BF16 decoder／merged dense状态和同输入真实输出。再从**该fresh保存后的模型**调用原`_collect_sorted_category_refs(category='up_proj')`，确认36份up来自本次返回的live dense权重；复用N的`train_group_vae_payload`＋`apply_group_vae_payload`做真实全36up、D32/K32、B8192、两stage各2次拟合更新，保留原保护／旋转／seed500031及已有类别统计刷新规则。这里只缩VAE更新数验证交接，不做10K质量训练、不接up500恢复。核fitter实际target digest等于刚merge／fresh后的up，不读取after_o旧up或另一分支donor；gate当前decoder和冻结码／路由／保护未被up拟合改写，替换后库存180native＋36compact＋36dense正确。

完成这条短链才可称“混合prefix恢复→中途保存回训→future LoRA融合→保存fresh→实际下一类别拟合／替换”已贯通。仍不代表第三阶／稀疏consumer、动态路由、up/down完整500恢复、最终216native＋36compact导出或九项通过；后继只对届时新增关键路径补验。当前无须再研究新算法或重复旧接口测试，实现按以上边界交sol/xhigh，root协调GPU0。
