# D16 固定两阶的尾阶短原型

用户现已批准下一批两项并行中的本尾阶短卡：仅 GPU2、整对共同60分钟绝对截止，覆盖一次真实捕获、两臂训练／编码／路由／保存／fresh／读出；不重置分段时限。Astra 已关闭原 CPU 核心窄审。`sequence.py` 为一次性薄入口，`run.py` 固定顺序执行，技术失败停止全部后继、无重试／降 batch。本文不是另一执行计划，唯一状态入口仍为[当前计划](../../docs/experience/records/w25_transfer/current_plan.md)。文档不进入源码 pins。

`bindings.py` 绑定原 D16 raw `02292fb6-5041-4f0d-bbb3-9827ec5f3a1f` 和共同 live after-up `5a94d57f-d3b9-4c97-adef-a099d659eb6a`。按 down 层号排序取索引 `0,floor(L/2),L-1`，当前为 Qwen 0/18/35；这不是最终固定层算法。前两阶 codes/decoder、保护、旋转 sign 和布局冻结，第三阶只允许 K8/K24。残差来自 live FP32 旋转目标减真实 compact 顺序 BF16 两阶重建，所有向量共享每层一个 mean/std，第三阶独立 BSQ 损失求和；不继承旧 pooled BSQ 语义。

`allocation.py` 只允许长度 2/3，仅选严格正 `e2-e3`，按收益稳定降序、相同收益沿原向量索引。最多四轮实际 zlib6/29B frame 重算费用，取最后可行轮；不按质量择轮。每层完整部署 tensor payload 上限 `22,033,116 B`，含原两阶全部字段、第三 decoder、实际码行和完整 route frame，三层不借额度。容器与元数据另报，不能把实际文件字节误称 payload。`artifact.py` 与原 `CompactRVQLinear` 消费真实逐阶 K 的短码和 BF16 decoder，不保存 dense 权重冒充压缩。

CPU 命令由已激活并核对的 `bitvae` 调用者在 tmux/nohup 后台执行：

```bash
CUDA_VISIBLE_DEVICES= python -B -u -m experiments.down_tail23_short_cmcc_20261104.validate_cpu \
  --output /home/cmcc/VAELLM/result/qwen3_8b_mmlu65_20261004/down_tail23_short_cmcc_20261104/cpu_reproduce
```

[原始报告](../../result/qwen3_8b_mmlu65_20261004/down_tail23_short_cmcc_20261104/cpu01/report.json)及[核心日志](../../result/qwen3_8b_mmlu65_20261004/down_tail23_short_cmcc_20261104/cpu01.log)：tmux `tail23_cpu01_1104`、PID `2435731`，主体 exit0／64.283 秒，CUDA 未初始化，训练更新为零。已核三层真实来源／完整共同残差／D16 每输入坐标 256 向量；两 K 独立 loss/全叶梯度；L0 两种完整真实未训练 bank 的异 K 保存和独立 fresh；原前缀／码位／路由／同输入 BF16 W 与 F.linear output exact；active 尾块 `8192+1`；禁第三阶时 bias/融合 mean 无泄漏。原 native packed 与 compact BF16 算子差仅记录，不设任意门槛。未训练候选及 fresh 散文件已在报告归并后删除，无后继用途。

CPU uniform-h 只是数值、路由与收费契约用例，不是实际 after-up 校准输入，不提供 K8/K24 收益或排序。CPU 实包 payload：K8 `20,558,904 B`、K24 `22,028,801 B`；容器／元数据分别 `11,085/11,140 B`。这不是训练结果、GPU 通过或完整 36 层路径验证。

窄审修补将梯度有限性检查放到真实更新前；未来首2步记录每层至少一个参数叶发生位移、每叶 Adam step 为1/2，末步核 step2000及参数／动量有限。参数和动量保持原 BF16，标量 step 保留 PyTorch 原生类型。受影响 CPU 检查使用真实 grouped VAE/AdamW、合成 `[17,3,16]` 两次更新，仅验证记录函数，12叶 step/精度／有限性及两次三层实际位移均通过，未执行真shape GPU 短验或2000步。证据归入原报告 `review_patch_validation`，变更见[窄审 diff](../../result/qwen3_8b_mmlu65_20261004/down_tail23_short_cmcc_20261104/review_patch.diff)；原核心 CPU 验证未重跑。

已批准短卡只含三层：真实 after-up 一次捕获原 Alpaca train512×256 前16条，前8估 h/路由、后8仅功能读出，仍属已用校准 train 池，不称盲测；每 K `[16384,3,16]`、独立 init/shuffle2031、原 BF16 AdamW 与原损失，2000 更新取原 linear10000 的前段末点，无择优。首两步真实技术检查直接并入各臂2000更新，不另建重复编码 short。完成2000与fresh即停；两者后8汇总功能误差均无改进则结束此短提名，有益也返回研究审定。**无自动扩层、扩步、full8、500 或 nine 后继。**实际运行状态及结果以独立输出 `sequence.json/report.json` 为准，启动不等于完成。

本次已完成：tmux `Tail23_pair_g2_20261104_1044`，driver `2459174/start49766641`、worker `2459184/start49766704`、fresh `2462002/start49787791`，物理GPU2。共同起点北京时间10:44:39、绝对截止11:44:39，实际10:48:26自然结束；sequence／worker／fresh均exit0，三进程退出，GPU2释放。独立输出为[运行报告](../../artifacts/down_tail23_short_cmcc_20261104/gpu_pair_20261104_1044/report.json)、[一次性执行状态](../../artifacts/down_tail23_short_cmcc_20261104/gpu_pair_20261104_1044/sequence.json)、[独立fresh](../../artifacts/down_tail23_short_cmcc_20261104/gpu_pair_20261104_1044/fresh_report.json)和各自日志。只读闭合核证、逐层未舍入值及原件SHA归[closure.json](../../artifacts/down_tail23_short_cmcc_20261104/gpu_pair_20261104_1044/closure.json)，源码pins mismatch为空。

## GPU 短卡完整结果（11月4日）

本节是此次真实训练后的结果，区别于上面的未训练CPU bank。只使用同一次真实 after-up train16 捕获：token digest `0bd1d7c6…`，input digest `45bc19c2…`，h digest `af0889cf…`；前8估h/路由，后8只作本次功能读出，仍是已有校准train池，不称盲测。三层h均为长度12032、均值1的真实非均匀二阶矩，L0/L18/L35范围分别 `.24793–6.10028/.55322–2.07452/.10659–16.81857`，未使用CPU uniform-h。

两臂均fresh初始化、真实2000更新，原linear10000／warmup0，末LR `.0024`；每层实际采样32,768,000向量，batch `[16384,3,16]`，各层init/shuffle2031，原AdamW及所有loss保持。首两步三层均有实际参数位移，12个原生参数叶的BF16参数／动量及原生FP32 scalar step1/2/2000有限且计数正确；未增加独立两步训练或择优节点。

汇总输出NMSE为三层SSE之和除以共同reference energy之和，W同样汇总；weighted residual为相同N的三层均值，不能与原坐标W指标混为一个量。相对变化由未舍入原值计算。

| 表示 | 原坐标W NMSE | 加权残差NMSE | 前8输出NMSE | 后8输出NMSE | 后8 SSE相对控制下降 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 同compact2 | .059880664 | .060840609 | .008384231 | .008177048 | — |
| K8第三阶 | .042242418 | .041785606 | .005802505 | .005660410 | 30.776853% |
| K24第三阶 | .043512706 | .037421661 | .004524818 | .004468227 | 45.356478% |

| 层 | 表示 | W NMSE | 加权残差NMSE | 前8输出NMSE | 后8输出NMSE | 第三阶覆盖 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | compact2 | .059440315 | .060629293 | .028364408 | .028227767 | 0 |
| 0 | K8 | .041984434 | .041383826 | .018992188 | .019164526 | 67.437257% |
| 0 | K24 | .043814658 | .036272534 | .016280827 | .018225718 | 23.573660% |
| 18 | compact2 | .060648780 | .061661736 | .049805702 | .050128062 | 0 |
| 18 | K8 | .042459434 | .042980074 | .034734542 | .035035056 | 66.399854% |
| 18 | K24 | .042625445 | .042352409 | .034143835 | .034712888 | 22.653620% |
| 35 | compact2 | .059520073 | .060230798 | .008312705 | .008115513 | 0 |
| 35 | K8 | .042290577 | .040992916 | .005752676 | .005617398 | 67.776911% |
| 35 | K24 | .044161407 | .033640041 | .004474100 | .004423961 | 23.736215% |

K8后8逐层SSE下降 `32.107538/30.108897/30.781976%`；K24为 `35.433369/30.751586/45.487592%`。K8的未加权W误差低于K24，K24的加权残差和实际输出误差低于K8，这是本三层／2000步／单seed条件的不同排序，不升级为完整模型精度或正式长程收益结论。

实际账本逐层如下。每层cap `22,033,116 B` 约束完整部署tensor payload，容器／元数据另报；真实文件较cap大不等于payload越界。compact2各层payload `19,571,295 B`，其中新增全2 frame为799B，原native两阶载荷仍为 `19,570,496 B`；其实际文件 `19,579,930 B`、容器／元数据8,635B。

| 层 | 第三K | 实际payload B | 未用cap B | frame B | 实际文件 B | 容器／元数据 B | 采用轮次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 22,033,078 | 38 | 378,441 | 22,044,229 | 11,151 | 4 |
| 18 | 8 | 22,032,906 | 210 | 410,223 | 22,043,973 | 11,067 | 4 |
| 35 | 8 | 22,033,044 | 72 | 367,945 | 22,044,165 | 11,121 | 4 |
| 0 | 24 | 22,030,717 | 2,399 | 270,839 | 22,041,797 | 11,080 | 3 |
| 18 | 24 | 22,030,721 | 2,395 | 355,860 | 22,041,861 | 11,140 | 3 |
| 35 | 24 | 22,030,200 | 2,916 | 255,301 | 22,041,349 | 11,149 | 3 |

K8三层payload／实际文件／容器元数据合计 `66,099,028/66,132,367/33,339 B`，K24为 `66,091,638/66,125,007/33,369 B`，平均覆盖67.204674%／23.321165%。fixed字段每层1,067,264B保持；两阶decoder为22,080B，第三decoder真实全部6叶另计6,944／11,040B，三阶总decoder为29,024／33,120B。各层原两码bank各3,080,192行从未删改，第三只存实际active行，码行按K为1／3B；本三层所有码的packed bits与logical code bits相等、码padding为0。frame已含原2bit lengths／zlib6／29B头，所有字段计费。实际完整模型未组装，不把此三层账本当作36层成品的实测位宽或磁盘大小。

六个候选均严格只有2/3长度、严格正加权实际增益／稳定tie，最多4轮实包取最后可行轮；冻结前两阶code／decoder／normalization／保护／sign的完整digest不变。独立fresh只读取captured inputs及9个实压缩包，所有持久字段、route/code、同输入BF16 W及F.linear输出完全一致，未读取父权重或保存dense权重冒充压缩。

整对包括capture、两臂fit/encode/route/save/readout和fresh共227.274秒，单卡总占用时间上界 `.063132 GPUh`；两臂fit加全向量编码约72.172／72.424秒，各臂完整阶段约83.104／82.551秒。主worker含完整after-up捕获的峰allocated／reserved为36.918／38.420GiB，非仅小VAE训练峰。保存时九个压缩包均通过同次独立fresh；两者后8均改善，本次“两者均无改善”的终止条件未触发。本短卡未进行36层／full8／500／九项。用户已据阶段汇报选择下一批仅K24全36down raw、总上限4 GPUh；由新隔离实现接续，本包不自动启动后续，启动与结果以唯一当前计划及新包证据为准。

本次没有分段reset GPU峰，无法从原记录分离K24训练本体峰；第2000步日志约70.757秒，至fit返回约72.424秒的差值还含同步／提取／编码／记录开销，不作为纯编码精确计时。保存内容只有部署decoder/codes/fixed/frame及真实capture/h，没有encoder、optimizer、scheduler或RNG可精确resume；不能将short2000部署包当作原10K轨迹续训状态。

11月4日按已确认后继用途收尾：保留三层K24及同层compact2六个部署包、真实`capture_inputs.pt`／`route_h.pt`，供新全36bridge核同源输入及局部对照；全部来源绑定、配置／源码pins、原始指标、核心日志、退出、fresh指纹及CPU证据继续保留。实现负责人已明确确认全36bridge不读取原K8且没有已安排K8后继，因此只删除本运行目录下层0／18／35的三个`model.layers.<层>.mlp.down_proj.K8.pt`，文件字节合计`66,132,367 B`，目标inode的allocated合计`66,134,016 B`。各文件nlink1、无符号链接；可读FD／maps无引用，受限的本用户会话管理进程及其它UID检查范围在原[closure收尾凭据](../../artifacts/down_tail23_short_cmcc_20261104/gpu_pair_20261104_1044/closure.json)中明示，未用共享磁盘空闲量变化计费。唯一schema／来源相对digest已先归并到该凭据，其余完整resolved元数据与原fresh指纹保留；K8部署载荷现已不可加载，原九包fresh清单仅作历史证据。删除后保留包、真实输入／h及同目录证据SHA一致，原D16 raw／final文件状态和GPU0 current36down prepared／1019源码pins一致。清理只反映没有后继用途，不推翻K8已测正收益或将其写成被支配算法。本次无新增经验，仅执行已确认生命周期收尾。
