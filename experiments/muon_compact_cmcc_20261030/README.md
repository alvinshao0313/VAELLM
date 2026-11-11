# Muon 的 native／compact 拓扑适配

本目录是隔离的真实模型接口验证，不修改旧 Muon、N snapshot 或正在运行的 dynamic／Rice 接续源码。唯一 GPU 预算为 GPU6 上完整 Qwen3-8B 的两次更新；不提供32步入口。当前状态和原始证据在 `/data1/cmcc/VAELLM/artifacts/muon_compact_cmcc_20261030/`。

真实来源为 B 最终成品 UUID `87571461-7310-46e5-bc2e-904baf1818ed`：`/data1/cmcc/VAELLM/artifacts/cat_structure_full_20261029/B/native/home_star_models_Qwen_Qwen3-8B_20261029_134231/final_model/`。仅36个gate经既有 `native_prefix.from_native` 转成固定两阶 `CompactRVQLinear`；复制已经部署的 BF16 decoder、已融合 norm／BSQ，不重拟合、不重复融合、不引入其它父或短验权重。216个native及所有非gate字段保持。它只是后续 C0 的拓扑接口父，不能称为 fresh BF16 基座或精度收益。

## 接口和最小验证

`optimizer.py` 按实际 metadata、decoder 模块及显式 `linear_in/out` 所有权派生矩阵。native packed Conv1d 的每stage独立；当前 compact importer 的实际叶为 `decoders.i.linear_in.weight`／`linear_out.weight`（`nn.Linear`，二维），每leaf一个矩阵。norm 为 `norm_out.norm.weight/bias`，与bias一起走辅助Adam；不按张量维度推测norm。首次CPU检查保留了将compact误记为 `.conv.weight` 引起的 `KeyError`，这是字段发现，不是算法负例。

Muon保留原NS5系数 `(3.4445,-4.7750,2.0315)`、FP32 momentum `.95`、Nesterov、BF16 NS、`.2*sqrt(max(m,n))`；辅助Adam保留 `(.9,.95)/eps1e-8/WD0`。只做一次joint clip1.5；κ从首个真实clip梯度的fresh Adam与base-Muon FP32矩阵候选norm比率重新计算，之后固定。两组均使用原500线性日程的前两步，不复用任何CPU随机梯度κ。

`lifecycle.py` 分别管理native group8／`_cached_weight`／trainable_decode和compact自己的decoder训练、缓存及route。生成时暂时用BF16参数data，回训恢复原FP32 master的Parameter和storage，逐项核optimizer对象、momentum／Adam对象和值、κ与LR连续；两个消费者缓存和KV必须清空。冻结补集由实际state排除decoder叶派生，包含码、长度、保护、旋转及Rice字段，不套用旧G09固定字段数。

数据仍是原H96配方的 group0/1，每组完整M32＋QA8、seed137、R0、原prompt和原生成上限。`inputs.py` 在迁入实体上验证原JSON／SHA／实际case和target，沿用既有数据验证与M混合函数，语义对应 `cat_aligned_opd.bindings.load_inputs`：只保留1024份一次半F半gold混合M目标；所有旧学生结果和QA目标丢弃。学生native OPD与QA全词表教师目标逐组重新生成。当前HF教师5shard／Tokenizer和源非投影147字段在CPU准备中核身份，训练教师直接从同一真实HF加载，保持BF16／FA2／FP32 RoPE。

末态真实BF16导出后，`fresh.py` 在独立Python进程仅用该成品重建；核全部持久字段、冻结补集、账本和完整同输入的BF16最后位置logits。跨native packed与compact独立stage算子的正常BF16差异单列记录，不作数值补偿，不将其解释为训练收益。

## 复现

调用者先激活 `bitvae`，在项目根运行。所有验证用tmux/nohup后台，启动前只核指定GPU6，不占GPU7。首次准备不带复用参数；`--reuse-fixed2-parent` 仅用于重新核对已有真实转换父后补CPU分组验证，不修改该父。

```bash
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.muon_compact_cmcc_20261030.prepare
CUDA_VISIBLE_DEVICES='' python -B -u -m experiments.muon_compact_cmcc_20261030.prepare_payloads
CUDA_VISIBLE_DEVICES=6 python -B -u -m experiments.muon_compact_cmcc_20261030.run --gpu 6 --timeout-seconds 7200
CUDA_VISIBLE_DEVICES=6 python -B -u -m experiments.muon_compact_cmcc_20261030.fresh
```

`prepare_payloads.py`只从已完成的真实36份dynamic3和36份Rice载荷检查decoder布局、冻结字段与实际serializer往返，不组装C1模型，不给出其完整GPU通过结论。各自正式final_model出现后仍需补该父的完整模型OPD生命周期验证。

## 实际结果（2026-10-30，已完成）

`prepare_v2.exit=0`，CPU分组、每stage NS扰动隔离／重排、两次随机梯度数学检查及momentum／Adam／κ／双LR内存保存重载通过；CPU随机κ没有用于GPU。实际1728个decoder叶／5,336,064参数，其中576个Muon叶拆1008矩阵、5,124,096元素，1152个辅助Adam叶／211,968元素；二维native norm明确归aux Adam。转换前后的全部非gate字段和36gate的decoder／码／保护／sign逐项exact；新接口父UUID为 `bc80c915-ed90-443f-b5ac-5aef6bbcbc42`，完整3675字段独立strict exact。当前HF五个真实shard的SHA、Tokenizer／配置SHA及源非投影147字段身份通过。

16:52在GPU6为1MiB、无compute时单次启动tmux `Muon_compact_short2_GPU6_20261030`，shell PID182337、训练worker182340；仅GPU6可见，GPU7未占。两组完整真实更新已完成，训练＋BF16导出耗时597.4831956620037秒；`short2/run.exit=0`。第二组生成时，1728份FP32 master／Parameter／storage和576momentum＋1152Adam对象及值、κ、两组LR全部exact。两个更新窗口各1008／1008独立矩阵梯度非零；所有其它实际字段冻结，教师不变，最终源恢复exact。

| 实测项 | 第一次更新 | 第二次更新 |
| --- | --- | --- |
| 原group／生成tokens | 0／205 | 1／232 |
| 矩阵LR／aux LR | 5.567205815587792e-6／1e-6 | 5.556071403956617e-6／9.98e-7 |
| NS5 wall秒 | 0.39111041788419243 | 0.39681585498328786 |
| BF16 decoder更新L2 | 0.0008193894695734796 | 0.0014764825174262582 |
| 训练峰值allocated bytes | 36,385,464,320 | 37,746,255,872 |

首真实joint clip后fresh κ为 **5.567205815587792**；fresh Adam矩阵候选norm为0.0022603391998985757，base Muon候选norm为0.00040600963477401567。上述匹配针对未舍入FP32候选，不是BF16 decoder／解码权重位移等价。真实完整gate native packed与compact独立stage消费者在一份完整原坐标Linear输入上的relative-L2为0.00706329680227732、max-abs0.0625，全部有限；该正常跨算子BF16差异未作补偿，不据此声称收益。

独立fresh worker189362仅加载新导出，`short2/fresh_audit.json`为verified，外层 `short2.exit=0`：**3675全字段strict exact、1947冻结字段exact、504bank、216native／36compact／0dense**；原完整148-token输入的最后位置全词表BF16 logits exact且有限。新短验UUID `ebb363af-750e-4152-b5c8-d2961814ddec`。末态实际账本logical **2.4987414816151494**／packed **2.5121161419412363** bpw，均计实际decoder、码、保护、旋转、length frame等，packed另计padding；全模型tensor bytes4,671,007,680，完整目录4,674,024,756字节（清理前，含容器和metadata），不将逻辑位宽或理论压缩率当文件大小。

`future_payloads_CPU.exit=0`：真实已完成36份dynamic3＋36份Rice的实际decoder布局、FP32训练／BF16值往返、冻结码／length／保护／旋转／Rice字段和完整serializer往返均通过；CUDA未初始化。该CPU层载荷证据没有组装C1、没有完整final父，也没有其GPU通过或质量结论。正式final出现后只补该路径，不能借当前C0的short2宣称C1已过。

本次只证明上述完整8B、两次真实更新和部署接口，未运行32步、九项质量或fresh BF16整链。初次CPU字段误记的失败日志 `prepare.log`／exit1保留；修订CPU证据为 `prepared.json`／`prepare_v2.log`／exit0。实际argv、PID及源码SHA见 `short2/execution.json`；两步完整轨迹、教师target绑定、冻结证明及位移见 `short2/report.json`；独导审计见 `short2/fresh_audit.json`，全持久state哈希见 `short2/expected_state.json`。原训练report保持独导前状态文字，独导完成事实由独立audit和外层真实exit0共同证明。

### 收尾与经验增量

GPU6已恢复1MiB、无本任务compute，全部一次性验证进程自然退出，未保留占卡或监控。核验完成后按持续授权清理两份验证生成的完整weight文件、decoder endpoint及同输入临时fixture共4文件，实际释放 **9,356,165,120 allocated bytes**；manifest／config、完整state哈希、报告、审计、核心日志保留，清理前后保护记录SHA及原始B权重stat不变，精确路径与保留理由见 `cleanup.json`。这些短验权重没有正式后续用途，不能作32步或质量实验初态。再次复验须使用独立输出根重新准备，不复用已收尾的权重路径。

新增实现证据：compact实际容器可能为`nn.Linear`和`norm_out.norm`，应按模块所有权及metadata分组，不能机械套native `.conv`字段；native与compact的缓存、BF16临时生成和FP32 master恢复必须分别处理。本配置真实两更新及独立导出支持该接口规则，不能外推为优化器收益或动态／Rice完整GPU支持。既有转换数值边界沿用，无新增质量经验。主线程统一维护当前计划与经验导航。

参考：[Muon原拓扑及数学证据](../g09_recovery_20261029/README.md#g09同父muon32对照10月29日)、[compact与已融合native前缀的转换边界](../../docs/experience/lessons/checkpoint_lifecycle.md#compact依赖闭合与native前缀转换)。
