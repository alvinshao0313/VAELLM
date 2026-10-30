# C/G 对齐恢复：J32 与 B32 局部更新基准

这是[同恢复链对照](../cat_aligned_recovery_20261028/README.md)的05／06阶段。固定历史层号与配方用作基准，明确记录父来源、更新范围、数据和预算，便于后续消融；本实现不把L18／L9或历史最佳步点称为可跨模型直接照搬的选择规则。

| 阶段 | 必须存在的实际父 | 本阶段更新 | 原生冻结补集 |
| --- | --- | --- | --- |
| `J32`／`05_J32` | 本profile已verified的 `04_C1/final_model` | L18 down的6个D张量＋2个原生主码bank；独立fresh32 | 3415字段，包括全部保护及其余投影 |
| `B32`／`06_B32` | 本profile已verified的 `05_J32/native_final/final_model` | L9 gate／up／down的18个D张量；独立fresh32 | 3405字段，包括全部504 bank、保护、L18及其他249个投影 |

所有父链都核实际审计、UUID、report、exit0、metadata SHA和权重stat；缺少新04或05便拒绝，不生成未来UUID或正式prepared。旧C1、C/G09只用于CPU几何可行性，不充当新父。

## 数据与共享的04初态保留目标

只读复用已冻结[03模块](../cat_aligned_protection_20261028/README.md)的 `data.load_materials()`，每阶段取其原N32的组0..31一次：Maux1024、Qasper32、NarrativeQA32、HotpotQA192，每组M32＋QA8。M目标仅混合一次 `.5raw-fullF+.5gold`；不使用旧QA teacher-forcing目标。当前学生每步以原生greedy读取完整prompt／真实EOS，Qasper／Narrative上限128、Hotpot32；full-F教师在同一完整prefix上给出全151936词表KL。

M／QA／R各权重.5，不再归一化。R沿原task、doc、branch、实际continuation token依次取均值。**两个阶段共同使用本profile真实新04冻结初态的R896**；06的学生父是新05，其R教师仍是新04，不改成05。缓存只由03的显式 `retention_cache.build_native` 生成一次，传入实际04目录、meta、audit binding、profile、schema和独立 `<profile>/04_initial_R`。原train七任务各128文档、所有分支、15462个continuation token不变；BF16 logits→FP32全词表log_softmax，raw概率载荷9,396,937,728 bytes/profile，含state/input/payload SHA和归一化回读。不能复用03的新02-R或旧C1-R。

## 05：保持原持续U与离散投影数学

数学沿旧 `qwen_c1_eta_retention128_20261020/pv_runtime.py` 的η=.001、R=.5配方；`pv_proxy.py`／`pv_tiles.py`直接按独立模块名只读复用，`selected_capture.py`／`selected_projection.py`仅改模块绑定，全部函数／类AST与原核一致。

六个D使用FP32 master／BF16计算，AdamW(.9,.95)、eps1e-8、WD0、clip1.5，LR为 `1e-6*(1-(step-1)/500)`。独立U为未保护原坐标绝对FP32 `[4096,12032]`，同betas／eps／WD，η=.001恒定，无D日程、不clip、不除16。每步M1＋QA8＋R7共16次实际前后向，在真实BF16 decoded W上捕获BF16 gW并累加FP32；hook返回None，保留原D反传，不另做VJP。

严格按D.step→U.step→投影执行。U只在第一步从实际training-packed BF16 W初始化一次；之后持续原U和Adam矩，不重锚、不平移旧delta、不重置矩。投影按新D＋旧码的实际BF16／group8 Wp，在48128个32² tile上稳定排序取top482。保留实际L18 K48／D32／S2／transpose／block32／保护256；stage0→1、位置0→31，每次比较current与48个真实单bit邻居，仅严格SSE下降，current赢平局，邻居取最早bit，后续选择基于此前已接受状态。

完整proposal再用实际group8解码，FP64计算真实tile位移，取最长优先前缀满足 `||ΔW_code||≤.01||Wp_unprotected||`；最终原生目标SSE变差只回滚两bank，D、U及两个Adam保留。零翻码、prefix0、拒码都是合法结果。

旧128终态／32、64、128快照合同未复用；这里实际计划和终态均为32。只保存6个BF16 D＋2个uint8 bank，不写U、gW、master、Adam或RNG，不声称exact resume。

## 06：实际18D与三组原生缓存

只沿旧 `mlpd_runtime.py`／`mlpd_native.py` 的B范围更新18D，单AdamW、共同clip1.5、原500步日程前32；无U、码更新、guard或新loss。参数量由实际父读取：现有C09／旧C1的18D为55744元素，G09因gate K34为56256；这只是历史几何可行性，正式仍从新05读取，不改G为旧形状。

原生生成按各类别真实signature独立选group8，L9对应L8–15的gate／up／down三组；保留249个冻结cache。05同理只涉及L16–23的一组down并保留251个cache。临时只把选中D切到BF16视图，生成后恢复同一Parameter、FP32 storage、梯度和非空Adam，清选中投影／KV缓存；整个模型的其他原生缓存引用保持。旧baseline100及16／32观察不参与本训练链，不借旧J32／C1面板。

两阶段均经冻结03的 `deployment.save_and_verify` 创建新UUID，独立strict3423、核504 bank和实际几何；logical≤2.6、rank0，并与**实际父**的logical、packed、逐组件ledger一致。没有调用写死旧2.497／2.510或max2.5的 `head_export.check_budget`。

## 入口与执行顺序

所有新模块名以 `selected_` 开头，避免遮蔽冻结03或旧 `pv_*` 的bindings。03的py／sh保持冻结，未修改旧源码、01／02／Llama或已有实验依赖。

调用者先激活 `bitvae`并核Python，cwd为 `rotation_factor_probe/source_snapshot_20261005_01`。下面是执行入口说明；每次由tmux／nohup后台执行、落独立日志，GPU须已分配。本轮没有启动GPU步骤。

```bash
export VAELLM_ALIGNED_PROFILE=G
# 无学生绑定的CPU材料检查，输出共同selected_cpu_materials.json。
CUDA_VISIBLE_DEVICES='' python -B /home/star/VAELLM/experiments/cat_aligned_selected_20261028/selected_prepare.py --materials-only

# 真04核证后，生成本profile唯一R缓存；先设置VAELLM_ALIGNED_GPU。
CUDA_VISIBLE_DEVICES="$VAELLM_ALIGNED_GPU" python -B -u /home/star/VAELLM/experiments/cat_aligned_selected_20261028/selected_cache.py --gpu "$VAELLM_ALIGNED_GPU"
CUDA_VISIBLE_DEVICES='' python -B /home/star/VAELLM/experiments/cat_aligned_selected_20261028/selected_prepare.py --stage J32

# 新native／32出口桥接待真实G两更新核验：QA[19,0]、R[17,28]。
CUDA_VISIBLE_DEVICES="$VAELLM_ALIGNED_GPU" python -B -u /home/star/VAELLM/experiments/cat_aligned_selected_20261028/selected_run.py \
  --stage J32 --prepared "/data/star/VAELLM/cat_aligned_recovery_20261028/$VAELLM_ALIGNED_PROFILE/05_J32/prepared.json" \
  --phase short2 --gpu "$VAELLM_ALIGNED_GPU" --timeout-seconds 7200
```

short2从真实父fresh启动，必须完成主循环、非空Adam往返、R、冻结、native compact及独立原生出口。通过后用 `run_J32.sh` 启动fresh formal32；完成后 `selected_export.py --report <05_J32>/formal/report.json --output-dir <05_J32>/native_final/final_model`独立导出。原生导出复用成熟加载上下文；它没有模型前向，不将其称为已验证的CUDA隐藏CPU构造。

05原生审计verified后才能 `selected_prepare.py --stage B32`，之后同样以 `--stage B32`、`06_B32/prepared.json` 运行short2，再用 `run_B32.sh`正式32及对应 `selected_export.py` 出口。两个阶段不自动串跑、等待、重试或轮询。

C可在formal入口显式给 `--short-evidence <G同阶段short2/report.json>` 复用G接线证据；必须以两份**实际新父prepared**证明相同代码／数据／方法及逐叶shape覆盖，同时独立核C自己的真实父和04-R。09／旧C1的CPU可行性不能代替该判断。

## 2026-10-28 实际CPU结果与未完成项

共同材料 `/data/star/VAELLM/cat_aligned_recovery_20261028/selected_cpu_materials.json` 与核心 `selected_cpu_materials.log` 已由真实 `bitvae`／tmux入口完成exit0，耗时50.68秒。通过原N32一次曝光及目标、04／05未就绪拒绝、真实C/G09和旧C1的动态scope／metadata组序、U原核两步数值及新η=.001绑定、诚实2／32保存合同与伪128拒绝、capture／projection定义AST一致、全部实际入口导入／Python AST／shell语法。复用冻结03已经实测的256 QA／768次EOS、cap与完整prefix重放，不重复伪造新源QA证据。

首次CPU检查误将每层rotation seed计入prewarm分组；实际原核signature按decoder／shape分组，随后各peer独立旋转。已按真实函数修正CPU检查；运行时始终调用真实 `_build_vae_linear_prewarm_signature`，未改变分组或旋转算法。失败根因与修复后退出保留在同一核心日志。

**新04／05仍未就绪：正式prepared、新04-R概率、GPU短验、05／06训练和原生导出均未执行。** CPU材料没有绑定新父，不证明GPU资源峰值、桥接或九项收益。旧数学／调用语义证据可复用；新接线待真实G两更新，05额外覆盖16项gW、U一次初始化、双Adam连续与真实投影，06覆盖G形状／三类group8及249冻结cache，再按实际证据决定C复用，不能机械扩成四份短验。参考[编码与投影经验](../../docs/experience/lessons/liftquant_recovery.md)、[checkpoint经验](../../docs/experience/lessons/checkpoint_lifecycle.md)及[KD边界](../../docs/experience/lessons/kd_and_data.md)；本次无新增精度结论，暂无新增技术经验。
