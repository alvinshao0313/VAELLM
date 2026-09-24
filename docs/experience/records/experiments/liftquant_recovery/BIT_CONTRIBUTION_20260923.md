# 硬码翻转是否有价值：受控诊断（2026-09-23）

用户要求判断编码翻转的实际价值。此前384步仅证明硬码可更新；2训练/2留出时留出MSE恶化7.99%，没有decoder-only对照，不能单独归因于编码。因此本次另设受控诊断，不使用旧运行充当新基线。

## 参考经验与改变条件

参考 [编码坐标和更新预算](../../../lessons/liftquant_recovery.md)、[公平对照](../../../lessons/experiment_design.md)。旧recovery_data.calibration有放回采文档，再按样本位置切分，可能同篇文档跨训练/留出；新诊断独立采用文本hash去重后按文档划分，拒绝跨集合相同token窗口。没有修改已有LiftQuant采样入口。

CPU预检退出0：原32篇均唯一且至少129token，能够划成16训练/16留出，每篇取1个128token随机窗口。固定seed42，token输入SHA256和全部文档hash/窗口起点记录到主运行manifest；抽样仍局限第一shard前32条，不能代表完整RedPajama。

## 固定对照

输入均为原始 `result/linear_output/distill_init`；只改block9现有七个压缩Linear，其他参数冻结；单stage、无旋转、无LoRA。GPU4共享，PyTorch allocator上限12GiB，bitvae环境。两臂串行。

- A：decoder-only，硬码和代理冻结；原decoder全部参数可学。
- B：joint，全部编码代理与原decoder联合更新。
- 相同16训练文档、batch2、384次更新（48轮固定顺序）、同FP teacher输入与目标、whole-block MSE、BF16 packed硬前向、seed42。
- decoder lr1.25e-5；joint code lr=min(2e-5, scaled_proxy.std()/50)；固定敏感度坐标和官方实际AdamW/余弦调用序列。
- 两臂都使用RecoveryLinear，避免A改走fused native路径；在首次实际翻码之前比较逐步loss，检查decoder LR序列完全相同、初始输出hash相同。
- 在固定0/96/192/384步记录逐文档MSE，报告固定384步，不用留出挑最优步骤或调参。
- 最后固定B最终decoder，单独恢复初始硬码，测量同样指标。该终态消融度量给定decoder下最终码的贡献，不等价于一个独立训练臂。

## 预先确定的判断口径

主比较为 B−A 在16个独立留出文档上的完整压缩student next-token NLL，辅以teacher-aligned block MSE；负差值表示改善。同时比较B和初始模型，报告改善文档数、原始逐文档值与10000次文档配对bootstrap区间。token和隐藏维度不当独立样本，bootstrap不覆盖训练seed差异。

- B优于A和初始模型、多数留出文档改善且NLL差区间完全低于0：当前有限条件下有正面预测证据；仍不能宣称正式下游准确率提升或超过LiftQuant。
- 只有block MSE改善、NLL无改善：局部蒸馏收益，没有完整模型预测收益证据。
- B不优于A：当前配置不支持把联合翻码作为更好的默认方案；不能推广为所有编码学习均无用。
- B优于A且恢复初始码后变差：同时支持允许编码学习的收益及最终编码的正贡献。
- B不优于A但恢复初始码后变差：说明共同适应，不能证明联合训练比decoder-only更值得。
- 区间跨0、方向不一致或无实际翻码：结论为证据不足，不能强行判有用。

## 验证与启动记录（完成结果见下节）

新增独立模块：ablation_train.py（配对训练和严格native对齐）、ablation_eval.py（文档切分/完整student NLL/配对统计）、bit_ablation.py（原态复位、两臂、code-revert、冻结检查）、run_bit_ablation.sh（固定配置）。已有训练源文件未改，没有Git提交或环境变更。

后台任务 `lq_bit_ablation_20260923_01`，启动shell PID1722437，先执行2训练/2留出×64token、每臂1步的最小GPU集成检查。NLL额外与已有streamed_inference包装下的完整模型普通前向核对。只有该进程退出0，才自动启动上述16/16×128、每臂384步对照；失败则停止，不自动绕过检查。

- 主入口：`bash experiments/liftquant_recovery/run_bit_ablation.sh .result/liftquant_recovery/bit_ablation_20260923_01`
- 集成检查日志/输出：`.result/liftquant_recovery/bit_ablation_integration_20260923_01.log`及同名前缀目录。
- 主诊断日志/输出：`.result/liftquant_recovery/bit_ablation_20260923_01.log`及同名前缀目录。
- 两阶段分别写同名前缀`.exit`；主结果为results.json、summary.json与manifest.json。

本次不导出大型模型，仅保留最小指标/配置/核心日志/复现入口。启动确认后按项目规范不持续监控、不等待完成；正式全量实验和下游基准未启动。此处记录启动时状态；新运行已完成，结论见下节。

启动确认：tmux会话存在，日志显示bitvae/Python3.11.13、CUDA环境识别和模型分片加载正常，尚未据此宣称GPU集成PASS或主对照完成；未持续监控。CPU文档预检的输入SHA256为 `ae7132f5e6f4f53fa11bf48d4933d281fa42b6081dfde295d1d69553f60bd872`。其退出0、32篇满足条件的事实已归并本文后，删除已完成且无使用者的CPU预检目录和日志/退出散文件，共83,466字节；保留正在执行的集成任务产物及初始模型。共享磁盘收尾核查可用约956GiB，其总量变化来自共享活动，不能全部归因于本任务。

## 2026-09-23 完成结果：相对decoder-only有正面证据

集成和主诊断均退出0、summary为PASS。主诊断362.049秒（约6分钟），两臂各384步；峰值allocated7.809GiB。独立实现的分层完整student NLL与已有streamed_inference普通前向逐文档比较最大差0。

配对控制核查：初始block输出hash相同，decoder每步实际学习率完全相同；前107步loss最大差0。第107步optimizer更新后首次翻码；decoder-only最终0码位改变，joint最终相对初始改变2,644,443 / 385,875,968码位（0.685309%）。累计翻转事件4,694,457含反复翻转，不作为独立码位数。两臂冻结检查通过，32篇文档的训练硬前向/native输出最大差0。

| 状态 | 训练block MSE | 留出block MSE | 完整模型留出NLL（越低越好） |
|---|---:|---:|---:|
| 原始初始化 | — | 0.0706919909 | 7.4591273069 |
| decoder-only | 0.0619549991 | 0.0626463748 | 7.5222655833 |
| joint编码+decoder | 0.0219377386 | 0.0554548462 | 7.3834794611 |
| 固定joint decoder、恢复初始编码 | — | 0.0634748631 | 7.4734945446 |

固定终点配对比较（差值均为candidate−reference）：

- joint vs decoder-only：留出block MSE降低11.4796%，16/16篇改善，差值bootstrap95%区间[-0.00837388,-0.00604073]；完整模型NLL降低1.8450%，12/16篇改善，差值-0.13878612，区间[-0.26884450,-0.02283976]。两个区间都不跨0。
- joint vs初始模型：留出block MSE降低21.5543%，16/16篇改善；完整模型NLL平均降低1.0142%，12/16篇改善，但差值区间[-0.19493594,0.04142993]跨0，不能认定稳定超越初始模型。
- 恢复初始码 vs保留joint码：留出block MSE增加14.4622%，16/16篇恶化；NLL增加1.2191%，13/16篇恶化，但NLL差值区间[-0.02258780,0.20668060]跨0。给定joint decoder的编码贡献在block误差上有力，在NLL上只有方向性支持，不能夸大为已经单独证明NLL因果收益。

当前判断：在这个固定样本、单block、同预算对照里，允许编码学习比只训练decoder更有效，已有独立留出预测收益证据，值得保留该方向；不是只有训练集拟合能力增强。但尚不足以宣布相对初始模型稳定提升、全模型逐层恢复成功或正式下游准确率提升。预先规定的强条件（同时稳定胜过A和初始模型）尚未全部满足，不能事后降低口径。本次没有启动新的实验或正式训练。

结果与早前2训练样本的过拟合现象条件不同：这次16训练/16文档分离留出、128token、固定384步。不能据此独立证明仅扩大数据量是差异原因，也不能用单seed、小样本bootstrap覆盖跨seed和领域不确定性。

原始证据：主目录 `.result/liftquant_recovery/bit_ablation_20260923_01/` 的manifest/results/summary/calibration_ids；集成目录同名加integration；两者核心日志仍在目录旁。源码入口和hash、逐文档误差、每步loss/LR、翻码统计均保留。

收尾：确认任务结束且四份progress JSON的每个字段都已完整包含于各自results对应训练臂后，删除4份重复progress及2份已归并summary的.exit散文件，共731,570字节。主结果保留约780KiB，集成约176KiB；本次未导出模型，无新增模型权重待清理，初始distill_init受保护。共享盘当时可用约956GiB。未改训练源码、环境或提交Git。
