# 逐层恢复工程优化与等价性验证（2026-09-23）

run_id：`engineering_20260923_01`。状态：已完成；CPU、真实GPU新旧等价性、真实两层边界恢复及正式入口参数解析均PASS。结论：在注明的小规模配置下，工程改动保持数值轨迹并降低开销；不代表正式下游效果提升。未运行正式实验、未提交Git、未修改环境。

## 固定条件与源码

初始模型为 `result/linear_output/distill_init`，无旋转、单stage、无LoRA。本次保持 FP teacher align=1、whole-block MSE、原敏感度码坐标/round-clamp STE、BF16 packed 前向、FP32代理与Adam、学习率及官方实际调度序列，数据采样协议和可学习参数范围不变。

参考：[编码恢复经验](../../../lessons/liftquant_recovery.md)、[保存与恢复](../../../lessons/checkpoint_lifecycle.md)。码VJP继续使用前向实际BF16舍入权重，保留1/s及饱和门控，不直接套用SparseBit的FP16/符号阈值更新核。没有启用新的Triton投影/VJP核，也没有以低精度优化器换取速度。

修改前实际源码与SHA在 `.result/liftquant_recovery/engineering_20260923_01/reference_source/`，作为永久回归参考保留。实际测试源码SHA、配置、token与原始指标均在结果JSON。其他未提交改动和旧结果不覆盖。

## 实现内容

| 文件 | 行为变化 |
| --- | --- |
| `all_bits.py` | 按需跳过无用梯度；投影/统计在GPU归约后成批回传；缓存popcount。每步硬码投影与原数学不变，旧Python统计接口保持兼容。 |
| `block_train.py` | 单遍累计整体/训练/留出MSE；完整审计改为首步、末步及每32步；每步仍检查loss/梯度/更新参数有限性并记录loss/LR/翻码。显存峰值明确标为进程峰值。 |
| `recovery_runtime.py` | CPU hidden预分配并逐批填充，避免正式4096×2048配置构造时额外约64GiB拼接副本；输入+目标约128GiB常驻需求仍存在。 |
| `run_checkpoint_recovery.sh` | 修正Arrow参数位置；固定4096条、2048token、batch2、2epochs、code2e-5、decoder1.25e-5不变。12GiB仍是当前冒烟资源限制，未据此证明正式长序列可运行。 |
| `recovery_resume.py` / `recover.py` | 每个block完成后原子保存单个latest边界。恢复检查初始模型和FP teacher实际权重、代码/库版本、配置、token及完成前缀，刷新native缓存并严格核对输入/输出，在原边界恢复RNG。 |

断点只包含累计已完成块的native硬码/decoder、指标、严格重载样例和RNG，不保存block内proxy/Adam。因此支持层边界恢复，不称作block内精确续训；恢复仍要求新输出目录，旧结果不覆盖。

## 验证与结果

GPU为共享A800的物理4号卡，bitvae，单block驻留，12GiB PyTorch allocator上限。任务通过tmux串行执行，启动后没有轮询等待循环。

1. **独立数学与梯度分支**：14个真实decoder通过既有独立数学参考标准（未放宽误差门限）；7种requires_grad组合的新旧前向与所需梯度逐元素相等，含decoder-only、code-only及bias-only。
2. **边界投影**：实际1,572,864行压缩Linear，跨65,536行边界构造阈值/饱和值。新旧默认/设备/无统计模式的packed码一致；更新计数5,921,410bit一致，统计一致。人工翻转仅是回归输入，不代表有益训练翻码。
3. **真实block9新旧对照**：2训练/2留出文档×64token、batch2、33步。每步loss、实际LR、翻码bits/bytes严格相等；最终FP32代理、全部decoder参数、梯度、Adam状态SHA相等；native state/output及训练前后整体/训练/留出MSE相等。新完整审计仅在1、32、33步，未改变更新轨迹。预分配embedding输入与旧列表拼接逐元素相等。
4. **CPU边界恢复**：真实torch两层AdamW，连续与恢复后的全部tensor及Python/NumPy/Torch RNG逐值一致。初始payload和teacher indexed shards变化可识别，错误代码/config/token/前缀/越界状态/shape/NaN/步数被拒绝。
5. **真实GPU边界恢复**：block9/10各1步，恢复后native状态、已完成块及续训块输出、loss/LR、RNG、冻结状态严格相等。注入7处过期grouped packed缓存，恢复后全部重建正确。原始checkpoint完整payload未变。该1步验证恢复链路，不证明训练翻码收益。
6. **正式入口**：真实argparse解析确认唯一Python命令、输出/Arrow位置和全部固定参数，未调用训练。

新旧block对照峰值allocated **7.6141GiB**、reserved **8.9746GiB**；真实边界恢复峰值allocated **6.5616GiB**。这不是2048token显存实测。

| 被测部分 | 旧版中位数 | 优化版中位数 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 投影与计数 | 7.180ms | 6.033ms | 约16% |
| 全码统计 | 12.934ms | 10.789ms | 约17% |

以上均2次预热、5次交错同步wall-time，原始分布保留。完整33步含准备和端点评测、扣除最终状态观测哈希耗时：旧20.815s、新9.660s。该单次先旧后新的比较存在缓存/顺序影响，仅报告观测，不保证正式全量获得2.15倍加速。正式loop同时减少了重统计调用次数。

首轮回归退出1：测试helper做人工边界扰动后只恢复state_dict，漏恢复非持久化grouped packed，导致训练前MSE不等。补齐helper缓存还原及严格断言后，复测全部通过；生产数学与比较门限未为此修改。此问题及故障注入说明，只比较persistent状态不足以确保前向一致。

## 复现与证据

远程结果根目录：`/home/shaoyuantian/program/VAELLM/.result/liftquant_recovery/engineering_20260923_01/`。

- `summary.json`：每项退出状态、证据索引、清理说明。
- `equivalence_02/results.json`：独立decoder、梯度分支、投影边界、新旧33步、组件原始计时；同目录`calibration_ids.pt`为实际输入。
- `resume_gpu/validation.json`：真实两层恢复、权重/teacher/代码/token身份、冻结及最终native状态SHA。
- `resume_cpu_03.json`、`formal_cli_parse.json`：CPU及正式参数解析。
- `validation.log`：合并后的唯一核心日志，保留首轮失败和最终成功信息。
- `reference_source/`：实际修改前源码，长期数值回归所需。

永久GPU入口为 `python -m experiments.liftquant_recovery.verify_engineering --checkpoint result/linear_output/distill_init --reference .result/liftquant_recovery/engineering_20260923_01/reference_source --ids .result/liftquant_recovery/engineering_20260923_01/equivalence_02/calibration_ids.pt --output <NEW_OUTPUT>`；边界入口为 `python -m experiments.liftquant_recovery.verify_resume_gpu --checkpoint result/linear_output/distill_init --ids <SAVED_IDS> --output <NEW_OUTPUT>`。需已激活bitvae、限定CUDA_VISIBLE_DEVICES=4并按项目规范后台执行，命令本身不构成后续资源授权。

恢复正式入口增加 `--resume <OLD_RUN/latest_boundary.pt>`，其余训练参数必须匹配，`--output`为新目录。正式全量运行仍未执行。

## 经验与清理

经验已更新到上述编码恢复文档，并补充保存/恢复经验。文档索引已用项目工具更新。

在核实后台任务结束、退出0、指标与日志归并后，删除50,750,775字节的小型测试边界、失败重复指标/token、测试源码pycache及退出/重复日志。此轮清理51,016,819字节，加早前CPU重复清理2,668字节，共 **51,019,487字节（约48.66MiB）**。结果目录保留约664KiB最小证据和回归源码。未导出完整模型；原始distill_init受保护，未生成正式实验结果。
