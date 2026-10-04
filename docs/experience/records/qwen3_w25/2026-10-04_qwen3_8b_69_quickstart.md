# Qwen3 8B 达到69的简版说明与最快复现路径

**从原始 BF16 模型开始完整复现：** 使用[一键复现脚本与配套文档](../../../guides/qwen3_8b_69_reproduction.md)。本页下方的快捷命令则使用已保留的压缩成品或末轮起点。

**不是只调超参数。** 达标方案沿用原框架的压缩与蒸馏算法，做了必要的计算实现修复，并增加模型组装、局部训练和独立导出工具。最终结果为八任务均分 **69.312980%**，投影权重存储 **2.456989 bit/weight**，成品无外置LoRA。

## 相对原框架究竟改了什么

这里以本轮代码起点`95c1dd9`为比较基准，不能只检出这个commit再套参数就视为完整复现。

| 类型 | 本轮做法 |
| --- | --- |
| 已有机制的参数选择 | q/k/v/gate/up主码2bit，o/down主码3bit；D32、两stage、decoder H128/无残块；双侧block32旋转；普通输入INT8保护64通道、down保护256通道。CAT、旋转、保护、decoder蒸馏、SparseBit和敏感度坐标原本已有。 |
| 实现修复 | 修正融合decoder的LayerNorm/SiLU、低精度舍入及相关梯度/参数转换行为，使训练、导出、推理对应同一设计；评测缓存显式使用BF16计算精度。这是代码修复，不能靠改学习率代替。 |
| 新增训练和组装流程 | 从不同训练分支抽取attention/MLP类别组合；使用实验入口只联合训练第一层down的decoder和码位；按保存点独立导出、重载、评测。核心损失和码位更新规则仍复用原框架。 |
| 新增结果核验 | 保存完整八任务原始结果，固定fast tokenizer、0-shot、全量题目和64/32评测batch；按实际张量统计位宽，核查冻结权重和导出结果。 |

探索期间还修过多GPU设备上下文，但最后达标训练本身是单卡，不能把这项修复单独说成精度来源。D16、H256、更深decoder、四stage、BlockLDLQ、额外head/norm等探索**没有进入该达标成品**。[计算核修复依据](../../lessons/checkpoint_lifecycle.md) · [详细方案与实验](2026-10-04_qwen3_8b_69_report.md)

## 真正有效的训练路线

1. **原HF模型逐类别压缩与恢复 → 66.5762%。** 每类压缩后训练已有压缩层decoder，尚未压缩层用临时rank12 LoRA补偿并融合；成品不留LoRA分支。
2. **不同蒸馏分支按类别组装 → 67.8045%。** 包含CE50 attention与CE10 MLP组合、后续attention恢复及第一层down局部联合更新。不是一条从头到尾只改学习率的训练。
3. **从67.8045%成品重新训练第一层down → 第1200步69.313%。** 只更新该投影decoder和两组硬码；task100、0.5CE+0.5mass-KL、prompt权重0、batch4、长度1024、decoder LR1e-5、码位Adam LR4e-5、换轮上限512、seed0，其余层冻结。

最后一轮采用**2000步cosine调度、warmup20**，每400步独立全评；1200步是最佳保存点，2000步只有68.589%。**不要把`--steps`改成1200来声称复现原轨迹**，这会改变学习率曲线。

## 最快复现选哪条路径

| 目的 | 最短路径 | 资源与时间依据 |
| --- | --- | --- |
| 验证69.313%的成绩 | 直接加载已保留的step1200成品，重跑完整八任务与位宽汇总 | 1张空闲A800 80GB；历史导出、审计、全评合计约8.7分钟，仅复评未另行测时 |
| 重做最后一轮训练 | 从已保留的67.8045%成品启动相同2000步任务，检查独立step1200 | 2张不同的空闲A800，1张训练、1张导出评测；历史到1200验收约52分钟，整轮约79分钟 |
| 从原HF完整重训 | 重做上面全部三阶段及其分支组装 | 不是最快路径；历史祖先权重部分已清理，不能假设只跑一次CAT就达标 |

时间是该服务器的历史记录，不是性能保证。这里只准备复现入口，**本次没有启动训练或评测**。

## 可直接使用的命令

使用当前服务器工作区和`bitvae`。还需要原HF目录`/home/star/models/Qwen/Qwen3-8B`、本地评测数据缓存；训练另需任务JSONL。下面两条路径二选一，GPU编号仅作示例，执行前替换为实际空闲卡。

```bash
cd /home/star/VAELLM
source /home/star/miniconda3/etc/profile.d/conda.sh
conda activate bitvae
which python
python -V
```

**最快验证成绩：**

```bash
RUN_DIR="result/qwen3_8b_w25_20260930/reproduce_eval_$(date +%Y%m%d_%H%M%S)"
(
  set -o noclobber
  CUDA_VISIBLE_DEVICES=0 nohup bash scripts/reproduce_qwen3_w25_69_eval.sh "$RUN_DIR" > "$RUN_DIR.log" 2>&1 < /dev/null
) &
```

[复评脚本](../../../../scripts/reproduce_qwen3_w25_69_eval.sh)依次执行MMLU batch64、其余七任务batch32，再生成`$RUN_DIR/acceptance.json`；日志为`$RUN_DIR.log`，退出码为`$RUN_DIR/run.exit`。无样本上限。汇总使用原指标和位宽口径，不靠手工四舍五入判断。

**最快重做训练：**

```bash
RUN_DIR="result/qwen3_8b_w25_20260930/reproduce_train_$(date +%Y%m%d_%H%M%S)"
(
  set -o noclobber
  CUDA_VISIBLE_DEVICES=0 nohup bash scripts/reproduce_qwen3_w25_69_train.sh "$RUN_DIR" 1 > "$RUN_DIR.log" 2>&1 < /dev/null
) &
```

[训练脚本](../../../../scripts/reproduce_qwen3_w25_69_train.sh)已把原始实际命令写全；第一个参数是全新输出目录，第二个是独立导出评测卡号。示例为GPU0训练、GPU1导出。结果看`$RUN_DIR/step_1200/acceptance.json`，该点导出不会随原生训练断点轮转删除。脚本保持完整2000步预算；不能把旧分数当作新运行已成功的证据。

两个脚本都会拒绝复用已存在的输出目录。脚本只负责运行既有入口，没有引入新的训练算法、调度器或自动重试。

## 必须保留的输入和复现边界

- **直接复评模型：** [step1200 final_model](../../../../result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/final_model)，UUID `f927ffda-dd5b-4d00-b63a-5e839a182d70`。
- **最后一轮训练起点：** [67.8045% final_model](../../../../result/qwen3_8b_w25_20260930/assembled_ce50_chatml_joint_bits00_s2000_attn/final_model)，UUID `1facb463-1832-4f10-8165-0a4b47ef7cf5`。两份权重本次核对均存在。
- **真实配置与版本：** [原始launch](../../../../result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/launch.json)。本次核对其固定的66份源码SHA256，均与当前文件一致；复现脚本由其中实际argv和[原评测命令](../../../../result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/step_export.json)整理，仅调整输出位置及由调用者指定GPU。
- **正确解读结果：** 2.457bit只针对252个投影，含embedding/head等的完整模型张量约4.31GiB；69.313%是八任务均分，MMLU为61.024%。原训练未启用完全确定性模式，新训练的精确末位及最佳步数不作保证。

本次只做脚本语法、原始命令参数对应和文档链接检查，复用已有运行证据；未重新跑GPU。需要展开原理、位宽计算或成功与失败对照时，查看[详细汇报](2026-10-04_qwen3_8b_69_report.md)。
