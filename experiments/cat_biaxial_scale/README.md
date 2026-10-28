# CAT 旋转后双向归一化原型

隔离实现，公共 CAT 代码不变。算法、六臂配置、数值边界、验证和运行结果统一见[实验记录](../../docs/experience/records/qwen3_w25/2026-10-22_cat_biaxial.md)。本模块输出局部 packed endpoint，尚未集成完整模型 v6。

在项目根目录、已激活 `bitvae` 的 shell 中，通过 tmux 执行。单个训练任务：

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m experiments.cat_biaxial_scale.run \
  --input-file result/qwen_cat_biaxial_20261022/prepared/inputs.pt \
  --output-dir result/qwen_cat_biaxial_20261022/formal/biaxial_int8/gate_proj \
  --category gate_proj --normalization biaxial --protection int8 --steps 10000
```

- `--category`：`gate_proj`、`up_proj`、`down_proj`，每次并行训练 L0/L18/L35 的三个完整矩阵。
- `--normalization`：`current`、`biaxial`（固定尺度）、`learned`（共同尺度可学习）。
- `--protection`：`int8` 或 `none`；`--steps` 是每个残差 stage 的更新数。
- 输出目录必须新建；不会自动覆盖、降配置或重试。

已安排的两条有限流程为 `run_int8.sh`、`run_none.sh`，GPU 由调用者指定，实际分配见实验记录；各执行九个训练任务，每完成一个归一化条件即做完整局部读出。脚本不负责环境激活或后台调度。

`prepare.py prepare` 提取原始权重/保护索引并固定诊断文本；`prepare.py capture` 采集原 BF16 模型输入。文本不进入权重拟合。`observe.py --prepared-dir ... --arm-dir ... --output-file ...` 从每臂的真实 packed 端点计算权重、单 Linear 和完整 MLP 误差。

每个训练任务生成 `report.json`、`run.exit` 和三个 `endpoints/<target_with_underscores>.pt`。报告保存配置、源码 SHA、实际张量存储账本和重载结果。读取接口为：

```python
from experiments.cat_biaxial_scale.runtime import load_endpoint
weight = load_endpoint(path, device='cuda:0').decode_weight()
```

`python -m experiments.cat_biaxial_scale.checks` 检查 CPU 代数、转置索引、损失/梯度、动态输入和尺度参数化。真实 GPU 验证复用训练入口，保持 M3/B8192/两 stage，仅缩短更新步数；短测通过不代表压缩效果。
