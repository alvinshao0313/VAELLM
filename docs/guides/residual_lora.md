# 可选残差 LoRA

每个 Transformer block 的 attention、MLP 两处 skip 分别使用独立的低秩映射，通过唯一参数 `--residual_lora_mode` 选择 `none`、`additive` 或 `replace`，默认 `none`。令 `D(z)=(alpha/rank)*B(A(dropout(z)))`，启用 `additive` 时的计算为：

```text
h = x + D_attn(x) + Attention(Norm(x))
y = h + D_mlp(h) + MLP(Norm(h))
```

`additive` 保留原始 skip，并添加低秩增量。初始化 `B=0`，所以新装模块时保持原基座输出。显式选择 `--residual_lora_mode replace` 时，使用此前的纯替换计算：

```text
h = D_attn(x) + Attention(Norm(x))
y = D_mlp(h) + MLP(Norm(h))
```

`replace` 不保留原始 skip；初始化使用正交行矩阵 `A` 和 `B=A.T/(alpha/rank)`，初始 skip 是 rank-r 正交投影。两种模式都支持当前项目环境的 Qwen3、Llama decoder。

`none` 不新增或训练残差模块，新模型沿用原计算。加载已包含残差模块的 checkpoint 时，`none` 仅冻结已有残差参数，仍保留 checkpoint 的模式和推理计算。

## 配置

| 参数 | 默认值 |
| --- | --- |
| `--residual_lora_mode` | `none`；可选 `additive`、`replace` |
| `--residual_lora_rank` | `8`，要求 `1 <= rank <= 8` |
| `--residual_lora_alpha` | `16` |
| `--residual_lora_dropout` | `0` |
| `--residual_lora_lr` | 未设置时跟随 `learning_rate` |

残差参数使用独立 optimizer 参数组，可与 projection LoRA、decoder 等训练参数同时启用。若希望残差参数保持 FP32，将 `residual_lora` 加入 `--distill_fp32_components`，并保留实验原来所需的组件。

两个入口脚本默认 `none`，末尾的 `"$@"` 允许覆盖配置：

```bash
# E2E：保留该脚本 dp 配置已有的 FP32 组件。
bash compressed_e2e_fintuning/scripts/e2e_decoder.sh \
  --resume_from_checkpoint "" \
  --residual_lora_mode additive \
  --residual_lora_lr 1e-4 \
  --distill_fp32_components lora,lm_head,norm,residual_lora

# CAT：保留该脚本已有的 lora FP32 组件。
bash scripts/catlora_simple2.sh \
  --residual_lora_mode additive \
  --distill_fp32_components lora,residual_lora
```

E2E 脚本当前包含旧的 `--resume_from_checkpoint` 行。首次启用残差模块时，按上例用空字符串覆盖该参数，或去掉该行，从原 `student_checkpoint_dir` 开始新训练；旧 step checkpoint 没有这套训练拓扑，不能据此进行 exact-step resume。已启用残差的 step checkpoint 续训时，需保持残差配置（包括 `residual_lora_mode`）不变，不能在 exact-step resume 时切换两种计算方式。

CAT 在每个类别压缩后的 `after_category` 模型恢复蒸馏阶段训练残差模块，跨类别保留其权重，最后一个类别也可继续训练。它不在仅使用权重重建损失的 VAE 训练阶段优化残差模块。

两处 skip 各有一个 `d -> r -> d` 映射，额外参数量为 `4 * L * d * r`。对于 `L=36, d=4096, r=8`，新增 **4,718,592** 个参数。保存和加载 checkpoint 时保留残差模块及其权重。

当前实现与 CPU 测试验证的是计算、梯度及训练/保存链路；下游精度是否提升，需要使用相同数据和评估口径进行实验确认。

历史验证结果和实现经验见 [残差 LoRA 验证总结](../experience/records/docs/exp_results/residual_lora_validation.md)。
