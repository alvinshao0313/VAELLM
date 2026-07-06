# MLP Activation-Weighted Aligned Channel Selection 实现思路

## 目标

在现有通道保护 / outlier channel selection / residual VAE 相关代码中，新增一种 **MLP 对齐保护通道选择模式**。

该模式用于 SwiGLU / GLU-style MLP，例如 Qwen、LLaMA 系列中的：

- `gate_proj`
- `up_proj`
- `down_proj`

核心目标是：

> 根据校准数据的激活统计，对 `up_proj`、`gate_proj`、`down_proj` 的权重向量做输入通道加权，然后在同一个 intermediate channel index 上融合三者得分，选择 top-k 作为保护通道。

最终选择的是一组 aligned intermediate indices：

```text
protected_indices ⊂ [0, d_ffn)
```

它们同时对应：

```text
up_proj.weight[protected_indices, :]
gate_proj.weight[protected_indices, :]
down_proj.weight[:, protected_indices]
```

不要让 `up_proj`、`gate_proj`、`down_proj` 分别独立选择保护通道。

---

## 背景与直觉

标准 SwiGLU MLP 的形式为：

```text
u = up_proj(x)
g = silu(gate_proj(x))
z = g * u
y = down_proj(z)
```

其中第 `i` 个 intermediate channel 是一条完整路径：

```text
x -> up_proj row i
x -> gate_proj row i
z_i -> down_proj column i
```

因此，选择 MLP 保护通道时，应该保护完整 intermediate path，而不是某个单独矩阵里的孤立 row / column。

---


## 算法输入

对每个 Transformer block 的 MLP，需要拿到：

```text
W_up:   up_proj.weight,   shape [d_ffn, d_model]
W_gate: gate_proj.weight, shape [d_ffn, d_model]
W_down: down_proj.weight, shape [d_model, d_ffn]
```

以及校准数据前向过程中统计得到的激活范数：

```text
a_in:  MLP 输入 hidden state 的通道 RMS/L2 范数，shape [d_model]
a_mid: down_proj 输入，即 SwiGLU intermediate activation 的通道 RMS/L2 范数，shape [d_ffn]
```

注意：

- `a_in` 用于加权 `up_proj` 和 `gate_proj` 的输入维度。
- `a_mid` 用于加权 `down_proj` 的输入维度，也就是 `down_proj.weight` 的列。
- `down_proj` 不能使用 `a_in` 加权，因为它的输入不是原始 hidden state，而是 `silu(gate_proj(x)) * up_proj(x)`。

---

## 激活统计逻辑

对每个 MLP block，校准前向时统计两个量：

```text
sum_sq_in:  shape [d_model]
sum_sq_mid: shape [d_ffn]
num_tokens: scalar
```

对于每个 batch 的 MLP 输入：

```text
X: shape [batch, seq_len, d_model]
```

reshape 为：

```text
X_flat: shape [num_tokens, d_model]
```

更新：

```text
sum_sq_in += sum(X_flat ** 2, dim=0)
```

同时计算 MLP intermediate activation：

```text
U = X_flat @ W_up.T
G = silu(X_flat @ W_gate.T)
Z = G * U
```

更新：

```text
sum_sq_mid += sum(Z ** 2, dim=0)
num_tokens += X_flat.shape[0]
```

统计结束后：

```text
a_in  = sqrt(sum_sq_in  / num_tokens + eps)
a_mid = sqrt(sum_sq_mid / num_tokens + eps)
```

实现时建议用 `float32` 做统计，避免 fp16/bf16 溢出或精度不足。

---

## 权重加权与单项得分

### 1. up_proj 行得分

`up_proj.weight` 的 shape 是：

```text
[d_ffn, d_model]
```

第 `i` 个 intermediate channel 对应第 `i` 行。

用 MLP 输入激活范数 `a_in` 对输入维度加权：

```text
weighted_up[i, j] = abs(W_up[i, j]) * a_in[j]
```

然后计算第 `i` 行的加权 L1 mean：

```text
score_up[i] = mean_j(weighted_up[i, j])
```

即：

```text
score_up = mean(abs(W_up) * a_in[None, :], dim=1)
```

### 2. gate_proj 行得分

同理：

```text
score_gate = mean(abs(W_gate) * a_in[None, :], dim=1)
```

shape：

```text
score_gate: [d_ffn]
```

### 3. down_proj 列得分

`down_proj.weight` 的 shape 是：

```text
[d_model, d_ffn]
```

第 `i` 个 intermediate channel 对应第 `i` 列。

用 `down_proj` 的输入激活范数 `a_mid` 加权列：

```text
weighted_down[r, i] = abs(W_down[r, i]) * a_mid[i]
```

然后计算第 `i` 列的加权 L1 mean：

```text
score_down[i] = mean_r(weighted_down[r, i])
```

即：

```text
score_down = mean(abs(W_down) * a_mid[None, :], dim=0)
```

shape：

```text
score_down: [d_ffn]
```

---

## 三者归一化与融合

由于 `up_proj`、`gate_proj`、`down_proj` 的权重尺度和激活尺度不同，三组 score 不能直接相加。

先分别做 mean normalization：

```text
norm_score = score / (mean(score) + eps)
```

得到：

```text
score_up_norm
score_gate_norm
score_down_norm
```

默认融合方式：

```text
score = (score_up_norm + score_gate_norm + score_down_norm) / 3
```

也可以预留权重参数：

```text
score = (
    alpha_up   * score_up_norm
  + alpha_gate * score_gate_norm
  + alpha_down * score_down_norm
) / (alpha_up + alpha_gate + alpha_down)
```

默认：

```text
alpha_up = alpha_gate = alpha_down = 1.0
```

如果后续实验发现 `down_proj` 更敏感，可以允许用户手动增大 `alpha_down`。

---

## 选择保护通道

给定保护比例：

```text
protect_ratio
```

计算：

```text
k = round(protect_ratio * d_ffn)
```

选择：

```text
protected_indices = topk(score, k, largest=True)
```

该 `protected_indices` 是 MLP intermediate channel indices。

后续应用保护时，必须同时映射到：

```text
up_proj:   rows protected_indices
gate_proj: rows protected_indices
down_proj: columns protected_indices
```

---

## 建议新增函数结构

可以新增一个独立函数，例如：

```text
select_mlp_aligned_activation_weighted_channels(
    mlp_module,
    activation_stats,
    protect_ratio,
    norm_type="l1_mean",
    fuse_weights=(1.0, 1.0, 1.0),
    eps=1e-8,
) -> protected_indices, score_detail
```

其中 `activation_stats` 至少包含：

```text
sum_sq_in
sum_sq_mid
num_tokens
```

或者直接包含：

```text
a_in
a_mid
```

返回：

```text
protected_indices: Tensor[int], shape [k]
score_detail: dict，包括 score_up/score_gate/score_down/score/fuse 后分数
```

`score_detail` 用于 debug 和实验日志保存。

---

## 与现有 outlier protection 逻辑的衔接

如果当前代码以单个 Linear module 为单位做 outlier channel 选择，需要新增 MLP group-level selection 逻辑。

建议处理方式：

1. 按 Transformer layer 找到同一层的 MLP 三件套：

```text
gate_proj, up_proj, down_proj
```

2. 对这一组 module 调用 aligned selection。

3. 得到一组共享的 `protected_indices`。

4. 应用到三个模块：

```text
up_proj protected axis: output / row
gate_proj protected axis: output / row
down_proj protected axis: input / column
```

5. 如果现有代码只支持 `input` 或 `output` 单方向保护，需要显式适配：

```text
up_proj/gate_proj: protect output channels
down_proj: protect input channels
```

不要把 `down_proj` 当成 output channel 保护。

---

## 最小伪代码

```text
for each transformer_layer:
    mlp = transformer_layer.mlp

    W_up = mlp.up_proj.weight.float()
    W_gate = mlp.gate_proj.weight.float()
    W_down = mlp.down_proj.weight.float()

    stats = activation_stats[layer_id]

    a_in = sqrt(stats.sum_sq_in / stats.num_tokens + eps)
    a_mid = sqrt(stats.sum_sq_mid / stats.num_tokens + eps)

    score_up = mean(abs(W_up) * a_in[None, :], dim=1)
    score_gate = mean(abs(W_gate) * a_in[None, :], dim=1)
    score_down = mean(abs(W_down) * a_mid[None, :], dim=0)

    score_up = score_up / (mean(score_up) + eps)
    score_gate = score_gate / (mean(score_gate) + eps)
    score_down = score_down / (mean(score_down) + eps)

    score = (score_up + score_gate + score_down) / 3

    k = round(protect_ratio * d_ffn)
    protected_indices = topk(score, k)

    protect rows protected_indices in up_proj
    protect rows protected_indices in gate_proj
    protect columns protected_indices in down_proj
```

---

## 测试与检查

实现后至少做以下检查。

### 1. shape 检查

对于每层 MLP：

```text
W_up.shape[0] == W_gate.shape[0] == W_down.shape[1]
```

否则不能做 aligned intermediate channel selection。

### 2. 激活统计检查

```text
a_in.shape == [d_model]
a_mid.shape == [d_ffn]
```

### 3. down 映射检查

确认 `down_proj` 保护的是 column/input channel，不是 row/output channel。

### 4. top-k 数量检查

```text
len(protected_indices) == round(protect_ratio * d_ffn)
```

### 5. score 日志

建议保存每层：

```text
mean(score_up), mean(score_gate), mean(score_down)
max(score), min(score)
protected_indices[:20]
```

用于确认没有 NaN、Inf 或全零。

---

## 常见错误

### 错误 1：用 MLP 输入激活加权 down_proj

错误：

```text
score_down = mean(abs(W_down) * a_in[None, :], dim=0)
```

原因：`a_in` shape 是 `[d_model]`，而 `down_proj` 的输入维度是 `[d_ffn]`。down 应该使用 SwiGLU intermediate activation norm `a_mid`。

### 错误 2：down_proj 按 row 选通道

错误：

```text
score_down = mean(abs(W_down) * something, dim=1)
```

正确：

```text
score_down = mean(abs(W_down) * a_mid[None, :], dim=0)
```

因为 intermediate channel 对应 `down_proj.weight[:, i]`。

### 错误 3：up/gate/down 分别 top-k

错误：

```text
up_indices = topk(score_up)
gate_indices = topk(score_gate)
down_indices = topk(score_down)
```

正确：

```text
score = fuse(score_up, score_gate, score_down)
protected_indices = topk(score)
```

三者必须共享同一组 intermediate indices。

### 错误 4：三者不归一化直接相加

错误：

```text
score = score_up + score_gate + score_down
```

正确：

```text
score = norm(score_up) + norm(score_gate) + norm(score_down)
```

否则尺度大的 projection 会主导结果。

---

## 推荐默认设置

```text
norm_type: l1_mean
activation_norm: rms
fuse_weights: (1.0, 1.0, 1.0)
eps: 1e-8
selection_scope: per-layer MLP
protected mapping:
    up_proj   -> output rows
    gate_proj -> output rows
    down_proj -> input columns
```

---

## 预期效果

该方法相比单独按每个 Linear 的权重幅值或激活幅值选通道，更符合 SwiGLU MLP 的结构，因为它选择的是完整 intermediate path：

```text
up row i + gate row i + down column i
```

因此更适合用于：

- MLP outlier channel protection
- residual VAE protected channel selection
- MLP block sparse / dense channel split
- SpenseGPT-style dense intermediate channel selection
