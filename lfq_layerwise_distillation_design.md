# LFQ/VQ 低精度模型的逐层蒸馏实现设计文档（修订版）

## 1. 目标与约束

当前模型采用 LFQ/VQ 对权重做量化。部署时不保存完整权重，只保留：

- `index`
- `decoder`

推理时通过 `index -> decoder -> 重构权重块` 的方式恢复线性层权重。

当前有三个关键前提：

1. `index + decoder` 已经通过加权重建损失优化完成。
2. 计算资源有限，无法支持端到端蒸馏，也无法支持 SFT。
3. 不希望离线保存大量中间层缓存，但可以支持 teacher 和 quantized model 以相同输入顺序进行分层推理。

因此，本方案的目标不是继续优化权重重建，而是在**不改变现有 LFQ/VQ 部署形式**的前提下，通过**逐层对齐**恢复量化后模型的函数能力。

---

## 2. 总体思路

由于 `index + decoder` 已经优化较好，后续性能下降的主要原因更可能是：

- 量化误差在网络中逐层传播
- 某些 block 的输出偏离 teacher
- residual stream 逐层漂移
- attention 输出在进入后续归一化前发生分布偏移

所以恢复阶段不再以权重重建为主，而是以**逐层函数对齐**为主。

整个训练流程采用严格的 layer-wise 方式：

1. 从第 0 层开始到最后一层依次处理。
2. 每次只训练当前层。
3. teacher 和 quantized model 用同一批输入做前向。
4. 当前层只比较该层的局部输出，不做端到端反传。
5. 当前层训练完后，进入下一层。

这样可以避免：

- 保存大规模离线缓存
- 端到端蒸馏的大显存占用
- 复杂的数据调度

---

## 3. 可训练参数设计

### 3.1 默认冻结项

默认冻结以下参数：

- 所有层的 `index`
- 非当前层的 `decoder`
- 其他 backbone 参数

原因：

1. `index` 已经被重建目标优化好，不适合再大范围改动。
2. 当前目标是修复函数偏差，而不是重新搜索离散分配。
3. 逐层训练时只开放当前层参数更稳定，也更省资源。

### 3.2 当前层可训练项

对于第 `l` 层，只开放以下参数：

- 第 `l` 层的 `decoder` 参数
- 第 `l` 层原本就存在且可融合的 bias / norm 参数（按需）

不建议额外新增独立的 per-channel scale 模块，原因是：

- 对现有架构而言，这类 scale 不能自然融合进 decoder
- 如果额外保留，会破坏最终部署形式

如果确实需要轻量修正，更推荐使用原有结构中本就存在、且训练后可以吸收的参数：

1. **bias 侧修正**
   - 若目标线性层本身带 bias，则可直接微调该 bias
   - 训练后不增加额外计算

2. **LayerNorm 权重侧修正**
   - 若某个 correction 本质上是对 LN 输出通道做缩放，则优先通过微调 LN 的 `weight` 实现
   - 训练后不增加额外计算

也就是说，本方案尽量不新增真正的运行时模块，而是只微调：

- `decoder`
- 原有 `bias`
- 原有 `LayerNorm.weight`

---

## 4. 逐层蒸馏的基本形式

记：

- $B_l^T$：teacher 的第 $l$ 个 block
- $B_l^Q$：quantized model 的第 $l$ 个 block
- $h_l^T$：teacher 在第 $l$ 层的输入 hidden
- $h_l^Q$：quantized model 在第 $l$ 层的输入 hidden
- $o_l^T$：teacher 第 $l$ 个 block 的输出
- $o_l^Q$：quantized model 第 $l$ 个 block 的输出
- $r_l^T$：teacher 在该 block 后的 residual 输出
- $r_l^Q$：quantized model 在该 block 后的 residual 输出

最关键的一点是：

- teacher 使用自己的输入 $h_l^T$
- quantized block 使用自己的真实输入 $h_l^Q$

也就是说，我们不是在比较“同一输入下的两个 block”，而是在比较：

- teacher 的真实局部行为
- quantized model 在误差已经累积后的真实局部行为

这样更符合实际推理路径。

---

## 5. Loss 设计

### 5.1 主体 loss

对于第 $l$ 层，推荐使用：

$$
L_l = \lambda_{blk} L_{blk}^{(l)} + \lambda_{res} L_{res}^{(l)} + \lambda_{anchor} L_{anchor}^{(l)}
$$

其中如果该层是 attention block，并且你希望额外约束归一化前后的幅值关系，可以再加一个可选项：

$$
L_l = \lambda_{blk} L_{blk}^{(l)} + \lambda_{res} L_{res}^{(l)} + \lambda_{anchor} L_{anchor}^{(l)} + \lambda_{norm} L_{norm}^{(l)}
$$

默认建议先从前三项开始，不必一上来就加第四项。

### 5.2 Block 输出对齐

$$
L_{blk}^{(l)} = \| o_l^Q - o_l^T \|_2^2
$$

其中：

$$
o_l^Q = B_l^Q(h_l^Q), \qquad o_l^T = B_l^T(h_l^T)
$$

这是最核心的 loss，用于直接修复当前 block 的局部函数偏差。

### 5.3 Residual 对齐

$$
L_{res}^{(l)} = \| r_l^Q - r_l^T \|_2^2
$$

这个 loss 的作用是减少 residual stream 的逐层漂移。

### 5.4 Anchor loss

由于 `decoder` 已经通过重建损失优化过，不希望逐层蒸馏时把它拉偏太多，因此加入一个很小的 anchor：

$$
L_{anchor}^{(l)} = \| \theta_l - \theta_l^{(0)} \|_2^2
$$

其中：

- $\theta_l$ 表示当前层所有可训练参数
- $\theta_l^{(0)}$ 表示蒸馏开始前的参数快照

注意这里的 anchor 只是为了防止训练不稳定，不是再次逼近 full-precision 权重。

### 5.5 可选的归一化侧对齐

如果某一层是 attention block，并且你决定开放该层后续 LayerNorm 的 `weight`，则可以加入一个轻量约束：

$$
L_{norm}^{(l)} = \| \operatorname{mean}(z_l^Q) - \operatorname{mean}(z_l^T) \|_2^2
$$

其中 $z_l^Q$ 和 $z_l^T$ 表示某个归一化相关位置的张量。

这一项是可选项，不建议在第一版实现中强制加入。

---

## 6. Attention 与 MLP 的推荐模板

### 6.1 Attention block

对 attention block，推荐先使用：

$$
L_l^{attn} = \lambda_{blk} L_{blk}^{(l)} + \lambda_{res} L_{res}^{(l)} + \lambda_{anchor} L_{anchor}^{(l)}
$$

如果观察到该层在 post-attention 附近存在明显偏移，再考虑加入：

$$
L_l^{attn} = \lambda_{blk} L_{blk}^{(l)} + \lambda_{res} L_{res}^{(l)} + \lambda_{anchor} L_{anchor}^{(l)} + \lambda_{norm} L_{norm}^{(l)}
$$

### 6.2 MLP block

对 MLP block，保持简单即可：

$$
L_l^{mlp} = \lambda_{blk} L_{blk}^{(l)} + \lambda_{res} L_{res}^{(l)} + \lambda_{anchor} L_{anchor}^{(l)}
$$

一般不需要额外归一化约束。

---

## 7. 推荐默认权重

建议先从下面这组默认值开始：

$$
\lambda_{blk} = 0.7, \qquad \lambda_{res} = 0.25, \qquad \lambda_{anchor} = 0.05
$$

如果启用归一化相关项，则可以先试：

$$
\lambda_{blk} = 0.6, \qquad \lambda_{res} = 0.25, \qquad \lambda_{norm} = 0.1, \qquad \lambda_{anchor} = 0.05
$$

对应配置可写成：

```python
loss_cfg = {
    "lambda_blk": 0.70,
    "lambda_res": 0.25,
    "lambda_anchor": 0.05,
}
```

如果开启归一化对齐：

```python
loss_cfg_attn = {
    "lambda_blk": 0.60,
    "lambda_res": 0.25,
    "lambda_norm": 0.10,
    "lambda_anchor": 0.05,
}
```

---

## 8. 训练流程

本方案不依赖离线缓存大规模中间结果，而是采用**双模型顺序前向**。

### Step 1. 准备同一批输入

对一批 calibration 数据，同时送入：

- teacher
- quantized model

两者都以 `eval()` 模式运行，但当前层可训练参数允许反传。

### Step 2. 在第 $l$ 层注册 hook

为了只拿当前层所需的中间结果，建议在当前层注册 forward hook，获取：

- 当前层输入 hidden
- 当前层输出 hidden
- residual 输出
- 如果需要，某个 norm 相关位置的张量

不需要保存所有层的所有结果。

### Step 3. teacher 前向

teacher 前向时：

- 用 `torch.no_grad()`
- 只收集当前层对应的目标张量

### Step 4. quantized model 前向

quantized model 前向时：

- 只训练当前层参数
- 通过 hook 拿到当前层的输出
- 用当前 batch 即时计算 loss
- 立刻反传和更新

### Step 5. 切到下一层

当前层训练完后：

- 冻结当前层
- 解冻下一层
- 重新注册下一层 hook
- 重复上述过程

整个流程中不需要额外的全量中间缓存文件。

---

## 9. 推荐伪代码

```python
for layer_id in range(num_layers):
    freeze_all_params(model_q)
    unfreeze_layer_decoder(model_q, layer_id)
    optionally_unfreeze_layer_bias_or_ln(model_q, layer_id)

    init_snapshot = snapshot_trainable_params(model_q, layer_id)
    optimizer = build_optimizer(get_trainable_params(model_q, layer_id))

    teacher_hook = register_layer_hooks(model_t, layer_id)
    quant_hook = register_layer_hooks(model_q, layer_id)

    for epoch in range(layer_epochs):
        for batch in calib_loader:
            with torch.no_grad():
                teacher_outputs = model_t(batch)
                teacher_states = teacher_hook.fetch()

            quant_outputs = model_q(batch)
            quant_states = quant_hook.fetch()

            loss_blk = mse(quant_states["block_out"], teacher_states["block_out"])
            loss_res = mse(quant_states["residual_out"], teacher_states["residual_out"])
            loss_anchor = l2_to_snapshot(
                get_trainable_params(model_q, layer_id),
                init_snapshot,
            )

            loss = (
                loss_cfg["lambda_blk"] * loss_blk
                + loss_cfg["lambda_res"] * loss_res
                + loss_cfg["lambda_anchor"] * loss_anchor
            )

            if use_norm_loss and "norm_tensor" in quant_states:
                loss_norm = mse(
                    quant_states["norm_tensor"].mean(dim=(0, 1)),
                    teacher_states["norm_tensor"].mean(dim=(0, 1)),
                )
                loss = loss + loss_cfg_attn["lambda_norm"] * loss_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    teacher_hook.remove()
    quant_hook.remove()
```

---

## 10. 实现建议

### 10.1 不要做离线大缓存

这版实现明确不推荐：

- 保存每条样本的所有层 hidden
- 保存庞大的 `teacher_cache.pt`
- 每层反复读取磁盘缓存

推荐做法是：

- 用同一批输入同时前向两个模型
- 只通过 hook 取当前层结果
- 当前 batch 算完立刻训练

这样逻辑更直接，也更适合用 Codex 写。

### 10.2 hook 的最小采集范围

每次只采集当前层需要的内容，例如：

- block 输出
- residual 输出
- 可选的 norm tensor

不要默认采集所有层。

### 10.3 参数融合原则

本方案下，参数设计应满足：

1. `decoder` 本来就是部署参数。
2. 若开放某层原有 `bias`，训练后它自然留在原结构中，不增加计算。若没有，增加0初始化的bias。
3. 若开放某层原有 `LayerNorm.weight`，训练后它自然留在原结构中，不增加计算。若没有，增加1初始化的参数。

因此，这里不额外引入新的运行时模块，也就不需要额外考虑复杂的 merge 逻辑。

---

## 11. 最小可行版本（MVP）

建议第一版只做下面这些：

### 可训练参数

- 当前层 `decoder`
- 不额外开放 bias
- 不额外开放 LN 参数

### loss

只使用：

$$
L_l = 0.7 L_{blk}^{(l)} + 0.25 L_{res}^{(l)} + 0.05 L_{anchor}^{(l)}
$$

### 流程

- teacher 和 quantized model 同 batch 顺序前向
- hook 只抓当前层
- 逐层训练
- 不做缓存文件
- 不做 index 更新

这版最容易实现，也最适合先验证“逐层函数对齐是否有效”。

---

## 12. 后续扩展方向

在 MVP 跑通后，可以逐步增加：

1. 对 attention 层开放原有 bias
2. 对部分层开放 LayerNorm.weight
3. 对 attention 层增加 $L_{norm}$
4. 只对误差更大的层增加训练步数
5. 做两轮 layer-wise sweep

不建议一开始就：

- 更新 index
- 引入额外 side module
- 加复杂的二阶损失
- 做端到端 token-level KD

---

## 13. 一句话总结

本方案的核心是：

> 在 `index + decoder` 已通过重建目标优化完成的前提下，不再依赖端到端蒸馏，也不保存大量中间缓存，而是通过 teacher 与 quantized model 的同批次顺序前向，逐层对齐 block 输出与 residual 输出，只更新当前层 decoder 及少量原有可融合参数，从而以低资源方式恢复量化模型性能。
