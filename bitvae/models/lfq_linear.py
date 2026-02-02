import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LFQLinear(nn.Module):
    def __init__(self, vq_idx, decoder, in_features, out_features, bias=None, transpose=False):
        super().__init__()
        # vq_idx 是 AutoEncoder quantizer 的输出的 bool 张量 (0/1)
        # 形状通常为 [N_Chunks, 1, Latent_Dim] (当 parallel_layers=1 时)
        if not isinstance(vq_idx, torch.Tensor):
            vq_idx = torch.tensor(vq_idx)

        self.register_buffer("vq_idx", vq_idx)

        # q_scale 是量化缩放因子，用于将 bool->float (+/- q_scale)
        # 1. / sqrt(latent_dim)
        self.register_buffer("q_scale", torch.tensor(1. / math.sqrt(self.vq_idx.shape[-1])))

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

        self.decoder = decoder
        # 冻结 decoder 参数，避免在后续 fine-tuning 中被更新
        for p in self.decoder.parameters():
            p.requires_grad = False

        self.in_features = in_features
        self.out_features = out_features
        self.transpose = transpose

    def forward(self, x):
        # 1. 还原 z: bool(0/1) -> float(-1/1) -> float(+/- q_scale)
        # 0 -> -1, 1 -> 1
        z = (self.vq_idx.float() - 0.5) * 2 * self.q_scale

        # 2. parallel_layers=1 时的 AutoEncoder 解码过程
        # Input: z -> [N_Chunks, 1, Latent_Dim]
        # Decoder Output: [N_Chunks, 1, Chunk_Size]

        w_recon_chunks = self.decoder(z)

        # 展平并重塑为权重矩阵
        # [N_Chunks, 1, Chunk_Size] -> [N_Chunks, Chunk_Size] -> [Total_Elements]
        w_recon = w_recon_chunks.view(-1)

        # 3. Scale & Shift (Normalization) has been fused into decoder weights!
        # Reshape to [Out_Features, In_Features]
        # 注意：这里假设 total elements 正好等于 in * out
        if self.transpose:
            # 如果训练时使用了转置 (In, Out)，则还原时先还原 (In, Out) 再转置回 (Out, In)
            w_recon = w_recon.view(self.in_features, self.out_features).t()
        else:
            w_recon = w_recon.view(self.out_features, self.in_features)

        # 执行线性变换
        # F.linear 计算 x @ w_recon.T + bias
        return F.linear(x, w_recon, self.bias)
