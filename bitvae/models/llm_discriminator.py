import torch
import torch.nn as nn
from .llm_vae import ParallelLinear


class MLPDiscriminator(nn.Module):
    """
    针对 1D 向量的 MLP 判别器，支持并行多模型处理。
    利用 ParallelLinear 实现 N 个独立的判别器。
    """

    def __init__(self, input_dim, hidden_dim=256, num_models=1):
        super().__init__()
        self.num_models = num_models

        self.main = nn.Sequential(
            ParallelLinear(input_dim, hidden_dim, num_models=num_models),
            nn.LeakyReLU(0.2),
            ParallelLinear(hidden_dim, hidden_dim, num_models=num_models),
            nn.LeakyReLU(0.2),
            ParallelLinear(hidden_dim, 1, num_models=num_models)
        )

    def forward(self, x):
        # x: [Batch, num_models, input_dim]
        # ParallelLinear 会自动处理形状并返回 [Batch, num_models, out_dim]
        return self.main(x)
