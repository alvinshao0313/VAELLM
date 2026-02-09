import argparse
import math
import torch
import torch.utils.checkpoint as checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from litebsq.bsq import BSQ
from litebsq.misc import ptdtype


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class ParallelLinear(nn.Module):
    """
    使用分组卷积实现的并行线性层。
    输入形如 [Batch, num_models, in_features]
    等价于 num_models 个独立的 Linear(in_features, out_features)
    """

    def __init__(self, in_features, out_features, num_models=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_models = num_models

        if num_models == 1:
            self.linear = nn.Linear(in_features, out_features)
        else:
            # groups=num_models 保证了每组通道独立运算，互不干扰
            self.conv = nn.Conv1d(
                in_channels=in_features * num_models,
                out_channels=out_features * num_models,
                kernel_size=1,
                groups=num_models,
                bias=True
            )

    def forward(self, x):
        # x: [Batch, num_models, in_features]
        if self.num_models == 1:
            # 如果输入是 [B, 1, C] 且我们使用的是 Linear
            if x.dim() == 3 and x.shape[1] == 1:
                return self.linear(x.squeeze(1)).unsqueeze(1)
            return self.linear(x)

        B = x.shape[0]
        # Reshape to [Batch, Channels, Length=1] for Conv1d
        # Channel 排列必须是: M1_F1, M1_F2... M2_F1, M2_F2...
        # x 必须是连续的
        x = x.reshape(B, self.num_models * self.in_features, 1)
        out = self.conv(x)
        out = out.reshape(B, self.num_models, self.out_features)
        return out

    def get_sub_linear(self, model_idx: int):
        """
        提取特定模型索引的权重并封装为标准的 nn.Linear 返回。
        注意：返回的是权重的副本。
        """
        if model_idx >= self.num_models or model_idx < 0:
            raise ValueError(f"Index {model_idx} out of range [0, {self.num_models-1}]")

        if self.num_models == 1:
            # 单模型情况，为了一致性返回副本
            layer = nn.Linear(self.in_features, self.out_features)
            layer.load_state_dict(self.linear.state_dict())
            return layer

        # 多模型 Parallel 模式
        start = model_idx * self.out_features
        end = (model_idx + 1) * self.out_features

        layer = nn.Linear(self.in_features, self.out_features, bias=(self.conv.bias is not None))

        with torch.no_grad():
            # Conv Weight: [N*Out, In, 1] -> Slice -> [Out, In, 1] -> Squeeze -> [Out, In]
            # Linear Weight: [Out, In]
            layer.weight.copy_(self.conv.weight[start:end, :, 0])

            if self.conv.bias is not None:
                layer.bias.copy_(self.conv.bias[start:end])

        return layer

    def fuse_q_scale(self, q_scale: float):
        # Apply q_scale fusion to weights/biases in-place.
        # For Linear: w <- w * (2*q_scale), b <- b + (-q_scale * sum_j w_ij)
        if self.num_models == 1:
            weight = self.linear.weight.data
            bias_delta = -q_scale * weight.sum(dim=1)
            weight.mul_(q_scale * 2)
            if self.linear.bias is not None:
                self.linear.bias.data.add_(bias_delta)
            else:
                self.linear.bias = nn.Parameter(bias_delta)
            return

        # Conv1d grouped case: weight shape [num_models*out, in, 1]
        weight = self.conv.weight.data  # [M*Out, In, 1]
        bias_delta = -q_scale * weight[:, :, 0].sum(dim=1)
        weight.mul_(q_scale * 2)
        if self.conv.bias is not None:
            self.conv.bias.data.add_(bias_delta)
        else:
            self.conv.bias = nn.Parameter(bias_delta)


class Normalize(nn.Module):
    def __init__(self, in_channels, norm_type='group', num_models=1):
        super().__init__()
        self.num_models = num_models
        self.in_channels = in_channels  # Per model channels

        assert norm_type in ['group', 'batch', 'layer', 'no']

        if norm_type == 'group':
            # 仿照 d_vae 的逻辑，优先尝试 16/8 分组 (Per Model)
            if in_channels >= 16 and in_channels % 16 == 0:
                groups_per_model = 16
            elif in_channels >= 8 and in_channels % 8 == 0:
                groups_per_model = 8
            else:
                groups_per_model = 4
                if in_channels < groups_per_model or in_channels % groups_per_model != 0:
                    groups_per_model = 1

            self.total_groups = groups_per_model * num_models
            self.total_channels = in_channels * num_models

            self.norm = nn.GroupNorm(num_groups=self.total_groups,
                                     num_channels=self.total_channels, eps=1e-6, affine=True)

        elif norm_type == 'batch':
            # 使用适合 1D 的 BatchNorm
            # 注意: BN 统计整个 Batch 的均值方差。
            # 这里 num_features = num_models * in_channels，所以统计的是 "Batch中对应Model的对应Channel" 的均值
            # 这是符合预期的 (Batch Norm independently per channel)
            self.norm = nn.BatchNorm1d(in_channels * num_models)

        elif norm_type == 'layer':
            # LayerNorm is applied over the last dimension.
            # Tensor shape is [Batch, NumModels, InChannels]
            # LayerNorm(InChannels) perfectly works on the last dim logic.
            # nn.LayerNorm(normalized_shape)
            self.norm = nn.LayerNorm(in_channels)

        elif norm_type == 'no':
            self.norm = nn.Identity()

        self.norm_type = norm_type

    def forward(self, x):
        # x: [Batch, NumModels, InChannels]
        if self.norm_type == 'no' or self.norm_type == 'layer':
            return self.norm(x)

        B = x.shape[0]
        if self.norm_type == 'group':
            # GroupNorm 需要 [B, C] 或者 [B, C, L]
            # 我们把他展平成 [B, NumModels*InChannels] (当作 1D 数据处理，虽然 GroupNorm 本质不关心空间维度)
            # 或者为了更准确模拟，视为 [B, TotalChannels]
            x_reshaped = x.view(B, self.total_channels)
            out = self.norm(x_reshaped)
            return out.view(B, self.num_models, self.in_channels)

        if self.norm_type == 'batch':
            # BatchNorm1d 需要 [B, C, L] 或者 [B, C]
            # 这里我们当作 [B, TotalChannels, 1] 或者直接 [B, TotalChannels]
            x_reshaped = x.view(B, self.in_channels * self.num_models)
            out = self.norm(x_reshaped)
            return out.view(B, self.num_models, self.in_channels)

    def get_sub_norm(self, model_idx: int):
        """
        提取特定模型索引的 Norm 参数并封装为新的 Normalize 返回。
        """
        if model_idx >= self.num_models or model_idx < 0:
            raise ValueError(f"Index {model_idx} out of range [0, {self.num_models-1}]")

        new_norm = Normalize(self.in_channels, self.norm_type, num_models=1)

        if self.norm_type == 'no':
            return new_norm

        elif self.norm_type == 'layer':
            # LayerNorm 参数在 parallel 模式下通常是共享的 (如果实现是 nn.LayerNorm(in_len))
            # 或者如果是 nn.LayerNorm(total_len) 则需要切分。
            # 查看 __init__，使用的是 nn.LayerNorm(in_channels)，这意味着所有模型共享同一组参数。
            # 直接加载即可。
            new_norm.norm.load_state_dict(self.norm.state_dict())
            return new_norm

        elif self.norm_type in ['group', 'batch']:
            # GroupNorm 和 BatchNorm 的参数是拼接的 [N*C]
            # 我们需要切片提取
            start = model_idx * self.in_channels
            end = (model_idx + 1) * self.in_channels

            with torch.no_grad():
                if self.norm.weight is not None:
                    new_norm.norm.weight.copy_(self.norm.weight[start:end])
                if self.norm.bias is not None:
                    new_norm.norm.bias.copy_(self.norm.bias[start:end])

                if self.norm_type == 'batch':
                    new_norm.norm.running_mean.copy_(self.norm.running_mean[start:end])
                    new_norm.norm.running_var.copy_(self.norm.running_var[start:end])
                    new_norm.norm.num_batches_tracked.copy_(self.norm.num_batches_tracked)

            return new_norm

        return new_norm


class ResnetBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type='group', num_models=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.num_models = num_models

        # 定义 1D 适用的 Normalize
        self.norm1 = Normalize(self.in_channels, norm_type, num_models=num_models)
        self.linear1 = ParallelLinear(self.in_channels, self.out_channels, num_models=num_models)

        # self.norm2 = Normalize(self.out_channels, norm_type, num_models=num_models)
        # self.linear2 = ParallelLinear(self.out_channels, self.out_channels, num_models=num_models)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = ParallelLinear(self.in_channels, self.out_channels, num_models=num_models)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        # x: [B, N, C]
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.linear1(h)

        # h = self.norm2(h)
        # h = swish(h)
        # h = self.linear2(h)

        x = self.nin_shortcut(x)

        return x + h


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,  # equal to codebook_dim (chunk size)
        hidden_dim: int,  # equal to base_ch
        num_res_blocks: int,
        out_dim: int,  # equal to log2(codebook_size)
        norm_type='group',
        use_checkpoint=False,
        num_models=1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks
        self.out_dim = out_dim
        self.use_checkpoint = use_checkpoint
        self.num_models = num_models

        # Input projection
        self.linear_in = ParallelLinear(in_dim, hidden_dim, num_models=num_models)

        # Blocks
        self.blocks = nn.ModuleList()
        curr_dim = hidden_dim

        for _ in range(num_res_blocks):
            self.blocks.append(ResnetBlock1D(in_channels=curr_dim, out_channels=curr_dim,
                               norm_type=norm_type, num_models=num_models))

        # End
        self.norm_out = Normalize(curr_dim, norm_type, num_models=num_models)
        self.linear_out = ParallelLinear(curr_dim, out_dim, num_models=num_models)

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

    def _forward(self, x):
        # x: (B, num_models, in_dim)
        h = self.linear_in(x)

        for block in self.blocks:
            h = block(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.linear_out(h)

        return h


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        norm_type='group',
        decoder_type='linear',
        use_checkpoint=False,
        num_models=1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.use_checkpoint = use_checkpoint
        self.decoder_type = decoder_type
        self.num_models = num_models
        self.out_dim = out_dim
        self._q_scale_fused = False

        if self.decoder_type == 'linear':
            self.linear = ParallelLinear(in_dim, out_dim, num_models=num_models)
        elif self.decoder_type == 'symmetric':
            # 1. Input Projection
            self.linear_in = ParallelLinear(in_dim, hidden_dim, num_models=num_models)

            # 2. Residual Blocks
            self.blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                self.blocks.append(ResnetBlock1D(in_channels=hidden_dim, out_channels=hidden_dim,
                                   norm_type=norm_type, num_models=num_models))

            # 3. Output Projection
            self.norm_out = Normalize(hidden_dim, norm_type, num_models=num_models)
            self.linear_out = ParallelLinear(hidden_dim, out_dim, num_models=num_models)
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

    def _forward(self, x):
        # x: [Batch, num_models, in_dim]
        if self.decoder_type == 'linear':
            return self.linear(x)
        else:
            h = self.linear_in(x)
            for block in self.blocks:
                h = block(h)
            h = self.norm_out(h)
            h = swish(h)
            return self.linear_out(h)

    @torch.no_grad()
    def _fuse_q_scale(self):
        if self._q_scale_fused:
            return
        q_scale = 1. / math.sqrt(self.in_dim)
        if self.decoder_type == 'linear':
            self.linear.fuse_q_scale(q_scale)
        elif self.decoder_type == 'symmetric':
            self.linear_in.fuse_q_scale(q_scale)
        self._q_scale_fused = True

    def get_sub_decoder(self, model_idx: int):
        """
        Creates a new Decoder instance for a single model (num_models=1),
        initialized with the weights from this multi-model decoder at model_idx.
        """
        if model_idx >= self.num_models or model_idx < 0:
            raise ValueError(f"Index {model_idx} out of range [0, {self.num_models-1}]")

        # Create a new Decoder instance with num_models=1.
        # NOTE: keep attributes consistent with current Decoder/ResnetBlock1D implementation.
        if self.decoder_type == "linear":
            in_dim = self.in_dim
            hidden_dim = 128
            num_res_blocks = 2
            norm_type = "group"
        else:
            in_dim = self.in_dim
            hidden_dim = self.linear_in.out_features
            num_res_blocks = len(self.blocks)
            norm_type = self.norm_out.norm_type

        new_decoder = Decoder(
            in_dim=in_dim,
            out_dim=self.out_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            norm_type=norm_type,
            decoder_type=self.decoder_type,
            use_checkpoint=self.use_checkpoint,
            num_models=1,
        )

        with torch.no_grad():
            if self.decoder_type == 'linear':
                # linear: ParallelLinear
                new_decoder.linear = self.linear.get_sub_linear(model_idx)
            elif self.decoder_type == 'symmetric':
                # 1. Input Projection
                new_decoder.linear_in = self.linear_in.get_sub_linear(model_idx)

                # 2. ResBlocks
                for i, block in enumerate(self.blocks):
                    # block is ResnetBlock1D (Parallel)
                    # new_decoder.blocks[i] is ResnetBlock1D (Single)
                    src_block = block
                    tgt_block = new_decoder.blocks[i]

                    # 逐层提取
                    tgt_block.norm1 = src_block.norm1.get_sub_norm(model_idx)
                    tgt_block.linear1 = src_block.linear1.get_sub_linear(model_idx)

                    if isinstance(src_block.nin_shortcut, ParallelLinear):
                        tgt_block.nin_shortcut = src_block.nin_shortcut.get_sub_linear(model_idx)
                    else:
                        tgt_block.nin_shortcut = nn.Identity()

                # 3. Output
                new_decoder.norm_out = self.norm_out.get_sub_norm(model_idx)
                new_decoder.linear_out = self.linear_out.get_sub_linear(model_idx)

        return new_decoder


class AutoEncoder(nn.Module):
    def __init__(self, args, num_models=1):
        super().__init__()
        self.args = args
        self.num_models = num_models

        # 1. 维度计算
        # codebook_dim 在此处被解释为 输入的切块大小 (Chunk Size)
        self.chunk_size = args.codebook_dim

        # out_channels (Latent Dim) = codebook_bits
        # 例如 codebook_bits=16
        self.latent_dim = args.codebook_bits

        # 2. 初始化 Encoder (并行版)
        self.encoder = Encoder(
            in_dim=self.chunk_size,
            hidden_dim=args.base_ch,
            num_res_blocks=args.num_res_blocks,
            out_dim=self.latent_dim,
            norm_type=getattr(args, 'norm_type', 'group'),
            use_checkpoint=args.use_checkpoint,
            num_models=num_models
        )

        # 3. 初始化 Decoder (并行版)
        self.decoder = Decoder(
            in_dim=self.latent_dim,
            out_dim=self.chunk_size,
            hidden_dim=args.base_ch,
            num_res_blocks=args.num_res_blocks,
            norm_type=getattr(args, 'norm_type', 'group'),
            decoder_type=getattr(args, 'decoder_type', 'linear'),
            use_checkpoint=args.use_checkpoint,
            num_models=num_models
        )

        # Loss weights
        self.recon_loss_type = args.recon_loss_type
        self.l1_weight = args.l1_weight
        self.lfq_weight = args.lfq_weight
        self.commitment_loss_weight = args.commitment_loss_weight

        # 4. 初始化 Quantizer (BSQ)
        # BSQ 天生支持 [Batch, Sequence, Dim] 这样的 3D 输入
        # 我们在这里把 num_models 当作 Sequence 维度处理
        self.quantizer_type = getattr(args, 'quantizer_type', 'BSQ')
        if self.quantizer_type == 'BSQ':
            self.quantizer = BSQ(
                dim=self.latent_dim,
                codebook_scale=1,  # 默认
                entropy_loss_weight=getattr(args, 'entropy_loss_weight', 0.1),
                commitment_loss_weight=getattr(args, 'commitment_loss_weight', 0.25),
                diversity_gamma=getattr(args, 'diversity_gamma', 1.0),
                has_projections=False,
                spherical=True,
                new_quant=getattr(args, 'new_quant', False),
                # 其他参数传递...
                gamma0=getattr(args, 'gamma0', 1.0),
                gamma=getattr(args, 'gamma', 1.0),
                zeta=getattr(args, 'zeta', 1.0),
                inv_temperature=getattr(args, 'inv_temperature', 100.0),
            )
        elif self.quantizer_type == 'Identity':
            self.quantizer = nn.Identity()
        else:
            raise NotImplementedError(f"{self.quantizer_type} not supported, use BSQ")

        weight_dtype = getattr(args, "vae_weight_dtype", "fp32").lower()
        if weight_dtype == "bf16":
            self.params_dtype = torch.bfloat16
            self.to(dtype=torch.bfloat16)
        elif weight_dtype == "fp16":
            self.params_dtype = torch.float16
            self.to(dtype=torch.float16)
        else:
            self.params_dtype = torch.float32

    def _parse_quantizer_output(self, quant_ret, device):
        if self.quantizer_type == 'Identity':
            return quant_ret, None, None, torch.tensor(0.0, device=device)
        if isinstance(quant_ret, tuple):
            z, bit_indices, aux_loss = quant_ret
            return z, bit_indices, aux_loss
        return quant_ret.quantized, quant_ret.bit_indices, quant_ret.entropy_aux_loss

    def _compute_recon_loss(self, x_recon, x):
        if self.recon_loss_type == 'l1':
            return F.l1_loss(x_recon, x)
        if self.recon_loss_type == 'huber':
            return F.huber_loss(x_recon, x, reduction='mean', delta=1.0)
        if self.recon_loss_type == 'relative_l1':
            return (x_recon - x).abs().sum() / (x.abs().sum() + 1e-10)
        if self.recon_loss_type == 'top_k_mse':
            k = max(1, int(0.1 * x.shape[-1]))
            errors = (x_recon - x).pow(2)
            topk_errors, _ = torch.topk(errors, k, dim=-1)
            return topk_errors.sum()
        if self.recon_loss_type == 'mse':
            return F.mse_loss(x_recon, x)
        return torch.tensor(0.0, device=x.device)

    def forward(self, x, is_train=True):
        """
        x: 输入张量, 形状为 [Batch, num_models, chunk_size]
        """
        # Run Encoder/Decoder with autocast
        enc_dtype = ptdtype[self.args.vae_autocast_dtype]
        with torch.amp.autocast("cuda", dtype=enc_dtype):
            h = self.encoder(x)  # [Batch, num_models, latent_dim]
            # h = h.to(dtype=torch.float32)

            quant_ret = self.quantizer(h)  # BSQ expects [B, N, C] shape.
            z, bit_indices, aux_loss = self._parse_quantizer_output(quant_ret, x.device)

            z = z.to(self.params_dtype)
            x_recon = self.decoder(z)  # [Batch, num_models, chunk_size]

        if not is_train:
            return x_recon, bit_indices

        recon_loss = self._compute_recon_loss(x_recon, x)

        target_recon_loss = recon_loss * self.l1_weight * self.num_models

        loss_dict = {
            "train/recon_loss": target_recon_loss,
            "train/commitment_loss": aux_loss * self.lfq_weight * self.num_models,
        }

        return (x_recon, x.detach(), loss_dict)  # 返回 (重建值, 输入值, 重建值作为显示, loss字典)

    @staticmethod
    def add_model_specific_args(parent_parser):
        from train_utils.train_args import add_model_specific_args
        return add_model_specific_args(parent_parser)


class MultiLayerVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_models = args.parallel_layers

        # 使用单个并行化的 AutoEncoder 替代原本的 ModuleList 循环
        # 这个 AutoEncoder 内部会处理 [Batch, num_models, chunk_size] 的数据流
        self.model = AutoEncoder(args, num_models=self.num_models)

    def forward(self, x, global_step=None, is_train=True):
        """
        x: [Batch, num_models, chunk_size]
        """
        # 直接调用并行化的模型
        # returns: (x_recon, x_orig, x_recon_disp, loss_dict) or (x_recon, vq_output)
        if is_train:
            x_recon, _, loss_dict = self.model(x, is_train=True)

            recon_loss = loss_dict["train/recon_loss"]
            commit_loss = loss_dict["train/commitment_loss"]
            loss_dict["loss"] = recon_loss + commit_loss
            return x_recon, loss_dict

        else:
            x_recon, vq_output = self.model(x, is_train=False)
            return x_recon, vq_output
