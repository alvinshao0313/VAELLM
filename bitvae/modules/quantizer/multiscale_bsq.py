"""
Binary Spherical Quantization
Proposed in https://arxiv.org/abs/2406.07548

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.
"""

import random
from math import log2, ceil
from functools import partial, cache
from collections import namedtuple
from contextlib import nullcontext

import torch.distributed as dist
from torch.distributed import nn as dist_nn

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module
from torch.amp import autocast

from einops import rearrange, reduce, pack, unpack

from einx import get_at

from .dynamic_resolution import predefined_HW_Scales_dynamic

# constants

Return = namedtuple('Return', ['quantized', 'indices', 'bit_indices', 'entropy_aux_loss'])

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])

# distributed helpers


@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist_nn.all_reduce(t)
    t = t / dist.get_world_size()
    return t

# helper functions


def exists(v):
    return v is not None


def identity(t):
    return t


def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def round_up_multiple(num, mult):
    return ceil(num / mult) * mult


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def l2norm(t):
    return F.normalize(t, dim=-1)

# entropy


def log(t, eps=1e-5):
    return t.clamp(min=eps).log()


def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

# cosine sim linear


class CosineSimLinear(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scale=1.
    ):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=0)
        return (x @ w) * self.scale


def get_latent2scale_schedule(T: int, H: int, W: int, mode="original"):
    assert mode in ["original", "dynamic", "dense", "same1", "same2", "same3", "half", "dense_f8"]
    predefined_HW_Scales = {
        # 256 * 256
        (32, 32): [(1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (9, 9), (13, 13), (18, 18), (24, 24), (32, 32)],
        (16, 16): [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)],
        # 1024x1024
        (64, 64): [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (9, 9), (12, 12), (16, 16), (21, 21), (27, 27), (36, 36), (48, 48), (64, 64)],

        (36, 64): [(1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (9, 12), (13, 16), (18, 24), (24, 32), (32, 48), (36, 64)],
    }
    if mode == "dynamic":
        predefined_HW_Scales.update(predefined_HW_Scales_dynamic)
    elif mode == "dense":
        predefined_HW_Scales[(16, 16)] = [(x, x) for x in range(1, 16 + 1)]
        predefined_HW_Scales[(32, 32)] = predefined_HW_Scales[(16, 16)] + [(20, 20), (24, 24), (28, 28), (32, 32)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(40, 40), (48, 48), (56, 56), (64, 64)]
    elif mode == "dense_f8":
        # predefined_HW_Scales[(16, 16)] = [(x, x) for x in range(1, 16+1)]
        predefined_HW_Scales[(32, 32)] = [(x, x) for x in range(1, 16 + 1)] + [(20, 20), (24, 24), (28, 28), (32, 32)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(40, 40), (48, 48), (56, 56), (64, 64)]
        predefined_HW_Scales[(128, 128)] = predefined_HW_Scales[(64, 64)] + [(80, 80), (96, 96), (112, 112), (128, 128)]
    elif mode.startswith("same"):
        num_quant = int(mode[len("same"):])
        predefined_HW_Scales[(16, 16)] = [(16, 16) for _ in range(num_quant)]
        predefined_HW_Scales[(32, 32)] = [(32, 32) for _ in range(num_quant)]
        predefined_HW_Scales[(64, 64)] = [(64, 64) for _ in range(num_quant)]
    elif mode == "half":
        predefined_HW_Scales[(32, 32)] = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
                                          (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)]
        predefined_HW_Scales[(64, 64)] = [(1, 1), (2, 2), (4, 4), (6, 6), (8, 8), (12, 12), (16, 16)]

    predefined_T_Scales = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 17, 17, 17, 17]
    patch_THW_shape_per_scale = predefined_HW_Scales[(H, W)]
    if len(predefined_T_Scales) < len(patch_THW_shape_per_scale):
        # print("warning: the length of predefined_T_Scales is less than the length of patch_THW_shape_per_scale!")
        predefined_T_Scales += [predefined_T_Scales[-1]] * (len(patch_THW_shape_per_scale) - len(predefined_T_Scales))
    patch_THW_shape_per_scale = [(min(T, t), h, w) for (h, w), t in zip(
        patch_THW_shape_per_scale, predefined_T_Scales[:len(patch_THW_shape_per_scale)])]
    return patch_THW_shape_per_scale


class MultiScaleBSQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        dim,
        soft_clamp_input_value=None,
        aux_loss=False,  # intermediate auxiliary loss
        use_decay_factor=False,
        use_stochastic_depth=False,
        drop_rate=0.,
        schedule_mode="original",  # ["original", "dynamic", "dense"]
        keep_first_quant=False,
        keep_last_quant=False,
        remove_residual_detach=False,
        random_flip=False,
        flip_prob=0.5,
        flip_mode="stochastic",  # "stochastic", "deterministic"
        max_flip_lvl=1,
        random_flip_1lvl=False,  # random flip one level each time
        flip_lvl_idx=None,
        drop_when_test=False,
        drop_lvl_idx=None,
        drop_lvl_num=0,
        random_short_schedule=False,  # randomly use short schedule (schedule for images of 256x256)
        short_schedule_prob=0.5,
        disable_flip_prob=0.0,  # disable random flip in this image
        uniform_short_schedule=False,
        **kwargs
    ):
        super().__init__()
        codebook_dim = dim

        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection
        self.layernorm = nn.Identity()
        self.use_stochastic_depth = use_stochastic_depth
        self.drop_rate = drop_rate
        self.remove_residual_detach = remove_residual_detach
        self.random_flip = random_flip
        self.flip_prob = flip_prob
        self.flip_mode = flip_mode
        self.max_flip_lvl = max_flip_lvl
        self.random_flip_1lvl = random_flip_1lvl
        self.flip_lvl_idx = flip_lvl_idx
        assert (random_flip and random_flip_1lvl) == False
        self.disable_flip_prob = disable_flip_prob

        self.drop_when_test = drop_when_test
        self.drop_lvl_idx = drop_lvl_idx
        self.drop_lvl_num = drop_lvl_num
        if self.drop_when_test:
            assert drop_lvl_idx is not None
            assert drop_lvl_num > 0
        self.random_short_schedule = random_short_schedule
        self.short_schedule_prob = short_schedule_prob
        self.full2short = {7: 7, 10: 7, 13: 7, 16: 16, 20: 16, 24: 16}
        self.full2short_f8 = {20: 20, 24: 20, 28: 20}
        self.uniform_short_schedule = uniform_short_schedule
        assert not (self.random_short_schedule and self.uniform_short_schedule)

        self.lfq = BSQ(
            dim=codebook_dim,
            codebook_scale=1,
            soft_clamp_input_value=soft_clamp_input_value,
            **kwargs
        )

        self.z_interplote_up = 'trilinear'
        self.z_interplote_down = 'area'

        self.use_decay_factor = use_decay_factor
        self.schedule_mode = schedule_mode
        self.keep_first_quant = keep_first_quant
        self.keep_last_quant = keep_last_quant
        if self.use_stochastic_depth and self.drop_rate > 0:
            assert self.keep_first_quant or self.keep_last_quant

    @property
    def codebooks(self):
        return self.lfq.codebook

    def get_codes_from_indices(self, indices_list):
        all_codes = []
        for indices in indices_list:
            codes = self.lfq.indices_to_codes(indices)
            all_codes.append(codes)
        _, _, T, H, W = all_codes[-1].size()
        summed_codes = 0
        for code in all_codes:
            summed_codes += F.interpolate(code, size=(T, H, W), mode=self.z_interplote_up)
        return summed_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def flip_quant(self, x):
        if self.flip_mode == 'stochastic':
            flip_mask = torch.rand_like(x) < self.flip_prob
        else:
            raise NotImplementedError
        x = x.clone()
        x[flip_mask] = -x[flip_mask]
        return x

    def forward(
        self,
        x,
        mask=None,
        return_all_codes=False,
    ):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        B, C, T, H, W = x.size()

        if self.schedule_mode.startswith("same"):
            scale_num = int(self.schedule_mode[len("same"):])
            assert T == 1
            scale_schedule = [(1, H, W)] * scale_num
        else:
            scale_schedule = get_latent2scale_schedule(T, H, W, mode=self.schedule_mode)
            scale_num = len(scale_schedule)

        if self.uniform_short_schedule:
            scale_num_short = self.full2short_f8[scale_num] if self.schedule_mode == "dense_f8" else self.full2short[scale_num]
            scale_num = random.randint(scale_num_short, scale_num)
            scale_schedule = scale_schedule[:scale_num]
        elif self.random_short_schedule and random.random() < self.short_schedule_prob:
            if self.schedule_mode == "dense_f8":
                scale_num = self.full2short_f8[scale_num]
            else:
                scale_num = self.full2short[scale_num]
            scale_schedule = scale_schedule[:scale_num]

        # x = self.project_in(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (b, c, t, h, w) => (b, t, h, w, c)
        x = self.project_in(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (b, t, h, w, c) => (b, c, t, h, w)
        x = self.layernorm(x)

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        all_bit_indices = []

        # go through the layers
        out_fact = init_out_fact = 1.0
        # residual_list = []
        # interpolate_residual_list = []
        # quantized_list = []
        if self.drop_when_test:
            drop_lvl_start = self.drop_lvl_idx
            drop_lvl_end = self.drop_lvl_idx + self.drop_lvl_num
        disable_flip = True if random.random() < self.disable_flip_prob else False  # disable random flip in this image
        with autocast('cuda', enabled=False):
            for si, (pt, ph, pw) in enumerate(scale_schedule):

                out_fact = max(0.1, out_fact) if self.use_decay_factor else init_out_fact
                if (pt, ph, pw) != (T, H, W):
                    interpolate_residual = F.interpolate(residual, size=(pt, ph, pw), mode=self.z_interplote_down)
                else:
                    interpolate_residual = residual
                # residual_list.append(torch.norm(residual.detach(), dim=1).mean())
                # interpolate_residual_list.append(torch.norm(interpolate_residual.detach(), dim=1).mean())
                if self.training and self.use_stochastic_depth and random.random() < self.drop_rate:
                    if (si == 0 and self.keep_first_quant) or (si == scale_num - 1 and self.keep_last_quant):
                        quantized, indices, bit_indices, loss = self.lfq(interpolate_residual)
                        if self.random_flip and si < self.max_flip_lvl and (not disable_flip):
                            quantized = self.flip_quant(quantized)
                        quantized = quantized * out_fact
                        all_indices.append(indices)
                        all_losses.append(loss)
                        all_bit_indices.append(bit_indices)
                    else:
                        quantized = torch.zeros_like(interpolate_residual)
                elif self.drop_when_test and drop_lvl_start <= si < drop_lvl_end:
                    continue
                else:
                    # residual_norm = torch.norm(interpolate_residual.detach(), dim=1) # (b, t, h, w)
                    # print(si, residual_norm.min(), residual_norm.max(), residual_norm.mean())
                    quantized, indices, bit_indices, loss = self.lfq(interpolate_residual)
                    if self.random_flip and si < self.max_flip_lvl and (not disable_flip):
                        quantized = self.flip_quant(quantized)
                    if self.random_flip_1lvl and si == self.flip_lvl_idx and (not disable_flip):
                        quantized = self.flip_quant(quantized)
                    quantized = quantized * out_fact
                    all_indices.append(indices)
                    all_losses.append(loss)
                    all_bit_indices.append(bit_indices)
                # quantized_list.append(torch.norm(quantized.detach(), dim=1).mean())
                if (pt, ph, pw) != (T, H, W):
                    quantized = F.interpolate(quantized, size=(T, H, W), mode=self.z_interplote_up).contiguous()

                if self.remove_residual_detach:
                    residual = residual - quantized
                else:
                    residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                if self.use_decay_factor:
                    out_fact -= 0.1
        # print("residual_list:", residual_list)
        # print("interpolate_residual_list:", interpolate_residual_list)
        # print("quantized_list:", quantized_list)
        # import ipdb; ipdb.set_trace()
        # project out, if needed
        quantized_out = quantized_out.permute(0, 2, 3, 4, 1).contiguous()  # (b, c, t, h, w) => (b, t, h, w, c)
        quantized_out = self.project_out(quantized_out)
        quantized_out = quantized_out.permute(0, 4, 1, 2, 3).contiguous()  # (b, t, h, w, c) => (b, c, t, h, w)

        # image
        if quantized_out.size(2) == 1:
            quantized_out = quantized_out.squeeze(2)

        # stack all losses and indices

        all_losses = torch.stack(all_losses, dim=-1)

        ret = (quantized_out, all_indices, all_bit_indices, all_losses)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers
        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)


class BSQ(Module):
    def __init__(
        self,
        *,
        dim=None,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        diversity_gamma=1.,
        straight_through_activation=nn.Identity(),
        num_codebooks=1,
        keep_num_codebooks_dim=None,
        codebook_scale=1.,                        # for residual LFQ, codebook scaled down by 2x at each layer
        frac_per_sample_entropy=1.,               # make less than 1. to only use a random fraction of the probs for per sample entropy
        has_projections=None,
        projection_has_bias=True,
        soft_clamp_input_value=None,
        cosine_sim_project_in=False,
        cosine_sim_project_in_scale=None,
        channel_first=None,
        experimental_softplus_entropy_loss=False,
        entropy_loss_offset=5.,                   # how much to shift the loss before softplus
        spherical=True,                          # from https://arxiv.org/abs/2406.07548
        force_quantization_f32=True,               # will force the quantization step to be full precision
        inv_temperature=100.0,
        gamma0=1.0, gamma=1.0, zeta=1.0,
        new_quant=False,  # new quant function，
        use_out_phi=False,  # use output phi network
        use_out_phi_res=False,  # residual out phi
    ):
        super().__init__()

        # some assert validations
        assert exists(dim), 'dim must be specified for BSQ'

        codebook_dim = dim
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        self.codebook_dims = codebook_dims

        has_projections = default(has_projections, dim != codebook_dims)

        if cosine_sim_project_in:
            cosine_sim_project_in = default(cosine_sim_project_in_scale, codebook_scale)
            project_in_klass = partial(CosineSimLinear, scale=cosine_sim_project_in)
        else:
            project_in_klass = partial(nn.Linear, bias=projection_has_bias)

        self.project_in = project_in_klass(dim, codebook_dims) if has_projections else nn.Identity()  # nn.Identity()
        self.project_out = nn.Linear(
            codebook_dims, dim, bias=projection_has_bias) if has_projections else nn.Identity()  # nn.Identity()
        self.has_projections = has_projections

        self.out_phi = nn.Linear(codebook_dims, codebook_dims) if use_out_phi else nn.Identity()
        self.use_out_phi_res = use_out_phi_res
        if self.use_out_phi_res:
            self.out_phi_scale = nn.Parameter(torch.zeros(codebook_dims), requires_grad=True)  # init as zero

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # channel first

        self.channel_first = channel_first

        # straight through activation

        self.activation = straight_through_activation

        # For BSQ (binary spherical quantization)
        if not spherical:
            raise ValueError("For BSQ, spherical must be True.")
        self.persample_entropy_compute = 'analytical'
        self.inv_temperature = inv_temperature
        self.gamma0 = gamma0  # loss weight for entropy penalty
        self.gamma = gamma  # loss weight for entropy penalty
        self.zeta = zeta    # loss weight for entire entropy penalty
        self.new_quant = new_quant

        # entropy aux loss related weights

        assert 0 < frac_per_sample_entropy <= 1.
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # whether to soft clamp the input value from -value to value

        self.soft_clamp_input_value = soft_clamp_input_value
        assert not exists(soft_clamp_input_value) or soft_clamp_input_value >= codebook_scale

        # whether to make the entropy loss positive through a softplus (experimental, please report if this worked or not in discussions)

        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss

        # for no auxiliary loss, during inference

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent=False)

        # whether to force quantization step to be f32

        self.force_quantization_f32 = force_quantization_f32

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    # @property
    # def dtype(self):
    #     return self.codebook.dtype

    def indices_to_codes(
        self,
        indices,
        label_type='int_label',
        project_out=True
    ):
        assert label_type in ['int_label', 'bit_label']
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = default(self.channel_first, is_img_or_video)

        if not self.keep_num_codebooks_dim:
            if label_type == 'int_label':
                indices = rearrange(indices, '... -> ... 1')
            else:
                indices = indices.unsqueeze(-2)

        # indices to codes, which are bits of either -1 or 1

        if label_type == 'int_label':
            assert indices[..., None].int().min() > 0
            bits = ((indices[..., None].int() & self.mask) != 0).float()  # .to(self.dtype)
        else:
            bits = indices

        codes = self.bits_to_codes(bits)

        codes = l2norm(codes)  # must normalize when using BSQ

        codes = rearrange(codes, '... c d -> ... (c d)')

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        if should_transpose:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def quantize(self, z):
        assert z.shape[-1] == self.codebook_dims, f"Expected {self.codebook_dims} dimensions, got {z.shape[-1]}"

        zhat = torch.where(z > 0,
                           torch.tensor(1, dtype=z.dtype, device=z.device),
                           torch.tensor(-1, dtype=z.dtype, device=z.device))
        return z + (zhat - z).detach()

    def quantize_new(self, z):
        assert z.shape[-1] == self.codebook_dims, f"Expected {self.codebook_dims} dimensions, got {z.shape[-1]}"

        zhat = torch.where(z > 0,
                           torch.tensor(1, dtype=z.dtype, device=z.device),
                           torch.tensor(-1, dtype=z.dtype, device=z.device))

        q_scale = 1. / (self.codebook_dims ** 0.5)
        zhat = q_scale * zhat  # on unit sphere

        return z + (zhat - z).detach()

    def soft_entropy_loss(self, z):
        if self.persample_entropy_compute == 'analytical':
            # if self.l2_norm:
            p = torch.sigmoid(-4 * z / (self.codebook_dims ** 0.5) * self.inv_temperature)
            # else:
            #     p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob = torch.stack([p, 1 - p], dim=-1)  # (b, h, w, 18, 2)
            per_sample_entropy = self.get_entropy(
                prob, dim=-1, normalize=False).sum(dim=-1).mean()  # (b,h,w,18)->(b,h,w)->scalar
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        # macro average of the probability of each subgroup
        avg_prob = reduce(prob, '... g d ->g d', 'mean')  # (18, 2)
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)

        # the approximation of the entropy is the sum of the entropy of each subgroup
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        if normalize:  # False
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim=True)
        else:  # True
            probs = count
        H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
        return H

    def forward(
        self,
        x,
        return_loss_breakdown=False,
        mask=None,
        entropy_weight=0.1
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = x.ndim >= 4
        # 如果没有显式指定 channel_first (默认None)，则认为 4/5 维张量 (Image/Video) 是 channel-first 的
        should_transpose = default(self.channel_first, is_img_or_video)

        # standardize image or video into (batch, seq, dimension)
        if should_transpose:
            # 如果是 (B, C, H, W) 或 (B, C, T, H, W)，先调整为 (B, ..., C)
            x = rearrange(x, 'b d ... -> b ... d')
            # 这里的 ps 保存了 '...' 的形状信息，方便后续还原
            # x 变成 (Batch, SequenceLength, FeatureDim)，统一处理为序列
            x, ps = pack_one(x, 'b * d')  # x.shape [b, hwt, c]

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'

        x = self.project_in(x)

        # split out number of codebooks
        # 如果用了多个 codebook (num_codebooks > 1)，将最后一个维度拆分
        # 一般默认 num_codebooks=1，所以此操作相当于没变
        x = rearrange(x, 'b n (c d) -> b n c d', c=self.num_codebooks)
        # 将特征向量投影到球面上 (L2 Normalize)，因为 BSQ 是球面量化
        x = l2norm(x)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32
        # 决定是否关闭 autocast (混合精度)，以确保量化计算精度（强制 FP32）
        quantization_context = partial(autocast, 'cuda', enabled=False) if force_f32 else nullcontext

        indices = None  # BSQ 不像传统 VQ 查表，所以这里的 indices 概念比较特别，通常其实用不到
        with quantization_context():

            if force_f32:
                orig_dtype = x.dtype
                x = x.float()  # 转为 float32 计算

            # use straight-through gradients (optionally with custom activation fn) if training
            if self.new_quant:
                quantized = self.quantize_new(x)
            else:
                # 核心：执行二值化操作
                # 这里的 quantize 函数包含 Straight-Through Estimator (STE)
                # 即 Forward 阶段返回二值化的 zhat，Backward 阶段让梯度直接流过 z
                quantized = self.quantize(x)
                # 缩放回单位球面上
                # 因为原始二值向量模长很大，除以 sqrt(d) 把它拉回单位圆
                q_scale = 1. / (self.codebook_dims ** 0.5)
                quantized = q_scale * quantized  # on unit sphere

            # calculate indices
            # 记录此时的二进制位状态 (0 或 1)，用于返回给用户看具体的码本索引
            bit_indices = (quantized > 0).int()

            # entropy aux loss
            if self.training:
                # 计算各种熵：每个样本的熵，整个 Codebook 的平均熵
                # 这一步是为了鼓励码本被均匀利用，防止某些二进制位永远是 0 或永远是 1
                persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(x)  # compute entropy
                # 组合 entropy loss: 最大化样本熵，最小化码本熵
                entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
            else:
                # if not training, just return dummy 0
                # 推理时不需要计算这些 loss，直接置 0
                entropy_penalty = persample_entropy = cb_entropy = self.zero

            # commit loss

            if self.training and self.commitment_loss_weight > 0.:
                # 计算量化前的特征 x 和量化后的 quantized 之间的 MSE
                # 这拉近“连续特征”和“离散特征”的距离，让 Encoder 吐出的特征更接近量化点
                commit_loss = F.mse_loss(x, quantized.detach(), reduction='none')

                if exists(mask):
                    commit_loss = commit_loss[mask]

                commit_loss = commit_loss.mean()
            else:
                commit_loss = self.zero

            # input back to original dtype if needed
            if force_f32:
                x = x.type(orig_dtype)  # 转回原来的精度 (如 FP16/BF16)

        # merge back codebook dim
        x = quantized  # rename quantized to x for output, x 现在变成了量化后的离散特征

        if self.use_out_phi_res:  # 可选：通过 MLP 对量化结果再做一次变换 (Out Phi)
            x = x + self.out_phi_scale * self.out_phi(x)  # apply out_phi on quant output as residual, 残差形式
        else:
            x = self.out_phi(x)  # apply out_phi on quant output, 直接变换

        x = rearrange(x, 'b n c d -> b n (c d)')  # 将 codebook 维度合并回去 (如果是多 codebook)

        # project out to feature dimension if needed
        # 投影回原始维度 (如果输入输出维度不一致)
        x = self.project_out(x)

        # reconstitute image or video dimensions

        if should_transpose:
            x = unpack_one(x, ps, 'b * d')  # 使用 unpack 还原回 (B, SequenceLength, C)
            x = rearrange(x, 'b ... d -> b d ...')  # 还原回 (B, C, H, W)
            # 还原 indices 的形状
            bit_indices = unpack_one(bit_indices, ps, 'b * c d')

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            bit_indices = rearrange(bit_indices, '... 1 d -> ... d')

        # complete aux loss
        # 加权求和最终的辅助 Loss (Commitment + Entropy)
        aux_loss = commit_loss * self.commitment_loss_weight + \
            (self.zeta * entropy_penalty / self.inv_temperature) * entropy_weight
        # returns
        # 打包返回：(量化后特征，索引(占位)，二进制位索引，辅助损失)
        ret = Return(x, indices, bit_indices, aux_loss)

        if not return_loss_breakdown:
            return ret
        # 如果需要详细 debug 信息，返回更多
        return ret, LossBreakdown(persample_entropy, cb_entropy, commit_loss)
