"""
Binary Spherical Quantization (BSQ)
Simplified for litebsq: only BSQ is kept.
"""

from collections import namedtuple
from contextlib import nullcontext
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.amp import autocast
from einops import rearrange, reduce, pack, unpack

Return = namedtuple('Return', ['quantized', 'bit_indices', 'entropy_aux_loss'])
LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def l2norm(t):
    return F.normalize(t, dim=-1)


class BSQ(nn.Module):
    def __init__(
        self,
        *,
        dim=None,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        diversity_gamma=1.0,
        num_codebooks=1,
        keep_num_codebooks_dim=None,
        codebook_scale=1.0,
        has_projections=None,
        projection_has_bias=True,
        channel_first=None,
        spherical=True,
        force_quantization_f32=True,
        inv_temperature=100.0,
        gamma0=1.0,
        gamma=1.0,
        zeta=1.0,
        new_quant=False,
    ):
        super().__init__()

        assert exists(dim), 'dim must be specified for BSQ'
        if not spherical:
            raise ValueError('For BSQ, spherical must be True.')

        codebook_dim = dim
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = default(has_projections, dim != codebook_dims)
        self.project_in = nn.Linear(dim, codebook_dims, bias=projection_has_bias) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim, bias=projection_has_bias) if has_projections else nn.Identity()

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.codebook_dims = codebook_dims
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.channel_first = channel_first
        self.inv_temperature = inv_temperature
        self.gamma0 = gamma0
        self.gamma = gamma
        self.zeta = zeta
        self.new_quant = new_quant
        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight
        self.codebook_scale = codebook_scale
        self.commitment_loss_weight = commitment_loss_weight
        self.force_quantization_f32 = force_quantization_f32

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.0), persistent=False)

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
        q_scale = 1.0 / (self.codebook_dims ** 0.5)
        zhat = q_scale * zhat
        return z + (zhat - z).detach()

    def soft_entropy_loss(self, z):
        p = torch.sigmoid(-4 * z / (self.codebook_dims ** 0.5) * self.inv_temperature)
        prob = torch.stack([p, 1 - p], dim=-1)
        per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
        avg_prob = reduce(prob, '... g d -> g d', 'mean')
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        if normalize:
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim=True)
        else:
            probs = count
        return -(probs * torch.log(probs + 1e-8)).sum(dim=dim)

    def forward(self, x, return_loss_breakdown=False, mask=None, entropy_weight=0.1):
        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        if should_transpose:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d')

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'

        x = self.project_in(x)
        x = rearrange(x, 'b n (c d) -> b n c d', c=self.num_codebooks)
        x = l2norm(x)

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled=False) if force_f32 else nullcontext

        with quantization_context():
            if force_f32:
                orig_dtype = x.dtype
                x = x.float()

            if self.new_quant:
                quantized = self.quantize_new(x)
            else:
                quantized = self.quantize(x)
                q_scale = 1.0 / (self.codebook_dims ** 0.5)
                quantized = q_scale * quantized

            bit_indices = (quantized > 0).bool()

            if self.training:
                persample_entropy, cb_entropy, _ = self.soft_entropy_loss(x)
                entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
            else:
                entropy_penalty = persample_entropy = cb_entropy = self.zero

            if self.training and self.commitment_loss_weight > 0.0:
                commit_loss = F.mse_loss(x, quantized.detach(), reduction='none')
                if exists(mask):
                    commit_loss = commit_loss[mask]
                commit_loss = commit_loss.mean()
            else:
                commit_loss = self.zero

            if force_f32:
                x = x.type(orig_dtype)

        x = quantized
        x = rearrange(x, 'b n c d -> b n (c d)')
        x = self.project_out(x)

        if should_transpose:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')
            bit_indices = unpack_one(bit_indices, ps, 'b * c d')

        if not self.keep_num_codebooks_dim:
            bit_indices = rearrange(bit_indices, '... 1 d -> ... d')

        aux_loss = commit_loss * self.commitment_loss_weight + \
            (self.zeta * entropy_penalty / self.inv_temperature) * entropy_weight

        ret = Return(x, bit_indices, aux_loss)

        if not return_loss_breakdown:
            return ret
        return ret, LossBreakdown(persample_entropy, cb_entropy, commit_loss)
