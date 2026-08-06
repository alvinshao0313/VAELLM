import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor

from litebsq.bsq import BSQ
from litebsq.misc import ptdtype
from litebsq.parallel_layers import (
    Normalize,
    ParallelLinear,
    ResnetBlock1D,
    apply_activation,
    pack_normalizes,
    pack_parallel_linears,
    pack_resnet_blocks,
)
from litebsq.vae_args import add_autoencoder_model_args, resolve_autoencoder_arch_spec


@dataclass(frozen=True)
class QuantizerResult:
    z: Tensor
    bit_indices: Optional[Tensor]
    aux_loss: Tensor


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        num_res_blocks: int,
        out_dim: int,
        norm_type: str = "group",
        activation_type: str = "swish",
        use_checkpoint: bool = False,
        num_models: int = 1,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_res_blocks = int(num_res_blocks)
        self.out_dim = int(out_dim)
        self.activation_type = str(activation_type).strip().lower()
        self.use_checkpoint = bool(use_checkpoint)
        self.num_models = int(num_models)

        self.linear_in = ParallelLinear(self.in_dim, self.hidden_dim, num_models=self.num_models)
        self.blocks = nn.ModuleList(
            [
                ResnetBlock1D(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    norm_type=norm_type,
                    activation_type=self.activation_type,
                    num_models=self.num_models,
                )
                for _ in range(self.num_res_blocks)
            ]
        )
        self.norm_out = Normalize(self.hidden_dim, norm_type, num_models=self.num_models)
        self.linear_out = ParallelLinear(self.hidden_dim, self.out_dim, num_models=self.num_models)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tensor:
        h = self.linear_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        h = apply_activation(h, self.activation_type)
        return self.linear_out(h)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        norm_type: str = "group",
        activation_type: str = "swish",
        decoder_type: str = "linear",
        use_checkpoint: bool = False,
        num_models: int = 1,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_res_blocks = int(num_res_blocks)
        self.norm_type = str(norm_type).strip().lower()
        self.activation_type = str(activation_type).strip().lower()
        self.decoder_type = str(decoder_type).strip().lower()
        self.use_checkpoint = bool(use_checkpoint)
        self.num_models = int(num_models)
        self._q_scale_fused = False

        if self.decoder_type == "linear":
            self.linear = ParallelLinear(self.in_dim, self.out_dim, num_models=self.num_models)
        elif self.decoder_type in {"symmetric", "asymmetric"}:
            self.linear_in = ParallelLinear(self.in_dim, self.hidden_dim, num_models=self.num_models)
            self.blocks = nn.ModuleList(
                [
                    ResnetBlock1D(
                        in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim,
                        norm_type=self.norm_type,
                        activation_type=self.activation_type,
                        num_models=self.num_models,
                    )
                    for _ in range(self.num_res_blocks)
                ]
            )
            self.norm_out = Normalize(self.hidden_dim, self.norm_type, num_models=self.num_models)
            self.linear_out = ParallelLinear(self.hidden_dim, self.out_dim, num_models=self.num_models)
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(self, x: Tensor) -> Tensor:
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tensor:
        if self.decoder_type == "linear":
            return self.linear(x)

        h = self.linear_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        h = apply_activation(h, self.activation_type)
        return self.linear_out(h)

    @torch.no_grad()
    def _fuse_q_scale(self, q_scale: float = None) -> None:
        if self._q_scale_fused:
            return
        if q_scale is None:
            q_scale = 1.0 / math.sqrt(self.in_dim)
        else:
            q_scale = float(q_scale)

        def _fuse_linear_or_parallel(layer) -> None:
            if hasattr(layer, "fuse_q_scale"):
                layer.fuse_q_scale(q_scale)
                return
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data
                bias_delta = -q_scale * weight.sum(dim=1)
                weight.mul_(q_scale * 2.0)
                if layer.bias is not None:
                    layer.bias.data.add_(bias_delta)
                else:
                    layer.bias = nn.Parameter(bias_delta)
                return
            raise TypeError(f"Unsupported layer type for q_scale fusion: {type(layer)}")

        if self.decoder_type == "linear":
            _fuse_linear_or_parallel(self.linear)
        elif self.decoder_type in {"symmetric", "asymmetric"}:
            _fuse_linear_or_parallel(self.linear_in)
        self._q_scale_fused = True

    def extract_single(self, model_idx: int) -> "Decoder":
        if model_idx < 0 or model_idx >= self.num_models:
            raise ValueError(f"Index {model_idx} out of range [0, {self.num_models - 1}]")

        decoder = Decoder(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            hidden_dim=self.hidden_dim,
            num_res_blocks=self.num_res_blocks,
            norm_type=self.norm_type,
            activation_type=self.activation_type,
            decoder_type=self.decoder_type,
            use_checkpoint=self.use_checkpoint,
            num_models=1,
        )
        decoder._q_scale_fused = bool(self._q_scale_fused)

        if self.decoder_type == "linear":
            decoder.linear = self.linear.extract_single(model_idx)
            decoder.train(self.training)
            return decoder

        decoder.linear_in = self.linear_in.extract_single(model_idx)
        decoder.blocks = nn.ModuleList(
            [block.extract_single(model_idx) for block in self.blocks]
        )
        decoder.norm_out = self.norm_out.extract_single(model_idx)
        decoder.linear_out = self.linear_out.extract_single(model_idx)
        decoder.train(self.training)
        return decoder

    def get_sub_decoder(self, model_idx: int) -> "Decoder":
        return self.extract_single(model_idx)


@torch.no_grad()
def pack_decoders(decoders: Sequence[Decoder]) -> Decoder:
    if not decoders:
        raise ValueError("pack_decoders expects at least one decoder.")

    first = decoders[0]
    if not isinstance(first, Decoder):
        raise TypeError(f"pack_decoders expects Decoder instances, got {type(first)}.")
    if int(first.num_models) != 1:
        raise ValueError(f"pack_decoders expects single-model decoders, got num_models={first.num_models}.")

    training = bool(first.training)
    device = None
    dtype = None
    for param in first.parameters():
        if param.is_floating_point():
            device = param.device
            dtype = param.dtype
            break

    for idx, decoder in enumerate(decoders[1:], start=1):
        if not isinstance(decoder, Decoder):
            raise TypeError(f"pack_decoders expects Decoder instances, got {type(decoder)} at idx={idx}.")
        if int(decoder.num_models) != 1:
            raise ValueError(
                f"pack_decoders expects single-model decoders, got num_models={decoder.num_models} at idx={idx}."
            )
        if (
            int(decoder.in_dim) != int(first.in_dim)
            or int(decoder.out_dim) != int(first.out_dim)
            or int(decoder.hidden_dim) != int(first.hidden_dim)
            or int(decoder.num_res_blocks) != int(first.num_res_blocks)
            or str(decoder.norm_type) != str(first.norm_type)
            or str(decoder.activation_type) != str(first.activation_type)
            or str(decoder.decoder_type) != str(first.decoder_type)
            or bool(decoder.use_checkpoint) != bool(first.use_checkpoint)
        ):
            raise ValueError(
                f"pack_decoders config mismatch at idx={idx}: "
                f"got in={int(decoder.in_dim)}, out={int(decoder.out_dim)}, hidden={int(decoder.hidden_dim)}, "
                f"blocks={int(decoder.num_res_blocks)}, norm={str(decoder.norm_type)}, "
                f"activation={str(decoder.activation_type)}, "
                f"type={str(decoder.decoder_type)}, ckpt={bool(decoder.use_checkpoint)} "
                f"vs first decoder."
            )
        if bool(decoder.training) != training:
            raise ValueError("pack_decoders requires all decoders to share the same training mode.")
        if bool(decoder._q_scale_fused) != bool(first._q_scale_fused):
            raise ValueError("pack_decoders requires identical _q_scale_fused across all decoders.")
        for param in decoder.parameters():
            if not param.is_floating_point():
                continue
            if device is None:
                device = param.device
                dtype = param.dtype
            elif param.device != device or param.dtype != dtype:
                raise ValueError(
                    f"pack_decoders dtype/device mismatch at idx={idx}: "
                    f"device={param.device}, dtype={param.dtype} vs device={device}, dtype={dtype}."
                )
            break

    packed = Decoder(
        in_dim=first.in_dim,
        out_dim=first.out_dim,
        hidden_dim=first.hidden_dim,
        num_res_blocks=first.num_res_blocks,
        norm_type=first.norm_type,
        activation_type=first.activation_type,
        decoder_type=first.decoder_type,
        use_checkpoint=first.use_checkpoint,
        num_models=len(decoders),
    )
    if device is not None:
        packed = packed.to(device=device, dtype=dtype)
    packed.requires_grad_(False)
    packed._q_scale_fused = bool(first._q_scale_fused)

    if packed.decoder_type == "linear":
        packed.linear = pack_parallel_linears([decoder.linear for decoder in decoders])
        packed.train(training)
        return packed

    packed.linear_in = pack_parallel_linears([decoder.linear_in for decoder in decoders])
    packed.blocks = nn.ModuleList(
        pack_resnet_blocks([decoder.blocks[block_idx] for decoder in decoders])
        for block_idx in range(int(first.num_res_blocks))
    )
    packed.norm_out = pack_normalizes([decoder.norm_out for decoder in decoders])
    packed.linear_out = pack_parallel_linears([decoder.linear_out for decoder in decoders])
    packed.train(training)
    return packed


class AutoEncoder(nn.Module):
    def __init__(self, args, num_models: int = 1):
        super().__init__()
        self.args = args
        self.num_models = int(num_models)
        self.arch_spec = resolve_autoencoder_arch_spec(args)

        self.chunk_size = int(self.arch_spec.codebook_dim)
        self.latent_dim = int(self.arch_spec.codebook_bits)
        self.encoder = Encoder(
            in_dim=self.chunk_size,
            hidden_dim=self.arch_spec.encoder_hidden_dim,
            num_res_blocks=self.arch_spec.encoder_num_res_blocks,
            out_dim=self.latent_dim,
            norm_type=self.arch_spec.norm_type,
            activation_type=self.arch_spec.activation_type,
            use_checkpoint=self.arch_spec.use_checkpoint,
            num_models=self.num_models,
        )
        self.decoder = Decoder(
            in_dim=self.latent_dim,
            out_dim=self.chunk_size,
            hidden_dim=self.arch_spec.decoder_hidden_dim,
            num_res_blocks=self.arch_spec.decoder_num_res_blocks,
            norm_type=self.arch_spec.norm_type,
            activation_type=self.arch_spec.activation_type,
            decoder_type=self.arch_spec.decoder_type,
            use_checkpoint=self.arch_spec.use_checkpoint,
            num_models=self.num_models,
        )

        self.recon_loss_type = str(getattr(args, "recon_loss_type", "mse")).strip().lower()
        self.l1_weight = float(getattr(args, "l1_weight", 1.0))
        self.lfq_weight = float(getattr(args, "lfq_weight", 1.0))
        self.commitment_loss_weight = float(getattr(args, "commitment_loss_weight", 0.25))
        self.quantizer_type = str(getattr(args, "quantizer_type", "BSQ"))
        self.quantizer = self._build_quantizer(args)

        weight_dtype = str(getattr(args, "vae_weight_dtype", "fp32")).lower()
        if weight_dtype == "bf16":
            self.params_dtype = torch.bfloat16
            self.to(dtype=torch.bfloat16)
        elif weight_dtype == "fp16":
            self.params_dtype = torch.float16
            self.to(dtype=torch.float16)
        else:
            self.params_dtype = torch.float32

    def _build_quantizer(self, args):
        if self.quantizer_type == "BSQ":
            return BSQ(
                dim=self.latent_dim,
                codebook_scale=1,
                entropy_loss_weight=getattr(args, "entropy_loss_weight", 0.1),
                commitment_loss_weight=getattr(args, "commitment_loss_weight", 0.25),
                has_projections=False,
                spherical=True,
                new_quant=getattr(args, "new_quant", False),
                gamma0=getattr(args, "gamma0", 1.0),
                gamma=getattr(args, "gamma", 1.0),
                zeta=getattr(args, "zeta", 1.0),
                inv_temperature=getattr(args, "inv_temperature", 100.0),
            )
        if self.quantizer_type == "Identity":
            return nn.Identity()
        raise NotImplementedError(f"{self.quantizer_type} not supported, use BSQ")

    def _autocast_context(self, x: Tensor):
        autocast_name = getattr(self.args, "vae_autocast_dtype", "fp32")
        autocast_dtype = ptdtype.get(autocast_name, torch.float32)
        if x.device.type != "cuda" or autocast_dtype == torch.float32:
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=autocast_dtype)

    def _parse_quantizer_output(self, quant_ret, device) -> QuantizerResult:
        if self.quantizer_type == "Identity":
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return QuantizerResult(z=quant_ret, bit_indices=None, aux_loss=zero)
        if isinstance(quant_ret, tuple):
            z, bit_indices, aux_loss = quant_ret
            return QuantizerResult(z=z, bit_indices=bit_indices, aux_loss=aux_loss)
        return QuantizerResult(
            z=quant_ret.quantized,
            bit_indices=quant_ret.bit_indices,
            aux_loss=quant_ret.entropy_aux_loss,
        )

    def _run_encode_quantize_decode(self, x: Tensor) -> tuple[Tensor, QuantizerResult]:
        with self._autocast_context(x):
            h = self.encoder(x)
            quant_result = self._parse_quantizer_output(self.quantizer(h), x.device)
            z = quant_result.z.to(self.params_dtype)
            x_recon = self.decoder(z)
        return x_recon, quant_result

    def _compute_recon_loss(self, x_recon: Tensor, x: Tensor, act_max: Optional[Tensor] = None) -> Tensor:
        if self.recon_loss_type == "l1":
            return F.l1_loss(x_recon, x)
        if self.recon_loss_type == "huber":
            return F.huber_loss(x_recon, x, reduction="mean", delta=1.0)
        if self.recon_loss_type == "relative_l1":
            return (x_recon - x).abs().sum() / (x.abs().sum() + 1e-10)
        if self.recon_loss_type == "mse":
            return F.mse_loss(x_recon, x)
        if self.recon_loss_type == "w_mse":
            return ((x_recon - x).pow(2) * x.abs()).mean()
        if self.recon_loss_type == "w2_mse":
            return ((x_recon - x).pow(2) * x.pow(2)).mean()
        if self.recon_loss_type == "wa_mse":
            if act_max is None:
                raise ValueError("recon_loss_type=wa_mse requires act_max tensor.")
            if act_max.shape != x.shape:
                raise ValueError(
                    f"wa_mse shape mismatch: act_max={tuple(act_max.shape)} vs x={tuple(x.shape)}"
                )
            x_f = x.float()
            x_recon_f = x_recon.float()
            act_f = act_max.float()
            errors = (x_recon_f - x_f).pow(2)
            weights = x_f.abs() * act_f
            return (errors * weights).mean()
        if self.recon_loss_type == "amse":
            if act_max is None:
                raise ValueError("recon_loss_type=amse requires hessian_diag/channel_weight tensor.")
            if act_max.shape != x.shape:
                raise ValueError(
                    f"amse shape mismatch: hessian_diag={tuple(act_max.shape)} vs x={tuple(x.shape)}"
                )
            x_f = x.float()
            x_recon_f = x_recon.float()
            h_f = act_max.float()
            errors = (x_recon_f - x_f).pow(2)
            return (errors * h_f).mean()
        raise ValueError(
            f"Unsupported recon_loss_type={self.recon_loss_type!r}."
        )

    def _forward_train(self, x: Tensor, act_max: Optional[Tensor] = None):
        x_recon, quant_result = self._run_encode_quantize_decode(x)
        recon_loss = self._compute_recon_loss(x_recon, x, act_max=act_max)
        loss_dict = {
            "train/recon_loss": recon_loss * self.l1_weight * self.num_models,
            "train/commitment_loss": quant_result.aux_loss * self.lfq_weight * self.num_models,
        }
        return x_recon, x.detach(), loss_dict

    def _forward_eval(self, x: Tensor):
        x_recon, quant_result = self._run_encode_quantize_decode(x)
        return x_recon, quant_result.bit_indices

    def forward(self, x: Tensor, global_step=None, is_train: bool = True, act_max: Optional[Tensor] = None):
        del global_step
        if is_train:
            return self._forward_train(x, act_max=act_max)
        return self._forward_eval(x)

    @staticmethod
    def add_model_specific_args(parent_parser):
        return add_autoencoder_model_args(parent_parser)


class MultiLayerVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_models = int(getattr(args, "parallel_layers", 1))
        if self.num_models < 1:
            raise ValueError(f"parallel_layers must be >= 1, got {self.num_models}")
        self.model = AutoEncoder(args, num_models=self.num_models)

    def forward(self, x: Tensor, is_train: bool = True, act_max: Optional[Tensor] = None):
        if is_train:
            x_recon, _, loss_dict = self.model(x, is_train=True, act_max=act_max)
            recon_loss = loss_dict["train/recon_loss"]
            commit_loss = loss_dict["train/commitment_loss"]
            loss_dict["loss"] = recon_loss + commit_loss
            return x_recon, loss_dict
        return self.model(x, is_train=False)


__all__ = [
    "AutoEncoder",
    "Decoder",
    "Encoder",
    "MultiLayerVAE",
    "QuantizerResult",
]
