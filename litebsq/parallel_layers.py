from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_ACTIVATION_CHOICES = ("swish", "relu", "none", "sigmoid", "gelu", "hard_swish")


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def hard_swish(x: Tensor) -> Tensor:
    return x * F.relu6(x + 3.0) / 6.0


def apply_activation(x: Tensor, activation_type: str) -> Tensor:
    kind = str(activation_type).strip().lower()
    if kind == "swish":
        return swish(x)
    if kind == "relu":
        return F.relu(x)
    if kind == "none":
        return x
    if kind == "sigmoid":
        return torch.sigmoid(x)
    if kind == "gelu":
        return F.gelu(x)
    if kind == "hard_swish":
        return hard_swish(x)
    raise ValueError(
        f"Unsupported activation_type={activation_type!r}. Expected one of: {', '.join(_ACTIVATION_CHOICES)}."
    )


def _validate_model_index(model_idx: int, *, num_models: int) -> None:
    if model_idx < 0 or model_idx >= int(num_models):
        raise ValueError(f"Index {model_idx} out of range [0, {num_models - 1}]")


class ParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_models: int = 1):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_models = int(num_models)

        if self.num_models == 1:
            self.linear = nn.Linear(self.in_features, self.out_features)
        else:
            self.conv = nn.Conv1d(
                in_channels=self.in_features * self.num_models,
                out_channels=self.out_features * self.num_models,
                kernel_size=1,
                groups=self.num_models,
                bias=True,
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.num_models == 1:
            if x.dim() == 3 and x.shape[1] == 1:
                return self.linear(x.squeeze(1)).unsqueeze(1)
            return self.linear(x)

        batch_size = int(x.shape[0])
        x = x.reshape(batch_size, self.num_models * self.in_features, 1)
        out = self.conv(x)
        return out.reshape(batch_size, self.num_models, self.out_features)

    def extract_single(self, model_idx: int) -> nn.Linear:
        _validate_model_index(model_idx, num_models=self.num_models)

        if self.num_models == 1:
            layer = nn.Linear(self.in_features, self.out_features, bias=(self.linear.bias is not None))
            layer = layer.to(device=self.linear.weight.device, dtype=self.linear.weight.dtype)
            layer.load_state_dict(self.linear.state_dict())
            layer.train(self.training)
            return layer

        start = model_idx * self.out_features
        end = (model_idx + 1) * self.out_features
        layer = nn.Linear(self.in_features, self.out_features, bias=(self.conv.bias is not None))
        layer = layer.to(device=self.conv.weight.device, dtype=self.conv.weight.dtype)
        with torch.no_grad():
            layer.weight.copy_(self.conv.weight[start:end, :, 0])
            if self.conv.bias is not None:
                layer.bias.copy_(self.conv.bias[start:end])
        layer.train(self.training)
        return layer

    def get_sub_linear(self, model_idx: int) -> nn.Linear:
        return self.extract_single(model_idx)

    def fuse_q_scale(self, q_scale: float) -> None:
        if self.num_models == 1:
            weight = self.linear.weight.data
            bias_delta = -q_scale * weight.sum(dim=1)
            weight.mul_(q_scale * 2)
            if self.linear.bias is not None:
                self.linear.bias.data.add_(bias_delta)
            else:
                self.linear.bias = nn.Parameter(bias_delta)
            return

        weight = self.conv.weight.data
        bias_delta = -q_scale * weight[:, :, 0].sum(dim=1)
        weight.mul_(q_scale * 2)
        if self.conv.bias is not None:
            self.conv.bias.data.add_(bias_delta)
        else:
            self.conv.bias = nn.Parameter(bias_delta)


def _resolve_single_linear(module: nn.Module) -> nn.Linear:
    if isinstance(module, nn.Linear):
        return module
    if isinstance(module, ParallelLinear):
        if int(module.num_models) != 1:
            raise ValueError(
                f"pack_parallel_linears expects single-model layers, got ParallelLinear(num_models={module.num_models})."
            )
        return module.linear
    raise TypeError(f"Unsupported linear layer type for packing: {type(module)}")


@torch.no_grad()
def pack_parallel_linears(layers: Sequence[nn.Module]) -> ParallelLinear:
    if not layers:
        raise ValueError("pack_parallel_linears expects at least one layer.")

    singles = [_resolve_single_linear(layer) for layer in layers]
    first = singles[0]
    in_features = int(first.in_features)
    out_features = int(first.out_features)
    has_bias = first.bias is not None
    training = bool(first.training)
    device = first.weight.device
    dtype = first.weight.dtype

    for idx, layer in enumerate(singles[1:], start=1):
        if int(layer.in_features) != in_features or int(layer.out_features) != out_features:
            raise ValueError(
                f"pack_parallel_linears shape mismatch at idx={idx}: "
                f"got ({int(layer.in_features)}, {int(layer.out_features)}) vs ({in_features}, {out_features})."
            )
        if (layer.bias is not None) != has_bias:
            raise ValueError("pack_parallel_linears requires all layers to agree on bias presence.")
        if bool(layer.training) != training:
            raise ValueError("pack_parallel_linears requires all layers to share the same training mode.")
        if layer.weight.device != device:
            raise ValueError(
                f"pack_parallel_linears device mismatch at idx={idx}: {layer.weight.device} vs {device}."
            )
        if layer.weight.dtype != dtype:
            raise ValueError(
                f"pack_parallel_linears dtype mismatch at idx={idx}: {layer.weight.dtype} vs {dtype}."
            )

    packed = ParallelLinear(in_features, out_features, num_models=len(singles)).to(device=device, dtype=dtype)
    packed.requires_grad_(False)
    if int(packed.num_models) == 1:
        packed.linear.weight.copy_(first.weight)
        if has_bias:
            if packed.linear.bias is None:
                raise RuntimeError("packed single-model linear unexpectedly has no bias.")
            packed.linear.bias.copy_(first.bias)
        else:
            packed.linear.register_parameter("bias", None)
        packed.train(training)
        return packed

    packed.conv.weight.zero_()
    packed.conv.bias.zero_()
    for model_idx, layer in enumerate(singles):
        start = model_idx * out_features
        end = start + out_features
        packed.conv.weight[start:end, :, 0].copy_(layer.weight)
        if has_bias:
            packed.conv.bias[start:end].copy_(layer.bias)
    packed.train(training)
    return packed


class Normalize(nn.Module):
    def __init__(self, in_channels: int, norm_type: str = "group", num_models: int = 1):
        super().__init__()
        self.num_models = int(num_models)
        self.in_channels = int(in_channels)
        self.norm_type = str(norm_type)

        if self.norm_type not in {"group", "batch", "layer", "rms", "no"}:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        if self.norm_type == "group":
            groups_per_model = self._resolve_groups_per_model(self.in_channels)
            self.total_groups = groups_per_model * self.num_models
            self.total_channels = self.in_channels * self.num_models
            self.norm = nn.GroupNorm(
                num_groups=self.total_groups,
                num_channels=self.total_channels,
                eps=1e-6,
                affine=True,
            )
        elif self.norm_type == "batch":
            self.total_channels = self.in_channels * self.num_models
            self.norm = nn.BatchNorm1d(self.total_channels)
        elif self.norm_type == "layer":
            if self.num_models == 1:
                self.norm = nn.LayerNorm(self.in_channels)
            else:
                self.norm = nn.LayerNorm(self.in_channels, elementwise_affine=False)
                self.weight = nn.Parameter(torch.ones(self.num_models, self.in_channels))
                self.bias = nn.Parameter(torch.zeros(self.num_models, self.in_channels))
        elif self.norm_type == "rms":
            if self.num_models == 1:
                self.norm = nn.RMSNorm(self.in_channels, eps=1e-6)
            else:
                self.norm = nn.RMSNorm(self.in_channels, eps=1e-6, elementwise_affine=False)
                self.weight = nn.Parameter(torch.ones(self.num_models, self.in_channels))
        else:
            self.norm = nn.Identity()

    @staticmethod
    def _resolve_groups_per_model(in_channels: int) -> int:
        if in_channels >= 16 and in_channels % 16 == 0:
            return 16
        if in_channels >= 8 and in_channels % 8 == 0:
            return 8
        if in_channels >= 4 and in_channels % 4 == 0:
            return 4
        return 1

    def _flatten_parallel(self, x: Tensor) -> Tensor:
        batch_size = int(x.shape[0])
        return x.view(batch_size, self.in_channels * self.num_models)

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_type == "no":
            return self.norm(x)
        if self.norm_type == "layer":
            out = self.norm(x)
            if self.num_models == 1:
                return out
            return out * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        if self.norm_type == "rms":
            out = self.norm(x)
            if self.num_models == 1:
                return out
            return out * self.weight.unsqueeze(0)

        flat = self._flatten_parallel(x)
        out = self.norm(flat)
        if x.dim() == 2 and self.num_models == 1:
            return out
        return out.view(int(x.shape[0]), self.num_models, self.in_channels)

    def extract_single(self, model_idx: int) -> "Normalize":
        _validate_model_index(model_idx, num_models=self.num_models)
        new_norm = Normalize(self.in_channels, self.norm_type, num_models=1)
        ref_weight = getattr(self.norm, "weight", None)
        if isinstance(ref_weight, torch.Tensor):
            new_norm = new_norm.to(device=ref_weight.device, dtype=ref_weight.dtype)
        elif self.norm_type in {"layer", "rms"} and isinstance(getattr(self, "weight", None), torch.Tensor):
            new_norm = new_norm.to(device=self.weight.device, dtype=self.weight.dtype)
        elif self.norm_type == "batch":
            new_norm = new_norm.to(device=self.norm.running_mean.device, dtype=self.norm.running_mean.dtype)
        new_norm.train(self.training)

        if self.norm_type == "no":
            return new_norm

        if self.norm_type == "layer":
            if self.num_models == 1:
                new_norm.norm.load_state_dict(self.norm.state_dict())
            else:
                with torch.no_grad():
                    new_norm.norm.weight.copy_(self.weight[model_idx])
                    new_norm.norm.bias.copy_(self.bias[model_idx])
            return new_norm

        if self.norm_type == "rms":
            if self.num_models == 1:
                new_norm.norm.load_state_dict(self.norm.state_dict())
            else:
                with torch.no_grad():
                    new_norm.norm.weight.copy_(self.weight[model_idx])
            return new_norm

        start = model_idx * self.in_channels
        end = (model_idx + 1) * self.in_channels
        with torch.no_grad():
            if getattr(self.norm, "weight", None) is not None:
                new_norm.norm.weight.copy_(self.norm.weight[start:end])
            if getattr(self.norm, "bias", None) is not None:
                new_norm.norm.bias.copy_(self.norm.bias[start:end])
            if self.norm_type == "batch":
                new_norm.norm.running_mean.copy_(self.norm.running_mean[start:end])
                new_norm.norm.running_var.copy_(self.norm.running_var[start:end])
                new_norm.norm.num_batches_tracked.copy_(self.norm.num_batches_tracked)
        return new_norm

    def get_sub_norm(self, model_idx: int) -> "Normalize":
        return self.extract_single(model_idx)


def _state_dict_allclose(left: dict, right: dict) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for key in left:
        one = left[key]
        two = right[key]
        if isinstance(one, torch.Tensor) and isinstance(two, torch.Tensor):
            if one.dtype.is_floating_point or two.dtype.is_floating_point:
                if not torch.allclose(one, two):
                    return False
            elif not torch.equal(one, two):
                return False
        else:
            if one != two:
                return False
    return True


@torch.no_grad()
def pack_normalizes(norms: Sequence["Normalize"]) -> "Normalize":
    if not norms:
        raise ValueError("pack_normalizes expects at least one Normalize.")

    first = norms[0]
    if not isinstance(first, Normalize):
        raise TypeError(f"pack_normalizes expects Normalize instances, got {type(first)}.")
    if int(first.num_models) != 1:
        raise ValueError(f"pack_normalizes expects single-model Normalize, got num_models={first.num_models}.")

    in_channels = int(first.in_channels)
    norm_type = str(first.norm_type)
    training = bool(first.training)
    device = None
    dtype = None
    ref_weight = getattr(first.norm, "weight", None)
    if isinstance(ref_weight, torch.Tensor):
        device = ref_weight.device
        dtype = ref_weight.dtype

    for idx, norm in enumerate(norms[1:], start=1):
        if not isinstance(norm, Normalize):
            raise TypeError(f"pack_normalizes expects Normalize instances, got {type(norm)} at idx={idx}.")
        if int(norm.num_models) != 1:
            raise ValueError(
                f"pack_normalizes expects single-model Normalize, got num_models={norm.num_models} at idx={idx}."
            )
        if int(norm.in_channels) != in_channels or str(norm.norm_type) != norm_type:
            raise ValueError(
                f"pack_normalizes config mismatch at idx={idx}: "
                f"in_channels={int(norm.in_channels)}, norm_type={str(norm.norm_type)} "
                f"vs ({in_channels}, {norm_type})."
            )
        if bool(norm.training) != training:
            raise ValueError("pack_normalizes requires all modules to share the same training mode.")
        cur_weight = getattr(norm.norm, "weight", None)
        if isinstance(cur_weight, torch.Tensor):
            if device is None:
                device = cur_weight.device
                dtype = cur_weight.dtype
            elif cur_weight.device != device or cur_weight.dtype != dtype:
                raise ValueError(
                    f"pack_normalizes dtype/device mismatch at idx={idx}: "
                    f"device={cur_weight.device}, dtype={cur_weight.dtype} vs device={device}, dtype={dtype}."
                )

    packed = Normalize(in_channels, norm_type, num_models=len(norms))
    if device is not None:
        packed = packed.to(device=device, dtype=dtype)
    packed.requires_grad_(False)
    packed.train(training)

    if norm_type == "no":
        return packed

    if norm_type == "layer":
        if len(norms) == 1:
            packed.norm.load_state_dict(first.norm.state_dict())
            return packed
        for idx, norm in enumerate(norms):
            with torch.no_grad():
                packed.weight[idx].copy_(norm.norm.weight)
                packed.bias[idx].copy_(norm.norm.bias)
        return packed

    if norm_type == "rms":
        if len(norms) == 1:
            packed.norm.load_state_dict(first.norm.state_dict())
            return packed
        for idx, norm in enumerate(norms):
            with torch.no_grad():
                packed.weight[idx].copy_(norm.norm.weight)
        return packed

    if norm_type == "group":
        packed.norm.weight.zero_()
        packed.norm.bias.zero_()
        for idx, norm in enumerate(norms):
            start = idx * in_channels
            end = start + in_channels
            packed.norm.weight[start:end].copy_(norm.norm.weight)
            packed.norm.bias[start:end].copy_(norm.norm.bias)
        return packed

    if norm_type == "batch":
        packed.norm.weight.zero_()
        packed.norm.bias.zero_()
        ref_batches = norms[0].norm.num_batches_tracked
        packed.norm.num_batches_tracked.copy_(ref_batches)
        for idx, norm in enumerate(norms):
            if not torch.equal(norm.norm.num_batches_tracked, ref_batches):
                raise ValueError(
                    f"pack_normalizes requires identical BatchNorm num_batches_tracked; mismatch at idx={idx}."
                )
            start = idx * in_channels
            end = start + in_channels
            packed.norm.weight[start:end].copy_(norm.norm.weight)
            packed.norm.bias[start:end].copy_(norm.norm.bias)
            packed.norm.running_mean[start:end].copy_(norm.norm.running_mean)
            packed.norm.running_var[start:end].copy_(norm.norm.running_var)
        return packed

    raise ValueError(f"Unsupported norm_type={norm_type} for pack_normalizes.")


class ResnetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "group",
        activation_type: str = "swish",
        num_models: int = 1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = self.in_channels if out_channels is None else int(out_channels)
        self.num_models = int(num_models)
        self.activation_type = str(activation_type).strip().lower()

        self.norm1 = Normalize(self.in_channels, norm_type, num_models=self.num_models)
        self.linear1 = ParallelLinear(self.in_channels, self.out_channels, num_models=self.num_models)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = ParallelLinear(self.in_channels, self.out_channels, num_models=self.num_models)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = apply_activation(h, self.activation_type)
        h = self.linear1(h)
        x = self.nin_shortcut(x)
        return x + h

    def extract_single(self, model_idx: int) -> "ResnetBlock1D":
        _validate_model_index(model_idx, num_models=self.num_models)
        block = ResnetBlock1D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            norm_type=self.norm1.norm_type,
            activation_type=self.activation_type,
            num_models=1,
        )
        block.norm1 = self.norm1.extract_single(model_idx)
        block.linear1 = self.linear1.extract_single(model_idx)
        if isinstance(self.nin_shortcut, ParallelLinear):
            block.nin_shortcut = self.nin_shortcut.extract_single(model_idx)
        else:
            block.nin_shortcut = nn.Identity()
        block.train(self.training)
        return block


@torch.no_grad()
def pack_resnet_blocks(blocks: Sequence["ResnetBlock1D"]) -> "ResnetBlock1D":
    if not blocks:
        raise ValueError("pack_resnet_blocks expects at least one block.")

    first = blocks[0]
    if not isinstance(first, ResnetBlock1D):
        raise TypeError(f"pack_resnet_blocks expects ResnetBlock1D, got {type(first)}.")
    if int(first.num_models) != 1:
        raise ValueError(
            f"pack_resnet_blocks expects single-model blocks, got num_models={first.num_models}."
        )
    in_channels = int(first.in_channels)
    out_channels = int(first.out_channels)
    norm_type = str(first.norm1.norm_type)
    activation_type = str(first.activation_type)
    training = bool(first.training)
    first_linear = _resolve_single_linear(first.linear1)
    device = first_linear.weight.device
    dtype = first_linear.weight.dtype
    shortcut_is_linear = isinstance(first.nin_shortcut, (nn.Linear, ParallelLinear))

    for idx, block in enumerate(blocks[1:], start=1):
        if not isinstance(block, ResnetBlock1D):
            raise TypeError(f"pack_resnet_blocks expects ResnetBlock1D, got {type(block)} at idx={idx}.")
        if int(block.num_models) != 1:
            raise ValueError(
                f"pack_resnet_blocks expects single-model blocks, got num_models={block.num_models} at idx={idx}."
            )
        if (
            int(block.in_channels) != in_channels
            or int(block.out_channels) != out_channels
            or str(block.norm1.norm_type) != norm_type
            or str(block.activation_type) != activation_type
        ):
            raise ValueError(
                f"pack_resnet_blocks config mismatch at idx={idx}: "
                f"in={int(block.in_channels)}, out={int(block.out_channels)}, norm={str(block.norm1.norm_type)}, "
                f"activation={str(block.activation_type)} "
                f"vs ({in_channels}, {out_channels}, {norm_type}, {activation_type})."
            )
        if bool(block.training) != training:
            raise ValueError("pack_resnet_blocks requires all blocks to share the same training mode.")
        block_linear = _resolve_single_linear(block.linear1)
        if block_linear.weight.device != device or block_linear.weight.dtype != dtype:
            raise ValueError(
                f"pack_resnet_blocks dtype/device mismatch at idx={idx}: "
                f"device={block_linear.weight.device}, dtype={block_linear.weight.dtype} "
                f"vs device={device}, dtype={dtype}."
            )
        if isinstance(block.nin_shortcut, (nn.Linear, ParallelLinear)) != shortcut_is_linear:
            raise ValueError("pack_resnet_blocks requires all shortcuts to share the same type.")

    packed = ResnetBlock1D(
        in_channels=in_channels,
        out_channels=out_channels,
        norm_type=norm_type,
        activation_type=activation_type,
        num_models=len(blocks),
    ).to(device=device, dtype=dtype)
    packed.requires_grad_(False)
    packed.norm1 = pack_normalizes([block.norm1 for block in blocks])
    packed.linear1 = pack_parallel_linears([block.linear1 for block in blocks])
    if shortcut_is_linear:
        packed.nin_shortcut = pack_parallel_linears([block.nin_shortcut for block in blocks])
    else:
        packed.nin_shortcut = nn.Identity()
    packed.train(training)
    return packed


__all__ = [
    "Normalize",
    "ParallelLinear",
    "ResnetBlock1D",
    "apply_activation",
    "pack_normalizes",
    "pack_parallel_linears",
    "pack_resnet_blocks",
    "hard_swish",
    "swish",
]
