import torch
import torch.nn as nn
from torch import Tensor


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


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
            layer.load_state_dict(self.linear.state_dict())
            return layer

        start = model_idx * self.out_features
        end = (model_idx + 1) * self.out_features
        layer = nn.Linear(self.in_features, self.out_features, bias=(self.conv.bias is not None))
        with torch.no_grad():
            layer.weight.copy_(self.conv.weight[start:end, :, 0])
            if self.conv.bias is not None:
                layer.bias.copy_(self.conv.bias[start:end])
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


class Normalize(nn.Module):
    def __init__(self, in_channels: int, norm_type: str = "group", num_models: int = 1):
        super().__init__()
        self.num_models = int(num_models)
        self.in_channels = int(in_channels)
        self.norm_type = str(norm_type)

        if self.norm_type not in {"group", "batch", "layer", "no"}:
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
            self.norm = nn.LayerNorm(self.in_channels)
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
        if self.norm_type in {"no", "layer"}:
            return self.norm(x)

        flat = self._flatten_parallel(x)
        out = self.norm(flat)
        if x.dim() == 2 and self.num_models == 1:
            return out
        return out.view(int(x.shape[0]), self.num_models, self.in_channels)

    def extract_single(self, model_idx: int) -> "Normalize":
        _validate_model_index(model_idx, num_models=self.num_models)
        new_norm = Normalize(self.in_channels, self.norm_type, num_models=1)

        if self.norm_type == "no":
            return new_norm

        if self.norm_type == "layer":
            new_norm.norm.load_state_dict(self.norm.state_dict())
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


class ResnetBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "group", num_models: int = 1):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = self.in_channels if out_channels is None else int(out_channels)
        self.num_models = int(num_models)

        self.norm1 = Normalize(self.in_channels, norm_type, num_models=self.num_models)
        self.linear1 = ParallelLinear(self.in_channels, self.out_channels, num_models=self.num_models)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = ParallelLinear(self.in_channels, self.out_channels, num_models=self.num_models)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = swish(h)
        h = self.linear1(h)
        x = self.nin_shortcut(x)
        return x + h

    def extract_single(self, model_idx: int) -> "ResnetBlock1D":
        _validate_model_index(model_idx, num_models=self.num_models)
        block = ResnetBlock1D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            norm_type=self.norm1.norm_type,
            num_models=1,
        )
        block.norm1 = self.norm1.extract_single(model_idx)
        block.linear1 = self.linear1.extract_single(model_idx)
        if isinstance(self.nin_shortcut, ParallelLinear):
            block.nin_shortcut = self.nin_shortcut.extract_single(model_idx)
        else:
            block.nin_shortcut = nn.Identity()
        return block


__all__ = [
    "Normalize",
    "ParallelLinear",
    "ResnetBlock1D",
    "swish",
]
