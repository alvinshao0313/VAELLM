from typing import Tuple

import torch


def build_block_data_loaders(
    stacked_data: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    block_indices = torch.arange(stacked_data.shape[0], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(stacked_data, block_indices)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, eval_loader


def reshape_blocks_for_codebook_dim(
    stacked_data: torch.Tensor,
    *,
    codebook_dim: int,
) -> torch.Tensor:
    target_dim = int(codebook_dim)
    if target_dim < 1:
        raise ValueError(f"codebook_dim must be >=1, got {target_dim}")
    if int(stacked_data.shape[-1]) == target_dim:
        return stacked_data
    num_models = int(stacked_data.shape[1])
    flat = stacked_data.permute(1, 0, 2).contiguous().view(num_models, -1)
    total_numel = int(flat.shape[1])
    if total_numel % target_dim != 0:
        raise ValueError(
            f"Cannot reshape residual blocks: total_numel_per_model={total_numel} not divisible by codebook_dim={target_dim}"
        )
    return flat.view(num_models, -1, target_dim).permute(1, 0, 2).contiguous()


def compute_stage_norm_stats(
    stage_data: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if stage_data.ndim != 3:
        raise ValueError(f"stage_data must be 3D [N_blocks, P, C], got shape={tuple(stage_data.shape)}")
    num_models = int(stage_data.shape[1])
    flat = stage_data.permute(1, 0, 2).contiguous().view(num_models, -1)
    mean = flat.mean(dim=1, keepdim=True)
    scale = flat.std(dim=1, keepdim=True).clamp_min(float(eps))
    return mean, scale


def apply_stage_norm(
    stage_data: torch.Tensor,
    *,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    num_blocks = int(stage_data.shape[0])
    num_models = int(stage_data.shape[1])
    codebook_dim = int(stage_data.shape[2])
    flat = stage_data.permute(1, 0, 2).contiguous().view(num_models, -1)
    norm_flat = (flat - mean) / scale
    return norm_flat.view(num_models, num_blocks, codebook_dim).permute(1, 0, 2).contiguous()


def restore_stage_norm(
    stage_data_norm: torch.Tensor,
    *,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    num_blocks = int(stage_data_norm.shape[0])
    num_models = int(stage_data_norm.shape[1])
    codebook_dim = int(stage_data_norm.shape[2])
    flat = stage_data_norm.permute(1, 0, 2).contiguous().view(num_models, -1)
    raw_flat = flat * scale + mean
    return raw_flat.view(num_models, num_blocks, codebook_dim).permute(1, 0, 2).contiguous()
