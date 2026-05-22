from typing import Callable, Iterator, Optional, Sequence, Tuple

import torch


def build_block_index_loader(
    *,
    num_blocks: int,
    batch_size: int,
    shuffle_seed: Optional[int],
) -> torch.utils.data.DataLoader:
    if int(num_blocks) < 1:
        raise ValueError(f"num_blocks must be >= 1, got {int(num_blocks)}.")
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {int(batch_size)}.")
    block_indices = torch.arange(int(num_blocks), dtype=torch.long)
    generator = None
    if shuffle_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(shuffle_seed))
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(block_indices),
        batch_size=min(int(batch_size), int(num_blocks)),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=False,
    )


def next_block_index_batch(
    *,
    iterator: Iterator[Tuple[torch.Tensor]],
    loader: torch.utils.data.DataLoader,
    device: str,
) -> Tuple[Iterator[Tuple[torch.Tensor]], torch.Tensor, torch.Tensor]:
    try:
        (block_idx_batch,) = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        (block_idx_batch,) = next(iterator)
    block_idx_cpu = block_idx_batch.to(device="cpu", dtype=torch.long, non_blocking=False).contiguous()
    block_idx_device = block_idx_cpu.to(device=device, non_blocking=True)
    return iterator, block_idx_cpu, block_idx_device


def compute_joint_decoder_recon_loss(
    *,
    group_tag: str,
    packed_stage_decoders: Sequence[torch.nn.Module],
    stage_bits_on_device: Sequence[torch.Tensor],
    joint_restore_plan,
    target_common: torch.Tensor,
    target_common_result,
    codebook_dim: int,
    train_device: str,
    train_dtype: torch.dtype,
    recon_loss_type: str,
    sort_mode: str,
    restore_fn: Callable[..., torch.Tensor],
    loss_fn: Callable[..., torch.Tensor],
    max_blocks: Optional[int],
) -> Tuple[float, int]:
    num_blocks = int(target_common.shape[0])
    if num_blocks < 1:
        raise ValueError(f"[{group_tag}] joint eval target has no blocks.")

    if max_blocks is None:
        block_count = num_blocks
        active_target = target_common
        active_restore_plan = joint_restore_plan
        active_stage_bits = list(stage_bits_on_device)
    else:
        eval_blocks = min(max(1, int(max_blocks)), num_blocks)
        block_idx_cpu = torch.arange(eval_blocks, dtype=torch.long)
        block_idx_device = block_idx_cpu.to(device=train_device, non_blocking=True)
        block_count = int(block_idx_cpu.numel())
        active_target = target_common.index_select(0, block_idx_device)
        active_restore_plan = joint_restore_plan.slice_blocks(block_idx_cpu)
        active_stage_bits = [
            stage_bits.index_select(0, block_idx_device)
            for stage_bits in stage_bits_on_device
        ]
    resolved_sort_mode = str(sort_mode).strip().lower()
    if resolved_sort_mode == "none":
        active_stage_src_idx_flat = [None for _stage_plan in active_restore_plan.stage_plans]
    else:
        active_stage_src_idx_flat = [
            stage_plan.stage_src_idx_flat.to(device=train_device, non_blocking=True)
            for stage_plan in active_restore_plan.stage_plans
        ]

    act_max = None
    if str(recon_loss_type).strip().lower() == "wa_mse":
        from train_utils.cat_data_prep import gather_wa_mse_act_max_batch

        block_idx_cpu = torch.arange(num_blocks, dtype=torch.long) if max_blocks is None else block_idx_cpu
        act_max = gather_wa_mse_act_max_batch(
            block_idx_batch=block_idx_cpu,
            part_metas=target_common_result.part_metas,
            codebook_dim=int(codebook_dim),
            train_device=train_device,
            target_dtype=train_dtype,
        )

    was_training = [bool(decoder.training) for decoder in packed_stage_decoders]
    try:
        for packed_decoder in packed_stage_decoders:
            packed_decoder.eval()
        with torch.no_grad():
            total_recon = None
            for packed_decoder, stage_bits, stage_src_idx_flat in zip(
                packed_stage_decoders, active_stage_bits, active_stage_src_idx_flat
            ):
                param = next(packed_decoder.parameters(), None)
                decode_dtype = param.dtype if param is not None else train_dtype
                stage_out = packed_decoder(stage_bits.to(dtype=decode_dtype))
                if resolved_sort_mode == "none":
                    if tuple(stage_out.shape) != tuple(active_restore_plan.common_shape):
                        raise RuntimeError(
                            f"[{group_tag}] joint eval decoder output shape mismatch: "
                            f"out={tuple(stage_out.shape)} vs common={tuple(active_restore_plan.common_shape)}"
                        )
                    stage_common = stage_out.to(dtype=train_dtype)
                else:
                    stage_common = restore_fn(
                        stage_stacked_data=stage_out,
                        stage_src_idx_flat=stage_src_idx_flat,
                        common_shape=active_restore_plan.common_shape,
                    ).to(dtype=train_dtype)
                total_recon = stage_common if total_recon is None else (total_recon + stage_common)
            if total_recon is None:
                raise RuntimeError(f"[{group_tag}] joint eval produced no reconstruction.")
            loss = loss_fn(
                recon_loss_type=recon_loss_type,
                x_recon=total_recon,
                x=active_target,
                act_max=act_max,
            )
            return float(loss.detach().float().cpu().item()), int(block_count)
    finally:
        for packed_decoder, training in zip(packed_stage_decoders, was_training):
            packed_decoder.train(training)
