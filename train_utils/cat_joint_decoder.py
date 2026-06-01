import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import nn

from train_utils.cat_data_prep import gather_wa_mse_act_max_batch
from train_utils.train_args import create_optimizer
from train_utils.utils import extract_layer_idx


@dataclass
class JointStageRestorePlan:
    stage_src_idx_global: torch.Tensor
    stage_src_idx_flat: torch.Tensor


@dataclass
class JointRestorePlan:
    common_shape: Tuple[int, int, int]
    stage_plans: List[JointStageRestorePlan]

    def slice_blocks(self, block_idx: torch.Tensor) -> "JointRestorePlan":
        if block_idx.ndim != 1:
            raise ValueError(f"block_idx must be 1D, got shape={tuple(block_idx.shape)}")
        if block_idx.numel() <= 0:
            raise ValueError("block_idx cannot be empty.")
        num_blocks, num_models, codebook_dim = self.common_shape
        block_idx_cpu = block_idx.to(device="cpu", dtype=torch.long, non_blocking=False).contiguous()
        if int(block_idx_cpu.min().item()) < 0 or int(block_idx_cpu.max().item()) >= int(num_blocks):
            raise ValueError(
                f"block_idx out of range: min={int(block_idx_cpu.min().item())}, "
                f"max={int(block_idx_cpu.max().item())}, num_blocks={num_blocks}"
            )
        block_width = int(num_models) * int(codebook_dim)
        inverse_block = torch.full((int(num_blocks),), -1, dtype=torch.long)
        inverse_block[block_idx_cpu] = torch.arange(int(block_idx_cpu.numel()), dtype=torch.long)

        sliced_stage_plans: List[JointStageRestorePlan] = []
        for stage_plan in self.stage_plans:
            src_idx = stage_plan.stage_src_idx_global.index_select(0, block_idx_cpu)
            src_block = torch.div(src_idx, block_width, rounding_mode="floor")
            src_offset = torch.remainder(src_idx, block_width)
            local_src_block = inverse_block.index_select(0, src_block.reshape(-1)).view_as(src_block)
            if bool((local_src_block < 0).any().item()):
                raise ValueError("slice_blocks got source block outside selected block_idx.")
            local_src_idx = (local_src_block * block_width + src_offset).contiguous()
            sliced_stage_plans.append(
                JointStageRestorePlan(
                    stage_src_idx_global=local_src_idx.contiguous(),
                    stage_src_idx_flat=local_src_idx.reshape(-1).contiguous(),
                )
            )
        return JointRestorePlan(
            common_shape=(int(block_idx_cpu.numel()), int(num_models), int(codebook_dim)),
            stage_plans=sliced_stage_plans,
        )


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


def build_joint_restore_plan(
    *,
    common_shape: Tuple[int, int, int],
    all_stage_split_metas: Sequence[Sequence[object]],
    common_split_metas: Sequence[object],
    codebook_dim: int,
    convert_stage_to_common_fn: Callable[..., torch.Tensor],
) -> JointRestorePlan:
    if len(common_shape) != 3:
        raise ValueError(f"common_shape must be 3D, got {common_shape}")
    if len(all_stage_split_metas) == 0:
        raise ValueError("all_stage_split_metas cannot be empty.")
    num_blocks, num_models, block_dim = [int(v) for v in common_shape]
    total_numel = int(num_blocks) * int(num_models) * int(block_dim)
    stage_template = torch.arange(total_numel, dtype=torch.long).view(num_blocks, num_models, block_dim)
    expected_perm = torch.arange(total_numel, dtype=torch.long)
    stage_plans: List[JointStageRestorePlan] = []
    for stage_split_metas in all_stage_split_metas:
        stage_common = convert_stage_to_common_fn(
            stage_stacked_data=stage_template,
            stage_split_metas=stage_split_metas,
            common_split_metas=common_split_metas,
            codebook_dim=int(codebook_dim),
        )
        if tuple(stage_common.shape) != (num_blocks, num_models, block_dim):
            raise ValueError(
                f"joint restore plan shape mismatch: got {tuple(stage_common.shape)}, "
                f"expected {(num_blocks, num_models, block_dim)}"
            )
        src_idx_flat = stage_common.reshape(-1).to(dtype=torch.long).contiguous()
        if int(src_idx_flat.min().item()) < 0 or int(src_idx_flat.max().item()) >= int(total_numel):
            raise ValueError("joint restore plan index out of range.")
        if not torch.equal(torch.sort(src_idx_flat).values, expected_perm):
            raise ValueError("joint restore plan index is not bijective.")
        src_idx_global = src_idx_flat.view(num_blocks, num_models, block_dim).contiguous()
        stage_plans.append(
            JointStageRestorePlan(
                stage_src_idx_global=src_idx_global,
                stage_src_idx_flat=src_idx_flat,
            )
        )
    return JointRestorePlan(
        common_shape=(num_blocks, num_models, block_dim),
        stage_plans=stage_plans,
    )


def apply_joint_restore_plan_full(
    *,
    stage_stacked_data: torch.Tensor,
    stage_src_idx_flat: torch.Tensor,
    common_shape: Tuple[int, int, int],
) -> torch.Tensor:
    flat = stage_stacked_data.reshape(-1)
    idx = stage_src_idx_flat
    if idx.device != flat.device:
        idx = idx.to(device=flat.device, non_blocking=True)
    common_flat = torch.take(flat, idx)
    return common_flat.view(int(common_shape[0]), int(common_shape[1]), int(common_shape[2]))


def finetune_stage_decoders(
    *,
    group_tag: str,
    shared_stage_args,
    joint_steps: int,
    joint_lr: float,
    joint_decoder_batch_size: Optional[int],
    train_device: str,
    train_dtype: torch.dtype,
    log_every: int,
    eval_every: int,
    eval_blocks: int,
    codebook_dim: int,
    recon_loss_type: str,
    intra_part_sort_mode: str,
    target_common_result,
    all_stage_bits: Sequence[torch.Tensor],
    all_stage_decoders: Sequence[Sequence[nn.Module]],
    all_stage_split_metas: Sequence[Sequence[object]],
    convert_stage_to_common_fn: Callable[..., torch.Tensor],
    recon_loss_fn: Callable[..., torch.Tensor],
    logger,
    shuffle_seed: Optional[int] = None,
) -> List[List[nn.Module]]:
    from litebsq.autoencoder import pack_decoders

    if int(joint_steps) <= 0:
        return [list(stage_decoders) for stage_decoders in all_stage_decoders]
    if len(all_stage_bits) < 2:
        return [list(stage_decoders) for stage_decoders in all_stage_decoders]

    resolved_joint_batch_size = None if joint_decoder_batch_size is None else int(joint_decoder_batch_size)
    if resolved_joint_batch_size is not None and resolved_joint_batch_size < 1:
        raise ValueError(f"[{group_tag}] joint_decoder_batch_size must be >= 1 or none.")
    resolved_sort_mode = str(intra_part_sort_mode).strip().lower()
    if resolved_joint_batch_size is not None and resolved_sort_mode != "none":
        raise ValueError(
            f"[{group_tag}] joint_decoder_batch_size is only supported when intra_part_sort_mode=none, "
            f"got {resolved_sort_mode}."
        )

    target_common = target_common_result.stacked_data.to(device=train_device, dtype=train_dtype, non_blocking=True)
    joint_restore_plan = build_joint_restore_plan(
        common_shape=tuple(int(v) for v in target_common.shape),
        all_stage_split_metas=all_stage_split_metas,
        common_split_metas=target_common_result.split_metas,
        codebook_dim=int(codebook_dim),
        convert_stage_to_common_fn=convert_stage_to_common_fn,
    )
    if len(joint_restore_plan.stage_plans) != len(all_stage_bits):
        raise ValueError(
            f"[{group_tag}] joint restore plan stage count mismatch: "
            f"plan={len(joint_restore_plan.stage_plans)} vs bits={len(all_stage_bits)}"
        )
    use_patch_joint = resolved_joint_batch_size is not None
    resolved_recon_loss = str(recon_loss_type).strip().lower()
    act_max_full = None
    if resolved_recon_loss == "wa_mse" and not use_patch_joint:
        full_block_idx = torch.arange(target_common.shape[0], dtype=torch.long)
        act_max_full = gather_wa_mse_act_max_batch(
            block_idx_batch=full_block_idx,
            part_metas=target_common_result.part_metas,
            codebook_dim=int(codebook_dim),
            train_device=train_device,
            target_dtype=train_dtype,
        )

    packed_stage_decoders: List[nn.Module] = []
    stage_bits_on_device: List[torch.Tensor] = []
    stage_src_idx_flat_on_device: List[torch.Tensor] = []
    for stage_bits_cpu, stage_decoders in zip(all_stage_bits, all_stage_decoders):
        packed_decoder = pack_decoders(list(stage_decoders)).to(train_device)
        packed_decoder.requires_grad_(True)
        packed_decoder.train()
        packed_stage_decoders.append(packed_decoder)
        stage_bits_on_device.append(stage_bits_cpu.to(device=train_device, non_blocking=True))
    if not use_patch_joint:
        for stage_plan in joint_restore_plan.stage_plans:
            stage_src_idx_flat_on_device.append(stage_plan.stage_src_idx_flat.to(device=train_device, non_blocking=True))

    joint_train_loader = None
    joint_train_iter = None
    if use_patch_joint:
        joint_train_loader = build_block_index_loader(
            num_blocks=int(target_common.shape[0]),
            batch_size=int(resolved_joint_batch_size),
            shuffle_seed=shuffle_seed,
        )
        joint_train_iter = iter(joint_train_loader)

    params = []
    for packed_decoder in packed_stage_decoders:
        params.extend([param for param in packed_decoder.parameters() if param.requires_grad])
    if not params:
        raise RuntimeError(f"[{group_tag}] joint decoder fine-tune has no trainable parameters.")
    optimizer = create_optimizer(params, shared_stage_args, float(joint_lr))
    lr_scheduler = None
    lr_scheduler_name = str(getattr(shared_stage_args, "lr_scheduler", "none"))
    if lr_scheduler_name != "none":
        import transformers

        lr_scheduler = transformers.get_scheduler(
            lr_scheduler_name,
            optimizer,
            num_warmup_steps=int(getattr(shared_stage_args, "lr_warmup_steps", 0)),
            num_training_steps=int(joint_steps),
        )

    full_recon_loss_before, full_recon_blocks = compute_joint_decoder_recon_loss(
        group_tag=group_tag,
        packed_stage_decoders=packed_stage_decoders,
        stage_bits_on_device=stage_bits_on_device,
        joint_restore_plan=joint_restore_plan,
        target_common=target_common,
        target_common_result=target_common_result,
        codebook_dim=int(codebook_dim),
        train_device=train_device,
        train_dtype=train_dtype,
        recon_loss_type=recon_loss_type,
        sort_mode=resolved_sort_mode,
        restore_fn=apply_joint_restore_plan_full,
        loss_fn=recon_loss_fn,
        max_blocks=None,
    )
    logger.info(
        "[%s/joint] full_recon_loss_before=%.6e blocks=%d",
        group_tag,
        full_recon_loss_before,
        full_recon_blocks,
    )

    start = time.time()
    for step in range(int(joint_steps)):
        target_batch = target_common
        active_restore_plan = joint_restore_plan
        active_stage_bits = stage_bits_on_device
        active_stage_src_idx_flat = stage_src_idx_flat_on_device
        act_max_batch = act_max_full
        if use_patch_joint:
            if joint_train_loader is None or joint_train_iter is None:
                raise RuntimeError(f"[{group_tag}] joint patch DataLoader is not initialized.")
            joint_train_iter, block_idx_cpu, block_idx_device = next_block_index_batch(
                iterator=joint_train_iter,
                loader=joint_train_loader,
                device=train_device,
            )
            target_batch = target_common.index_select(0, block_idx_device)
            active_restore_plan = joint_restore_plan.slice_blocks(block_idx_cpu)
            active_stage_bits = [
                stage_bits.index_select(0, block_idx_device)
                for stage_bits in stage_bits_on_device
            ]
            active_stage_src_idx_flat = [
                stage_plan.stage_src_idx_flat.to(device=train_device, non_blocking=True)
                for stage_plan in active_restore_plan.stage_plans
            ]
            if resolved_recon_loss == "wa_mse":
                act_max_batch = gather_wa_mse_act_max_batch(
                    block_idx_batch=block_idx_cpu,
                    part_metas=target_common_result.part_metas,
                    codebook_dim=int(codebook_dim),
                    train_device=train_device,
                    target_dtype=train_dtype,
                )

        optimizer.zero_grad(set_to_none=True)
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
                        f"[{group_tag}] joint decoder output shape mismatch: "
                        f"out={tuple(stage_out.shape)} vs common={tuple(active_restore_plan.common_shape)}"
                    )
                stage_common = stage_out.to(dtype=train_dtype)
            else:
                stage_common = apply_joint_restore_plan_full(
                    stage_stacked_data=stage_out,
                    stage_src_idx_flat=stage_src_idx_flat,
                    common_shape=active_restore_plan.common_shape,
                ).to(dtype=train_dtype)
            total_recon = stage_common if total_recon is None else (total_recon + stage_common)
        if total_recon is None:
            raise RuntimeError(f"[{group_tag}] joint decoder fine-tune produced no reconstruction.")
        loss = recon_loss_fn(
            recon_loss_type=recon_loss_type,
            x_recon=total_recon,
            x=target_batch,
            act_max=act_max_batch,
        )
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if log_every > 0 and (step + 1) % int(log_every) == 0:
            speed = (time.time() - start) / int(log_every)
            logger.info(
                "[%s/joint] mode=%s step=%d/%d loss=%.4e speed=%.4fs/it",
                group_tag,
                "patch" if use_patch_joint else "full",
                step + 1,
                int(joint_steps),
                float(loss.detach().float().item()),
                speed,
            )
            start = time.time()

        if eval_every > 0 and (step + 1) % int(eval_every) == 0:
            eval_recon_loss, eval_block_count = compute_joint_decoder_recon_loss(
                group_tag=group_tag,
                packed_stage_decoders=packed_stage_decoders,
                stage_bits_on_device=stage_bits_on_device,
                joint_restore_plan=joint_restore_plan,
                target_common=target_common,
                target_common_result=target_common_result,
                codebook_dim=int(codebook_dim),
                train_device=train_device,
                train_dtype=train_dtype,
                recon_loss_type=recon_loss_type,
                sort_mode=resolved_sort_mode,
                restore_fn=apply_joint_restore_plan_full,
                loss_fn=recon_loss_fn,
                max_blocks=int(eval_blocks),
            )
            logger.info(
                "[%s/joint] eval@step=%d recon_loss=%.6e blocks=%d",
                group_tag,
                step + 1,
                eval_recon_loss,
                eval_block_count,
            )

    full_recon_loss_after, full_recon_blocks_after = compute_joint_decoder_recon_loss(
        group_tag=group_tag,
        packed_stage_decoders=packed_stage_decoders,
        stage_bits_on_device=stage_bits_on_device,
        joint_restore_plan=joint_restore_plan,
        target_common=target_common,
        target_common_result=target_common_result,
        codebook_dim=int(codebook_dim),
        train_device=train_device,
        train_dtype=train_dtype,
        recon_loss_type=recon_loss_type,
        sort_mode=resolved_sort_mode,
        restore_fn=apply_joint_restore_plan_full,
        loss_fn=recon_loss_fn,
        max_blocks=None,
    )
    loss_delta = full_recon_loss_after - full_recon_loss_before
    loss_ratio = full_recon_loss_after / full_recon_loss_before if full_recon_loss_before != 0.0 else float("nan")
    logger.info(
        "[%s/joint] full_recon_loss_after=%.6e blocks=%d delta=%.6e ratio=%.6f",
        group_tag,
        full_recon_loss_after,
        full_recon_blocks_after,
        loss_delta,
        loss_ratio,
    )

    updated_stage_decoders: List[List[nn.Module]] = []
    num_models = int(target_common.shape[1])
    for packed_decoder in packed_stage_decoders:
        packed_decoder.eval()
        updated_stage_decoders.append(
            [packed_decoder.get_sub_decoder(model_idx).to(device="cpu") for model_idx in range(num_models)]
        )

    del target_common, act_max_full, stage_bits_on_device, packed_stage_decoders, optimizer, stage_src_idx_flat_on_device
    if use_patch_joint:
        del joint_train_loader, joint_train_iter
    if lr_scheduler is not None:
        del lr_scheduler
    torch.cuda.empty_cache()
    return updated_stage_decoders


def _build_joint_target_subset(
    target_common_result,
    *,
    linear_start: int,
    linear_end: int,
    model_start: int,
    model_end: int,
):
    return SimpleNamespace(
        stacked_data=target_common_result.stacked_data[:, model_start:model_end, :].contiguous(),
        part_metas=list(target_common_result.part_metas[model_start:model_end]),
        split_metas=list(target_common_result.split_metas[linear_start:linear_end]),
    )


def finetune_stage_decoders_in_subgroups(
    *,
    group_tag: str,
    group_refs: Sequence[object],
    shared_stage_args,
    joint_steps: int,
    joint_lr: float,
    joint_group_size: int,
    joint_decoder_batch_size: Optional[int],
    train_device: str,
    train_dtype: torch.dtype,
    log_every: int,
    eval_every: int,
    eval_blocks: int,
    codebook_dim: int,
    recon_loss_type: str,
    intra_part_sort_mode: str,
    target_common_result,
    all_stage_bits: Sequence[torch.Tensor],
    all_stage_decoders: Sequence[Sequence[nn.Module]],
    all_stage_split_metas: Sequence[Sequence[object]],
    parts_per_linear: int,
    convert_stage_to_common_fn: Callable[..., torch.Tensor],
    recon_loss_fn: Callable[..., torch.Tensor],
    logger,
    shuffle_seed: Optional[int] = None,
) -> List[List[nn.Module]]:
    num_linears = int(len(group_refs))
    if num_linears <= 0:
        return [list(stage_decoders) for stage_decoders in all_stage_decoders]
    resolved_joint_group_size = max(1, min(int(joint_group_size), num_linears))
    if resolved_joint_group_size >= num_linears:
        return finetune_stage_decoders(
            group_tag=group_tag,
            shared_stage_args=shared_stage_args,
            joint_steps=joint_steps,
            joint_lr=joint_lr,
            joint_decoder_batch_size=joint_decoder_batch_size,
            train_device=train_device,
            train_dtype=train_dtype,
            log_every=log_every,
            eval_every=eval_every,
            eval_blocks=eval_blocks,
            codebook_dim=codebook_dim,
            recon_loss_type=recon_loss_type,
            intra_part_sort_mode=intra_part_sort_mode,
            target_common_result=target_common_result,
            all_stage_bits=all_stage_bits,
            all_stage_decoders=all_stage_decoders,
            all_stage_split_metas=all_stage_split_metas,
            convert_stage_to_common_fn=convert_stage_to_common_fn,
            recon_loss_fn=recon_loss_fn,
            logger=logger,
            shuffle_seed=shuffle_seed,
        )

    resolved_recon_loss = str(recon_loss_type).strip().lower()
    if resolved_recon_loss in {"cosine", "relative_l1"}:
        raise ValueError(
            f"[{group_tag}] joint_decoder_group_size={resolved_joint_group_size} is not supported "
            f"with recon_loss_type={resolved_recon_loss}. Set --joint_decoder_group_size to the full group size."
        )

    logger.info(
        "[%s/joint] split full-batch group into subgroups: joint_decoder_group_size=%d linears=%d",
        group_tag,
        resolved_joint_group_size,
        num_linears,
    )

    updated_stage_decoders: List[List[nn.Module]] = [list(stage_decoders) for stage_decoders in all_stage_decoders]
    for linear_start in range(0, num_linears, resolved_joint_group_size):
        linear_end = min(linear_start + resolved_joint_group_size, num_linears)
        model_start = int(linear_start) * int(parts_per_linear)
        model_end = int(linear_end) * int(parts_per_linear)
        subgroup_target_common_result = _build_joint_target_subset(
            target_common_result,
            linear_start=linear_start,
            linear_end=linear_end,
            model_start=model_start,
            model_end=model_end,
        )
        subgroup_stage_bits = [
            stage_bits[:, model_start:model_end, :].contiguous()
            for stage_bits in all_stage_bits
        ]
        subgroup_stage_decoders = [
            list(stage_decoders[model_start:model_end])
            for stage_decoders in updated_stage_decoders
        ]
        subgroup_stage_split_metas = [
            list(stage_split_metas[linear_start:linear_end])
            for stage_split_metas in all_stage_split_metas
        ]
        start_layer_idx = extract_layer_idx(group_refs[linear_start].name)
        end_layer_idx = extract_layer_idx(group_refs[linear_end - 1].name)
        if start_layer_idx is not None and end_layer_idx is not None:
            subgroup_tag = f"{group_tag}.subL{start_layer_idx}-{end_layer_idx}"
        else:
            subgroup_tag = f"{group_tag}.sub{linear_start}-{linear_end - 1}"
        subgroup_updated_decoders = finetune_stage_decoders(
            group_tag=subgroup_tag,
            shared_stage_args=shared_stage_args,
            joint_steps=joint_steps,
            joint_lr=joint_lr,
            joint_decoder_batch_size=joint_decoder_batch_size,
            train_device=train_device,
            train_dtype=train_dtype,
            log_every=log_every,
            eval_every=eval_every,
            eval_blocks=eval_blocks,
            codebook_dim=codebook_dim,
            recon_loss_type=recon_loss_type,
            intra_part_sort_mode=intra_part_sort_mode,
            target_common_result=subgroup_target_common_result,
            all_stage_bits=subgroup_stage_bits,
            all_stage_decoders=subgroup_stage_decoders,
            all_stage_split_metas=subgroup_stage_split_metas,
            convert_stage_to_common_fn=convert_stage_to_common_fn,
            recon_loss_fn=recon_loss_fn,
            logger=logger,
            shuffle_seed=None if shuffle_seed is None else int(shuffle_seed) + int(linear_start),
        )
        for stage_idx in range(len(updated_stage_decoders)):
            updated_stage_decoders[stage_idx][model_start:model_end] = subgroup_updated_decoders[stage_idx]

    return updated_stage_decoders
