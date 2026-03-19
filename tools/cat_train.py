import os
import sys
import time
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch import nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train_utils.train_args import (
    process_cat_train_args,
    create_optimizer,
    resolve_autoencoder_arch_args,
    resolve_codebook_int_for_category,
    resolve_intra_parallel_for_category,
    resolve_stage_value,
    resolve_skip_layer_matches,
)
from train_utils.cat_data_prep import (
    LinearPrepRef,
    format_intra_part_sort_mode,
    gather_wa_mse_act_max_batch,
    load_activation_weight_dict,
    normalize_intra_part_sort_mode,
    prepare_group_weight_data,
    resolve_intra_parallel,
)
from train_utils.activation_utils import (
    ActivationCalibrationCache,
    collect_act_max_for_linears,
)
from train_utils.model_checkpoint_io import (
    _build_run_output_dir,
    _safe_path_token,
    save_model_checkpoint,
)
from train_utils.utils import (
    LinearRef,
    clone_namespace as _clone_namespace,
    collect_linears as _collect_linears,
    extract_layer_idx as _extract_layer_idx,
    format_intra_parallel_desc as _format_intra_parallel_desc,
    format_namespace as _format_namespace,
    get_logger,
    resolve_category_order as _resolve_category_order,
    set_seed,
    split_csv as _split_csv,
)


log = get_logger("linear_by_category")


def _fuse_q_scale_linear(linear: nn.Linear, q_scale: float) -> None:
    with torch.no_grad():
        weight = linear.weight.data
        bias_delta = -q_scale * weight.sum(dim=1)
        weight.mul_(q_scale * 2)
        if linear.bias is not None:
            linear.bias.data.add_(bias_delta)
        else:
            linear.bias = nn.Parameter(bias_delta)


def _fuse_q_scale_into_decoder(decoder: nn.Module, q_scale: float) -> None:
    if hasattr(decoder, "_fuse_q_scale"):
        decoder._fuse_q_scale(float(q_scale))
        return

    # 回退逻辑: 没有 Decoder._fuse_q_scale 时直接融合到第一层线性。
    decoder_type = str(getattr(decoder, "decoder_type"))
    if decoder_type == "linear":
        _fuse_q_scale_linear(decoder.linear, q_scale)
    elif decoder_type in {"symmetric", "asymmetric"}:
        _fuse_q_scale_linear(decoder.linear_in, q_scale)


def _fuse_norm_into_decoder(decoder: nn.Module, mean: float, std: float) -> None:
    decoder_type = str(getattr(decoder, "decoder_type"))
    if decoder_type == "linear":
        last = decoder.linear
    elif decoder_type in {"symmetric", "asymmetric"}:
        last = decoder.linear_out
    else:
        raise ValueError(f"Unsupported decoder_type={decoder_type} for norm fusion")

    if not isinstance(last, nn.Linear):
        raise TypeError(f"Expected nn.Linear as last layer, got {type(last)}")

    with torch.no_grad():
        last.weight.mul_(std)
        if last.bias is None:
            last.bias = nn.Parameter(torch.zeros(last.out_features, device=last.weight.device, dtype=last.weight.dtype))
        last.bias.mul_(std).add_(mean)


def _build_block_data_loaders(
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


def _reshape_blocks_for_codebook_dim(
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


def _contains_stage_choice(value, target: str) -> bool:
    target_norm = str(target).strip().lower()
    if isinstance(value, (list, tuple)):
        return any(str(v).strip().lower() == target_norm for v in value)
    return str(value).strip().lower() == target_norm


def _compute_stage_norm_stats(
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


def _apply_stage_norm(
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


def _restore_stage_norm(
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


def _eval_ppl_after_category(model: nn.Module, vae_args, ppl_limit: int, category: str, eval_device: str = "cuda") -> None:
    from train_utils.eval_utils import calculate_ppl

    log.info("开始类别 %s 的 PPL 评估...", category)
    model.eval()
    model.to(eval_device)
    with torch.no_grad():
        setattr(vae_args, "limit", int(ppl_limit))
        ppl_result = calculate_ppl(model, vae_args)
    model.to("cpu")
    torch.cuda.empty_cache()
    log.info("类别 %s 训练后 PPL: %.2f", category, float(ppl_result.get("wiki_ppl", float("nan"))))


def _train_group_vae_and_replace(
    *,
    model: nn.Module,
    group_refs: Sequence[LinearRef],
    group_tag: str,
    vae_args,
    training_args,
    train_device: str,
    convert_device: str,
    do_convert: bool,
    steps: Union[int, Sequence[int]],
    batch_size: int,
    log_every: int,
    eval_every: int,
    eval_blocks: int,
    output_dir: str,
    intra_parallel,
    intra_part_sort_mode: Union[str, Sequence[str]],
    skip_layer_keys: Optional[Set[Tuple[int, str]]] = None,
    activation_weight_by_linear: Optional[Dict[str, torch.Tensor]] = None,
    wa_mse_runtime: Optional[Dict[str, object]] = None,
    outlier_protect_ratio: float = 0.0,
    outlier_protect_axis: str = "input",
) -> None:
    from litebsq.llm_vae import MultiLayerVAE
    from litebsq.vae_linear import VAELinear
    from litebsq.bsq_linear import set_module_by_name

    # 根据训练参数选择输入精度。
    if bool(getattr(training_args, "bf16", False)):
        train_dtype = torch.bfloat16
    elif bool(getattr(training_args, "fp16", False)):
        train_dtype = torch.float16
    else:
        train_dtype = torch.float32

    residual_stages = int(getattr(vae_args, "residual_stages", 1))
    if residual_stages < 1:
        raise ValueError(f"residual_stages must be >= 1, got {residual_stages}")
    if len(group_refs) == 0:
        raise ValueError(f"[{group_tag}] group_refs cannot be empty.")
    group_category = str(group_refs[0].category)

    stage0_sort_mode = resolve_stage_value(intra_part_sort_mode, 0, arg_name="--intra_part_sort_mode")
    stage0_codebook_dim = resolve_codebook_int_for_category(
        resolve_stage_value(getattr(vae_args, "codebook_dim"), 0, arg_name="--codebook_dim"),
        group_category,
        arg_name="codebook_dim",
    )
    stage0_recon_loss = str(
        resolve_stage_value(getattr(vae_args, "recon_loss_type", "mse"), 0, arg_name="--recon_loss_type")
    ).strip().lower()
    use_wa_mse_loss = any(
        str(resolve_stage_value(getattr(vae_args, "recon_loss_type", "mse"), i, arg_name="--recon_loss_type")).strip().lower()
        == "wa_mse"
        for i in range(residual_stages)
    )
    row_parts, col_parts = resolve_intra_parallel(intra_parallel)
    parts_per_linear = int(row_parts) * int(col_parts)
    effective_activation_weight = activation_weight_by_linear
    if use_wa_mse_loss and wa_mse_runtime is not None:
        if bool(wa_mse_runtime.get("dynamic", False)):
            calib_device = str(wa_mse_runtime.get("device") or train_device)
            linear_items = [(r.name, r.module) for r in group_refs]
            dynamic_act_max, new_cache = collect_act_max_for_linears(
                model=model,
                linear_items=linear_items,
                model_path=str(wa_mse_runtime["model_path"]),
                access_token=wa_mse_runtime.get("access_token"),
                dataset=str(wa_mse_runtime.get("dataset", "wikitext2")),
                nsamples=int(wa_mse_runtime.get("nsamples", 512)),
                seqlen=int(wa_mse_runtime.get("seqlen", 512)),
                seed=int(wa_mse_runtime.get("seed", 0)),
                device=calib_device,
                cache=wa_mse_runtime.get("cache"),  # type: ignore[arg-type]
                log_every=int(wa_mse_runtime.get("log_every", 0)),
                logger=log,
            )
            wa_mse_runtime["cache"] = new_cache
            effective_activation_weight = dynamic_act_max
            log.info(
                "[%s] refreshed act_max from current model (linears=%d, dataset=%s, nsamples=%d, seqlen=%d).",
                group_tag,
                len(dynamic_act_max),
                str(wa_mse_runtime.get("dataset", "wikitext2")),
                int(wa_mse_runtime.get("nsamples", 512)),
                int(wa_mse_runtime.get("seqlen", 512)),
            )
        elif effective_activation_weight is None:
            static_dict = wa_mse_runtime.get("static_dict")
            if isinstance(static_dict, dict):
                effective_activation_weight = static_dict

    prep_refs = [
        LinearPrepRef(
            name=r.name,
            weight=r.module.weight,
            in_features=int(r.module.in_features),
            out_features=int(r.module.out_features),
            transpose=bool(r.transpose),
        )
        for r in group_refs
    ]
    prep_result = prepare_group_weight_data(
        group_refs=prep_refs,
        intra_parallel=(row_parts, col_parts),
        codebook_dim=int(stage0_codebook_dim),
        batch_size=int(batch_size),
        # 多阶残差独立 norm：这里保持原始域，后续在每个 stage 内单独做标准化。
        normalize_weight=False,
        recon_loss_type="wa_mse" if use_wa_mse_loss else stage0_recon_loss,
        activation_weight_by_linear=effective_activation_weight,
        train_device=train_device,
        intra_part_sort_mode=stage0_sort_mode,
        outlier_protect_ratio=float(outlier_protect_ratio),
        outlier_protect_axis=str(outlier_protect_axis),
    )
    num_models = int(prep_result.num_models)
    group_vae_args = _clone_namespace(vae_args, parallel_layers=num_models)
    stacked_data = prep_result.stacked_data
    use_wa_mse = bool(prep_result.use_wa_mse)
    part_metas = prep_result.part_metas
    split_metas = prep_result.split_metas
    if float(outlier_protect_ratio) > 0.0:
        per_linear_protected = []
        zero_protected = []
        any_protected = False
        for ref, meta in zip(group_refs, split_metas):
            if str(outlier_protect_axis) == "output":
                protected_idx = meta.protected_output_indices
                total_channels = int(ref.module.out_features)
            else:
                protected_idx = meta.protected_input_indices
                total_channels = int(ref.module.in_features)
            protected_count = int(protected_idx.numel()) if isinstance(protected_idx, torch.Tensor) else 0
            any_protected = any_protected or protected_count > 0
            per_linear_protected.append(
                f"{ref.name}:{protected_count}/{total_channels}"
            )
            if protected_count == 0:
                zero_protected.append(ref.name)
        log.info(
            "[%s] outlier protection axis=%s ratio=%.6f protected_channels=%s",
            group_tag,
            str(outlier_protect_axis),
            float(outlier_protect_ratio),
            ",".join(per_linear_protected),
        )
        if zero_protected:
            log.info(
                "[%s] outlier protection skipped due to floor(...) == 0 for %d/%d linears: %s",
                group_tag,
                len(zero_protected),
                len(split_metas),
                ",".join(zero_protected),
            )
        if not any_protected:
            log.info("[%s] outlier protection produced no protected channels in this group.", group_tag)
    if len(split_metas) != len(group_refs):
        raise RuntimeError(
            f"[{group_tag}] split metadata mismatch: len(split_metas)={len(split_metas)} "
            f"vs len(group_refs)={len(group_refs)}"
        )
    if use_wa_mse:
        log.info("[%s] wa_mse enabled with online act_max gather.", group_tag)

    residual_data = stacked_data.detach().clone().contiguous()
    all_stage_bits: List[torch.Tensor] = []
    all_stage_decoders: List[List[nn.Module]] = []
    all_stage_codebook_dims: List[int] = []

    for stage_idx in range(residual_stages):
        stage_tag = f"{group_tag}/stage{stage_idx + 1}"
        stage_steps = int(resolve_stage_value(steps, stage_idx, arg_name="--steps_per_category/--steps_per_group"))
        stage_codebook_bits = resolve_codebook_int_for_category(
            resolve_stage_value(getattr(group_vae_args, "codebook_bits"), stage_idx, arg_name="--codebook_bits"),
            group_category,
            arg_name="codebook_bits",
        )
        stage_codebook_dim = resolve_codebook_int_for_category(
            resolve_stage_value(getattr(group_vae_args, "codebook_dim"), stage_idx, arg_name="--codebook_dim"),
            group_category,
            arg_name="codebook_dim",
        )
        residual_data = _reshape_blocks_for_codebook_dim(residual_data, codebook_dim=int(stage_codebook_dim))
        stage_recon_loss = str(
            resolve_stage_value(getattr(group_vae_args, "recon_loss_type", "mse"),
                                stage_idx, arg_name="--recon_loss_type")
        ).strip().lower()
        stage_base_ch = int(resolve_stage_value(
            getattr(group_vae_args, "base_ch", 128), stage_idx, arg_name="--base_ch"))
        stage_num_res_blocks = int(
            resolve_stage_value(getattr(group_vae_args, "num_res_blocks", 1), stage_idx, arg_name="--num_res_blocks")
        )
        stage_norm_type = str(
            resolve_stage_value(getattr(group_vae_args, "norm_type", "group"), stage_idx, arg_name="--norm_type")
        ).strip().lower()
        stage_decoder_type = str(
            resolve_stage_value(getattr(group_vae_args, "decoder_type", "linear"), stage_idx, arg_name="--decoder_type")
        ).strip().lower()
        stage_decoder_base_ch = resolve_stage_value(
            getattr(group_vae_args, "decoder_base_ch", None), stage_idx, arg_name="--decoder_base_ch"
        )
        stage_decoder_num_res_blocks = resolve_stage_value(
            getattr(group_vae_args, "decoder_num_res_blocks", None), stage_idx, arg_name="--decoder_num_res_blocks"
        )
        stage_vae_args = _clone_namespace(
            group_vae_args,
            codebook_bits=int(stage_codebook_bits),
            codebook_dim=int(stage_codebook_dim),
            base_ch=int(stage_base_ch),
            num_res_blocks=int(stage_num_res_blocks),
            norm_type=stage_norm_type,
            decoder_type=stage_decoder_type,
            decoder_base_ch=None if stage_decoder_base_ch is None else int(stage_decoder_base_ch),
            decoder_num_res_blocks=(
                None if stage_decoder_num_res_blocks is None else int(stage_decoder_num_res_blocks)
            ),
            recon_loss_type=stage_recon_loss,
        )
        resolve_autoencoder_arch_args(stage_vae_args)

        if bool(getattr(stage_vae_args, "normalize_weight", False)):
            stage_norm_mean, stage_norm_scale = _compute_stage_norm_stats(residual_data)
            stage_train_data = _apply_stage_norm(
                residual_data,
                mean=stage_norm_mean,
                scale=stage_norm_scale,
            )
        else:
            stage_norm_mean = None
            stage_norm_scale = None
            stage_train_data = residual_data

        train_loader, eval_loader = _build_block_data_loaders(stage_train_data, batch_size=int(batch_size))
        vae = MultiLayerVAE(stage_vae_args).to(train_device)

        # 2) 训练当前 residual stage 对应的 VAE。
        optimizer = create_optimizer(vae.parameters(), stage_vae_args, stage_vae_args.lr)
        lr_scheduler = None
        lr_scheduler_name = str(getattr(stage_vae_args, "lr_scheduler", "none"))
        if lr_scheduler_name != "none":
            import transformers

            lr_scheduler = transformers.get_scheduler(
                lr_scheduler_name,
                optimizer,
                num_warmup_steps=int(getattr(stage_vae_args, "lr_warmup_steps", 0)),
                num_training_steps=int(stage_steps),
            )

        residual_rms_before = float(residual_data.float().pow(2).mean().sqrt().item())
        log.info(
            "[%s] start (residual_rms=%.6e, steps=%d, blocks=%d, bits=%d, dim=%d, recon_loss=%s, base_ch=%d, num_res_blocks=%d, norm_type=%s, decoder_type=%s, stage_norm=%s)",
            stage_tag,
            residual_rms_before,
            int(stage_steps),
            int(residual_data.shape[0]),
            int(stage_codebook_bits),
            int(stage_codebook_dim),
            stage_recon_loss,
            int(stage_base_ch),
            int(stage_num_res_blocks),
            stage_norm_type,
            stage_decoder_type,
            "on" if bool(getattr(stage_vae_args, "normalize_weight", False)) else "off",
        )
        start = time.time()
        train_iter = iter(train_loader)
        for step in range(int(stage_steps)):
            try:
                x_batch, block_idx_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_batch, block_idx_batch = next(train_iter)

            x = x_batch.to(device=train_device, dtype=train_dtype, non_blocking=True)
            act_max_batch = None
            if stage_recon_loss == "wa_mse":
                act_max_batch = gather_wa_mse_act_max_batch(
                    block_idx_batch=block_idx_batch,
                    part_metas=part_metas,
                    codebook_dim=int(stage_codebook_dim),
                    train_device=train_device,
                    target_dtype=train_dtype,
                )
            optimizer.zero_grad(set_to_none=True)
            _, loss_dict = vae(x, is_train=True, act_max=act_max_batch)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if log_every > 0 and (step + 1) % int(log_every) == 0:
                speed = (time.time() - start) / int(log_every)
                recon = loss_dict.get("train/recon_loss")
                commit = loss_dict.get("train/commitment_loss")
                log.info(
                    "[%s] step=%d/%d loss=%.6f recon=%.6f commit=%.6f speed=%.4fs/it",
                    stage_tag,
                    step + 1,
                    stage_steps,
                    float(loss.detach().float().item()),
                    float(recon.detach().float().item()) if isinstance(recon, torch.Tensor) else float("nan"),
                    float(commit.detach().float().item()) if isinstance(commit, torch.Tensor) else float("nan"),
                    speed,
                )
                start = time.time()

            if eval_every > 0 and (step + 1) % int(eval_every) == 0:
                vae.eval()
                with torch.no_grad():
                    mse_acc = []
                    top_k_mse_acc = []
                    total = 0
                    for x_eval_batch, _eval_idx_batch in eval_loader:
                        if total >= int(eval_blocks):
                            break
                        x_eval_batch = x_eval_batch[: max(0, int(eval_blocks) - total)]
                        total += x_eval_batch.shape[0]
                        x_eval = x_eval_batch.to(device=train_device, dtype=train_dtype, non_blocking=True)
                        x_recon, _ = vae(x_eval, is_train=False)
                        x_eval_f = x_eval.float()
                        x_recon_f = x_recon.float()
                        mse_acc.append(torch.nn.functional.mse_loss(x_recon_f, x_eval_f))

                        # 对每个并行模型（P 维）独立选 top-k：
                        # x_eval/x_recon: [B, P, C] -> [P, B*C]
                        flat_eval = x_eval_f.permute(1, 0, 2).reshape(x_eval_f.shape[1], -1)
                        flat_recon = x_recon_f.permute(1, 0, 2).reshape(x_recon_f.shape[1], -1)
                        k = min(100, flat_eval.shape[1])
                        _, topk_idx = torch.topk(flat_eval.abs(), k=k, dim=1)
                        top_eval = torch.gather(flat_eval, dim=1, index=topk_idx)
                        top_recon = torch.gather(flat_recon, dim=1, index=topk_idx)
                        top_k_mse_acc.append(torch.nn.functional.mse_loss(top_recon, top_eval))
                    mse = torch.stack(mse_acc).mean() if mse_acc else torch.tensor(0.0)
                    top_k_mse = torch.stack(top_k_mse_acc).mean() if top_k_mse_acc else torch.tensor(0.0)
                log.info(
                    "[%s] eval@step=%d mse=%.6e top_k_mse(k=100)=%.6e",
                    stage_tag,
                    step + 1,
                    float(mse.detach().cpu().item()),
                    float(top_k_mse.detach().cpu().item()),
                )
                vae.train()

        # 3) 对当前 stage 的 residual 生成重构，更新下一阶段 residual。
        vae.eval()
        stage_recon_chunks: List[torch.Tensor] = []
        stage_bit_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for x_in_batch, _eval_idx_batch in eval_loader:
                x_in = x_in_batch.to(device=train_device, dtype=train_dtype, non_blocking=True)
                x_recon, bit_idx = vae(x_in, is_train=False)
                stage_recon_chunks.append(x_recon.detach().to(device="cpu", dtype=residual_data.dtype))
                if do_convert:
                    stage_bit_chunks.append(bit_idx.detach().to("cpu"))

        stage_recon_full_norm = torch.cat(stage_recon_chunks, dim=0)
        if tuple(stage_recon_full_norm.shape) != tuple(stage_train_data.shape):
            raise RuntimeError(
                f"[{stage_tag}] recon shape mismatch: recon={tuple(stage_recon_full_norm.shape)} "
                f"vs stage_train_data={tuple(stage_train_data.shape)}"
            )
        if stage_norm_mean is not None and stage_norm_scale is not None:
            stage_recon_full = _restore_stage_norm(
                stage_recon_full_norm,
                mean=stage_norm_mean,
                scale=stage_norm_scale,
            )
        else:
            stage_recon_full = stage_recon_full_norm
        if tuple(stage_recon_full.shape) != tuple(residual_data.shape):
            raise RuntimeError(
                f"[{stage_tag}] denorm recon shape mismatch: recon={tuple(stage_recon_full.shape)} "
                f"vs residual={tuple(residual_data.shape)}"
            )
        residual_data = (residual_data - stage_recon_full).contiguous()
        residual_rms_after = float(residual_data.float().pow(2).mean().sqrt().item())
        log.info(
            "[%s] residual rms: before=%.6e after=%.6e",
            stage_tag,
            residual_rms_before,
            residual_rms_after,
        )

        if do_convert:
            if not stage_bit_chunks:
                raise RuntimeError(f"[{stage_tag}] no bit indices collected during conversion.")
            stage_full_bits = torch.cat(stage_bit_chunks, dim=0)  # [N_blocks, P, latent_dim]
            all_stage_bits.append(stage_full_bits)
            all_stage_codebook_dims.append(int(stage_codebook_dim))

            decoder_in_dim = int(getattr(vae.model.decoder, "in_dim"))
            use_new_quant = bool(getattr(stage_vae_args, "new_quant", False))
            quant_q_scale = (1.0 / math.sqrt(decoder_in_dim)) if use_new_quant else 1.0

            decoders: List[nn.Module] = []
            for i in range(num_models):
                dec = vae.model.decoder.get_sub_decoder(i)
                _fuse_q_scale_into_decoder(dec, q_scale=float(quant_q_scale))
                if bool(getattr(stage_vae_args, "normalize_weight", False)):
                    if stage_norm_mean is None or stage_norm_scale is None:
                        raise RuntimeError(f"[{stage_tag}] stage norm stats missing while normalize_weight=True")
                    _fuse_norm_into_decoder(
                        dec,
                        mean=float(stage_norm_mean[i].item()),
                        std=float(stage_norm_scale[i].item()),
                    )
                decoders.append(dec)
            all_stage_decoders.append(decoders)

        del vae, train_loader, eval_loader, optimizer
        if lr_scheduler is not None:
            del lr_scheduler
        torch.cuda.empty_cache()

    # # 保存分组 VAE，便于复现实验和离线分析。
    # group_dir = os.path.join(output_dir, "vae_by_category", group_tag.replace("/", "_"))
    # os.makedirs(group_dir, exist_ok=True)
    # torch.save(vae.state_dict(), os.path.join(group_dir, "vae_state.pt"))

    if not do_convert:
        del stacked_data, residual_data
        torch.cuda.empty_cache()
        return

    if (
        len(all_stage_bits) != residual_stages
        or len(all_stage_decoders) != residual_stages
        or len(all_stage_codebook_dims) != residual_stages
    ):
        raise RuntimeError(
            f"[{group_tag}] stage payload mismatch: bits={len(all_stage_bits)} "
            f"decoders={len(all_stage_decoders)} codebook_dims={len(all_stage_codebook_dims)} "
            f"residual_stages={residual_stages}"
        )

    for i, r in enumerate(group_refs):
        old = r.module
        split_meta = split_metas[i]
        if str(split_meta.linear_name) != str(r.name):
            raise RuntimeError(
                f"[{group_tag}] split metadata order mismatch at idx={i}: "
                f"meta={split_meta.linear_name}, ref={r.name}"
            )
        if int(split_meta.parallel_rows) * int(split_meta.parallel_cols) != int(parts_per_linear):
            raise RuntimeError(
                f"[{group_tag}] split parts mismatch at idx={i}: "
                f"meta={split_meta.parallel_rows}x{split_meta.parallel_cols}, expected={parts_per_linear}"
            )
        layer_idx = _extract_layer_idx(r.name)
        skip_this = bool(
            skip_layer_keys
            and layer_idx is not None
            and (int(layer_idx), str(r.category)) in skip_layer_keys
        )
        start_idx = i * parts_per_linear
        end_idx = start_idx + parts_per_linear
        stage_part_bits_payload: List[object] = []
        stage_part_decoders_payload: List[object] = []
        for stage_idx in range(residual_stages):
            stage_bits = all_stage_bits[stage_idx]
            stage_decoders = all_stage_decoders[stage_idx]
            part_bits = []
            part_decoders = []
            for model_idx in range(start_idx, end_idx):
                part_bits.append(stage_bits[:, model_idx, :].unsqueeze(1))  # [N_blocks, 1, latent_dim]
                part_decoders.append(stage_decoders[model_idx])
            if parts_per_linear > 1:
                stage_part_bits_payload.append(part_bits)
                stage_part_decoders_payload.append(part_decoders)
            else:
                stage_part_bits_payload.append(part_bits[0])
                stage_part_decoders_payload.append(part_decoders[0])

        if residual_stages == 1:
            new_linear = VAELinear(
                in_features=old.in_features,
                out_features=old.out_features,
                bias=old.bias,
                original_weight=old.weight,
                vq_weight=stage_part_bits_payload[0],
                decoder=stage_part_decoders_payload[0],
                codebook_dim=int(all_stage_codebook_dims[0]),
                transpose=r.transpose,
                parallel_parts=parts_per_linear,
                parallel_rows=row_parts,
                parallel_cols=col_parts,
                restore_row_indices=split_meta.restore_row_indices,
                restore_col_indices=split_meta.restore_col_indices,
                compressed_in_features=int(split_meta.compressed_in_features),
                compressed_out_features=int(split_meta.compressed_out_features),
                protected_input_indices=split_meta.protected_input_indices,
                protected_input_weight=split_meta.protected_input_weight,
                protected_output_indices=split_meta.protected_output_indices,
                protected_output_weight=split_meta.protected_output_weight,
                always_use_original=skip_this,
                protect_original_weight=skip_this,
            ).to(convert_device)
        else:
            new_linear = VAELinear(
                in_features=old.in_features,
                out_features=old.out_features,
                bias=old.bias,
                original_weight=old.weight,
                vq_weight=None,
                decoder=None,
                stage_vq_weights=stage_part_bits_payload,
                stage_decoders=stage_part_decoders_payload,
                codebook_dim=int(all_stage_codebook_dims[0]),
                stage_codebook_dims=list(all_stage_codebook_dims),
                transpose=r.transpose,
                parallel_parts=parts_per_linear,
                parallel_rows=row_parts,
                parallel_cols=col_parts,
                restore_row_indices=split_meta.restore_row_indices,
                restore_col_indices=split_meta.restore_col_indices,
                compressed_in_features=int(split_meta.compressed_in_features),
                compressed_out_features=int(split_meta.compressed_out_features),
                protected_input_indices=split_meta.protected_input_indices,
                protected_input_weight=split_meta.protected_input_weight,
                protected_output_indices=split_meta.protected_output_indices,
                protected_output_weight=split_meta.protected_output_weight,
                always_use_original=skip_this,
                protect_original_weight=skip_this,
            ).to(convert_device)
        # 替换时预热：后续 LoRA / PPL 前向可直接复用缓存权重，避免重复重构。
        try:
            new_linear.prime_decoded_weight_cache(dtype=train_dtype)
        except Exception as e:
            log.warning("[%s] cache warmup failed for %s: %s", group_tag, r.name, e)
        # 替换后将模块放回 CPU，降低显存占用。
        new_linear.to("cpu")
        set_module_by_name(model, r.name, new_linear)

    del stacked_data, residual_data, all_stage_bits, all_stage_decoders, all_stage_codebook_dims
    torch.cuda.empty_cache()


def main(argv: Optional[Sequence[str]] = None) -> None:
    global log
    cat_args, hf_args, training_args, vae_args = process_cat_train_args(argv)
    set_seed(cat_args.seed)

    os.makedirs(cat_args.output_dir, exist_ok=True)
    run_output_dir = _build_run_output_dir(cat_args.output_dir, vae_args.model_path)
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "linear_by_category.log")
    log = get_logger("linear_by_category")
    cat_args.output_dir = run_output_dir

    log.info("Run output directory: %s", run_output_dir)
    log.info(
        "Args:\nscript=%s\nvae=%s\ntraining=%s",
        _format_namespace(cat_args),
        _format_namespace(vae_args),
        _format_namespace(training_args),
    )

    log.info("Loading model: %s", vae_args.model_path)
    from rotation.model_utils import get_model

    model = get_model(vae_args.model_path, hf_args.access_token)
    if bool(getattr(cat_args, "rot_llm", False)):
        from rotation.model_rotation import prepare_model

        log.info("Applying offline LLM rotation fusion before VAE compression.")
        model = prepare_model(model)
    intra_part_sort_mode = getattr(cat_args, "intra_part_sort_mode", "l2")
    stage0_intra_part_sort_mode = resolve_stage_value(
        intra_part_sort_mode,
        0,
        arg_name="--intra_part_sort_mode",
    )
    stage0_row_sort_mode, stage0_col_sort_mode = normalize_intra_part_sort_mode(
        stage0_intra_part_sort_mode,
        arg_name="--intra_part_sort_mode",
    )
    stage0_sort_mode_desc = format_intra_part_sort_mode((stage0_row_sort_mode, stage0_col_sort_mode))
    if isinstance(intra_part_sort_mode, (list, tuple)):
        stage_modes = [format_intra_part_sort_mode(v) for v in intra_part_sort_mode]
        if len(set(stage_modes)) > 1:
            log.warning(
                "--intra_part_sort_mode provided as multi-stage list (%s). "
                "当前实现的分块排序在 stage0 固化，后续 stage 将沿用 stage0 排序。",
                ",".join(stage_modes),
            )
    intra_parallel_raw = getattr(cat_args, "intra_parallel", 1)
    if isinstance(intra_parallel_raw, dict):
        log.info(
            "intra_parallel category overrides enabled: keys=%s",
            ",".join(sorted(str(k) for k in intra_parallel_raw.keys())),
        )
    activation_weight_by_linear: Optional[Dict[str, torch.Tensor]] = None
    wa_mse_runtime: Optional[Dict[str, object]] = None
    outlier_protect_ratio = float(getattr(cat_args, "outlier_protect_ratio", 0.0))
    outlier_protect_axis = str(getattr(cat_args, "outlier_protect_axis", "input")).strip().lower()
    act_path = getattr(cat_args, "activation_weight_path", None)
    if act_path:
        activation_weight_by_linear = load_activation_weight_dict(str(act_path))
        log.info(
            "Loaded static activation abs-max dict: %s (entries=%d)",
            act_path,
            len(activation_weight_by_linear),
        )

    if _contains_stage_choice(getattr(vae_args, "recon_loss_type", ""), "wa_mse"):
        wa_mse_runtime = {
            "dynamic": str(getattr(cat_args, "wa_mse_act_mode", "dynamic")).strip().lower() == "dynamic",
            "cache": None,  # type: Optional[ActivationCalibrationCache]
            "dataset": str(getattr(cat_args, "wa_mse_calib_dataset", "wikitext2")),
            "nsamples": int(getattr(cat_args, "wa_mse_calib_nsamples", 512)),
            "seqlen": int(getattr(cat_args, "wa_mse_calib_seqlen", 512)),
            "seed": int(getattr(cat_args, "wa_mse_calib_seed", 0)),
            "device": str(getattr(cat_args, "wa_mse_calib_device", "")).strip() or str(cat_args.train_device),
            "log_every": int(getattr(cat_args, "wa_mse_calib_log_every", 0)),
            "model_path": str(vae_args.model_path),
            "access_token": hf_args.access_token,
            "static_dict": activation_weight_by_linear,
        }
        if not bool(wa_mse_runtime["dynamic"]) and activation_weight_by_linear is None:
            raise ValueError(
                "wa_mse requires either --wa_mse_act_mode dynamic or --activation_weight_path in static mode."
            )
        if bool(wa_mse_runtime["dynamic"]):
            log.info(
                "wa_mse dynamic act_max enabled: dataset=%s nsamples=%d seqlen=%d seed=%d device=%s",
                str(wa_mse_runtime["dataset"]),
                int(wa_mse_runtime["nsamples"]),
                int(wa_mse_runtime["seqlen"]),
                int(wa_mse_runtime["seed"]),
                str(wa_mse_runtime["device"]),
            )
    if outlier_protect_ratio > 0.0:
        has_reusable_dynamic_act = bool(wa_mse_runtime and bool(wa_mse_runtime.get("dynamic", False)))
        if activation_weight_by_linear is None and not has_reusable_dynamic_act:
            raise ValueError(
                "outlier_protect_ratio requires activation vectors. "
                "Please provide --activation_weight_path or enable a wa_mse dynamic act_max source."
            )
        log.info("Outlier protection enabled: axis=%s ratio=%.6f", outlier_protect_axis, outlier_protect_ratio)
    transpose_modules = _split_csv(cat_args.transpose_modules)
    projection_suffixes = _split_csv(cat_args.projection_suffixes)
    only_decoder_projections = bool(cat_args.only_decoder_projections) and not bool(cat_args.include_all_linears)
    all_linears = _collect_linears(
        model,
        transpose_modules,
        only_decoder_projections=only_decoder_projections,
        projection_suffixes=projection_suffixes,
    )
    discovered_categories = [r.category for r in all_linears]
    category_order = _resolve_category_order(cat_args.category_order, discovered_categories)

    refs_by_cat: Dict[str, List[LinearRef]] = {}
    for r in all_linears:
        refs_by_cat.setdefault(r.category, []).append(r)
    discovered_skip_keys = []
    for r in all_linears:
        li = _extract_layer_idx(r.name)
        if li is not None:
            discovered_skip_keys.append((li, r.category))
    skip_layer_keys, matched, missing = resolve_skip_layer_matches(
        getattr(cat_args, "skip_layers", ""),
        discovered_skip_keys,
    )
    if skip_layer_keys:
        if matched:
            log.info(
                "skip_layers 生效: %s",
                ",".join(f"{li}.{cat}" for li, cat in matched),
            )
        if missing:
            log.warning(
                "skip_layers 未匹配到任何 Linear: %s",
                ",".join(f"{li}.{cat}" for li, cat in missing),
            )

    steps_per_group = cat_args.steps_per_group if cat_args.steps_per_group is not None else cat_args.steps_per_category
    linear_group_size = int(cat_args.linear_group_size)
    if linear_group_size < 1:
        raise ValueError(f"linear_group_size must be >= 1, got {linear_group_size}")

    active_categories = [c for c in category_order if c in refs_by_cat]
    category_intra_parallel: Dict[str, Tuple[int, int]] = {}
    category_codebook: Dict[str, Tuple[int, int]] = {}
    for cat in active_categories:
        category_intra_parallel[cat] = resolve_intra_parallel_for_category(intra_parallel_raw, cat)
        stage0_codebook_bits = resolve_stage_value(getattr(vae_args, "codebook_bits"), 0, arg_name="--codebook_bits")
        stage0_codebook_dim = resolve_stage_value(getattr(vae_args, "codebook_dim"), 0, arg_name="--codebook_dim")
        category_codebook[cat] = (
            resolve_codebook_int_for_category(
                stage0_codebook_bits,
                cat,
                arg_name="codebook_bits",
            ),
            resolve_codebook_int_for_category(
                stage0_codebook_dim,
                cat,
                arg_name="codebook_dim",
            ),
        )

    sort_needs_act = bool(category_intra_parallel) and (
        stage0_row_sort_mode == "act_l2" or stage0_col_sort_mode == "act_l2"
    )
    if sort_needs_act and activation_weight_by_linear is None and wa_mse_runtime is None:
        raise ValueError(
            "intra_part_sort_mode requires activation vectors when enabled dimension uses act_l2. "
            "Please provide --activation_weight_path."
        )

    unique_parallel = sorted(set(category_intra_parallel.values()))
    if int(getattr(vae_args, "parallel_layers", 1)) != 1:
        log.warning("检测到 --parallel_layers=%d，但当前脚本不再使用该参数；请使用 --intra_parallel。", int(vae_args.parallel_layers))
    if unique_parallel:
        if len(unique_parallel) == 1:
            intra_row_parts, intra_col_parts = unique_parallel[0]
            intra_parts_per_linear = int(intra_row_parts) * int(intra_col_parts)
            intra_parallel_desc = _format_intra_parallel_desc(intra_row_parts, intra_col_parts)
            log.info(
                "并行配置: linear_group_size=%d, intra_parallel=%s (rows=%d, cols=%d), intra_part_sort_mode=%s, total_num_models=%d",
                linear_group_size,
                intra_parallel_desc,
                intra_row_parts,
                intra_col_parts,
                stage0_sort_mode_desc,
                linear_group_size * intra_parts_per_linear,
            )
        else:
            per_cat_desc = ",".join(
                f"{cat}:{_format_intra_parallel_desc(*category_intra_parallel[cat])}"
                for cat in active_categories
            )
            models_per_group_values = sorted(
                linear_group_size * int(rp) * int(cp)
                for rp, cp in unique_parallel
            )
            log.info(
                "并行配置: linear_group_size=%d, intra_parallel=per_category{%s}, intra_part_sort_mode=%s, total_num_models_per_group=[%d,%d]",
                linear_group_size,
                per_cat_desc,
                stage0_sort_mode_desc,
                models_per_group_values[0],
                models_per_group_values[-1],
            )
    unique_codebook = sorted(set(category_codebook.values()))
    if unique_codebook:
        if len(unique_codebook) == 1:
            cb_bits, cb_dim = unique_codebook[0]
            log.info("codebook 配置: bits=%d, dim=%d", cb_bits, cb_dim)
        else:
            per_cat_cb_desc = ",".join(
                f"{cat}:[bits={category_codebook[cat][0]},dim={category_codebook[cat][1]}]"
                for cat in active_categories
            )
            log.info("codebook 配置: per_category{%s}", per_cat_cb_desc)
    lora_round_idx = 0
    lora_schedule = getattr(cat_args, "lora_schedule", None)
    if isinstance(lora_schedule, dict) and lora_schedule:
        log.info(
            "LoRA category schedule enabled. keys=%s",
            ",".join(sorted(str(k) for k in lora_schedule.keys())),
        )
    for cat_idx, cat in enumerate(active_categories):
        if cat not in refs_by_cat:
            continue

        refs = refs_by_cat[cat]
        if not refs:
            continue

        cat_row_parts, cat_col_parts = category_intra_parallel[cat]
        cat_codebook_bits, cat_codebook_dim = category_codebook[cat]
        cat_vae_args = _clone_namespace(vae_args)
        cat_parts_per_linear = int(cat_row_parts) * int(cat_col_parts)
        cat_intra_parallel_desc = _format_intra_parallel_desc(cat_row_parts, cat_col_parts)
        log.info(
            "=== Category: %s (%d linears, intra_parallel=%s rows=%d cols=%d, codebook_bits=%d, codebook_dim=%d) ===",
            cat,
            len(refs),
            cat_intra_parallel_desc,
            cat_row_parts,
            cat_col_parts,
            int(cat_codebook_bits),
            int(cat_codebook_dim),
        )

        refs_sorted = []
        missing = 0
        for r in refs:
            li = _extract_layer_idx(r.name)
            if li is None:
                missing += 1
                continue
            refs_sorted.append((li, r))
        if missing:
            log.warning("[%s] %d modules missing layer_idx, skipped.", cat, missing)
        refs_sorted.sort(key=lambda x: x[0])
        ordered_refs = [r for _, r in refs_sorted]

        for start in range(0, len(ordered_refs), linear_group_size):
            group_refs = ordered_refs[start:start + linear_group_size]
            if len(group_refs) < linear_group_size and not cat_args.allow_tail_group:
                log.info("[%s] tail group size=%d skipped (set --allow_tail_group to include).", cat, len(group_refs))
                break
            layer_indices = [idx for idx, _ in refs_sorted[start:start + linear_group_size]]
            group_tag = f"{cat}.L{layer_indices[0]}-{layer_indices[-1]}"
            log.info(
                "---- Group: %s (linears=%d, intra_parallel=%s, num_models=%d) ----",
                group_tag,
                len(group_refs),
                cat_intra_parallel_desc,
                len(group_refs) * cat_parts_per_linear,
            )
            _train_group_vae_and_replace(
                model=model,
                group_refs=group_refs,
                group_tag=group_tag,
                vae_args=cat_vae_args,
                training_args=training_args,
                train_device=cat_args.train_device,
                convert_device=cat_args.convert_device,
                do_convert=bool(cat_args.convert),
                steps=steps_per_group,
                batch_size=cat_args.batch_size,
                log_every=cat_args.log_every,
                eval_every=cat_args.eval_every,
                eval_blocks=cat_args.eval_blocks,
                output_dir=cat_args.output_dir,
                intra_parallel=(cat_row_parts, cat_col_parts),
                intra_part_sort_mode=intra_part_sort_mode,
                skip_layer_keys=skip_layer_keys,
                activation_weight_by_linear=activation_weight_by_linear,
                wa_mse_runtime=wa_mse_runtime,
                outlier_protect_ratio=outlier_protect_ratio,
                outlier_protect_axis=outlier_protect_axis,
            )
            # _eval_ppl_after_category(
            #     model=model,
            #     vae_args=vae_args,
            #     ppl_limit=cat_args.ppl_limit,
            #     category=cat,
            #     eval_device=cat_args.train_device,
            # )

        if cat_args.lora_after_category:
            from train_utils.lora_utils import lora_finetune_remaining_categories
            log.info("LoRA 微调前评估...")
            _eval_ppl_after_category(
                model=model,
                vae_args=vae_args,
                ppl_limit=cat_args.ppl_limit,
                category=cat,
                eval_device=cat_args.train_device,
            )

            remaining_categories = active_categories[cat_idx + 1:]
            model = lora_finetune_remaining_categories(
                model=model,
                remaining_categories=remaining_categories,
                collect_linears_fn=_collect_linears,
                transpose_modules=transpose_modules,
                projection_suffixes=projection_suffixes,
                only_decoder_projections=only_decoder_projections,
                cat_args=cat_args,
                vae_args=vae_args,
                training_args=training_args,
                logger=log,
                lora_round_idx=lora_round_idx,
                after_category=cat,
            )
            lora_round_idx += 1

        _eval_ppl_after_category(
            model=model,
            vae_args=vae_args,
            ppl_limit=cat_args.ppl_limit,
            category=cat,
            eval_device=cat_args.train_device,
        )
        # cat_dir_name = _safe_path_token(cat)
        # cat_model_dir = os.path.join(run_output_dir, cat_dir_name)
        # save_paths = save_model_checkpoint(
        #     model,
        #     cat_model_dir,
        #     base_model_path=vae_args.model_path,
        #     tokenizer=None,
        #     save_config=True,
        #     extra_meta={
        #         "stage": "after_category",
        #         "category": cat,
        #         "category_index": int(cat_idx),
        #         "lora_after_category": bool(cat_args.lora_after_category),
        #     },
        # )
        # log.info("Saved category checkpoint (%s): %s", cat, save_paths["output_dir"])

    _eval_ppl_after_category(
        model=model,
        vae_args=vae_args,
        ppl_limit=cat_args.ppl_limit,
        category="none",
        eval_device=cat_args.train_device,
    )
    if cat_args.save_model:
        if not cat_args.convert:
            raise ValueError("--save_model requires --convert")
        from transformers import AutoTokenizer
        from litebsq.vae_linear import clear_model_vae_linear_cache

        model_out = os.path.join(run_output_dir, "final_model")
        tok = AutoTokenizer.from_pretrained(vae_args.model_path, use_fast=True, token=hf_args.access_token)
        cleared = clear_model_vae_linear_cache(model)
        log.info("Final save: cleared decoded cache for %d VAELinear modules.", cleared)
        save_paths = save_model_checkpoint(
            model,
            model_out,
            base_model_path=vae_args.model_path,
            tokenizer=tok,
            save_config=True,
            extra_meta={"stage": "final"},
            unload_vae_original_weights=bool(cat_args.unload_vae_original_weights_on_final_save),
        )
        log.info("Saved final model to %s", save_paths["output_dir"])

    log.info("Done.")


if __name__ == "__main__":
    main()
