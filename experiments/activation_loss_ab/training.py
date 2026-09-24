"""Bounded CAT-equivalent training: one full Linear, no transpose/protection/rotation.

Uses production data preparation, BSQ VAE, normalization, optimizer and payload
conversion. Only the reconstruction term differs; no monkeypatches are used.
The restricted MSE path is checked against train_group_vae_payload in tests.
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import torch
from transformers import get_scheduler

from experiments.activation_loss_ab.objectives import block_output_mse, gather_grams
from experiments.weight_rotation_ab import _host_for_weight
from litebsq.llm_vae import MultiLayerVAE
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_data_prep import (
    LinearPrepRef, materialize_prepared_group_data, prepare_group_linear_entries,
)
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_pipeline import (
    _apply_stage_norm, _clone_namespace, _compute_stage_norm_stats,
    _fuse_norm_into_decoder, _fuse_q_scale_into_decoder,
    _resolve_train_dtype, _restore_stage_norm, apply_group_vae_payload,
)
from train_utils.train_args import create_optimizer
from train_utils.utils import LinearRef

MODES = ("mse", "amse", "block_output")


def configure(category: str, mode: str, seed: int, steps: int, batch_size: int):
    if mode not in MODES:
        raise ValueError(f"Unknown objective: {mode}")
    base_loss = "amse" if mode == "amse" else "mse"
    cli = [
        "--model_path", "Qwen/Qwen3-8B", "--compression_categories", category,
        "--after_category_mode", "none", "--seed", str(seed), "--data_seed", str(seed),
        "--bf16", "true", "--deterministic", "true", "--codebook_bits", "default=32",
        "--codebook_dim", "default=32", "--residual_stages", "default=2",
        "--base_ch", "default=128", "--num_res_blocks", "default=0",
        "--decoder_base_ch", "default=128", "--decoder_num_res_blocks", "default=1",
        "--norm_type", "default=layer", "--activation_type", "default=swish",
        "--decoder_type", "default=symmetric", "--recon_loss_type", f"default={base_loss}",
        "--normalize_weight", "--new_quant", "--vae_steps", f"default={steps}",
        "--vae_batch_size", str(batch_size), "--vae_learning_rate", "0.003",
        "--vae_weight_decay", "0", "--vae_optim", "adamw", "--vae_lr_scheduler_type", "linear",
        "--vae_warmup_ratio", "0", "--beta1", "0.9", "--beta2", "0.95",
        "--l1_weight", "1", "--lfq_weight", "2.5", "--commitment_loss_weight", "0.25",
        "--entropy_loss_weight", "0.01", "--vae_decoder_checkpoint", "true",
        "--channel_protect_mode", "none", "--channel_protect_count", "default=0",
        "--channel_axis", "input", "--channel_rank_metric", "channel_weight_abs",
        "--channel_quant", "int8", "--weight_rotation", "none", "--weight_rotation_block_size", "32",
    ]
    cat_args, _, training_args, vae_args = parse_cat_runtime_args(cli)
    training_args.seed = cat_args.seed
    cfg = resolve_category_runtime_configs(cat_args, vae_args, [category])[category]
    return cfg, vae_args, training_args


def _state_digest(model) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def train_weight(
    weight: torch.Tensor, *, module_name: str, mode: str, grams: torch.Tensor,
    seed: int, steps: int, batch_size: int, device: str, save_path: Path | None = None,
):
    if weight.ndim != 2 or weight.shape[1] % 32:
        raise ValueError("Only full, row-aligned input-axis matrices are supported.")
    if grams.shape != (weight.shape[1] // 32, 32, 32):
        raise ValueError("Activation groups do not match the full Linear input width.")
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive.")
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    host, linear = _host_for_weight(weight.detach().cpu().float(), module_name)
    category = module_name.rsplit(".", 1)[1]
    cfg, vae_args, training_args = configure(category, mode, seed, steps, batch_size)
    train_dtype = _resolve_train_dtype(training_args)
    grams = grams.detach().to(device=device, dtype=torch.float32)
    refs = [LinearRef(module_name, linear, category, False)]
    prep = prepare_group_linear_entries(
        group_refs=[LinearPrepRef(module_name, linear.weight, linear.in_features, linear.out_features, False)],
        activation_weight_by_linear=None, channel_protect_count=0,
        channel_axis="input", recon_loss_type="mse", apply_outlier_channel_removal=False,
    )
    shared = _clone_namespace(
        vae_args, parallel_layers=1, residual_stages=2, codebook_bits=32, codebook_dim=32,
        base_ch=int(cfg.base_ch), num_res_blocks=int(cfg.num_res_blocks),
        norm_type=str(cfg.norm_type), activation_type=str(cfg.activation_type),
        decoder_type=str(cfg.decoder_type), decoder_base_ch=cfg.decoder_base_ch,
        decoder_num_res_blocks=cfg.decoder_num_res_blocks,
        recon_loss_type="amse" if mode == "amse" else "mse",
    )
    current = weight.detach().cpu().float().contiguous()
    all_bits, all_decoders, all_metas, stage_records = [], [], [], []
    started = time.perf_counter()
    for stage in range(2):
        data = materialize_prepared_group_data(
            prepared_entries=prep, intra_parallel=(1, 1), codebook_dim=32,
            batch_size=batch_size, normalize_weight=False, recon_loss_type="mse",
            train_device=device, split_weights_by_linear=[current], shuffle_seed=seed + stage,
        )
        residual = data.stacked_data.detach().clone().contiguous()
        mean, scale = _compute_stage_norm_stats(residual)
        train_cpu = _apply_stage_norm(residual, mean=mean, scale=scale)
        train_gpu = train_cpu.to(device=device, dtype=train_dtype).contiguous()
        stage_args = _clone_namespace(shared)
        vae = MultiLayerVAE(stage_args).to(device)
        initial_digest = _state_digest(vae)
        optimizer = create_optimizer(vae.parameters(), stage_args, stage_args.lr)
        scheduler = get_scheduler(
            str(stage_args.lr_scheduler), optimizer,
            num_warmup_steps=int(stage_args.lr_warmup_steps), num_training_steps=steps,
        )
        generator = torch.Generator().manual_seed(seed + stage)
        order = None
        pos = 0
        for step in range(steps):
            if order is None or pos >= len(train_gpu):
                order = torch.randperm(len(train_gpu), generator=generator, dtype=torch.long)
                pos = 0
            stop = min(pos + batch_size, len(train_gpu))
            ids = order[pos:stop].to(device=device)
            pos = stop
            x = train_gpu.index_select(0, ids)
            diagonal = None
            if mode == "amse":
                diagonal = gather_grams(grams, ids).diagonal(dim1=-2, dim2=-1).unsqueeze(1).to(train_dtype)
            optimizer.zero_grad(set_to_none=True)
            x_recon, losses = vae(x, is_train=True, act_max=diagonal)
            if mode == "block_output":
                recon = block_output_mse(x_recon, x, grams, ids)
                losses["train/recon_loss"] = recon * vae.model.l1_weight * vae.model.num_models
                losses["loss"] = losses["train/recon_loss"] + losses["train/commitment_loss"]
            losses["loss"].backward()
            optimizer.step()
            scheduler.step()
            if (step + 1) % 100 == 0 or step + 1 == steps:
                value = float(losses["loss"].detach())
                if not math.isfinite(value):
                    raise RuntimeError("Nonfinite training loss.")
                print(f"TRAIN {mode} {module_name} stage={stage+1} step={step+1}/{steps} loss={value:.6g}", flush=True)
        del optimizer, scheduler
        vae.eval()
        recons, bits = [], []
        with torch.no_grad():
            for offset in range(0, len(train_gpu), batch_size):
                out, codes = vae(train_gpu[offset:offset + batch_size], is_train=False)
                recons.append(out.detach().to(device="cpu", dtype=residual.dtype))
                bits.append(codes.detach().cpu())
        recon_norm = torch.cat(recons)
        recon = _restore_stage_norm(recon_norm, mean=mean, scale=scale)
        current = (residual - recon).reshape_as(weight).contiguous()
        decoder = vae.model.decoder.get_sub_decoder(0)
        _fuse_q_scale_into_decoder(decoder, q_scale=1.0 / math.sqrt(32))
        _fuse_norm_into_decoder(decoder, mean=float(mean[0]), std=float(scale[0]))
        all_bits.append(torch.cat(bits))
        all_decoders.append([decoder.cpu()])
        all_metas.append(data.split_metas)
        stage_records.append({
            "stage": stage, "steps": steps, "initial_state_sha256": initial_digest,
            "normalization_mean": float(mean[0]), "normalization_scale": float(scale[0]),
        })
        del vae, train_gpu, residual, train_cpu, data, recons, bits, x, x_recon, losses
        torch.cuda.empty_cache()
    payload = {
        "format": "vaellm_group_vae_payload", "version": 1,
        "target_common_split_metas": all_metas[0], "parts_per_linear": 1,
        "row_parts": 1, "col_parts": 1, "residual_stages": 2,
        "all_stage_bits": all_bits, "all_stage_decoders": all_decoders,
        "all_stage_codebook_dims": [32, 32], "all_stage_split_metas": all_metas,
        "protected_channel_quant_format": "none", "weight_rotation_specs": [None],
    }
    apply_group_vae_payload(model=host, group_refs=refs, group_tag=module_name, payload=payload, convert_device=device)
    compressed = host.get_submodule(module_name).to(device)
    with torch.no_grad():
        decoded = compressed._decode_weight(dtype=torch.float32).detach().cpu()
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("Nonfinite production-decoded weight.")
    state = {k: v.detach().cpu() for k, v in compressed.state_dict().items()}
    packed_bytes = sum(
        compressed.get_stage_part_vq_storage(stage_idx=s, part_idx=0).numel() for s in range(2)
    )
    record = {
        "module": module_name, "mode": mode, "seed": seed, "shape": list(weight.shape),
        "transpose": False, "rotation": "none", "channel_protection": "none",
        "steps_per_stage": steps, "batch_size": batch_size, "stages": stage_records,
        "code_payload_bpw": packed_bytes * 8 / weight.numel(),
        "state_bpw": sum(t.numel() * t.element_size() for t in state.values()) * 8 / weight.numel(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if save_path is not None:
        torch.save({"metadata": record, "state_dict": state}, save_path)
    del compressed, host, payload, all_decoders, all_bits
    torch.cuda.empty_cache()
    return decoded, record
