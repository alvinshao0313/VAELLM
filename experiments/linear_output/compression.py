"""Residual-stage orchestration for one full, non-transposed Linear."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from train_utils.cat_data_prep import LinearPrepRef, prepare_group_linear_entries, materialize_prepared_group_data
from train_utils.cat_train_data import compute_stage_norm_stats, apply_stage_norm, restore_stage_norm
from train_utils.cat_train_pipeline import _fuse_norm_into_decoder, _fuse_q_scale_into_decoder

from .artifacts import dump_json, export_linear, tensor_digest
from .config import stage_steps, validate
from .training import optimize_stage


def train_linear(linear: nn.Linear, *, name: str, args,
                 next_inputs: Callable[[str], torch.Tensor], output: Path) -> dict:
    validate(args)
    if not isinstance(linear, nn.Linear) or linear.in_features % 32:
        raise ValueError("Only full nn.Linear matrices with input width divisible by 32 are supported.")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    original = linear.weight.detach().cpu().float().contiguous()
    if not torch.isfinite(original).all():
        raise ValueError(f"Nonfinite source weights: {name}")
    prep = prepare_group_linear_entries(
        group_refs=[LinearPrepRef(name, original, linear.in_features, linear.out_features, False)],
        activation_weight_by_linear=None, channel_protect_count=0, channel_axis="input",
        recon_loss_type="mse", apply_outlier_channel_removal=False,
    )
    data = materialize_prepared_group_data(
        prepared_entries=prep, intra_parallel=(1, 1), codebook_dim=32,
        batch_size=args.vae_chunk_vectors, normalize_weight=False, recon_loss_type="mse",
        train_device="cpu", split_weights_by_linear=[original], shuffle_seed=args.seed,
    )
    split_metas = data.split_metas
    residual = data.stacked_data.detach().clone().contiguous()
    if not torch.equal(residual.reshape_as(original), original):
        raise RuntimeError("Preparation changed the row-major input-axis layout.")
    del data
    counts = stage_steps(args.steps, args.residual_stages)
    all_bits, all_decoders, records = [], [], []
    expected_weight = torch.zeros_like(original)
    offset = 0
    with (output / "training.jsonl").open("x", encoding="utf-8") as log:
        for stage, count in enumerate(counts):
            if args.normalize_weight:
                mean, scale = compute_stage_norm_stats(residual)
            else:
                mean, scale = torch.zeros(1, 1), torch.ones(1, 1)
            normalized = apply_stage_norm(residual, mean=mean, scale=scale)
            vae, decoded, bits, initial_hash = optimize_stage(
                normalized, residual.reshape_as(original), mean, scale,
                name=name, stage=stage, count=count, offset=offset, args=args,
                next_inputs=next_inputs, log=log,
            )
            stage_recon = restore_stage_norm(decoded, mean=mean, scale=scale)
            expected_weight.add_(stage_recon.reshape_as(original))
            residual = (residual - stage_recon).contiguous()
            decoder = vae.model.decoder.get_sub_decoder(0)
            q_scale = 1 / math.sqrt(args.codebook_bits) if args.new_quant else 1.0
            _fuse_q_scale_into_decoder(decoder, q_scale=q_scale)
            _fuse_norm_into_decoder(decoder, mean=float(mean.item()), std=float(scale.item()))
            all_decoders.append([decoder.cpu()])
            all_bits.append(bits)
            records.append(dict(stage=stage, steps=count, initial_state_sha256=initial_hash,
                                normalization_mean=float(mean.item()), normalization_scale=float(scale.item())))
            offset += count
            del vae, normalized, stage_recon, decoded
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
    payload = dict(
        format="vaellm_group_vae_payload", version=1, target_common_split_metas=split_metas,
        parts_per_linear=1, row_parts=1, col_parts=1, residual_stages=args.residual_stages,
        all_stage_bits=all_bits, all_stage_decoders=all_decoders,
        all_stage_codebook_dims=[32] * args.residual_stages,
        all_stage_split_metas=[split_metas] * args.residual_stages,
        protected_channel_quant_format="none", weight_rotation_specs=[None],
    )
    parity = export_linear(name, linear, payload, expected_weight, output / "packed", args)
    record = dict(module=name, shape=list(original.shape), objective=args.objective,
                  transpose=False, rotation="none", protection="none", steps=offset,
                  batch_size=args.batch_size, stages=records, source_weight_sha256=tensor_digest(linear.weight),
                  elapsed_seconds=time.perf_counter() - started, **parity)
    dump_json(output / "record.json", record)
    return record
