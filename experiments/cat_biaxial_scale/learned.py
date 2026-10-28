"""Shared learned scales with live inputs and a frozen first-stage code/decoder."""
from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import logging
import math

import torch
from torch import nn
from transformers import get_scheduler

from litebsq.llm_vae import MultiLayerVAE
from train_utils.cat_data_prep import materialize_prepared_group_data
from train_utils.cat_train_data import compute_stage_norm_stats
from train_utils.cat_train_pipeline import (_build_vae_linear_from_stage_payload,
    _fuse_norm_into_decoder, _fuse_q_scale_into_decoder)
from train_utils.train_args import create_optimizer

from .scaling import BiaxialScale, block_scale_factors
from .training import BATCH, DIM, SEED, prepare_scaled, vae_model_args

LOG = logging.getLogger(__name__)


class LearnedBiaxialScale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer('row_initial', scale.row.detach().clone())
        self.register_buffer('column_initial', scale.column.detach().clone())
        self.row_delta = nn.Parameter(torch.zeros_like(scale.row, dtype=torch.float32))
        self.column_delta = nn.Parameter(torch.zeros_like(scale.column, dtype=torch.float32))

    def effective(self):
        result = []
        for axis in ('row', 'column'):
            base, delta = getattr(self, axis + '_initial'), getattr(self, axis + '_delta')
            raw = (base.float().log() + delta - delta.mean()).exp()
            rounded = raw.to(torch.bfloat16).float()
            # BF16 deployment values in the forward pass; FP32 log-master gradient.
            result.append(raw + (rounded - raw).detach())
        return tuple(result)

    def block_factors(self, indices, *, transpose, dim=DIM):
        row, column = self.effective()
        return block_scale_factors(row, column, indices, transpose=transpose, dim=dim)

    def snapshot(self):
        row, column = self.effective()
        return BiaxialScale(row.detach().to(torch.bfloat16), column.detach().to(torch.bfloat16))


def block_view(weight, transpose):
    return (weight.T.contiguous() if transpose else weight).reshape(-1, 1, DIM)


def codes_digest(bits):
    return hashlib.sha256(bits.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def native_stage_outputs(refs, metas, bits, decoders, codebook_bits, device):
    """Read the frozen first-stage function through the production packed decoder."""
    outputs = []
    with torch.random.fork_rng(devices=[]):
        for model, (ref, meta, decoder) in enumerate(zip(refs, metas, decoders)):
            layer = _build_vae_linear_from_stage_payload(old_module=ref.module, transpose=ref.transpose,
                split_meta=meta, stage_split_metas=[meta],
                stage_part_bits_payload=[bits[:, model, :].unsqueeze(1)],
                stage_part_decoders_payload=[deepcopy(decoder)], stage_codebook_dims=[DIM],
                parallel_rows=1, parallel_cols=1, parallel_parts=1, bias=None,
                protected_channel_quant_format='int8' if meta.protected_input_indices is not None else 'none',
                weight_rotation_spec=None).to(device).eval().requires_grad_(False)
            layer._decoder_compute_dtype = torch.bfloat16
            outputs.append(layer._decode_compressed_weight(dtype=torch.bfloat16).float())
            del layer
    return outputs


def fit_learned(refs, plans, config, vae_args, device, steps):
    entries, initial_scales, specs, rotated_weights = prepare_scaled(refs, plans, config, device, retain_rotated=True)
    material = materialize_prepared_group_data(prepared_entries=entries, intra_parallel=(1, 1),
        codebook_dim=DIM, batch_size=BATCH, normalize_weight=False, recon_loss_type='mse',
        train_device=str(device), shuffle_seed=SEED)
    metas = material.split_metas
    del material
    original = torch.cat([block_view(value.to(device), ref.transpose)
        for value, ref in zip(rotated_weights, refs)], dim=1)
    learners = nn.ModuleList([LearnedBiaxialScale(scale).to(device) for scale in initial_scales])
    prior = torch.zeros_like(original)
    args = vae_model_args(vae_args, config, len(refs))
    all_bits, all_decoders, reports = [], [], []
    indices_all_cpu = torch.arange(len(original))
    for stage in range(config.residual_stages):
        # Stage statistics are frozen at this boundary; only the per-batch U/S
        # target and its derivative are live during optimization.
        with torch.no_grad():
            factors_initial = torch.cat([scale.block_factors(indices_all_cpu.to(device), transpose=ref.transpose)
                for scale, ref in zip(learners, refs)], dim=1)
            initial_residual = original / factors_initial - prior
            mean, std = compute_stage_norm_stats(initial_residual.cpu())
            _, physical_std = compute_stage_norm_stats((initial_residual * factors_initial).cpu())
        mean_gpu, std_gpu, physical_std_gpu = mean.to(device), std.to(device), physical_std.to(device)
        vae = MultiLayerVAE(deepcopy(args)).to(device).train()
        initial_parameters = {key: value.detach().clone() for key, value in vae.named_parameters()}
        initial_scale = [scale.snapshot() for scale in learners]
        # Reset Adam state at each native residual-stage boundary, preserving
        # learned master coordinates and the first-stage frozen payload.
        optimizer = create_optimizer([*vae.parameters(), *learners.parameters()], args, args.lr)
        scheduler = get_scheduler(args.lr_scheduler, optimizer,
            num_warmup_steps=args.lr_warmup_steps, num_training_steps=steps)
        generator = torch.Generator().manual_seed(SEED + stage)
        order, position = None, 0
        trajectory = []
        for step in range(steps):
            if order is None or position >= len(original):
                order = torch.randperm(len(original), generator=generator); position = 0
            indices = order[position:position + BATCH].to(device); position += len(indices)
            factors = torch.cat([scale.block_factors(indices, transpose=ref.transpose)
                for scale, ref in zip(learners, refs)], dim=1)
            residual = original.index_select(0, indices) / factors - prior.index_select(0, indices)
            target_float = (residual - mean_gpu.reshape(1, -1, 1)) / std_gpu.reshape(1, -1, 1)
            x = target_float.to(torch.bfloat16)
            optimizer.zero_grad(set_to_none=True)
            decoded, native = vae(x, is_train=True)
            physical = factors * (prior.index_select(0, indices)
                + mean_gpu.reshape(1, -1, 1) + std_gpu.reshape(1, -1, 1) * decoded.float())
            recon = ((physical - original.index_select(0, indices))
                / physical_std_gpu.reshape(1, -1, 1)).square().mean()
            loss = args.l1_weight * len(refs) * recon + native['train/commitment_loss']
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite learned-scale objective.')
            loss.backward()
            observe_step = step == 0 or (step + 1) % 100 == 0 or step == steps - 1
            if observe_step and not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                        for p in [*vae.parameters(), *learners.parameters()]):
                raise FloatingPointError('Invalid learned-scale native codec/master gradient.')
            scale_gradient = math.sqrt(sum(float(p.grad.square().sum()) for p in learners.parameters())) if observe_step else None
            if step == 0 and scale_gradient == 0:
                raise RuntimeError('The shared scale has no gradient through the live target.')
            optimizer.step(); scheduler.step()
            if observe_step:
                row = dict(step=step + 1, loss=float(loss.detach()), reconstruction=float(recon.detach()),
                    auxiliary=float(native['train/commitment_loss'].detach()), scale_gradient_l2=scale_gradient,
                    lr=scheduler.get_last_lr()[0])
                trajectory.append(row); LOG.info('learned stage%d %s', stage + 1, row)
            del factors, residual, target_float, x, decoded, native, recon, loss, physical, indices
        changes = {group: sum(int((value.detach() != initial_parameters[key]).sum())
            for key, value in vae.named_parameters() if key.startswith('model.' + group + '.'))
            for group in ('encoder', 'decoder')}
        if not all(changes.values()):
            raise RuntimeError('No surviving BF16 encoder or decoder update.')
        del optimizer, scheduler, initial_parameters, initial_residual, factors_initial
        vae.eval(); bits, input_cast_sse = [], torch.zeros(len(refs), device=device, dtype=torch.float64)
        with torch.no_grad():
            for index_chunk in indices_all_cpu.split(BATCH):
                index = index_chunk.to(device)
                factors = torch.cat([scale.block_factors(index, transpose=ref.transpose)
                    for scale, ref in zip(learners, refs)], dim=1)
                residual = original.index_select(0, index) / factors - prior.index_select(0, index)
                target_float = (residual - mean_gpu.reshape(1, -1, 1)) / std_gpu.reshape(1, -1, 1)
                x = target_float.to(torch.bfloat16)
                _, code = vae(x, is_train=False)
                bits.append(code.cpu())
                error = (x.float() - target_float) * std_gpu.reshape(1, -1, 1) * factors
                input_cast_sse += error.double().square().sum(dim=(0, 2))
            bits = torch.cat(bits)
            decoders = []
            for model in range(len(refs)):
                decoder = vae.model.decoder.get_sub_decoder(model)
                _fuse_q_scale_into_decoder(decoder, q_scale=1 / math.sqrt(config.codebook_bits))
                _fuse_norm_into_decoder(decoder, mean=float(mean[model]), std=float(std[model]))
                decoders.append(decoder.cpu())
        all_bits.append(bits); all_decoders.append(decoders)
        frozen_digest = codes_digest(all_bits[0])
        decoded_stage = native_stage_outputs(refs, metas, bits, decoders, config.codebook_bits, device)
        prior.add_(torch.cat([block_view(value, ref.transpose) for value, ref in zip(decoded_stage, refs)], dim=1))
        snapshots = [scale.snapshot() for scale in learners]
        scale_changes = {ref.name: {axis: int((getattr(before, axis) != getattr(after, axis)).sum())
            for axis in ('row', 'column')} for ref, before, after in zip(refs, initial_scale, snapshots)}
        reports.append(dict(stage=stage + 1, normalized_mean=mean.flatten().tolist(), normalized_std=std.flatten().tolist(),
            physical_std=physical_std.flatten().tolist(), bf16_changed_elements=changes,
            bf16_scale_changed_elements=scale_changes, input_cast_physical_sse=input_cast_sse.cpu().tolist(),
            first_stage_code_sha256=frozen_digest, scale_optimizer_reset=True, trajectory=trajectory,
            target_boundary='Encoder consumes live BF16(U/currentS - frozen priorV - stage_mean)/stage_std; reconstruction compares currentS*(priorV+mean+std*decoded) to original FP32 U. All scale/input paths remain differentiable.'))
        if codes_digest(all_bits[0]) != reports[0]['first_stage_code_sha256']:
            raise RuntimeError('Frozen first-stage codes changed during the second stage.')
        del vae, decoded_stage, x, code, error, factors, residual, target_float, initial_scale, snapshots
        gc.collect(); torch.cuda.empty_cache()
    payload = dict(format='vaellm_group_vae_payload', version=1, target_common_split_metas=metas,
        parts_per_linear=1, row_parts=1, col_parts=1, residual_stages=config.residual_stages,
        all_stage_bits=all_bits, all_stage_decoders=all_decoders, all_stage_codebook_dims=[DIM] * config.residual_stages,
        all_stage_split_metas=[metas] * config.residual_stages,
        protected_channel_quant_format='int8' if plans is not None else 'none', weight_rotation_specs=specs)
    return payload, [scale.snapshot().cpu() for scale in learners], reports
