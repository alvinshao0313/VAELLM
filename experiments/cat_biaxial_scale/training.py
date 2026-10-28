"""Native sampled CAT control and fixed-scale reconstruction in the same BSQ population."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import gc
import logging
import math
import time

import torch
from transformers import get_scheduler

from experiments.rotation_factor_probe.train import host_for_weight, recipe as native_recipe
from litebsq.llm_vae import MultiLayerVAE
from litebsq.vae_args import apply_autoencoder_arch_defaults
from rotation.weight_preprocess import build_weight_rotation, rotation_spec
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_data_prep import LinearPrepRef, materialize_prepared_group_data, prepare_group_linear_entries
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_data import apply_stage_norm, compute_stage_norm_stats, restore_stage_norm
from train_utils.cat_train_pipeline import (_fuse_norm_into_decoder, _fuse_q_scale_into_decoder,
    _group_data_to_compressed_weights, train_group_vae_payload)
from train_utils.train_args import create_optimizer
from train_utils.utils import LinearRef, set_seed

from .scaling import BiaxialScale, weighted_reconstruction_loss

SEED, BATCH, DIM = 31, 8192, 32
LOG = logging.getLogger(__name__)


def recipe(model_path, category, steps, protected):
    argv, _, _, _ = native_recipe(model_path, 'down_proj' if category == 'down_proj' else 'gate_proj',
                                  steps=steps, seed=SEED)
    argv[argv.index('--compression_categories') + 1] = category
    argv[argv.index('--weight_rotation') + 1] = 'two_sided'
    argv += ['--weight_rotation_block_size', '32']
    if protected:
        argv[argv.index('--channel_protect_mode') + 1] = 'channel'
        argv += ['--channel_protect_count', 'default=' + ('256' if category == 'down_proj' else '64'),
                 '--channel_quant', 'int8']
    cat, _, train, vae = parse_cat_runtime_args(argv)
    config = resolve_category_runtime_configs(cat, vae, [category])[category]
    return argv, config, train, vae


def build_host(targets, category):
    names = sorted(targets, key=lambda name: int(name.split('.')[2]))
    host, _ = host_for_weight(targets[names[0]]['reference_weight'], names[0])
    refs = []
    for name in names:
        if name != names[0]:
            other, _ = host_for_weight(targets[name]['reference_weight'], name)
            # Different layers share only their leading model/layers containers.
            parent_name = name.rsplit('.', 2)[0]
            parent = host
            for part in parent_name.split('.'):
                if not hasattr(parent, part):
                    parent.add_module(part, torch.nn.Module())
                parent = getattr(parent, part)
            parent.add_module('mlp', other.get_submodule(parent_name + '.mlp'))
        refs.append(LinearRef(name, host.get_submodule(name), category, category == 'down_proj'))
    return host, refs


def fit(targets, model_path, category, normalization, protection, steps, device):
    protected = protection == 'int8'
    argv, config, train_args, vae_args = recipe(model_path, category, steps, protected)
    host, refs = build_host(targets, category)
    plans = {name: item['protected_input_indices'] for name, item in targets.items()} if protected else None
    set_seed(SEED)
    started = time.time()
    if normalization == 'current':
        payload = train_group_vae_payload(model=host, group_refs=refs, group_tag=category,
            runtime_cfg=config, vae_args=vae_args, training_args=train_args,
            train_device=str(device), convert_device=str(device), do_convert=True,
            batch_size=BATCH, gpu_resident_data=True, log_every=100 if steps > 2 else 1,
            eval_every=0, eval_blocks=256, channel_protect_mode='channel' if protected else 'none',
            channel_plan=plans, channel_rank_metric='channel_weight_abs', channel_axis='input',
            channel_quant='int8' if protected else 'none', deterministic=True, shuffle_seed=SEED)
        scales, stages = [None] * len(refs), []
    elif normalization == 'biaxial':
        payload, scales, stages = fit_biaxial(refs, plans, config, vae_args, device, steps)
    elif normalization == 'learned':
        from .learned import fit_learned
        payload, scales, stages = fit_learned(refs, plans, config, vae_args, device, steps)
    else:
        raise ValueError('Unknown normalization: ' + normalization)
    return host, refs, payload, scales, dict(native_argument_recipe=argv,
        resolved_category=asdict(config), normalization=normalization, protection=protection,
        names=[ref.name for ref in refs], independent_linears=len(refs),
        batch_vectors_per_linear=BATCH, steps_per_stage=steps, seed=SEED,
        elapsed_seconds=time.time() - started, stages=stages,
        boundary='Each category uses the same M3 population across comparison arms; historical CAT M36 is a different BSQ marginal population. Parsed target_layers0 is only a constructor template; the actual refs are the listed layers0/18/35.')


def fit_biaxial(refs, plans, config, vae_args, device, steps):
    entries, scales, specs, _ = prepare_scaled(refs, plans, config, device)
    return fit_fixed_prepared(refs, plans, config, vae_args, device, steps, entries, scales, specs)


def prepare_scaled(refs, plans, config, device, *, retain_rotated=False):
    protected = plans is not None
    entries = prepare_group_linear_entries(group_refs=[LinearPrepRef(r.name, r.module.weight,
        r.module.in_features, r.module.out_features, r.transpose) for r in refs],
        activation_weight_by_linear=None, channel_protect_count=config.channel_protect_count if protected else 0,
        channel_axis='input', channel_quant='int8' if protected else 'none', recon_loss_type='mse',
        channel_plan=plans, apply_outlier_channel_removal=True)
    specs, scales, rotated_weights = [], [], []
    for i, entry in enumerate(entries):
        prepared = entry.prepared_weight
        spec = rotation_spec('two_sided', 32, SEED, entry.ref.name)
        rotation = build_weight_rotation(prepared.compressed_out_features, prepared.compressed_in_features, spec).to(device)
        with torch.no_grad():
            rotated = rotation(prepared.split_weight.to(device=device, dtype=torch.float32))
            scale = BiaxialScale.from_rotated(rotated)
            normalized = scale.normalize(rotated).cpu()
        entries[i] = replace(entry, prepared_weight=replace(prepared, split_weight=normalized))
        specs.append(spec); scales.append(scale.cpu())
        if retain_rotated:
            rotated_weights.append(rotated.cpu())
        del rotation, rotated, normalized
    return entries, scales, specs, rotated_weights


def vae_model_args(vae_args, config, num_models):
    args = deepcopy(vae_args)
    for key in ('residual_stages', 'codebook_bits', 'codebook_dim', 'base_ch', 'num_res_blocks',
                'norm_type', 'activation_type', 'decoder_type', 'decoder_base_ch', 'decoder_num_res_blocks',
                'recon_loss_type'):
        setattr(args, key, getattr(config, key))
    args.parallel_layers = num_models
    apply_autoencoder_arch_defaults(args)
    return args


def fit_fixed_prepared(refs, plans, config, vae_args, device, steps, entries, scales, specs):
    protected = plans is not None
    material_kwargs = dict(prepared_entries=entries, intra_parallel=(1, 1), codebook_dim=DIM,
        batch_size=BATCH, normalize_weight=False, recon_loss_type='mse', train_device=str(device),
        intra_part_sort_mode='none')
    common = materialize_prepared_group_data(**material_kwargs, shuffle_seed=SEED)
    common_metas = common.split_metas
    residual_weights = [entry.prepared_weight.split_weight.detach().cpu() for entry in entries]
    args = vae_model_args(vae_args, config, len(refs))
    all_bits, all_decoders, stage_metas, reports = [], [], [], []
    gpu_scales = [scale.to(device) for scale in scales]
    for stage in range(config.residual_stages):
        material = materialize_prepared_group_data(**material_kwargs, split_weights_by_linear=residual_weights,
            shuffle_seed=SEED + stage)
        residual = material.stacked_data.detach().clone().contiguous()
        mean, std = compute_stage_norm_stats(residual)
        physical_std = torch.stack([scale.restore(value.to(device)).std().clamp_min(1e-6).cpu()
            for scale, value in zip(gpu_scales, residual_weights)]).reshape(-1, 1)
        blocks_cpu = apply_stage_norm(residual, mean=mean, scale=std)
        blocks = blocks_cpu.to(device=device, dtype=torch.bfloat16)
        std_gpu, physical_std_gpu = std.to(device), physical_std.to(device)
        vae = MultiLayerVAE(deepcopy(args)).to(device).train()
        initial = {k: p.detach().clone() for k, p in vae.named_parameters()}
        optimizer = create_optimizer(vae.parameters(), args, args.lr)
        scheduler = get_scheduler(args.lr_scheduler, optimizer,
            num_warmup_steps=args.lr_warmup_steps, num_training_steps=steps)
        generator = torch.Generator().manual_seed(SEED + stage)
        order, position = None, 0
        trajectory = []
        for step in range(steps):
            if order is None or position >= len(blocks):
                order = torch.randperm(len(blocks), generator=generator)
                position = 0
            indices = order[position:position + BATCH].to(device)
            position += len(indices)
            x = blocks.index_select(0, indices)
            factors = torch.cat([scale.block_factors(indices, transpose=ref.transpose, dim=DIM)
                for scale, ref in zip(gpu_scales, refs)], dim=1)
            optimizer.zero_grad(set_to_none=True)
            decoded, loss_dict = vae(x, is_train=True)
            recon = weighted_reconstruction_loss(decoded, x, factors, std_gpu, physical_std_gpu)
            loss = args.l1_weight * len(refs) * recon + loss_dict['train/commitment_loss']
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite scaled reconstruction loss.')
            loss.backward()
            if step in (0, steps - 1):
                if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in vae.parameters()):
                    raise FloatingPointError('Invalid native BSQ encoder/decoder gradient.')
            optimizer.step(); scheduler.step()
            if step == 0 or (step + 1) % 100 == 0 or step == steps - 1:
                row = dict(step=step + 1, loss=float(loss.detach()), reconstruction=float(recon.detach()),
                    auxiliary=float(loss_dict['train/commitment_loss'].detach()), lr=scheduler.get_last_lr()[0])
                trajectory.append(row); LOG.info('biaxial stage%d %s', stage + 1, row)
            del x, factors, decoded, loss, loss_dict, recon, indices
        changes = {group: sum(int((p.detach() != initial[k]).sum()) for k, p in vae.named_parameters()
            if k.startswith('model.' + group + '.')) for group in ('encoder', 'decoder')}
        if not all(changes.values()):
            raise RuntimeError('No surviving native BF16 encoder or decoder update.')
        del optimizer, scheduler, initial
        vae.eval(); reconstructions, bits = [], []
        with torch.no_grad():
            for chunk in blocks.split(BATCH):
                reconstructed, code = vae(chunk, is_train=False)
                reconstructions.append(reconstructed.to(device='cpu', dtype=residual.dtype))
                bits.append(code.cpu())
        reconstructed = restore_stage_norm(torch.cat(reconstructions), mean=mean, scale=std)
        following = (residual - reconstructed).contiguous()
        residual_weights = _group_data_to_compressed_weights(stacked_data=following, split_metas=common_metas)
        cast_error = blocks.cpu().float() - blocks_cpu
        cast_physical_sse = []
        cast_weights = _group_data_to_compressed_weights(stacked_data=cast_error * std.reshape(1, -1, 1), split_metas=common_metas)
        for scale, value in zip(gpu_scales, cast_weights):
            cast_physical_sse.append(float(scale.restore(value.to(device)).double().square().sum()))
        reports.append(dict(stage=stage + 1, normalized_mean=mean.flatten().tolist(),
            normalized_std=std.flatten().tolist(), physical_std=physical_std.flatten().tolist(),
            input_cast_physical_sse=cast_physical_sse, bf16_changed_elements=changes, trajectory=trajectory,
            residual_rms_after=following.square().mean(dim=(0, 2)).sqrt().tolist(),
            target_boundary='Native BF16 normalized input; weighted loss maps its error to physical rotated coordinates, not to the unrounded FP32 target.'))
        decoders = []
        for model in range(len(refs)):
            decoder = vae.model.decoder.get_sub_decoder(model)
            _fuse_q_scale_into_decoder(decoder, q_scale=1 / math.sqrt(config.codebook_bits))
            _fuse_norm_into_decoder(decoder, mean=float(mean[model]), std=float(std[model]))
            decoders.append(decoder.cpu())
        all_bits.append(torch.cat(bits)); all_decoders.append(decoders); stage_metas.append(material.split_metas)
        del vae, material, blocks, blocks_cpu, reconstructed, reconstructions, following, residual, cast_error, cast_weights
        gc.collect(); torch.cuda.empty_cache()
    payload = dict(format='vaellm_group_vae_payload', version=1, target_common_split_metas=common_metas,
        parts_per_linear=1, row_parts=1, col_parts=1, residual_stages=config.residual_stages,
        all_stage_bits=all_bits, all_stage_decoders=all_decoders,
        all_stage_codebook_dims=[DIM] * config.residual_stages, all_stage_split_metas=stage_metas,
        protected_channel_quant_format='int8' if protected else 'none', weight_rotation_specs=specs)
    return payload, [scale.cpu() for scale in gpu_scales], reports
