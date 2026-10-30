"""Hand the actual fresh merged dense up category to the unchanged N fitter."""
from dataclasses import replace
import gc
from pathlib import Path

import torch

from experiments.adaptive_vector_rvq import common as c
from train_utils import cat_train_pipeline as native
from train_utils.cat_runtime_state_v6 import restore_cat_runtime_state
from train_utils.cat_data_prep import LinearPrepRef, build_outlier_channel_index_plan
from . import checkpoint, RUNTIME
from .recovery import CATEGORIES


def fit_live_up(model, cat, hf, training, vae, cfgs, raw_runtime, output):
    c.require(Path(native.__file__).resolve().is_relative_to(RUNTIME), 'Use only the actual N category-aware fitter.')
    c.require(cat.channel_scope == 'layer', 'This narrow handoff keeps the actual existing layer protection policy.')
    pairs, missing = native._collect_sorted_category_refs(model, category='up_proj',
        transpose_modules=cat.transpose_modules.split(','), only_decoder_projections=True, compression_categories=CATEGORIES)
    refs = [ref for _index, ref in pairs]
    c.require(not missing and len(refs) == 36 and all(type(ref.module) is torch.nn.Linear for ref in refs),
              'The next category must be all36 actual fresh merged ordinary up projections.')
    target = {ref.name: ref.module.weight for ref in refs}
    initial_target_digest = c.digest(target)
    gates = {name: tensor for name, tensor in model.state_dict().items() if '.mlp.gate_proj.' in name}
    gate_digest = c.digest(gates)
    activation_runtime, plan, identity = restore_cat_runtime_state(raw_runtime, access_token=hf.access_token)
    c.require(plan is None, 'This raw branch uses the actual layer protection path, not a new global allocation.')
    activation_runtime, plan, _budget = native._refresh_channel_protection_after_category(model=model,
        current_category='gate_proj', current_category_idx=4, active_categories=CATEGORIES, cat_args=cat,
        resolved_category_cfgs=cfgs, transpose_modules=cat.transpose_modules.split(','), only_decoder_projections=True,
        compression_categories=CATEGORIES, target_layers='all', skip_layer_keys=set(), activation_runtime=activation_runtime,
        global_adaptive_plan=plan, global_raw_budget_value=0, resolved_channel_mode=cat.channel_protect_mode,
        resolved_channel_rank_metric=cat.channel_rank_metric, resolved_channel_mlp_rank_metric=cat.channel_mlp_rank_metric,
        channel_axis=cat.channel_axis, category_channel_protect_count={name: cfg.channel_protect_count for name, cfg in cfgs.items()},
        linear_group_size=36, plan_is_main=True, plan_world_size=1, run_output_dir=str(output), logger=native.log)
    c.require(c.digest(target) == initial_target_digest, 'Activation-statistics refresh changed a live dense target.')
    cfg = cfgs['up_proj']
    c.require(cfg.codebook_dim == cfg.codebook_bits == 32 and cfg.residual_stages == 2,
              'Keep the actual original D32K32/S2 up policy.')
    actmax, actmean = native._activation_views_for_refs(refs, activation_runtime, category='up_proj', rank_metric=cat.channel_rank_metric)
    protection = build_outlier_channel_index_plan(group_refs=[LinearPrepRef(name=ref.name, weight=ref.module.weight,
        in_features=ref.module.in_features, out_features=ref.module.out_features, transpose=ref.transpose) for ref in refs],
        activation_weight_by_linear=actmax, activation_abs_mean_by_linear=actmean,
        channel_protect_count=cfg.channel_protect_count, channel_axis=cat.channel_axis, channel_scope='layer',
        channel_rank_metric=cat.channel_rank_metric, channel_min_per_layer=cat.channel_min_per_layer)
    model.to('cpu')
    for module in model.modules():
        clear = getattr(module, 'clear_decoded_weight_cache', None)
        if callable(clear):
            clear()
    gc.collect()
    torch.cuda.empty_cache()
    c.require(c.digest(target) == initial_target_digest, 'The fitter target is not the exact fresh returned up weight state.')
    optimizer_rows = []
    factory = native.create_optimizer

    def observed_optimizer(*args, **kwargs):
        optimizer = factory(*args, **kwargs)
        row = dict(actual_updates=0, learning_rates=[], state_step_counters=[])
        optimizer_rows.append(row)

        def updated(current, _args, _kwargs):
            row['actual_updates'] += 1
            row['learning_rates'].append([group['lr'] for group in current.param_groups])
            row['state_step_counters'].append(sorted({int(state['step']) for state in current.state.values()}))
        optimizer.register_step_post_hook(updated)
        return optimizer

    try:
        native.create_optimizer = observed_optimizer
        payload = native.train_group_vae_payload(model=model, group_refs=refs, group_tag='mixed_handoff.up.L0-35',
            runtime_cfg=replace(cfg, steps=2), vae_args=vae, training_args=training,
            train_device=cat.train_device, convert_device=cat.convert_device, do_convert=True, batch_size=8192,
            log_every=cat.log_every, eval_every=cat.eval_every, eval_blocks=cat.eval_blocks,
            gpu_resident_data=cat.gpu_resident_data, activation_runtime=activation_runtime,
            channel_protect_mode=cat.channel_protect_mode, channel_plan=protection, channel_scope=cat.channel_scope,
            channel_rank_metric=cat.channel_rank_metric, channel_axis=cat.channel_axis, channel_quant=cat.channel_quant,
            channel_min_per_layer=cat.channel_min_per_layer, deterministic=cat.deterministic, shuffle_seed=500031)
    finally:
        native.create_optimizer = factory
    c.require(len(optimizer_rows) == 2 and all(row['actual_updates'] == 2 for row in optimizer_rows),
              'Both real next-up residual stages must execute two actual optimizer updates.')
    c.require(c.digest(target) == initial_target_digest, 'N fitting changed its dense source before actual native replacement.')
    native.apply_group_vae_payload(model=model, group_refs=refs, payload=payload,
        group_tag='mixed_handoff.up.L0-35', convert_device=cat.convert_device)
    c.require(c.digest({name: tensor for name, tensor in model.state_dict().items() if '.mlp.gate_proj.' in name}) == gate_digest,
              'The actual next-up fit/replacement changed compact gate decoders or frozen code/route/protection.')
    groups = checkpoint.inventories(model)
    c.require([len(groups[key]) for key in ('native', 'compact', 'pending_dense')] == [180, 36, 36], 'Incorrect actual next-up mixed inventory.')
    return dict(actual_fitter=str(Path(native.__file__).resolve()), source_target_digest=initial_target_digest,
        actual_updates=optimizer_rows, actual_layers=36, D=32, K=32, batch_vectors_per_linear=8192, stages=2,
        shuffle_seed=500031, source='Actual merged/fresh returned live up weights; no old after_o up or donor loaded',
        compact_gate_state_exact=True, inventories=groups, full_class_fit_only_two_updates=True,
        activation_refresh_enabled=cat.channel_refresh_after_category, channel_protection_count=cfg.channel_protect_count,
        rotation_block_size=cfg.weight_rotation_block_size, no_up500_recovery=True, no_quality_claim=True)
