"""Original live-down native fit with only the approved per-arm geometry change."""
from dataclasses import replace
import gc

from . import bindings as b
from experiments.adaptive_class_20261030 import initialization
from .validation import updates
from train_utils.cat_data_prep import LinearPrepRef, build_outlier_channel_index_plan


def execute(model, parsed, activation_runtime, plan, *, arm, output, short):
    cat,hf,training,vae,cfgs,_runtime = parsed
    native = b.recovery.native
    settings, category, index = b.ARMS[arm], 'down_proj', 6
    pairs,missing = native._collect_sorted_category_refs(model,category=category,
        transpose_modules=cat.transpose_modules.split(','),only_decoder_projections=True,compression_categories=b.recovery.CATEGORIES)
    refs = [ref for _layer,ref in pairs]
    b.require(not missing and len(refs) == 36 and all(type(ref.module) is b.torch.nn.Linear for ref in refs)
        and all(ref.transpose == settings['transpose'] for ref in refs),
        'Fit precisely the36 actual live dense down matrices with the selected direction.')
    targets = {ref.name:ref.module.weight for ref in refs}
    digest = b.c.digest(targets)
    prefix = lambda:{name:value for name,value in model.state_dict().items() if '.mlp.down_proj.' not in name}
    before = b.c.digest(prefix())
    prepared = b.read(b.RESULT / 'prepared.json')
    b.require(digest == prepared['live_down_source_digest'] and before == prepared['frozen_prefix_digest'],
        'The fork must start at the exact same live source and frozen prefix as original C0.')
    activation_runtime,plan,_budget = native._refresh_channel_protection_after_category(model=model,
        current_category='up_proj',current_category_idx=5,active_categories=b.recovery.CATEGORIES,cat_args=cat,
        resolved_category_cfgs=cfgs,transpose_modules=cat.transpose_modules.split(','),only_decoder_projections=True,
        compression_categories=b.recovery.CATEGORIES,target_layers='all',skip_layer_keys=set(),activation_runtime=activation_runtime,
        global_adaptive_plan=plan,global_raw_budget_value=0,resolved_channel_mode=cat.channel_protect_mode,
        resolved_channel_rank_metric=cat.channel_rank_metric,resolved_channel_mlp_rank_metric=cat.channel_mlp_rank_metric,
        channel_axis=cat.channel_axis,category_channel_protect_count={name:cfg.channel_protect_count for name,cfg in cfgs.items()},
        linear_group_size=36,plan_is_main=True,plan_world_size=1,run_output_dir=str(output),logger=native.log)
    cfg = cfgs[category]
    actmax,actmean = native._activation_views_for_refs(refs,activation_runtime,category=category,rank_metric=cat.channel_rank_metric)
    protection = build_outlier_channel_index_plan(group_refs=[LinearPrepRef(name=ref.name,weight=ref.module.weight,
        in_features=ref.module.in_features,out_features=ref.module.out_features,transpose=ref.transpose) for ref in refs],
        activation_weight_by_linear=actmax,activation_abs_mean_by_linear=actmean,channel_protect_count=cfg.channel_protect_count,
        channel_axis=cat.channel_axis,channel_scope='layer',channel_rank_metric=cat.channel_rank_metric,channel_min_per_layer=cat.channel_min_per_layer)
    model.to('cpu')
    for module in model.modules():
        clear = getattr(module,'clear_decoded_weight_cache',None)
        if callable(clear):
            clear()
    gc.collect()
    b.torch.cuda.empty_cache()
    b.require(b.c.digest(targets) == digest, 'Statistics refresh/offload changed the live dense source.')
    native.set_seed(cat.seed)
    with initialization.observe(native) as initializations, updates(native,enabled=short) as checks:
        payload = native.train_group_vae_payload(model=model,group_refs=refs,group_tag='whole_class.down_proj.L0-35',
            runtime_cfg=replace(cfg,steps=2) if short else cfg,vae_args=vae,training_args=training,
            train_device=cat.train_device,convert_device=cat.convert_device,do_convert=True,batch_size=settings['batch'],
            log_every=1 if short else cat.log_every,eval_every=cat.eval_every,eval_blocks=cat.eval_blocks,
            gpu_resident_data=cat.gpu_resident_data,activation_runtime=activation_runtime,
            channel_protect_mode=cat.channel_protect_mode,channel_plan=protection,channel_scope=cat.channel_scope,
            channel_rank_metric=cat.channel_rank_metric,channel_axis=cat.channel_axis,channel_quant=cat.channel_quant,
            channel_min_per_layer=cat.channel_min_per_layer,deterministic=cat.deterministic,shuffle_seed=600031)
    b.require(len(initializations) == 2 and b.c.digest(targets) == digest, 'Require both genuine stage initializations and unchanged dense source.')
    if short:
        b.require(len(checks) == 2 and all(len(stage['updates']) == 2 for stage in checks),
            'The real new path must complete two finite backward/optimizer updates per residual stage.')
    native.apply_group_vae_payload(model=model,group_refs=refs,payload=payload,
        group_tag='whole_class.down_proj.L0-35',convert_device=cat.convert_device)
    b.require(b.c.digest(prefix()) == before, 'Down fitting/replacement changed a frozen prefix or backbone field.')
    inventory = b.checkpoint.inventories(model)
    b.require([len(inventory[key]) for key in ('native','compact','pending_dense')] == [216,36,0],
        'The complete native216/compact36/no-dense endpoint is required.')
    old = prepared['source_binding']['original_down_fit']['actual_optimizer_initializations']
    matched = [row['initial_parameter_digest'] == reference['initial_parameter_digest']
        for row,reference in zip(initializations,old,strict=True)]
    if arm == 'transpose_false' and not short:
        b.require(all(matched), 'Same-structure direction control must retain both actual C0 optimizer initialization digests.')
    control = b.torch.load(b.CONTROL / b.checkpoint.STATE_FILE,map_location='cpu',weights_only=True,mmap=True)
    invariant = {name:value for name,value in model.state_dict().items() if '.mlp.down_proj.' in name
        and ('.protected_input_' in name or '.weight_rotation.' in name)}
    expected = {name:value for name,value in control.items() if '.mlp.down_proj.' in name
        and ('.protected_input_' in name or '.weight_rotation.' in name)}
    b.require(invariant.keys() == expected.keys() and b.c.digest(invariant) == b.c.digest(expected),
        'Down protection indices/INT8 values/scales and rotation signs must equal the original same-parent raw control.')
    specs = b.checkpoint.native._collect_vae_linear_specs(model)
    downs=[row for row in specs if row['name'].endswith('.down_proj')]
    b.require(len(downs) == 36 and all(row['codebook_dim'] == settings['dim']
        and row['stage_codebook_dims'] == [settings['dim']]*2 and row['transpose'] == settings['transpose']
        and row['residual_stages'] == 2 and all(stage['in_dim'] == settings['bits']
            and stage['out_dim'] == settings['dim'] and stage['hidden_dim'] == 128 and stage['num_res_blocks'] == 0
            for stage in row['stage_decoders']) for row in downs),
        'Actual native down payload geometry differs from this approved factor.')
    reference = b.read(b.CONTROL / b.checkpoint.META_FILE)['native_modules']
    rotations = {row['name']:row['weight_rotation'] for row in specs if row['name'].endswith('.down_proj')}
    expected_rotations = {row['name']:row['weight_rotation'] for row in reference if row['name'].endswith('.down_proj')}
    b.require(rotations == expected_rotations,'Keep the actual down rotation specs unchanged.')
    return dict(category=category,source_target_digest=digest,frozen_prefix_digest=before,all_frozen_prefix_fields_exact=True,
        raw_live_source=str(b.PARENT),original_M36_pooled_BSQ=True,actual_stages=2,
        actual_updates_per_stage=2 if short else 10000,schedule_steps=2 if short else 10000,
        formal_schedule_steps=10000,vector_batch_per_layer=settings['batch'],scalar_exposure_per_layer_update=settings['batch']*settings['dim'],
        shuffle_seed=600031,fresh_VAE_initialization_seed=31,actual_optimizer_initializations=initializations,
        original_C0_initialization_digest_match=matched,new_path_update_checks=checks,
        protection_and_rotation_fields_exact_to_control=True,protected_rotation_digest=b.c.digest(invariant),
        rotation_specs_exact_to_control=True,
        inventories=inventory,category_protection=cfg.channel_protect_count,category_rotation_block=cfg.weight_rotation_block_size)
