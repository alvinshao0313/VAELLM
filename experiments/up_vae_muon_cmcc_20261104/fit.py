"""Actual continuation up path; change only optimizer or the D/K/B geometry."""
from contextlib import contextmanager, ExitStack

from . import bindings as b
from .optimizer import install
from experiments.adaptive_class_20261030 import continuation as original
from experiments.down_geometry_baseline_cmcc_20261103.validation import updates


@contextmanager
def actual_call(native, settings, *, short):
    original_fit, original_scheduler = native.train_group_vae_payload, native.transformers.get_scheduler
    calls, schedules = [], []
    def fit(**kwargs):
        b.require(kwargs['batch_size'] == 8192 and len(kwargs['group_refs']) == 36
            and all(ref.name.endswith('.up_proj') and not ref.transpose for ref in kwargs['group_refs']), 'Intercept only the actual full up call.')
        kwargs['batch_size'] = settings['batch']
        cfg = kwargs['runtime_cfg']
        b.require(cfg.codebook_dim == settings['dim'] and cfg.codebook_bits == settings['bits'], 'The real runtime D/K must agree with actual B.')
        calls.append(dict(B=kwargs['batch_size'],D=cfg.codebook_dim,K=cfg.codebook_bits,steps=cfg.steps))
        return original_fit(**kwargs)
    def scheduler(*args, **kwargs):
        name = kwargs.get('name',args[0] if args else None)
        if short:
            b.require(name == 'linear' and kwargs['num_warmup_steps'] == 0 and kwargs['num_training_steps'] == 2,
                'Technical horizon hook accepts only the two original VAE linear/no-warmup schedulers.')
            kwargs['num_training_steps'] = 10000
            schedules.append(dict(actual_updates=2,schedule_max=10000))
        return original_scheduler(*args,**kwargs)
    native.train_group_vae_payload = fit
    if short:
        native.transformers.get_scheduler = scheduler
    try:
        yield calls, schedules
        b.require(len(calls) == 1 and (not short or len(schedules) == 2), 'Require exactly one whole-M up call and two technical scheduler hooks.')
    finally:
        native.train_group_vae_payload, native.transformers.get_scheduler = original_fit, original_scheduler


def execute(model, parsed, activation, plan, *, arm, output, short, records):
    prepared = b.read(b.RESULT / 'prepared.json')
    native, settings = b.recovery.native, b.ARMS[arm]
    prefix = lambda:{name:value for name,value in model.state_dict().items() if '.mlp.up_proj.' not in name}
    before = b.c.digest(prefix())
    b.require(before == prepared['frozen_prefix_digest'], 'Start from the same complete frozen after-gate prefix.')
    cat,hf,training,vae,cfgs,_runtime = parsed
    with ExitStack() as stack:
        if arm == 'muon':
            stack.enter_context(install(native,records,short=short))
        checks = stack.enter_context(updates(native,enabled=short and arm == 'd16'))
        calls, schedules = stack.enter_context(actual_call(native,settings,short=short))
        _activation,_plan,report = original.fit_next(model,cat,hf,training,vae,cfgs,activation,plan,
            previous='gate_proj',category='up_proj',output=output,short=short)
    b.require(report['source_target_digest'] == prepared['source_binding']['original_up_fit']['source_target_digest']
        and b.c.digest(prefix()) == before, 'Preserve actual live up targets and every non-up persistent field.')
    expected = prepared['source_binding']['original_up_fit']['actual_optimizer_initializations']
    matches = ([row['canonical_original_flat_order_initial_digest'] == old['initial_parameter_digest']
        for row,old in zip(records,expected,strict=True)] if arm == 'muon' else [])
    if arm == 'muon':
        b.require(len(records) == 2 and all(row['calibration']['relative_error'] <= .01 for row in records), 'Both actual up Muon stage kappa matches must pass.')
        if not short:
            b.require(all(matches), 'Both formal Muon initializations must equal the actual original C0 flat order.')
    if short and arm == 'd16':
        b.require(len(checks) == 2 and all(len(row['updates']) == 2 for row in checks), 'Both genuine D16 stages must perform two finite nonzero updates.')
    control = b.torch.load(b.CONTROL / b.checkpoint.STATE_FILE,map_location='cpu',weights_only=True,mmap=True)
    select = lambda values:{name:value for name,value in values.items() if '.mlp.up_proj.' in name
        and ('.protected_input_' in name or '.weight_rotation.' in name)}
    b.require(b.c.digest(select(model.state_dict())) == b.c.digest(select(control)), 'Keep original input64 INT8 values/scales/indices and rotation signs exact.')
    specs = b.checkpoint.native._collect_vae_linear_specs(model)
    up = [row for row in specs if row['name'].endswith('.up_proj')]
    b.require(len(up) == 36 and all(row['codebook_dim'] == settings['dim'] and row['stage_codebook_dims'] == [settings['dim']]*2
        and row['residual_stages'] == 2 and not row['transpose']
        and all(stage['in_dim'] == settings['bits'] and stage['out_dim'] == settings['dim']
            and stage['hidden_dim'] == 128 and stage['num_res_blocks'] == 0 and stage['decoder_type'] == 'symmetric'
            for stage in row['stage_decoders']) for row in up), 'Actual up consumer geometry differs from the selected single factor.')
    reference = b.read(b.CONTROL / b.checkpoint.META_FILE)['native_modules']
    rotations = lambda rows:{row['name']:row['weight_rotation'] for row in rows if row['name'].endswith('.up_proj')}
    b.require(rotations(specs) == rotations(reference), 'Actual up rotation specs must match the control.')
    report.update(vector_batch_per_layer=settings['batch'], scalar_exposure_per_layer_update=settings['batch']*settings['dim'],
        schedule_steps=10000, actual_call=calls, technical_original_scheduler=schedules,
        optimizer_stages=records, original_C0_initialization_digest_match=matches,
        all_frozen_prefix_fields_exact=True, protection_and_rotation_fields_exact_to_control=True,
        rotation_specs_exact_to_control=True, new_shape_update_checks=checks,
        actual_optimizer='BF16 VAE Muon + native auxiliary AdamW' if arm == 'muon' else 'Original BF16 AdamW')
    return report
