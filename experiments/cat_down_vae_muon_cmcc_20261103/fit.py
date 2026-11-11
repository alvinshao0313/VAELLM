"""Keep the unchanged live-down fitting path; replace only its two factories."""
from . import bindings as b
from .optimizer import install
from experiments.adaptive_class_20261030 import continuation as original


def execute(model,parsed,activation,plan,*,output,short,records):
    prepared=b.read(b.RESULT / 'prepared.json')
    prefix=lambda:{name:value for name,value in model.state_dict().items() if '.mlp.down_proj.' not in name}
    before=b.c.digest(prefix())
    b.require(before==prepared['reused_source_evidence']['frozen_prefix_digest'],'Load the exact same frozen prefix as the control.')
    cat,hf,training,vae,cfgs,_runtime=parsed
    with install(b.recovery.native,records,short=short):
        _activation,_plan,report=original.fit_next(model,cat,hf,training,vae,cfgs,activation,plan,
            previous='up_proj',category='down_proj',output=output,short=short)
    b.require(len(records)==2 and report['source_target_digest']==prepared['reused_source_evidence']['live_down_source_digest']
        and b.c.digest(prefix())==before,'Keep the actual same live source and complete non-down prefix.')
    expected=prepared['source_binding']['original_down_fit']['actual_optimizer_initializations']
    matches=[actual['canonical_original_flat_order_initial_digest']==old['initial_parameter_digest']
        for actual,old in zip(records,expected,strict=True)]
    b.require(all(matches),'Both original flat-order initialization digests must match the actual C0 control.')
    b.require(all(row['calibration']['relative_error']<=.01 and len(row['updates'])==(2 if short else 3) for row in records),
        'Require both actual stage amplitude calibrations and declared real update observations.')
    control=b.torch.load(b.CONTROL / b.checkpoint.STATE_FILE,map_location='cpu',weights_only=True,mmap=True)
    select=lambda values:{name:value for name,value in values.items() if '.mlp.down_proj.' in name
        and ('.protected_input_' in name or '.weight_rotation.' in name)}
    b.require(b.c.digest(select(model.state_dict()))==b.c.digest(select(control)),
        'Original input256 INT8 protection and rotation sign fields must remain exact.')
    specs=b.checkpoint.native._collect_vae_linear_specs(model)
    expected_specs=b.read(b.CONTROL / b.checkpoint.META_FILE)['native_modules']
    rotations=lambda rows:{row['name']:row['weight_rotation'] for row in rows if row['name'].endswith('.down_proj')}
    b.require(rotations(specs)==rotations(expected_specs),'The original actual rotation specs must remain exact.')
    report.update(actual_optimizer='BF16 VAE per-linear Muon plus original BF16 auxiliary AdamW',
        canonical_original_initialization_matches=matches,optimizer_stages=records,
        canonical_initialization_boundary='Original one-group flat order reconstructed before grouping; native observer below records the new two-group order.',
        all_frozen_prefix_fields_exact=True,protection_and_rotation_fields_exact_to_control=True,
        rotation_specs_exact_to_control=True,persistent_FP32_parameter_master=False)
    return report
