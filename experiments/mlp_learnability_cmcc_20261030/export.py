"""Build a separate BF16 deployment object; never rewrite live FP32 masters."""
import gc

from . import bindings as b
from . import lifecycle as life
from .free_dense import FreeDenseLinear, protection_binding

torch = b.torch


def deployment_contract(model,arm,source_format):
    b.require(arm in ('decoder','dense') and source_format in ('native','mixed'), 'Unknown actual arm/source format.')
    b.require(all(p.dtype == torch.bfloat16 and not p.requires_grad for p in model.parameters()),
              'Every deployment Parameter must be the actual frozen BF16 cast.')
    rope = [(name,value) for name,value in model.named_buffers() if 'inv_freq' in name]
    b.require(rope and all(value.dtype == torch.float32 and torch.isfinite(value).all().item() for name,value in rope),
              'Keep the original finite FP32 RoPE buffers; never cast the complete model.')
    state = model.state_dict()
    b.require(all(torch.is_tensor(value) for value in state.values()), 'Canonical deployment must contain no custom extra_state.')
    groups = b.checkpoint.inventories(model)
    expected = [144,0,108] if arm == 'dense' else ([252,0,0] if source_format == 'native' else [216,36,0])
    b.require([len(groups[key]) for key in ('native','compact','pending_dense')] == expected,
              'The exported actual native/compact/dense projection topology differs.')
    if arm == 'dense':
        b.require(set(groups['pending_dense']) == set(life.mlp_modules(model)), 'Only all108 MLP may become ordinary dense projections.')
    return groups


def protection_proof(model,records,*,deployed=False):
    modules = life.mlp_modules(model)
    b.require(set(modules) == set(records), 'Protection audit must cover the actual108 MLP projections.')
    for name,module in modules.items():
        if type(module) is torch.nn.Linear:
            weight = module.weight.detach()
        else:
            b.require(protection_binding(module) == records[name]['protection'], 'Compressed protection fields changed: '+name)
            if not deployed:
                continue
            from .free_dense import deployed_weight
            weight = deployed_weight(module,name)
        ids = torch.tensor(records[name]['protected_indices'],dtype=torch.int64,device=weight.device)
        b.require(list(weight.shape) == records[name]['shape'] and weight.dtype == torch.bfloat16
            and b.digest({'columns':weight.index_select(1,ids)}) == records[name]['protected_BF16_sha256'],
            'Deployment changed actual protected BF16 columns: '+name)
    return dict(projections=len(modules),persistent_compressed_fields_exact=True,
                actual_protected_BF16_exact=bool(deployed or all(type(module) is torch.nn.Linear for module in modules.values())))


@torch.no_grad()
def build(model,arm,source_binding,active,source_non_mlp,protection_records):
    b.require(arm in ('decoder','dense'), 'Unknown all-MLP deployment arm.')
    b.require(all(p.dtype == torch.float32 and p.requires_grad for p in active.values()), 'Export only from actual FP32 optimizer masters.')
    expected_active = (life.decoder_parameters(life.mlp_modules(model)) if arm == 'decoder'
                       else {name+'.active_weight':module.active_weight for name,module in life.mlp_modules(model).items()})
    b.require(set(active) == set(expected_active) and all(active[name] is p for name,p in expected_active.items()),
              'The export must copy every actual all108 MLP active leaf, without unrelated parameters.')
    master_stamp = {name:(id(p),p.data_ptr(),p.dtype,p._version) for name,p in active.items()}
    deployment,_source_meta,_source_budget = b.load_source(source_binding)
    b.require(life.non_mlp_proof(deployment) == source_non_mlp, 'Independent CPU source changed outside the MLP family.')
    source_state = life.state_records(deployment)['tensors'] if arm == 'decoder' else None
    if arm == 'dense':
        for name,module in life.mlp_modules(model).items():
            b.require(isinstance(module,FreeDenseLinear), 'Never pass a custom free wrapper to the canonical serializer.')
            module.validate_structure()
            original = deployment.get_submodule(name)
            b.require(module.source['source_checkpoint_id'] == source_binding['checkpoint_id']
                and module.source['protection'] == protection_binding(original), 'Dense export is bound to another source/protection: '+name)
            weight = module.decode_weight().to(device='cpu',dtype=torch.bfloat16).contiguous()
            linear = torch.nn.Linear(module.in_features,module.out_features,bias=module.bias is not None,
                                     dtype=torch.bfloat16,device='meta')
            linear.weight = torch.nn.Parameter(weight,requires_grad=False)
            if module.bias is not None:
                linear.bias = torch.nn.Parameter(module.bias.detach().cpu().clone(),requires_grad=False)
            b.checkpoint.native.set_module_by_name(deployment,name,linear)
    else:
        parameters = dict(deployment.named_parameters())
        b.require(set(active).issubset(parameters), 'The actual source decoder leaf topology changed before BF16 export.')
        for name,p in active.items():
            target = parameters[name]
            b.require(target.dtype == torch.bfloat16 and target.shape == p.shape, 'Native/compact source decoder BF16 storage differs: '+name)
            target.copy_(p.detach().to(device='cpu',dtype=torch.bfloat16))
            b.require(torch.equal(target,p.detach().to(device='cpu',dtype=torch.bfloat16)), 'Decoder export differs from its actual BF16 cast: '+name)
        actual = life.state_records(deployment)['tensors']
        b.require({name:value for name,value in actual.items() if name not in active}
                  == {name:value for name,value in source_state.items() if name not in active},
                  'Independent D export changed a frozen source field.')
    b.require(life.non_mlp_proof(deployment) == source_non_mlp
        and master_stamp == {name:(id(p),p.data_ptr(),p.dtype,p._version) for name,p in active.items()},
        'CPU export changed live FP32 Parameter/storage/version or frozen non-MLP state.')
    deployment.eval().requires_grad_(False)
    for module in deployment.modules():
        if isinstance(module,b.VAELinear):
            module.disable_trainable_decode()
        elif isinstance(module,b.CompactRVQLinear):
            module.refresh_runtime()
    b.checkpoint.native.refresh_vae_linear_runtime_after_state_load(deployment)
    b.install_precision_runtime(deployment,torch.bfloat16)
    groups = deployment_contract(deployment,arm,source_binding['source_format'])
    protection = protection_proof(deployment,protection_records)
    b.require(all(value.device.type == 'cpu' for value in deployment.state_dict().values()), 'The independent deployment clone must start on CPU.')
    gc.collect()
    return deployment,dict(groups=groups,live_masters_unchanged=True,independent_CPU_source_strict=True,
        all_deployment_parameters_BF16=True,FP32_RoPE_preserved=True,diagnostic_overbudget=arm=='dense',
        source_non_mlp_exact=True,protection=protection)


@torch.no_grad()
def save(deployment,directory,identity,source_non_mlp,protection_records):
    deployment_contract(deployment,identity['arm'],identity['source_binding']['source_format'])
    b.require(life.non_mlp_proof(deployment) == source_non_mlp, 'Saving deployment changed frozen source attention/backbone.')
    protection_proof(deployment,protection_records,deployed=True)
    provenance = dict(schema='all_MLP_learnability_short2_export_v1',arm=identity['arm'],
        prototype_only=identity['prototype_only'],validation_only=True,diagnostic_overbudget=identity['arm']=='dense',
        actual_updates=2,schedule_steps=500,group_indices=[2,10],source_binding=identity['source_binding'],
        execution=identity,source_non_mlp=source_non_mlp,source_protection=protection_records,
        scope='Two-update mechanism diagnostic only; not formal quality, a new C0 parent or exact-resume state.')
    meta,budget,state = b.checkpoint.save(deployment,directory,provenance=provenance)
    b.require(all(torch.is_tensor(value) for value in state.values()), 'Canonical deployment must contain no custom extra_state.')
    expected = {name:life.tensor_record(value) for name,value in state.items()}
    if identity['arm'] == 'dense':
        b.require(meta['checkpoint_kind'] == 'mixed_dense_boundary' and not budget['budget_passed']
            and not budget['final_budget_applicable'] and budget['pending_dense_target_count']==108,
            'The dense diagnostic must be charged honestly and never claim a low-bit final pass.')
    else:
        b.require(meta['checkpoint_kind'] == 'mixed_final_model' and budget['budget_passed'], 'D endpoint lost its real compressed budget contract.')
    return meta,budget,expected
