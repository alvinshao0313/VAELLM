"""Verify the real B conversion, current teacher, and actual Muon layout on CPU."""
from copy import deepcopy
import argparse
import gc
import io
import json
import os
import time

from . import bindings as b
from . import inputs, lifecycle
from . import optimizer as opt
from experiments.adaptive_native_model_20261029.native_prefix import from_native


def exact(left, right, label):
    b.require(set(left) == set(right), 'Different keys: '+label)
    for name, value in left.items():
        b.require(value.dtype == right[name].dtype and value.shape == right[name].shape
            and b.torch.equal(value.cpu(),right[name].cpu()), 'Changed tensor: '+label+'/'+name)
    return len(left)


def optimizer_checks(active, layout):
    active = {name:p.detach().float().cpu().clone().requires_grad_() for name,p in active.items()}
    matrices, affine = opt.split(active,layout)
    b.torch.manual_seed(137)
    independence = []
    for shape in sorted({tuple(row['matrix_shape']) for row in layout.values()}):
        value = b.torch.randn(shape)
        result = opt.zeropower(value)
        changed = value.clone()
        changed[0] = b.torch.randn_like(changed[0])
        b.require(b.torch.equal(result[1:],opt.zeropower(changed)[1:])
            and b.torch.equal(result,opt.zeropower(value.flip(0)).flip(0)), 'NS crossed independent matrix boundaries.')
        independence.append(list(shape))
    optimizer = opt.MuonAdam(active,layout)
    for step in (1,2):
        optimizer.zero_grad(set_to_none=True)
        for p in active.values(): p.grad = b.torch.randn_like(p)*1e-4
        b.torch.nn.utils.clip_grad_norm_(list(active.values()),1.5,error_if_nonfinite=True)
        if step == 1:
            before = lifecycle.leaves(active)
            optimizer.calibrate()
            b.require(lifecycle.leaves(active) == before and not optimizer.state, 'CPU calibration changed actual tensors.')
        opt.schedule(optimizer,step)
        optimizer.step()
        opt.state_steps(active,optimizer,step)
    stream = io.BytesIO()
    b.torch.save(optimizer.state_dict(),stream)
    stream.seek(0)
    saved = b.torch.load(stream,map_location='cpu',weights_only=True)
    copied = {name:p.detach().clone().requires_grad_() for name,p in active.items()}
    restored = opt.MuonAdam(copied,layout)
    restored.load_state_dict(saved)
    def values(parameters,optim):
        return {name:{key:b.core.tensor_record(value) if isinstance(value,b.torch.Tensor) else value
            for key,value in optim.state[p].items()} for name,p in parameters.items()}
    b.require(values(active,optimizer) == values(copied,restored) and opt.metadata(optimizer) == opt.metadata(restored),
        'CPU momentum/Adam/kappa/LR state roundtrip changed values.')
    return dict(matrix_leaves=len(matrices), auxiliary_Adam_leaves=len(affine),
        actual_matrix_stages=sum(row['stages'] for row in layout.values()),matrix_elements=sum(p.numel() for p in matrices.values()),
        affine_elements=sum(p.numel() for p in affine.values()),total_D_elements=sum(p.numel() for p in active.values()),
        norm_affine_excluded_from_Muon=all('.norm_' not in name for name in matrices),
        CPU_only_random_gradient_test=True, CPU_kappa_not_reused=True, NS_stage_independence_shapes=independence,
        two_updates_finite=True,momentum_Adam_kappa_LR_roundtrip_exact=True)


def convert(reuse=False):
    meta = b.read(b.SOURCE / 'checkpoint_meta.json')
    b.require(meta['checkpoint_id'] == '87571461-7310-46e5-bc2e-904baf1818ed'
        and len(meta['converted_modules']) == 252 and not meta['pending_dense_targets'] and meta['lora_config'] is None,
        'Use the real complete B endpoint, never a short validation weight.')
    model,state,loaded = b.checkpoint.load_parent(b.SOURCE,meta)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys, 'Original B strict load failed.')
    identity = inputs.teacher_identity(state,[item['name'] for item in meta['converted_modules']])
    converted = []
    for spec in meta['converted_modules']:
        name = spec['name']
        if not name.endswith('.gate_proj'): continue
        native = model.get_submodule(name)
        relative = {key[len(name)+1:]:value for key,value in state.items() if key.startswith(name+'.')}
        bundle,lengths,resolved = from_native(spec,relative,dict(parent_checkpoint=str(b.SOURCE),
            parent_checkpoint_id=meta['checkpoint_id'],purpose='C0 topology interface only; no refit/refusion/quality claim'))
        compact = b.CompactRVQLinear(bundle,lengths)
        copied = 0
        for stage in range(2):
            copied += exact(native.get_stage_part_decoder(stage,0).state_dict(),compact.decoders[stage].state_dict(),name)
            b.require(b.torch.equal(native.get_stage_part_vq_storage(stage,0)[:,0,:],getattr(compact,f'codes_{stage}')),
                'Native packed bits changed in conversion.')
        for field in ('protected_input_indices','protected_input_qvalues','protected_input_scales'):
            b.require(b.torch.equal(getattr(native,field),getattr(compact,field)), 'Protection changed.')
        for axis in ('input','output'):
            b.require(b.torch.equal(getattr(native.weight_rotation,axis+'_signs'),getattr(compact.weight_rotation,axis+'_signs')), 'Rotation changed.')
        b.require((lengths == 2).all().item(), 'Fixed2 route must cover every actual vector.')
        b.checkpoint.native.set_module_by_name(model,name,compact)
        converted.append(dict(name=name,decoder_fields_copied_exact=copied,code_banks=2,n_vectors=resolved['n_vectors']))
    groups = b.checkpoint.inventories(model)
    b.require([len(groups[key]) for key in ('native','compact','pending_dense')] == [216,36,0], 'Wrong real converted topology.')
    gate_names = groups['compact']
    non_gate = {key:value for key,value in state.items() if not key.startswith(tuple(name+'.' for name in gate_names))}
    preserved = exact(non_gate,{key:value for key,value in model.state_dict().items() if key in non_gate},'all non-gate fields')
    parent = b.OUTPUT / 'fixed2_parent'
    if reuse:
        previous,saved,loaded,budget = b.checkpoint.load(parent)
        b.require(saved['provenance']['source_checkpoint_id'] == meta['checkpoint_id'] and not loaded.missing_keys
            and not loaded.unexpected_keys, 'Prepared parent belongs to another actual B conversion.')
        exact(model.state_dict(),previous.state_dict(),'reconstructed real B conversion vs preserved parent')
        expected = previous.state_dict()
        del previous
    else:
        saved,budget,expected = b.checkpoint.save(model,parent,provenance=dict(original_parent=b.binding(b.SOURCE/'checkpoint_meta.json'),
            source_checkpoint_id=meta['checkpoint_id'],source_weight=b.binding(b.SOURCE/meta['state_dict_file']),
            scope='Only36 deployed native asymmetric gates converted using from_native; no refusion or refit; preserve216 native/all non-gate fields'))
    return model,saved,expected,identity,dict(converted=converted,non_gate_fields_preserved_exact=preserved,
        source_checkpoint_id=meta['checkpoint_id'],mixed_checkpoint_id=saved['checkpoint_id'],bit_budget=budget,
        purpose='C0 topology interface father; not a fresh BF16 base and not a quality improvement')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-fixed2-parent',action='store_true')
    args = parser.parse_args()
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not b.torch.cuda.is_initialized(), 'CPU preparation only.')
    b.require(not (b.OUTPUT/'prepared.json').exists()
        and (args.reuse_fixed2_parent or not (b.OUTPUT/'fixed2_parent').exists()), 'Never overwrite preparation.')
    b.OUTPUT.mkdir(parents=True,exist_ok=True)
    b.torch.set_num_threads(4)
    started = time.monotonic()
    model,meta,state,identity,conversion = convert(args.reuse_fixed2_parent)
    dataset,recovery,cache,evidence = inputs.load_inputs()
    modules = b.selected_modules(model)
    active = lifecycle.decoder_parameters(modules)
    layout = opt.derive_layout(model,meta)
    check = optimizer_checks(active,layout)
    record = dict(groups=dataset['groups'],train_ids=dataset['train_ids'],
        decoder_parameters={name:b.core.tensor_record(value) for name,value in active.items()},
        trainable_tensors=len(active),trainable_elements=sum(p.numel() for p in active.values()),
        validated_frozen_state=lifecycle.frozen_proof(model,set(active)),matrix_layout=layout,
        model_config=identity['model_config'],teacher_identity=identity,source_checkpoint_id=meta['checkpoint_id'])
    del active,modules,model,state,recovery
    gc.collect()
    fresh,loaded,result,budget = b.checkpoint.load(b.OUTPUT/'fixed2_parent')
    saved_state = b.torch.load(b.OUTPUT/'fixed2_parent'/meta['state_dict_file'],map_location='cpu',weights_only=True,mmap=True)
    compared = exact(fresh.state_dict(),saved_state,'fixed2 parent fresh strict')
    b.require(not result.missing_keys and not result.unexpected_keys
        and lifecycle.frozen_proof(fresh,set(record['decoder_parameters'])) == record['validated_frozen_state'], 'Fresh parent frozen inventory changed.')
    conversion.update(fresh_strict_fields_compared=compared,fresh_strict_exact=True)
    paths = [*b.HERE.glob('*.py'),b.SOURCE/'checkpoint_meta.json',b.OUTPUT/'fixed2_parent/adaptive_model_meta.json',
        Path(b.aligned.__file__),Path(b.base.__file__),Path(b.mmlu.__file__),Path(b.qa.__file__),Path(b.core.__file__)]
    prepared = dict(schema='mixed_compact_Muon_short2_prepared_v1',status='CPU_verified_GPU_pending',
        record=record,contract=b.CONTRACT,steps=b.STEPS,conversion=conversion,optimizer_CPU=check,
        dataset_binding=b.binding(HOT/'dataset.pt'),target_evidence=evidence,target_cache=cache,
        bindings={str(path.resolve()):b.binding(path) for path in paths},
        CUDA_initialized=b.torch.cuda.is_initialized(),elapsed_seconds=time.monotonic()-started)
    b.require(not prepared['CUDA_initialized'], 'CPU preparation initialized CUDA.')
    b.core.write(b.OUTPUT/'prepared.json',prepared)
    print(json.dumps(dict(status=prepared['status'],optimizer=check,conversion_fields=compared,seconds=prepared['elapsed_seconds'])),flush=True)


from pathlib import Path
HOT = inputs.HOT

if __name__ == '__main__':
    main()
