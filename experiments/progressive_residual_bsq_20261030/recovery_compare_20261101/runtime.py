"""Original factory composition, prefix activation and frozen group8 caching."""
from collections import defaultdict
import gc
import logging

from . import bindings as b
from train_utils.cat_train_runtime import load_model_for_cat_train
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.decoder_execution import enable_vae_linear_by_execution_plan
from train_utils.v6_model_loader import load_v6_model_checkpoint


def projections(model):
    return {name:module for name,module in model.named_modules()
        if name.rsplit('.',1)[-1] in b.CATEGORIES and name.startswith('model.layers.')}


def records(values):
    return {name:b.old_life.tensor_record(value) for name,value in values.items()}


def build(arm):
    argv=list(b.read(b.BASE/'formal/request.json')['recipe']['native_argv'])
    argv[argv.index('--output_dir')+1]=str(b.OUTPUT/'unused_factory_path')
    cat,hf,_training,vae=parse_cat_runtime_args(argv)
    b.require(not cat.rot_llm and cat.after_category_mode == 'none', 'Use the actual original CAT BF16 factory policy.')
    factory = load_model_for_cat_train(cat_args=cat,hf_args=hf,vae_args=vae).eval().requires_grad_(False)
    bank,meta,loaded = load_v6_model_checkpoint(str(b.SOURCE),base_model_path=str(b.HF),map_location='cpu',strict=True)
    bank.eval().requires_grad_(False)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys and meta['checkpoint_id'] == b.SOURCE_ID,
        'Strictly load the real S4 bank.')
    modules = projections(bank)
    b.require(len(modules)==252 and all(isinstance(m,b.VAELinear) and m.residual_stages==4 for m in modules.values()),
        'All252 genuine four-stage modules are required.')
    prefixes = tuple(name+'.' for name in modules)
    outside = {name:value for name,value in factory.state_dict().items() if not name.startswith(prefixes)}
    bank_outside = {name:value for name,value in bank.state_dict().items() if not name.startswith(prefixes)}
    b.require(len(outside)==147 and outside.keys()==bank_outside.keys()
        and all(value.dtype==bank_outside[name].dtype and b.torch.equal(value,bank_outside[name]) for name,value in outside.items()),
        'Original BF16 factory and S4 bank nonprojection fields must match exactly.')
    dense = records({name:value for name,value in factory.state_dict().items() if name.startswith(prefixes)})
    immutable = records({name:value for name,value in bank.state_dict().items() if not b.decoder_name(name)})
    teacher = b.data.original.teacher_identity(factory.state_dict(),list(modules))
    model = bank if arm=='terminal' else factory
    del bank, factory, outside, bank_outside
    model.eval().requires_grad_(False)
    gc.collect()
    return model,modules,dict(nonprojection_exact=True,nonprojection_fields=147,
        factory='N load_model_for_cat_train with the original S4 recipe; rot_llm=False',
        future_dense_records=dense, original_nondecoder_records=immutable, teacher_identity=teacher)


def install(model, bank, stage, arm, active):
    old = {name:(id(p),p.data_ptr(),p._version) for name,p in active.items()}
    if arm=='sequential':
        for name,module in bank.items():
            if name.rsplit('.',1)[-1] not in b.CATEGORIES[:stage]:continue
            previous = model.get_submodule(name)
            if isinstance(previous,b.VAELinear):continue
            b.require(type(previous) is b.torch.nn.Linear, 'Install only over this factory dense projection.')
            device = previous.weight.device
            b.native.set_module_by_name(model,name,module.to(device))
    new = {}
    for name,module in projections(model).items():
        if name.rsplit('.',1)[-1] not in b.CATEGORIES[:stage]:continue
        b.require(isinstance(module,b.VAELinear) and module.residual_stages==4, 'Active prefix needs the realS4 consumer.')
        prefix = name+'._parallel_stage_decoder.'
        if any(key.startswith(prefix) for key in active):continue
        module.packed_vq_decoder_linear = True
        plan = enable_vae_linear_by_execution_plan(module,mode='trainable_decoder')
        b.require(plan.use_packed, 'Use the unchanged native packed decoder training path.')
        module._parallel_stage_decoder.use_checkpoint = True
        for relative,p in module._parallel_stage_decoder.named_parameters():
            p.data = p.data.float()
            p.requires_grad_(True)
            new[prefix+relative] = p
    b.install_precision_runtime(model,b.torch.bfloat16)
    b.require(old == {name:(id(active[name]),active[name].data_ptr(),active[name]._version) for name in old},
        'Installing the next bank must preserve all earlier recovered master identities/versions.')
    active.update(new)
    b.require(set(active)=={name for name,p in model.named_parameters() if p.requires_grad}
        and len({name.split('._parallel_stage_decoder.')[0] for name in active})==36*stage
        and all(p.dtype==b.torch.float32 for p in active.values()), 'Only the same prefix decoder masters may train.')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train(); model.config.use_cache=False
    return new


@b.torch.no_grad()
def warm_frozen(model):
    from litebsq import vae_linear_prewarm as native
    original = native._decode_named_vae_linear_weights_chunk
    chunks = 0
    def release(*args,**kwargs):
        nonlocal chunks
        results,count = original(*args,**kwargs);chunks+=1
        for item in results:
            layer=item.base_layer
            for attr in ('_parallel_stage_grouped_vq_weight','_parallel_stage_grouped_vq_runtime',
                         '_parallel_stage_grouped_vq_packed_runtime'):
                scratch=getattr(layer,attr,None)
                b.require(not isinstance(scratch,b.torch.Tensor) or item.decoded_weight.device!=scratch.device
                    or item.decoded_weight.untyped_storage().data_ptr()!=scratch.untyped_storage().data_ptr(),
                    'Completed group8 output must not alias scratch.')
            layer._clear_parallel_stage_decode_runtime_cache()
        return results,count
    groups=defaultdict(list)
    for name,module in projections(model).items():
        if isinstance(module,b.VAELinear) and not module.trainable_decode:
            groups[str(next(module.parameters()).device)].append((name,module))
    native._decode_named_vae_linear_weights_chunk=release
    try:
        result={device:native.prime_named_vae_linear_cache(targets,dtype=b.torch.bfloat16,clear_existing=True,
            group_size=8,compute_device=device,logger=logging.getLogger('S4_frozen_group8')) for device,targets in groups.items()}
    finally:native._decode_named_vae_linear_weights_chunk=original
    for device,targets in groups.items():
        b.require(result[device]==dict(total=len(targets),warmed=len(targets),skipped=0,failed=0), 'Every frozen module must prewarm.')
        for name,module in targets:
            cache=module._cached_weight
            b.require(cache is not None and cache.dtype==b.torch.bfloat16 and not cache.requires_grad
                and not b.torch.is_inference(cache), 'Frozen BF16 cache must support hidden backward: '+name)
    return dict(devices=result,group_size=8,chunks=chunks,ordinary_no_grad=True)


def composition(model, evidence, active, stage, arm):
    state=model.state_dict(); expected=evidence['original_nondecoder_records'];dense=evidence['future_dense_records']
    found=projections(model)
    b.require(len(found)==252, 'No projection may disappear at category transitions.')
    for name,value in state.items():
        if name in active:continue
        if b.decoder_name(name):continue  # frozen future decoder is checked in the phase frozen complement.
        target=expected.get(name,dense.get(name))
        b.require(target is not None and b.old_life.tensor_record(value)==target, 'Factory/bank frozen source changed: '+name)
    pending=[name for name,module in found.items() if type(module) is b.torch.nn.Linear]
    b.require(len(pending)==(252-36*stage if arm=='sequential' else 0), 'Wrong partial/fullS4 topology.')
    return dict(stage=stage,native=252-len(pending),future_dense=len(pending),
        actual_factory_and_nondecoder_fields_exact=True,active_projections=36*stage)
