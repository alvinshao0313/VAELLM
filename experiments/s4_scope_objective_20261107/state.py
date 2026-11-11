"""Exact external-master training states and native-only independent CPU exports."""
from pathlib import Path
from . import bindings as b,core
from .protection import effective_scale,integer_coordinate
from experiments.s4_recovery_screen_20261105.state import cpu,same
from experiments.mlp_learnability_formal_cmcc_20261101.snapshot import rng,restore_rng,rng_exact
from train_utils.v6_model_loader import load_v6_model_checkpoint
from train_utils.checkpoint_bpw import audit_state_dict


def load(directory):
    directory=Path(directory).resolve(strict=True)
    b.require(directory==b.SOURCE or directory.is_relative_to(b.OUTPUT.resolve()),'Unknown native model source.')
    model,meta,loaded=load_v6_model_checkpoint(str(directory),base_model_path=str(b.HF),map_location='cpu',strict=True)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys,'Native strict load failed.')
    if directory==b.SOURCE:b.require(meta['checkpoint_id']==b.SOURCE_ID,'Common parent changed.')
    else:
        p=meta['extra_meta']['distill_stage']
        b.require(p['algorithm']==b.ALGORITHM and p['parent']==b.source() and p['contract']==b.CONTRACT,
            'Endpoint provenance changed.')
    b.require(core.runtime.records(model.state_dict())==b.read(directory.parent/'expected_state.json'),
        'Complete persistent state differs.')
    budget=audit_state_dict(meta,model.state_dict(),max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'] and budget['compressed_target_count']==252
        and abs(budget['user_linear_bpw']-b.BPW)<1e-12
        and budget['packed_linear_bpw']==budget['user_linear_bpw'],'Native S4 ledger changed.')
    return model.eval().requires_grad_(False),meta,budget


def identity(request):
    return {k:request[k] for k in ('arm','recipe','validation_only','groups','parent','data_digest','source_sha256','contract')}


def check_payload(saved,request):
    b.require(saved['schema']=='S4_scope_objective_named_state_v1' and saved['identity']==identity(request),
        'Saved training identity changed.')
    names=saved['parameter_names'];step=saved['completed_updates'];groups=request['groups']
    b.require(names==list(saved['masters'])==request['parameter_names']
        and saved['next_cursor']==step and 0<step<=len(groups)
        and saved['next_group_index']==(groups[step] if step<len(groups) else None),'Saved names/cursor changed.')
    b.require(all(p.dtype==b.torch.float32 and b.torch.isfinite(p).all() for p in saved['masters'].values()),
        'Saved masters must be finite FP32.')
    optim=saved['optimizer'];order=[i for g in optim['param_groups'] for i in g['params']]
    b.require(len(order)==len(names)==len(optim['state']) and len(set(order))==len(names)
        and all(int(optim['state'][i]['step'])==step
            and optim['state'][i]['exp_avg'].dtype==optim['state'][i]['exp_avg_sq'].dtype==b.torch.float32
            and b.torch.isfinite(optim['state'][i]['exp_avg']).all()
            and b.torch.isfinite(optim['state'][i]['exp_avg_sq']).all()
            and optim['state'][i]['exp_avg'].shape==saved['masters'][name].shape
            for name,i in zip(names,order)),'Complete named FP32 Adam state required.')
    for group in optim['param_groups']:
        b.require(group['lr']==group['initial_recipe_lr']*(1-(step-1)/500), 'Saved LR schedule changed.')
    return saved


def save(model,active,optimizer,controller,request,completed,path,frozen):
    b.require(all(p.grad is None for p in active.values()),'Save at a completed update boundary.')
    groups=request['groups']
    payload=dict(schema='S4_scope_objective_named_state_v1',identity=identity(request),
        completed_updates=completed,next_cursor=completed,next_group_index=groups[completed] if completed<len(groups) else None,
        parameter_names=list(active),masters=cpu(active),optimizer=cpu(optimizer.state_dict()),rng=cpu(rng()),
        frozen_complement=frozen,mutable_native_names=sorted(core.mutable_names(active,request['arm'])),
        first_hard_change_step=controller.first_hard_step if controller else None)
    check_payload(payload,request);path=Path(path);b.require(not path.exists(),'Never overwrite actual state.')
    b.torch.save(payload,path)
    loaded=b.torch.load(path,map_location='cpu',weights_only=False)
    b.require(same(payload['masters'],loaded['masters']) and same(payload['optimizer'],loaded['optimizer'])
        and rng_exact(loaded['rng']),'Saved training state differs.')
    return b.binding(path)


def restore(path,model,active,optimizer,controller,request):
    saved=check_payload(b.torch.load(path,map_location='cpu',weights_only=False),request)
    b.require(list(active)==saved['parameter_names'] and core.mutable_names(active,request['arm'])==set(saved['mutable_native_names'])
        and b.old.old_life.frozen_proof(model,saved['mutable_native_names'])==saved['frozen_complement'],
        'Named scope/frozen complement changed at restore.')
    with b.torch.no_grad():
        for name,p in active.items():p.copy_(saved['masters'][name].to(p.device))
    optimizer.load_state_dict(saved['optimizer'])
    b.require(same(cpu(active),saved['masters']) and same(cpu(optimizer.state_dict()),saved['optimizer']),
        'Restored masters or Adam differs.')
    restore_rng(saved['rng']);b.require(rng_exact(saved['rng']),'Actual RNG restoration failed.')
    if controller:
        controller.first_hard_step=saved['first_hard_change_step'];controller.reset_previous()
    return saved


@b.torch.no_grad()
def export(request,saved,path):
    check_payload(saved,request);path=Path(path);path.mkdir(parents=True,exist_ok=False)
    model,meta,_=load(b.SOURCE);parameters=dict(model.named_parameters());modules=core.runtime.projections(model)
    frozen=b.old.old_life.frozen_proof(model,saved['mutable_native_names'])
    b.require(frozen==saved['frozen_complement'],'Export source frozen state changed.')
    changed=dict(early=0,middle=0,late=0)
    for name,value in saved['masters'].items():
        if '._parallel_stage_decoder.' in name:
            hard=value.bfloat16();layer=int(name.split('.')[2]);key='early' if layer<12 else 'middle' if layer<24 else 'late'
            changed[key]+=int(b.torch.count_nonzero(parameters[name]!=hard));parameters[name].copy_(hard)
        else:
            module=modules[name.rsplit('.',1)[0]]
            if request['recipe']['protection']=='scale':
                module.protected_input_scales.copy_(effective_scale(module.protected_input_scales,value))
            else:module.protected_input_qvalues.copy_(integer_coordinate(value).to(b.torch.int8).view(b.torch.uint8))
    for module in modules.values():module.disable_trainable_decode()
    model.gradient_checkpointing_disable();model.eval().requires_grad_(False)
    b.old.native.refresh_vae_linear_runtime_after_state_load(model)
    b.old.install_precision_runtime(model,b.torch.bfloat16)
    b.require(b.old.old_life.frozen_proof(model,saved['mutable_native_names'])==frozen,
        'Native export changed frozen fields.')
    provenance=dict(algorithm=b.ALGORITHM,parent=request['parent'],contract=b.CONTRACT,arm=request['arm'],
        recipe=request['recipe'],actual_updates=saved['completed_updates'],next_group_index=saved['next_group_index'],
        validation_only=request['validation_only'],data_digest=request['data_digest'],source_sha256=request['source_sha256'],
        mutable_native_names=saved['mutable_native_names'],native_only=True)
    extra=dict(meta['extra_meta']);extra['distill_stage']=provenance
    extra['distill_stage_history']=[*extra['distill_stage_history'],provenance]
    result=b.old.native.save_v6_full_checkpoint(model,str(path/'model'),checkpoint_kind='category_boundary',
        compressed_targets=list(modules),pending_dense_targets=[],skip_targets=[],train_mode='decoder_only',
        after_category_mode='none',completed_categories=list(b.CATEGORIES),compression_categories=list(b.CATEGORIES),
        target_modules=list(b.CATEGORIES),base_model_path=str(b.HF),
        cat_runtime_state=b.torch.load(b.SOURCE/'cat_runtime_state.pt',map_location='cpu',weights_only=True),extra_meta=extra)
    b.write(path/'expected_state.json',core.runtime.records(model.state_dict()))
    budget=audit_state_dict(result['meta_payload'],model.state_dict(),max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'] and budget['user_linear_bpw']==budget['packed_linear_bpw']
        and abs(budget['user_linear_bpw']-b.BPW)<1e-12,'Exported ledger changed.')
    b.require(all(changed.values()),'Real decoder changes must survive BF16 in early/middle/late layers.')
    report=dict(checkpoint=b.event(path/'model'),bit_budget=budget,BF16_changed_decoder=changed,
        actual_updates=saved['completed_updates'],named_state_verified=True,complete_frozen_complement_exact=True,
        native_fields=len(model.state_dict()),protection_export='Original UINT8 q and BF16 scale only; no recalibration or external adapter')
    b.write(path/'export.json',report);return report
