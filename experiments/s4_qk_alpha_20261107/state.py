"""Exact named single-GPU training state and independent native BF16 export."""
from copy import deepcopy
from pathlib import Path
import gc
from . import bindings as b,core
from experiments.mlp_learnability_formal_cmcc_20261101.snapshot import rng,restore_rng,rng_exact,live_stamp
from train_utils.v6_model_loader import load_v6_model_checkpoint
from train_utils.checkpoint_bpw import audit_state_dict


def cpu(value):
    if b.torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(cpu(v) for v in value)
    return deepcopy(value)


def same(left,right):
    if b.torch.is_tensor(left):return b.torch.is_tensor(right) and left.dtype==right.dtype and b.torch.equal(left.cpu(),right.cpu())
    if isinstance(left,dict):return isinstance(right,dict) and list(left)==list(right) and all(same(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):return type(left) is type(right) and len(left)==len(right) and all(same(a,c) for a,c in zip(left,right))
    return type(left) is type(right) and left==right


def load(directory):
    directory=Path(directory).resolve(strict=True)
    b.require(directory==b.SOURCE or directory.is_relative_to(b.OUTPUT.resolve()),'Unknown model source.')
    model,meta,loaded=load_v6_model_checkpoint(str(directory),base_model_path=str(b.HF),map_location='cpu',strict=True)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys,'Native strict load failed.')
    if directory==b.SOURCE:b.require(meta['checkpoint_id']==b.SOURCE_ID,'Common down/pre UUID changed.')
    else:
        provenance=meta['extra_meta']['distill_stage']
        b.require(provenance['algorithm']==b.ALGORITHM and provenance['parent']==b.source()
            and provenance['contract']==b.contract(provenance['arm'])
            and provenance['teacher_alpha']==b.ARMS[provenance['arm']]['alpha'],'Scope endpoint provenance changed.')
    expected=b.read(directory.parent/'expected_state.json')
    b.require(core.runtime.records(model.state_dict())==expected,'Full persistent BF16 state differs.')
    budget=audit_state_dict(meta,model.state_dict(),max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'] and budget['compressed_target_count']==252
        and abs(budget['user_linear_bpw']-4.094031292459239)<1e-12
        and budget['packed_linear_bpw']==budget['user_linear_bpw'],'S4 storage ledger changed.')
    return model.eval().requires_grad_(False),meta,budget


def identity(request):
    return {k:request[k] for k in ('arm','teacher_alpha','validation_only','groups','parent','data_digest','source_sha256','learning_contract')}


def save(model,active,optimizer,request,completed,path):
    path=Path(path);b.require(not path.exists(),'Never overwrite an actual training state.')
    b.require(all(p.grad is None for p in active.values()),'Save only at a completed update boundary.')
    groups=request['groups'];next_index=groups[completed] if completed<len(groups) else None
    payload=dict(schema='S4_scope_named_training_state_v1',identity=identity(request),completed_updates=completed,
        next_cursor=completed,next_group_index=next_index,parameter_names=list(active),masters=cpu(active),
        saved_execution_policy=dict(contract=request['contract'],first_batch_stop=request['stop_step'],stop_step=completed),
        optimizer=cpu(optimizer.state_dict()),rng=cpu(rng()),frozen_complement=b.old.old_life.frozen_proof(model,set(active)))
    b.torch.save(payload,path)
    loaded=b.torch.load(path,map_location='cpu',weights_only=False)
    b.require(same(payload['masters'],loaded['masters']) and same(payload['optimizer'],loaded['optimizer']) and rng_exact(loaded['rng']),
        'Saved actual FP32 master/Adam payload differs.')
    return b.binding(path)


def restore(path,model,active,optimizer,request):
    saved=b.torch.load(path,map_location='cpu',weights_only=False)
    b.require(saved['schema']=='S4_scope_named_training_state_v1' and saved['identity']==identity(request)
        and saved['parameter_names']==list(active)
        and saved['frozen_complement']==b.old.old_life.frozen_proof(model,set(active)),
        'Actual named state source/data/scope/code/frozen complement changed.')
    step=saved['completed_updates'];groups=request['groups']
    expected=groups[step] if step<len(groups) else None
    b.require(saved['next_cursor']==step and 0<=step<=len(groups) and saved['next_group_index']==expected,
        'Saved cursor/next group violates the fixed endpoint.')
    with b.torch.no_grad():
        for name,p in active.items():p.copy_(saved['masters'][name].to(p.device))
    optimizer.load_state_dict(saved['optimizer'])
    b.require(same(cpu(active),saved['masters']) and same(cpu(optimizer.state_dict()),saved['optimizer']),
        'Restored FP32 masters or complete Adam state differs.')
    for p in active.values():
        item=optimizer.state[p]
        b.require(p.dtype==item['exp_avg'].dtype==item['exp_avg_sq'].dtype==b.torch.float32
            and int(item['step'])==step,'Restored FP32/Adam step contract failed.')
    restore_rng(saved['rng']);b.require(rng_exact(saved['rng']),'Restored actual RNG differs.')
    return saved


@b.torch.no_grad()
def export(model,active,optimizer,request,saved,path):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    stamp,random_state=live_stamp(active,optimizer),rng()
    deployment,meta,_=load(b.SOURCE)
    parameters=dict(deployment.named_parameters())
    for name,p in active.items():parameters[name].copy_(p.detach().to(device='cpu',dtype=b.torch.bfloat16))
    for module in core.runtime.projections(deployment).values():module.disable_trainable_decode()
    deployment.gradient_checkpointing_disable();deployment.eval().requires_grad_(False)
    b.old.native.refresh_vae_linear_runtime_after_state_load(deployment)
    b.old.install_precision_runtime(deployment,b.torch.bfloat16)
    b.require(b.old.old_life.frozen_proof(deployment,set(active))==saved['frozen_complement'],'Export changed frozen persistent fields.')
    provenance=dict(algorithm=b.ALGORITHM,parent=request['parent'],contract=request['contract'],arm=request['arm'],teacher_alpha=request['teacher_alpha'],
        active_projection_count=b.ARMS[request['arm']]['projections'],active_decoder_leaves=len(active),compressed_target_count=252,
        saved_execution_policy=saved['saved_execution_policy'],learning_contract=request['learning_contract'],
        actual_updates=saved['completed_updates'],next_group_index=saved['next_group_index'],
        validation_only=request['validation_only'],data_digest=request['data_digest'],source_sha256=request['source_sha256'])
    extra=dict(meta['extra_meta']);extra['distill_stage']=provenance
    extra['distill_stage_history']=[*extra['distill_stage_history'],provenance]
    result=b.old.native.save_v6_full_checkpoint(deployment,str(path/'model'),checkpoint_kind='category_boundary',
        compressed_targets=list(core.runtime.projections(deployment)),pending_dense_targets=[],skip_targets=[],train_mode='decoder_only',
        after_category_mode='none',completed_categories=list(b.CATEGORIES),compression_categories=list(b.CATEGORIES),
        target_modules=list(b.CATEGORIES),base_model_path=str(b.HF),
        cat_runtime_state=b.torch.load(b.SOURCE/'cat_runtime_state.pt',map_location='cpu',weights_only=True),extra_meta=extra)
    b.write(path/'expected_state.json',core.runtime.records(deployment.state_dict()))
    budget=audit_state_dict(result['meta_payload'],deployment.state_dict(),max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'] and abs(budget['user_linear_bpw']-4.094031292459239)<1e-12,'Export storage changed.')
    del deployment;gc.collect();restore_rng(random_state)
    b.require(rng_exact(random_state) and live_stamp(active,optimizer)==stamp,'Export disturbed live training state.')
    record=dict(checkpoint=b.event(path/'model'),bit_budget=budget,from_actual_named_training_state=True)
    b.write(path/'snapshot.json',record);return record
