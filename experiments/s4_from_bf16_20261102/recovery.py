"""One fresh Adam on the complete installed S4 prefix; no future dense updates."""
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from . import bindings as b,budget
from experiments.progressive_residual_bsq_20261030.recovery_compare_20261101 import runtime,training
from experiments.mlp_learnability_formal_cmcc_20261101 import readout
from experiments.mlp_learnability_formal_cmcc_20261101.snapshot import rng,restore_rng,rng_exact,live_stamp
from train_utils import checkpoint_v6 as native_save
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.distill_precision import prepare_model_export


def panel(model,dataset,targets,baseline=None):
    values={}
    for key,indices in [('old_prefix',[2,10]),('new_prefix',[34,42])]:
        values[key]=readout.evaluate(model,dataset,targets,indices,None if baseline is None else baseline[key])
    values['scope']='Train directional panel:128M,32QA presentations/16unique; not heldout or full-pool fitting.'
    return values


def cpu(value):
    if b.torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {key:cpu(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(cpu(item) for item in value)
    return deepcopy(value)


def training_state(model,active,optimizer,step,root,request,stage,next_group=None):
    payload=dict(schema='fresh_S4_prefix_training_state_v1',category=b.CATEGORIES[stage-1],
        completed_updates=step,next_group_index=step%128 if next_group is None else next_group,
        contract=b.CONTRACT,data_digest=request['data_digest'],validation_only=request['validation_only'],
        source_sha256=request['source_sha256'],HF_identity=request['HF_identity'],parameter_names=list(active),
        masters={name:cpu(p) for name,p in active.items()},optimizer=cpu(optimizer.state_dict()),rng=cpu(rng()),
        budget_state=request.get('_budget_state'),
        round_source=request['_round_source_binding'],execution_root=request['output'],
        frozen_complement=b.old.old_life.frozen_proof(model,set(active)),
        boundary='Actual current stage FP32 masters/Adam/RNG. Next stage intentionally creates fresh Adam after BF16 commit.')
    b.torch.save(payload,root/'training_state.pt')
    return b.binding(root/'training_state.pt')


def restore_training_state(path,model,active,optimizer,request,stage):
    saved=b.torch.load(path,map_location='cpu',weights_only=False)
    b.require(saved['schema']=='fresh_S4_prefix_training_state_v1'
        and saved['category']==b.CATEGORIES[stage-1] and saved['contract']==b.CONTRACT
        and saved['data_digest']==request['data_digest'] and saved['source_sha256']==request['source_sha256']
        and saved['HF_identity']==request['HF_identity'] and saved['parameter_names']==list(active)
        and saved['validation_only']==request['validation_only'] and saved['execution_root']==request['output']
        and saved['round_source']==request['_round_source_binding']
        and saved['frozen_complement']==b.old.old_life.frozen_proof(model,set(active)),
        'Actual named continuation source/data/contract changed.')
    with b.torch.no_grad():
        for name,p in active.items():p.copy_(saved['masters'][name].to(p.device))
    optimizer.load_state_dict(saved['optimizer'])
    restored=optimizer.state_dict()
    for name,p in active.items():
        b.require(b.torch.equal(p.detach().cpu(),saved['masters'][name])
            and int(optimizer.state[p]['step'])==saved['completed_updates'],'Restored FP32 master/Adam clock differs: '+name)
    for key,values in saved['optimizer']['state'].items():
        for field,value in values.items():
            if b.torch.is_tensor(value):b.require(b.torch.equal(restored['state'][key][field].cpu(),value),
                'Restored Adam tensor differs: '+field)
    restore_rng(saved['rng']);b.require(rng_exact(saved['rng']),'Restored actual training RNG differs.')
    return saved


def snapshot(model,active,optimizer,stage,step,root,request,save_kwargs):
    root.mkdir(parents=True,exist_ok=False)
    stamp,random_state=live_stamp(active,optimizer),rng()
    from train_utils.v6_model_loader import load_v6_model_checkpoint
    deployment,meta,loaded=load_v6_model_checkpoint(str(request['_round_source']),
        base_model_path=str(b.HF),map_location='cpu',strict=True)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys,'Strict current-round snapshot factory failed.')
    parameters=dict(deployment.named_parameters())
    with b.torch.no_grad():
        for name,p in active.items():parameters[name].copy_(p.detach().to(device='cpu',dtype=b.torch.bfloat16))
    for module in runtime.projections(deployment).values():
        if isinstance(module,b.old.VAELinear):module.disable_trainable_decode()
    deployment.gradient_checkpointing_disable();deployment.eval().requires_grad_(False)
    prepare_model_export(deployment,save_kwargs['training_args'])
    native_save.refresh_vae_linear_runtime_after_state_load(deployment)
    # Use the original inventory/save API on the independent BF16 consumer.
    from train_utils.cat_checkpoint_v6 import save_cat_v6_full_checkpoint
    kwargs=dict(save_kwargs)
    kwargs.update(checkpoint_kind='category_boundary',category=b.CATEGORIES[stage-1],
        distill_stage_meta=dict(algorithm='fresh_BF16_S4_prefix_recovery',origin='original_HF_BF16',
            phase='midpoint_observation',did_train=True,actual_updates=step,active_projection_count=stage*36,
            source_sha256=request['source_sha256'],data_digest=request['data_digest'],contract=b.CONTRACT),
        distill_stage_history=list(request.get('recovery_history',[])))
    saved=save_cat_v6_full_checkpoint(deployment,str(root/'model'),**kwargs)
    expected=runtime.records(deployment.state_dict());b.write(root/'expected_state.json',expected)
    budget=audit_state_dict(saved['meta_payload'],deployment.state_dict(),max_bpw=16,max_lora_rank=0)
    b.require(not budget['contract_violations'],'Snapshot native storage contract failed.')
    del deployment;gc.collect()
    for device in range(2):
        with b.torch.cuda.device(device):b.torch.cuda.empty_cache()
    restore_rng(random_state)
    b.require(rng_exact(random_state) and live_stamp(active,optimizer)==stamp,
        'Independent midpoint snapshot changed masters/Adam/LR or Python/NumPy/CPU/CUDA RNG.')
    b.write(root/'snapshot.json',dict(status='BF16_snapshot_saved',checkpoint=b.event(root/'model'),
        actual_updates=step,persistent_fields=len(expected),source_sha256=request['source_sha256'],
        complete_live_state_preserved=True,bit_budget=budget))
    return root/'model'


def run(model,stage,request,root,save_kwargs,observe,evaluate):
    dataset,targets=request['_dataset'],request['_targets']
    root.mkdir(parents=True,exist_ok=False)
    fit_random=rng()
    random.seed(137);np.random.seed(137);b.torch.manual_seed(137)
    handles=[];optimizer=None;active={};began=time.monotonic()
    parallel_attributes={key:(hasattr(model,key),getattr(model,key,None))
        for key in ('hf_device_map','is_parallelizable','model_parallel')}
    before=runtime.records(model.state_dict())
    try:
        model.eval().requires_grad_(False)
        b.restore_rope_alias(model)
        handles,mapping=b.old.old_life.apply_map(model)
        runtime.install(model,{},stage,'terminal',active)
        b.require(len(active)==216*stage,'Each installed projection has6 native packed decoder leaves; train the full prefix.')
        frozen=b.old.old_life.frozen_proof(model,set(active))
        optimizer=b.torch.optim.AdamW(list(active.values()),lr=1e-6,betas=(.9,.95),eps=1e-8,weight_decay=0.,foreach=False)
        baseline=panel(model,dataset,targets)
        b.write(root/'panel_000000.json',baseline)
        short_indices=[2,10,45] if stage==7 else [2 if stage==1 else 10 if stage==2 else 0]
        allocation=None if request['validation_only'] else budget.initial(
            request['_previous_post_correct'],request['_pre_correct'],b.CATEGORIES[stage-1])
        preceding=None if allocation is None else allocation['pre_correct']
        rows=[];indices=[];decisions=[]
        for step in range(1,(len(short_indices) if request['validation_only'] else 384)+1):
            index=short_indices[step-1] if request['validation_only'] else (step-1)%128
            indices.append(index)
            optimizer.zero_grad(set_to_none=True);b.old.old_life.synchronize()
            for device in range(2):b.torch.cuda.reset_peak_memory_stats(device)
            started=time.monotonic()
            losses,case_ids=training.losses(model,dataset,targets,index,True)
            gradients=training.gradient_contract(model,active)
            norm=float(b.torch.nn.utils.clip_grad_norm_(list(active.values()),1.5,error_if_nonfinite=True,foreach=False))
            b.require(math.isfinite(norm) and norm>0,'Finite nonzero joint M+QA gradient required.')
            lr=1e-6*(1-(step-1)/500);optimizer.param_groups[0]['lr']=lr;optimizer.step()
            for name,p in active.items():
                item=optimizer.state[p]
                b.require(int(item['step'])==step and item['exp_avg'].dtype==item['exp_avg_sq'].dtype==b.torch.float32
                    and b.torch.isfinite(p).all() and b.torch.isfinite(item['exp_avg']).all()
                    and b.torch.isfinite(item['exp_avg_sq']).all(),'Prefix Adam state/value contract failed: '+name)
            b.old.old_life.synchronize()
            row=dict(step=step,group_index=index,lr=lr,losses=losses,weighted_objective=sum(losses.values()),
                gradients=gradients,gradient_norm_before_clip=norm,case_ids=case_ids,
                exposure=b.data.exposure(dataset,targets,[index]),seconds=time.monotonic()-started,memory=b.old.old_life.memory())
            rows.append(row);b.write(root/'updates'/f'step_{step:06d}.json',row)
            observe('prefix_recovery_update',dict(category=b.CATEGORIES[stage-1],active_projections=36*stage,
                **{key:row[key] for key in ('step','group_index','lr','weighted_objective','seconds','memory')}))
            if (request['validation_only'] and stage==7 and step==2) or (not request['validation_only'] and step in budget.NODES):
                optimizer.zero_grad(set_to_none=True)
                measured=panel(model,dataset,targets,baseline);b.write(root/f'panel_{step:06d}.json',measured)
                midpoint=snapshot(model,active,optimizer,stage,step,root/f'step_{step:06d}',request,save_kwargs)
                resume_path=None
                if request['validation_only']:
                    resume_path=training_state(model,active,optimizer,step,root/f'step_{step:06d}',request,stage,next_group=45)
                stamp,random_before=live_stamp(active,optimizer),rng()
                scored=evaluate(midpoint,root/f'step_{step:06d}'/'evaluation','mmlu',request['validation_only'],
                    fresh=request['validation_only'])
                restore_rng(random_before)
                b.require(live_stamp(active,optimizer)==stamp and rng_exact(random_before),
                    'Midpoint readout changed the paused live masters/Adam/RNG.')
                if resume_path is not None:
                    optimizer=b.torch.optim.AdamW(list(active.values()),lr=1e-6,betas=(.9,.95),eps=1e-8,
                        weight_decay=0.,foreach=False)
                    restored=restore_training_state(resume_path['path'],model,active,optimizer,request,stage)
                    b.require(restored['completed_updates']==step and restored['next_group_index']==short_indices[step],
                        'Actual validation continuation cursor differs from its next real group.')
                    b.write(root/'resume_validation.json',dict(status='actual_FP32_Adam_RNG_state_restored',
                        saved_step=step,next_group_index=restored['next_group_index'],state=resume_path,
                        boundary='The following real group45 update consumes this restored optimizer/state.'))
                if not request['validation_only']:
                    current=budget.correct(scored)
                    decision=budget.decide(step,current,preceding,allocation['initial_budget'],allocation['previous_post_correct'])
                    decisions.append(decision);preceding=current
                    request['_budget_state']=dict(allocation=allocation,decisions=decisions)
                    b.write(root/'budget.json',request['_budget_state'])
                    observe('adaptive_category_budget',dict(category=b.CATEGORIES[stage-1],**decision))
                cleanup=b.cleanup_model(midpoint,'Midpoint full state/fresh/readout is complete; live Adam continues or final post is independently saved.')
                snapshot_record=b.read(Path(midpoint).parent/'snapshot.json')
                snapshot_record['completed_payload_cleanup']=cleanup;b.write(Path(midpoint).parent/'snapshot.json',snapshot_record)
                if not request['validation_only'] and not decision['continue_training']:break
        optimizer.zero_grad(set_to_none=True)
        final=panel(model,dataset,targets,baseline);b.write(root/f'panel_{len(indices):06d}.json',final)
        b.require(b.old.old_life.frozen_proof(model,set(active))==frozen,'Recovery changed a frozen future/code/protection/rotation/backbone field.')
        state_binding=training_state(model,active,optimizer,len(indices),root,request,stage) if stage==7 else None
        result=dict(category=b.CATEGORIES[stage-1],actual_updates=len(indices),active_projections=36*stage,
            budget_allocation=allocation,budget_decisions=decisions,group_indices=indices,
            exposure=b.data.exposure(dataset,targets,indices),
            fresh_Adam=True,schedule_steps=500,first_lr=1e-6,last_lr=rows[-1]['lr'],
            full_frozen_complement_exact=True,initial_panel=baseline,final_panel=final,
            layer_device_map=mapping,training_state=state_binding,
            elapsed_seconds=time.monotonic()-began,max_memory={str(d):{key:max(row['memory'][str(d)][key] for row in rows)
                for key in ('peak_allocated','peak_reserved')} for d in range(2)})
        b.write(root/'training.json',result)
        model.gradient_checkpointing_disable();model.eval().requires_grad_(False)
        for name,p in active.items():p.data=p.data.bfloat16()
        for module in runtime.projections(model).values():
            if isinstance(module,b.old.VAELinear):module.disable_trainable_decode();module.clear_decoded_weight_cache()
        native_save.refresh_vae_linear_runtime_after_state_load(model)
        b.old.install_precision_runtime(model,b.torch.bfloat16)
        after=runtime.records(model.state_dict())
        b.require({key:value for key,value in after.items() if key not in active}==
            {key:value for key,value in before.items() if key not in active},'BF16 stage commit changed the frozen complement.')
        b.require(any(before[name]!=after[name] for name in active),'Actual decoder updates must survive BF16 stage commit.')
        return result
    finally:
        for handle in handles:handle.remove()
        del optimizer
        model.to('cpu');gc.collect()
        b.restore_rope_alias(model)
        for key,(present,value) in parallel_attributes.items():
            if present:setattr(model,key,value)
            elif hasattr(model,key):delattr(model,key)
        for device in range(2):
            with b.torch.cuda.device(device):b.torch.cuda.empty_cache()
        restore_rng(fit_random)
        b.require(rng_exact(fit_random),'Recovery/readout did not restore the exact VAE fitting RNG stream.')
