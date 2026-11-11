"""Unchanged prefix update mathematics plus the explicit zero-update up hook."""
from pathlib import Path
import gc
import math
import random
import time
import numpy as np
from . import bindings as b
from experiments.s4_from_bf16_20261102 import budget

_math=b.bound_implementation('recovery')
runtime,training,native_save=_math.runtime,_math.training,_math.native_save
rng,restore_rng,rng_exact,live_stamp=_math.rng,_math.restore_rng,_math.rng_exact,_math.live_stamp
panel,snapshot,training_state=_math.panel,_math.snapshot,_math.training_state
restore_training_state=_math.restore_training_state


def skip(model,request,root):
    began=time.monotonic();root.mkdir(parents=True,exist_ok=False)
    state=runtime.records(model.state_dict());random_state=rng()
    identities={name:(id(p),p.data_ptr(),p._version,p.requires_grad) for name,p in model.named_parameters()}
    mode=model.training
    installed=[m for m in runtime.projections(model).values() if isinstance(m,b.old.VAELinear)]
    b.require(len(installed)==216 and all(m.residual_stages==4 for m in installed),
        'The skipped up boundary must contain all216 actual S4 prefix projections.')
    b.require(runtime.records(model.state_dict())==state and rng_exact(random_state)
        and mode==model.training and identities=={name:(id(p),p.data_ptr(),p._version,p.requires_grad)
            for name,p in model.named_parameters()},'Skip changed complete model/parameter/RNG state.')
    result=dict(category='up_proj',actual_updates=0,did_train=False,active_projections=0,
        installed_prefix_projections=216,fresh_Adam=False,optimizer_created=False,training_groups_consumed=0,
        budget_allocation=dict(initial_budget=0,rule='predeclared_skip_up_only'),budget_decisions=[],group_indices=[],
        exposure=dict(presentations=dict(mmlu=0,qasper=0,hotpotqa=0),total_input_tokens=0,total_answer_positions=0,groups=[]),
        first_lr=None,last_lr=None,schedule_steps=500,full_frozen_complement_exact=True,complete_state_exact=True,
        parameter_identity_exact=True,RNG_exact=True,initial_panel=None,final_panel=None,
        training_state=None,elapsed_seconds=time.monotonic()-began,max_memory={})
    b.write(root/'training.json',result)
    return result


def run(model,stage,request,root,save_kwargs,observe,evaluate):
    if stage==6:return skip(model,request,root)
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
        short_indices=[0]
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
