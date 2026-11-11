"""One continuous56-update Adam run with symmetric prefix activation."""
from collections import defaultdict
import math
import time

from . import bindings as b, runtime
from experiments.mlp_learnability_cmcc_20261030.fit import group_cases


def losses(model,dataset,recovery,index,backward):
    group,m_cases,qa_keys=group_cases(dataset,recovery,index)
    target=b.torch.cat([recovery[case['case_id']]['target_log_probs'] for case in m_cases])
    loss,_=b.data.legacy.mmlu.weighted_loss(b.data.mmlu_log_distribution(model,m_cases),target)
    b.require(b.torch.isfinite(loss).item(), 'Nonfinite original once-mixed M objective.')
    values=defaultdict(float,mmlu=float(loss.detach()))
    if backward:loss.backward()
    del loss,target
    for key in qa_keys:
        case=recovery[key]['case']
        b.require(group['weights'][key]==.5/8, 'Keep the original equal QA document weights.')
        logits=b.data.legacy.base.answer_logits(model,case)
        loss=b.data.legacy.base.answer_kl(logits,case,recovery[key]['target_log_probs'])*(.5/8)
        b.require(b.torch.isfinite(loss).item(), 'Nonfinite original gold-prefix static QA KL.')
        values[case['domain']]+=float(loss.detach())
        if backward:loss.backward()
        del loss,logits
    return dict(values),group['case_ids']


@b.torch.no_grad()
def probe(model,dataset,recovery):
    previous=model.training
    model.eval()
    values,ids=losses(model,dataset,recovery,2,False)
    model.train(previous)
    return dict(group_index=2,case_ids=ids,domain_losses=values,weighted_objective=sum(values.values()),
        boundary='One fixed original train group; not a full-pool or downstream quality score.')


def gradient_contract(model,active):
    norms=defaultdict(float)
    for name,p in active.items():
        b.require(p.grad is not None and p.grad.dtype==b.torch.float32
            and b.torch.isfinite(p.grad).all().item(), 'Missing/nonfinite FP32 decoder gradient: '+name)
        norms[name.split('._parallel_stage_decoder.')[0]]+=float(p.grad.double().square().sum())
    b.require(all(value>0 and math.isfinite(value) for value in norms.values()),
        'Every active projection needs a finite nonzero aggregate decoder gradient.')
    b.require(all(p.grad is None for name,p in model.named_parameters() if name not in active),
        'Frozen code/protection/rotation/future/backbone received a Parameter gradient.')
    return dict(active_projections=len(norms),all_finite_nonzero=True)


def adam_stamp(optimizer,active):
    return {name:dict(master=(id(p),p.data_ptr(),p._version),state_id=id(optimizer.state[p]),
        moments={key:(value.data_ptr(),value._version) for key,value in optimizer.state[p].items()
            if b.torch.is_tensor(value)}) for name,p in active.items() if p in optimizer.state}


def fit(model,bank,evidence,arm,short,dataset,recovery,root,observe,deadline):
    active={};optimizer=None;activation={};rows=[];phases=[];step=0
    fixture_case=recovery[dataset['groups'][2]['case_ids'][0]]['case']
    fixture=dict(input_ids=fixture_case['input_ids'].unsqueeze(0),
        attention_mask=b.torch.ones_like(fixture_case['input_ids']).unsqueeze(0),
        source_case_id=fixture_case['case_id'],input_signature=fixture_case['input_signature'])
    initial=None
    for stage,indices in b.segments(short):
        old_stamp=adam_stamp(optimizer,active) if optimizer is not None else {}
        new=runtime.install(model,bank,stage,arm,active)
        for name in new:activation[name]=step
        if optimizer is None:
            optimizer=b.torch.optim.AdamW(list(new.values()),lr=b.CONTRACT['lr'],
                betas=tuple(b.CONTRACT['betas']),eps=b.CONTRACT['eps'],weight_decay=0.,foreach=False)
        else:optimizer.add_param_group(dict(params=list(new.values())))
        b.require(adam_stamp(optimizer,{name:active[name] for name in old_stamp})==old_stamp,
            'Category transition reset or changed an earlier master/Adam moment/step.')
        b.require(all(p not in optimizer.state for p in new.values()), 'New decoder Adam state must start uninitialized.')
        warm=runtime.warm_frozen(model)
        topology=runtime.composition(model,evidence,active,stage,arm) if short or stage in (1,7) else dict(
            stage=stage,native=len([m for m in runtime.projections(model).values() if isinstance(m,b.VAELinear)]),
            active_projections=36*stage)
        frozen=b.old_life.frozen_proof(model,set(active)) if short else None
        if initial is None:initial=probe(model,dataset,recovery)
        observe('prefix_segment_initialized',dict(stage=stage,actual_global_step=step,
            new_projections=len({name.split('._parallel_stage_decoder.')[0] for name in new}),
            topology=topology,old_master_Adam_identity_preserved=True,warmup=warm,memory=b.old_life.memory()))
        phase_rows=[]
        for index in indices:
            b.require(time.monotonic()<deadline, 'Fixed comparison runtime budget exceeded.')
            optimizer.zero_grad(set_to_none=True)
            b.old_life.synchronize()
            for device in range(2):b.torch.cuda.reset_peak_memory_stats(device)
            began=time.monotonic();step+=1
            values,case_ids=losses(model,dataset,recovery,index,True)
            gradients=gradient_contract(model,active)
            norm=float(b.torch.nn.utils.clip_grad_norm_(list(active.values()),1.5,error_if_nonfinite=True,foreach=False))
            b.require(math.isfinite(norm) and norm>0, 'Joint M+QA gradient norm must be finite and nonzero.')
            lr=1e-6*(1-(step-1)/500)
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step()
            expected_steps={}
            for name,p in active.items():
                state=optimizer.state[p];count=step-activation[name]
                b.require(int(state['step'])==count and state['exp_avg'].dtype==state['exp_avg_sq'].dtype==b.torch.float32
                    and b.torch.isfinite(p).all().item() and b.torch.isfinite(state['exp_avg']).all().item()
                    and b.torch.isfinite(state['exp_avg_sq']).all().item(), 'Per-leaf Adam activation clock failed: '+name)
                expected_steps[b.category(name)]=count
            b.old_life.synchronize()
            exposure=b.data.exposure(dataset,recovery,[index])
            row=dict(step=step,stage=stage,group_index=index,case_ids=case_ids,lr=lr,domain_losses=values,
                input_tokens=exposure['total_input_tokens'],supervised_answer_positions=exposure['total_answer_positions'],
                weighted_objective=sum(values.values()),gradient_norm_before_clip=norm,gradients=gradients,
                per_category_Adam_steps=expected_steps,seconds=time.monotonic()-began,memory=b.old_life.memory())
            b.write(root/'updates'/f'step_{step:06d}.json',row)
            rows.append(row);phase_rows.append(row)
            observe('static_TF_update', {key:row[key] for key in ('step','stage','group_index','lr','weighted_objective','seconds','memory')})
        optimizer.zero_grad(set_to_none=True)
        if short:b.require(b.old_life.frozen_proof(model,set(active))==frozen, 'Short changed the complete frozen complement.')
        phase=dict(stage=stage,category=b.CATEGORIES[stage-1],global_step=step,
            group_indices=indices,actual_updates=len(phase_rows),active_projections=36*stage,
            per_category_Adam_steps=phase_rows[-1]['per_category_Adam_steps'],
            last_objective=phase_rows[-1]['weighted_objective'],seconds=sum(row['seconds'] for row in phase_rows),
            exposure=b.data.exposure(dataset,recovery,indices),topology=topology,
            prior_Adam_state_preserved=True,complete_frozen_complement_exact=True if short else None)
        phases.append(phase)
        b.write(root/'segments'/f'stage_{stage:02d}.json',phase)
    b.require(step==(4 if short else 56), 'Finish the fixed update budget.')
    if not short:b.require(rows[-1]['per_category_Adam_steps']==dict(zip(b.CATEGORIES,[56,48,40,32,24,16,8])),
        'Both arms must end with the same per-category update counts.')
    final=probe(model,dataset,recovery)
    source_state=b.torch.load(b.SOURCE/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    changed=sum(int((p.detach().cpu().bfloat16()!=source_state[name]).sum()) for name,p in active.items())
    b.require(changed>0, 'At least one real decoder update must survive BF16 deployment.')
    final_topology=runtime.composition(model,evidence,active,7,arm)
    return active,optimizer,dict(actual_updates=step,phases=phases,initial_probe=initial,final_probe=final,
        final_per_category_Adam_steps=rows[-1]['per_category_Adam_steps'],BF16_changed_decoder_elements=changed,
        final_topology=final_topology,exposure=b.data.exposure(dataset,recovery,[row['group_index'] for row in rows]),
        max_memory={str(d):{key:max(row['memory'][str(d)][key] for row in rows)
            for key in ('peak_allocated','peak_reserved')} for d in range(2)}),fixture
