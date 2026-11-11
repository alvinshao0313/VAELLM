"""Continuous original AdamW mathematics with fixed readout/snapshot boundaries."""
from collections import defaultdict
import math
import time

from . import bindings as b
from . import readout, snapshot
from experiments.mlp_learnability_cmcc_20261030.fit import group_cases


def fit(model, arm, active, dataset, recovery, references, identity, root, non_mlp, protection,
        fixture, baseline, observe, deadline):
    torch = b.torch
    optimizer = torch.optim.AdamW(list(active.values()),lr=b.BASE_LR[arm],betas=(.9,.95),eps=1e-8,
        weight_decay=0.,foreach=False)
    master_ids = {name:(id(p),p.data_ptr()) for name,p in active.items()}
    frozen = b.life.frozen_proof(model,set(active))
    indices = [2,10] if identity['validation_only'] else list(range(32))
    boundaries = b.snapshot_steps(identity['validation_only'])
    updates, snapshots, readouts = [], [], []
    for step,index in enumerate(b.groups(identity['validation_only']),1):
        b.require(time.monotonic()<deadline, 'The fixed learning run exceeded its explicit runtime budget.')
        optimizer.zero_grad(set_to_none=True)
        b.life.synchronize()
        for device in range(2): torch.cuda.reset_peak_memory_stats(device)
        began = time.monotonic()
        group,m_cases,qa_keys = group_cases(dataset,recovery,index)
        target = torch.cat([recovery[case['case_id']]['target_log_probs'] for case in m_cases])
        loss,_values = b.data.legacy.mmlu.weighted_loss(b.data.mmlu_log_distribution(model,m_cases),target)
        b.require(torch.isfinite(loss).item(), 'Nonfinite original M32 objective.')
        losses = defaultdict(float,mmlu=float(loss.detach()))
        loss.backward()
        del loss,target,_values
        qa_rows = []
        for key in qa_keys:
            b.require(time.monotonic()<deadline, 'The fixed learning run exceeded its explicit runtime budget.')
            case = recovery[key]['case']
            b.require(group['weights'][key] == .5/8, 'Keep the original QA document weights.')
            logits = b.data.legacy.base.answer_logits(model,case)
            loss = b.data.legacy.base.answer_kl(logits,case,recovery[key]['target_log_probs'])*(.5/8)
            b.require(torch.isfinite(loss).item(), 'Nonfinite static full-vocabulary QA objective.')
            value = float(loss.detach())
            losses[case['domain']] += value
            loss.backward()
            qa_rows.append(dict(case_id=key,domain=case['domain'],input_tokens=len(case['input_ids']),
                answer_positions=len(case['answer_query_positions']),weighted_loss=value))
            del loss,logits
        gradients = b.life.gradient_check(model,active)
        norm = float(torch.nn.utils.clip_grad_norm_(list(active.values()),1.5,error_if_nonfinite=True,foreach=False))
        b.require(math.isfinite(norm) and norm>0, 'Require one finite nonzero joint M+QA clip.')
        lr = b.BASE_LR[arm]*(1-(step-1)/500)
        optimizer.param_groups[0]['lr'] = lr
        previous_states = {name:id(optimizer.state[p]) for name,p in active.items() if p in optimizer.state}
        optimizer.step()
        state = b.life.optimizer_check(optimizer,active,step)
        b.require(master_ids == {name:(id(p),p.data_ptr()) for name,p in active.items()}
            and all(id(optimizer.state[active[name]]) == value for name,value in previous_states.items()),
            'Continuous live master/storage/Adam identities changed, including after snapshot.')
        b.require(b.life.frozen_proof(model,set(active)) == frozen, 'A frozen code/protection/non-MLP field changed.')
        b.life.synchronize()
        row = dict(step=step,group_index=index,lr=lr,domain_losses=dict(losses),weighted_objective=sum(losses.values()),
            gradient_norm_before_clip=norm,gradient_norm_after_clip=min(norm,1.5),
            projection_gradient_norms=gradients,Adam=state,QA=qa_rows,M_case_ids=group['case_ids'][:32],
            all108_gradient_finite_nonzero=True,frozen_persistent_fields_exact=True,
            master_and_state_identity_continuous=True,seconds=time.monotonic()-began,memory=b.life.memory())
        path = root/'updates'/f'step_{step:06d}.json'
        b.write(path,row)
        updates.append(b.binding(path))
        observe('all_MLP_learning_update',dict(step=step,group_index=index,lr=lr,
            seconds=row['seconds'],objective=row['weighted_objective'],memory=row['memory']))
        if step in boundaries:
            optimizer.zero_grad(set_to_none=True)
            measured = readout.evaluate(model,dataset,recovery,indices,baseline)
            b.require(b.life.frozen_proof(model,set(active)) == frozen, 'Boundary readout changed frozen state.')
            measured['actual_updates'] = step
            readouts.append(measured)
            snapshots.append(snapshot.save(model,arm,active,optimizer,identity,step,root,non_mlp,protection,references,fixture))
            observe('all_MLP_snapshot_complete',dict(step=step,snapshot=snapshots[-1],
                objective=measured['weighted_objective'],M_accuracy=measured['M_accuracy'],memory=b.life.memory()))
    optimizer.zero_grad(set_to_none=True)
    return dict(steps=len(updates),group_indices=b.groups(identity['validation_only']),updates=updates,
        snapshots=snapshots,readouts=readouts,training_contract=b.CONTRACT,
        frozen_before=frozen,frozen_after=b.life.frozen_proof(model,set(active)),
        final_memory=b.life.memory(),no_optimizer_resume_artifact=True),optimizer
