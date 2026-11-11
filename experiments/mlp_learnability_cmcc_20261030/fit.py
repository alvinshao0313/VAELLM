"""Two real static teacher-forcing updates, with one joint mixed-device clip."""
from collections import defaultdict
import math
import time

from . import bindings as b
from . import data,lifecycle as life

torch = b.torch
GROUPS = [2,10]
BASE_LR = dict(decoder=1e-6,dense=1e-5)


def group_cases(dataset,recovery,index):
    group = dataset['groups'][index]
    b.require(group['group_index'] == index and len(group['case_ids']) == 40, 'Keep each complete original M32+QA8 group.')
    return group,[recovery[key]['case'] for key in group['case_ids'][:32]],group['case_ids'][32:]


@torch.no_grad()
def evaluate(model,dataset,recovery):
    previous = model.training
    model.eval()
    rows = []
    for index in GROUPS:
        group,m_cases,qa_keys = group_cases(dataset,recovery,index)
        target = torch.cat([recovery[case['case_id']]['target_log_probs'] for case in m_cases])
        distribution = data.mmlu_log_distribution(model,m_cases)
        m_loss,_values = data.legacy.mmlu.weighted_loss(distribution,target)
        losses = defaultdict(float,mmlu=float(m_loss))
        for key in qa_keys:
            case = recovery[key]['case']
            logits = data.legacy.base.answer_logits(model,case)
            loss = data.legacy.base.answer_kl(logits,case,recovery[key]['target_log_probs'])*(.5/8)
            losses[case['domain']] += float(loss)
            del logits,loss
        rows.append(dict(group_index=index,domain_losses=dict(losses),weighted_objective=sum(losses.values()),
                         M_case_ids=group['case_ids'][:32],QA_case_ids=qa_keys))
    model.train(previous)
    b.require(all(math.isfinite(row['weighted_objective']) for row in rows), 'Nonfinite fixed-group TF readout.')
    return rows


def fit(model,arm,active,dataset,recovery,references,observe,deadline):
    b.require(arm in BASE_LR, 'Only the two explicit all-MLP parameterizations are supported.')
    optimizer = torch.optim.AdamW(list(active.values()),lr=BASE_LR[arm],betas=(.9,.95),eps=1e-8,
                                  weight_decay=0.,foreach=False)
    master_ids = {name:(id(p),p.data_ptr()) for name,p in active.items()}
    decoder_initial = ({name:p.detach().cpu().bfloat16() for name,p in active.items()} if arm == 'decoder' else {})
    frozen_before = life.frozen_proof(model,set(active))
    before = evaluate(model,dataset,recovery)
    b.require(life.frozen_proof(model,set(active)) == frozen_before, 'Initial fixed-group readout changed persistent frozen state.')
    observe('fixed_TF_groups_before_updates',dict(groups=before))
    rows = []
    for step,index in enumerate(GROUPS,1):
        b.require(time.monotonic()<deadline, 'The bounded mechanism short exceeded its wall-clock budget.')
        optimizer.zero_grad(set_to_none=True)
        life.synchronize()
        for device in range(2): torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        group,m_cases,qa_keys = group_cases(dataset,recovery,index)
        target = torch.cat([recovery[case['case_id']]['target_log_probs'] for case in m_cases])
        loss,_values = data.legacy.mmlu.weighted_loss(data.mmlu_log_distribution(model,m_cases),target)
        b.require(torch.isfinite(loss).item(), 'Nonfinite unchanged M32 objective.')
        losses = defaultdict(float,mmlu=float(loss.detach()))
        loss.backward()
        del loss,target,_values
        observe('M32_backward_complete',dict(step=step,group_index=index,weighted_loss=losses['mmlu']))
        qa_rows = []
        for key in qa_keys:
            b.require(time.monotonic()<deadline, 'The bounded mechanism short exceeded its wall-clock budget.')
            case = recovery[key]['case']
            b.require(group['weights'][key] == .5/8, 'The original QA document weighting changed.')
            logits = data.legacy.base.answer_logits(model,case)
            loss = data.legacy.base.answer_kl(logits,case,recovery[key]['target_log_probs'])*(.5/8)
            b.require(torch.isfinite(loss).item(), 'Nonfinite unchanged static full-vocabulary QA objective.')
            value = float(loss.detach())
            losses[case['domain']] += value
            loss.backward()
            qa_rows.append(dict(case_id=key,domain=case['domain'],input_tokens=len(case['input_ids']),
                answer_positions=len(case['answer_query_positions']),weighted_loss=value))
            del loss,logits
            observe('QA_backward_complete',dict(step=step,group_index=index,**qa_rows[-1]))
        gradients = life.gradient_check(model,active)
        norm = float(torch.nn.utils.clip_grad_norm_(list(active.values()),1.5,error_if_nonfinite=True,foreach=False))
        b.require(math.isfinite(norm) and norm>0, 'The single joint M+QA mixed-device gradient norm must be finite and nonzero.')
        lr = BASE_LR[arm]*(1-(step-1)/500)
        optimizer.param_groups[0]['lr'] = lr
        previous_state_ids = {name:id(optimizer.state[p]) for name,p in active.items() if p in optimizer.state}
        optimizer.step()
        state = life.optimizer_check(optimizer,active,step)
        b.require(master_ids == {name:(id(p),p.data_ptr()) for name,p in active.items()}
            and all(id(optimizer.state[active[name]]) == identity for name,identity in previous_state_ids.items()),
            'Live FP32 Parameter/storage or Adam state identities changed.')
        motion = life.parameter_motion(model,active,references,decoder_initial)
        b.require(life.frozen_proof(model,set(active)) == frozen_before, 'A code, route, protection, attention or non-MLP field changed.')
        life.synchronize()
        row = dict(step=step,group_index=index,lr=lr,domain_losses=dict(losses),weighted_objective=sum(losses.values()),
            gradient_norm_before_clip=norm,gradient_norm_after_clip=min(norm,1.5),clip_coefficient=min(1.,1.5/(norm+1e-6)),
            projection_gradient_norms=gradients,Adam=state,parameter_motion=motion,QA=qa_rows,
            M_case_ids=group['case_ids'][:32],QA_case_ids=qa_keys,student_M32_FW_BW=1,student_QA_FW_BW=8,
            seconds=time.monotonic()-started,memory_by_logical_device=life.memory(),
            all108_gradient_finite_nonzero=True,frozen_persistent_fields_exact=True,master_and_state_identity_continuous=True)
        rows.append(row)
        observe('all_MLP_static_TF_update',row)
    optimizer.zero_grad(set_to_none=True)
    after = evaluate(model,dataset,recovery)
    b.require(life.frozen_proof(model,set(active)) == frozen_before, 'Final fixed-group readout changed frozen state.')
    return dict(steps=2,group_indices=GROUPS,base_lr=BASE_LR[arm],schedule_steps=500,warmup_steps=0,
        optimizer='AdamW',betas=[.9,.95],eps=1e-8,weight_decay=0.,foreach=False,joint_clip=1.5,
        trainable_leaves=len(active),trainable_elements=sum(p.numel() for p in active.values()),
        fixed_groups_before=before,fixed_groups_after=after,updates=rows,frozen_before=frozen_before,
        frozen_after=life.frozen_proof(model,set(active)),final_memory_by_logical_device=life.memory(),
        no_optimizer_resume_artifact=True),optimizer
