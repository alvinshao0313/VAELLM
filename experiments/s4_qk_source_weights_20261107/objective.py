"""Original M32 four-option and static full-vocabulary QA KL with explicit case weights."""
from . import bindings as b,data
from experiments.mlp_learnability_cmcc_20261030.fit import group_cases
from experiments.s4_webmed_recovery_20261105 import data as original
from experiments.progressive_residual_bsq_20261030.recovery_compare_20261101.training import gradient_contract


def knowledge_loss(log_probs,target,group):
    b.require(log_probs.shape==target.shape==(32,4),'Keep the original M32 four-option supports.')
    values=b.torch.nn.functional.kl_div(log_probs,target.to(log_probs.device),log_target=True,reduction='none').sum(-1)
    weights=log_probs.new_tensor([group['weights'][key] for key in group['case_ids'][:32]])
    loss=(values*weights).sum()
    details={}
    for source in ('webinstruct','medmcqa'):
        mask=b.torch.tensor([data.source_name(key)==source for key in group['case_ids'][:32]],device=values.device)
        b.require(int(mask.sum())==16,'Keep each original source16 in the same M32 forward.')
        details[source]=dict(unweighted_KL=float(values[mask].detach().mean()),
            weighted_objective=float((values[mask]*weights[mask]).detach().sum()),
            actual_total_weight=float(weights[mask].sum()),cases=16)
    return loss,details


def losses(model,dataset,targets,index,backward):
    group,m_cases,qa_keys=group_cases(dataset,targets,index)
    target=b.torch.cat([targets[case['case_id']]['target_log_probs'] for case in m_cases])
    loss,details=knowledge_loss(original.mmlu_log_distribution(model,m_cases),target,group)
    b.require(b.torch.isfinite(loss).item(),'Nonfinite source-weight M objective.')
    values=dict(mmlu=float(loss.detach()))
    if backward:loss.backward()
    del loss,target
    qa_raw=0.;qa_weighted=0.
    for key in qa_keys:
        case=targets[key]['case'];b.require(group['weights'][key]==.5/8,'Original QA weights required.')
        logits=original.legacy.base.answer_logits(model,case)
        raw=original.legacy.base.answer_kl(logits,case,targets[key]['target_log_probs'])
        loss=raw*group['weights'][key]
        b.require(b.torch.isfinite(loss).item(),'Nonfinite original gold-prefix static QA KL.')
        values[case['domain']]=values.get(case['domain'],0.)+float(loss.detach())
        qa_raw+=float(raw.detach());qa_weighted+=float(loss.detach())
        if backward:loss.backward()
        del loss,raw,logits
    details['QA']=dict(unweighted_KL=qa_raw/8,weighted_objective=qa_weighted,actual_total_weight=.5,cases=8)
    return values,group['case_ids'],details
