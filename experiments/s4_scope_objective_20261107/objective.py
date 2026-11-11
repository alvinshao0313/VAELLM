"""Retain the original target; change only the knowledge KL direction in B."""
import math
from . import bindings as b
from experiments.s4_webmed_recovery_20261105 import data
from experiments.mlp_learnability_cmcc_20261030.fit import group_cases


def knowledge(log_probs,mixed,raw,adaptive,coefficient=None):
    t=b.torch
    b.require(log_probs.shape==mixed.shape==raw.shape==(32,4),'Keep real M32/four-option support.')
    b.require(all(x.dtype==t.float32 and t.isfinite(x).all() for x in (log_probs,mixed,raw)),
        'Finite FP32 normalized log-probabilities required.')
    mixed=mixed.to(log_probs.device);raw=raw.to(log_probs.device)
    entropy=-(raw.exp()*raw).sum(-1)
    lam=(entropy.clamp(max=math.log(4))/math.log(4)).mean().detach()
    b.require(0<=float(lam)<=1,'Teacher entropy outside normalized probability range.')
    forward=t.nn.functional.kl_div(log_probs,mixed,log_target=True,reduction='none').sum(-1).mean()
    reverse=(log_probs.exp()*(log_probs-mixed)).sum(-1).mean()
    selected=lam if coefficient is None else log_probs.new_tensor(coefficient)
    loss=selected*forward+(1-selected)*reverse if adaptive else forward
    return loss,dict(lambda_raw_teacher=float(lam),forward_KL=float(forward.detach()),
        reverse_KL=float(reverse.detach()),selected_KL=float(loss.detach()))


def losses(model,dataset,targets,raw_targets,index,arm,backward):
    group,cases,qa_keys=group_cases(dataset,targets,index)
    prediction=data.mmlu_log_distribution(model,cases)
    mixed=b.torch.cat([targets[c['case_id']]['target_log_probs'] for c in cases]).to(prediction.device)
    raw=b.torch.cat([raw_targets[c['case_id']]['target_log_probs'] for c in cases]).to(prediction.device)
    value,details=knowledge(prediction,mixed,raw,b.ARMS[arm]['objective']=='adaptive')
    gold=b.torch.tensor([c['gold_index'] for c in cases],device=prediction.device)
    details['gold_NLL']=float(-prediction.detach().gather(1,gold[:,None]).mean())
    details['sources']={}
    for name,start in [('webinstruct',0),('medmcqa',16)]:
        sl=slice(start,start+16);v=(mixed[sl].exp()*(mixed[sl]-prediction[sl])).sum(-1)
        details['sources'][name]=dict(common_mixed_KL=float(v.detach().mean()),cases=16,
            gold_NLL=float(-prediction.detach()[sl].gather(1,gold[sl,None]).mean()),weight=.25)
    loss=.5*value;b.require(b.torch.isfinite(loss),'Nonfinite knowledge loss.')
    values=dict(mmlu=float(loss.detach()))
    if backward:loss.backward()
    del loss,value,prediction,mixed,raw
    for key in qa_keys:
        case=targets[key]['case'];b.require(group['weights'][key]==.5/8,'Original QA weight changed.')
        logits=data.legacy.base.answer_logits(model,case)
        loss=data.legacy.base.answer_kl(logits,case,targets[key]['target_log_probs'])*(.5/8)
        b.require(b.torch.isfinite(loss),'Nonfinite original static QA KL.')
        values[case['domain']]=values.get(case['domain'],0.)+float(loss.detach())
        if backward:loss.backward()
        del loss,logits
    return values,group['case_ids'],details
