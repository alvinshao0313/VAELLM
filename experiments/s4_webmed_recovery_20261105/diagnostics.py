"""Fixed128 early/late/untrained panels, split by actual Web/Med source."""
from . import bindings as b


@b.torch.no_grad()
def gold(model,cases,targets=None,baseline=None):
    values={};correct=0;nll=0.;kl=0.
    for start in range(0,len(cases),32):
        batch=cases[start:start+32];b.require(len(batch)==32,'Preserve the actual M32 diagnostic math.')
        prediction=b.data.mmlu_log_distribution(model,batch).cpu()
        for index,case in enumerate(batch):
            key=case['case_id'];loss=float(-prediction[index,case['gold_index']]);ok=bool(prediction[index].argmax()==case['gold_index'])
            row=dict(correct=ok,gold_NLL=loss)
            if targets is not None:
                target=targets[key]['target_log_probs'][0]
                row['actual_mixed_KL']=float((target.exp()*(target-prediction[index])).sum());kl+=row['actual_mixed_KL']
            values[key]=row;correct+=int(ok);nll+=loss
    result=dict(count=len(cases),M_correct=correct,M_accuracy=correct/len(cases),M_gold_NLL=nll/len(cases),M_cases=values)
    if targets is not None:result['actual_mixed_KL']=kl/len(cases)
    if baseline is not None:
        b.require(values.keys()==baseline['M_cases'].keys(),'Fixed diagnostic identity changed.')
        result.update(M_repairs=sum(not baseline['M_cases'][k]['correct'] and row['correct'] for k,row in values.items()),
            M_damages=sum(baseline['M_cases'][k]['correct'] and not row['correct'] for k,row in values.items()))
    return result


@b.torch.no_grad()
def evaluate(model,dataset,targets,heldout,baseline=None,limited=False):
    mode=model.training;model.eval();result={}
    for name in ['early','late','heldout']:
        if name=='heldout':cases=heldout
        else:
            groups=b.CONTRACT[name+'_groups']
            cases=[targets[key]['case'] for index in groups for key in dataset['groups'][index]['case_ids'][:32]]
        item={}
        for source in ['webinstruct','medmcqa']:
            selected=[case for case in cases if case['case_id'].startswith(source+'/')]
            if limited:selected=selected[:32]
            b.require(len(selected)==(32 if limited else 64),'Each source needs its fixed64-case diagnostic.')
            item[source]=gold(model,selected,None if name=='heldout' else targets,
                None if baseline is None else baseline[name][source])
        item['count']=sum(item[source]['count'] for source in ['webinstruct','medmcqa'])
        item['M_correct']=sum(item[source]['M_correct'] for source in ['webinstruct','medmcqa'])
        item['M_gold_NLL']=sum(item[source]['M_gold_NLL'] for source in ['webinstruct','medmcqa'])/2
        if name!='heldout':item['actual_M_weighted_objective']=.5*sum(item[source]['actual_mixed_KL'] for source in ['webinstruct','medmcqa'])/2
        result[name]=item
    result.update(limited_validation=limited,QA_diagnostic_presentations=0,
        scope='Each formal panel128=64Web+64Med; early/late actual once-mixed target KL; untrained official-train panel gold only')
    model.train(mode);return result
