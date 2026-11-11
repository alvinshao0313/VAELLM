"""CPU proof of exact inputs/targets, actual weighted KL consumer, and fresh QK64."""
import ast
from collections import Counter
import os
import time
from . import bindings as b,data,train,result,objective


def numerical(dataset,targets,weighted):
    maximum=0.;gradient_max=0.
    for index,group in enumerate(dataset['groups']):
        target=b.torch.cat([targets[key]['target_log_probs'] for key in group['case_ids'][:32]])
        # Real cached targets form a finite, unequal source diagnostic input. No model,
        # forward path, or training target is replaced by this local CPU math check.
        logits=target.flip(-1).detach().clone().requires_grad_(True)
        logp=b.torch.log_softmax(logits,-1)
        old,_=data.original.legacy.mmlu.weighted_loss(logp,target)
        equal,_=objective.knowledge_loss(logp,target,group)
        b.require(b.torch.equal(old,equal),'Equal-source objective must exactly match the old weighted_loss.')
        if index in (0,63):
            old_grad=b.torch.autograd.grad(old,logits,retain_graph=True)[0]
            new_grad=b.torch.autograd.grad(equal,logits,retain_graph=True)[0]
            b.require(b.torch.equal(old_grad,new_grad),'Equal-source gradient differs from old consumer.')
        for arm,current in weighted.items():
            actual=current['groups'][index];loss,details=objective.knowledge_loss(logp,target,actual)
            values=b.torch.nn.functional.kl_div(logp,target,log_target=True,reduction='none').sum(-1)
            manual=logp.new_zeros(())
            for source in ('webinstruct','medmcqa'):
                mask=b.torch.tensor([data.source_name(key)==source for key in group['case_ids'][:32]])
                manual=manual+values[mask].mean()*b.ARMS[arm]['source_weights'][source]
                b.require(details[source]['actual_total_weight']==b.ARMS[arm]['source_weights'][source],
                    'The actual loss consumer did not receive the source weights.')
            difference=abs(float((loss-manual).detach()));maximum=max(maximum,difference)
            b.require(b.torch.allclose(loss,manual,rtol=1e-6,atol=1e-7),'Consumer differs from source-mean objective.')
            if index in (0,63):
                actual_grad=b.torch.autograd.grad(loss,logits,retain_graph=True)[0]
                manual_grad=b.torch.autograd.grad(manual,logits,retain_graph=True)[0]
                gradient_max=max(gradient_max,float((actual_grad-manual_grad).abs().max()))
                b.require(b.torch.isfinite(actual_grad).all() and actual_grad.abs().sum()>0
                    and b.torch.allclose(actual_grad,manual_grad,rtol=1e-6,atol=1e-7),
                    'Actual weighted gradients differ from the independent source-mean formula.')
    return dict(groups_checked=128,real_cached_targets=4096,equal_source_old_objective_exact=True,
        equal_source_old_gradients_exact_groups=[0,63],actual_consumer_gradient_groups=[0,63],
        source_mean_formula_atol=1e-7,source_mean_formula_rtol=1e-6,max_objective_difference=maximum,
        max_gradient_difference=gradient_max,actual_case_weights_passed_to_KL=True,no_extra_batch_or_M_factor=True)


def main():
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU prepare must hide CUDA.')
    b.require(not (b.RUN/'plan.json').exists(),'Never overwrite an existing source-weight plan.')
    started=time.monotonic();b.RUN.mkdir(parents=True,exist_ok=True)
    for path in b.HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    dataset,targets,old_evidence=data.original.load();parent=b.source()
    equal,unchanged,equal_proof=data.rebuild(dataset,targets,old_evidence,dict(webinstruct=.25,medmcqa=.25,QA=.5))
    b.require(equal['groups']==dataset['groups'] and unchanged is targets,'Equal-source control group contract changed.')
    qa_keys=[case['case_id'] for case in dataset['training_cases'] if case['domain']!='mmlu']
    b.require(len(qa_keys)==128 and len(targets)==4224,'Original4224 targets required.')
    counts=Counter(key for group in dataset['groups'][:64] for key in group['case_ids'])
    for name in ('webinstruct','medmcqa'):
        found=[key for key in counts if key.startswith(name+'/')]
        b.require(len(found)==1024 and all(counts[key]==1 for key in found),'Each source1024 once required.')
    b.require(all(counts[key]==4 for key in qa_keys),'OriginalQA128 each four times required.')
    source=b.torch.load(b.SOURCE/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    reference_request=b.read(b.CONTROL_ROOT/'request.json');names=reference_request['exact_scope_parameter_names']
    expected={f'model.layers.{layer}.self_attn.{category}' for layer in range(36) for category in ('q_proj','k_proj')}
    actual=[name for name in source if '._parallel_stage_decoder.' in name and name.split('._parallel_stage_decoder.')[0] in expected]
    b.require(set(actual)==set(names) and len(actual)==432 and len(expected)==72,'Actual QK scope changed.')
    masters={name:b.torch.nn.Parameter(source[name].float().clone()) for name in names};optimizer=train.make_optimizer(masters)
    b.require(not optimizer.state and all(p.dtype==b.torch.float32 and b.torch.equal(p,source[n].float()) for n,p in masters.items()),
        'Fresh Adam/master source contract failed.')
    qk=b.read(b.QK_VALIDATION);control=b.read(b.CONTROL_ROOT/'result.json');control_run=b.read(b.CONTROL_ROOT/'run.json')
    b.require(qk['status']=='verified_exact_scope_real_two_export_fresh' and qk['full_frozen_complement_exact']
        and qk['fresh']['full_state_exact'] and qk['fresh']['persistent_fields']==3927 and qk['fresh']['BF16_forward_finite']
        and control_run['exit_code']==0 and control['checkpoint_id']==b.CONTROL_ID
        and reference_request['parent']==parent and reference_request['learning_contract']==b.previous.LEARNING_CONTRACT
        and reference_request['data_digest']==old_evidence['actual_input_target_records_sha256'],
        'Reuse only verified same-QK path and equal-source control.')
    weighted={};proofs={}
    for arm,spec in b.ARMS.items():
        current,current_targets,evidence=data.rebuild(dataset,targets,old_evidence,spec['source_weights'])
        b.require(current_targets is targets and current['training_cases'] is dataset['training_cases']
            and current['train_ids'] is dataset['train_ids']
            and all(current_targets[key]['case'] is targets[key]['case']
                and b.torch.equal(current_targets[key]['target_log_probs'],targets[key]['target_log_probs']) for key in targets)
            and evidence['input_order_without_weights_sha256']==equal_proof['input_order_without_weights_sha256']
            and evidence['QA_target_records_sha256']==equal_proof['QA_target_records_sha256'],
            'All actual inputs/order/4224 targets/QA must remain exact.')
        weighted[arm]=current;directory=b.RUN/arm;directory.mkdir(exist_ok=True)
        b.write(directory/'data_contract.json',evidence)
        proofs[arm]=dict(source_weights=spec['source_weights'],all4224_targets_exact=True,
            all_inputs_order_domain_QA_weights_exact=True,all128_groups_source_weight_sums_exact=True,
            first64_exposure=evidence['first64_exposure'],data_contract=b.binding(directory/'data_contract.json'),
            data_digest=evidence['actual_input_target_records_sha256'])
    math_proof=numerical(dataset,targets,weighted)
    paired_self=result.paired_samples(list(b.CONTROL_EIGHT.values()),list(b.CONTROL_EIGHT.values()))
    b.require(all(row['repaired']==row['damaged']==0 for row in paired_self['tasks'].values())
        and len(paired_self['subject_diagnostics'])==57,'Full8 self-pairing/subject diagnostic failed.')
    prepared=b.read(b.prior.DATA/'dataset_prepared.json');pins=b.pins();requests={}
    for arm,spec in b.ARMS.items():
        directory=b.RUN/arm;request=dict(schema='S4_QK_source_weights_fresh64_request_v1',arm=arm,gpu=spec['gpu'],teacher_alpha=.5,
            source_weights=spec['source_weights'],validation_only=False,groups=list(range(128)),stop_step=64,parent=parent,
            data_digest=proofs[arm]['data_digest'],data_contract=proofs[arm]['data_contract'],
            source_sha256=pins,contract=b.contract(arm),learning_contract=b.learning_contract(arm),heldout=prepared['heldout'],
            fixture=b.binding(b.prior.RUN/'fixture.pt'),exact_scope_parameter_names=names,
            parent_eight={key:b.binding(path) for key,path in b.PARENT_EIGHT.items()},
            control_eight={key:b.binding(path) for key,path in b.CONTROL_EIGHT.items()},
            control_result=b.binding(b.CONTROL_ROOT/'result.json'),reused_QK_validation=b.binding(b.QK_VALIDATION))
        b.write(directory/'request.json',request);requests[arm]=b.binding(directory/'request.json')
    proof=dict(status='CPU_QK_source_weights_fresh64_verified',parent=parent,source_sha256=pins,arms=proofs,requests=requests,
        all4224_targets_exact=True,knowledge4096_exact=True,QA128_targets_exact=True,numerical_consumer=math_proof,
        exact_scope_parameter_names=names,active_projections=72,active_leaves=432,active_elements=sum(p.numel() for p in masters.values()),
        initial_FP32_masters_exact_source=True,fresh_Adam_state_entries=len(optimizer.state),
        shared_input_order_without_weights_sha256=equal_proof['input_order_without_weights_sha256'],
        shared_QA_target_records_sha256=equal_proof['QA_target_records_sha256'],
        groups=list(range(64)),unique_M_per_source=1024,QA_unique=128,QA_presentations_per_case=4,
        full8_actual_sample_self_pair_verified=paired_self,reused_QK_validation=b.binding(b.QK_VALIDATION),
        original_data_contract=b.binding(b.prior.DATA/'data_contract.json'),control_result=b.binding(b.CONTROL_ROOT/'result.json'),
        formal_terminal=64,next_group_index=64,automatic128=False,LongBench=False,
        CUDA_initialized=b.torch.cuda.is_initialized(),seconds=time.monotonic()-started)
    b.require(not proof['CUDA_initialized'],'CPU prepare initialized CUDA.')
    b.write(b.RUN/'CPU_validation.json',proof)
    b.write(b.RUN/'plan.json',dict(root=str(b.RUN),requests=requests,source_sha256=pins,
        CPU_validation=b.binding(b.RUN/'CPU_validation.json'),reused_QK_validation=b.binding(b.QK_VALIDATION),automatic_retry=False))
    print({k:proof[k] for k in ('status','all4224_targets_exact','active_leaves','active_elements','seconds')},flush=True)


if __name__=='__main__':main()
