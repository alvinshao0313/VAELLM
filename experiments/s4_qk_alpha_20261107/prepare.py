"""CPU proof for once-mixed targets, unchanged QK/inputs/QA, and two fresh64 requests."""
import ast
import os
import time
from collections import Counter
from . import bindings as b,data,train,result


def main():
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU prepare must hide CUDA.')
    b.require(not (b.RUN/'plan.json').exists(),'Never overwrite an existing alpha plan.')
    started=time.monotonic();b.RUN.mkdir(parents=True,exist_ok=True)
    for path in b.HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    dataset,old,old_evidence,raw=data.base();parent=b.source()
    half,half_proof=data.rebuild(dataset,old,old_evidence,raw,.5)
    b.require(all(b.torch.equal(half[key]['target_log_probs'],old[key]['target_log_probs']) for key in old),
        'Alpha.5 must exactly reconstruct every original FP32 target without normalization changes.')
    qa_keys=[case['case_id'] for case in dataset['training_cases'] if case['domain']!='mmlu']
    b.require(len(qa_keys)==128,'Original QA128 required.')
    counts=Counter(key for group in dataset['groups'][:64] for key in group['case_ids'])
    for name in ('webinstruct','medmcqa'):
        found=[key for key in counts if key.startswith(name+'/')]
        b.require(len(found)==1024 and all(counts[key]==1 for key in found),'Each original source1024 once required.')
    b.require(all(counts[key]==4 for key in qa_keys),'OriginalQA128 must appear exactly four times.')
    source=b.torch.load(b.SOURCE/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    reference_request=b.read(b.CONTROL_ROOT/'request.json');names=reference_request['exact_scope_parameter_names']
    expected={f'model.layers.{layer}.self_attn.{category}' for layer in range(36) for category in ('q_proj','k_proj')}
    actual=[name for name in source if '._parallel_stage_decoder.' in name and name.split('._parallel_stage_decoder.')[0] in expected]
    b.require(set(actual)==set(names) and len(actual)==432 and len(expected)==72,'Actual QK scope changed from alpha.5 control.')
    masters={name:b.torch.nn.Parameter(source[name].float().clone()) for name in names};optimizer=train.make_optimizer(masters)
    b.require(not optimizer.state and all(p.dtype==b.torch.float32 and b.torch.equal(p,source[n].float()) for n,p in masters.items()),
        'Fresh Adam/master source contract failed.')
    qk=b.read(b.QK_VALIDATION);control=b.read(b.CONTROL_ROOT/'result.json');control_run=b.read(b.CONTROL_ROOT/'run.json')
    b.require(qk['status']=='verified_exact_scope_real_two_export_fresh' and qk['full_frozen_complement_exact']
        and qk['fresh']['full_state_exact'] and qk['fresh']['persistent_fields']==3927 and qk['fresh']['BF16_forward_finite']
        and control_run['exit_code']==0 and control['checkpoint_id']==b.CONTROL_ID
        and reference_request['parent']==parent and reference_request['learning_contract']==b.previous.LEARNING_CONTRACT
        and reference_request['data_digest']==old_evidence['actual_input_target_records_sha256'],
        'Reuse only the verified same-QK path and same-source alpha.5 control.')
    paired_self=result.paired_samples(list(b.CONTROL_EIGHT.values()),list(b.CONTROL_EIGHT.values()))
    b.require(all(row['repaired']==row['damaged']==0 for row in paired_self['tasks'].values()),'Actual full8 self-pairing failed.')
    prepared=b.read(b.prior.DATA/'dataset_prepared.json');pins=b.pins();requests={};proofs={}
    for arm,spec in b.ARMS.items():
        targets,evidence=data.rebuild(dataset,old,old_evidence,raw,spec['alpha'])
        b.require(evidence['input_order_weights_sha256']==half_proof['input_order_weights_sha256']
            and evidence['QA_target_records_sha256']==half_proof['QA_target_records_sha256']
            and evidence['first64_exposure']==half_proof['first64_exposure']
            and all(targets[key]['target_log_probs'] is old[key]['target_log_probs'] for key in qa_keys),
            'Alpha must leave all actual inputs/order/weights/QA/exposure exact.')
        directory=b.RUN/arm;directory.mkdir(exist_ok=True);b.write(directory/'data_contract.json',evidence)
        request=dict(schema='S4_QK_alpha_fresh64_request_v1',arm=arm,gpu=spec['gpu'],teacher_alpha=spec['alpha'],
            validation_only=False,groups=list(range(128)),stop_step=64,parent=parent,
            data_digest=evidence['actual_input_target_records_sha256'],data_contract=b.binding(directory/'data_contract.json'),
            source_sha256=pins,contract=b.contract(arm),learning_contract=b.learning_contract(arm),heldout=prepared['heldout'],
            fixture=b.binding(b.prior.RUN/'fixture.pt'),exact_scope_parameter_names=names,
            parent_eight={key:b.binding(path) for key,path in b.PARENT_EIGHT.items()},
            control_eight={key:b.binding(path) for key,path in b.CONTROL_EIGHT.items()},
            control_result=b.binding(b.CONTROL_ROOT/'result.json'),reused_QK_validation=b.binding(b.QK_VALIDATION))
        b.write(directory/'request.json',request);requests[arm]=b.binding(directory/'request.json')
        proofs[arm]=dict(teacher_alpha=spec['alpha'],data_contract=request['data_contract'],data_digest=request['data_digest'],
            all_QA_targets_inputs_order_domain_weights_exact=True,formula_probability_atol=1e-7,normalization_atol=2e-6,
            finite_detached_FP32=True,no_clamp_or_renormalization=True)
    proof=dict(status='CPU_QK_alpha25_75_fresh64_verified',parent=parent,source_sha256=pins,arms=proofs,requests=requests,
        alpha_half_rebuilt_all4224_exact=True,alpha_half_knowledge4096_exact=True,QA128_targets_exact=True,
        exact_scope_parameter_names=names,active_projections=72,active_leaves=432,active_elements=sum(p.numel() for p in masters.values()),
        initial_FP32_masters_exact_source=True,fresh_Adam_state_entries=len(optimizer.state),
        shared_input_order_weights_sha256=half_proof['input_order_weights_sha256'],shared_QA_target_records_sha256=half_proof['QA_target_records_sha256'],
        groups=list(range(64)),unique_M_per_source=1024,QA_unique=128,QA_presentations_per_case=4,
        full8_actual_sample_self_pair_verified=paired_self,reused_QK_validation=b.binding(b.QK_VALIDATION),
        original_data_contract=b.binding(b.prior.DATA/'data_contract.json'),control_result=b.binding(b.CONTROL_ROOT/'result.json'),
        formal_terminal=64,next_group_index=64,automatic128=False,LongBench=False,
        CUDA_initialized=b.torch.cuda.is_initialized(),seconds=time.monotonic()-started)
    b.require(not proof['CUDA_initialized'],'CPU prepare initialized CUDA.')
    b.write(b.RUN/'CPU_validation.json',proof)
    b.write(b.RUN/'plan.json',dict(root=str(b.RUN),requests=requests,source_sha256=pins,
        CPU_validation=b.binding(b.RUN/'CPU_validation.json'),reused_QK_validation=b.binding(b.QK_VALIDATION),automatic_retry=False))
    print({k:proof[k] for k in ('status','alpha_half_rebuilt_all4224_exact','active_leaves','active_elements','seconds')},flush=True)


if __name__=='__main__':main()
