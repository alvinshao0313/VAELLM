"""CPU bind real states, unchanged recipes, actual source weights and exposure."""
import ast
from collections import Counter
from copy import deepcopy
import os
import time
from . import bindings as b,core,state


def main():
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Hide CUDA for preparation.')
    b.require(not b.RUN.exists(),'Never overwrite a continuation attempt.')
    began=time.monotonic()
    for path in b.HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    validation=b.read(b.weighted.QK_VALIDATION)
    b.require(validation['status']=='verified_exact_scope_real_two_export_fresh'
        and validation['full_frozen_complement_exact'] and validation['independent_export_named_state_exact']
        and validation['fresh']['full_state_exact'] and validation['fresh']['BF16_forward_finite'],
        'Existing QK GPU proof is required.')
    resume_proof=b.read(b.equal.REFERENCE_VALIDATION)
    b.require(resume_proof['status']=='verified_attention_scope_cross_process_2plus1',
        'Existing named-state GPU resume proof required.')
    prepared={};reference=None
    for arm,spec in b.ARMS.items():
        original=b.read(spec['predecessor']/'request.json');b.parent.s.check_pins(original['source_sha256'])
        result=b.read(spec['predecessor']/'result.json');training=b.read(spec['predecessor']/'training/training.json')
        b.require(result['checkpoint_id']==spec['checkpoint_id'] and training['exit_code']==0
            and training['completed_updates']==64,'Completed original64 required.')
        saved=b.torch.load(b.predecessor_state(arm),map_location='cpu',weights_only=False)
        b.require(saved['schema']=='S4_scope_named_training_state_v1' and saved['identity']==state.identity(original)
            and saved['completed_updates']==saved['next_cursor']==saved['next_group_index']==64,'State identity/cursor changed.')
        b.require(saved['parameter_names']==list(saved['masters']) and len(saved['masters'])==432
            and set(saved['parameter_names'])==set(original['exact_scope_parameter_names'])
            and all(t.dtype==b.torch.float32 and b.torch.isfinite(t).all() for t in saved['masters'].values()),
            'Actual432 ordered FP32 masters required.')
        b.require(len(saved['optimizer']['state'])==432 and all(int(item['step'])==64
            and item['exp_avg'].dtype==item['exp_avg_sq'].dtype==b.torch.float32
            and b.torch.isfinite(item['exp_avg']).all() and b.torch.isfinite(item['exp_avg_sq']).all()
            for item in saved['optimizer']['state'].values()),'Actual complete Adam step64 required.')
        b.require(set(saved['rng'])=={'python','numpy','CPU','CUDA'} and len(saved['rng']['CUDA'])==1,
            'Original single-GPU RNG payload required.')
        dataset,targets,evidence=core.load_data(arm)
        b.require(evidence['actual_input_target_records_sha256']==original['data_digest'],'Data contract changed.')
        sequence=[{k:v for k,v in group.items() if k!='weights'} for group in dataset['groups']]
        target_records={key:core.equal.data.legacy.core.tensor_record(item['target_log_probs']) for key,item in targets.items()}
        common=(sequence,target_records,saved['parameter_names'])
        if reference is None:reference=common
        else:b.require(common==reference,'Both arms must use identical inputs, targets, order and parameter order.')
        b.require(all(core.weighted.data.group_weights(group)==spec['source_weights'] for group in dataset['groups']),
            'Actual source loss weights changed.')
        counts=Counter(key for group in dataset['groups'] for key in group['case_ids'])
        for source in ('webinstruct','medmcqa'):
            keys=[key for key in counts if key.startswith(source+'/')]
            b.require(len(keys)==2048 and all(counts[key]==1 for key in keys),'Cumulative source exposure changed.')
        qa=[case['case_id'] for case in dataset['training_cases'] if case['domain']!='mmlu']
        b.require(len(qa)==128 and all(counts[key]==8 for key in qa),'QA128 each8 required.')
        labels={str(cursor):core.exposure_labels(dataset,cursor) for cursor in (64,128)}
        b.require(not labels['64']['late']['trained_by_cursor'] and labels['128']['late']['trained_by_cursor']
            and not labels['128']['heldout']['trained_by_cursor'],'Actual panel consumption labels differ.')
        request=deepcopy(original)
        request['continuation']=dict(contract=b.CONTRACT,gpu=spec['gpu'],source_sha256=b.pins(),
            source_weights=spec['source_weights'],original_request=b.binding(spec['predecessor']/'request.json'),
            predecessor_state=b.binding(b.predecessor_state(arm)),predecessor_result=b.binding(spec['predecessor']/'result.json'),
            QK_validation=b.binding(b.weighted.QK_VALIDATION),resume_validation=b.binding(b.equal.REFERENCE_VALIDATION),
            learning_operators='Unchanged original arm build/update/restore; no algorithm or precision change')
        b.check_request(request)
        prepared[arm]=(request,dict(original_identity_preserved=state.identity(request)==saved['identity'],
            masters=432,Adam_step=64,cursor=64,next_group=64,source_weights=spec['source_weights'],panel_labels=labels,
            new_exposure=core.exposure(dataset,targets,range(64,128)),
            cumulative_exposure=core.exposure(dataset,targets,range(128))))
    b.require(not b.torch.cuda.is_initialized(),'CPU preparation initialized CUDA.')
    b.RUN.mkdir(parents=True)
    for arm,(request,proof) in prepared.items():
        b.write(b.RUN/arm/'request.json',request)
        proof.update(request=b.binding(b.RUN/arm/'request.json'))
    b.write(b.RUN/'CPU_validation.json',dict(status='verified_original64_pair_continuation',
        arms={arm:proof for arm,(_,proof) in prepared.items()},same_inputs_targets_order_parameters=True,
        step65_lr=5e-7*(1-64/500),step128_lr=5e-7*(1-127/500),CUDA_initialized=False,
        seconds=time.monotonic()-began))
    print(dict(status='verified_original64_pair_continuation',run=str(b.RUN)),flush=True)


if __name__=='__main__':main()
