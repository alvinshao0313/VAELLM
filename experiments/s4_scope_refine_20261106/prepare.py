"""Bind exact QK/VO scopes and reuse the original data and resume evidence."""
import ast
import os
import time
from . import bindings as b,core


def main():
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preparation must hide CUDA.')
    b.require(not (b.RUN/'plan.json').exists(),'Never overwrite this prepared trial.')
    started=time.monotonic();b.RUN.mkdir(parents=True,exist_ok=True)
    for path in b.HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    dataset,targets,evidence=core.load_data();parent=b.source();pins=b.pins()
    prepared=b.read(b.prior.DATA/'dataset_prepared.json');shapes=prepared['shapes'][:64]
    peaks=dict(M_peak=max(row['M_max_length'] for row in shapes),QA_peak=max(row['QA_max_length'] for row in shapes))
    b.require(shapes[2]['QA_max_length']==peaks['QA_peak'] and shapes[9]['M_max_length']==peaks['M_peak'],
        'Validation groups2/9 must cover actual first64 peaks.')
    source=b.torch.load(b.SOURCE/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    names={}
    for arm,spec in b.ARMS.items():
        projections={f'model.layers.{layer}.self_attn.{category}' for layer in range(36) for category in spec['categories']}
        names[arm]=[name for name in source if name.split('._parallel_stage_decoder.')[0] in projections
            and '._parallel_stage_decoder.' in name]
        b.require(len(names[arm])==spec['leaves'] and len({n.split('._parallel_stage_decoder.')[0] for n in names[arm]})==72,
            'Actual exact QK/VO decoder inventory differs.')
    b.require(not set(names['qk'])&set(names['vo']),'QK and VO must be disjoint.')
    original=b.read(b.REFERENCE_VALIDATION)
    b.require(original['status']=='verified_attention_scope_cross_process_2plus1' and original['resume']['masters_exact']
        and original['resume']['complete_Adam_exact'] and original['resume']['RNG_exact'],
        'Original real2+1 exact-resume evidence required.')
    parent_eight={key:b.binding(path) for key,path in b.PARENT_EIGHT.items()}
    b.require(all(b.read(path)['checkpoint_id']==b.SOURCE_ID for path in b.PARENT_EIGHT.values()),'Parent eight UUID differs.')
    requests={}
    for arm,spec in b.ARMS.items():
        for validation in (False,True):
            tag='validation_'+arm if validation else arm
            directory=b.RUN/tag;directory.mkdir(exist_ok=True)
            request=dict(schema='S4_same_parent_QK_VO64_request_v1',arm=arm,gpu='6' if validation else spec['gpu'],
                validation_only=validation,groups=[2,9] if validation else list(range(128)),stop_step=2 if validation else 64,parent=parent,
                data_digest=evidence['actual_input_target_records_sha256'],data_contract=b.binding(b.prior.DATA/'data_contract.json'),
                source_sha256=pins,contract=b.CONTRACT,learning_contract=b.LEARNING_CONTRACT,heldout=prepared['heldout'],
                fixture=b.binding(b.prior.RUN/'fixture.pt'),parent_eight=parent_eight,required_shapes=peaks,
                exact_scope_parameter_names=names[arm],reference_resume_validation=b.binding(b.REFERENCE_VALIDATION))
            b.write(directory/'request.json',request);requests[tag]=b.binding(directory/'request.json')
    proof=dict(status='CPU_exact_QK_VO64_contracts_verified',parent=parent,source_sha256=pins,requests=requests,
        original_data_contract=b.binding(b.prior.DATA/'data_contract.json'),parent_eight=parent_eight,
        unique_M_per_source=1024,QA_unique=128,QA_presentations_per_case=4,groups=list(range(64)),
        actual_native_scope_leaves={key:len(value) for key,value in names.items()},
        actual_scope_parameters={key:sum(source[n].numel() for n in value) for key,value in names.items()},
        exact_scope_parameter_names=names,**peaks,all252_S4=True,validation_groups=[2,9],
        reference_resume_validation=b.binding(b.REFERENCE_VALIDATION),formal_terminal=64,next_group_index=64,
        automatic128=False,LongBench=False,CUDA_initialized=b.torch.cuda.is_initialized(),seconds=time.monotonic()-started)
    b.require(not proof['CUDA_initialized'],'CPU preparation initialized CUDA.')
    b.write(b.RUN/'CPU_validation.json',proof)
    b.write(b.RUN/'plan.json',dict(root=str(b.RUN),requests=requests,source_sha256=pins,contract=b.CONTRACT,
        CPU_validation=b.binding(b.RUN/'CPU_validation.json'),reference_resume_validation=b.binding(b.REFERENCE_VALIDATION),automatic_retry=False))
    print({key:proof[key] for key in ('status','actual_native_scope_leaves','actual_scope_parameters','seconds')},flush=True)


if __name__=='__main__':main()
