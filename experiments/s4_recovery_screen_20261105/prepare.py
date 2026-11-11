"""CPU source/data/scope contracts and the bounded requests; no GPU setup."""
import ast
from collections import Counter
from pathlib import Path
import os
import time
from . import bindings as b,core


def main():
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preparation must hide CUDA.')
    b.require(not (b.RUN/'plan.json').exists(),'Never overwrite a prepared scope trial.')
    started=time.monotonic();b.RUN.mkdir(parents=True,exist_ok=True)
    for path in b.HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    dataset,targets,evidence=core.load_data();parent=b.source();pins=b.pins()
    prepared=b.read(b.prior.DATA/'dataset_prepared.json')
    shapes=prepared['shapes'][:64]
    peaks=dict(M_peak=max(row['M_max_length'] for row in shapes),QA_peak=max(row['QA_max_length'] for row in shapes))
    b.require(max(prepared['shapes'][i]['M_max_length'] for i in (2,9,10))==peaks['M_peak']
        and max(prepared['shapes'][i]['QA_max_length'] for i in (2,9,10))==peaks['QA_peak'],
        'The declared validation groups must cover both actual first64 peaks.')
    prefix={k for group in dataset['groups'][:64] for k in group['case_ids'][:32]}
    for name,indices in [('early',b.CONTRACT['early_groups']),('late',b.CONTRACT['late_groups'])]:
        panel={k for i in indices for k in dataset['groups'][i]['case_ids'][:32]}
        b.require(len(panel)==128 and ((panel<=prefix) if name=='early' else not panel&prefix),'Panel exposure label mismatch.')
    source=b.torch.load(b.SOURCE/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    names={arm:[name for name in source if '._parallel_stage_decoder.' in name
        and any(name.startswith(f'model.layers.{layer}.{block}.{category}.')
            for layer in range(36) for block,categories in (('self_attn',b.CATEGORIES[:4]),('mlp',b.CATEGORIES[4:]))
            for category in categories if category in b.CATEGORIES[:spec['stage']])]
        for arm,spec in b.ARMS.items()}
    for arm,spec in b.ARMS.items():
        b.require(len(names[arm])==spec['leaves'],'CPU actual native decoder scope differs.')
    parent_eight={key:b.binding(path) for key,path in b.PARENT_EIGHT.items()}
    b.require(all(b.read(path)['checkpoint_id']==b.SOURCE_ID for path in b.PARENT_EIGHT.values()),'Reused parent eight UUID differs.')
    requests={}
    for arm,spec in b.ARMS.items():
        for validation in (False,True) if arm=='attention' else (False,):
            tag='validation' if validation else arm
            directory=b.RUN/tag;directory.mkdir(exist_ok=True)
            request=dict(schema='S4_same_parent_scope64_request_v1',arm=arm,gpu='7' if validation else spec['gpu'],
                validation_only=validation,groups=[2,9,10] if validation else list(range(128)),stop_step=3 if validation else 64,parent=parent,
                data_digest=evidence['actual_input_target_records_sha256'],data_contract=b.binding(b.prior.DATA/'data_contract.json'),
                source_sha256=pins,contract=b.CONTRACT,learning_contract=b.LEARNING_CONTRACT,
                heldout=prepared['heldout'],fixture=b.binding(b.prior.RUN/'fixture.pt'),
                parent_eight=parent_eight)
            request['required_shapes']=peaks
            b.write(directory/'request.json',request);requests[tag]=b.binding(directory/'request.json')
    proof=dict(status='CPU_scope64_contracts_verified',parent=parent,source_sha256=pins,requests=requests,
        original_data_contract=b.binding(b.prior.DATA/'data_contract.json'),parent_eight=parent_eight,
        unique_M_per_source=1024,QA_unique=128,QA_presentations_per_case=4,groups=list(range(64)),
        actual_native_scope_leaves={key:len(value) for key,value in names.items()},
        actual_scope_parameters={key:sum(source[n].numel() for n in value) for key,value in names.items()},
        **peaks,all252_S4=True,late_panel_untrained_by64=True,
        validation_groups=[2,9,10],validation_split='2 updates save; new process restore + one update',
        formal_terminal=64,next_group_index=64,automatic128=False,LongBench=False,
        CUDA_initialized=b.torch.cuda.is_initialized(),seconds=time.monotonic()-started)
    b.require(not proof['CUDA_initialized'],'CPU preparation initialized CUDA.')
    b.write(b.RUN/'CPU_validation.json',proof)
    b.write(b.RUN/'plan.json',dict(root=str(b.RUN),requests=requests,source_sha256=pins,contract=b.CONTRACT,
        CPU_validation=b.binding(b.RUN/'CPU_validation.json'),formal_dependency=str(b.prior.RUN/'webmed/run.json'),automatic_retry=False))
    print(dict(status=proof['status'],leaves=proof['actual_native_scope_leaves'],parameters=proof['actual_scope_parameters'],
        source_pins=len(pins),seconds=proof['seconds']),flush=True)


if __name__=='__main__':main()
