"""Bind unchanged full8, canonical archive and actual four-card task partitions."""
import ast
import math
import os
from pathlib import Path
import time
from . import bindings as b
from experiments.longbench_v1.models import make_config,check_config
from experiments.longbench_v1.common import file_stat
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.protocol import DATASETS,EXPECTED_COUNTS,model_policy


def main():
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not b.RUN.exists(),'Fresh CPU-only LB preparation required.')
    started=time.monotonic()
    for path in b.HERE.glob('*.py'):ast.parse(path.read_text())
    completed=b.read(b.prior.RUN/'completion_verification.json');result=b.read(b.SOURCE_ROOT/'result.json')
    run=b.read(b.SOURCE_ROOT/'run.json');fresh=b.read(b.SOURCE_ROOT/'post/evaluation/fresh.json')
    b.require(completed['status']=='verified_complete_three_native_merges_and_full_eight'
        and run['exit_code']==0 and result['checkpoint_id']==fresh['checkpoint_id']==b.SOURCE_ID
        and result['M_correct']==10146 and abs(result['eight_mean_percent']-70.80321507638564)<1e-12
        and fresh['full_state_exact'] and fresh['BF16_forward_finite'],'Verified selected merge/full8 required.')
    reference_files=list((b.REFERENCE/'groups/part_01/runs').glob('*.json'))
    b.require(len(reference_files)==1,'One mature fullS4 LB evidence required.')
    reference=b.read(reference_files[0])
    b.require(reference['status']=='partition_complete' and reference['exit_code']==0 and reference['new_predictions']==4750
        and reference['load_audit']['model_policy']==model_policy(b.HF)
        and abs(reference['load_audit']['bit_budget']['user_linear_bpw']-b.BPW)<1e-12,
        'Reuse only the actual same-topology fullS4 LongBench evidence.')
    checkpoint=b.event(b.SOURCE);config=make_config(b.parent.MANIFEST,kind='dense',hf_path=b.HF,mode='full')
    config['model'].update(kind='compressed',checkpoint=str(b.SOURCE),checkpoint_id=b.SOURCE_ID,
        metadata_sha256=checkpoint['metadata_sha256'],weight_stat=file_stat(b.SOURCE/'pytorch_model.bin'))
    config['research_parent']=dict(stage_count=4,unprotected_main_bpw=4,model_merge=True,ratio_late=.75,
        final_lowbit_acceptance_evaluated=False)
    pins=b.pins();config['source_sha256'].update(pins);check_config(config)
    manifest,rows=load_archive(config['archive']['path'])
    b.require(manifest['data']['counts']==EXPECTED_COUNTS and len(rows)==4750
        and all(row['input_tokens']+row['max_new_tokens']<=32768 for row in rows),'Canonical full21 geometry changed.')
    groups=b.partition_api().task_groups(b.GPUS)
    b.require(len(groups)==4 and set(t for group in groups for t in group)==set(DATASETS),'Four-part coverage changed.')
    eight={k:b.binding(b.SOURCE_ROOT/'post/evaluation'/(k+'.json')) for k in ('mmlu','seven')}
    for record in eight.values():b.require(b.read(record['path'])['checkpoint_id']==b.SOURCE_ID,'Reused eight UUID differs.')
    request=dict(schema='S4_merge75_complete21_request_v1',contract=b.CONTRACT,checkpoint=checkpoint,
        source_sha256=pins,eight=eight,merge_result=b.binding(b.SOURCE_ROOT/'result.json'),
        source_fresh=b.binding(b.SOURCE_ROOT/'post/evaluation/fresh.json'),
        reference_generation_evidence=b.binding(reference_files[0]),GPUs=b.GPUS,task_groups=groups,
        config_sha256=None)
    b.RUN.mkdir(parents=True);config_path=b.RUN/'config.json';b.write(config_path,config)
    request['config_sha256']=b.sha(config_path);b.write(b.RUN/'request.json',request)
    units=[]
    for index,(gpu,tasks) in enumerate(zip(b.GPUS,groups,strict=True)):
        output=b.RUN/'groups'/f'part_{index+1:02d}'
        argv=[str(b.PYTHON),'-B','-u','-m','experiments.s4_merge75_longbench_20261108.predict',
            '--request',str(b.RUN/'request.json'),'--config',str(config_path),'--output',str(output),'--datasets',*tasks]
        units.append(dict(output_dir=str(output),datasets=tasks,expected_samples=sum(EXPECTED_COUNTS[t] for t in tasks),
            native_reports=[],outer_exit=None,launch_records=[],terminal_attempts=[],gpu=gpu,argv=argv,
            native_runtime_seconds=86400,outer_timeout_seconds=86580,termination_grace_seconds=20))
    merged=b.RUN/'merged'
    plan=dict(schema='fixed_full21_merge_plan_v1',status='draft_awaiting_explicit_terminal_binding',checkpoint_id=b.SOURCE_ID,
        expected_tasks=list(DATASETS),expected_samples=4750,config_path=str(config_path),config_sha256=b.sha(config_path),
        protocol=config['protocol'],execution_units=units,
        task_file_sources={t:str(Path(u['output_dir'])/'predictions'/(t+'.jsonl')) for u in units for t in u['datasets']},
        pins={**pins,str(config_path):b.sha(config_path),str(b.RUN/'request.json'):b.sha(b.RUN/'request.json')},
        merged_output=str(merged),scoring_cwd=str(b.ROOT),python_executable=str(b.PYTHON),
        nine_task_argv=[str(b.PYTHON),'-B','-u','-m','experiments.s4_merge75_longbench_20261108.result',
            '--request',str(b.RUN/'request.json'),'--output',str(merged/'nine_task_acceptance.json')],
        automatic_retry=False,automatic_resume=False,partial_scoring=False)
    b.write(b.RUN/'merge_plan.json',plan)
    b.require(not b.torch.cuda.is_initialized(),'CPU preparation initialized CUDA.')
    b.write(b.RUN/'CPU_validation.json',dict(status='verified_same_merge_eight_and_canonical_full21',
        checkpoint_id=b.SOURCE_ID,reused_eight=eight,reference_fullS4_generation=b.binding(reference_files[0]),
        partitions=[dict(gpu=u['gpu'],datasets=u['datasets'],samples=u['expected_samples']) for u in units],
        complete21_samples=4750,original32K_geometry=True,CUDA_initialized=False,seconds=time.monotonic()-started))
    print(dict(status='CPU_full21_prepared',run=str(b.RUN)),flush=True)


if __name__=='__main__':main()
