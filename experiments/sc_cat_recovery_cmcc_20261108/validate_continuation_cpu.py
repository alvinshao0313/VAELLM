"""Actual saved375 continuation plus settled budget/reuse/failure regression."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import torch
from transformers import BertTokenizerFast
from experiments.cat_auto_protocol_cmcc_20261104 import data
from experiments.cat_auto_protocol_cmcc_20261104.validate_native_builder_cpu import fixture_sources
from . import io
from .builder import make_native_recovery_trainer, RecoveryExecutionPolicy
from .continuation import reuse_completed
from .identity import source_pins
from .processes import Budget, Processes
from .settings import RecoverySettings
from .validate_cpu import protected_root


def checks(root):
    vocab=root/'vocab.txt'
    vocab.write_text('\n'.join(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','hello','world','tiny','train','answer']+[f'unused{i}' for i in range(10,128)])+'\n')
    tokenizer=BertTokenizerFast(vocab_file=str(vocab),do_lower_case=False)
    tokenizer.eos_token=tokenizer.sep_token
    inventory, raw=protected_root(root,tokenizer)
    sources=fixture_sources(root)
    prepared=data.freeze_data_plan(sources,sources[0],tokenizer,model_identity=inventory['model_identity'],category_ids=['k_proj'],search_seed=31)
    common=dict(round_base_dir=raw,inventory=inventory,category_id='k_proj',settings=RecoverySettings(),
        prepared_data=prepared,sources=sources,tokenizer=tokenizer,
        execution_policy=RecoveryExecutionPolicy('cpu','sdpa','none','none',teacher_output_pin_memory=False),phase='cpu_fixture')
    run=make_native_recovery_trainer(**common,output_dir=root/'original',stop_at=375)
    try:
        first=run.train()
        checkpoint=Path(run.trainer.args.output_dir)/'checkpoint-375'
        assert first['actual_steps']==375 and (checkpoint/'node_ready.json').is_file()
    finally:
        run.close()
    resumed=make_native_recovery_trainer(**common,output_dir=root/'continued',resume_from=checkpoint)
    try:
        result=resumed.train()
        assert result['saved_state_restore_verified'] and result['actual_steps']==500
        assert result['starting_step']==375 and result['actual_update_count']==125
        assert result['actual_data']['resumed_updates']==125
    finally:
        resumed.close()
    auth=dict(budget_by_component=dict(technical=4,training=10),total_GPU_hours=14,gpus=[4,5,6,7],
        maximum_trajectories=2,maximum_KD_updates=1000)
    b=Budget(root/'old_budget.json',auth,[dict(started_epoch=0,ended_epoch=60)],[])
    b.reserve({'a/train':dict(component='training',seconds=2700)},counts=dict(trajectories=1,KD_updates=500))
    request=root/'request.json';io.write(request,dict(operation='train',settings=asdict(RecoverySettings())))
    b.start('a/train',gpu=4,start=100,deadline=2800,command=['actual_cpu_fixture'],log=root/'log',request=request)
    b.finish('a/train',end=300,exit_code=-15,error='interrupted fixture')
    continued=Budget(root/'new_budget.json',auth,b.value['technical_windows'],[],previous=b.value)
    assert continued.value['total_actual_GPU_hours']==b.value['total_actual_GPU_hours']
    assert continued.value['counter_reserved']==b.value['counter_reserved']
    assert continued.value['tasks']['a/train']['status']=='reserved'
    result_path=root/'result.json';io.write(result_path,{'actual':True})
    entry=dict(request=io.binding(request),result=io.binding(result_path))
    assert reuse_completed(entry,io.read(request))==entry['result']
    try:
        reuse_completed(entry,dict(operation='train',settings={}))
    except ValueError:
        pass
    else:
        raise AssertionError('Changed settings were reused.')
    processes=Processes(root/'execution',continued,{})
    processes.record_failure('original',ValueError('disk space'))
    processes.record_failure('cancelled',ValueError('dependency failed'))
    assert io.read(root/'execution/first_failure.json')['task_id']=='original'
    return dict(native375_resume_to500=True,actual_KD_updates=500,unchanged_original_schedule=True,
        exact_optimizer_scheduler_RNG_masters=True,actual_resumed_input_hashes=True,
        inherited_actual_fees_once=True,first_failure_preserved=True,changed_reuse_inputs_rejected=True,
        GPU_verified=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    output=Path(args.output);assert not output.exists()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='sc_cat_resume375_') as temporary:
        result=checks(Path(temporary))
    io.write(output,dict(result,status='CPU_verified',source_pins=source_pins()))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
