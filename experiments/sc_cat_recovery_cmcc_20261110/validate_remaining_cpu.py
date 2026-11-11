"""New continuation decisions plus real scoped/final native CPU train/export paths."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import functools
import os
from pathlib import Path
import tempfile
import time

import torch
from transformers import BertTokenizerFast
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from train_utils import checkpoint_v6 as v6
from experiments.cat_auto_protocol_cmcc_20261104 import data as data_module
from experiments.cat_auto_protocol_cmcc_20261104.live import architecture, fresh_native
from experiments.cat_auto_protocol_cmcc_20261104.nodes import NodeScore
from experiments.cat_auto_protocol_cmcc_20261104.validate_recovery_cpu import score
from experiments.cat_auto_protocol_cmcc_20261104.validate_native_builder_cpu import fixture_sources
from . import io, search
from .boundary import export_selected_node
from .builder import make_native_recovery_trainer, RecoveryExecutionPolicy
from .identity import source_pins
from .mutable import capture_mutable
from .processes import Budget
from .remaining import owners
from .selection import TrajectoryResult, SelectedCategory, terminal_endpoints, long_check
from .settings import RecoverySettings, normalize, trajectory_id
from .validate_cpu import protected_root, rule_checks


def decisions(root):
    rules=rule_checks();settings=RecoverySettings()
    inv=dict(prefix=('q','k'),current=('k',),protected=('q','k'),future=('v',))
    state=dict(current_category='k',approved_categories=['k'],categories={'k':dict(default_winner_raw='raw',rounds=[])},
        short_screen={'combination':dict(status='passed')})
    def row(identifier, config, values, *, status='complete', candidate='a', raw='raw', raw_value=.4):
        return TrajectoryResult('k','parent',raw,'recipe_'+candidate,candidate,identifier,100,100,config,score(raw_value,raw),
            tuple(NodeScore(s,score(v,identifier+str(s)),identifier+str(s)) for s,v in zip((125,250,375,500),values)),status)
    identifier=trajectory_id('raw',settings,'data','code');completed={identifier:row(identifier,settings,(.5,)*4)}
    planned=search.begin_round(state,category='k',parameter='combination',base=settings,raw_id='raw',inventory=inv,
        data_identity='data',code_identity='code',completed=completed)
    for index,candidate in enumerate(planned['candidates']):
        if candidate['reuse']:continue
        assert candidate['first_stop']==250
        config=RecoverySettings(**candidate['settings']);value=.8-index*.03
        completed[candidate['trajectory_id']]=row(candidate['trajectory_id'],config,(value,value),status='pending250')
    decision=search.finish_round(state,'k',completed);assert len(decision['continue_to500'])==2
    for key,value in list(completed.items()):
        if key in decision['continue_to500']:
            completed[key]=row(key,value.settings,(float(value.nodes[0].score.mean),)*4)
        elif value.status=='pending250':completed[key]=replace(value,status='screened250')
    assert isinstance(search.finish_round(state,'k',completed),RecoverySettings)
    assert sum(v.status=='screened250' for v in completed.values())==4
    assert planned['status']=='complete'
    # B raw beats B recovery but cannot displace B's terminal endpoint when A
    # owns the sole globally qualifying raw. Both actual recovery endpoints matter.
    a=row('A',settings,(.7,)*4,candidate='A',raw='rawA',raw_value=.95)
    b=row('B',settings,(.6,)*4,candidate='B',raw='rawB',raw_value=.9)
    endpoints=terminal_endpoints([a,b]);assert [c['selected_step'] for _,c in endpoints]==[0,500]
    assert long_check([row('T',settings,(.6,.65,.7,.8))],'recipe_a',['one','two']) is None
    authorization=dict(gpus=[4,5,6,7],budget_by_component=dict(technical=1,training=1),total_GPU_hours=2,maximum_trajectories=1)
    prior=Budget(root/'old_budget.json',authorization,[],[])
    prior.reserve({'done':dict(component='training',seconds=30)},counts=dict(trajectories=1))
    request=root/'request.json';io.write(request,{'fixture':True})
    prior.start('done',gpu=4,start=10,deadline=40,command=['CPU_budget_fixture'],log=root/'unused.log',request=request)
    prior.finish('done',end=20,exit_code=1,error='explicit fixture failed task')
    next_auth=dict(authorization,batch='remaining_and_replay',budget_by_component=dict(technical=1,training=2),total_GPU_hours=3,maximum_trajectories=2)
    inherited=Budget(root/'new_budget.json',next_auth,[],[],previous=prior.value,retry_tasks=(),new_batch=True)
    assert inherited.value['tasks']['done']['status']=='failed' and inherited.value['total_actual_GPU_hours']==10/3600
    assert inherited.value['counter_reserved']['trajectories']==1
    return dict(previous_rules=rules,actual_short_screen_decision=True,global_raw0_terminal_eligibility=True,
        no_anchor_tie_priority=True,fees_failed_windows_and_counts_inherited=True)


def full_raw(root, raw_dir, inventory):
    model,meta,_=v6.load_v6_full_checkpoint_into_model(architecture(raw_dir,attn_implementation='sdpa'),str(raw_dir),strict=True)
    for projection in inventory['projections']:
        name=projection['actual_name']
        if name in meta['compressed_targets']:continue
        out_features,in_features=projection['shape'];remaining=in_features-8
        decoders=[Decoder(in_dim=8,out_dim=8,hidden_dim=8,num_res_blocks=0,norm_type='layer',decoder_type='asymmetric',
            use_checkpoint=False,num_models=1).to(dtype=torch.bfloat16).extract_single(0) for _ in range(2)]
        codes=[torch.randint(2,(out_features*remaining//8,1,8),dtype=torch.bool) for _ in range(2)]
        layer=VAELinear(in_features=in_features,out_features=out_features,compressed_in_features=remaining,bias=None,original_weight=None,
            vq_weight=codes[0],decoder=decoders[0],codebook_dim=8,transpose=False,stage_vq_weights=codes,stage_decoders=decoders,
            stage_codebook_dims=[8,8],protected_input_indices=torch.arange(8),protected_input_qvalues=torch.randint(1,100,(8,out_features),dtype=torch.uint8),
            protected_input_scales=torch.full((8,),.01,dtype=torch.bfloat16),protected_channel_quant_format='int8')
        layer.pack_parallel_stage_decoder_(trainable=False);parent,field=name.rsplit('.',1);setattr(model.get_submodule(parent),field,layer)
    directory=root/'last_raw';order=meta['compression_categories']
    v6.save_v6_full_checkpoint(model,str(directory),checkpoint_kind='round_base',compressed_targets=[p['actual_name'] for p in inventory['projections']],
        pending_dense_targets=[],completed_categories=order[:-1],compression_categories=order,target_layers=meta['target_layers'],target_modules=meta['target_modules'],
        cat_runtime_state=v6.load_v6_cat_runtime_state(str(raw_dir),required=True),base_model_path=meta['base_model_path'],
        extra_meta=dict(cat_auto_phase='cpu_fixture',distill_stage_history=[dict(category=c,did_train=False) for c in order[:-1]]))
    return directory,order[-1]


def actual(root):
    vocab=root/'vocab.txt';vocab.write_text('\n'.join(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','hello','world','tiny','train','answer']+[f'unused{i}' for i in range(10,128)])+'\n')
    tokenizer=BertTokenizerFast(vocab_file=str(vocab),do_lower_case=False);tokenizer.eos_token=tokenizer.sep_token
    inventory,raw_dir=protected_root(root,tokenizer);last_dir,last=full_raw(root,raw_dir,inventory)
    source=fixture_sources(root)
    prepared=data_module.freeze_data_plan(source,source[0],tokenizer,model_identity=inventory['model_identity'],category_ids=['k_proj',last],search_seed=31)
    policy=RecoveryExecutionPolicy('cpu','sdpa','none','none',teacher_output_pin_memory=False)
    inv=owners(dict(round_base_dir=str(raw_dir),current_category='k_proj'),inventory)
    assert len(inv['prefix'])==len(inv['protected'])==4 and len(inv['current'])==2
    final_inv=owners(dict(round_base_dir=str(last_dir),current_category=last),inventory)
    assert len(final_inv['prefix'])==14 and not final_inv['future']
    assert normalize(RecoverySettings(),**final_inv).groups=='D'
    configs=[('currentD_prefixP','k_proj',raw_dir,RecoverySettings(groups='DPL',decoder_scope='current',protection_scope='prefix',bound_optimizer='muon',lora_optimizer='muon'),2),
        ('prefixD_currentP','k_proj',raw_dir,RecoverySettings(groups='DPL',decoder_scope='prefix',protection_scope='current',bound_optimizer='muon',lora_optimizer='muon'),2),
        ('final_replay125',last,last_dir,RecoverySettings(groups='DP',bound_optimizer='muon'),125)]
    proofs=[]
    for label,category,directory,settings,steps in configs:
        options=dict(technical_stop_after=2) if steps==2 else dict(replay_selected_step=125)
        run=make_native_recovery_trainer(round_base_dir=directory,inventory=inventory,category_id=category,settings=settings,prepared_data=prepared,
            sources=source,tokenizer=tokenizer,execution_policy=policy,phase='cpu_fixture',output_dir=root/label,**options)
        try:
            result=run.train();values,_,_=capture_mutable(run.selection)
            assert result['actual_steps']==steps and result['schedule_max']==500 and all(torch.isfinite(t).all() for t in values.values())
            current=[p['actual_name'] for p in inventory['projections'] if p['category_id']==category]
            actual_d={n for n,_ in run.selection.decoder_targets};actual_p=set(run.selection.protection.modules)
            wanted=owners(dict(round_base_dir=str(directory),current_category=category),inventory)
            assert actual_d==set(wanted[settings.decoder_scope]) and actual_p==set(wanted[settings.protection_scope])
            if category==last:assert not run.selection.dense_targets and all(g['group_name']!='lora' for g in run.optimizer.param_groups)
            node=Path(run.trainer.args.output_dir)/f'checkpoint-{steps}';meta=v6.load_v6_training_step_meta(str(node))
            selected=SelectedCategory(category,'CPU_recipe','CPU_candidate',settings,steps,label,meta['checkpoint_id'],
                run.provenance['round_base_id'],run.provenance['semantic_seeds'],'cpu_fixture')
        finally:run.close()
        report=export_selected_node(node_directory=node,destination=root/(label+'_export'),selected=selected,current_targets=current,
            phase='cpu_fixture',model_factory=functools.partial(architecture,attn_implementation='sdpa'))
        fresh,proof=fresh_native(root/(label+'_export'),model_factory=functools.partial(architecture,attn_implementation='sdpa'))
        with torch.inference_mode():assert torch.isfinite(fresh(input_ids=prepared['calibration'][0],use_cache=False).logits).all()
        assert proof['state_exact'] and report['export_evidence']['actual_updates']==steps
        proofs.append(dict(case=label,actual_updates=steps,schedule_max=500,decoder_owners=len(actual_d),protection_owners=len(actual_p),export_and_fresh=True))
    return proofs


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(2);output=Path(args.output);assert not output.exists();output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='remaining_CPU_',dir=output.parent) as directory:
        root=Path(directory);rules=decisions(root);print('decision/budget checks passed',flush=True);proofs=actual(root)
    io.write(output,dict(status='passed',rules=rules,actual=proofs,actual_CPU_updates=129,source_pins=source_pins(),
        GPU_executed=False,formal_quality_verified=False,elapsed_seconds=time.monotonic()-started))
    print('New scoped/final native CPU checks passed',flush=True)


if __name__=='__main__':main()
