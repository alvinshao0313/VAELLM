"""Original HF factory and native S4 fits with an explicit new prefix recovery hook."""
import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b,recovery,budget
from experiments.progressive_residual_bsq_20261030.recovery_compare_20261101 import runtime
from experiments.mlp_learnability_formal_cmcc_20261101.snapshot import rng,restore_rng,rng_exact


def execute(request_path):
    request=b.read(request_path);b.parent.s.check_pins(request['source_sha256'])
    b.initialize();resources=b.resources()
    b.require(b.parent.s.hf_identity(b.HF)==request['HF_identity'],'Original HF identity changed.')
    root=Path(request['output']);short=request['validation_only']
    cat,hf,training,vae,cfgs,identity,normalized,argv=b.native_recipe(root/'native')
    b.require(not cat.rot_llm and cat.after_category_mode=='none' and vae.model_path==str(b.HF),
        'Native factory is original HF; experiment owns explicit prefix recovery dispatch.')
    from train_utils import cat_train_pipeline as native
    import transformers
    b.require(Path(native.__file__).resolve().is_relative_to(b.parent.s.SNAPSHOT),'Use the pinned native mathematical implementation.')
    fitting_rng=rng()
    dataset,targets,evidence=b.data.load()
    restore_rng(fitting_rng)
    b.require(evidence['actual_input_target_records_sha256']==request['data_digest'],'Actual static target digest changed.')
    request['_dataset'],request['_targets']=dataset,targets
    request['recovery_history']=[]
    record=dict(status='running_fresh_BF16_S4_prefix_recovery',**b.parent.process_event(os.getpid()),
        started_unix=time.time(),source_sha256=request['source_sha256'],contract=b.CONTRACT,
        factory_origin=str(b.HF),old_compressed_inputs=[],native_dispatch='none; custom experiment prefix hook enabled',
        resources=resources,native_argv=argv,runtime_identity=identity,normalized=normalized,
        groups=[],nodes=[],recoveries=[])
    b.write(root/'fit.json',record)
    names=('train_group_vae_payload','apply_group_vae_payload','_load_model_for_cat_train',
        'save_cat_v6_full_checkpoint','collect_activation_stats_for_linears')
    originals={name:getattr(native,name) for name in names}
    original_scheduler=transformers.get_scheduler
    current=None;frozen_fit=None;factory_records=None
    observe=lambda event,value:print(json.dumps(dict(event=event,**value)),flush=True)

    def evaluator(checkpoint,output,component,limited,fresh=False):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        if fresh:
            call(checkpoint,output,'fresh','6,7',limited)
        return call(checkpoint,output,component,'7',limited)

    def call(checkpoint,output,component,gpus,limited):
        command=[str(b.PYTHON),'-B','-u','-m','experiments.s4_from_bf16_20261102.evaluate',
            '--request',str(request_path),'--checkpoint',str(checkpoint),'--output',str(output),'--component',component]
        if limited:command.append('--short')
        with (output/(component+'.log')).open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=b.ROOT,env=b.environment(gpus),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
            b.write(output/(component+'_launch.json'),dict(command=command,gpus=gpus,
                **b.parent.process_event(process.pid),resident_parent=b.parent.process_event(os.getpid()),
                paused_parent_memory=b.old.old_life.memory()))
            code=process.wait()
        b.require(code==0,component+' failed; stop the affected fixed chain without retry.')
        return b.read(output/(component+'.json'))

    def factory(**kwargs):
        nonlocal factory_records
        model=originals['_load_model_for_cat_train'](**kwargs)
        b.require(all(type(module) is b.torch.nn.Linear for module in runtime.projections(model).values())
            and len(runtime.projections(model))==252,'Start with252 actual original BF16 dense projections.')
        factory_records=runtime.records(model.state_dict())
        b.write(root/'factory.json',dict(origin=str(b.HF),HF_identity=request['HF_identity'],
            compressed_inputs=[],persistent_fields=len(factory_records),initial_state=factory_records))
        return model

    def scheduler(name,optimizer,**kwargs):
        if current is not None:
            b.require(str(name)=='linear' and kwargs['num_warmup_steps']==0,'Original VAE linear schedule required.')
            kwargs['num_training_steps']=10000
        return original_scheduler(name,optimizer,**kwargs)

    def fit(**kwargs):
        nonlocal current,frozen_fit
        stage=len(record['groups'])+1;cfg=kwargs['runtime_cfg']
        b.require(cfg.category==b.CATEGORIES[stage-1] and cfg==cfgs[cfg.category] and cfg.residual_stages==4
            and cfg.steps==10000 and len(kwargs['group_refs'])==36
            and kwargs['shuffle_seed']==31+(stage-1)*100000,'Fixed original seven-category M36 S4 recipe changed.')
        model=kwargs['model'];prefix=tuple(name+'.' for name in b.parent.s.projection_names((cfg.category,)))
        frozen_fit=runtime.records({name:value for name,value in model.state_dict().items() if not name.startswith(prefix)})
        for ref in kwargs['group_refs']:
            b.require(type(ref.module) is b.torch.nn.Linear
                and b.old.old_life.tensor_record(ref.module.weight)==factory_records[ref.name+'.weight'],
                'Future category fit must read this execution original BF16 dense weight: '+ref.name)
        current=dict(category=cfg.category,stage=stage,started_unix=time.time(),shuffle_seed=kwargs['shuffle_seed'],
            fit_updates_per_stage=2 if short else 10000,schedule_steps=10000,prior_prefix_frozen_before_fit=True)
        record['groups'].append(current);b.write(root/'fit.json',record)
        kwargs['batch_size']=4096 if cfg.category=='gate_proj' else 8192
        if short:kwargs['runtime_cfg']=replace(cfg,steps=2)
        result=originals['train_group_vae_payload'](**kwargs)
        b.require(result['residual_stages']==4 and len(result['all_stage_bits'])==4
            and all(value.shape[1]==36 and value.shape[2]==cfg.codebook_bits
                and b.torch.all((value==0)|(value==1)) for value in result['all_stage_bits']),
            'Every new category needs four complete actual hard-code banks.')
        current.update(fit_seconds=time.time()-current['started_unix'],all_four_hard_code_shapes=[list(x.shape) for x in result['all_stage_bits']])
        return result

    def apply(**kwargs):
        result=originals['apply_group_vae_payload'](**kwargs)
        prefix=tuple(name+'.' for name in b.parent.s.projection_names((current['category'],)))
        after=runtime.records({name:value for name,value in kwargs['model'].state_dict().items() if not name.startswith(prefix)})
        b.require(after==frozen_fit,'Next category fitting/installation changed a prior recovered BF16 prefix or future dense/backbone.')
        current['complete_noncurrent_fields_exact_after_fit']=True
        b.write(root/'fit.json',record)
        return result

    def calibration(**kwargs):
        b.require(kwargs['dataset']=='alpaca=1' and kwargs['nsamples']==512 and kwargs['seqlen']==256
            and kwargs['seed']==31,'Retain original BF16 train-only protection calibration.')
        result=originals['collect_activation_stats_for_linears'](**kwargs)
        record['calibration']=dict(dataset='alpaca=1',samples=512,seqlen=256,seed=31,
            input_ids_sha256=b.old.data.runtime.digest({str(i):v for i,v in enumerate(result[1].input_ids)}))
        b.write(root/'fit.json',record)
        return result

    def save(model,directory,**kwargs):
        if kwargs['checkpoint_kind']=='final_model':
            # after_down already is the selected complete endpoint; avoid a duplicate model.
            record['native_duplicate_final_save_omitted']=True
            return dict(output_dir=record['nodes'][-1]['post']['checkpoint']['path'],
                meta_payload=b.read(Path(record['nodes'][-1]['post']['checkpoint']['path'])/'checkpoint_meta.json'))
        b.require(kwargs['checkpoint_kind']=='category_boundary' and kwargs['category']==current['category']
            and len(record['recoveries'])==current['stage']-1,'Each genuinely fitted category receives exactly one recovery.')
        stage=current['stage'];node=root/'nodes'/current['category'];node.mkdir(parents=True)
        state_before=rng()
        try:
            shared=dict(kwargs)
            shared['distill_stage_history']=list(request['recovery_history'])
            shared['distill_stage_meta']=dict(algorithm='fresh_BF16_S4_prefix_recovery',origin='original_HF_BF16',
                phase='before_current_recovery',did_train=False,actual_updates=0,stage=stage,contract=b.CONTRACT,
                source_sha256=request['source_sha256'],data_digest=request['data_digest'])
            pre=originals['save_cat_v6_full_checkpoint'](model,str(node/'pre'/'model'),**shared)
            pre_event=b.event(pre['output_dir'])
            node_record=dict(category=current['category'],pre=dict(checkpoint=pre_event))
            record['nodes'].append(node_record)
            b.write(node/'pre'/'expected_state.json',runtime.records(model.state_dict()))
            model.to('cpu');gc.collect();b.torch.cuda.empty_cache()
            if not short or stage in (1,2,7):
                pre_score=evaluator(pre['output_dir'],node/'pre'/'evaluation','mmlu',short)
                if not short:
                    request['_pre_correct']=budget.correct(pre_score)
                    request['_previous_post_correct']=10251 if stage==1 else record['nodes'][-2]['post']['M_correct']
                    node_record['pre']['M_correct']=request['_pre_correct']
            b.write(root/'fit.json',record)
            request['_round_source']=pre['output_dir']
            request['_round_source_binding']=pre_event
            saved_kwargs={key:value for key,value in kwargs.items() if key!='checkpoint_kind' and key!='category'}
            recovered=recovery.run(model,stage,request,node/'recovery',saved_kwargs,observe,evaluator)
            history=dict(algorithm='fresh_BF16_S4_prefix_recovery',origin='original_HF_BF16',category=current['category'],
                did_train=True,stage=stage,active_projection_count=36*stage,actual_updates=recovered['actual_updates'],
                budget_allocation=recovered['budget_allocation'],budget_decisions=recovered['budget_decisions'],
                fitting_RNG_preserved=True,new_Adam=True,data_digest=request['data_digest'],
                pre_checkpoint_id=pre_event['checkpoint_id'],contract=b.CONTRACT,source_sha256=request['source_sha256'])
            request['recovery_history'].append(history);record['recoveries'].append(recovered)
            shared['distill_stage_meta']=dict(history,phase='after_current_recovery')
            shared['distill_stage_history']=list(request['recovery_history'])
            post=originals['save_cat_v6_full_checkpoint'](model,str(node/'post'/'model'),**shared)
            b.write(node/'post'/'expected_state.json',runtime.records(model.state_dict()))
            node_record['post']=dict(checkpoint=b.event(post['output_dir']))
            if short and stage==7:evaluator(post['output_dir'],node/'post'/'evaluation','mmlu',True,fresh=True)
            elif not short or stage in (1,2):
                post_score=evaluator(post['output_dir'],node/'post'/'evaluation','mmlu',short)
                if not short:node_record['post']['M_correct']=budget.correct(post_score)
            if stage==7 and not short:
                node_record['pre']['retained_for']='Actual final FP32/Adam/RNG state and its implemented continuation/snapshot round source.'
            else:
                node_record['pre']['completed_payload_cleanup']=b.cleanup_model(pre['output_dir'],
                    'All recovery snapshots for this class have finished reading this round source; pre MMLU is complete.')
            if stage<7:
                node_record['post']['completed_payload_cleanup']=b.cleanup_model(post['output_dir'],
                    'Post MMLU is complete; the next fitter consumes the same live BF16 model, not this on-disk duplicate.')
            b.write(root/'fit.json',record)
            observe('category_fit_and_prefix_recovery_complete',dict(category=current['category'],stage=stage,
                actual_recovery_updates=recovered['actual_updates'],pre=pre_event,
                post=node_record['post']['checkpoint']))
            return post
        finally:
            restore_rng(state_before)
            b.require(rng_exact(state_before),'Custom category recovery/readout consumed the VAE fitting RNG stream.')

    replacements=dict(train_group_vae_payload=fit,apply_group_vae_payload=apply,
        _load_model_for_cat_train=factory,save_cat_v6_full_checkpoint=save,
        collect_activation_stats_for_linears=calibration)
    try:
        for name,value in replacements.items():setattr(native,name,value)
        transformers.get_scheduler=scheduler
        native.run_cat_train(cat_args=cat,hf_args=hf,training_args=training,vae_args=vae)
        b.require(len(record['groups'])==len(record['recoveries'])==len(record['nodes'])==7,'Complete seven real fits/recoveries/boundaries required.')
        record.update(status='complete_fresh_BF16_S4_prefix_recovery',exit_code=0)
    except BaseException:
        record.update(status='failed',exit_code=1,error=traceback.format_exc());raise
    finally:
        for name,value in originals.items():setattr(native,name,value)
        transformers.get_scheduler=original_scheduler
        record['elapsed_seconds']=time.time()-record['started_unix'];b.write(root/'fit.json',record)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();root=Path(b.read(args.request)['output']);code=1
    try:execute(args.request);code=0
    finally:(root/'fit.exit').write_text(str(code)+'\n')


if __name__=='__main__':main()
