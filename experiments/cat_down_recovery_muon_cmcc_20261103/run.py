"""Fresh raw C0 -> canonical category Muon2/500 -> genuine BF16 mixed deployment."""
import argparse
from dataclasses import asdict
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

from . import bindings as b
from . import training
from experiments.adaptive_class_20261030.run_pipeline import warm,logits,fresh


def verified_short():
    report=b.read(b.OUTPUT/'short/report.json')
    strict=b.read(b.OUTPUT/'short/strict_reload.json')
    b.require(report['status']=='completed_original_C0_down_Muon_short' and report['exit_code']==0
        and report['source_binding']==b.prepared()['source_binding'] and report['recovery']['actual_steps']==2
        and report['recovery']['optimizer_updates'][0]['external_scheduled_lr']==0.
        and report['recovery']['optimizer_updates'][1]['external_scheduled_lr']>0
        and strict['checkpoint_id']==report['final_checkpoint_id'] and strict['exit_code']==0
        and strict['same_input_logits_exact'],'Require the actual complete new standard-optimizer2/save/fresh path.')
    return dict(report=str(b.OUTPUT/'short/report.json'),report_sha256=b.sha(b.OUTPUT/'short/report.json'),
        fresh=str(b.OUTPUT/'short/strict_reload.json'),fresh_sha256=b.sha(b.OUTPUT/'short/strict_reload.json'),
        checkpoint_id=report['final_checkpoint_id'],scope='Engineering2 only, never a model/optimizer source for formal500.')


def execute(phase,seconds):
    b.require(Path(sys.executable).resolve()==b.PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV')=='bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES')==str(b.GPU) and b.torch.cuda.device_count()==1,
        'Use activated bitvae and only the assigned GPU3.')
    prepared=b.prepared()
    short_proof=verified_short() if phase=='formal' else None
    output=b.OUTPUT/phase
    b.require(not output.exists(),'Never overwrite, resume or automatically retry this category comparison.')
    output.mkdir()
    cat,hf,args,vae,runtime,stage=b.recipe(output)
    b.recovery.native.configure_deterministic_mode(cat.deterministic)
    started=time.monotonic()
    report=dict(schema='C0_down_category_Muon_execution_v1',status='running',phase=phase,validation_only=phase=='short',
        pid=os.getpid(),start_ticks=b.executor.proc_start(os.getpid()),command=[sys.executable,*sys.argv],started_unix=time.time(),
        physical_GPU=b.GPU,source_binding=prepared['source_binding'],source_sha256=prepared['source_sha256'],
        prepared_sha256=b.sha(b.PREPARED),contract=b.CONTRACT,original_stage_reference=asdict(stage),
        original_stage_optimizer_is_reference_only=True,
        effective_optimizer_override='experiments.cat_down_recovery_muon_cmcc_20261103.training.MuonMixedCatTrainer.create_optimizer',
        effective_runtime_identity=dict(canonical=runtime['runtime_identity'],category_optimizer_override=b.CONTRACT),
        original_schedule_steps=500,actual_updates=2 if phase=='short' else 500,short_lifecycle_evidence=short_proof)
    b.write(output/'report.json',report)
    from transformers import TrainerCallback
    class TimeBudget(TrainerCallback):
        def on_step_begin(self,*_args,**_kwargs):
            b.require(time.monotonic()-started<seconds,'Declared category training time budget exceeded; stop dependent stages.')
    try:
        model,meta,strict,budget=b.checkpoint.load(b.RAW)
        b.require(meta['checkpoint_id']==b.RAW_ID and not strict.missing_keys and not strict.unexpected_keys,
            'Each phase loads the exact original raw C0 independently, never short weights.')
        evidence={}
        observer=training.ObserveMuon(evidence,short=phase=='short')
        saved=training.original.MixedCatTrainer
        training.original.MixedCatTrainer=training.MuonMixedCatTrainer
        try:
            trainer,teacher,native,_compact,inventory=b.recovery.make_trainer(model,stage,vae,hf,output,b.RAW,meta,
                callbacks=[observer,TimeBudget()],validation_save_steps=1 if phase=='short' else None)
        finally:
            training.original.MixedCatTrainer=saved
        observer.trainer=trainer
        result=trainer.train()
        b.require(evidence['frozen_before']==prepared['frozen_state'],
            'Actual training frozen complement differs from the genuine prepared raw C0.')
        evidence.update(actual_inventory=inventory,trainer_metrics=result.metrics,actual_history=trainer.state.log_history,
            schedule_max_steps=500)
        probe=observer.probe
        report['recovery']=evidence
        b.recovery.canonical._release_cat_trainer_training_state(trainer,logger=logging.getLogger(__name__))
        teacher.offload_for_eval()
        trainer.teacher_runtime=None
        model,merge=b.recovery.finalize(trainer,stage,native)
        evidence['merge']=merge
        observer.trainer=None
        del teacher,trainer
        gc.collect();b.torch.cuda.empty_cache()
        warmup=warm(model,216)
        expected=logits(model,probe)
        provenance=dict(schema='C0_down_category_Muon_provenance_v1',phase=phase,validation_only=phase=='short',
            source_binding=prepared['source_binding'],source_raw_checkpoint_id=b.RAW_ID,actual_category='down_proj',
            actual_recovery_updates=report['actual_updates'],planned_schedule_steps=500,contract=b.CONTRACT,
            effective_optimizer=evidence['effective_optimizer'],effective_optimizer_override=report['effective_optimizer_override'],
            original_stage_optimizer_is_reference_only=True,short_lifecycle_evidence=short_proof)
        saved,budget,payload=b.checkpoint.save(model,output/'final_model',provenance=provenance)
        b.require(all(budget[key]==prepared['source_budget'][key] for key in
            ('user_linear_bits','packed_linear_bits','original_linear_weights','full_model_tensor_bytes')),
            'Decoder optimizer recovery must preserve actual deployed bit/tensor budget.')
        b.torch.save(dict(checkpoint_id=saved['checkpoint_id'],deployment_digest=b.c.digest(payload),inputs=probe,logits=expected),
            output/'same_input_deployment.pt')
        report.update(status='completed_original_C0_down_Muon_'+phase,exit_code=0,
            final_model=str(output/'final_model'),final_checkpoint_id=saved['checkpoint_id'],final_budget=budget,
            deployment_warmup=warmup,expected_deployment_digest=b.c.digest(payload),
            strict_reload_report=str(output/'strict_reload.json'))
        b.check_pins(prepared['source_sha256'])
    except BaseException:
        report.update(status='failed',exit_code=1,error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(),elapsed_seconds=time.monotonic()-started)
        b.write(output/'report.json',report)
    print(json.dumps(dict(status=report['status'],checkpoint_id=report['final_checkpoint_id'],seconds=report['elapsed_seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('short','formal'))
    parser.add_argument('--timeout-seconds',type=int)
    parser.add_argument('--fresh',type=Path)
    parser.add_argument('--evidence',type=Path)
    args=parser.parse_args()
    b.torch.set_num_threads(4)
    b.torch.backends.cuda.matmul.allow_tf32=False
    b.torch.backends.cudnn.allow_tf32=False
    if args.fresh:
        b.require(args.evidence is not None,'Fresh requires the actual saved same-input deployment evidence.')
        fresh(args.fresh,args.evidence)
    else:
        b.require(args.phase is not None and args.timeout_seconds is not None,'Run requires a declared phase and budget.')
        execute(args.phase,args.timeout_seconds)


if __name__=='__main__':
    main()
