"""Fresh actual D16/B16384 VAE-Muon fitting, original native deployment and fresh."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b
from . import fit,validation
from experiments.adaptive_class_20261030.run_pipeline import warm,logits,fresh


def verified_short():
    report=b.read(b.OUTPUT/'short/report.json');strict=b.read(b.OUTPUT/'short/strict_reload.json')
    p=b.prepared()
    b.require(report['status']=='completed_D16_VAE_Muon_short' and report['exit_code']==0
        and report['source_sha256']==p['source_sha256'] and report['contract']==b.CONTRACT
        and report['fit']['original_D16_flat_initialization_matches']==[True,True]
        and report['fit']['vector_batch_per_layer']==16384 and report['fit']['actual_updates_per_stage']==2
        and report['fit']['all_frozen_prefix_fields_exact'] and report['fit']['protection_and_rotation_fields_exact_to_control']
        and report['packed_codes']['all_packed_bits_roundtrip_exact']
        and strict['checkpoint_id']==report['raw_checkpoint_id'] and strict['same_input_logits_exact'] and strict['exit_code']==0,
        'Require actual D16/M36/B16384 both-stage short, D16 initialization/kappa/frozen/pack and independent BF16 fresh.')
    return dict(report=str(b.OUTPUT/'short/report.json'),report_sha256=b.sha(b.OUTPUT/'short/report.json'),
        fresh=str(b.OUTPUT/'short/strict_reload.json'),fresh_sha256=b.sha(b.OUTPUT/'short/strict_reload.json'),
        checkpoint_id=report['raw_checkpoint_id'],seconds=report['elapsed_seconds'])


def execute(phase):
    b.require(Path(sys.executable).resolve()==b.PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV')=='bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and os.environ.get('PYTHONHASHSEED')=='31'
        and os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True',
        'Use caller bitvae, assigned GPU1 and original seed/allocator.')
    p=b.prepared();short=phase=='short'
    proof=None if short else verified_short()
    output=b.OUTPUT/phase
    b.require(not output.exists(),'Do not overwrite or repeat the fixed fit phase.')
    parsed,identity,baseline=b.recipe(output)
    records=[];output.mkdir(parents=True)
    report=dict(schema='D16_VAE_Muon_fit_execution_v1',status='running',phase=phase,validation_only=short,
        command=[sys.executable,*sys.argv],pid=os.getpid(),start_ticks=b.executor.proc_start(os.getpid()),started_unix=time.time(),
        source_binding=p['source_binding'],controls=p['controls'],contract=b.CONTRACT,source_sha256=p['source_sha256'],
        prepared_sha256=b.sha(b.PREPARED),runtime_identity=identity,settings=b.SETTINGS,optimizer_stages=records,short_evidence=proof)
    path=output/'report.json';b.write(path,report)
    try:
        b.recovery.native.configure_deterministic_mode(parsed[0].deterministic)
        activation,plan,restored=b.base.restore_cat_runtime_state(parsed[-1],access_token=parsed[1].access_token)
        b.require(restored==baseline,'Restore the original train calibration runtime, not a fitted control state.')
        model,meta,loaded,_=b.checkpoint.load(b.PARENT)
        b.require(meta['checkpoint_id']==b.CONTRACT['parent'] and not loaded.missing_keys and not loaded.unexpected_keys,
            'Freshly load only the actual original after-up before each independent fit.')
        b.recovery.native.set_seed(31)
        report['fit']=fit.execute(model,parsed,activation,plan,output=output/'fit',short=short,records=records)
        report['packed_codes']=validation.codes(model)
        warmup=warm(model,216)
        inputs=dict(input_ids=parsed[-1]['activation_runtime']['cache']['input_ids'][0].reshape(1,256).clone())
        expected=logits(model,inputs)
        provenance=dict(schema='D16_VAE_Muon_raw_provenance_v1',phase=phase,validation_only=short,
            source_binding=p['source_binding'],controls=p['controls'],contract=b.CONTRACT,settings=b.SETTINGS,
            actual_category='down_proj',actual_updates_per_stage=2 if short else 10000,recovery_updates=0,
            original_D16_flat_initialization_matches=report['fit']['original_D16_flat_initialization_matches'],source_sha256=p['source_sha256'])
        saved,budget,payload=b.checkpoint.save(model,output/'raw_model',provenance=provenance)
        reference=p['controls']['D16_raw_budget']
        b.require(all(budget[key]==reference[key] for key in
            ('user_linear_bits','packed_linear_bits','original_linear_weights','full_model_tensor_bytes')),
            'Measured complete D16 deployment budget must match the original same-geometry control.')
        evidence=output/'same_input_deployment.pt'
        b.torch.save(dict(checkpoint_id=saved['checkpoint_id'],deployment_digest=b.c.digest(payload),inputs=inputs,logits=expected),evidence)
        report.update(raw_model=str(output/'raw_model'),raw_checkpoint_id=saved['checkpoint_id'],raw_budget=budget,deployment_warmup=warmup)
        del model,payload,expected
        gc.collect();b.torch.cuda.empty_cache()
        subprocess.run([sys.executable,'-B','-u','-m','experiments.down_d16_vae_muon_cmcc_20261103.run',
            '--fresh',str(output/'raw_model'),'--evidence',str(evidence)],cwd=b.REPO,check=True)
        strict=b.read(output/'strict_reload.json')
        b.require(strict['checkpoint_id']==saved['checkpoint_id'] and strict['same_input_logits_exact'],'Independent native D16 BF16 fresh failed.')
        report['strict_reload_report']=str(output/'strict_reload.json')
        b.check_pins(p['source_sha256'])
        report.update(status='completed_D16_VAE_Muon_short' if short else 'completed_D16_VAE_Muon_formal_raw',exit_code=0)
    except BaseException:
        report.update(status='failed',exit_code=1,error=traceback.format_exc());raise
    finally:
        report.update(finished_unix=time.time(),elapsed_seconds=time.time()-report['started_unix'],
            peak_allocated_bytes=b.torch.cuda.max_memory_allocated(),peak_reserved_bytes=b.torch.cuda.max_memory_reserved())
        b.write(path,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('short','formal'));parser.add_argument('--fresh',type=Path);parser.add_argument('--evidence',type=Path)
    args=parser.parse_args()
    b.torch.set_num_threads(8);b.torch.backends.cuda.matmul.allow_tf32=False;b.torch.backends.cudnn.allow_tf32=False
    if args.fresh:
        b.require(args.evidence is not None,'Fresh requires the actual pre-save deployment evidence.');fresh(args.fresh,args.evidence)
    else:
        b.require(args.phase is not None,'Choose the actual fitting phase.');execute(args.phase)
