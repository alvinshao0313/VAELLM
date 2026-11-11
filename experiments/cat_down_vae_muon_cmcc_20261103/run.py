"""One fresh BF16 live-down Muon fit, followed by true deployment and fresh."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b
from . import fit
from experiments.adaptive_class_20261030.run_pipeline import warm,logits,fresh


def execute(phase):
    b.require(Path(sys.executable).resolve()==b.PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV')=='bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES')==str(b.GPU) and os.environ.get('PYTHONHASHSEED')=='31'
        and os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True',
        'Use caller bitvae, assigned GPU2, original seed and mature allocator environment.')
    prepared=b.read(b.RESULT / 'prepared.json')
    b.check_pins(prepared['source_sha256'])
    binding,reused=b.sources()
    b.require(binding==prepared['source_binding'] and reused==prepared['reused_source_evidence'], 'The actual source evidence changed.')
    short=phase=='short'
    output=b.OUTPUT / phase
    b.require(not output.exists(),'Never overwrite a short or formal fit.')
    output.mkdir(parents=True)
    parsed,identity=b.recipe(output)
    records=[]
    report=dict(schema='cat_down_VAE_Muon_fit_v1',status='running',phase=phase,validation_only=short,
        command=[sys.executable,*sys.argv],pid=os.getpid(),start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),
        started_unix=time.time(),source_binding=binding,reused_source_evidence=reused,contract=b.CONTRACT,
        source_sha256=prepared['source_sha256'],native_runtime_identity=identity,optimizer_stages=records)
    path=output / 'report.json'
    b.write(path,report)
    try:
        b.recovery.native.configure_deterministic_mode(parsed[0].deterministic)
        activation,plan,restored=b.base.restore_cat_runtime_state(parsed[-1],access_token=parsed[1].access_token)
        b.require(restored==identity,'Restore the exact original legal-train calibration runtime.')
        model,meta,loaded,_budget=b.checkpoint.load(b.PARENT)
        b.require(meta['checkpoint_id']==binding['after_up']['checkpoint_id'] and not loaded.missing_keys and not loaded.unexpected_keys,
            'Strictly load the original after-up; never inherit the short model.')
        b.recovery.native.set_seed(31)
        report['fit']=fit.execute(model,parsed,activation,plan,output=output / 'fit',short=short,records=records)
        warmup=warm(model,216)
        tokens=parsed[-1]['activation_runtime']['cache']['input_ids'][0].reshape(1,256).clone()
        inputs=dict(input_ids=tokens)
        expected=logits(model,inputs)
        provenance=dict(schema='cat_down_VAE_Muon_fit_v1',phase=phase,validation_only=short,source_after_up=binding['after_up'],
            actual_category='down_proj',updates_per_stage=2 if short else 10000,recovery_updates=0,contract=b.CONTRACT,
            source_sha256=prepared['source_sha256'])
        saved,budget,payload=b.checkpoint.save(model,output / 'raw_model',provenance=provenance)
        evidence=output / 'same_input_deployment.pt'
        b.torch.save(dict(checkpoint_id=saved['checkpoint_id'],deployment_digest=b.c.digest(payload),inputs=inputs,logits=expected),evidence)
        report.update(raw_model=str(output / 'raw_model'),raw_checkpoint_id=saved['checkpoint_id'],raw_budget=budget,deployment_warmup=warmup)
        del model,payload,expected
        gc.collect();b.torch.cuda.empty_cache()
        subprocess.run([sys.executable,'-B','-u','-m','experiments.cat_down_vae_muon_cmcc_20261103.run',
            '--fresh',str(output / 'raw_model'),'--evidence',str(evidence)],cwd=b.REPO,check=True)
        strict=b.read(output / 'strict_reload.json')
        b.require(strict['checkpoint_id']==saved['checkpoint_id'] and strict['same_input_logits_exact'], 'Independent true mixed BF16 fresh verification failed.')
        report['strict_reload_report']=str(output / 'strict_reload.json')
        b.check_pins(prepared['source_sha256'])
        report.update(status='completed_VAE_Muon_short' if short else 'completed_VAE_Muon_formal_raw',exit_code=0)
    except BaseException:
        report.update(status='failed',exit_code=1,error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(),elapsed_seconds=time.time()-report['started_unix'],
            peak_allocated_bytes=b.torch.cuda.max_memory_allocated(),peak_reserved_bytes=b.torch.cuda.max_memory_reserved())
        b.write(path,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('short','formal'))
    parser.add_argument('--fresh',type=Path)
    parser.add_argument('--evidence',type=Path)
    args=parser.parse_args()
    b.torch.set_num_threads(8)
    b.torch.backends.cuda.matmul.allow_tf32=False
    b.torch.backends.cudnn.allow_tf32=False
    if args.fresh:fresh(args.fresh,args.evidence)
    else:execute(args.phase)
