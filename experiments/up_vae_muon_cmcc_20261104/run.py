"""Fresh original-parent up fit, whole partial deployment and independent fresh."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b, fit
from experiments.adaptive_class_20261030.run_pipeline import warm, logits, fresh


def execute(arm,phase):
    gpu = b.ARMS[arm]['gpu']
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu) and os.environ.get('PYTHONHASHSEED') == '31'
        and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True', 'Use caller bitvae, assigned GPU and the mature allocator/seed.')
    prepared = b.read(b.RESULT / 'prepared.json');b.check_pins(prepared['source_sha256'])
    source = b.sources();b.require(source == prepared['source_binding'], 'Original up parent/control evidence changed.')
    short, output = phase == 'short', b.OUTPUT / arm / phase
    b.require(not output.exists(), 'Never overwrite a candidate short/formal output.')
    output.mkdir(parents=True)
    parsed,identity,baseline = b.recipe(arm,output)
    records = []
    report = dict(status='running',arm=arm,phase=phase,validation_only=short,pid=os.getpid(),started_unix=time.time(),
        source_binding=source,contract=b.CONTRACT,source_sha256=prepared['source_sha256'],
        native_runtime_identity=identity,original_runtime_identity=baseline,optimizer_stages=records,command=[sys.executable,*sys.argv])
    path = output / 'report.json';b.write(path,report)
    try:
        b.recovery.native.configure_deterministic_mode(parsed[0].deterministic)
        activation,plan,restored = b.restore_cat_runtime_state(parsed[-1],access_token=parsed[1].access_token)
        b.require(restored == baseline, 'Restore the actual original train calibration runtime; the D16 recipe is an explicit fork.')
        model,meta,loaded,_budget = b.checkpoint.load(b.PARENT)
        b.require(meta['checkpoint_id'] == source['after_gate']['checkpoint_id'] and not loaded.missing_keys and not loaded.unexpected_keys,
            'Fresh-load the original after-gate for every technical/formal arm.')
        b.recovery.native.set_seed(31)
        report['fit'] = fit.execute(model,parsed,activation,plan,arm=arm,output=output / 'fit',short=short,records=records)
        warmup = warm(model,180)
        inputs = dict(input_ids=parsed[-1]['activation_runtime']['cache']['input_ids'][0].reshape(1,256).clone())
        expected = logits(model,inputs)
        provenance = dict(schema='C0_up_one_factor_v1',arm=arm,phase=phase,validation_only=short,
            source_after_gate=source['after_gate'],actual_category='up_proj',updates_per_stage=2 if short else 10000,
            recovery_updates=0,contract=b.CONTRACT,source_sha256=prepared['source_sha256'])
        saved,budget,payload = b.checkpoint.save(model,output / 'raw_model',provenance=provenance)
        evidence = output / 'same_input_deployment.pt'
        b.torch.save(dict(checkpoint_id=saved['checkpoint_id'],deployment_digest=b.c.digest(payload),inputs=inputs,logits=expected),evidence)
        report.update(raw_model=str(output / 'raw_model'),raw_checkpoint_id=saved['checkpoint_id'],raw_budget=budget,deployment_warmup=warmup)
        del model,payload,expected;gc.collect();b.torch.cuda.empty_cache()
        subprocess.run([sys.executable,'-B','-u','-m','experiments.up_vae_muon_cmcc_20261104.run',
            '--fresh',str(output / 'raw_model'),'--evidence',str(evidence)],cwd=b.REPO,check=True)
        strict = b.read(output / 'strict_reload.json')
        b.require(strict['checkpoint_id'] == saved['checkpoint_id'] and strict['same_input_logits_exact'], 'Independent actual partial deployment fresh failed.')
        b.check_pins(prepared['source_sha256'])
        report.update(status='completed_up_short' if short else 'completed_up_formal_raw',exit_code=0,
            strict_reload_report=str(output / 'strict_reload.json'))
    except BaseException:
        report.update(status='failed',exit_code=1,error=traceback.format_exc());raise
    finally:
        report.update(finished_unix=time.time(),elapsed_seconds=time.time()-report['started_unix'],
            peak_allocated_bytes=b.torch.cuda.max_memory_allocated(),peak_reserved_bytes=b.torch.cuda.max_memory_reserved())
        b.write(path,report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=tuple(b.ARMS));parser.add_argument('--phase',choices=('short','formal'))
    parser.add_argument('--fresh',type=Path);parser.add_argument('--evidence',type=Path)
    args = parser.parse_args()
    b.torch.set_num_threads(8);b.torch.backends.cuda.matmul.allow_tf32=False;b.torch.backends.cudnn.allow_tf32=False
    if args.fresh:fresh(args.fresh,args.evidence)
    else:execute(args.arm,args.phase)
