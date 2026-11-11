"""Fit one independent live-down fork and save real deployed/fresh evidence."""
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


def verify(directory,evidence):
    subprocess.run([sys.executable,'-B','-u','-m','experiments.down_geometry_baseline_cmcc_20261103.run',
        '--fresh',str(directory),'--evidence',str(evidence)],cwd=b.REPO,check=True)
    result = b.read(directory.parent / 'strict_reload.json')
    b.require(result['checkpoint_id'] == b.read(directory / b.checkpoint.META_FILE)['checkpoint_id']
        and result['same_input_logits_exact'], 'Independent fresh consumer evidence differs.')
    return directory.parent / 'strict_reload.json'


def deploy(model,destination,provenance,inputs):
    warmup = warm(model,216)
    expected = logits(model,inputs)
    meta,budget,payload = b.checkpoint.save(model,destination,provenance=provenance)
    evidence = destination.parent / 'same_input_deployment.pt'
    b.torch.save(dict(checkpoint_id=meta['checkpoint_id'],deployment_digest=b.c.digest(payload),inputs=inputs,logits=expected),evidence)
    return meta,budget,evidence,warmup


def execute(arm,phase):
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(b.ARMS[arm]['gpu'])
        and os.environ.get('PYTHONHASHSEED') == '31', 'Use caller bitvae and the fixed assigned single GPU.')
    prepared = b.read(b.RESULT / 'prepared.json')
    b.check_pins(prepared['source_sha256'])
    b.require(b.sources() == prepared['source_binding'], 'Real source binding changed after CPU verification.')
    output = b.OUTPUT / arm / phase
    b.require(not output.exists(), 'Never overwrite an existing fit.')
    output.mkdir(parents=True)
    parsed,identity,baseline = b.recipe(arm,output)
    short = phase == 'short'
    report = dict(schema='matched_live_down_geometry_v1',status='running',arm=arm,phase=phase,validation_only=short,
        command=[sys.executable,*sys.argv],pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),started_unix=time.time(),
        source_binding=prepared['source_binding'],runtime_identity=identity,source_sha256=prepared['source_sha256'],
        settings=b.ARMS[arm],prepared_sha256=b.sha(b.RESULT / 'prepared.json'))
    path = output / 'report.json'
    b.write(path,report)
    try:
        b.recovery.native.configure_deterministic_mode(parsed[0].deterministic)
        activation,plan,restored = b.restore_cat_runtime_state(parsed[-1],access_token=parsed[1].access_token)
        b.require(restored == baseline, 'Restore the same original calibration runtime.')
        model,meta,loaded,_budget = b.checkpoint.load(b.PARENT)
        b.require(meta['checkpoint_id'] == prepared['source_binding']['after_up']['checkpoint_id']
            and not loaded.missing_keys and not loaded.unexpected_keys, 'Strictly load the actual live after-up parent.')
        b.recovery.native.set_seed(31)
        report['fit'] = fit.execute(model,parsed,activation,plan,arm=arm,output=output / 'fit',short=short)
        provenance = dict(schema='matched_live_down_geometry_v1',arm=arm,phase=phase,validation_only=short,
            source_after_up=prepared['source_binding']['after_up'],source_C0_report=prepared['source_binding']['c0_report'],
            actual_category='down_proj',fit_updates_per_stage=2 if short else 10000,recovery_updates=0,settings=b.ARMS[arm])
        tokens = parsed[-1]['activation_runtime']['cache']['input_ids'][0].reshape(1,256).clone()
        meta,budget,evidence,warmup = deploy(model,output / 'raw_model',provenance,dict(input_ids=tokens))
        report.update(raw_model=str(output / 'raw_model'),raw_checkpoint_id=meta['checkpoint_id'],raw_budget=budget,
            raw_deployment_warmup=warmup)
        del model
        gc.collect()
        b.torch.cuda.empty_cache()
        report['strict_reload_report'] = str(verify(output / 'raw_model',evidence))
        b.check_pins(prepared['source_sha256'])
        report.update(status='completed_new_path_short' if short else 'completed_formal_raw_down',exit_code=0)
    except BaseException:
        report.update(status='failed',exit_code=1,error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(),elapsed_seconds=time.time()-report['started_unix'],
            peak_allocated=b.torch.cuda.max_memory_allocated(),peak_reserved=b.torch.cuda.max_memory_reserved())
        b.write(path,report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=tuple(b.ARMS))
    parser.add_argument('--phase',choices=('short','formal'))
    parser.add_argument('--fresh',type=Path)
    parser.add_argument('--evidence',type=Path)
    args = parser.parse_args()
    b.torch.set_num_threads(8)
    b.torch.backends.cuda.matmul.allow_tf32=False
    b.torch.backends.cudnn.allow_tf32=False
    if args.fresh:
        fresh(args.fresh,args.evidence)
    else:
        execute(args.arm,args.phase)
