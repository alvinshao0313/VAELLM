"""The predesignated D16 candidate's own unchanged final down500 recovery."""
import gc
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback

from . import bindings as b
from .run import deploy,verify
from experiments.adaptive_class_20261030 import continuation as flow


def execute():
    arm,output='d16',b.OUTPUT / 'd16/recovery'
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and not output.exists(),
        'Only the predesignated D16 branch may consume one original down500 suffix on GPU0.')
    prepared=b.read(b.RESULT / 'prepared.json')
    b.check_pins(prepared['source_sha256'])
    raw=b.read(b.OUTPUT / 'd16/formal/report.json')
    b.require(raw['status'] == 'completed_formal_raw_down' and raw['exit_code'] == 0
        and raw['source_binding'] == prepared['source_binding'], 'Require this D16 arm own completed raw down fit.')
    # The raw full8 is descriptive: its score does not cancel this fixed candidate.
    readout=b.read(b.RESULT / 'd16/raw_readout.json')
    screening=b.read(b.RESULT / 'd16/raw_full8/full8_summary.json')
    b.require(readout['checkpoint_id'] == screening['checkpoint_id'] == raw['raw_checkpoint_id'],
        'The complete raw diagnostics must refer to this exact raw UUID.')
    parsed,identity,_baseline=b.recipe(arm,output)
    cat,hf,training,vae,_cfgs,_runtime=parsed
    stage=b.recovery.canonical.resolve_cat_after_category_stage(cat,training,category='down_proj',round_idx=6)
    reference=b.read(b.C0 / 'report.json')['stages'][2]['original_stage']
    effective=json.loads(json.dumps(asdict(stage)))
    effective.pop('output_dir')
    reference.pop('output_dir')
    b.require(b.c.read(b.C0 / 'report.json')['stages'][2]['recovery']['actual_steps'] == 500
        and effective == reference,'Keep the original stage_seed/data_seed/data/500 schedule and parameter scope.')
    output.mkdir(parents=True)
    record=dict(schema='matched_live_down_recovery_v1',status='running',arm=arm,phase='formal',
        command=[sys.executable,*sys.argv],pid=os.getpid(),started_unix=time.time(),
        source_binding=prepared['source_binding'],source_sha256=prepared['source_sha256'],
        raw_report=str(b.OUTPUT / 'd16/formal/report.json'),raw_checkpoint_id=raw['raw_checkpoint_id'],
        raw_full8=str(b.RESULT / 'd16/raw_full8/full8_summary.json'),raw_readout=str(b.RESULT / 'd16/raw_readout.json'),
        original_stage=asdict(stage),original_stage_exact=True,runtime_identity=identity,
        RNG_contract='Unchanged canonical Trainer initializes stage_seed/data_seed. Independent raw readout processes do not affect it.')
    path=output / 'report.json'
    b.write(path,record)
    try:
        model,meta,loaded,_budget=b.checkpoint.load(raw['raw_model'])
        b.require(meta['checkpoint_id'] == raw['raw_checkpoint_id'] and not loaded.missing_keys and not loaded.unexpected_keys,
            'Recover only this candidate own raw native216/compact36 state.')
        b.recovery.native.configure_deterministic_mode(cat.deterministic)
        model,evidence,probe=flow.recover(model,stage,vae,hf,output,b.Path(raw['raw_model']),meta,short=False)
        record['recovery']=evidence
        provenance=dict(schema='matched_live_down_recovery_v1',arm=arm,phase='formal',actual_category='down_proj',
            source_binding=prepared['source_binding'],source_raw_checkpoint_id=raw['raw_checkpoint_id'],
            fit_updates_per_stage=10000,actual_recovery_updates=500,settings=b.ARMS[arm])
        saved,budget,evidence_path,warmup=deploy(model,output / 'final_model',provenance,probe)
        record.update(final_model=str(output / 'final_model'),final_checkpoint_id=saved['checkpoint_id'],
            final_budget=budget,deployment_warmup=warmup)
        del model
        gc.collect()
        b.torch.cuda.empty_cache()
        record['final_strict_reload_report']=str(verify(output / 'final_model',evidence_path))
        b.check_pins(prepared['source_sha256'])
        record.update(status='completed_own_down500',exit_code=0)
    except BaseException:
        record.update(status='failed',exit_code=1,error=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'])
        b.write(path,record)


if __name__ == '__main__':
    b.torch.set_num_threads(8)
    b.torch.backends.cuda.matmul.allow_tf32=False
    b.torch.backends.cudnn.allow_tf32=False
    execute()
