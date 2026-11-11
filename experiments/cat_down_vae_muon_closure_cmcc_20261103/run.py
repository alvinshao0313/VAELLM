"""One fresh original flow.recover AdamW500, actual deployment and independent fresh."""
import argparse
from dataclasses import asdict
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b
from experiments.adaptive_class_20261030 import continuation as flow
from experiments.adaptive_class_20261030.run_pipeline import warm,logits,fresh


def execute():
    b.require(Path(sys.executable).resolve()==b.PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV')=='bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES')=='2' and os.environ.get('PYTHONHASHSEED')=='31'
        and os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True',
        'Use caller bitvae, assigned GPU2, original seed and mature allocator.')
    prepared=b.prepared()
    output=b.OUTPUT/'formal'
    b.require(not output.exists(),'Never overwrite or repeat the single approved closure.')
    cat,hf,training,vae,runtime,stage=b.recipe(output)
    output.mkdir(parents=True)
    record=dict(schema='VAE_Muon_original_Adam500_execution_v1',status='running',phase='formal',validation_only=False,
        command=[sys.executable,*sys.argv],pid=os.getpid(),start_ticks=b.executor.proc_start(os.getpid()),started_unix=time.time(),
        contract=b.CONTRACT,source_binding=prepared['source_binding'],source_sha256=prepared['source_sha256'],
        prepared_sha256=b.sha(b.PREPARED),original_stage=asdict(stage),original_stage_exact=True,
        effective_optimizer=prepared['effective_optimizer'],optimizer_state_fresh=True,
        original_runtime_identity=runtime['runtime_identity'],
        boundary='Only VAE-fitting optimizer differs upstream; this suffix is the original C0 Adam500 with no VAE optimizer state import.')
    path=output/'report.json'
    b.write(path,record)
    try:
        b.recovery.native.configure_deterministic_mode(cat.deterministic)
        model,meta,strict,budget=b.checkpoint.load(b.RAW)
        b.require(meta['checkpoint_id']==b.RAW_ID and not strict.missing_keys and not strict.unexpected_keys,
            'Strictly load only this completed own raw VAE-Muon UUID.')
        model,evidence,probe=flow.recover(model,stage,vae,hf,output,b.RAW,meta,short=False)
        b.require(evidence['actual_steps']==500 and evidence['frozen_persistent_fields_exact']
            and evidence['frozen_before']==prepared['frozen_state'],'Finish the complete original500 and preserve its actual frozen complement.')
        record['recovery']=evidence
        provenance=dict(schema='VAE_Muon_original_Adam500_provenance_v1',phase='formal',validation_only=False,
            source_binding=prepared['source_binding'],source_raw_checkpoint_id=b.RAW_ID,
            source_after_up_checkpoint_id=prepared['source_binding']['upstream_binding']['after_up']['checkpoint_id'],
            actual_category='down_proj',VAE_updates_per_stage=10000,actual_recovery_updates=500,
            contract=b.CONTRACT,effective_optimizer=prepared['effective_optimizer'])
        warmup=warm(model,216)
        expected=logits(model,probe)
        saved,budget,payload=b.checkpoint.save(model,output/'final_model',provenance=provenance)
        evidence_path=output/'same_input_deployment.pt'
        b.torch.save(dict(checkpoint_id=saved['checkpoint_id'],deployment_digest=b.c.digest(payload),inputs=probe,logits=expected),evidence_path)
        record.update(final_model=str(output/'final_model'),final_checkpoint_id=saved['checkpoint_id'],final_budget=budget,
            deployment_warmup=warmup,expected_deployment_digest=b.c.digest(payload))
        del model,payload,probe,expected
        gc.collect()
        b.torch.cuda.empty_cache()
        subprocess.run([sys.executable,'-B','-u','-m','experiments.cat_down_vae_muon_closure_cmcc_20261103.run',
            '--fresh',str(output/'final_model'),'--evidence',str(evidence_path)],cwd=b.REPO,check=True)
        strict=b.read(output/'strict_reload.json')
        b.require(strict['checkpoint_id']==saved['checkpoint_id'] and strict['same_input_logits_exact'],
            'Independent actual final UUID deployment fresh failed.')
        record['strict_reload_report']=str(output/'strict_reload.json')
        b.check_pins(prepared['source_sha256'])
        record.update(status='completed_VAE_Muon_original_Adam500_and_fresh',exit_code=0)
    except BaseException:
        record.update(status='failed',exit_code=1,error=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'],
            peak_allocated_bytes=b.torch.cuda.max_memory_allocated(),peak_reserved_bytes=b.torch.cuda.max_memory_reserved())
        b.write(path,record)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh',type=Path)
    parser.add_argument('--evidence',type=Path)
    args=parser.parse_args()
    b.torch.set_num_threads(8)
    b.torch.backends.cuda.matmul.allow_tf32=False
    b.torch.backends.cudnn.allow_tf32=False
    if args.fresh:
        b.require(args.evidence is not None,'Independent fresh requires actual pre-save evidence.')
        fresh(args.fresh,args.evidence)
    else:
        execute()
