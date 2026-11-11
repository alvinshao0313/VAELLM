"""This candidate own raw -> unchanged mature original torch AdamW500/fresh."""
from dataclasses import asdict
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b
from .run import verified_short
from experiments.adaptive_class_20261030 import continuation as flow
from experiments.adaptive_class_20261030.run_pipeline import warm,logits


def execute():
    b.require(Path(sys.executable).resolve()==b.PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV')=='bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and os.environ.get('PYTHONHASHSEED')=='31'
        and os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True','Use caller bitvae/assigned GPU1/original seed and allocator.')
    p=b.prepared();proof=verified_short();raw,_=b.raw_source()
    output=b.OUTPUT/'recovery'
    b.require(not output.exists(),'Never repeat or overwrite the single original Adam500 suffix.')
    parsed,identity,_baseline,stage=b.recovery_recipe(output)
    cat,hf,_training,vae,_cfgs,_runtime=parsed
    output.mkdir(parents=True)
    record=dict(schema='D16_Muon_own_Adam500_execution_v1',status='running',phase='formal',validation_only=False,
        command=[sys.executable,*sys.argv],pid=os.getpid(),start_ticks=b.executor.proc_start(os.getpid()),started_unix=time.time(),
        source_binding=p['source_binding'],controls=p['controls'],contract=b.CONTRACT,prepared_sha256=b.sha(b.PREPARED),
        source_sha256=p['source_sha256'],original_stage=asdict(stage),original_stage_exact=True,short_evidence=proof,
        source_raw_checkpoint_id=raw['raw_checkpoint_id'],source_raw_report=str(b.OUTPUT/'formal/report.json'),
        source_raw_report_sha256=b.sha(b.OUTPUT/'formal/report.json'),optimizer_state_fresh=True,
        effective_optimizer=p['original_recovery_optimizer'],runtime_identity=identity)
    path=output/'report.json';b.write(path,record)
    try:
        b.recovery.native.configure_deterministic_mode(cat.deterministic)
        model,meta,loaded,budget=b.checkpoint.load(raw['raw_model'])
        b.require(meta['checkpoint_id']==raw['raw_checkpoint_id'] and not loaded.missing_keys and not loaded.unexpected_keys,
            'Strictly load this candidate own raw state; do not borrow any control recovery state.')
        model,evidence,probe=flow.recover(model,stage,vae,hf,output,Path(raw['raw_model']),meta,short=False)
        b.require(evidence['actual_steps']==500 and evidence['frozen_persistent_fields_exact']
            and evidence['actual_inventory']['decoder_leaves']==1728 and evidence['actual_inventory']['lora_leaves']==0,
            'Complete original all252 D-only FP32 masters500 and preserve its actual frozen payload.')
        record['recovery']=evidence
        provenance=dict(schema='D16_Muon_own_Adam500_provenance_v1',phase='formal',validation_only=False,
            source_binding=p['source_binding'],controls=p['controls'],contract=b.CONTRACT,
            source_raw_checkpoint_id=raw['raw_checkpoint_id'],actual_category='down_proj',actual_recovery_updates=500,
            actual_VAE_updates_per_stage=10000,effective_optimizer=p['original_recovery_optimizer'])
        warmup=warm(model,216);expected=logits(model,probe)
        saved,budget,payload=b.checkpoint.save(model,output/'final_model',provenance=provenance)
        b.require(all(budget[key]==raw['raw_budget'][key] for key in
            ('user_linear_bits','packed_linear_bits','original_linear_weights','full_model_tensor_bytes')),
            'Original decoder-only recovery must preserve the complete actual D16 storage inventory.')
        evidence_path=output/'same_input_deployment.pt'
        b.torch.save(dict(checkpoint_id=saved['checkpoint_id'],deployment_digest=b.c.digest(payload),inputs=probe,logits=expected),evidence_path)
        record.update(final_model=str(output/'final_model'),final_checkpoint_id=saved['checkpoint_id'],final_budget=budget,
            deployment_warmup=warmup,expected_deployment_digest=b.c.digest(payload))
        del model,payload,expected,probe
        gc.collect();b.torch.cuda.empty_cache()
        subprocess.run([sys.executable,'-B','-u','-m','experiments.down_d16_vae_muon_cmcc_20261103.run',
            '--fresh',str(output/'final_model'),'--evidence',str(evidence_path)],cwd=b.REPO,check=True)
        strict=b.read(output/'strict_reload.json')
        b.require(strict['checkpoint_id']==saved['checkpoint_id'] and strict['same_input_logits_exact'],'Actual final UUID independent BF16 fresh failed.')
        record['strict_reload_report']=str(output/'strict_reload.json')
        b.check_pins(p['source_sha256'])
        record.update(status='completed_D16_Muon_own_original_Adam500_fresh',exit_code=0)
    except BaseException:
        record.update(status='failed',exit_code=1,error=traceback.format_exc());raise
    finally:
        record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'],
            peak_allocated_bytes=b.torch.cuda.max_memory_allocated(),peak_reserved_bytes=b.torch.cuda.max_memory_reserved())
        b.write(path,record)


if __name__=='__main__':
    b.torch.set_num_threads(8);b.torch.backends.cuda.matmul.allow_tf32=False;b.torch.backends.cudnn.allow_tf32=False
    execute()
