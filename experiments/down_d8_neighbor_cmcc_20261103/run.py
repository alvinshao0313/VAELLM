"""Original live-down fitting, isolated masked consumer, actual save and fresh."""
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
from experiments.adaptive_class_20261030.run_pipeline import warm,logits,fresh as original_fresh


def verified_short():
    report=b.read(b.OUTPUT/'short/report.json')
    strict=b.read(b.OUTPUT/'short/strict_reload.json')
    prepared=b.prepared()
    b.require(report['status']=='completed_D8_new_path_short' and report['exit_code']==0
        and report['source_sha256']==prepared['source_sha256'] and report['contract']==b.CONTRACT
        and report['fit']['actual_updates_per_stage']==2 and report['fit']['all_frozen_prefix_fields_exact']
        and report['fit']['protection_and_rotation_fields_exact_to_control']
        and report['packed_codes']['all_packed_bits_roundtrip_exact'] and report['packed_codes']['high4_padding_zero']
        and report['decoder_numerics']['status']=='actual_D8_single_and_group8_masked_fused_vs_native_decomposed_verified'
        and strict['checkpoint_id']==report['raw_checkpoint_id'] and strict['same_input_logits_exact']
        and strict['effective_consumer']==b.consumer.contract() and strict['exit_code']==0,
        'Require the actual new-shape two-stage short, code/mask numerics, frozen complement and independent BF16 fresh.')
    return dict(report=str(b.OUTPUT/'short/report.json'),report_sha256=b.sha(b.OUTPUT/'short/report.json'),
        fresh=str(b.OUTPUT/'short/strict_reload.json'),fresh_sha256=b.sha(b.OUTPUT/'short/strict_reload.json'),
        checkpoint_id=report['raw_checkpoint_id'],elapsed_seconds=report['elapsed_seconds'],
        peak_allocated_bytes=report['peak_allocated_bytes'],peak_reserved_bytes=report['peak_reserved_bytes'])


def fresh(directory,evidence):
    b.consumer.require_meta(b.read(directory/b.checkpoint.META_FILE))
    original_fresh(directory,evidence)
    path=directory.parent/'strict_reload.json'
    record=b.read(path)
    record['effective_consumer']=b.consumer.contract()
    b.write(path,record)


def execute(phase):
    b.require(Path(sys.executable).resolve()==b.PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV')=='bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and os.environ.get('PYTHONHASHSEED')=='31'
        and os.environ.get('PYTORCH_CUDA_ALLOC_CONF')=='expandable_segments:True',
        'Use caller bitvae, only assigned GPU1 and the mature seed/allocator.')
    prepared=b.prepared()
    review=b.require_core_review()
    short=phase=='short'
    previous=None if short else verified_short()
    output=b.OUTPUT/phase
    b.require(not output.exists(),'Never overwrite or retry this fixed short/formal stage.')
    parsed,identity,baseline=b.recipe(output)
    output.mkdir(parents=True)
    record=dict(schema='D8_down_neighbor_execution_v1',status='running',phase=phase,validation_only=short,
        command=[sys.executable,*sys.argv],pid=os.getpid(),start_ticks=b.executor.proc_start(os.getpid()),started_unix=time.time(),
        source_binding=prepared['source_binding'],D16_control=prepared['D16_control'],contract=b.CONTRACT,
        source_sha256=prepared['source_sha256'],prepared_sha256=b.sha(b.PREPARED),core_review=review,
        settings=b.SETTINGS,runtime_identity=identity,effective_consumer=b.consumer.contract(),
        short_evidence=previous,predicted_budget=prepared['predicted_budget'])
    path=output/'report.json'
    b.write(path,record)
    try:
        b.recovery.native.configure_deterministic_mode(parsed[0].deterministic)
        activation,plan,restored=b.base.restore_cat_runtime_state(parsed[-1],access_token=parsed[1].access_token)
        b.require(restored==baseline,'Restore unchanged original legal train runtime.')
        model,meta,loaded,_budget=b.checkpoint.load(b.PARENT)
        b.require(meta['checkpoint_id']==b.CONTRACT['parent'] and not loaded.missing_keys and not loaded.unexpected_keys,
            'Fit only the original actual after-up parent, fresh for short/formal.')
        b.recovery.native.set_seed(31)
        record['fit']=fit.execute(model,parsed,activation,plan,arm='d8',output=output/'fit',short=short)
        record['packed_codes']=validation.codes(model)
        if short:
            record['decoder_numerics']=validation.decoder_numerics(model)
        warmup=warm(model,216)
        inputs=dict(input_ids=parsed[-1]['activation_runtime']['cache']['input_ids'][0].reshape(1,256).clone())
        expected=logits(model,inputs)
        provenance=dict(schema='D8_down_neighbor_provenance_v1',phase=phase,validation_only=short,
            source_binding=prepared['source_binding'],D16_control=prepared['D16_control'],actual_category='down_proj',
            actual_updates_per_stage=2 if short else 10000,recovery_updates=0,settings=b.SETTINGS,
            contract=b.CONTRACT,effective_consumer=b.consumer.contract(),source_sha256=prepared['source_sha256'])
        saved,budget,payload=b.checkpoint.save(model,output/'raw_model',provenance=provenance)
        # Keep the original native runtime identity explicit, and additionally
        # bind the effective consumer rather than claiming an unchanged decoder.
        saved['runtime']['effective_consumer']=b.consumer.contract()
        b.write(output/'raw_model'/b.checkpoint.META_FILE,saved)
        budget=b.checkpoint.ledger(saved,payload,output/'raw_model')
        b.consumer.require_meta(saved)
        predicted=prepared['predicted_budget']
        b.require(all(budget[key]==predicted[key] for key in
            ('user_linear_bits','packed_linear_bits','original_linear_weights','full_model_tensor_bytes')),
            'Measured all-component D8 budget disagrees with the actual-geometry expectation.')
        evidence=output/'same_input_deployment.pt'
        b.torch.save(dict(checkpoint_id=saved['checkpoint_id'],deployment_digest=b.c.digest(payload),inputs=inputs,logits=expected),evidence)
        record.update(raw_model=str(output/'raw_model'),raw_checkpoint_id=saved['checkpoint_id'],raw_budget=budget,
            deployment_warmup=warmup,expected_deployment_digest=b.c.digest(payload))
        del model,payload,expected
        gc.collect()
        b.torch.cuda.empty_cache()
        subprocess.run([sys.executable,'-B','-u','-m','experiments.down_d8_neighbor_cmcc_20261103.run',
            '--fresh',str(output/'raw_model'),'--evidence',str(evidence)],cwd=b.REPO,check=True)
        strict=b.read(output/'strict_reload.json')
        b.require(strict['checkpoint_id']==saved['checkpoint_id'] and strict['same_input_logits_exact']
            and strict['effective_consumer']==b.consumer.contract(),'Independent D8 effective-consumer fresh failed.')
        record['strict_reload_report']=str(output/'strict_reload.json')
        b.check_pins(prepared['source_sha256'])
        record.update(status='completed_D8_new_path_short' if short else 'completed_D8_formal_raw',exit_code=0)
    except BaseException:
        record.update(status='failed',exit_code=1,error=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'],
            peak_allocated_bytes=b.torch.cuda.max_memory_allocated(),peak_reserved_bytes=b.torch.cuda.max_memory_reserved())
        b.write(path,record)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('short','formal'))
    parser.add_argument('--fresh',type=Path)
    parser.add_argument('--evidence',type=Path)
    args=parser.parse_args()
    b.torch.set_num_threads(8)
    b.torch.backends.cuda.matmul.allow_tf32=False
    b.torch.backends.cudnn.allow_tf32=False
    if args.fresh:
        b.require(args.evidence is not None,'Fresh needs actual pre-save deployment evidence.')
        fresh(args.fresh,args.evidence)
    else:
        b.require(args.phase is not None,'Choose the actual short or formal phase.')
        execute(args.phase)
