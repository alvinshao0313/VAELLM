"""Fresh G09 Muon short2 or formal32; the source and loss remain unchanged."""
import argparse
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

import bindings as b
import fit
import motion

runtime = b.module(b.OPD / 'run.py', '_G09_Muon_original_pair_loader')


def check_short():
    path = b.OUTPUT / 'short2/native_audit.json'
    b.require(path.is_file(), 'Formal Muon32 requires the actual new Muon two-update native audit.')
    audit = b.read(path)
    report = b.read(b.OUTPUT / 'short2/report.json')
    b.require(audit['status'] == 'verified' and audit['phase'] == 'short2'
        and audit['source_checkpoint_id'] == b.SOURCE_ID
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
        and audit['frozen_tensors_compared'] == 1911 and audit['code_bank_count'] == 504
        and report['prepared_sha256'] == b.sha256(b.OUTPUT / 'prepared.json')
        and report['steps'] == 2 and report['training']['group_indices'] == [0,1]
        and report['source_restored_exact'] and report['saved_decoder_matches_final_FP32_cast_exact']
        and report['training']['trajectory'][1]['transition']['Muon_group_metadata_exact']
        and report['training']['trajectory'][1]['transition']['Muon_momentum_initialized_leaves'] == 504
        and report['training']['trajectory'][1]['transition']['auxiliary_Adam_initialized_leaves'] == 1008
        and (path.parent / 'run.exit').read_text().strip() == '0'
        and (b.OUTPUT / 'short2.exit').read_text().strip() == '0',
        'Actual Muon short2 is incomplete or bound to another implementation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=tuple(b.STEPS),required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--timeout-seconds',type=int,required=True)
    args = parser.parse_args()
    prepared,record = b.read_prepared()
    if args.phase == 'formal32': check_short()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and Path.cwd() == b.SNAPSHOT
        and args.gpu == 7 and os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
        and b.torch.cuda.device_count() == 1 and args.timeout_seconds > 0,
        'Use activated bitvae/snapshot cwd and the root-assigned GPU7.')
    run = b.OUTPUT / args.phase
    b.require(not run.exists(), 'Do not overwrite or automatically retry a Muon execution.')
    run.mkdir()
    b.torch.set_num_threads(4)
    b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    started = time.monotonic()
    observe = lambda event,values: print(json.dumps(dict(event=event,elapsed_seconds=time.monotonic()-started,**values)),flush=True)
    identity = dict(schema='G09_Muon_OPD_execution_v1',source_checkpoint_id=b.SOURCE_ID,profile='G',
        phase=args.phase,physical_GPU=7,seed=137,group_indices=b.STEPS[args.phase],
        prepared_path=str(b.OUTPUT / 'prepared.json'),prepared_sha256=b.sha256(b.OUTPUT / 'prepared.json'),
        contract=b.CONTRACT,implementation=prepared['bindings'],dataset_binding=prepared['dataset_binding'],
        pid=os.getpid(),argv=sys.argv,python=sys.executable,cwd=str(Path.cwd()))
    try:
        dataset,recovery,cache,evidence = b.load_inputs()
        b.require(sum('target_log_probs' in item for item in recovery.values()) == 1024,
            'Keep only the once-mixed full-F M targets; QA is current native OPD.')
        model,teacher,load_audit = runtime.load_pair(record,observe)
        specs = list(prepared['source_geometry'].values())
        reference_weights = motion.capture_source(model,specs)
        tokenizer = b.core.build_tokenizer(record['model_config']['model']['hf_path'])
        training,initial = fit.fit(model,teacher,tokenizer,recovery,record,args.phase,observe,started+args.timeout_seconds)
        del teacher
        gc.collect()
        b.torch.cuda.empty_cache()
        b.torch._C._cuda_clearCublasWorkspaces()
        b.torch.cuda.empty_cache()
        with b.torch.no_grad():
            modules = b.selected_modules(model,record)
            expected = {name:p.detach().cpu().bfloat16() for name,p in model.named_parameters() if name in initial}
            training['deployment'] = b.core.deployment_mode(model,modules)
            endpoint = b.fit_core.copy_parameters(modules)
            b.require(b.hot.scale_backend._exact(endpoint,expected), 'Native endpoint differs from final actual FP32 masters cast to BF16.')
            decoded_motion = motion.endpoint_motion(model,specs,reference_weights)
            patch = run / 'decoder_endpoint.pt'
            b.torch.save(dict(schema='G09_Muon_OPD_decoder_endpoint_v1',execution=identity,
                source_checkpoint_id=b.SOURCE_ID,steps=len(b.STEPS[args.phase]),state=endpoint,
                frozen_state=training['frozen_state_after']),patch)
            saved = b.torch.load(patch,map_location='cpu',weights_only=True)
            b.require(b.hot.scale_backend._exact(saved['state'],endpoint) and len(endpoint) == 1512
                and all(value.dtype == b.torch.bfloat16 for value in endpoint.values()), 'Saved BF16 endpoint differs.')
            b.fit_core.restore(model,saved['state'])
            b.require(b.hot.scale_backend._exact(b.fit_core.copy_parameters(modules),endpoint), 'Saved endpoint install differs.')
            b.fit_core.restore(model,initial)
            b.require(b.hot.scale_backend._exact(b.fit_core.copy_parameters(modules),initial)
                and b.fit_core.frozen_proof(model,set(initial)) == training['frozen_state_before'],
                'Exact G09 source/frozen-state rollback failed.')
        result = dict(**identity,status='G09_Muon_nativegreedy_OPD_'+args.phase+'_complete',
            steps=len(b.STEPS[args.phase]),training=training,load_audit=load_audit,teacher_cache=cache,
            unchanged_M_target_evidence=evidence,decoded_BF16_weight_motion=decoded_motion,
            saved_decoder_matches_final_FP32_cast_exact=True,source_restored_exact=True,
            endpoint=dict(path=str(patch),sha256=b.sha256(patch),bytes=patch.stat().st_size,
                tensors=1512,save_reload_install_exact=True),elapsed_seconds=time.monotonic()-started)
        b.core.write(run / 'report.json',result)
        (run / 'run.exit').write_text('0\n',encoding='utf-8')
        observe('G09_Muon_training_complete',dict(phase=args.phase,report=str(run / 'report.json')))
    except Exception as error:
        b.core.write(run / 'failure.json',dict(**identity,status='failed_G09_Muon',error=str(error),traceback=traceback.format_exc()))
        (run / 'run.exit').write_text('1\n',encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
