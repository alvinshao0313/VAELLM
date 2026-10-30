"""One fresh G09 direct32 trajectory using the unchanged native OPD implementation."""
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

runtime = b.module(b.OPD / 'run.py', '_g09_reused_native_OPD_runtime')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    args = parser.parse_args()
    prepared, record = b.read_prepared()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and Path.cwd() == b.SNAPSHOT
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu) and b.torch.cuda.device_count() == 1
        and args.gpu == 2 and args.timeout_seconds > 0, 'Use bitvae/snapshot cwd and the assigned GPU2.')
    run = b.OUTPUT / 'formal32'
    b.require(not run.exists(), 'Do not overwrite or automatically retry direct OPD32.')
    run.mkdir()
    b.torch.set_num_threads(4)
    b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    started = time.monotonic()
    observe = lambda event, values: print(json.dumps(dict(event=event,
        elapsed_seconds=time.monotonic()-started, **values)), flush=True)
    identity = dict(schema='G09_direct_OPD_execution_v1', source_checkpoint_id=b.SOURCE_ID,
        profile='G', phase='formal32', physical_GPU=args.gpu, seed=137, group_indices=list(range(32)),
        prepared_path=str(b.OUTPUT / 'prepared.json'), prepared_sha256=b.sha256(b.OUTPUT / 'prepared.json'),
        implementation=prepared['bindings'], dataset_binding=prepared['dataset_binding'],
        actual_target_contract=b.CONTRACT, teacher_kind='full7', student_scope='all', retention_lambda=0.,
        mmlu_execution_batch_size=32, pid=os.getpid(), argv=sys.argv, python=sys.executable, cwd=str(Path.cwd()))
    b.core.write(b.OUTPUT / 'execution.json', dict(status='running_direct_OPD32', **identity))
    try:
        dataset, recovery, cache, evidence = b.load_inputs()
        b.require(sum('target_log_probs' in item for item in recovery.values()) == 1024,
            'Only once-mixed F-M targets may be retained; QA uses current live trajectories.')
        model, teacher, load_audit = runtime.load_pair(record, observe)
        tokenizer = b.core.build_tokenizer(record['model_config']['model']['hf_path'])
        training, initial = runtime.fit.fit(model, teacher, tokenizer, recovery, record, 'formal32', observe,
            started+args.timeout_seconds)
        del teacher
        gc.collect()
        b.torch.cuda.empty_cache()
        b.torch._C._cuda_clearCublasWorkspaces()
        b.torch.cuda.empty_cache()
        with b.torch.no_grad():
            modules = b.selected_modules(model, record)
            training['deployment'] = b.core.deployment_mode(model, modules)
            endpoint = b.fit_core.copy_parameters(modules)
            patch = run / 'decoder_endpoint.pt'
            b.torch.save(dict(schema='G09_direct_OPD_decoder_endpoint_v1', execution=identity,
                source_checkpoint_id=b.SOURCE_ID, steps=32, state=endpoint,
                frozen_state=training['frozen_state_after']), patch)
            saved = b.torch.load(patch, map_location='cpu', weights_only=True)
            b.require(b.hot.scale_backend._exact(saved['state'], endpoint) and len(endpoint) == 1512
                and all(value.dtype == b.torch.bfloat16 for value in endpoint.values()),
                'Complete native BF16 decoder endpoint save/reload differs.')
            b.fit_core.restore(model, saved['state'])
            b.require(b.hot.scale_backend._exact(b.fit_core.copy_parameters(modules), endpoint),
                'Saved native decoder endpoint install differs.')
            b.fit_core.restore(model, initial)
            b.require(b.hot.scale_backend._exact(b.fit_core.copy_parameters(modules), initial)
                and b.fit_core.frozen_proof(model, set(initial)) == training['frozen_state_before'],
                'Actual G09 decoder/frozen complement restoration failed.')
        result = dict(**identity, status='G09_direct_nativegreedy_OPD32_complete', steps=32,
            training=training, load_audit=load_audit, teacher_cache=cache,
            unchanged_HOT_M_target_evidence=evidence, source_restored_exact=True,
            endpoint=dict(path=str(patch), sha256=b.sha256(patch), bytes=patch.stat().st_size,
                tensors=1512, save_reload_install_exact=True), elapsed_seconds=time.monotonic()-started)
        b.core.write(run / 'report.json', result)
        b.core.write(b.OUTPUT / 'execution.json', dict(**identity,
            status='training_complete_native_export_pending', exit_code=0, report=str(run / 'report.json')))
        (run / 'run.exit').write_text('0\n', encoding='utf-8')
        observe('G09_direct_OPD32_training_complete', dict(report=str(run / 'report.json')))
    except Exception as error:
        b.core.write(run / 'failure.json', dict(**identity, status='failed_direct_OPD32',
            error=str(error), traceback=traceback.format_exc()))
        (run / 'run.exit').write_text('1\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
