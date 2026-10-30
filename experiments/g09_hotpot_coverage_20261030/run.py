"""One fresh H192 Muon32 trajectory through the mature fit and native rollout."""
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

fit = b.module(b.MUON / 'fit.py', '_hotpot192_mature_Muon_fit')
sys.modules['fit'] = fit
motion = fit.motion
runtime = b.module(b.OPD / 'run.py', '_hotpot192_mature_pair_loader')


def supervision(training):
    rows = training['trajectory']
    result = {}
    for domain in ('qasper', 'hotpotqa'):
        generated = [item for row in rows for item in row['rollouts'] if item['domain'] == domain]
        targets = [item for row in rows for item in row['QA_documents'] if item['domain'] == domain]
        b.require([(item['case_id'], item['generated_tokens']) for item in generated]
            == [(item['case_id'], item['answer_tokens']) for item in targets], 'Actual OPD generation/target counts differ.')
        result[domain] = dict(document_presentations=len(generated),
            unique_documents=len({item['case_id'] for item in generated}),
            prompt_tokens=sum(item['prompt_tokens'] for item in generated),
            generated_supervision_tokens=sum(item['generated_tokens'] for item in generated),
            maximum_generated_tokens=max(item['generated_tokens'] for item in generated))
    result['total_generated_supervision_tokens'] = sum(result[key]['generated_supervision_tokens'] for key in ('qasper', 'hotpotqa'))
    result['reduction'] = 'Unchanged token mean per document, then document weight .5/8; generated totals are observation, not objective weights.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=tuple(b.STEPS), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    args = parser.parse_args()
    prepared, record = b.read_prepared()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and Path.cwd() == b.SNAPSHOT
        and args.gpu == 4 and os.environ.get('CUDA_VISIBLE_DEVICES') == '4'
        and b.torch.cuda.device_count() == 1 and args.timeout_seconds > 0,
        'Use activated bitvae, snapshot cwd and the assigned physical GPU4.')
    run = b.OUTPUT / args.phase
    b.require(not run.exists(), 'Do not overwrite or automatically retry coverage training.')
    run.mkdir()
    b.torch.set_num_threads(4)
    b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    started = time.monotonic()
    observe = lambda event, values: print(json.dumps(dict(event=event, elapsed_seconds=time.monotonic()-started, **values)), flush=True)
    identity = dict(schema='G09_Muon_OPD_execution_v1', experiment_schema='G09_Hotpot192_coverage_v1',
        source_checkpoint_id=b.SOURCE_ID, profile='G', phase=args.phase, physical_GPU=4, seed=137,
        group_indices=b.STEPS[args.phase], prepared_path=str(b.OUTPUT / 'prepared.json'),
        prepared_sha256=b.sha256(b.OUTPUT / 'prepared.json'), contract=b.CONTRACT,
        implementation=prepared['bindings'], dataset_binding=prepared['dataset_binding'],
        pid=os.getpid(), argv=sys.argv, python=sys.executable, cwd=str(Path.cwd()))
    try:
        _, recovery, cache, evidence = b.load_inputs()
        b.require(sum('target_log_probs' in value for value in recovery.values()) == 1024,
            'Reuse only unchanged mixed full-F M targets; all QA is online OPD.')
        model, teacher, load_audit = runtime.load_pair(record, observe)
        specs = list(prepared['source_geometry'].values())
        reference_weights = motion.capture_source(model, specs)
        tokenizer = b.core.build_tokenizer(record['model_config']['model']['hf_path'])
        training, initial = fit.fit(model, teacher, tokenizer, recovery, record, args.phase, observe, started+args.timeout_seconds)
        training['actual_OPD_supervision'] = supervision(training)
        observe('actual_OPD_supervision', training['actual_OPD_supervision'])
        del teacher
        gc.collect()
        b.torch.cuda.empty_cache()
        b.torch._C._cuda_clearCublasWorkspaces()
        b.torch.cuda.empty_cache()
        with b.torch.no_grad():
            modules = b.selected_modules(model, record)
            expected = {name:p.detach().cpu().bfloat16() for name,p in model.named_parameters() if name in initial}
            training['deployment'] = b.core.deployment_mode(model, modules)
            endpoint = b.fit_core.copy_parameters(modules)
            b.require(b.hot.scale_backend._exact(endpoint, expected), 'Final FP32 masters/BF16 native endpoint differ.')
            decoded_motion = motion.endpoint_motion(model, specs, reference_weights)
            patch = run / 'decoder_endpoint.pt'
            b.torch.save(dict(schema='G09_Muon_OPD_decoder_endpoint_v1', execution=identity,
                source_checkpoint_id=b.SOURCE_ID, steps=32, state=endpoint,
                frozen_state=training['frozen_state_after']), patch)
            saved = b.torch.load(patch, map_location='cpu', weights_only=True)
            b.require(b.hot.scale_backend._exact(saved['state'], endpoint) and len(endpoint) == 1512
                and all(value.dtype == b.torch.bfloat16 for value in endpoint.values()), 'Saved BF16 endpoint differs.')
            b.fit_core.restore(model, saved['state'])
            b.require(b.hot.scale_backend._exact(b.fit_core.copy_parameters(modules), endpoint), 'Endpoint reinstall differs.')
            b.fit_core.restore(model, initial)
            b.require(b.hot.scale_backend._exact(b.fit_core.copy_parameters(modules), initial)
                and b.fit_core.frozen_proof(model, set(initial)) == training['frozen_state_before'], 'G09 exact rollback failed.')
        result = dict(**identity, status='G09_Muon_nativegreedy_OPD_formal32_complete', steps=32,
            training=training, load_audit=load_audit, teacher_cache=cache, unchanged_M_target_evidence=evidence,
            decoded_BF16_weight_motion=decoded_motion, saved_decoder_matches_final_FP32_cast_exact=True,
            source_restored_exact=True, endpoint=dict(path=str(patch), sha256=b.sha256(patch),
                bytes=patch.stat().st_size, tensors=1512, save_reload_install_exact=True),
            elapsed_seconds=time.monotonic()-started)
        b.core.write(run / 'report.json', result)
        (run / 'run.exit').write_text('0\n', encoding='utf-8')
        observe('G09_Hotpot192_training_complete', dict(report=str(run / 'report.json')))
    except Exception as error:
        b.core.write(run / 'failure.json', dict(**identity, status='failed_G09_Hotpot192', error=str(error), traceback=traceback.format_exc()))
        (run / 'run.exit').write_text('1\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
