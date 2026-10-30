"""One fresh, explicitly assigned short2 or formal32 nativegreedy OPD trajectory."""
import argparse
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

from bindings import OUTPUT as OUT, SNAPSHOT as S, PYTHON, PROFILE, core, qa, torch, selected_modules, fit_core, engine, hot
from bindings import load_inputs, read_prepared, STEPS, CONTRACT
import fit
import rollout
from native import load_student


@torch.no_grad()
def load_pair(record, observe):
    # Same established native loader/restoration chain as old OPD, without a new whole-model hash gate.
    teacher, teacher_audit = qa.load_model(record['teacher_identity']['carrier_model_config'], logging.getLogger(__name__))
    policy = core.native_policy(teacher)
    restoration = engine.restore_targets(teacher, record['model_config']['model']['hf_path'],
        record['teacher_projection_names'])
    core.require(restoration['selected_count'] == 252
        and not any(isinstance(value, core.VAELinear) for value in teacher.modules()), 'F teacher must restore all252.')
    teacher.eval().requires_grad_(False)
    rollout.move_teacher(teacher, 'cpu', policy)
    gc.collect(); torch.cuda.empty_cache(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()
    core.require(torch.cuda.memory_allocated() == 0, 'Teacher left CUDA state before fresh student loading.')
    observe('teacher_on_CPU_before_student_load', {'allocated_bytes': 0, 'restored_projections': 252})
    student, student_audit = load_student(record['model_config'])
    core.require(core.native_policy(student) == policy, 'Fresh student/teacher FA2/FP32 RoPE policy differs.')
    return student, teacher, {'teacher_load_audit': teacher_audit, 'student_load_audit': student_audit,
        'teacher_restoration': restoration, 'native_policy': policy,
        'teacher_GPU_allocated_during_student_load': 0, 'teacher_retained_on_CPU_until_rollout_complete': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--phase', choices=tuple(STEPS), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    args = parser.parse_args()
    prepared, record = read_prepared(args.prepared.resolve())
    core.require(Path(sys.executable).resolve() == PYTHON.resolve() and Path.cwd() == S
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu) and torch.cuda.device_count() == 1
        and args.timeout_seconds > 0, 'Use bitvae/source cwd and exactly the GPU newly assigned by root.')
    if args.phase == 'formal32':
        short_path = OUT / 'short2/report.json'
        core.require(short_path.exists() and core.read(short_path)['status'] == 'aligned_nativegreedy_OPD_short2_complete'
            and (OUT / 'short2/run.exit').read_text().strip() == '0'
            and core.read(short_path)['prepared_sha256'] == core.sha256(args.prepared),
            'Fresh formal32 requires this implementation’s successful complete short2 validation.')
    run = OUT / args.phase
    core.require(not run.exists(), 'Do not overwrite or automatically retry an OPD execution.')
    run.mkdir()
    qa.check_config(record['model_config'])
    torch.set_num_threads(4); torch.manual_seed(137)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    started = time.monotonic()
    observe = lambda event, values: print(json.dumps({'event': event,
        'elapsed_seconds': time.monotonic()-started, **values}), flush=True)
    identity = {'source_checkpoint_id': record['source_checkpoint_id'], 'profile': PROFILE, 'phase': args.phase, 'physical_GPU': args.gpu,
        'seed': 137, 'group_indices': STEPS[args.phase], 'prepared_path': str(args.prepared.resolve()),
        'prepared_sha256': core.sha256(args.prepared), 'implementation': prepared['bindings'],
        'dataset_binding': prepared['dataset_binding'], 'actual_target_contract': CONTRACT,
        'teacher_kind': 'full7', 'student_scope': 'all', 'retention_lambda': 0., 'mmlu_execution_batch_size': 32}
    try:
        dataset, recovery, cache, target_evidence = load_inputs()
        # The immutable QA TF cache was verified for provenance; OPD never trains on its old answers.
        recovery = {key: (item if item['case']['domain'] == 'mmlu'
            else {k: v for k, v in item.items() if k != 'target_log_probs'}) for key, item in recovery.items()}
        model, teacher, load_audit = load_pair(record, observe)
        tokenizer = core.build_tokenizer(record['model_config']['model']['hf_path'])
        training, initial = fit.fit(model, teacher, tokenizer, recovery, record, args.phase, observe,
            started+args.timeout_seconds)
        del teacher
        gc.collect(); torch.cuda.empty_cache(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()
        with torch.no_grad():
            modules = selected_modules(model, record)
            training['deployment'] = core.deployment_mode(model, modules)
            endpoint = fit_core.copy_parameters(modules)
            patch = run / 'decoder_endpoint.pt'
            torch.save({'schema': 'CG_aligned_online_OPD_endpoint_v1', **identity,
                'steps': len(STEPS[args.phase]), 'learning_rate': 1e-6, 'state': endpoint,
                'frozen_state': training['frozen_state_after']}, patch)
            saved = torch.load(patch, map_location='cpu', weights_only=True)
            core.require(hot.scale_backend._exact(saved['state'], endpoint)
                and len(endpoint) == 1512 and all(value.dtype == torch.bfloat16 for value in endpoint.values()),
                'Complete native BF16 decoder endpoint save/reload differs.')
            fit_core.restore(model, saved['state'])
            core.require(hot.scale_backend._exact(fit_core.copy_parameters(modules), endpoint),
                'Complete native saved endpoint install differs.')
            fit_core.restore(model, initial)
            core.require(hot.scale_backend._exact(fit_core.copy_parameters(modules), initial)
                and fit_core.frozen_proof(model, set(initial)) == training['frozen_state_before'],
                'Exact source decoders/frozen-state restoration failed.')
        result = {**identity, 'status': 'aligned_nativegreedy_OPD_'+args.phase+'_complete',
            'steps': len(STEPS[args.phase]), 'training': training, 'load_audit': load_audit,
            'teacher_cache': cache, 'unchanged_HOT_M_target_evidence': target_evidence,
            'endpoint': {'path': str(patch), 'sha256': core.sha256(patch), 'bytes': patch.stat().st_size,
                'tensors': 1512, 'save_reload_install_exact': True}, 'source_restored_exact': True,
            'source_restore_contract': 'all1512 source decoder tensors exact plus unchanged complete frozen complement',
            'elapsed_seconds': time.monotonic()-started}
        core.write(run / 'report.json', result)
        (run / 'run.exit').write_text('0\n', encoding='utf-8')
        return 0
    except BaseException as error:
        core.write(run / 'report.json', {**identity, 'status': 'failed', 'scientific_endpoint_valid': False,
            'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
            'elapsed_seconds': time.monotonic()-started})
        traceback.print_exc()
        (run / 'run.exit').write_text('1\n', encoding='utf-8')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
