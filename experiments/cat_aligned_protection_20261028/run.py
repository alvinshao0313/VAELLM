"""One fresh assigned nativegreedy D/protection short2 or actual32/64 formal trajectory."""
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
import data
import fit
import rollout
from native import load_student
from snapshot import native_roundtrip


@b.torch.no_grad()
def load_pair(record, observe):
    teacher, teacher_audit = b.qa.load_model(record['teacher_identity']['carrier_model_config'], logging.getLogger(__name__))
    policy = b.core.native_policy(teacher)
    restoration = b.engine.restore_targets(teacher, record['model_config']['model']['hf_path'], record['teacher_projection_names'])
    b.require(restoration['selected_count'] == 252 and not any(isinstance(m, b.core.VAELinear) for m in teacher.modules()),
        'F must restore all252 original HF projections.')
    teacher.eval().requires_grad_(False)
    rollout.move_teacher(teacher, 'cpu', policy)
    gc.collect(); b.torch.cuda.empty_cache(); b.torch._C._cuda_clearCublasWorkspaces(); b.torch.cuda.empty_cache()
    b.require(b.torch.cuda.memory_allocated() == 0, 'Teacher must be fully onCPU before native student load.')
    observe('full_F_on_CPU_before_student', dict(restored_projections=252, GPU_allocated_bytes=0))
    student, student_audit = load_student(record['model_config'])
    b.require(b.core.native_policy(student) == policy, 'Student/F FA2/FP32RoPE policy differs.')
    return student, teacher, dict(teacher_load_audit=teacher_audit, student_load_audit=student_audit,
        teacher_restoration=restoration, native_policy=policy)


def short_evidence(path, prepared, arm):
    path = Path(path).resolve()
    report, audit = b.read(path), b.read(path.parent / 'native_audit.json')
    b.require(path.name == 'report.json' and report['status'] == 'aligned_protection_short2_complete'
        and report['phase'] == 'short2' and report['arm'] == arm and report['steps'] == 2
        and (path.parent / 'run.exit').read_text().strip() == '0' and report['contract'] == b.CONTRACT
        and report['source_restored_exact'] and report['training']['teacher_unchanged']
        and report['training']['endpoint']['before']['initialized_leaves'] == 1764
        and report['training']['endpoint']['before']['adam_steps'] == [2]
        and report['native_roundtrip']['native_endpoint_installed_exact']
        and audit['status'] == 'verified' and audit['endpoint_report'] == b.binding(path)
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423,
        'Formal execution requires actual complete short2/nonemptyAdam/native strict export evidence for this arm.')
    previous = b.read(report['prepared_path'])
    code = lambda p: {k: v for k, v in p['bindings'].items() if Path(k).suffix in ('.py', '.sh')}
    b.require(code(previous) == code(prepared), 'Short evidence uses different implementation.')
    if report['profile'] == b.PROFILE:
        b.require(report['prepared_sha256'] == b.sha256(b.OUTPUT / 'prepared.json')
            and report['source_checkpoint_id'] == prepared['source_checkpoint_id'], 'Same-profile source changed since short2.')
        reuse = 'same_profile_exact_preparation'
    else:
        # G has the larger new decoder geometry. Reuse its wiring evidence only when actual C shapes are covered.
        b.require(b.PROFILE == 'C' and report['profile'] == 'G'
            and path == b.PARENT.parent / 'G/03_protection' / arm / 'short2/report.json', 'Only explicit G-to-C wiring reuse is supported.')
        old, current = previous['records'][arm], prepared['records'][arm]
        for field in ('decoder_parameters', 'protection_parameters'):
            b.require(set(old[field]) == set(current[field]) and all(len(old[field][key]['shape']) == len(item['shape'])
                and all(a >= c for a, c in zip(old[field][key]['shape'], item['shape'], strict=True)) for key, item in current[field].items()),
                'Actual C geometry is not covered by G short2; run a necessary local short2.')
        b.require(old['protection_kind'] == current['protection_kind'] and prepared['materials'] == previous['materials'],
            'Cross-profile wiring reuse changed method/input protocol.')
        reuse = 'explicit_G_larger_geometry_to_C_wiring_only_actual_C_source_independently_audited'
    return dict(report=b.binding(path), native_audit=b.binding(path.parent / 'native_audit.json'), reuse= reuse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(b.ARMS), required=True)
    parser.add_argument('--phase', choices=('short2', 'formal'), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    parser.add_argument('--short-evidence', type=Path, help='Explicit same-arm G short2 when its actual geometry covers C.')
    args = parser.parse_args()
    prepared = b.read_prepared(args.prepared)
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and Path.cwd() == b.SNAPSHOT
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu) and b.torch.cuda.device_count() == 1
        and args.timeout_seconds > 0, 'Use bitvae/snapshot cwd/exactly one assigned GPU.')
    validation = None
    if args.phase == 'formal':
        validation = short_evidence(args.short_evidence or b.OUTPUT / args.arm / 'short2/report.json', prepared, args.arm)
    else:
        b.require(args.short_evidence is None, 'Short2 starts fresh and does not consume prior trajectory evidence.')
    run = b.OUTPUT / args.arm / args.phase
    b.require(not run.exists(), 'Never overwrite or automatically retry a protection trajectory.')
    run.mkdir(parents=True)
    record = data.record_for(prepared, args.arm, args.phase, run)
    b.torch.set_num_threads(4); b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    started = time.monotonic()
    observe = lambda event, values: print(json.dumps(dict(event=event, elapsed_seconds=time.monotonic()-started, **values)), flush=True)
    identity = dict(source_checkpoint_id=record['source_checkpoint_id'], profile=b.PROFILE, arm=args.arm, phase=args.phase,
        seed=137, physical_GPU=args.gpu, prepared_path=str(args.prepared.resolve()), prepared_sha256=b.sha256(args.prepared),
        steps=record['steps'], planned_steps=record['planned_steps'], group_indices=record['group_indices'],
        R_group_indices=record['R_group_indices'], contract=b.CONTRACT, implementation=prepared['bindings'],
        validation_evidence=validation, R_cache=prepared['R_cache'])
    result = dict(identity)
    try:
        recovery, retention, materials = data.load_materials()
        b.require(materials == prepared['materials'], 'Bound material values changed.')
        from retention_cache import load, CACHE
        b.require(prepared['R_cache'] == b.binding(CACHE / 'cache.json'), 'Profile initial-state R cache binding changed.')
        source_targets, cache = load(retention, materials)
        model, teacher, load_audit = load_pair(record, observe)
        source = b.core.state_proof(model)
        b.require(source == prepared['source_initial_state'] == cache['source_initial_state'], 'Student and R teacher must share complete actual02 initial state.')
        tokenizer = b.core.build_tokenizer(record['model_config']['model']['hf_path'])
        training, initial, initial_protection = fit.fit(model, teacher, tokenizer, recovery, retention, source_targets,
            record, observe, started+args.timeout_seconds)
        del teacher
        gc.collect(); b.torch.cuda.empty_cache(); b.torch._C._cuda_clearCublasWorkspaces(); b.torch.cuda.empty_cache()
        native = native_roundtrip(model, initial, initial_protection, training, record, source)
        result = dict(**identity, status='aligned_protection_'+args.phase+'_complete', training=training,
            native_roundtrip=native, source_restored_exact=True, load_audit=load_audit,
            R_cache_manifest=b.binding(CACHE / 'cache.json'), elapsed_seconds=time.monotonic()-started)
        b.core.write(run / 'report.json', result)
        (run / 'run.exit').write_text('0\n', encoding='utf-8')
        del model
        gc.collect(); b.torch.cuda.empty_cache(); b.torch._C._cuda_clearCublasWorkspaces(); b.torch.cuda.empty_cache()
        if args.phase == 'short2':
            # The short path covers the same independent native save/fresh-strict exit as the formal exporter.
            import export
            export.export(run / 'report.json', run / 'native_final/final_model')
        return 0
    except BaseException as error:
        # Preserve completed training/roundtrip evidence when only the export fails.
        result.update(status='failed', scientific_endpoint_valid=False, error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
        b.core.write(run / 'report.json', result)
        traceback.print_exc()
        (run / 'run.exit').write_text('1\n', encoding='utf-8')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
