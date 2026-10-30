"""One fresh real-parent selected stage, independent short2 or honest formal32."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import selected_bindings as b
import selected_fit
from selected_native import release_cuda


def short_evidence(path, prepared):
    path = Path(path).resolve()
    report, audit = b.read(path), b.read(path.parent / 'native_audit.json')
    stage = prepared['stage']
    training = report['training']
    b.require(path.name == 'report.json' and report['status'] == 'aligned_selected_short2_complete'
        and report['stage'] == stage and report['phase'] == 'short2' and report['steps'] == report['planned_steps'] == 2
        and report['contract'] == b.CONTRACT and report['source_restored_exact']
        and training['terminal_native_equals_compact'] and training['teacher_unchanged']
        and training['trajectory'][1]['transition']['initialized_Adam_leaves'] == b.STAGES[stage]['decoder_tensors']
        and training['endpoint']['D_Adam_steps'] == [2]
        and training['endpoint']['U_Adam_steps'] == (2 if stage == 'J32' else None)
        and audit['status'] == 'verified' and audit['endpoint_report'] == b.binding(path)
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
        and (path.parent / 'run.exit').read_text().strip() == '0', 'Need genuine selected short2/nonemptyAdam/native export evidence.')
    old = b.read(report['prepared_path'])
    code = lambda p: {key: item for key, item in p['bindings'].items() if Path(key).suffix in ('.py', '.sh')}
    b.require(code(old) == code(prepared), 'Short2 used a different implementation.')
    if report['profile'] == b.PROFILE:
        b.require(report['prepared_sha256'] == b.sha256(b.stage_dir(stage) / 'prepared.json')
            and report['source_checkpoint_id'] == prepared['source_checkpoint_id'], 'Same-profile short source changed.')
        reuse = 'same_profile_exact_preparation'
    else:
        b.require(b.PROFILE == 'C' and report['profile'] == 'G'
            and path == b.PARENT.parent / 'G' / b.STAGES[stage]['directory'] / 'short2/report.json'
            and old['materials'] == prepared['materials'], 'Only explicit same-stage G-to-C wiring reuse is allowed.')
        for field in ('decoder_parameters', 'code_parameters'):
            before, after = old['scope'][field], prepared['scope'][field]
            b.require(set(before) == set(after) and all(len(before[key]['shape']) == len(value['shape'])
                and all(a >= c for a, c in zip(before[key]['shape'], value['shape'], strict=True)) for key, value in after.items()),
                'Actual C parent geometry is not covered by G; run necessary local short2.')
        reuse = 'G_larger_geometry_wiring_only_actual_C_parent_and04_R_independently_bound'
    return dict(report=b.binding(path), native_audit=b.binding(path.parent / 'native_audit.json'), reuse=reuse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=tuple(b.STAGES), required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--phase', choices=('short2', 'formal'), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    parser.add_argument('--short-evidence', type=Path)
    args = parser.parse_args()
    prepared = b.read_prepared(args.stage, args.prepared)
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and Path.cwd() == b.SNAPSHOT
        and os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu) and b.torch.cuda.device_count() == 1
        and args.timeout_seconds > 0, 'Use bitvae/snapshot cwd/exactly one assigned GPU.')
    validation = None
    if args.phase == 'formal':
        validation = short_evidence(args.short_evidence or b.stage_dir(args.stage) / 'short2/report.json', prepared)
    else:
        b.require(args.short_evidence is None, 'Independent short2 starts fresh.')
    run = b.stage_dir(args.stage) / args.phase
    b.require(not run.exists(), 'Never overwrite or automatically retry a selected-stage trajectory.')
    run.mkdir(parents=True)
    record = b.runtime_record(prepared, args.phase, run)
    b.torch.set_num_threads(4)
    b.torch.manual_seed(137)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    b.torch.set_float32_matmul_precision('highest')
    started = time.monotonic()
    observe = lambda event, values: print(json.dumps(dict(event=event, elapsed_seconds=time.monotonic()-started, **values)), flush=True)
    identity = dict(profile=b.PROFILE, stage=args.stage, phase=args.phase, seed=137, physical_GPU=args.gpu,
        source_checkpoint_id=prepared['source_checkpoint_id'], R_source_checkpoint_id=prepared['R_source_checkpoint_id'],
        steps=record['steps'], planned_steps=record['planned_steps'], contract=b.CONTRACT,
        prepared_path=str(args.prepared.resolve()), prepared_sha256=b.sha256(args.prepared), validation_evidence=validation)
    try:
        recovery, archive, materials = b.inputs.load_materials()
        b.require(materials == prepared['materials'], 'Actual train materials changed.')
        r_directory, r_meta, _, r_audit_path = b.source04()
        targets, R_record = b.retention.load_native(archive, materials, directory=r_directory, meta=r_meta,
            source_audit=b.binding(r_audit_path), cache_dir=b.R_CACHE, profile=b.PROFILE, schema=b.R_SCHEMA)
        b.require(R_record['source_initial_state'] == prepared['R_source_initial_state'], 'Actual04 initial retention state changed.')
        model, teacher, load_audit = b.pair_loader.load_pair(record, observe)
        b.require(b.core.state_proof(model) == prepared['source_initial_state'], 'Student differs from its complete actual parent.')
        tokenizer = b.core.build_tokenizer(record['model_config']['model']['hf_path'])
        training = selected_fit.fit(model, teacher, tokenizer, recovery, archive, targets, record, observe, started+args.timeout_seconds)
        del model, teacher
        release_cuda()
        b.require(b.torch.cuda.memory_allocated() == 0, 'Release actual student/full-F before native export.')
        result = dict(**identity, status='aligned_selected_'+args.phase+'_complete', training=training,
            source_restored_exact=True, load_audit=load_audit, R_cache=prepared['R_cache'],
            GPU_allocated_after_release=0, elapsed_seconds=time.monotonic()-started)
        b.core.write(run / 'report.json', result)
        (run / 'run.exit').write_text('0\n', encoding='utf-8')
        if args.phase == 'short2':
            import selected_export
            selected_export.export(run / 'report.json', run / 'native_final/final_model')
        return 0
    except BaseException as error:
        b.core.write(run / 'report.json', dict(**identity, status='failed', scientific_endpoint_valid=False,
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started))
        traceback.print_exc()
        (run / 'run.exit').write_text('1\n', encoding='utf-8')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
