"""Independent endpoint-only strict mixed state and same-input BF16 readback."""
import argparse
import json
from pathlib import Path
import time
import traceback

from . import bindings as b
from . import export, fit, runtime
from .run_short import check_resources, probe

torch = b.torch


def comparison(reference, candidate):
    b.require(reference.shape == candidate.shape and reference.dtype == candidate.dtype == torch.bfloat16
        and bool(torch.isfinite(reference).all()) and bool(torch.isfinite(candidate).all()),
        'Compare complete finite BF16 logits with the same shape and deployment path.')
    difference = candidate.float()-reference.float()
    norm = float(reference.double().norm())
    return dict(exact=torch.equal(reference, candidate),
        relative_L2=float(difference.double().norm())/norm if norm else None,
        max_abs=float(difference.abs().max()), changed_elements=int((reference != candidate).sum()),
        changed_top1_positions=int((reference.argmax(-1) != candidate.argmax(-1)).sum()), finite=True)


@torch.no_grad()
def audit(directory):
    started = time.monotonic()
    directory = Path(directory).resolve(strict=True)
    report = b.read(directory/'report.json')
    b.require((directory/'run.exit').read_text(encoding='utf-8').strip() == '0'
        and report['schema'] == 'BR_adjacent_blocks_short2_v1' and report['exit_code'] == 0
        and report['status'] == 'training_and_BF16_export_complete_fresh_pending',
        'Require an actually completed adjacent-block two-update export.')
    b.require(report['physical_GPUs'] == [6] and report['logical_device'] == 'cuda:0'
        and report['layers'] == report['completed_layers'] == [0, 1] and report['updates_per_layer'] == 2
        and report['group_indices'] == list(fit.GROUPS) and report['validation_only'] is True
        and report['prototype_only'] == report['source_binding']['prototype_only'],
        'Unexpected adjacent-block short source, identity, execution or resource binding.')
    b.require(len(report['layer_reports']) == 2
        and all(row['layer'] == index and row['training']['steps'] == 2
            and row['training']['group_indices'] == list(fit.GROUPS)
            and row['commit']['all_D_BF16_cast_exact'] is True
            and row['student_advance']['each_unique_entry_advanced_once'] is True
            and row['teacher_advance']['each_unique_entry_advanced_once'] is True
            for index, row in enumerate(report['layer_reports']))
        and report['layer_reports'][1]['stream_transition']['previous_student_BF16_output_equals_current_student_input'] is True
        and report['layer_reports'][1]['stream_transition']['previous_teacher_output_equals_current_teacher_input'] is True
        and report['teacher_frozen_exact'] is True,
        'The source report must contain the actual two-block training, BF16 commits and independent stream transitions.')
    for path, expected in report['source_sha256'].items():
        b.require(b.sha(path) == expected, 'Bound implementation changed before fresh readback: ' + path)
    final = directory/'final_model'
    b.require(Path(report['export']['directory']).resolve(strict=True) == final,
              'Fresh readback must consume this short run\'s own single endpoint.')
    for key, filename in (('expected_state', 'expected_state.json'), ('fixture', 'same_input_output.pt')):
        binding = report['export'][key]
        b.require(Path(binding['path']).resolve(strict=True) == directory/filename
            and b.file_binding(directory/filename) == binding, 'Changed or external fresh fixture: ' + key)
    # This loader reads only the new full endpoint; no parent weights or teacher are needed.
    model, meta, loaded, budget = b.checkpoint.load(final)
    provenance = meta['provenance']
    b.require(meta['checkpoint_id'] == report['export']['checkpoint_id']
        and meta['checkpoint_kind'] == 'mixed_final_model' and budget == report['export']['bit_budget']
        and provenance['schema'] == 'BR_adjacent_blocks_short2_export_v1'
        and provenance['execution'] == {key: report[key] for key in provenance['execution']}
        and provenance['source_binding'] == report['source_binding']
        and provenance['source_audit'] == report['source_audit']
        and provenance['completed_layers'] == [0, 1] and provenance['updates_per_layer'] == 2
        and provenance['group_indices'] == list(fit.GROUPS) and provenance['schedule_steps'] == 500
        and provenance['trained_decoder_names'] == report['trained_decoder_names']
        and provenance['source_frozen_complement'] == report['source_frozen_complement']
        and provenance['validation_only'] is True and provenance['prototype_only'] == report['prototype_only'],
        'Fresh endpoint identity, actual ledger or BR provenance changed.')
    expected = b.read(directory/'expected_state.json')
    actual = fit.state_records(model)
    b.require(not loaded.missing_keys and not loaded.unexpected_keys and actual == expected,
              'The entire independent persistent mixed state did not reload strictly and exactly.')
    allowed = set(report['trained_decoder_names'])
    b.require(allowed == {name for layer in (0, 1) for name in runtime.decoder_parameters(model, layer)}
        and fit.proof_from_records(actual, allowed) == report['source_frozen_complement'],
        'The fresh frozen code, route, protection, rotation, norm, backbone or other-block complement changed.')
    groups = export.deployment_contract(model)
    storage = export.budget_contract(budget, report['source_audit']['post_conversion_budget'])
    fixture = torch.load(directory/'same_input_output.pt', map_location='cpu', weights_only=True)
    inputs = {name: fixture[name] for name in ('input_ids', 'attention_mask', 'position_ids') if name in fixture}
    b.require(fixture['schema'] == 'BR_BF16_same_input_v1'
        and set(inputs) == {'input_ids', 'attention_mask', 'position_ids'}
        and all(value.dtype == torch.int64 for value in inputs.values())
        and fixture['input_ids'].ndim == 2 and fixture['input_ids'].shape[0] == 1
        and all(value.shape == fixture['input_ids'].shape for value in inputs.values())
        and b.digest(inputs) == fixture['input_signature']
        and fixture['logits'].shape == (1, 1, model.config.vocab_size),
        'Unexpected actual complete same-input BF16 readback fixture.')
    warm = runtime.prepare_full_deployment(model, 'cuda:0')
    logits = probe(model, fixture)
    compared = comparison(fixture['logits'], logits)
    b.write(directory/'fresh_comparison.json', dict(status='verified' if compared['exact'] else 'failed',
        BF16_deployed_to_fresh=compared,
        interpretation='Identical committed BF16 mixed endpoint, same group8 deployment path and complete input.'))
    b.require(compared['exact'], 'Independent same-path BF16 logits differ; see fresh_comparison.json.')
    runtime.offload_full_deployment(model)
    b.require(fit.state_records(model) == expected,
              'Full deployment prewarm or same-input readout changed a persistent saved tensor.')
    result = dict(schema='BR_adjacent_blocks_short2_fresh_audit_v1', status='verified', validation_only=True,
        prototype_only=report['prototype_only'], source_checkpoint_id=report['source_binding']['checkpoint_id'],
        checkpoint_id=meta['checkpoint_id'], completed_layers=[0, 1], updates_per_layer=2,
        serialized_tensors_compared=len(expected), fresh_strict_tensors_compared=len(actual),
        strict_missing_keys=list(loaded.missing_keys), strict_unexpected_keys=list(loaded.unexpected_keys),
        full_state_exact=True, frozen_complement_exact=True, source_frozen_complement=report['source_frozen_complement'],
        same_complete_input_BF16_logits_exact=True, BF16_deployed_to_fresh=compared, finite=True,
        all_deployment_parameters_BF16=True, FP32_RoPE_preserved=True, groups=groups,
        storage=storage, bit_budget=budget, warmup=warm, physical_GPUs=[6],
        input_tokens=fixture['input_ids'].shape[-1], source_case_id=fixture['source_case_id'],
        input_signature=fixture['input_signature'], output=fit.tensor_record(logits),
        training_report=b.file_binding(directory/'report.json'), export_meta=b.file_binding(final/b.checkpoint.META_FILE),
        expected_state=b.file_binding(directory/'expected_state.json'), fixture=b.file_binding(directory/'same_input_output.pt'),
        source_sha256=report['source_sha256'], single_effective_state=str(final/meta['state_dict_file']),
        parent_weights_loaded=False, teacher_loaded=False, elapsed_seconds=time.monotonic()-started,
        scope='Adjacent-block mixed BF16 lifecycle only; no full36-block BR, quality or exact-resume proof.')
    b.write(directory/'fresh_audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    check_resources()
    args.directory = args.directory.resolve(strict=True)
    b.require(not (args.directory/'fresh_audit.json').exists() and not (args.directory/'fresh.exit').exists(),
              'Never overwrite an independent fresh result.')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    try:
        result = audit(args.directory)
    except BaseException as error:
        b.write(args.directory/'fresh_failure.json', dict(error=str(error), traceback=traceback.format_exc()))
        (args.directory/'fresh.exit').write_text('1\n', encoding='utf-8')
        raise
    (args.directory/'fresh.exit').write_text('0\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], checkpoint_id=result['checkpoint_id'],
        audit=str(args.directory/'fresh_audit.json'), elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
