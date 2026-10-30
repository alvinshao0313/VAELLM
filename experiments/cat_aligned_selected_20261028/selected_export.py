"""Export only an actual selected terminal2/32, using frozen03's native2.6/rank0 audit."""
import argparse
from pathlib import Path
import selected_bindings as b
from selected_state import terminal_steps
from train_utils.checkpoint_v6 import refresh_vae_linear_runtime_after_state_load
from train_utils.v6_model_loader import load_v6_model_checkpoint
from transformers import AutoTokenizer


@b.torch.no_grad()
def export(report_path, output):
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    report = b.read(report_path)
    stage, phase = report['stage'], report['phase']
    prepared = b.read_prepared(stage, report['prepared_path'])
    terminal_steps(phase, report['steps'], report['planned_steps'])
    run = b.stage_dir(stage) / phase
    expected_output = run / 'native_final/final_model' if phase == 'short2' else b.stage_dir(stage) / 'native_final/final_model'
    indices = [19, 0] if phase == 'short2' else list(range(32))
    r_indices = [17, 28] if phase == 'short2' else list(range(32))
    training = report['training']
    b.require(report_path == run / 'report.json' and output == expected_output
        and report['status'] == 'aligned_selected_'+phase+'_complete' and report['profile'] == b.PROFILE
        and report['source_checkpoint_id'] == prepared['source_checkpoint_id'] and report['R_source_checkpoint_id'] == prepared['R_source_checkpoint_id']
        and report['prepared_sha256'] == b.sha256(report['prepared_path']) and report['contract'] == b.CONTRACT
        and (run / 'run.exit').read_text().strip() == '0' and report['source_restored_exact']
        and training['terminal_native_equals_compact'] and training['teacher_unchanged']
        and training['frozen_state_before'] == training['frozen_state_after'] == prepared['scope']['frozen']
        and training['group_indices'] == indices and training['R_group_indices'] == r_indices
        and training['student_FW_BW'] == 16*report['steps'] and training['new_observation_FW'] == 0,
        'Only this profile actual fresh terminal2/32, not an old observation/128 artifact, may be exported.')
    endpoint = training['endpoint']
    path = run / 'selected_endpoint.pt'
    b.require(b.binding(path) == {k: endpoint[k] for k in ('path', 'bytes', 'sha256')}
        and endpoint['save_reload_exact'] and endpoint['live_D_and_U_optimization_unchanged']
        and endpoint['D_Adam_steps'] == [report['steps']]
        and endpoint['U_Adam_steps'] == (report['steps'] if stage == 'J32' else None), 'Changed actual continuous optimizer endpoint.')
    payload = b.torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    b.require(payload['schema'] == 'CG_aligned_selected_endpoint_v1' and payload['profile'] == b.PROFILE
        and payload['stage'] == stage and payload['phase'] == phase and payload['steps'] == payload['planned_steps'] == report['steps']
        and payload['contract'] == b.CONTRACT and payload['source_checkpoint_id'] == prepared['source_checkpoint_id']
        and payload['R_source_checkpoint_id'] == prepared['R_source_checkpoint_id'] and payload['prepared_sha256'] == report['prepared_sha256']
        and payload['inference_only'] and not payload['exact_resume_claimed'], 'Wrong selected-stage compact provenance.')
    source, meta, source_audit, source_audit_path = b.source(stage)
    original = b.load_state(source, meta)
    scope = b.scope(original, meta, stage)
    b.require(scope == prepared['scope'] and payload['frozen_state'] == scope['frozen']
        and set(payload['state']) == set(scope['mutable']), 'Actual source-derived selected scope changed.')
    for key, value in payload['state'].items():
        b.require(value.device.type == 'cpu' and value.dtype == original[key].dtype and value.shape == original[key].shape
            and value.is_contiguous() and b.torch.isfinite(value).all().item(), 'Invalid native selected compact tensor: ' + key)
    expected = {**original, **payload['state']}
    b.require(b.mature.frozen_state(expected, set(scope['mutable'])) == scope['frozen'], 'Selected export changed the frozen complement.')
    model, loaded, result = load_v6_model_checkpoint(str(source), map_location='cpu', strict=True, expected_kind='final_model')
    b.require(loaded == meta and not result.missing_keys and not result.unexpected_keys, 'Strict actual selected-parent loading failed.')
    live = model.state_dict()
    for key, value in payload['state'].items():
        live[key].copy_(value)
    model.eval().requires_grad_(False)
    refresh_vae_linear_runtime_after_state_load(model)
    tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], use_fast=True, local_files_only=True)
    provenance = dict(parent_checkpoint_ids=[meta['checkpoint_id']], profile=b.PROFILE, stage=stage, phase=phase,
        source_checkpoint_id=meta['checkpoint_id'], source_checkpoint_dir=str(source), source_audit=b.binding(source_audit_path),
        R_source_checkpoint_id=prepared['R_source_checkpoint_id'], R_source_audit=prepared['R_source_audit'], R_cache=prepared['R_cache'],
        actual_steps=report['steps'], planned_steps=report['planned_steps'], contract=b.CONTRACT,
        selected_modules=scope['selected'], replaced_tensor_keys=scope['mutable'],
        input_evidence_sha256={str(report_path): b.sha256(report_path), str(path): b.sha256(path),
            report['prepared_path']: report['prepared_sha256']},
        purpose='Inference-only selected actual32 benchmark endpoint; no U/gW/optimizer/master/RNG, no exact-resume claim.')
    audit = b.deployment.save_and_verify(model, output, meta, expected, provenance, tokenizer)
    b.require(all(audit['bit_budget'][key] == source_audit['bit_budget'][key] for key in b.BUDGET_FIELDS),
        'Selected native logical/packed/component ledger differs from the actual parent.')
    audit.update(schema='CG_aligned_selected_native_audit_v1', stage=stage, phase=phase,
        source_checkpoint_id=meta['checkpoint_id'], R_source_checkpoint_id=prepared['R_source_checkpoint_id'],
        actual_steps=report['steps'], planned_steps=report['planned_steps'], endpoint_report=b.binding(report_path),
        frozen_tensors_compared=scope['frozen']['tensors'], mutable_tensors=len(scope['mutable']),
        parent_bit_ledger_equal=True, inference_only=True, exact_resume_claimed=False)
    b.core.write((run if phase == 'short2' else b.stage_dir(stage)) / 'native_audit.json', audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    b.torch.set_num_threads(4)
    print(export(args.report, args.output_dir), flush=True)
