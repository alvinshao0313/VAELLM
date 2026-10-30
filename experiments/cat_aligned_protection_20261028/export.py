"""Export actual terminal03a32/03b64 or its genuine short2 using native2.6/rank0."""
import argparse
from pathlib import Path
import bindings as b
from deployment import save_and_verify
from train_utils.checkpoint_v6 import refresh_vae_linear_runtime_after_state_load
from train_utils.v6_model_loader import load_v6_model_checkpoint
from transformers import AutoTokenizer


@b.torch.no_grad()
def export(report_path, output):
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    report = b.read(report_path)
    prepared = b.read_prepared(report['prepared_path'])
    arm, phase = report['arm'], report['phase']
    choice = b.ARMS[arm]
    record = prepared['records'][arm]
    run = b.OUTPUT / arm / phase
    expected_output = run / 'native_final/final_model' if phase == 'short2' else b.OUTPUT / arm / 'native_final/final_model'
    steps = 2 if phase == 'short2' else choice['steps']
    order = [0, 19] if phase == 'short2' else list(range(32)) * (steps//32)
    r_order = [17, 28] if phase == 'short2' else order
    b.require(phase in ('short2', 'formal') and report_path == run / 'report.json' and output == expected_output
        and report['status'] == 'aligned_protection_'+phase+'_complete' and report['steps'] == report['planned_steps'] == steps
        and report['source_checkpoint_id'] == prepared['source_checkpoint_id'] and report['profile'] == b.PROFILE
        and report['prepared_sha256'] == b.sha256(report['prepared_path']) and report['contract'] == b.CONTRACT
        and (run / 'run.exit').read_text().strip() == '0' and report['source_restored_exact']
        and report['native_roundtrip']['native_endpoint_installed_exact']
        and report['training']['frozen_state_before'] == report['training']['frozen_state_after'] == record['validated_frozen_state']
        and report['training']['group_indices'] == order and report['training']['R_group_indices'] == r_order,
        'Only actual same-profile fresh terminal2/32/64 may be exported; never masquerade32 as64.')
    endpoint = report['training']['endpoint']
    patch = Path(endpoint['path'])
    b.require(patch == run / 'protected_endpoint.pt' and b.sha256(patch) == endpoint['sha256']
        and endpoint['save_reload_exact'] and endpoint['before'] == endpoint['after']
        and endpoint['before']['initialized_leaves'] == 1764 and endpoint['before']['adam_steps'] == [steps], 'Changed actual native compact endpoint.')
    payload = b.torch.load(patch, map_location='cpu', weights_only=True)
    b.require(payload['schema'] == 'CG_aligned_protection_endpoint_v1' and payload['profile'] == b.PROFILE
        and payload['arm'] == arm and payload['phase'] == phase and payload['steps'] == payload['planned_steps'] == steps
        and payload['contract'] == b.CONTRACT and payload['prepared_sha256'] == report['prepared_sha256']
        and payload['source_checkpoint_id'] == prepared['source_checkpoint_id'], 'No old compact/source or artificial terminal state.')
    source, meta, source_audit = b.source()
    original = b.load_state(source, meta)
    decoder, protection, frozen = b.inventory(original, record['student_projection_names'], choice['kind'])
    b.require(decoder == record['decoder_parameters'] and protection == record['protection_parameters']
        and frozen == payload['frozen_state'] == record['validated_frozen_state']
        and set(payload['state']) == set(decoder) and set(payload['protection']) == set(protection), 'Actual source inventory changed.')
    mutable = {**payload['state'], **payload['protection']}
    for key, value in mutable.items():
        b.require(value.device.type == 'cpu' and value.dtype == original[key].dtype and value.shape == original[key].shape
            and value.is_contiguous() and b.torch.isfinite(value).all().item(), 'Invalid native shape/storage: ' + key)
    expected = {**original, **mutable}
    b.require(b.frozen_state(expected, mutable) == frozen, 'Export changed a frozen source field.')
    model, loaded, result = load_v6_model_checkpoint(str(source), map_location='cpu', strict=True, expected_kind='final_model')
    b.require(loaded == meta and not result.missing_keys and not result.unexpected_keys, 'Fresh strict02 source load failed.')
    live = model.state_dict()
    for key, value in mutable.items():
        live[key].copy_(value)
    model.eval().requires_grad_(False)
    refresh_vae_linear_runtime_after_state_load(model)
    tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], use_fast=True, local_files_only=True)
    provenance = dict(parent_checkpoint_ids=[meta['checkpoint_id']], source_checkpoint_dir=str(source),
        source_checkpoint_id=meta['checkpoint_id'], source_audit=b.binding(b.SOURCE_AUDIT), profile=b.PROFILE,
        arm=arm, phase=phase, actual_steps=steps, planned_steps=steps, full_F_identity=prepared['teacher_identity'],
        R_cache=prepared['R_cache'], contract=b.CONTRACT, input_evidence_sha256={str(report_path): b.sha256(report_path),
            str(patch): b.sha256(patch), report['prepared_path']: report['prepared_sha256']},
        replaced_tensor_keys=sorted(mutable), frozen_tensors_compared=frozen['tensors'],
        purpose='Inference-only actual BF16D and original-shape native protection endpoint; no optimizer/master/resume payload.')
    audit = save_and_verify(model, output, meta, expected, provenance, tokenizer)
    audit.update(schema='CG_aligned_protection_native_audit_v1', source_checkpoint_id=meta['checkpoint_id'],
        arm=arm, phase=phase, actual_steps=steps, planned_steps=steps, frozen_tensors_compared=frozen['tensors'], endpoint_report=b.binding(report_path))
    b.core.write((run if phase == 'short2' else b.OUTPUT / arm) / 'native_audit.json', audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    b.torch.set_num_threads(4)
    print(export(args.report, args.output_dir), flush=True)
