"""Export this actual aligned OPD32 endpoint with unchanged native storage/rank0."""
import argparse
import gc
from pathlib import Path

import bindings as b
from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.qwen3_w25.audit_e2e_decoder import _inference_spec
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import save_v6_full_checkpoint, refresh_vae_linear_runtime_after_state_load
from train_utils.v6_model_loader import load_v6_model_checkpoint
from transformers import AutoTokenizer


@b.torch.no_grad()
def export(report_path, output):
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    report = b.read(report_path)
    prepared, record = b.read_prepared(report['prepared_path'])
    source_id = record['source_checkpoint_id']
    b.require(report_path == b.OUTPUT / 'formal32/report.json' and report['status'] == 'aligned_nativegreedy_OPD_formal32_complete'
        and report['phase'] == 'formal32' and report['steps'] == 32
        and (report_path.parent / 'run.exit').read_text().strip() == '0'
        and report['prepared_sha256'] == b.sha256(report['prepared_path']) and report['source_checkpoint_id'] == source_id
        and report['actual_target_contract'] == b.CONTRACT and report['source_restored_exact']
        and report['endpoint']['save_reload_install_exact']
        and [row['group_index'] for row in report['training']['trajectory']] == list(range(32))
        and report['training']['frozen_state_before'] == report['training']['frozen_state_after'] == record['validated_frozen_state'],
        'Require this actual fresh32 OPD endpoint and unchanged source complement.')
    b.require(output == b.OUTPUT / 'native_final' / 'final_model' and not output.exists() and not output.is_symlink(),
        'Use the isolated aligned native deployment output.')
    patch = Path(report['endpoint']['path'])
    b.require(patch == b.OUTPUT / 'formal32/decoder_endpoint.pt' and b.sha256(patch) == report['endpoint']['sha256'],
        'Changed actual OPD decoder endpoint.')
    payload = b.torch.load(patch, map_location='cpu', weights_only=True)
    b.require(payload['schema'] == 'CG_aligned_online_OPD_endpoint_v1' and payload['phase'] == 'formal32'
        and payload['steps'] == 32 and payload['prepared_sha256'] == report['prepared_sha256']
        and payload['actual_target_contract'] == b.CONTRACT and payload['source_checkpoint_id'] == source_id,
        'Do not substitute an old, teacher-forced, or short endpoint.')
    source, meta, source_audit = b.source()
    original = b.load_state(source, meta)
    declared, frozen = b.hot.state_inventory(original, record['student_projection_names'])
    decoder = payload['state']
    b.require(len(original) == 3423 and len(decoder) == 1512 and set(decoder) == set(declared)
        and declared == prepared['decoder_parameters'] and frozen == payload['frozen_state'] == record['validated_frozen_state']
        and sum(v.numel() for v in decoder.values()) == record['trainable_elements'], 'Changed source-derived full decoder inventory.')
    for key, value in decoder.items():
        b.require(value.device.type == 'cpu' and value.dtype == original[key].dtype == b.torch.bfloat16
            and value.shape == original[key].shape and value.is_contiguous() and b.torch.isfinite(value).all().item(),
            'Invalid native BF16 decoder: ' + key)
    expected = {**original, **decoder}
    budget = audit_state_dict(meta, expected, max_bpw=2.6, max_lora_rank=0)
    source_budget = audit_state_dict(meta, original, max_bpw=2.6, max_lora_rank=0)
    b.require(source_budget['budget_passed'] and budget['budget_passed']
        and all(budget[k] == source_budget[k] for k in b.BUDGET_FIELDS), 'OPD changed native logical/packed storage or rank0.')
    model, loaded, result = load_v6_model_checkpoint(str(source), map_location='cpu', strict=True, expected_kind='final_model')
    b.require(loaded == meta and not result.missing_keys and not result.unexpected_keys, 'Fresh source strict native loading failed.')
    live = model.state_dict()
    for key, value in decoder.items():
        live[key].copy_(value)
    model.eval().requires_grad_(False)
    refresh_vae_linear_runtime_after_state_load(model)
    _assert_state_equal(model.state_dict(), expected, 'All aligned OPD endpoint tensors')
    tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], use_fast=True, local_files_only=True)
    provenance = dict(base_checkpoint_dir=str(source), base_checkpoint_id=source_id,
        donor_kind='same_actual_aligned01_online_OPD32_decoder_endpoint', phase='formal32', steps=32,
        measured_report=str(report_path), measured_status=report['status'], source_audit=b.binding(b.SOURCE_AUDIT),
        frozen_tensors_compared=frozen['tensors'], replaced_tensor_keys=sorted(decoder), teacher_identity=record['teacher_identity'],
        actual_target_contract=b.CONTRACT, dataset_binding=prepared['dataset_binding'], retention_lambda=0.,
        input_evidence_sha256={str(report_path): b.sha256(report_path), str(patch): b.sha256(patch),
            report['prepared_path']: report['prepared_sha256']}, source_bit_budget=source_budget,
        purpose='Actual all252 BF16 decoder endpoint; BSQ/protection/norm/head frozen; no LoRA, R target or optimizer payload.')
    saved = save_v6_full_checkpoint(model, str(output), checkpoint_kind='final_model',
        compressed_targets=meta['compressed_targets'], pending_dense_targets=(), skip_targets=(), train_mode='none',
        after_category_mode='none', lora_config=None, base_model_path=meta['base_model_path'], tokenizer=tokenizer,
        extra_meta={'aligned_online_OPD': provenance})
    del live, model
    gc.collect()
    serialized = b.load_state(output)
    compared = _assert_state_equal(serialized, expected, 'Saved aligned OPD3423')
    fresh, loaded, result = load_v6_model_checkpoint(str(output), map_location='cpu', strict=True, expected_kind='final_model')
    fresh_compared = _assert_state_equal(fresh.state_dict(), expected, 'Fresh strict aligned OPD3423')
    b.require(not result.missing_keys and not result.unexpected_keys and saved['checkpoint_id'] == loaded['checkpoint_id'] != source_id
        and {_['name']: _inference_spec(_) for _ in loaded['converted_modules']}
        == {_['name']: _inference_spec(_) for _ in meta['converted_modules']}
        and loaded.get('immutable_resume_contract') is None and loaded['lora_config'] is None
        and type(fresh.lm_head) is b.torch.nn.Linear and not fresh.config.tie_word_embeddings,
        'Fresh endpoint identity, geometry or topology differs.')
    final_budget = audit_state_dict(loaded, serialized, max_bpw=2.6, max_lora_rank=0)
    b.require(final_budget['budget_passed'] and all(final_budget[k] == source_budget[k] for k in b.BUDGET_FIELDS),
        'Saved native storage differs from the actual source.')
    result = dict(schema='aligned_online_OPD_native_audit_v1', status='verified', profile=b.PROFILE,
        checkpoint_dir=str(output), checkpoint_id=loaded['checkpoint_id'], source_checkpoint_id=source_id,
        serialized_tensors_compared=compared, fresh_strict_tensors_compared=fresh_compared,
        fresh_strict_reload_state_exact=True, frozen_tensors_compared=frozen['tensors'], code_bank_count=504,
        bit_budget=final_budget, source_bit_budget=source_budget, source_geometry_unchanged=True,
        endpoint_report=b.binding(report_path), checkpoint_weight_stat=b.file_stat(output / loaded['state_dict_file']),
        checkpoint_meta_sha256=b.sha256(output / 'checkpoint_meta.json'), sources=provenance)
    b.core.write(b.OUTPUT / 'native_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    b.torch.set_num_threads(4)
    print(export(args.report, args.output_dir), flush=True)
