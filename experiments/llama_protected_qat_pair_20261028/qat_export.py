"""Canonical native3619 export replacing actual BF16 D and optional UINT8 q only."""
import gc
import qat_common as c


def export_endpoint(record, training, destination):
    from transformers import AutoTokenizer
    from experiments.qwen3_w25.assemble_categories import _load_state, _assert_state_equal
    from train_utils.checkpoint_bpw import audit_checkpoint, audit_state_dict
    from train_utils.checkpoint_v6 import refresh_vae_linear_runtime_after_state_load, save_v6_full_checkpoint
    c.require(training['steps'] == (2 if training['phase'] == 'short2' else 64)
        and training['source_restored_exact'], 'Export only actual verified pair endpoints.')
    c.check_binding(training['endpoint'])
    payload = c.torch.load(training['endpoint']['path'], map_location='cpu', weights_only=True)
    c.require(payload['source_checkpoint_id'] == c.SOURCE_ID and payload['steps'] == training['steps']
        and payload['arm'] == training['arm'] and payload['prepared_sha256'] == c.sha(c.HERE / 'prepared.json'), 'Wrong endpoint identity.')
    original = _load_state(c.SOURCE); meta = c.read(c.SOURCE / 'checkpoint_meta.json'); decoder = payload['state']; mutable = c.mutable(record, training['arm'])
    frozen = record['frozen_state'][training['arm']]
    count = 1568 if training['arm'] == 'DQ' else 1344
    c.require(set(decoder) == mutable and len(original) == 3619 and len(decoder) == count,
        'Actual native3619/D/q endpoint inventory differs.')
    for name, value in decoder.items():
        c.require(value.dtype == original[name].dtype == (c.torch.uint8 if name.endswith('.protected_input_qvalues') else c.torch.bfloat16)
            and value.shape == original[name].shape and value.is_contiguous() and c.torch.isfinite(value).all().item(), 'Invalid BF16 decoder endpoint: ' + name)
    c.require(c.state_proof(original, decoder) == frozen, 'Wrong actual frozen complement.')
    expected = {**original, **decoder}
    fields = ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'components', 'outside_linear_bits', 'code_padding_bits')
    before = record['bit_budget']; budget = audit_state_dict(meta, expected, max_bpw=c.MAX_BPW, max_lora_rank=0)
    c.require(budget['budget_passed'] and budget['user_linear_bpw'] == budget['packed_linear_bpw'] == c.BPW
        and all(before[key] == budget[key] for key in fields), 'Actual original storage/rank changed.')
    model, loaded_meta, loaded = c.native.load_v6_model_checkpoint(str(c.SOURCE), map_location='cpu', strict=True)
    c.require(loaded_meta['checkpoint_id'] == c.SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys, 'Source strict load failed.')
    for name, value in decoder.items(): model.state_dict()[name].copy_(value)
    model.eval().requires_grad_(False); refresh_vae_linear_runtime_after_state_load(model)
    _assert_state_equal(model.state_dict(), expected, 'Sparse005 paired D-only native3619')
    c.require(not destination.exists(), 'Fresh independent native endpoint required.')
    provenance = {'base_checkpoint_dir': str(c.SOURCE), 'base_checkpoint_id': c.SOURCE_ID,
        'donor_checkpoint_id': c.SOURCE_ID, 'donor_kind': 'Llama_sparse005_matched_fresh_D_vs_D_protected_q_QAT',
        'arm': training['arm'], 'phase': training['phase'], 'steps': training['steps'], 'seed': c.SEED,
        'replaced_tensor_keys': sorted(decoder), 'frozen_tensors_compared': frozen['tensors'],
        'projection_count': 224, 'code_banks': 512, 'sparse_tensors_frozen': 576,
        'endpoint': training['endpoint'], 'dataset': record['dataset'], 'R_teacher': c.SOURCE_ID,
        'purpose': 'Inference endpoint only; no optimizer/exact-resume claim. Both formal64 endpoints receive complete nine tasks.'}
    tokenizer = AutoTokenizer.from_pretrained(c.HF, use_fast=True, local_files_only=True)
    saved = save_v6_full_checkpoint(model, str(destination), checkpoint_kind='final_model',
        compressed_targets=meta['compressed_targets'], pending_dense_targets=(), skip_targets=(),
        train_mode='none', after_category_mode='none', norm_train_mode='none', lm_head_train_mode='none',
        lora_config=None, base_model_path=meta['base_model_path'], tokenizer=tokenizer,
        extra_meta={'llama_protected_qat_pair': provenance})
    del model; gc.collect()
    actual_meta = c.read(destination / 'checkpoint_meta.json'); serialized = _load_state(destination)
    result = {'status': 'invalid', 'checkpoint_dir': str(destination), 'checkpoint_id': saved['checkpoint_id'], 'sources': provenance}
    try:
        result['serialized_tensors_compared'] = _assert_state_equal(serialized, expected, 'Serialized3619')
        del serialized
        fresh, fresh_meta, loaded = c.native.load_v6_model_checkpoint(str(destination), map_location='cpu', strict=True)
        result['reloaded_tensors_compared'] = _assert_state_equal(fresh.state_dict(), expected, 'Fresh strict3619')
        c.require(not loaded.missing_keys and not loaded.unexpected_keys and fresh_meta == actual_meta
            and fresh_meta['checkpoint_id'] != c.SOURCE_ID and fresh_meta['converted_modules'] == meta['converted_modules']
            and fresh_meta['compressed_targets'] == meta['compressed_targets'] and fresh_meta['lora_config'] is None
            and not fresh_meta.get('immutable_resume_contract') and not fresh_meta['pending_dense_targets'] and not fresh_meta['skip_targets'],
            'Strict metadata/source identity or topology changed.')
        result['full_state'] = c.state_proof(fresh.state_dict()); del fresh; gc.collect()
        budget = audit_checkpoint(destination, max_bpw=c.MAX_BPW, max_lora_rank=0)
        c.require(budget['budget_passed'] and all(budget[key] == before[key] for key in fields), 'Actual serialized bit budget differs.')
        result.update(status='verified', bit_budget=budget, frozen_complement_exact=True, sparse576_exact=True,
            fresh_strict_reload_state_exact=True, weight=c.binding(destination / actual_meta['state_dict_file']))
    finally:
        c.write(destination / 'assembly_report.json', result)
    return result
