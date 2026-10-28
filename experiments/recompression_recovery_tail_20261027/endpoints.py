"""Native CPU assembly and independent complete-state audits at logical2.6/rank0."""
from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
import time

import common as c

import torch
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _expected_state, _load_state, _replace_modules, EXPECTED
from experiments.qwen3_w25.audit_e2e_decoder import _audit_decoder_export_state, _inference_spec, _validate_export_storage, persistent_inventory
from train_utils.checkpoint_v6 import save_v6_full_checkpoint


def save_native(model, output, meta, expected, provenance, tokenizer):
    c.require(not Path(output).exists(), 'Never overwrite an endpoint.')
    from train_utils.checkpoint_bpw import audit_state_dict
    budget = audit_state_dict(meta, expected, max_bpw=2.6, max_lora_rank=0)
    c.require(budget['budget_passed'] and not budget['contract_violations'], 'The complete assembled state exceeds logical2.6/rank0.')
    _assert_state_equal(model.state_dict(), expected, 'Actual native in-memory assembly')
    saved = save_v6_full_checkpoint(model, str(output), checkpoint_kind='final_model',
        compressed_targets=meta['compressed_targets'], pending_dense_targets=(), skip_targets=(),
        train_mode='none', after_category_mode='none', norm_train_mode='none', lm_head_train_mode='none',
        lora_config=None, base_model_path=meta['base_model_path'], tokenizer=tokenizer,
        extra_meta={'recompression_tail': provenance})
    actual = _load_state(output)
    count = _assert_state_equal(actual, expected, 'Actual native serialized3423')
    fresh, loaded_meta, loaded = c.load_cpu(output)
    c.require(not loaded.missing_keys and not loaded.unexpected_keys and loaded_meta['checkpoint_id'] == saved['checkpoint_id'], 'Independent native fresh strict load failed.')
    c.require(loaded_meta['checkpoint_id'] != meta['checkpoint_id']
        and loaded_meta.get('immutable_resume_contract') is None
        and not (loaded_meta.get('extra_meta') or {}).get('cat_runtime_state_file')
        and set(loaded_meta['compressed_targets']) == set(meta['compressed_targets'])
        and loaded_meta['base_model_path'] == meta['base_model_path']
        and loaded_meta['lora_config'] is None and not loaded_meta['pending_dense_targets'] and not loaded_meta['skip_targets']
        and {row['name']: _inference_spec(row) for row in loaded_meta['converted_modules']}
        == {row['name']: _inference_spec(row) for row in meta['converted_modules']}, 'Native writer changed deployment metadata or inherited a resume contract.')
    compared = _assert_state_equal(fresh.state_dict(), expected, 'Independent native fresh strict3423')
    _, actual_budget = c.checkpoint(output)
    c.require(actual_budget['components'] == budget['components'] and count == compared == 3423, 'Native save/fresh full-state ledger differs.')
    report = dict(status='verified', checkpoint_dir=str(output), checkpoint_id=saved['checkpoint_id'],
        serialized_tensors_compared=count, fresh_strict_tensors_compared=compared,
        fresh_strict_reload_state_exact=True, bit_budget=actual_budget, sources=provenance, export_used_cuda=False)
    c.write(Path(output) / 'assembly_report.json', report)
    del fresh, actual
    gc.collect()
    return report


def export_chatml(profile):
    from transformers import AutoTokenizer
    from experiments.qwen3_w25.kl_joint_export import CODE_NAMES, DECODER_NAMES, SELECTED_KEYS, apply_payload_, project_payload
    from experiments.qwen3_w25.kl_round_export import validate_codes, LOGICAL_SHAPE
    from experiments.qwen3_w25.local_bits import _changed_bits
    from experiments.qwen3_w25.format_recovery_diagnostic import check_inputs
    root = c.OUTPUT / profile / '07_chatml_joint256'
    launch = c.read(root / 'format_launch.json')
    check_inputs(launch)
    report = c.read(root / 'format_observations/diagnostic.json')
    source, output = c.SOURCES[profile], c.DATA / profile / '07_chatml_joint256/final_model'
    c.require(report['status'] == 'complete' and report['source_checkpoint_id'] == c.SOURCE_IDS[profile]
        and Path(report['source_dir']).resolve() == source and report['steps'] == 256 and report['joint']
        and report['training_format'] == 'chatml' and report['frozen_state']['unchanged']
        and report['frozen_cache_pointers_unchanged'] and report['source_pins_data_unchanged']
        and report['runtime']['frozen_tensor_count'] == 3415, 'Require the actual same-source fixed256 ChatML endpoint.')
    payload_path = Path(report['selected_payload']['path'])
    c.require(payload_path.parent == root / 'format_observations' and c.sha(payload_path) == report['selected_payload']['sha256'], 'Actual ChatML endpoint payload binding changed.')
    payload = torch.load(payload_path, map_location='cpu', weights_only=True)
    projected = project_payload(payload, expected_source_id=c.SOURCE_IDS[profile])
    c.require(payload['joint'] and payload['training_format'] == 'chatml' and payload['data_archive_sha256'] == launch['data_archive']['sha256'], 'ChatML payload provenance differs.')
    validate_codes({c.TARGET + '.' + name: payload['codes'][name] for name in CODE_NAMES})
    meta, source_budget = c.checkpoint(source)
    original = _load_state(source)
    expected = {**original, **projected}
    xor = [_changed_bits(original[c.TARGET + '.' + name], payload['codes'][name], LOGICAL_SHAPE) for name in CODE_NAMES]
    changes = {name: int(torch.count_nonzero(payload['decoder_fp32'][name] != original[c.TARGET + '.' + name].float())) for name in DECODER_NAMES}
    c.require(xor == report['net_xor_by_stage'] and changes == report['decoder_changed_elements'], 'Payload codes/FP32 decoder disagree with the actual observed terminal.')
    model, loaded_meta, loaded = c.load_cpu(source)
    c.require(not loaded.missing_keys and not loaded.unexpected_keys and loaded_meta == meta, 'ChatML source fresh strict load failed.')
    _assert_state_equal(model.state_dict(), original, 'Unmodified complete ChatML source3423')
    apply_payload_(model, payload, expected_source_id=meta['checkpoint_id'])
    frozen = {key: value for key, value in original.items() if key not in SELECTED_KEYS}
    c.require(_assert_state_equal({key: model.state_dict()[key] for key in frozen}, frozen, 'ChatML frozen3415') == 3415, 'ChatML frozen complement is incomplete.')
    tokenizer = AutoTokenizer.from_pretrained(str(c.HF), use_fast=True, local_files_only=True)
    provenance = dict(profile=profile, stage='07_chatml_joint256', source_dir=str(source), source_checkpoint_id=meta['checkpoint_id'],
        diagnostic=str(root / 'format_observations/diagnostic.json'), diagnostic_sha256=c.sha(root / 'format_observations/diagnostic.json'),
        payload=str(payload_path), payload_sha256=c.sha(payload_path), replaced_tensor_keys=sorted(SELECTED_KEYS),
        frozen_tensors_compared=3415, source_bit_budget=source_budget, code_net_xor_by_stage=xor,
        BF16_projection='Original exact torch.bfloat16 cast of six saved FP32 decoder leaves; two saved uint8 banks.')
    audit = save_native(model, output, meta, expected, provenance, tokenizer)
    for key in ('user_linear_bits', 'packed_linear_bits', 'components', 'code_padding_bits'):
        c.require(audit['bit_budget'][key] == source_budget[key], 'Fixed geometry ChatML export changed storage: ' + key)
    c.write(root / 'native_audit.json', audit)
    return audit


def assemble_attention(profile):
    from experiments.qwen3_w25.assemble_runtime_categories import _validate_sources
    chatml = c.read(c.OUTPUT / profile / '07_chatml_joint256/native_audit.json')
    decoder = c.read(c.OUTPUT / profile / '06_decoder_kl/native_audit.json')
    c.require(chatml['status'] == decoder['status'] == 'verified' and decoder['source_checkpoint_id'] == c.SOURCE_IDS[profile]
        and chatml['sources']['source_checkpoint_id'] == c.SOURCE_IDS[profile], 'Both inference donors must descend from the same fixed assembly.')
    base, donor = Path(chatml['checkpoint_dir']), Path(decoder['checkpoint_dir'])
    meta, donor_meta, names, original, tokenizer, runtime = _validate_sources(base, donor, ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    c.require(meta['checkpoint_id'] == chatml['checkpoint_id'] and donor_meta['checkpoint_id'] == decoder['checkpoint_id'] and len(names) == 144, 'Use144 same-source KL attention projections.')
    expected = _expected_state(_load_state(base), _load_state(donor), names, EXPECTED)
    model, _, _ = c.load_cpu(base)
    donor_model, _, _ = c.load_cpu(donor)
    _replace_modules(model, donor_model, names)
    del donor_model
    gc.collect()
    provenance = dict(profile=profile, stage='08_joint_mlp_decoder_attention', source_checkpoint_id=c.SOURCE_IDS[profile],
        base_checkpoint_dir=str(base), base_checkpoint_id=meta['checkpoint_id'], donor_checkpoint_dir=str(donor),
        donor_checkpoint_id=donor_meta['checkpoint_id'], categories=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        replaced_targets=names, source_config_runtime_fields=runtime, inference_only=True)
    audit = save_native(model, c.DATA / profile / '08_joint_mlp_decoder_attention/final_model', meta, expected, provenance, tokenizer)
    c.write(c.OUTPUT / profile / '08_joint_mlp_decoder_attention/native_audit.json', audit)
    return audit


def audit_training(profile, recipe):
    from experiments.qwen3_w25.e2e_subset_audit import audit_subset_state
    from train_utils.checkpoint_v6 import load_v6_training_model_state
    stage = '06_decoder_kl' if recipe == 'decoder' else '09_joint1200'
    runtime = c.read(c.DATA / profile / stage / 'stage_result.json')
    final, training = Path(runtime['saved_model_dir']), Path(runtime['training_checkpoint_dir'])
    source = c.SOURCES[profile] if recipe == 'decoder' else c.DATA / profile / '08_joint_mlp_decoder_attention/final_model'
    finish = 2000 if recipe == 'decoder' else 1200
    if recipe == 'decoder':
        from experiments.qwen3_w25.export_e2e_selected import inspect_step
    else:
        from experiments.qwen3_w25.e2e_subset_export import inspect_step
    step_meta, state, _snapshot, bound_source, _ = inspect_step(training)
    c.require(bound_source.resolve() == source and state['global_step'] == finish and state['max_steps'] == 2000
        and runtime['global_step'] == runtime['finish_step'] == finish and runtime['max_steps'] == 2000, 'Actual saved endpoint or original2000 schedule differs.')
    source_meta, source_budget = c.checkpoint(source)
    meta, budget = c.checkpoint(final)
    c.require(meta['checkpoint_id'] not in (source_meta['checkpoint_id'], step_meta['checkpoint_id'])
        and meta['immutable_resume_contract'] == step_meta['immutable_resume_contract']
        and meta['train_mode'] == ('decoder' if recipe == 'decoder' else 'decoder_sparse_bit'), 'Runtime native final metadata differs from its real saved training endpoint.')
    c.require({row['name']: _inference_spec(row, export_decoder_dtype=True) for row in source_meta['converted_modules']}
        == {row['name']: _inference_spec(row) for row in meta['converted_modules']}, 'Inference topology/protection/rotation changed.')
    original, actual = _load_state(source), _load_state(final)
    model, loaded_meta, loaded = c.load_cpu(source)
    c.require(not loaded.missing_keys and not loaded.unexpected_keys and loaded_meta == source_meta, 'Source independent CPU strict load failed.')
    inventory, banks = persistent_inventory(model)
    c.require(len(original) == len(actual) == 3423 and sum(map(len, inventory.values())) == 1512 and sum(map(len, banks.values())) == 504, 'Actual complete native inventory differs.')
    _assert_state_equal(model.state_dict(), original, 'Source fresh strict3423')
    mutable, manifest = load_v6_training_model_state(str(training), map_location='cpu')
    keys = {key for values in inventory.values() for key in values} if recipe == 'decoder' else set(inventory[c.TARGET])
    c.require(set(mutable) == keys and {row['name']: row['component_class'] for row in manifest} == dict.fromkeys(keys, 'decoder')
        and all(value.dtype == torch.float32 for value in mutable.values()), 'Saved FP32 mutable state differs from the actual decoder selection.')
    if recipe == 'decoder':
        report = _audit_decoder_export_state(original, actual, inventory, banks, train_mode='decoder')
        _assert_state_equal({key: actual[key] for key in keys}, {key: value.bfloat16() for key, value in mutable.items()}, 'Saved1512 FP32 decoder to native BF16 exact cast')
    else:
        from experiments.qwen3_w25.cpu_bit_export import restore_packed_bits_cpu
        restore = restore_packed_bits_cpu([(c.TARGET, model.get_submodule(c.TARGET))], checkpoint=training, contract=step_meta['immutable_resume_contract'])
        packed = {key: model.state_dict()[key].detach().clone() for key, _ in banks[c.TARGET]}
        report = audit_subset_state(original, actual, inventory, banks, selected_names=(c.TARGET,), training_decoder_state=mutable, packed_snapshot=packed)
        report['sparse_bit_restore'] = restore
        c.require(report['frozen_tensors_compared'] == 3415 and restore['active_score_signs_verified'] == 147849216, 'Complete frozen complement/current hard-code signs are required.')
    del model
    gc.collect()
    _validate_export_storage(source_budget, budget, report)
    fresh, loaded_meta, loaded = c.load_cpu(final)
    c.require(not loaded.missing_keys and not loaded.unexpected_keys and loaded_meta == meta, 'Independent final strict load failed.')
    compared = _assert_state_equal(fresh.state_dict(), actual, 'Independent native final fresh strict3423')
    report.update(status='verified', profile=profile, stage=stage, checkpoint_dir=str(final), checkpoint_id=meta['checkpoint_id'],
        source_checkpoint_id=source_meta['checkpoint_id'], source_dir=str(source), training_checkpoint=str(training),
        selected_global_step=finish, original_scheduler_steps=2000, bit_budget=budget, source_bit_budget=source_budget,
        serialized_tensors_compared=len(actual), fresh_strict_tensors_compared=compared, fresh_strict_reload_state_exact=True,
        saved_training_decoder_BF16_exact=True, actual_native_runtime_finalization=meta['finalized_status'],
        audit_used_cuda=False, native_runtime_finalization_used_cuda=True,
        source_training_state_sha256=c.sha(training / 'training_model_state.pt'))
    c.write(c.OUTPUT / profile / stage / 'native_audit.json', report)
    return report


@torch.no_grad()
def consume_chatml(profile):
    from tools.cat_eval import _prepare_model_for_eval
    from torch.nn import functional as F
    root = c.OUTPUT / profile / '07_chatml_joint256'
    audit = c.read(root / 'native_audit.json')
    model, meta, loaded = c.load_cpu(audit['checkpoint_dir'])
    c.require(not loaded.missing_keys and not loaded.unexpected_keys and meta['checkpoint_id'] == audit['checkpoint_id'], 'Actual saved ChatML consumer strict load failed.')
    model.eval().requires_grad_(False)
    warmup = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    c.require(warmup['total'] == warmup['warmed'] == 252 and not warmup['failed'] and not warmup['skipped'], 'Consume every actual native projection.')
    target = model.get_submodule(c.TARGET)
    calls = []
    def observe(layer, args, result):
        c.require(layer.cache_decoded_weight and layer.temporary and not layer.trainable_decode
            and torch.equal(result, F.linear(args[0], layer._cached_weight, layer.bias)), 'The actual whole-model forward must consume saved native L0-down cache.')
        calls.append(1)
    handle = target.register_forward_hook(observe)
    try:
        archive = torch.load(c.ARCHIVE / 'batches.pt', map_location='cpu', weights_only=True)
        batch = archive['formats']['chatml']['batches'][0]
        result = model(input_ids=batch['input_ids'].to('cuda:0'), attention_mask=batch['attention_mask'].to('cuda:0'),
                       use_cache=False, logits_to_keep=1)
        c.require(calls == [1] and result.logits.dtype == torch.bfloat16 and tuple(result.logits.shape) == (4, 1, 151936)
            and torch.isfinite(result.logits).all().item() and result.past_key_values is None, 'Actual unchanged train-only ChatML batch4 consumer failed.')
    finally:
        handle.remove()
    report = dict(status='verified', checkpoint_id=meta['checkpoint_id'], whole_model_forwards=1, batch_size=4,
        actual_native_L0_down_Flinear=True, warmup=warmup, archive_sha256=c.sha(c.ARCHIVE / 'batches.pt'),
        native_audit_sha256=c.sha(root / 'native_audit.json'), quality_selection=False)
    c.write(root / 'gpu_validation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['chatml', 'attention', 'audit', 'consumer'])
    parser.add_argument('--profile', choices=['C', 'G'], required=True)
    parser.add_argument('--recipe', choices=['decoder', 'last_joint'])
    args = parser.parse_args()
    c.check_prepared()
    torch.set_num_threads(4)
    c.require(not torch.cuda.is_initialized(), 'CPU endpoint work must remain on CPU.')
    result = consume_chatml(args.profile) if args.command == 'consumer' else export_chatml(args.profile) if args.command == 'chatml' else assemble_attention(args.profile) if args.command == 'attention' else audit_training(args.profile, args.recipe)
    if args.command != 'consumer':
        c.require(not torch.cuda.is_initialized(), 'Native CPU endpoint processing initialized CUDA.')
    print(dict(status=result['status'], checkpoint_id=result['checkpoint_id']), flush=True)


if __name__ == '__main__':
    main()
