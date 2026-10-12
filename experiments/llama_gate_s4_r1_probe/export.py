"""Export and independently audit an actual saved R0/R1 all-gate training step."""
import argparse
import gc
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from compressed_e2e_fintuning.mid_eval import temporary_inference_decode_mode
from compressed_e2e_fintuning.runtime_v6 import (
    _assert_final_runtime_clean, _build_v6_tokenizer, _finalize_decoders, _sync_model_padding_config)
from compressed_e2e_fintuning.runtime_v6_pipeline import (
    _assert_finalization_probe_close, _build_finalization_probe_inputs, _run_finalization_probe)
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.gate_student_context_evaluation.deploy import native_policy
from experiments.llama_gate_r1_probe.migrate import error, strict_load
from experiments.llama_gate_s4_r1_probe.common import (
    GATE_NAMES, SOURCE, SOURCE_ID, budget_check, check, expected_specs, read, require,
    state_inventory as inventory)
from experiments.longbench_v1.common import check_stat, file_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
from experiments.qwen3_w25.audit_e2e_decoder import _inference_spec
from experiments.qwen3_w25.export_e2e_selected import restore_decoder_state
from experiments.w25_eval.subset_state import (
    selected_layers, selected_modules, selected_names, validate_configuration, validate_sparse_sidecar)
from litebsq.vae_linear import clear_model_vae_linear_cache
from tools.cat_eval import _prepare_model_for_eval
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import (
    load_v6_training_model_state, load_v6_training_step_meta, resolve_training_step_round_base_ref,
    save_v6_full_checkpoint)
from train_utils.config.targets import collect_e2e_compressed_targets
from train_utils.distill_precision import prepare_model_export


def inspect_step(record, arm, step):
    require(arm in record['schedule']['arms'] and type(step) is int and step in record['schedule']['export_steps'],
            'Select an actual scheduled R0/R1 save boundary.')
    root = Path(record['output_dir'])
    matches = list((root / arm).glob(f'home_*/trainer_state/checkpoint-{step}'))
    require(len(matches) == 1, 'Require exactly one native checkpoint for this arm and step.')
    checkpoint = matches[0].resolve()
    meta = load_v6_training_step_meta(str(checkpoint))
    source, base_meta = resolve_training_step_round_base_ref(str(checkpoint), meta)
    expected_source = SOURCE if arm == 'r0' else root / 'r1_initial'
    source = Path(source).resolve()
    require(source == expected_source.resolve(), 'The saved step has a different round base.')
    expected_id = SOURCE_ID
    if arm == 'r1':
        migration = read(root / 'migration.json')
        require(migration['status'] == 'verified' and migration['source_checkpoint_id'] == SOURCE_ID,
                'The actual R1 migration has not been verified.')
        check_stat(migration['weight_stat'])
        require(sha256(source / 'checkpoint_meta.json') == migration['metadata_sha256'], 'R1 source metadata changed.')
        expected_id = migration['checkpoint_id']
    require(base_meta['checkpoint_id'] == meta['round_base_checkpoint_id'] == expected_id
            and base_meta['converted_modules'] == expected_specs(read(SOURCE / 'checkpoint_meta.json'), arm),
            'Saved source identity or S4/R1 topology differs.')
    snapshot = read(checkpoint.parent.parent / 'normalized_e2e_runtime_args.json')
    state = read(checkpoint / 'trainer_state.json')
    architecture = record['architecture']
    validate_configuration(meta, state, snapshot, architecture)
    validate_sparse_sidecar(checkpoint, 'decoder')
    require(state['global_step'] == step and meta['train_mode'] == 'decoder'
            and meta['base_model_path'] == base_meta['base_model_path']
            and set(meta['compressed_targets']) == set(architecture['projection_shapes'])
            and not meta['pending_dense_targets'] and not meta['skip_targets']
            and tuple(selected_names(meta['target_layers'], meta['target_modules'], architecture)) == tuple(GATE_NAMES),
            'Only the exact native all32-gate decoder scope may be exported.')
    return checkpoint, meta, state, snapshot, source, base_meta


def compare_saved(original, actual, mutable, manifest, record, arm):
    expected = record['inventories'][arm]
    selected = set(expected['selected'])
    require(inventory(original) == inventory(actual) == expected and set(mutable) == selected
            and len(manifest) == len(selected)
            and {row['name']: row['component_class'] for row in manifest} == {key: 'decoder' for key in selected},
            'Persistent or mutable decoder inventory differs.')
    projected = dict(original)
    fp32_changes, bf16_changes = {}, {}
    for key in sorted(selected):
        value = mutable[key]
        require(value.dtype == torch.float32 and value.shape == original[key].shape
                and original[key].dtype == torch.bfloat16 and bool(torch.isfinite(value).all()),
                'Expected a finite FP32 decoder endpoint: ' + key)
        projected[key] = value.bfloat16()
        require(torch.isfinite(projected[key]).all(), 'BF16 decoder overflow: ' + key)
        fp32_changes[key] = int(torch.count_nonzero(value != original[key].float()))
        bf16_changes[key] = int(torch.count_nonzero(projected[key] != original[key]))
    count = _assert_state_equal(actual, projected, 'Saved state equals exact selected FP32 cast plus frozen source')
    require(sum(fp32_changes.values()) > 0, 'The saved native step contains no actual decoder update.')
    return {'serialized_tensors_compared': count, 'selected_decoder_tensor_count': len(selected),
        'selected_decoder_fp32_changed_elements': sum(fp32_changes.values()),
        'selected_decoder_fp32_changed_elements_by_key': fp32_changes,
        'changed_decoder_elements': sum(bf16_changes.values()),
        'changed_decoder_tensor_count': sum(x > 0 for x in bf16_changes.values()),
        'selected_decoder_matches_fp32_cast_exactly': True,
        'selected_decoder_update_survived_bf16': sum(bf16_changes.values()) > 0,
        'frozen_tensors_compared': count - len(selected), 'frozen_bytes_unchanged': True,
        'code_bank_count': expected['banks'], 'frozen_code_bank_count': expected['banks'], 'changed_code_bits': 0}


@torch.no_grad()
def export_and_audit(path, arm, step):
    record = read(path); check(record)
    checkpoint, meta, state, snapshot, source, base_meta = inspect_step(record, arm, step)
    step_dir = Path(record['output_dir']) / arm / f'step_{step:04d}'
    output = step_dir / 'final_model'; audit_path = step_dir / 'audit.json'
    require(not output.exists() and not audit_path.exists(), 'Never overwrite an exported step or its audit.')
    require(int(os.environ.get('WORLD_SIZE', '1')) == 1 and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, 'Independent export uses the bound single visible CUDA GPU.')
    started = time.time(); training = SimpleNamespace(**snapshot['training_args'])
    configure_e2e_determinism(bool(training.full_determinism)); set_e2e_seed(training.seed)
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(0)
    if training.tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = bool(training.tf32)
        torch.backends.cudnn.allow_tf32 = bool(training.tf32)
    original = _load_state(source)
    require(inventory(original) == record['inventories'][arm], 'Source state differs from prepared inventory.')
    source_budget = audit_state_dict(base_meta, original, max_bpw=2.5, max_lora_rank=0)
    budget_check(source_budget, record, arm)
    model, loaded_meta = strict_load(source)
    require(loaded_meta == base_meta, 'Native round-base metadata changed.')
    _assert_state_equal(model.state_dict(), original, 'Source strict reconstruction')
    policy = native_policy(model, record['model_policy'])
    architecture = record['architecture']
    selected = collect_e2e_compressed_targets(model,
        target_layers=selected_layers(meta['target_layers'], architecture),
        target_modules=selected_modules(meta['target_modules'], architecture),
        num_layers=architecture['num_hidden_layers'])
    require(tuple(name for name, _ in selected) == tuple(GATE_NAMES), 'Runtime selected modules differ.')
    restored = restore_decoder_state(model, selected, checkpoint=checkpoint, snapshot=snapshot)
    require(restored == len(record['inventories'][arm]['selected']), 'Restored state escaped the selected decoder leaves.')
    tokenizer = _build_v6_tokenizer(str(base_meta['base_model_path']), SimpleNamespace(**snapshot['hf_args']))
    _sync_model_padding_config(model, tokenizer); model.config.use_cache = False
    model.eval().to('cuda:0')
    inputs = _build_finalization_probe_inputs(tokenizer)
    training_probe, dtype = _run_finalization_probe(model, inputs)
    with temporary_inference_decode_mode(model, cache_decoded_weight=True):
        deployment_probe, deployment_dtype = _run_finalization_probe(model, inputs)
    require(deployment_dtype == dtype, 'Deployment reference changed output dtype.')
    require(_finalize_decoders(selected) == len(GATE_NAMES), 'Not all selected gate decoders were finalized.')
    _assert_final_runtime_clean(model)
    finalized, finalized_dtype = _run_finalization_probe(model, inputs)
    require(finalized_dtype == dtype, 'Structural finalization changed output dtype.')
    structural = _assert_finalization_probe_close(deployment_probe, finalized, output_dtype=dtype,
                                                label='S4/R1 selected-step structural finalization')
    export_dtype = str(prepare_model_export(model, training))
    exported, exported_dtype = _run_finalization_probe(model, inputs)
    require(exported_dtype == dtype and native_policy(model, record['model_policy']) == policy,
            'Export changed native dtype or RoPE policy.')
    clear_model_vae_linear_cache(model); _assert_final_runtime_clean(model)
    model.requires_grad_(False).to('cpu')
    runtime_audit = {'runtime': 'experiments.llama_gate_s4_r1_probe.export', 'source_checkpoint': str(checkpoint),
        'source_checkpoint_id': meta['checkpoint_id'], 'round_base_dir': str(source),
        'round_base_checkpoint_id': base_meta['checkpoint_id'], 'source_global_step': step,
        'source_max_steps': state['max_steps'], 'restored_decoder_tensors': restored,
        'optimizer_steps_executed': 0, 'trainer_or_main_optimizer_constructed': False,
        'teacher_or_training_data_loaded': False, 'decoder_compute_path_difference': error(deployment_probe, training_probe),
        'core_structural_finalization_forward_parity': structural, 'export_dtype': export_dtype,
        'export_precision_probe': error(exported, finalized),
        'selected_training_state_sha256': sha256(checkpoint / 'training_model_state.pt'),
        'native_metadata_sha256': sha256(checkpoint / 'checkpoint_meta.json'), 'export_script_sha256': sha256(__file__)}
    saved = save_v6_full_checkpoint(model, str(output), checkpoint_kind='final_model',
        **{key: tuple(base_meta.get(key) or ()) for key in ('compressed_targets', 'pending_dense_targets', 'skip_targets',
            'legacy_original_only_sources', 'completed_categories', 'compression_categories')},
        train_mode='decoder', norm_train_mode='none', lm_head_train_mode='none', lora_config=None,
        resolved_learning_rates=meta['resolved_learning_rates'], target_layers=meta['target_layers'],
        target_modules=meta['target_modules'], immutable_resume_contract=meta['immutable_resume_contract'],
        base_model_path=base_meta['base_model_path'], tokenizer=tokenizer,
        finalized_status={'decoder_finalized': True, 'runtime_clean': True, 'inference_forward_parity': True},
        runtime_audit=runtime_audit)
    require(saved['checkpoint_id'] not in (meta['checkpoint_id'], base_meta['checkpoint_id']), 'Export reused an input UUID.')
    actual = _load_state(output); saved_meta = read(output / 'checkpoint_meta.json')
    require([_inference_spec(s) for s in saved_meta['converted_modules']]
            == [_inference_spec(s) for s in base_meta['converted_modules']], 'Export changed deployment geometry.')
    mutable, manifest = load_v6_training_model_state(str(checkpoint), map_location='cpu')
    audit = compare_saved(original, actual, mutable, manifest, record, arm)
    budget = audit_state_dict(saved_meta, actual, max_bpw=2.5, max_lora_rank=0)
    budget_check(budget, record, arm)
    del model, selected, mutable; gc.collect(); torch.cuda.empty_cache()
    fresh, fresh_meta = strict_load(output)
    require(fresh_meta == saved_meta and type(fresh.lm_head) is torch.nn.Linear
            and not fresh.config.tie_word_embeddings, 'Fresh native deployment topology differs.')
    count = _assert_state_equal(fresh.state_dict(), actual, 'Fresh strict saved state')
    require(inventory(fresh.state_dict()) == record['inventories'][arm], 'Fresh state inventory differs.')
    fresh.config.use_cache = False
    warmup = _prepare_model_for_eval(fresh, 'cuda:0', 8, logging.getLogger(__name__), {})
    require(warmup['total'] == 224 and warmup['failed'] == 0
            and native_policy(fresh, record['model_policy']) == policy, 'Fresh group8 deployment preparation differs.')
    before = {k: (v.data_ptr(), v._version) for k, v in fresh.state_dict().items()}
    fresh_probe, fresh_dtype = _run_finalization_probe(fresh, inputs)
    require(fresh_dtype == exported_dtype, 'Fresh inference changed logits dtype.')
    fresh_parity = _assert_finalization_probe_close(exported, fresh_probe, output_dtype=fresh_dtype,
                                                   label='Fresh S4/R1 group8 inference')
    require(before == {k: (v.data_ptr(), v._version) for k, v in fresh.state_dict().items()}
            and native_policy(fresh, record['model_policy']) == policy, 'Fresh inference mutated persistent state or RoPE.')
    check(record)
    report = {'status': 'verified', 'arm': arm, 'global_step': step, 'checkpoint_id': saved['checkpoint_id'],
        'checkpoint_dir': str(output), 'training_checkpoint_dir': str(checkpoint),
        'source_checkpoint_id': base_meta['checkpoint_id'], 'selected_checkpoint_id': meta['checkpoint_id'],
        **audit, 'reloaded_tensors_compared': count, 'fresh_strict_reload_state_exact': True,
        'group8_warmup': warmup, 'fresh_group8_forward_parity': fresh_parity,
        'persistent_state_unchanged_after_inference': True, 'native_decoder_dtype': 'torch.bfloat16',
        'rank_zero': True, 'bit_budget': budget, 'source_bit_budget': source_budget,
        'export': runtime_audit, 'weight_stat': file_stat(output / 'pytorch_model.bin'),
        'metadata_sha256': sha256(output / 'checkpoint_meta.json'), 'elapsed_seconds': time.time() - started,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(0),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(0), 'quality_target_evaluated': False}
    write_json(audit_path, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('r0', 'r1'))
    parser.add_argument('--step', required=True, type=int)
    args = parser.parse_args(); export_and_audit(args.launch_record, args.arm, args.step)
