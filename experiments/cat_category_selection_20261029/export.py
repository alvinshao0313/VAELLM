"""Restore and finalize all actual prefix decoders and future-dense adapters."""
from copy import deepcopy
import gc
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from . import sources as s
from e2e_common.full_lora import collect_exact_peft_lora_config, finalize_model_level_lora, iter_named_peft_lora_layers
from litebsq.vae_linear import VAELinear, clear_model_vae_linear_cache
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.config.configs import AuxTrainableConfig
from train_utils.distill_decoder import NamedMainDecoderTarget, finalize_main_decoder_targets
from train_utils.distill_precision import configure_distill_precision, prepare_model_export
from train_utils.model_level_checkpoint_state import collect_model_level_mutable_state, restore_model_level_mutable_state
from train_utils.model_level_trainables import build_model_level_trainable_selection


def compare_endpoint(actual, expected, dense_keys):
    s.require(set(actual) == set(expected), 'A500 control persistent inventory differs.')
    differences = []
    for name, value in actual.items():
        reference = expected[name]
        s.require(value.dtype == reference.dtype and value.shape == reference.shape, 'A500 control dtype/shape differs: ' + name)
        if torch.equal(value, reference):
            continue
        s.require(name in dense_keys and value.dtype == torch.bfloat16,
                  'Difference outside CPU/GPU BF16 dense-LoRA fusion: ' + name)
        delta2 = reference2 = 0.0
        changed = 0
        maximum = 0.0
        for left, right in zip(value.reshape(-1).split(1 << 20), reference.reshape(-1).split(1 << 20), strict=True):
            delta = left.float() - right.float()
            changed += int(torch.count_nonzero(delta))
            delta2 += float(delta.double().square().sum())
            reference2 += float(right.double().square().sum())
            maximum = max(maximum, float(delta.abs().max()))
        differences.append(dict(name=name, changed_elements=changed, max_abs=maximum,
                                squared_error=delta2, reference_squared_norm=reference2,
                                relative_l2=(delta2 / reference2) ** .5))
    return dict(tensors_compared=len(actual), all_tensors_exact=not differences,
                nonmerge_tensors_exact=len(actual) - len(dense_keys), dense_merge_differences=differences,
                total_changed_elements=sum(row['changed_elements'] for row in differences),
                max_abs=max((row['max_abs'] for row in differences), default=0.),
                boundary='Only ordinary CPU/GPU BF16 dense fusion may differ. Measured without compensation or arbitrary quality threshold; no forward or quality equivalence claimed.')


def restore_finalize(step, raw_meta, raw, node_meta, stage, training):
    contract = node_meta['immutable_resume_contract']
    checks = s.contract_config_checks(contract, stage, training)
    model, original = s.build_raw(raw_meta, raw)
    decoders = tuple((name, model.get_submodule(name)) for name in contract['decoder_target_names'])
    dense = tuple(contract['lora_target_names'])
    s.require(all(isinstance(layer, VAELinear) for _, layer in decoders), 'Actual native prefix target missing.')
    lora = contract['lora']
    selected = build_model_level_trainable_selection(model, aux=AuxTrainableConfig(**contract['aux']),
        compressed_modules=(), dense_target_modules=dense, decoder_modules=decoders,
        rank=lora['rank'], alpha=lora['alpha'], dropout=lora['dropout'], rank_explicit=True,
        fp32_components=contract['optimization']['distill_fp32_components'], train_decoder=True, train_lora=True, freeze=True)
    model = selected.peft_model
    precision = SimpleNamespace(**contract['precision'])
    configure_distill_precision(selected, components=contract['optimization']['distill_fp32_components'], training_args=precision)
    actual_lora = collect_exact_peft_lora_config(model, default_rank=lora['rank'], alpha=lora['alpha'], dropout=lora['dropout'])
    s.require(actual_lora == node_meta['lora_config'] == lora, 'Actual reconstructed LoRA topology differs from immutable source.')
    mutable, manifest = s.native.load_v6_training_model_state(str(s.NODES / f'gate{step}'), map_location='cpu')
    restore_model_level_mutable_state(model, selection=selected, selected_vae_modules=decoders,
                                    checkpoint_state=mutable, checkpoint_manifest=manifest)
    restored, classes, actual_manifest = collect_model_level_mutable_state(model, selection=selected, selected_vae_modules=decoders)
    s.equal(restored, mutable, 'Complete real FP32 mutable restoration')
    s.require(all(value.dtype == torch.float32 for value in restored.values()), 'Restore the actual FP32 D/LoRA leaves.')
    decoder_values = {entry['name'].removeprefix('base_model.model.'): mutable[entry['name']]
                      for entry in manifest if entry['component_class'] == 'decoder'}
    lora_leaves = sum(entry['component_class'] == 'lora' for entry in manifest)
    s.require(len(decoders) == len(raw_meta['compressed_targets']) == 180
              and len(dense) == len(list(iter_named_peft_lora_layers(model))) == 72
              and len(decoder_values) == len(selected.decoder_parameters) == 1080
              and lora_leaves == len(selected.lora_parameters) == 144 and len(restored) == 1224,
              'Restore all actual180 prefix D and all72 future dense adapters; none may be omitted.')
    print(dict(event='all_FP32_mutable_restored', step=step, decoder_leaves=len(decoder_values), lora_leaves=lora_leaves), flush=True)
    finalized = finalize_main_decoder_targets(tuple(NamedMainDecoderTarget(name, layer) for name, layer in decoders))
    s.require(finalized == len(decoders), 'Native prefix finalization omitted a projection.')
    model = finalize_model_level_lora(model, compressed_proxy_names=None)
    prepare_model_export(model, precision)
    model.eval().requires_grad_(False)
    clear_model_vae_linear_cache(model)
    s.require(not list(iter_named_peft_lora_layers(model)) and all(not layer.trainable_decode for _, layer in decoders),
              'Live PEFT or trainable decoder topology remains after deployment finalization.')
    actual = model.state_dict()
    for name, value in decoder_values.items():
        s.require(name in actual and torch.equal(actual[name], value.to(dtype=actual[name].dtype)),
                  'Actual packed decoder FP32→BF16 export differs: ' + name)
    dense_keys = {name + '.weight' for name in dense}
    changed = set(decoder_values) | dense_keys
    frozen = {name: value for name, value in original.items() if name not in changed}
    s.require(set(actual) == set(original), 'Finalization changed persistent field inventory.')
    frozen_count = s.equal({name: actual[name] for name in frozen}, frozen, 'All persistent frozen fields and packed banks')
    dense_changed = sum(not torch.equal(actual[name], original[name]) for name in dense_keys)
    s.require(dense_changed == len(dense_keys), 'An actual trained future-dense merge was omitted.')
    s.cpu_only(model)
    info = dict(actual_updates=step, planned_steps=contract['optimization']['steps'], raw_tensor_count=len(original),
        raw_tensor_digest=raw['source_raw_state_digest'], mutable_tensors_exact=len(restored), decoder_fp32_leaves=1080,
        lora_fp32_leaves=lora_leaves, prefix_decoders_finalized=finalized, dense_lora_merged=len(dense),
        dense_weights_changed=dense_changed, decoder_BF16_export_exact=1080, frozen_persistent_fields_exact=frozen_count,
        actual_mutable_digest=s.digest(mutable), source_immutable_contract_checks=checks,
        training_steps_executed=0, Trainer_constructed=False, finalization_device='cpu',
        all_parameters_frozen=all(not value.requires_grad for value in model.parameters()))
    del selected, restored, mutable, original, decoder_values, frozen
    gc.collect()
    return model, info, dense_keys


def budget(meta, state):
    result = audit_state_dict(meta, state, max_bpw=100., max_lora_rank=0)
    s.require(not result['contract_violations'], 'Actual deployment storage contract violation.')
    s.require(all(result['components'].get(key, {}).get('storage_bits', 0) == 0 for key in ('lora', 'residual_lora')),
              'Deployed adapters remain outside the future-dense merge.')
    return dict(final_budget_applicable=False, pending_dense_count=len(meta['pending_dense_targets']),
                final_limit_bpw=2.6, intermediate_storage_audit=result)


def save250(model, raw_meta, node_meta, info):
    runtime = torch.load(s.RAW / 'cat_runtime_state.pt', map_location='cpu', weights_only=True)
    extra = deepcopy(raw_meta['extra_meta'])
    history = deepcopy(extra['distill_stage_history'])
    s.require(len(history) == 4 and sum(item['did_train'] for item in history) == 4,
              'Preserve the four real prior recovery rounds before this gate node.')
    gate = dict(mode=node_meta['after_category_mode'], category='gate_proj', did_train=True,
        actual_updates=250, planned_steps=500, source_training_checkpoint_id=node_meta['checkpoint_id'],
        round_base_checkpoint_id=s.RAW_ID, selection_status='pending_read_only_full8_pair',
        prefix_decoder_target_count=180, remaining_lora_target_count=72,
        source_immutable_contract=node_meta['immutable_resume_contract'])
    history.append(gate)
    extra.update(stage='after_category', category='gate_proj', active_category=None, round_idx=5, distill_stage=gate, distill_stage_history=history)
    provenance = dict(purpose='Read-only A250 finalized candidate, not selected and not a500-completed or exact-resume state.',
        control_structure_fixed_before_evaluation='A D32/K32/S2 asymmetric gate',
        source_training_checkpoint_id=node_meta['checkpoint_id'], source_round_base_checkpoint_id=s.RAW_ID,
        original_round_base_ref=node_meta['round_base_ref'], actual_source_resolution='Preserved raw36gate archive + common after_o; missing/prunable original raw path is not read.',
        source_node=str(s.NODES / 'gate250'), source_raw_archive=str(s.RAW), common_after_o=str(s.BASE),
        actual_updates=250, planned_steps=500, selection_status='pending_read_only_full8_pair',
        immutable_contract_source_sha256=s.sha(s.NODES / 'gate250/checkpoint_meta.json'),
        restoration=info, runtime_sidecar='Actual raw A statistics, valid because original channel refresh after category is disabled.',
        source_sha256=s.own_pins())
    extra['selected_step_export'] = provenance
    tokenizer = AutoTokenizer.from_pretrained(node_meta['base_model_path'], use_fast=True, local_files_only=True)
    s.require(tokenizer.pad_token_id == s.read(s.CONTROL / 'config.json')['pad_token_id'] == 151643,
              'Original tokenizer/control padding identity differs.')
    model.config.pad_token_id = tokenizer.pad_token_id
    destination = s.DATA / 'A250/model'
    destination.parent.mkdir(parents=True, exist_ok=True)
    saved = s.native.save_v6_full_checkpoint(model, str(destination), checkpoint_kind='category_boundary',
        compressed_targets=raw_meta['compressed_targets'], pending_dense_targets=raw_meta['pending_dense_targets'],
        skip_targets=raw_meta['skip_targets'], legacy_original_only_sources=raw_meta['legacy_original_only_sources'],
        completed_categories=[*raw_meta['completed_categories'], 'gate_proj'],
        compression_categories=raw_meta['compression_categories'], target_layers=raw_meta['target_layers'],
        target_modules=raw_meta['target_modules'], train_mode='none', after_category_mode=node_meta['after_category_mode'],
        norm_train_mode='none', lm_head_train_mode='none', lora_config=None,
        resolved_learning_rates=node_meta['resolved_learning_rates'], base_model_path=node_meta['base_model_path'],
        finalized_status=dict(lora_finalized=True, decoder_finalized=True, aux_finalized=True,
                              stable_category_boundary=True, inference_ready=False),
        tokenizer=tokenizer, cat_runtime_state=runtime, extra_meta=extra)
    meta = s.native.load_v6_meta(str(destination))
    serialized = torch.load(destination / meta['state_dict_file'], weights_only=True, map_location='cpu', mmap=True)
    count = s.equal(model.state_dict(), serialized, 'Serialized complete A250 deployment')
    s.require(saved['checkpoint_id'] == meta['checkpoint_id'] not in (s.RAW_ID, node_meta['checkpoint_id']), 'A250 requires a new deployment UUID.')
    return dict(checkpoint_dir=str(destination), checkpoint_id=meta['checkpoint_id'],
        source_node_checkpoint_id=node_meta['checkpoint_id'], raw_parent_checkpoint_id=s.RAW_ID,
        serialized_tensors_exact=count, state_digest=s.digest(serialized),
        checkpoint_weight_stat=s.stat(destination / meta['state_dict_file']), checkpoint_meta_sha256=s.sha(destination / 'checkpoint_meta.json'),
        bit_budget=budget(meta, serialized), restoration=info)


def control500(model, raw_meta, node_meta, info, dense_keys):
    meta = s.native.load_v6_meta(str(s.CONTROL))
    s.require(meta['completed_categories'] == [*raw_meta['completed_categories'], 'gate_proj']
              and meta['compressed_targets'] == raw_meta['compressed_targets']
              and meta['pending_dense_targets'] == raw_meta['pending_dense_targets']
              and meta['lora_config'] is None and meta['immutable_resume_contract'] is None,
              'Use the real already-finalized A500 after_gate_proj control.')
    saved = torch.load(s.CONTROL / meta['state_dict_file'], weights_only=True, map_location='cpu', mmap=True)
    comparison = compare_endpoint(model.state_dict(), saved, dense_keys)
    s.require(len(meta['extra_meta']['distill_stage_history']) == 5
              and meta['extra_meta']['distill_stage_history'][:4] == raw_meta['extra_meta']['distill_stage_history'],
              'A500 control recovery history differs from this real raw parent.')
    return dict(checkpoint_dir=str(s.CONTROL), checkpoint_id=meta['checkpoint_id'],
        source_node_checkpoint_id=node_meta['checkpoint_id'], raw_parent_checkpoint_id=s.RAW_ID,
        state_digest=s.digest(saved), checkpoint_weight_stat=s.stat(s.CONTROL / meta['state_dict_file']),
        checkpoint_weight_sha256=s.sha(s.CONTROL / meta['state_dict_file']),
        checkpoint_meta_sha256=s.sha(s.CONTROL / 'checkpoint_meta.json'),
        bit_budget=budget(meta, saved), restoration=info, CPU_recomputed_vs_original_GPU_export=comparison,
        duplicate_full500_model_saved=False)


def verify(endpoint, expected_digest):
    from train_utils.cat_train_runtime import load_cat_resume_distill_progress
    model, meta = s.load_deployment(endpoint)
    state = torch.load(Path(endpoint) / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    count = s.equal(model.state_dict(), state, 'Independent fresh deployment')
    s.require(s.digest(state) == expected_digest, 'Independent fresh complete tensor digest differs.')
    progress = load_cat_resume_distill_progress(str(endpoint))
    s.require(progress.lora_round_idx == 5 and list(progress.completed_categories) == meta['completed_categories'],
              'Independent loader lost the original recovery history/next round identity.')
    s.require(meta['immutable_resume_contract'] is None and meta['lora_config'] is None, 'Deployment must not claim old training-step exact-resume.')
    s.cpu_only(model)
    return dict(status='fresh_independent_native_exact', checkpoint_id=meta['checkpoint_id'],
        persistent_fields_exact=count, all_parameters_frozen=all(not value.requires_grad for value in model.parameters()),
        complete_state_digest=expected_digest, cat_lora_round_idx=progress.lora_round_idx,
        completed_categories=list(progress.completed_categories), next_recovery_seed=31 + progress.lora_round_idx,
        source_weight_payloads_read=[str(Path(endpoint) / meta['state_dict_file'])],
        model_constructed_from_own_config=True, CUDA_initialized=torch.cuda.is_initialized(),
        forward_executed=False, FA2_policy=model.config._attn_implementation)
