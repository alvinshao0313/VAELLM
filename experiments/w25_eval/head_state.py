"""Native head-only contracts and state checks for the actual Qwen/Llama inventory."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

from compressed_e2e_fintuning.v6_runtime_state import validate_e2e_immutable_resume_contract
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _require
from experiments.qwen3_w25.audit_e2e_decoder import _decoder_storage
from experiments.qwen3_w25.e2e_fullhead_export import HEAD_KEY, assert_head_cast
from experiments.qwen3_w25.export_e2e_selected import _read_json
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_layers, selected_modules, validate_recipe
from train_utils.checkpoint_v6 import load_v6_training_step_meta, resolve_training_step_round_base_ref


def validate_configuration(meta, state, snapshot, architecture):
    contract = meta['immutable_resume_contract']
    cfg, training = snapshot['canonical_config'], snapshot['training_args']
    _require(meta['train_mode'] == contract['train_mode'] == cfg['train_mode'] == 'none',
             'Head-only export requires native train_mode=none, not zero decoder LR.')
    _require(meta['lora_config'] is None and contract['lora'] is None, 'LoRA is forbidden.')
    for key, value in (('norm_train_mode', 'none'), ('lm_head_train_mode', 'full')):
        _require(meta[key] == contract['aux'][key] == cfg['aux'][key] == value, f'Expected {key}={value}.')
    _require(cfg['aux']['residual_lora_mode'] == 'none', 'Residual LoRA is forbidden.')
    _require(contract['runtime']['parallel_mode'] == 'dp' and contract['runtime']['offload_mode'] == 'none'
             and not contract['runtime']['distill_hif4_act'] and contract['distributed']['world_size'] == 1,
             'Use native single-process DP without student offload or HiFloat4.')
    _require(contract['precision']['bf16'] and not contract['precision']['fp16']
             and contract['optimization']['distill_fp32_components'] == ['lm_head'],
             'Only the head is promoted to FP32; compute/export use BF16.')
    _require('proxy_coordinates' not in contract['sparse_bit'], 'Head-only training cannot own active scores.')
    _require(type(state['global_step']) is int and 0 < state['global_step'] <= state['max_steps']
             and state['max_steps'] == training['max_steps'] == contract['optimization']['steps'],
             'Saved step or original training schedule disagrees.')
    _require(meta['target_layers'] == contract['target_layers']
             == list(selected_layers(cfg['target_layers'], architecture))
             and meta['target_modules'] == contract['target_modules']
             == list(selected_modules(cfg['target_modules'], architecture)), 'Native target metadata disagrees.')
    _require(training['seed'] == contract['data']['seed'], 'Training seed disagrees.')
    # Preserve runtime-resolved workers and identities, and compare the original
    # normalized mathematical configuration using the production contract checker.
    expected = deepcopy(contract)
    for key in ('train_mode', 'data', 'loss', 'runtime'):
        expected[key] = deepcopy(cfg[key])
    expected['input_checkpoint_id'] = meta['round_base_checkpoint_id']
    expected['optimization'] = {key: value for key, value in cfg['opt'].items() if key != 'logging_steps'}
    expected['aux'] = {key: value for key, value in cfg['aux'].items() if not key.startswith('residual_lora_')}
    expected['precision'] = {key: training[key] for key in contract['precision']}
    expected['sparse_bit'] = {key: cfg['bit_' + key] for key in ('active_ratio', 'optimizer', 'weight_decay', 'round_steps')}
    expected['sparse_bit']['bit_lr'] = cfg['bit_lr']
    for key in contract['evaluation_execution']:
        value = cfg['runtime']['evaluation'][key] if key == 'eval_after_save' else training[key]
        if key == 'eval_strategy':
            value = str(IntervalStrategy(value)).lower()
        elif key == 'save_strategy':
            value = str(SaveStrategy(value)).lower()
        expected['evaluation_execution'][key] = value
    validate_e2e_immutable_resume_contract(contract, expected)
    return contract


def inspect_step(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    meta = load_v6_training_step_meta(str(checkpoint))
    source, base_meta = resolve_training_step_round_base_ref(str(checkpoint), meta)
    _require(checkpoint.parent.name == 'trainer_state', 'Select an original native training checkpoint.')
    snapshot = _read_json(checkpoint.parent.parent / 'normalized_e2e_runtime_args.json')
    state = _read_json(checkpoint / 'trainer_state.json')
    architecture = model_inventory(base_meta['base_model_path'])
    validate_configuration(meta, state, snapshot, architecture)
    validate_recipe(base_meta, architecture)
    _require(set(meta['compressed_targets']) == set(architecture['projection_shapes'])
             and not meta['pending_dense_targets'] and not meta['skip_targets'], 'Full model coverage changed.')
    _require(Path(meta['base_model_path']).resolve() == Path(base_meta['base_model_path']).resolve()
             and not base_meta.get('post_norm_head_linear'), 'Source identity/head topology differs.')
    _require(not (checkpoint / 'sparse_bit_tuning').exists(), 'Head-only checkpoint contains a bit sidecar.')
    return meta, state, snapshot, Path(source), base_meta, architecture


def load_head_state(checkpoint):
    from train_utils.checkpoint_v6 import load_v6_training_model_state
    mutable, manifest = load_v6_training_model_state(str(checkpoint), map_location='cpu')
    _require(set(mutable) == {HEAD_KEY} and len(manifest) == 1
             and {row['name']: row['component_class'] for row in manifest} == {HEAD_KEY: 'lm_head'}
             and mutable[HEAD_KEY].dtype == torch.float32, 'The sole mutable payload must be the FP32 original head.')
    return mutable, manifest


def restore_head_state(model, selected, *, checkpoint, snapshot):
    from compressed_e2e_fintuning.v6_runtime_state import collect_e2e_mutable_state, restore_e2e_mutable_state
    from train_utils.config.configs import AuxTrainableConfig
    from train_utils.distill_precision import configure_distill_precision
    from train_utils.model_level_trainables import build_model_level_trainable_selection
    cfg = snapshot['canonical_config']
    _require(cfg['train_mode'] == 'none' and cfg['aux']['lm_head_train_mode'] == 'full'
             and cfg['aux']['norm_train_mode'] == cfg['aux']['residual_lora_mode'] == 'none'
             and cfg['opt']['distill_fp32_components'] == ['lm_head'], 'Unexpected restore components.')
    _require(type(model.lm_head) is torch.nn.Linear and model.lm_head.bias is None
             and not model.config.tie_word_embeddings
             and model.lm_head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr(),
             'Require the original untied Linear head.')
    selection = build_model_level_trainable_selection(
        model, aux=AuxTrainableConfig(**cfg['aux']), compressed_modules=selected,
        rank=cfg['lora']['rank'], alpha=cfg['lora']['alpha'], dropout=cfg['lora']['dropout'],
        fp32_components=('lm_head',), train_decoder=False, train_lora=False, freeze=True)
    _require(selection.peft_model is model and len(selection.lm_head_parameters) == 1
             and not any((selection.decoder_parameters, selection.norm_parameters,
                          selection.lora_parameters, selection.residual_lora_parameters)),
             'Native trainable selection escaped the head.')
    configure_distill_precision(selection, components=('lm_head',),
                               training_args=SimpleNamespace(**snapshot['training_args']))
    mutable, manifest = load_head_state(checkpoint)
    restore_e2e_mutable_state(model, selection=selection, selected_vae_modules=selected,
                             checkpoint_state=mutable, checkpoint_manifest=manifest)
    restored, classes, _ = collect_e2e_mutable_state(model, selection=selection, selected_vae_modules=selected)
    _require(classes == {HEAD_KEY: 'lm_head'}, 'Native mutable inventory differs from head-only state.')
    return {'tensors_compared': _assert_state_equal(restored, mutable, 'Native FP32 head restore'),
            'component_counts': {'lm_head': 1}, 'optimizer_restored': False}


def audit_head_state(original, actual, inventory, banks, selected_head):
    _require(original.keys() == actual.keys() and HEAD_KEY in original, 'Complete persistent inventory changed.')
    decoder_keys = {key for keys in inventory.values() for key in keys}
    code_keys = {key for rows in banks.values() for key, _shape in rows}
    _require(decoder_keys | code_keys <= original.keys() and HEAD_KEY not in decoder_keys | code_keys,
             'Invalid persistent decoder/code inventory.')
    frozen = {key: value for key, value in original.items() if key != HEAD_KEY}
    compared = _assert_state_equal({key: actual[key] for key in frozen}, frozen, 'All non-head bytes frozen')
    before, after = original[HEAD_KEY], actual[HEAD_KEY]
    _require(before.dtype == after.dtype == torch.bfloat16 and before.shape == after.shape,
             'Head shape/dtype must remain the original BF16 representation.')
    assert_head_cast(after, selected_head)
    fp32_changed = bf16_changed = 0
    fp32_square = bf16_square = 0.0
    digest = hashlib.sha256()
    for start in range(0, before.shape[0], 1024):
        old, saved, new = before[start:start + 1024], selected_head[start:start + 1024], after[start:start + 1024]
        _require(bool(torch.isfinite(old).all()), 'Non-finite source head.')
        fp32_changed += int(torch.count_nonzero(saved != old.float()))
        bf16_changed += int(torch.count_nonzero(new != old))
        fp32_square += float((saved.double() - old.double()).square().sum())
        bf16_square += float((new.double() - old.double()).square().sum())
        digest.update(saved.detach().cpu().contiguous().numpy().tobytes())
    _require(fp32_changed > 0, 'No actual FP32 head update in the saved step.')
    storages = [_decoder_storage(state, decoder_keys) for state in (original, actual)]
    return {'train_mode': 'none', 'lm_head_train_mode': 'full', 'mutable_tensor_count': 1,
        'frozen_tensors_compared': compared, 'frozen_bytes_unchanged': True,
        'decoder_tensor_count': len(decoder_keys), 'changed_decoder_elements': 0,
        'code_bank_count': len(code_keys), 'changed_code_bits': 0, 'packed_codes_unchanged': True,
        'decoder_storage_source': storages[0], 'decoder_storage_target': storages[1], 'decoder_storage_saved_bits': 0,
        'head_shape': list(after.shape), 'head_dtype': str(after.dtype),
        'head_storage_bytes': after.untyped_storage().nbytes(), 'head_fusion_performed': False,
        'changed_head_fp32_elements': fp32_changed, 'changed_head_bf16_elements': bf16_changed,
        'head_fp32_delta_l2': fp32_square ** .5, 'head_bf16_delta_l2': bf16_square ** .5,
        'head_update_survived_bf16': bf16_changed > 0, 'head_matches_selected_fp32_cast_exactly': True,
        'head_selected_fp32_sha256': digest.hexdigest(), 'tensor_schema_unchanged': True}
