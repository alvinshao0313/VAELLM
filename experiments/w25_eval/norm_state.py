"""Native all-backbone-norm contracts, selected state, and exact export checks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

from compressed_e2e_fintuning.v6_runtime_state import validate_e2e_immutable_resume_contract
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _require
from experiments.qwen3_w25.audit_e2e_decoder import _decoder_storage
from experiments.qwen3_w25.export_e2e_selected import _read_json
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_layers, selected_modules, selected_names, validate_recipe
from train_utils.checkpoint_v6 import load_v6_training_step_meta, resolve_training_step_round_base_ref
from train_utils.model_level_trainables import iter_named_backbone_norm_modules


def norm_parameters(model):
    found = {}
    for name, module in iter_named_backbone_norm_modules(model):
        for relative, parameter in module.named_parameters():
            key = name + '.' + relative
            _require(key not in found, 'Backbone norm parameter appeared twice.')
            found[key] = parameter
    _require(bool(found), 'No native backbone norms found.')
    return found


def norm_shapes(base_model_path):
    config = AutoConfig.from_pretrained(base_model_path, local_files_only=True)
    _require(config.model_type in ('qwen3', 'llama'), 'Unsupported norm inventory architecture.')
    config._attn_implementation = 'sdpa'  # Meta shape discovery only; no inference.
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config)
    found = norm_parameters(model)
    _require(all(p.device.type == 'meta' for p in model.parameters()), 'Norm shape discovery allocated real weights.')
    return {key: list(p.shape) for key, p in found.items()}


def validate_runtime(cfg, args):
    meta = _read_json(Path(cfg.student_checkpoint_dir) / 'checkpoint_meta.json')
    architecture = model_inventory(meta['base_model_path'])
    validate_recipe(meta, architecture)
    _require(set(selected_names(cfg.target_layers, cfg.target_modules, architecture))
             == set(architecture['projection_shapes']), 'Norm-only metadata requires the complete projection inventory.')
    _require(cfg.train_mode == 'none' and cfg.aux.norm_train_mode == 'all'
             and cfg.aux.lm_head_train_mode == cfg.aux.residual_lora_mode == 'none'
             and tuple(cfg.opt.distill_fp32_components) == ('norm',),
             'Only all native backbone norms may train; decoders/codes/head/LoRA must remain frozen.')
    _require(cfg.runtime.parallel_mode == 'dp' and cfg.runtime.offload_mode == 'none'
             and not cfg.runtime.distill_hif4_act and args.world_size == 1,
             'Use single-process native DP without student offload or HiFloat4.')
    _require(not cfg.runtime.evaluation.eval_after_save and not str(cfg.runtime.evaluation.eval_tasks or '').strip(),
             'Export only after this training process exits.')
    _require(cfg.resume_from_checkpoint is None and cfg.stop_after_step is None, 'Require a fresh source without resume.')
    _require(args.bf16 and not args.fp16, 'Require BF16 compute/export and FP32 norm parameters.')
    return meta, architecture


def validate_configuration(meta, state, snapshot, architecture):
    contract = meta['immutable_resume_contract']
    cfg, training = snapshot['canonical_config'], snapshot['training_args']
    _require(meta['train_mode'] == contract['train_mode'] == cfg['train_mode'] == 'none', 'Require native train_mode=none.')
    _require(meta['lora_config'] is None and contract['lora'] is None, 'LoRA is forbidden.')
    for key, value in (('norm_train_mode', 'all'), ('lm_head_train_mode', 'none')):
        _require(meta[key] == contract['aux'][key] == cfg['aux'][key] == value, f'Expected {key}={value}.')
    _require(cfg['aux']['residual_lora_mode'] == 'none', 'Residual LoRA is forbidden.')
    _require(contract['runtime']['parallel_mode'] == 'dp' and contract['runtime']['offload_mode'] == 'none'
             and not contract['runtime']['distill_hif4_act'] and contract['distributed']['world_size'] == 1,
             'Require native single-process DP without student offload or HiFloat4.')
    _require(contract['precision']['bf16'] and not contract['precision']['fp16']
             and contract['optimization']['distill_fp32_components'] == ['norm'], 'Only norms may be promoted to FP32.')
    _require('proxy_coordinates' not in contract['sparse_bit'], 'Norm-only training cannot own active scores.')
    _require(type(state['global_step']) is int and 0 < state['global_step'] <= state['max_steps']
             and state['max_steps'] == training['max_steps'] == contract['optimization']['steps'], 'Saved schedule differs.')
    _require(meta['target_layers'] == contract['target_layers'] == list(selected_layers(cfg['target_layers'], architecture))
             and meta['target_modules'] == contract['target_modules']
             == list(selected_modules(cfg['target_modules'], architecture)), 'Native target metadata differs.')
    _require(training['seed'] == contract['data']['seed'], 'Training seed differs.')
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
    _require(checkpoint.parent.name == 'trainer_state', 'Select the original native training checkpoint.')
    snapshot = _read_json(checkpoint.parent.parent / 'normalized_e2e_runtime_args.json')
    state = _read_json(checkpoint / 'trainer_state.json')
    architecture = model_inventory(base_meta['base_model_path'])
    validate_configuration(meta, state, snapshot, architecture)
    validate_recipe(base_meta, architecture)
    _require(set(meta['compressed_targets']) == set(architecture['projection_shapes'])
             and not meta['pending_dense_targets'] and not meta['skip_targets'], 'Full model coverage differs.')
    _require(Path(meta['base_model_path']).resolve() == Path(base_meta['base_model_path']).resolve()
             and not base_meta.get('post_norm_head_linear'), 'Source identity/head topology differs.')
    _require(not (checkpoint / 'sparse_bit_tuning').exists(), 'Norm-only checkpoint contains bit state.')
    return meta, state, snapshot, Path(source), base_meta, architecture


def load_norm_state(checkpoint, expected_shapes):
    from train_utils.checkpoint_v6 import load_v6_training_model_state
    mutable, manifest = load_v6_training_model_state(str(checkpoint), map_location='cpu')
    _require(set(mutable) == set(expected_shapes) and len(manifest) == len(expected_shapes)
             and {row['name']: row['component_class'] for row in manifest} == {key: 'norm' for key in expected_shapes},
             'The complete mutable payload must contain exactly the native backbone norms.')
    _require(all(value.dtype == torch.float32 and list(value.shape) == expected_shapes[key]
                 and bool(torch.isfinite(value).all()) for key, value in mutable.items()), 'Norm shape/dtype/value differs.')
    return mutable, manifest


def restore_norm_state(model, selected, *, checkpoint, snapshot):
    from compressed_e2e_fintuning.v6_runtime_state import collect_e2e_mutable_state, restore_e2e_mutable_state
    from train_utils.config.configs import AuxTrainableConfig
    from train_utils.distill_precision import configure_distill_precision
    from train_utils.model_level_trainables import build_model_level_trainable_selection
    cfg = snapshot['canonical_config']
    expected = norm_parameters(model)
    selection = build_model_level_trainable_selection(
        model, aux=AuxTrainableConfig(**cfg['aux']), compressed_modules=selected,
        rank=cfg['lora']['rank'], alpha=cfg['lora']['alpha'], dropout=cfg['lora']['dropout'],
        fp32_components=('norm',), train_decoder=False, train_lora=False, freeze=True)
    _require(selection.peft_model is model
             and {id(p) for p in selection.norm_parameters.values()} == {id(p) for p in expected.values()}
             and not any((selection.decoder_parameters, selection.lm_head_parameters,
                          selection.lora_parameters, selection.residual_lora_parameters)), 'Native selection escaped norms.')
    configure_distill_precision(selection, components=('norm',), training_args=SimpleNamespace(**snapshot['training_args']))
    mutable, manifest = load_norm_state(checkpoint, {key: list(p.shape) for key, p in expected.items()})
    restore_e2e_mutable_state(model, selection=selection, selected_vae_modules=selected,
                             checkpoint_state=mutable, checkpoint_manifest=manifest)
    restored, classes, _ = collect_e2e_mutable_state(model, selection=selection, selected_vae_modules=selected)
    _require(classes == {key: 'norm' for key in expected}, 'Native mutable norm inventory differs.')
    return {'tensors_compared': _assert_state_equal(restored, mutable, 'Native FP32 norm restore'),
            'component_counts': {'norm': len(expected)}, 'optimizer_restored': False}


def assert_norm_cast(actual, selected):
    _require(actual.keys() == selected.keys(), 'Selected/exported norm inventory differs.')
    for key, value in actual.items():
        _require(value.dtype == torch.bfloat16 and selected[key].dtype == torch.float32
                 and value.shape == selected[key].shape and bool(torch.isfinite(selected[key]).all()),
                 f'Norm shape/dtype/value differs: {key}')
        _require(torch.equal(value.detach().cpu(), selected[key].detach().cpu().to(torch.bfloat16)),
                 f'Exported norm differs from the selected FP32 cast: {key}')


def audit_norm_state(original, actual, inventory, banks, selected):
    _require(original.keys() == actual.keys() and set(selected) < set(original), 'Persistent inventory differs.')
    decoder_keys = {key for keys in inventory.values() for key in keys}
    code_keys = {key for rows in banks.values() for key, _shape in rows}
    _require(decoder_keys | code_keys <= original.keys() and not set(selected) & (decoder_keys | code_keys),
             'Backbone norm inventory intersects compression state.')
    frozen = {key: value for key, value in original.items() if key not in selected}
    compared = _assert_state_equal({key: actual[key] for key in frozen}, frozen, 'Every non-norm byte frozen')
    assert_norm_cast({key: actual[key] for key in selected}, selected)
    fp32_changed = bf16_changed = 0
    fp32_square = bf16_square = 0.0
    digest = hashlib.sha256()
    for key in sorted(selected):
        before, saved, after = original[key], selected[key], actual[key]
        _require(before.dtype == torch.bfloat16 and before.shape == after.shape
                 and bool(torch.isfinite(before).all()), f'Original norm representation differs: {key}')
        fp32_changed += int(torch.count_nonzero(saved != before.float()))
        bf16_changed += int(torch.count_nonzero(after != before))
        fp32_square += float((saved.double() - before.double()).square().sum())
        bf16_square += float((after.double() - before.double()).square().sum())
        digest.update(key.encode())
        digest.update(saved.detach().cpu().contiguous().numpy().tobytes())
    _require(fp32_changed > 0, 'No real FP32 norm update in the saved step.')
    return {'train_mode': 'none', 'norm_train_mode': 'all', 'lm_head_train_mode': 'none',
        'mutable_tensor_count': len(selected), 'norm_elements': sum(value.numel() for value in selected.values()),
        'frozen_tensors_compared': compared, 'frozen_bytes_unchanged': True,
        'decoder_tensor_count': len(decoder_keys), 'changed_decoder_elements': 0,
        'code_bank_count': len(code_keys), 'changed_code_bits': 0, 'packed_codes_unchanged': True,
        'decoder_storage_source': _decoder_storage(original, decoder_keys),
        'decoder_storage_target': _decoder_storage(actual, decoder_keys), 'decoder_storage_saved_bits': 0,
        'norm_storage_bytes': sum(actual[key].untyped_storage().nbytes() for key in selected),
        'changed_norm_fp32_elements': fp32_changed, 'changed_norm_bf16_elements': bf16_changed,
        'norm_fp32_delta_l2': fp32_square ** .5, 'norm_bf16_delta_l2': bf16_square ** .5,
        'norm_update_survived_bf16': bf16_changed > 0, 'norm_matches_selected_fp32_cast_exactly': True,
        'norm_selected_fp32_sha256': digest.hexdigest(), 'tensor_schema_unchanged': True}
