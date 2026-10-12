"""Native all-decoder plus backbone-norm mutable-state boundary."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from compressed_e2e_fintuning.v6_runtime_state import validate_e2e_immutable_resume_contract
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _require
from experiments.qwen3_w25.export_e2e_selected import _read_json
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.norm_state import norm_parameters, norm_shapes, assert_norm_cast
from experiments.w25_eval.subset_state import selected_layers, selected_modules, selected_names, validate_recipe, persistent_inventory, checkpoint_inventory
from train_utils.checkpoint_v6 import load_v6_training_step_meta, resolve_training_step_round_base_ref, load_v6_training_model_state


def validate_configuration(meta, state, snapshot, architecture):
    contract = meta['immutable_resume_contract']
    cfg, training = snapshot['canonical_config'], snapshot['training_args']
    _require(meta['train_mode'] == contract['train_mode'] == cfg['train_mode'] == 'decoder', 'Require native decoder training.')
    _require(meta['lora_config'] is None and contract['lora'] is None, 'LoRA is forbidden.')
    for key, value in (('norm_train_mode', 'all'), ('lm_head_train_mode', 'none')):
        _require(meta[key] == contract['aux'][key] == cfg['aux'][key] == value, f'Expected {key}={value}.')
    _require(cfg['aux']['residual_lora_mode'] == 'none', 'Residual LoRA is forbidden.')
    _require(contract['runtime']['parallel_mode'] == 'dp' and contract['runtime']['offload_mode'] == 'none'
             and not contract['runtime']['distill_hif4_act'] and contract['distributed']['world_size'] == 1,
             'Require native single-process DP without student offload or HiFloat4.')
    _require(contract['precision']['bf16'] and not contract['precision']['fp16']
             and contract['optimization']['distill_fp32_components'] == ['decoder', 'norm'], 'Require FP32 decoder and norm masters.')
    _require('proxy_coordinates' not in contract['sparse_bit'], 'Frozen-code training cannot own active scores.')
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
    _require(not (checkpoint / 'sparse_bit_tuning').exists(), 'Decoder/norm checkpoint contains bit state.')
    return meta, state, snapshot, Path(source), base_meta, architecture


def expected_inventory(source, architecture):
    from experiments.qwen3_w25.assemble_categories import _load_state
    from experiments.w25_eval.validate_decoder_subset_training import expected_inventory as decoder_inventory
    names = tuple(architecture['projection_shapes'])
    result = decoder_inventory(source, architecture, names)
    original = _load_state(Path(source))
    shapes = norm_shapes(architecture['base_model_path'])
    decoder_keys = {key for name in names for key in original if key.startswith(name + '._parallel_stage_decoder.')}
    _require(not decoder_keys & shapes.keys() and all(key in original and original[key].dtype == torch.bfloat16
        and list(original[key].shape) == shape for key, shape in shapes.items()), 'Native norm shape/storage differs.')
    result.update(norm_shapes=shapes, norm_tensors=len(shapes), norm_elements=sum(original[key].numel() for key in shapes),
                  mutable_tensors=len(decoder_keys) + len(shapes),
                  frozen_tensors=len(original) - len(decoder_keys) - len(shapes))
    return result


def load_union_state(checkpoint, decoder_shapes, expected_norm_shapes):
    mutable, manifest = load_v6_training_model_state(str(checkpoint), map_location='cpu')
    expected = {**{key: 'decoder' for key in decoder_shapes}, **{key: 'norm' for key in expected_norm_shapes}}
    shapes = {**decoder_shapes, **expected_norm_shapes}
    _require(not decoder_shapes.keys() & expected_norm_shapes.keys() and set(mutable) == set(expected)
             and len(manifest) == len(expected)
             and {row['name']: row['component_class'] for row in manifest} == expected,
             'Mutable checkpoint must contain exactly all decoder and native backbone norm leaves.')
    _require(all(t.dtype == torch.float32 and list(t.shape) == shapes[key] and bool(torch.isfinite(t).all())
                 for key, t in mutable.items()), 'Mutable shape/dtype/value differs.')
    return mutable, manifest


def restore_union_state(model, selected, *, checkpoint, snapshot, architecture):
    from compressed_e2e_fintuning.v6_runtime_state import collect_e2e_mutable_state, restore_e2e_mutable_state
    from train_utils.config.configs import AuxTrainableConfig
    from train_utils.distill_precision import configure_distill_precision
    from train_utils.model_level_trainables import build_model_level_trainable_selection
    cfg = snapshot['canonical_config']
    inventory, _banks = persistent_inventory(model, architecture)
    params = dict(model.named_parameters())
    decoder_keys = {key for keys in inventory.values() for key in keys}
    norms = norm_parameters(model)
    selection = build_model_level_trainable_selection(model, aux=AuxTrainableConfig(**cfg['aux']),
        compressed_modules=selected, rank=cfg['lora']['rank'], alpha=cfg['lora']['alpha'], dropout=cfg['lora']['dropout'],
        fp32_components=('decoder', 'norm'), train_decoder=True, train_lora=False, freeze=True)
    _require(selection.peft_model is model
             and {id(p) for p in selection.decoder_parameters.values()} == {id(params[key]) for key in decoder_keys}
             and {id(p) for p in selection.norm_parameters.values()} == {id(p) for p in norms.values()}
             and not any((selection.lm_head_parameters, selection.lora_parameters, selection.residual_lora_parameters)),
             'Native trainable selection escaped the decoder/norm union.')
    configure_distill_precision(selection, components=('decoder', 'norm'), training_args=SimpleNamespace(**snapshot['training_args']))
    mutable, manifest = load_union_state(checkpoint, {key: list(params[key].shape) for key in decoder_keys},
                                       {key: list(p.shape) for key, p in norms.items()})
    restore_e2e_mutable_state(model, selection=selection, selected_vae_modules=selected,
                             checkpoint_state=mutable, checkpoint_manifest=manifest)
    restored, classes, _ = collect_e2e_mutable_state(model, selection=selection, selected_vae_modules=selected)
    _require(classes == {**{key: 'decoder' for key in decoder_keys}, **{key: 'norm' for key in norms}},
             'Native restored component manifest differs.')
    return {'tensors_compared': _assert_state_equal(restored, mutable, 'Native FP32 decoder/norm restore'),
            'component_counts': {'decoder': len(decoder_keys), 'norm': len(norms)}, 'optimizer_restored': False}
