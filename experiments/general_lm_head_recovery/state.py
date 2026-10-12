"""Native all-decoder plus original dense-head mutable-state boundary."""
from copy import deepcopy
from pathlib import Path
import torch
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from compressed_e2e_fintuning.v6_runtime_state import validate_e2e_immutable_resume_contract
from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen3_w25.export_e2e_selected import _read_json
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.norm_state import norm_shapes
from experiments.qwen3_w25.e2e_fullhead_export import HEAD_KEY, assert_head_cast, restore_fullhead_state
from experiments.w25_eval.subset_state import selected_layers, selected_modules, selected_names, validate_recipe, persistent_inventory, checkpoint_inventory
from train_utils.checkpoint_v6 import load_v6_training_step_meta, resolve_training_step_round_base_ref, load_v6_training_model_state


def validate_configuration(meta, state, snapshot, architecture):
    contract = meta['immutable_resume_contract']
    cfg, training = snapshot['canonical_config'], snapshot['training_args']
    _require(meta['train_mode'] == contract['train_mode'] == cfg['train_mode'] == 'decoder', 'Require native decoder training.')
    _require(meta['lora_config'] is None and contract['lora'] is None, 'LoRA is forbidden.')
    for key, value in (('norm_train_mode', 'none'), ('lm_head_train_mode', 'full')):
        _require(meta[key] == contract['aux'][key] == cfg['aux'][key] == value, f'Expected {key}={value}.')
    _require(cfg['aux']['residual_lora_mode'] == 'none', 'Residual LoRA is forbidden.')
    _require(contract['runtime']['parallel_mode'] == 'dp' and contract['runtime']['offload_mode'] == 'none'
             and not contract['runtime']['distill_hif4_act'] and contract['distributed']['world_size'] == 1,
             'Require native single-process DP without student offload or HiFloat4.')
    _require(contract['precision']['bf16'] and not contract['precision']['fp16']
             and contract['optimization']['distill_fp32_components'] == ['decoder', 'lm_head'], 'Require FP32 decoder and original head masters.')
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
    _require(not (checkpoint / 'sparse_bit_tuning').exists(), 'Decoder/head checkpoint contains bit state.')
    return meta, state, snapshot, Path(source), base_meta, architecture


def expected_inventory(source, architecture):
    from experiments.qwen3_w25.assemble_categories import _load_state
    from experiments.w25_eval.validate_decoder_subset_training import expected_inventory as decoder_inventory
    names = tuple(architecture['projection_shapes'])
    result = decoder_inventory(source, architecture, names)
    original = _load_state(Path(source))
    config = _read_json(Path(source) / 'config.json')
    shapes = norm_shapes(architecture['base_model_path'])
    head = original[HEAD_KEY]
    _require(config['model_type'] == architecture['model_type'] == 'llama'
             and not config['tie_word_embeddings'] and list(head.shape) == [config['vocab_size'], config['hidden_size']]
             and head.dtype == torch.bfloat16 and all(key in original for key in shapes),
             'Require the native untied Llama head and backbone norm inventory.')
    result.update(head_tensors=1, head_elements=head.numel(), head_shape=list(head.shape),
                  norm_tensors=len(shapes), norm_shapes=shapes,
                  mutable_tensors=result['selected_decoder_tensors'] + 1,
                  frozen_tensors=len(original) - result['selected_decoder_tensors'] - 1)
    return result


def load_union_state(checkpoint, decoder_shapes, head_shape):
    mutable, manifest = load_v6_training_model_state(str(checkpoint), map_location='cpu')
    expected = {**{key: 'decoder' for key in decoder_shapes}, HEAD_KEY: 'lm_head'}
    shapes = {**decoder_shapes, HEAD_KEY: list(head_shape)}
    _require(HEAD_KEY not in decoder_shapes and set(mutable) == set(expected)
             and len(manifest) == len(expected)
             and {row['name']: row['component_class'] for row in manifest} == expected,
             'Mutable state must contain exactly all decoder leaves and the original dense head.')
    _require(all(t.dtype == torch.float32 and list(t.shape) == shapes[key]
                 for key, t in mutable.items()), 'Mutable shape or FP32 storage differs.')
    return mutable, manifest


def restore_union_state(model, selected, *, checkpoint, snapshot, architecture):
    inventory, _banks = persistent_inventory(model, architecture)
    _require(set(name for name, _ in selected) == set(inventory)
             and list(model.lm_head.weight.shape) == [model.config.vocab_size, model.config.hidden_size],
             'Restore the actual complete model inventory and original head shape.')
    result = restore_fullhead_state(model, selected, checkpoint=checkpoint, snapshot=snapshot)
    _require(result['component_counts'] == {'decoder': sum(map(len, inventory.values())), 'lm_head': 1},
             'Restored component inventory differs.')
    return result
