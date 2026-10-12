"""Shape-derived joint native state boundary; no model-name or projection-count branch."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
from functools import lru_cache

import torch
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

from compressed_e2e_fintuning.v6_runtime_state import validate_e2e_immutable_resume_contract
from experiments.llama_ce_reproduction.train import read
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.lora_state import saved_lora_selection, validate_source, project_lora_state, lora_storage
from train_utils.checkpoint_v6 import _build_decoder_from_spec
from train_utils.checkpoint_bpw import _code_specs
from litebsq.autoencoder import pack_decoders
from experiments.w25_eval.subset_state import validate_recipe
from train_utils.checkpoint_v6 import load_v6_training_step_meta, resolve_training_step_round_base_ref


@lru_cache(maxsize=16)
def _decoder_geometry(serialized_specs):
    # Construct only the small native decoders, with no model weights or RNG side effect.
    with torch.random.fork_rng(devices=[]):
        decoder = pack_decoders([_build_decoder_from_spec(s) for s in json.loads(serialized_specs)])
    return len(decoder.state_dict()), sum(v.numel() for v in decoder.state_dict().values())


def geometry(meta, architecture):
    validate_source(meta, architecture); validate_recipe(meta, architecture)
    specs = {s['name']: s for s in meta['converted_modules']}
    decoders = [_decoder_geometry(json.dumps(s['stage_decoders'], sort_keys=True)) for s in specs.values()]
    lora = lora_storage(architecture['projection_shapes'], rank=2)
    return {'projections': len(specs), 'decoder_tensors': sum(n for n, _ in decoders),
        'decoder_elements': sum(n for _, n in decoders), 'code_banks': sum(len(list(_code_specs(s))) for s in specs.values()),
        'lora_tensors': lora['tensors'], 'lora_elements': lora['parameters'], 'lora_bits': lora['storage_bits']}


def inspect_step(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    meta = load_v6_training_step_meta(str(checkpoint))
    source, base = resolve_training_step_round_base_ref(str(checkpoint), meta)
    architecture = validate_source(base)
    validate_recipe(base, architecture)
    require(checkpoint.parent.name == 'trainer_state' and not (checkpoint / 'sparse_bit_tuning').exists(),
            'Require the original native continuous-state checkpoint, without SparseBit.')
    state = read(checkpoint / 'trainer_state.json')
    snapshot = read(checkpoint.parent.parent / 'normalized_e2e_runtime_args.json')
    cfg, training, contract = snapshot['canonical_config'], snapshot['training_args'], meta['immutable_resume_contract']
    shapes, lora = saved_lora_selection(meta, architecture)
    require(meta['train_mode'] == cfg['train_mode'] == contract['train_mode'] == 'decoder_lora'
            and set(shapes) == set(architecture['projection_shapes'])
            and lora['rank'] == 2 and lora['alpha'] == 4 and lora['dropout'] == 0,
            'Require every actual projection with ordinary rank2/alpha4 topology.')
    require(meta['round_base_checkpoint_id'] == base['checkpoint_id']
            and meta['base_model_path'] == base['base_model_path']
            and meta['lora_config'] == lora and set(meta['compressed_targets']) == set(shapes)
            and not meta['pending_dense_targets'] and not meta['skip_targets'], 'Source or saved scope differs.')
    require(all(cfg['aux'][k] == contract['aux'][k] == meta[k] == 'none'
                for k in ('norm_train_mode', 'lm_head_train_mode'))
            and cfg['aux']['residual_lora_mode'] == contract['aux'].get('residual_lora_mode', 'none') == 'none'
            and cfg['opt']['distill_fp32_components'] == contract['optimization']['distill_fp32_components'] == ['lora', 'decoder']
            and contract['precision']['bf16'] and not contract['precision']['fp16']
            and contract['distributed']['world_size'] == 1 and cfg['runtime']['parallel_mode'] == 'dp',
            'Only FP32 decoder and ordinary LoRA may train, with BF16 computation/export.')
    require(type(state['global_step']) is int and 0 < state['global_step'] <= 2000
            and state['max_steps'] == training['max_steps'] == cfg['opt']['steps'] == 2000
            and training['seed'] == cfg['data']['seed'] == 0
            and all(cfg['lora'][k] == lora[k] for k in ('rank', 'alpha', 'dropout')), 'Saved schedule or rank differs.')
    expected = deepcopy(contract)
    for key in ('train_mode', 'data', 'loss', 'runtime'):
        expected[key] = deepcopy(cfg[key])
    expected['input_checkpoint_id'] = meta['round_base_checkpoint_id']
    expected['optimization'] = {k: v for k, v in cfg['opt'].items() if k != 'logging_steps'}
    expected['aux'] = {k: v for k, v in cfg['aux'].items() if not k.startswith('residual_lora_')}
    expected['precision'] = {k: training[k] for k in contract['precision']}
    expected['sparse_bit'] = {k: cfg['bit_' + k] for k in ('active_ratio', 'optimizer', 'weight_decay', 'round_steps')}
    expected['sparse_bit']['bit_lr'] = cfg['bit_lr']
    for key in contract['evaluation_execution']:
        value = cfg['runtime']['evaluation'][key] if key == 'eval_after_save' else training[key]
        if key == 'eval_strategy': value = str(IntervalStrategy(value)).lower()
        if key == 'save_strategy': value = str(SaveStrategy(value)).lower()
        expected['evaluation_execution'][key] = value
    validate_e2e_immutable_resume_contract(contract, expected)
    return meta, state, snapshot, Path(source), base, architecture


def partition(mutable, manifest, shapes, decoder_count):
    classes = {row['name']: row['component_class'] for row in manifest}
    require(len(classes) == len(manifest) == len(mutable) == decoder_count + 2 * len(shapes) and set(classes) == set(mutable)
            and set(classes.values()) == {'decoder', 'lora'}, 'Unexpected joint mutable inventory.')
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in mutable.values()),
            'Joint mutable tensors must be finite FP32.')
    groups = {kind: {k: mutable[k] for k in mutable if classes[k] == kind} for kind in ('decoder', 'lora')}
    require(len(groups['decoder']) == decoder_count and len(groups['lora']) == 2 * len(shapes), 'Joint leaf counts differ.')
    projected = project_lora_state(groups['lora'], [row for row in manifest if row['component_class'] == 'lora'],
                                   shapes, rank=2, alpha=4)
    return groups, projected


def restore_joint(model, selected, checkpoint, meta, snapshot, architecture):
    from compressed_e2e_fintuning.runtime_v6_pipeline import _enable_internal_decode_runtime
    from compressed_e2e_fintuning.v6_runtime_state import restore_e2e_mutable_state
    from e2e_common.full_lora import collect_exact_peft_lora_config
    from train_utils.checkpoint_v6 import load_v6_training_model_state
    from train_utils.config.configs import AuxTrainableConfig
    from train_utils.distill_precision import configure_distill_precision
    from train_utils.model_level_trainables import build_model_level_trainable_selection
    cfg = snapshot['canonical_config']
    _enable_internal_decode_runtime(selected, decoder_enabled=True, sparse_enabled=False,
                                    vae_decoder_checkpoint=cfg['runtime']['vae_decoder_checkpoint'])
    selection = build_model_level_trainable_selection(model, aux=AuxTrainableConfig(**cfg['aux']),
        compressed_modules=selected, dense_target_modules=(), rank=2, alpha=4, dropout=0, rank_explicit=True,
        fp32_components=('decoder', 'lora'), train_decoder=True, train_lora=True, freeze=True)
    model = selection.peft_model or model
    configure_distill_precision(selection, components=('decoder', 'lora'),
                               training_args=SimpleNamespace(bf16=True, fp16=False))
    require(collect_exact_peft_lora_config(model, default_rank=2, alpha=4, dropout=0) == meta['lora_config'],
            'Reconstructed exact LoRA topology differs.')
    mutable, manifest = load_v6_training_model_state(str(checkpoint), map_location='cpu')
    decoder_count = sum(len(m._parallel_stage_decoder.state_dict()) for _, m in selected)
    groups, factors = partition(mutable, manifest, architecture['projection_shapes'], decoder_count)
    # The native manifest matches actual Parameter identities after PEFT wrapping;
    # no string-prefix rewriting or decoder/LoRA key guessing is used.
    restore_e2e_mutable_state(model, selection=selection, selected_vae_modules=selected,
                             checkpoint_state=mutable, checkpoint_manifest=manifest)
    decoders = {name + '._parallel_stage_decoder.' + key: value.detach().cpu()
                for name, module in selected for key, value in module._parallel_stage_decoder.state_dict().items()}
    require(len(decoders) == len(groups['decoder']) == decoder_count, 'Native decoder mapping is incomplete.')
    return model, decoders, factors
