"""One narrow remaining-dense-LoRA + native/compact prefix-decoder recovery."""
from dataclasses import asdict, replace
from copy import deepcopy
import gc
import importlib.util
import logging
from pathlib import Path
import sys

import torch

from experiments.adaptive_vector_rvq import common as c
from train_utils import cat_after_category_common as canonical
from train_utils import cat_train_pipeline as native
from train_utils.cat_runtime_state_v6 import restore_cat_runtime_state, validate_cat_runtime_identity
from train_utils.distill_precision import configure_distill_precision, install_precision_runtime, prepare_model_export
from train_utils.model_level_trainables import build_model_level_trainable_selection
from train_utils.model_level_optimizer import ModelLevelOptimizerLRConfig, attach_model_level_optimizer_contract, build_model_level_param_groups
from train_utils.distill_decoder import NamedMainDecoderTarget, finalize_main_decoder_targets
from e2e_common.full_lora import finalize_model_level_lora, iter_named_peft_lora_layers, _merge_dense_peft_lora_into_base
from . import RUNTIME
from . import checkpoint
from .archive_cat_gate_prefixes import BASE, BASE_ID, base_source, load_archive, subset
from .consumer import CompactRVQLinear
from .native_prefix import from_native

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path('/data/star/VAELLM/adaptive_native_model_20261029/cat_gate_prefixes/B')
STRUCTURE = ROOT / 'experiments/cat_structure_full_20261029'
PREPARED = Path('/data/star/VAELLM/cat_structure_full_20261029/prepared.json')
CATEGORIES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
log = logging.getLogger(__name__)


def recipe(output):
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(STRUCTURE))
        spec = importlib.util.spec_from_file_location('_mixed_actual_structure_recipe', STRUCTURE / 'recipe.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous
    c.require(Path(native.__file__).resolve().is_relative_to(RUNTIME), 'Use the explicit category-aware N fitter/runtime.')
    runtime = torch.load(ARCHIVE / 'cat_runtime_state.pt', map_location='cpu', weights_only=True)
    saved = c.read(PREPARED)['arms']['B']
    cat, hf, training, vae, cfgs, identity, normalized = module.parse_recipe('B', saved['native_argv'], runtime)
    validate_cat_runtime_identity(runtime['runtime_identity'], identity)
    cat.output_dir = str(output)
    stage = canonical.resolve_cat_after_category_stage(cat, training, category='gate_proj', round_idx=4)
    c.require(stage.mode == 'remaining_lora_prefix_decoder' and stage.config.opt.steps == 500
              and stage.config.opt.batch_size == 4 and stage.config.data.model_max_length == 1024
              and stage.config.opt.gradient_checkpointing and stage.config.data.dynamic_padding
              and stage.config.lora.rank == 12 and stage.config.opt.resolved_decoder_lr() == 1e-5
              and stage.config.opt.learning_rate == 1e-4, 'Keep the original real500/B4/1024/D+remaining-rank12 contract.')
    return cat, hf, training, vae, cfgs, runtime, stage


@torch.no_grad()
def build_raw_mixed():
    base_meta, base_state, names, _targets = base_source()
    gate, specs, archive = load_archive(ARCHIVE)
    c.require(archive['common_after_o']['checkpoint_id'] == BASE_ID and archive['after_o_complement_exact'],
              'Use the verified actual raw B + after_o reconstruction.')
    model = checkpoint.architecture(ARCHIVE / 'raw_config.json', dict(attn_implementation='flash_attention_2', seqlen=2048))
    checkpoint.native._rebuild_converted_modules(model, base_meta['converted_modules'])
    state = subset(base_state, names, False)
    for spec in specs:
        name = spec['name']
        relative = {key[len(name) + 1:]: value for key, value in gate.items() if key.startswith(name + '.')}
        bundle, lengths, resolved = from_native(spec, relative, dict(source_raw_checkpoint_id=archive['source_raw']['checkpoint_id'],
            source_module=name, actual_raw_archive=str(ARCHIVE), purpose='Real raw B fixture only; no structure selection or third stage'))
        layer = CompactRVQLinear(bundle, lengths)
        checkpoint.native.set_module_by_name(model, name, layer)
        state.update({name + '.' + key: tensor for key, tensor in layer.state_dict().items()})
    checkpoint.restore_parameter_dtypes(model, state)
    loaded = model.load_state_dict(state, strict=True)
    checkpoint.native.refresh_vae_linear_runtime_after_state_load(model)
    install_precision_runtime(model)
    groups = checkpoint.inventories(model)
    c.require([len(groups[key]) for key in ('native', 'compact', 'pending_dense')] == [144, 36, 72]
              and not loaded.missing_keys and not loaded.unexpected_keys, 'Incomplete real raw mixed projection inventory.')
    return model.eval().requires_grad_(False), dict(raw_archive=str(ARCHIVE),
        raw_source_checkpoint_id=archive['source_raw']['checkpoint_id'], archive_manifest_sha256=c.sha(ARCHIVE / 'manifest.json'),
        common_after_o_checkpoint_id=BASE_ID, common_after_o=str(BASE), inventories=groups,
        persistent_tensors=len(model.state_dict()), runtime='N category-aware snapshot', original_non_gate_fields_exact=True)


def selection(model, stage):
    groups = checkpoint.inventories(model)
    counts = [len(groups[key]) for key in ('native', 'compact', 'pending_dense')]
    c.require(counts in ([144, 36, 72], [180, 36, 36], [216, 36, 0]),
              'Use the actual gate/up/down boundary with the complete36 compact gates.')
    cfg = stage.config
    native_targets = tuple((name, model.get_submodule(name)) for name in groups['native'])
    compact_targets = tuple((name, model.get_submodule(name)) for name in groups['compact'])
    selected = build_model_level_trainable_selection(model, aux=cfg.aux, fp32_components=cfg.opt.distill_fp32_components,
        compressed_modules=(), dense_target_modules=groups['pending_dense'], decoder_modules=native_targets,
        rank=cfg.lora.rank, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout, rank_explicit=cfg.lora.rank_explicit,
        train_decoder=True, train_lora=bool(groups['pending_dense']), freeze=True)
    ids = {id(value) for value in selected.decoder_parameters.values()}
    compact_leaves = 0
    for name, module in compact_targets:
        module.enable_decoder_training()
        for relative, parameter in module.decoders.named_parameters():
            c.require(id(parameter) not in ids, 'A compact decoder Parameter appears twice in the optimizer inventory.')
            ids.add(id(parameter))
            selected.decoder_parameters[f'compact::{name}.decoders.{relative}'] = parameter
            compact_leaves += 1
    configure_distill_precision(selected, components=cfg.opt.distill_fp32_components, training_args=stage, logger=log)
    lr = ModelLevelOptimizerLRConfig(learning_rate=cfg.opt.learning_rate, decoder_lr=cfg.opt.resolved_decoder_lr(),
        weight_decay=cfg.opt.weight_decay, norm_lr=cfg.aux.norm_lr, lm_head_lr=cfg.aux.lm_head_lr,
        residual_lora_lr=cfg.aux.residual_lora_lr)
    optimizer_groups = build_model_level_param_groups(selected, lr_config=lr, model=selected.peft_model)
    c.require(not selected.compressed_lora_targets and all(parameter.dtype == torch.float32
        for inventory in (selected.decoder_parameters, selected.lora_parameters) for parameter in inventory.values()),
        'All actual D and dense-LoRA masters must be FP32, without compressed proxies.')
    info = dict(native_modules=len(native_targets), compact_modules=len(compact_targets), dense_lora_targets=len(groups['pending_dense']),
        decoder_leaves=len(selected.decoder_parameters), native_decoder_leaves=len(selected.decoder_parameters) - compact_leaves,
        compact_decoder_leaves=compact_leaves, lora_leaves=len(selected.lora_parameters),
        actual_trainable_leaves=sum(parameter.requires_grad for parameter in selected.peft_model.parameters()),
        optimizer_groups=[dict(name=group['group_name'], leaves=len(group['params']), lr=group['lr'], weight_decay=group['weight_decay'])
                          for group in optimizer_groups])
    return selected, native_targets, compact_targets, lr, info


def make_trainer(model, stage, vae, hf, output, round_base, round_meta, callbacks, *, validation_save_steps=None):
    from .training_checkpoint import MixedCatTrainer
    cfg = stage.config
    tokenizer, data = canonical._prepare_after_category_dataset(model=model, vae_args=vae, cfg=cfg)
    previous_use_cache = bool(model.config.use_cache)
    model.config.use_cache = False
    install_precision_runtime(model, torch.bfloat16)
    selected, native_targets, compact_targets, lr, info = selection(model, stage)
    selected.peft_model.enable_input_require_grads()
    selected.peft_model.to(stage.train_device).train()
    args = canonical._build_training_arguments(stage, is_iterable=data.is_iterable, group_by_length=data.group_by_length)
    args.output_dir = str(output / 'trainer_state')
    # The original regular250/500 checkpoint path uses this same override; only
    # the short validation trigger is1. The training schedule remains500.
    regular_save = dict(strategy=stage.save_strategy, save_steps=stage.save_steps, save_only_model=stage.save_only_model)
    if validation_save_steps is not None:
        args.save_steps = int(validation_save_steps)
    teacher = native.DistillTeacherRuntime(model_path=vae.model_path, access_token=hf.access_token,
        forward_device=stage.train_device, dtype=native.resolve_distill_teacher_dtype(args, model),
        model_offload=cfg.runtime.teacher_model_offload, logger=log)
    trainer = MixedCatTrainer(model=selected.peft_model, train_dataset=data.train_dataset, eval_dataset=data.eval_dataset,
        args=args, callbacks=callbacks, processing_class=tokenizer,
        data_collator=canonical.build_distill_data_collator(tokenizer, model_max_length=cfg.data.model_max_length,
            dynamic_padding=cfg.data.dynamic_padding), loss_config=cfg.loss, teacher_runtime=teacher,
        teacher_output_offload=cfg.runtime.teacher_output_offload,
        teacher_output_pin_memory=cfg.runtime.teacher_output_pin_memory,
        teacher_output_chunk_tokens=cfg.runtime.teacher_output_chunk_tokens,
        mixed_round_base=round_base, mixed_round_meta=round_meta, selected_native=native_targets,
        selected_compact=compact_targets, regular_save=regular_save)
    attach_model_level_optimizer_contract(trainer, selection=selected, lr_config=lr)
    trainer.mixed_original_use_cache = previous_use_cache
    return trainer, teacher, native_targets, compact_targets, info


@torch.no_grad()
def finalize(trainer, stage, native_targets):
    model = trainer.model
    # Snapshot exactly the actual ordinary PEFT merge rule before modifying it.
    expected = {}
    for name, layer in iter_named_peft_lora_layers(model):
        logical = name.removeprefix('base_model.model.')
        expected[logical] = c.digest(_merge_dense_peft_lora_into_base(deepcopy(layer), torch.bfloat16).state_dict())
    expected_dense = len(trainer.mixed_round_meta['pending_dense_modules'])
    c.require(len(expected) == expected_dense, 'Merge exactly the actual remaining dense LoRA targets at this category boundary.')
    finalized = finalize_main_decoder_targets(tuple(NamedMainDecoderTarget(name=name, base_layer=module) for name, module in native_targets))
    c.require(finalized == len(native_targets), 'Native decoder finalization omitted an actual prefix projection.')
    if expected_dense:
        model = finalize_model_level_lora(model, compressed_proxy_names=None)
    prepare_model_export(model, stage)
    for name, expected_digest in expected.items():
        c.require(type(model.get_submodule(name)) is torch.nn.Linear and c.digest(model.get_submodule(name).state_dict()) == expected_digest,
                  'The actual BF16 dense PEFT merge rule was not preserved: ' + name)
    c.require(not list(iter_named_peft_lora_layers(model)), 'Live PEFT adapter remains after stable export.')
    model.eval().requires_grad_(False)
    for module in model.modules():
        clear = getattr(module, 'clear_decoded_weight_cache', None)
        if callable(clear):
            clear()
    model.config.use_cache = trainer.mixed_original_use_cache
    return model, dict(actual_dense_merges=len(expected), BF16_dense_merge_rule_exact=True,
                       native_decoder_finalized=finalized, inventories=checkpoint.inventories(model))
