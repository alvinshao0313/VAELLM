"""Architecture boundaries for the existing H128/H256 R0 two-stage recipe.

The live Qwen boundary cannot be edited while its callbacks are running. Numeric
restore/audit helpers remain shared; only model inventory and recipe checks live here.
"""
from copy import deepcopy
from pathlib import Path

from torch import nn
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

from experiments.qwen3_w25.assemble_categories import _require
from experiments.qwen3_w25.e2e_subset_export import _read_json
from compressed_e2e_fintuning.v6_runtime_state import validate_e2e_immutable_resume_contract
from experiments.qwen3_w25.local_bits import _code_banks
from experiments.w25_eval.acceptance import validate_checkpoint
from experiments.w25_eval.model_inventory import model_inventory
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_v6 import load_v6_training_step_meta, resolve_training_step_round_base_ref
from train_utils.config.targets import parse_target_layers, parse_target_modules, resolve_target_layers
from train_utils.utils import extract_layer_idx


def selected_layers(target_layers, architecture):
    if isinstance(target_layers, str):
        parsed = parse_target_layers(target_layers)
    else:
        _require(isinstance(target_layers, (tuple, list)) and bool(target_layers)
                 and all(type(layer) is int and layer >= 0 for layer in target_layers),
                 'Selected layers must be a nonempty set of nonnegative integer indices.')
        _require(len(set(target_layers)) == len(target_layers), 'Duplicate selected layer indices.')
        parsed = tuple(sorted(target_layers))
    return resolve_target_layers(parsed, num_layers=architecture['num_hidden_layers'])


def selected_modules(target_modules, architecture):
    available = tuple(dict.fromkeys(name.rsplit('.', 1)[-1]
                                   for name in architecture['projection_shapes']))
    if isinstance(target_modules, str):
        requested = parse_target_modules(target_modules)
    else:
        _require(isinstance(target_modules, (tuple, list)) and bool(target_modules)
                 and all(isinstance(module, str) for module in target_modules),
                 'Selected modules must be a nonempty collection of projection suffixes.')
        requested = tuple(target_modules)
        _require(len(set(requested)) == len(requested), 'Duplicate selected module suffixes.')
    modules = available if requested == 'all' else tuple(requested)
    _require(bool(modules) and set(modules) <= set(available),
             f'Unknown projection suffixes: {sorted(set(modules) - set(available))}.')
    # Native runtime stores suffixes in model traversal order, not CLI order.
    return tuple(module for module in available if module in modules)


def selected_names(target_layers, target_modules, architecture):
    layers = selected_layers(target_layers, architecture)
    modules = selected_modules(target_modules, architecture)
    names = tuple(name for name in architecture['projection_shapes']
                  if name.rsplit('.', 1)[-1] in modules and extract_layer_idx(name) in layers)
    pairs = {(extract_layer_idx(name), name.rsplit('.', 1)[-1]) for name in names}
    expected = {(layer, module) for layer in layers for module in modules}
    _require(len(names) == len(expected) and pairs == expected,
             'Selected projections are absent or ambiguous in the real architecture.')
    return names


def validate_recipe(meta, architecture):
    shapes = architecture['projection_shapes']
    specs = {spec['name']: spec for spec in meta['converted_modules']}
    _require(len(specs) == len(meta['converted_modules']) and set(specs) == set(shapes)
             and set(meta['compressed_targets']) == set(shapes)
             and not meta['pending_dense_targets'] and not meta['skip_targets'],
             'The complete real architecture must be compressed exactly once.')
    for name, spec in specs.items():
        _require([spec['out_features'], spec['in_features']] == shapes[name],
                 f'Projection shape differs from the actual model: {name}.')
        _require(spec['parallel_parts'] == 1 and spec['residual_stages'] == 2
                 and spec['parallel_stage_decode'], f'Expected two packed single-part stages: {name}.')
        decoders = spec['stage_decoders']
        _require(len(decoders) == 2 and all(
                 (d['hidden_dim'], d['decoder_type']) in ((128, 'symmetric'), (256, 'asymmetric'))
                 and d['num_res_blocks'] == 0 and d['norm_type'] == 'layer'
                 and d['activation_type'] == 'swish' for d in decoders),
                 f'Expected H128 symmetric or H256 asymmetric R0/LayerNorm/Swish: {name}.')
        packed_fields = ('in_dim', 'out_dim', 'hidden_dim', 'num_res_blocks', 'norm_type',
                         'activation_type', 'decoder_type', 'param_dtype')
        _require(all(decoders[0][key] == decoders[1][key] for key in packed_fields)
                 and spec['decoders'] == decoders[:1],
                 f'Stage decoder geometry or its stage-zero alias differs: {name}.')


def checkpoint_inventory(directory, architecture=None):
    directory = Path(directory).resolve()
    if architecture is None:
        architecture = model_inventory(_read_json(directory / 'checkpoint_meta.json')['base_model_path'])
    _, meta, budget = validate_checkpoint(directory, architecture)
    validate_recipe(meta, architecture)
    return meta, budget, architecture


def persistent_inventory(model, architecture):
    modules = {name: module for name, module in model.named_modules() if isinstance(module, VAELinear)}
    _require(set(modules) == set(architecture['projection_shapes']),
             'Runtime compressed projections differ from the actual model inventory.')
    inventory, banks = {}, {}
    for name, module in modules.items():
        _require([module.out_features, module.in_features] == architecture['projection_shapes'][name],
                 f'Runtime projection shape differs: {name}.')
        packed = getattr(module, '_parallel_stage_decoder', None)
        _require(isinstance(packed, nn.Module) and module.parallel_stage_decode
                 and module.residual_stages == 2 and module.parallel_parts == 1,
                 f'Expected the native two-stage packed decoder: {name}.')
        inventory[name] = sorted(name + '._parallel_stage_decoder.' + key for key in packed.state_dict())
        banks[name] = [(name + '.' + key, shape) for key, shape in _code_banks(module)]
        _require(len(inventory[name]) == 6 and len(banks[name]) == 2,
                 f'The recipe requires six persistent decoder leaves and two code banks: {name}.')
    return inventory, banks


def validate_configuration(meta, state, snapshot, architecture):
    """Compare saved facts without the live Qwen boundary's fixed layer scope.

    This narrow copy preserves its immutable-contract checks; the old callback
    remains pinned by running jobs and cannot import an expanded boundary.
    """
    contract = meta["immutable_resume_contract"]
    cfg, training = snapshot["canonical_config"], snapshot["training_args"]
    _require(meta["train_mode"] in ("decoder", "decoder_sparse_bit")
             and meta["train_mode"] == contract["train_mode"],
             "Subset export requires matching native decoder or decoder_sparse_bit training.")
    _require(meta["lora_config"] is None and contract["lora"] is None, "LoRA is outside this export topology.")
    for key in ("norm_train_mode", "lm_head_train_mode"):
        _require(meta[key] == contract["aux"][key] == "none", f"Expected {key}=none.")
    _require(contract["aux"].get("residual_lora_mode", "none") == cfg["aux"]["residual_lora_mode"] == "none", "Residual LoRA is forbidden.")
    _require(contract["runtime"]["parallel_mode"] in ("dp", "layer_mp")
             and contract["runtime"]["offload_mode"] == "none"
             and not contract["runtime"]["distill_hif4_act"], "Only single-process DP or layer_mp, without offload or HiFloat4, is supported.")
    _require(contract["distributed"]["world_size"] == 1, "This recipe uses single-process training/export.")
    _require(contract["precision"]["bf16"] and not contract["precision"]["fp16"], "BF16 compute/export is required.")
    _require(contract["optimization"].get("distill_fp32_components") == ["decoder"], "Expected FP32 decoder parameters only.")
    if meta["train_mode"] == "decoder_sparse_bit":
        _require(contract["sparse_bit"].get("proxy_coordinates") == "decoder_sensitivity"
                 and contract["sparse_bit"]["active_ratio"] == 1.0,
                 "Joint subset export requires all bits and decoder_sensitivity coordinates.")
    else:
        _require("proxy_coordinates" not in contract["sparse_bit"],
                 "Decoder-only native contracts must not claim active bit coordinates.")
    _require(type(state["global_step"]) is int and 0 < state["global_step"] <= state["max_steps"], "Invalid selected optimizer step.")
    _require(state["max_steps"] == training["max_steps"] == contract["optimization"]["steps"], "Original total steps disagree; do not edit them to export.")
    selected_names(meta["target_layers"], meta["target_modules"], architecture)
    _require(meta["target_layers"] == contract["target_layers"]
             == list(selected_layers(cfg["target_layers"], architecture)),
             "Resolved target layers disagree across metadata, contract and snapshot.")
    _require(meta["target_modules"] == contract["target_modules"]
             == list(selected_modules(cfg["target_modules"], architecture)),
             "Resolved target modules disagree across metadata, contract and snapshot.")
    _require(training["seed"] == contract["data"]["seed"], "Snapshot training seed disagrees.")

    # The normalized snapshot precedes dataset construction, which resolves the
    # worker count. Preserve that saved runtime fact and teacher/tokenizer identity;
    # all user-configurable mathematical fields below must match the original run.
    expected = deepcopy(contract)
    for key in ("train_mode", "data", "loss", "runtime"):
        expected[key] = deepcopy(cfg[key])
    expected["input_checkpoint_id"] = meta["round_base_checkpoint_id"]
    expected["optimization"] = {key: value for key, value in cfg["opt"].items() if key != "logging_steps"}
    expected["aux"] = {key: value for key, value in cfg["aux"].items() if not key.startswith("residual_lora_")}
    expected["precision"] = {key: training[key] for key in contract["precision"]}
    expected["sparse_bit"] = {key: cfg["bit_" + key] for key in ("active_ratio", "optimizer", "weight_decay", "round_steps")}
    expected["sparse_bit"]["bit_lr"] = cfg["bit_lr"]
    coordinates = cfg.get("bit_proxy_coordinates", "unit")
    _require(coordinates in ("unit", "decoder_sensitivity"), "Unsupported snapshot bit coordinates.")
    if meta["train_mode"] == "decoder_sparse_bit" and coordinates != "unit":
        expected["sparse_bit"]["proxy_coordinates"] = coordinates
    for key in contract["evaluation_execution"]:
        value = cfg["runtime"]["evaluation"][key] if key == "eval_after_save" else training[key]
        # The snapshot serializes str-enums as their values; the production
        # contract uses str(enum).lower() after TrainingArguments normalization.
        if key == "eval_strategy":
            value = str(IntervalStrategy(value)).lower()
        elif key == "save_strategy":
            value = str(SaveStrategy(value)).lower()
        expected["evaluation_execution"][key] = value
    validate_e2e_immutable_resume_contract(contract, expected)
    return contract


def validate_sparse_sidecar(checkpoint, train_mode):
    sparse = Path(checkpoint) / 'sparse_bit_tuning'
    if train_mode == 'decoder_sparse_bit':
        _require((sparse / 'exact_state.pt').is_file(), 'Missing exact SparseBit sidecar.')
    elif train_mode == 'decoder':
        _require(not sparse.exists(), 'Decoder-only checkpoint unexpectedly contains SparseBit state.')
    else:
        raise ValueError('Unsupported subset training mode.')


def inspect_step(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    meta = load_v6_training_step_meta(str(checkpoint))
    source, base_meta = resolve_training_step_round_base_ref(str(checkpoint), meta)
    _require(checkpoint.parent.name == 'trainer_state', 'Select a checkpoint in its original trainer_state directory.')
    snapshot = _read_json(checkpoint.parent.parent / 'normalized_e2e_runtime_args.json')
    state = _read_json(checkpoint / 'trainer_state.json')
    architecture = model_inventory(base_meta['base_model_path'])
    validate_configuration(meta, state, snapshot, architecture)
    _require(Path(meta['base_model_path']).resolve() == Path(base_meta['base_model_path']).resolve(),
             'Step/base model identities disagree.')
    validate_recipe(base_meta, architecture)
    _require(set(meta['compressed_targets']) == set(architecture['projection_shapes'])
             and not meta['pending_dense_targets'] and not meta['skip_targets'],
             'The selected training step does not cover the complete actual architecture.')
    _require(set(selected_names(meta['target_layers'], meta['target_modules'], architecture)) <= architecture['projection_shapes'].keys(),
             'Selected layers are absent from this architecture.')
    validate_sparse_sidecar(checkpoint, meta['train_mode'])
    return meta, state, snapshot, Path(source), base_meta, architecture
