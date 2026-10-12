"""The fixed13bd mixed S2/S4 all-decoder inventory and native step contract."""
from pathlib import Path

import torch
from torch import nn

from experiments.llama_ce_reproduction.train import read
from experiments.long_context_recovery.contract import require
from experiments.qwen3_w25.assemble_categories import _load_state
from experiments.qwen3_w25.audit_e2e_decoder import _inference_spec
from experiments.w25_eval.acceptance import validate_checkpoint
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import selected_names, validate_configuration, validate_sparse_sidecar
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_bpw import _code_specs
from train_utils.checkpoint_v6 import load_v6_training_step_meta, resolve_training_step_round_base_ref

PUBLIC = Path('/home/star/VAELLM')
R2 = PUBLIC / 'result/qwen3_8b_mmlu65_20261004'
ROOT = R2 / 'rotation_factor_probe/source_snapshot_20261005_01'
SOURCE = R2 / 'llama_gate_stage4/s4/final_model'
SOURCE_ID = '13bd608f-c3a4-43d3-b3c0-be269743c544'
TOTAL_BITS = 17099800576


def validate_recipe(meta, architecture):
    reference = read(SOURCE / 'checkpoint_meta.json')
    specs = {s['name']: s for s in meta['converted_modules']}
    original = {s['name']: s for s in reference['converted_modules']}
    require(reference['checkpoint_id'] == SOURCE_ID and architecture['model_type'] == 'llama'
            and len(specs) == len(meta['converted_modules']) == 224
            and set(specs) == set(original) == set(meta['compressed_targets']) == set(architecture['projection_shapes'])
            and not meta['pending_dense_targets'] and not meta['skip_targets'], 'Require the complete original13bd topology.')
    for name, spec in specs.items():
        stages = 4 if name.endswith('.gate_proj') else 2
        require(_inference_spec(spec) == _inference_spec(original[name])
                and [spec['out_features'], spec['in_features']] == architecture['projection_shapes'][name]
                and spec['parallel_parts'] == 1 and spec['parallel_stage_decode'] and spec['residual_stages'] == stages
                and len(spec['stage_decoders']) == len(list(_code_specs(spec))) == stages,
                'Mixed-stage deployment geometry changed: ' + name)
    return specs


def persistent_inventory(model, architecture):
    specs = validate_recipe(read(SOURCE / 'checkpoint_meta.json'), architecture)
    modules = {n: m for n, m in model.named_modules() if isinstance(m, VAELinear)}
    require(set(modules) == set(specs), 'Runtime compressed inventory differs from13bd.')
    inventory, banks = {}, {}
    for name, module in modules.items():
        spec = specs[name]; decoder = getattr(module, '_parallel_stage_decoder', None)
        require(isinstance(decoder, nn.Module) and module.parallel_stage_decode and module.parallel_parts == 1
                and module.residual_stages == decoder.num_models == spec['residual_stages'], 'Wrong grouped stage inventory: ' + name)
        inventory[name] = sorted(name + '._parallel_stage_decoder.' + k for k in decoder.state_dict())
        expected = [(k, tuple(v['logical_shape'])) for k, v in _code_specs(spec)]
        state = module.state_dict(); banks[name] = []
        for stage, (key, shape) in enumerate(expected):
            value = module.get_stage_part_vq_storage(stage, 0)
            actual = module.get_stage_part_vq_spec(stage, 0)
            aliases = [n for n, v in state.items() if v.data_ptr() == value.data_ptr()
                       and v.shape == value.shape and v.dtype == value.dtype and v.stride() == value.stride()]
            require(value.dtype == torch.uint8 and tuple(actual['logical_shape']) == shape
                    and aliases == [key[len(name) + 1:]], 'Native packed bank/alias differs: ' + key)
            banks[name].append((key, shape))
        require(len(inventory[name]) == 6 and banks[name] == expected, 'Actual stage inventory differs: ' + name)
    return inventory, banks


def expected_inventory(architecture):
    specs = validate_recipe(read(SOURCE / 'checkpoint_meta.json'), architecture)
    state = _load_state(SOURCE)
    keys = [k for n in specs for k in state if k.startswith(n + '._parallel_stage_decoder.')]
    banks = [k for spec in specs.values() for k, _ in _code_specs(spec)]
    require(len(keys) == len(set(keys)) == 1344 and len(banks) == len(set(banks)) == 512
            and len(state) == 3043 and all(state[k].dtype == torch.uint8 for k in banks)
            and all(state[k].dtype == torch.bfloat16 for k in keys), 'Actual source state inventory differs.')
    count = sum(state[k].numel() for k in keys)
    require(count == 4078592, 'Actual decoder parameter count differs.')
    return {'deployed_projections': 224, 'serialized_tensors': len(state), 'all_decoder_tensors': len(keys),
        'all_code_banks': len(banks), 'selected_projections': 224, 'selected_decoder_tensors': len(keys),
        'selected_code_banks': 0, 'active_score_elements': 0, 'frozen_tensors': len(state) - len(keys),
        'selected_projection_code_banks': len(banks), 'selected_decoder_elements': count}


def checkpoint_inventory(directory, architecture=None):
    if architecture is None:
        architecture = model_inventory(read(Path(directory) / 'checkpoint_meta.json')['base_model_path'])
    _, meta, budget = validate_checkpoint(directory, architecture)
    validate_recipe(meta, architecture)
    require(budget['user_linear_bits'] == budget['packed_linear_bits'] == TOTAL_BITS
            and budget['code_padding_bits'] == 0 and budget['original_linear_weights'] == 6979321856,
            'Original S4 deployment budget changed.')
    return meta, budget, architecture


def inspect_step(checkpoint):
    checkpoint = Path(checkpoint).resolve(); meta = load_v6_training_step_meta(str(checkpoint))
    source, base_meta = resolve_training_step_round_base_ref(str(checkpoint), meta)
    require(Path(source).resolve() == SOURCE and base_meta['checkpoint_id'] == SOURCE_ID
            and checkpoint.parent.name == 'trainer_state', 'Use a native step from the original13bd round base.')
    snapshot = read(checkpoint.parent.parent / 'normalized_e2e_runtime_args.json')
    state = read(checkpoint / 'trainer_state.json'); architecture = model_inventory(base_meta['base_model_path'])
    validate_configuration(meta, state, snapshot, architecture)
    validate_recipe(base_meta, architecture)
    require(meta['train_mode'] == 'decoder' and meta['base_model_path'] == base_meta['base_model_path']
            and set(meta['compressed_targets']) == set(architecture['projection_shapes'])
            and not meta['pending_dense_targets'] and not meta['skip_targets']
            and set(selected_names(meta['target_layers'], meta['target_modules'], architecture))
            == set(architecture['projection_shapes']), 'Only all-decoder restoration is supported here.')
    validate_sparse_sidecar(checkpoint, 'decoder')
    return meta, state, snapshot, SOURCE, base_meta, architecture
