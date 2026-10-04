"""CPU inventory checks; no Llama compressed checkpoint or GPU save is claimed."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from typing import Optional

import pytest
import torch
import transformers

from experiments.qwen3_w25.summarize_results import _validate_checkpoint
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.subset_state import checkpoint_inventory, persistent_inventory, validate_recipe
from test_qwen3_w25_e2e_aux_export import _model, TARGETS


ROOT = Path(__file__).resolve().parents[1]
QWEN = Path('/home/star/models/Qwen/Qwen3-8B')
LLAMA = Path('/home/star/models/meta-llama/Llama-3.1-8B-Instruct')
SOURCE = ROOT / 'result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/final_model'


@pytest.fixture(scope='module')
def qwen_source():
    if not (SOURCE / 'pytorch_model.bin').is_file():
        pytest.skip('Real source endpoint not installed.')
    return checkpoint_inventory(SOURCE)


def test_real_qwen_checkpoint_budget_and_meta_match_old_boundary(qwen_source):
    meta, budget, architecture = qwen_source
    old_meta, old_budget = _validate_checkpoint(SOURCE)
    assert meta == old_meta and budget == old_budget
    assert architecture['projection_count'] == 252
    assert architecture['original_linear_weights'] == 6945767424
    state = torch.load(SOURCE / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
    # Inspect actual serialized keys/shapes only; this is not a fresh model load.
    decoder = [key for key in state if '._parallel_stage_decoder.' in key]
    banks = [key for key in state if key.endswith(('.vq_weight', '.vq_weight_s1'))]
    assert len(state) == 3423 and len(decoder) == 1512 and len(banks) == 504
    assert all(state[key].dtype == torch.bfloat16 for key in decoder)
    assert all(state[key].dtype == torch.uint8 for key in banks)


def test_llama_actual_meta_inventory_and_cross_architecture_rejection(qwen_source):
    if not (LLAMA / 'tokenizer.json').is_file():
        pytest.skip('Official Llama metadata/tokenizer not installed.')
    llama = model_inventory(LLAMA)
    assert llama['model_type'] == 'llama' and llama['projection_count'] == 224
    assert llama['num_hidden_layers'] == 32 and llama['original_linear_weights'] == 6979321856
    assert not llama['weights_loaded_for_inventory']
    with pytest.raises(ValueError, match='architecture'):
        validate_recipe(qwen_source[0], llama)


@pytest.mark.parametrize('violation', ['shape', 'hidden', 'residual', 'missing'])
def test_recipe_and_complete_inventory_fail_closed(qwen_source, violation):
    meta, _, architecture = qwen_source
    bad = deepcopy(meta)
    if violation == 'shape':
        bad['converted_modules'][0]['in_features'] += 1
    elif violation == 'missing':
        bad['converted_modules'].pop()
    else:
        key = 'hidden_dim' if violation == 'hidden' else 'num_res_blocks'
        bad['converted_modules'][0]['stage_decoders'][0][key] += 1
    with pytest.raises(ValueError):
        validate_recipe(bad, architecture)


def test_runtime_inventory_uses_real_shapes_and_actual_persistent_leaves():
    # Small real VAELinear fixture checks inventory plumbing, not Llama quality.
    model = _model()
    architecture = {'projection_shapes': {name: [model.get_submodule(name).out_features,
        model.get_submodule(name).in_features] for name in TARGETS}}
    decoder, banks = persistent_inventory(model, architecture)
    assert set(decoder) == set(banks) == set(TARGETS)
    assert sum(map(len, decoder.values())) == 6 * len(TARGETS)
    assert sum(map(len, banks.values())) == 2 * len(TARGETS)
    actual = model.state_dict()
    assert {key for keys in decoder.values() for key in keys} <= actual.keys()
    assert {key for rows in banks.values() for key, _ in rows} <= actual.keys()
    architecture['projection_shapes'][TARGETS[0]][0] += 1
    with pytest.raises(ValueError, match='shape'):
        persistent_inventory(model, architecture)


@pytest.mark.parametrize('base,misleading_name,expected', [(QWEN, 'llama-renamed', 'get_qwen3'),
                                                         (LLAMA, 'Qwen3-renamed', 'get_llama')])
def test_pending_config_dispatch_with_actual_configs_and_misleading_paths(tmp_path, base, misleading_name, expected):
    if not (base / 'config.json').is_file():
        pytest.skip('Official model config not installed.')
    # Apply only to temporary copies and execute the isolated config/dispatch nodes.
    # Loader sentinels observe selection; pretrained weight loading is not tested.
    (tmp_path / 'rotation').mkdir()
    (tmp_path / 'train_utils').mkdir()
    shutil.copy2(ROOT / 'rotation/model_utils.py', tmp_path / 'rotation/model_utils.py')
    shutil.copy2(ROOT / 'train_utils/base_reference.py', tmp_path / 'train_utils/base_reference.py')
    subprocess.run(['patch', '--batch', '-p1', '-i', str(ROOT / 'experiments/w25_eval/model_utils.pending.patch')],
                   cwd=tmp_path, check=True, capture_output=True)
    tree = ast.parse((tmp_path / 'rotation/model_utils.py').read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('get_model_config', 'get_model')]
    assert len(functions) == 2
    reference = ast.parse((tmp_path / 'train_utils/base_reference.py').read_text())
    functions.extend(node for node in reference.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('_reference_attn_implementation', '_reference_seqlen'))
    assert len(functions) == 4
    scope = {'transformers': transformers, 'Optional': Optional}
    for name in ('get_llama', 'get_mistral', 'get_qwen2', 'get_qwen3', 'get_opt', 'get_auto_causal_lm'):
        scope[name] = lambda *args, _name=name: _name
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<pending-config-dispatch>', 'exec'), scope)
    renamed = tmp_path / misleading_name
    renamed.mkdir()
    shutil.copy2(base / 'config.json', renamed / 'config.json')
    assert scope['get_model'](str(renamed)) == expected
    config = scope['get_model_config'](str(renamed))
    assert scope['_reference_attn_implementation'](config) == (
        'sdpa' if config.model_type == 'llama' else 'flash_attention_2')
    assert scope['_reference_seqlen'](config) == 2048


RUNS = ROOT / 'result/qwen3_8b_mmlu65_20261004'


@pytest.mark.parametrize('run_name', ['best1200_task100_joint_s400', 'best1200_down03_task100_joint_s400'])
def test_actual_native_400_contract_unchanged(run_name):
    from experiments.qwen3_w25.e2e_subset_export import inspect_step as old_inspect
    from experiments.w25_eval.subset_state import inspect_step, selected_names, validate_configuration
    paths = list((RUNS / run_name).glob('*/trainer_state/checkpoint-400'))
    if len(paths) != 1 or not (paths[0] / 'sparse_bit_tuning/exact_state.pt').is_file():
        pytest.skip('Actual native step400 or retired SparseBit sidecar not available.')
    old = old_inspect(paths[0])
    new = inspect_step(paths[0])
    assert new[:5] == old
    before = deepcopy(new[:3])
    assert validate_configuration(*new[:3], new[5]) == old[0]['immutable_resume_contract']
    assert new[:3] == before
    names = selected_names(new[0]['target_layers'], new[0]['target_modules'], new[5])
    assert names == tuple(f'model.layers.{layer}.mlp.down_proj' for layer in new[0]['target_layers'])
    # Scope may change, but unrelated mathematical fields still fail the original contract.
    changed = deepcopy(new[2])
    changed['canonical_config']['bit_lr'] = '1e-3'
    with pytest.raises(ValueError):
        validate_configuration(new[0], new[1], changed, new[5])


def test_real_architecture_middle_scope_and_invalid_indices(qwen_source):
    from experiments.w25_eval.subset_state import selected_names
    architecture = qwen_source[2]
    expected = tuple(f'model.layers.{layer}.mlp.down_proj' for layer in range(16, 20))
    assert selected_names([19, 17, 16, 18], ['down_proj'], architecture) == expected
    assert selected_names('16-19', 'down_proj', architecture) == expected
    assert len(selected_names('all', ['down_proj'], architecture)) == 36
    for layers in ([], [-1], [36], [0, 0], [True], [1.0]):
        with pytest.raises(ValueError):
            selected_names(layers, ['down_proj'], architecture)
    assert selected_names([16], ['gate_proj'], architecture) == ('model.layers.16.mlp.gate_proj',)
    mlp = selected_names('all', ['down_proj', 'gate_proj', 'up_proj'], architecture)
    assert len(mlp) == 108
    assert mlp[:3] == tuple(f'model.layers.0.mlp.{module}' for module in ('gate_proj', 'up_proj', 'down_proj'))
    assert len(selected_names('all', 'all', architecture)) == 252
    for modules in ([], ['missing_proj'], ['down_proj', 'down_proj'], ['down_proj,up_proj'], ['all'], 'down', '*'):
        with pytest.raises((ValueError, argparse.ArgumentTypeError)):
            selected_names([16], modules, architecture)
    llama = model_inventory(LLAMA)
    assert len(selected_names('16-19', ['down_proj'], llama)) == 4
    with pytest.raises(ValueError):
        selected_names([32], ['down_proj'], llama)


def test_actual_h256_mixed_metadata_and_incompatible_packed_stages(qwen_source):
    # Read the actual independent migration metadata; no H256 training or load is claimed.
    path = RUNS / 'best1200_down0_h256_initial/final_model/checkpoint_meta.json'
    if not path.is_file():
        pytest.skip('Actual H256 migration metadata not available.')
    meta = json.loads(path.read_text())
    validate_recipe(meta, qwen_source[2])
    widened = next(row for row in meta['converted_modules'] if row['name'] == 'model.layers.0.mlp.down_proj')
    assert [(row['hidden_dim'], row['decoder_type']) for row in widened['stage_decoders']] == [(256, 'asymmetric')] * 2
    assert sum(row['stage_decoders'][0]['hidden_dim'] == 256 for row in meta['converted_modules']) == 1
    for key, value in [('activation_type', 'relu'), ('norm_type', 'none'), ('decoder_type', 'symmetric')]:
        changed = deepcopy(meta)
        row = next(row for row in changed['converted_modules'] if row['name'] == widened['name'])
        for decoder in row['stage_decoders'] + row['decoders']:
            decoder[key] = value
        with pytest.raises(ValueError):
            validate_recipe(changed, qwen_source[2])
    changed = deepcopy(meta)
    row = next(row for row in changed['converted_modules'] if row['name'] == widened['name'])
    row['stage_decoders'][1].update(hidden_dim=128, decoder_type='symmetric')
    with pytest.raises(ValueError, match='geometry'):
        validate_recipe(changed, qwen_source[2])
