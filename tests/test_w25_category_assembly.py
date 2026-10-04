"""Real local tokenizer/config/meta contracts; no pretrained weight loading."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.w25_eval.assemble_categories import (
    assembly_tokenizer, validate_runtime_configs, validate_source_metadata, validate_sources,
)
from experiments.w25_eval.model_inventory import model_inventory


ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / 'result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/final_model'
LLAMA_ROOT = ROOT / 'result/llama31_8b_w25_20261004/validation_remaining_prefix/native_runs'


@pytest.fixture(scope='module', params=['qwen3', 'llama'])
def source(request):
    paths = [QWEN] if request.param == 'qwen3' else list(LLAMA_ROOT.glob('*/final_model'))
    if len(paths) != 1 or not (paths[0] / 'checkpoint_meta.json').is_file():
        pytest.skip('Actual source metadata is not installed.')
    directory = paths[0]
    meta = json.loads((directory / 'checkpoint_meta.json').read_text())
    config = json.loads((directory / 'config.json').read_text())
    architecture = model_inventory(meta['base_model_path'])
    tokenizer, original_pad = assembly_tokenizer(meta['base_model_path'])
    return directory, meta, config, architecture, tokenizer, original_pad


def test_actual_inventory_and_tokenizer_runtime_rules(source):
    _, meta, config, architecture, tokenizer, original_pad = source
    _, specs = validate_source_metadata(meta, architecture, complete=True)
    assert len(specs) == architecture['projection_count']
    assert not architecture['weights_loaded_for_inventory']
    other = deepcopy(config)
    other.update(use_cache=not config['use_cache'], pad_token_id=None)
    runtime = validate_runtime_configs(config, other, architecture, tokenizer, original_pad)
    if architecture['model_type'] == 'llama':
        assert len(specs) == 224 and original_pad is None
        assert config['pad_token_id'] == tokenizer.pad_token_id == tokenizer.eos_token_id == 128009
    else:
        assert len(specs) == 252 and original_pad == tokenizer.pad_token_id == 151643
    assert runtime['resolved_tokenizer_pad_token_id'] == tokenizer.pad_token_id
    for field, bad_value in [('hidden_size', config['hidden_size'] + 1),
                             ('use_cache', 1), ('pad_token_id', tokenizer.pad_token_id + 1),
                             ('model_type', 'opt'), ('num_hidden_layers', 1)]:
        bad = deepcopy(other)
        bad[field] = bad_value
        with pytest.raises(ValueError):
            validate_runtime_configs(config, bad, architecture, tokenizer, original_pad)


def test_changed_protection_partial_donor_keeps_real_model_geometry(source, tmp_path):
    _, meta, config, architecture, tokenizer, original_pad = source
    donor = deepcopy(meta)
    donor['converted_modules'] = [spec for spec in donor['converted_modules']
                                  if spec['name'].endswith(('gate_proj', 'down_proj'))]
    donor['converted_module_count'] = len(donor['converted_modules'])
    donor['compressed_targets'] = [spec['name'] for spec in donor['converted_modules']]
    donor['pending_dense_targets'] = []
    donor['skip_targets'] = []
    for spec in donor['converted_modules']:
        count = 128 if spec['name'].endswith('gate_proj') else 384
        spec['compressed_in_features'] = spec['in_features'] - count
        for field in ('protected_input_indices', 'protected_input_scales', 'protected_input_qvalues'):
            spec[field]['shape'][0] = count
        blocks = spec['compressed_in_features'] * spec['compressed_out_features'] // spec['codebook_dim']
        for field in ('vq_weights', 'stage_vq_weights'):
            for bank in spec[field]:
                bank['shape'][0] = bank['logical_shape'][0] = blocks
    validate_source_metadata(donor, architecture, complete=False)
    with pytest.raises(ValueError, match='every actual projection'):
        validate_source_metadata(donor, architecture, complete=True)
    base_dir, donor_dir = tmp_path / 'base', tmp_path / 'donor'
    for path, payload in ((base_dir, meta), (donor_dir, donor)):
        path.mkdir()
        (path / 'checkpoint_meta.json').write_text(json.dumps(payload))
        saved_config = dict(config)
        if path == donor_dir:
            saved_config.update(use_cache=True, pad_token_id=None)
        (path / 'config.json').write_text(json.dumps(saved_config))
    # No pytorch_model.bin exists: this entire preflight is metadata-only.
    result = validate_sources(base_dir, donor_dir, ['gate_proj', 'down_proj'])
    assert len(result[2]) == 2 * architecture['num_hidden_layers']
    assert all('compressed_in_features' in fields for fields in result[-1].values())
    with pytest.raises(ValueError, match='every selected category'):
        validate_sources(base_dir, donor_dir, ['q_proj'])


def test_metadata_rejects_architecture_shape_dense_lora_and_illegal_protection(source):
    _, meta, _, architecture, _, _ = source
    mutations = [
        ('in_features', meta['converted_modules'][0]['in_features'] + 1),
        ('compressed_in_features', 1),
        ('has_original_weight', True),
        ('low_rank_a', {'shape': [1, 1], 'dtype': 'bfloat16'}),
        ('protected_input_indices', {'shape': [65], 'dtype': 'int64'}),
        ('protected_input_weight', {'shape': [64, 1], 'dtype': 'bfloat16'}),
        ('protected_residual_shared_decoder_refs', ['not_a_local_decoder']),
    ]
    for field, value in mutations:
        bad = deepcopy(meta)
        bad['converted_modules'][0][field] = value
        with pytest.raises((ValueError, TypeError)):
            validate_source_metadata(bad, architecture, complete=True)
    wrong = deepcopy(architecture)
    wrong['base_model_path'] = '/not/the/original/model'
    with pytest.raises(ValueError, match='architecture/model'):
        validate_source_metadata(meta, wrong, complete=True)
