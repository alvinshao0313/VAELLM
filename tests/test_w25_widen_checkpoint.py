"""Selection and actual source layout only; existing width math tests are reused."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.widen_checkpoint import (
    _compare_budgets, build_parser, expected_widened_state, selected_projection_names, widened_specs,
)
from train_utils.checkpoint_bpw import audit_state_dict


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/final_model'
TARGET = 'model.layers.0.mlp.down_proj'


@pytest.fixture(scope='module')
def source():
    if not (SOURCE / 'pytorch_model.bin').is_file():
        pytest.skip('Actual Qwen source checkpoint is unavailable.')
    meta = json.loads((SOURCE / 'checkpoint_meta.json').read_text())
    return meta, model_inventory(meta['base_model_path'])


def test_cli_and_real_architecture_selection(source, tmp_path):
    _, architecture = source
    args = build_parser().parse_args(['--source_dir', str(SOURCE), '--output_dir', str(tmp_path / 'new'),
        '--report', str(tmp_path / 'report.json'), '--target_layers', '0', '--target_modules', 'down_proj'])
    assert selected_projection_names(architecture, args.target_layers, args.target_modules) == (TARGET,)
    assert len(selected_projection_names(architecture, '0-3', 'down_proj')) == 4
    assert selected_projection_names(architecture, '35', 'down_proj') == ('model.layers.35.mlp.down_proj',)
    for layers, modules in [('36', 'down_proj'), ('0,0', 'down_proj'), ('0', 'unknown_proj'), ('', 'down_proj')]:
        with pytest.raises((ValueError, __import__('argparse').ArgumentTypeError)):
            selected_projection_names(architecture, layers, modules)


def test_llama_selection_uses_actual_32_layers():
    base = Path('/home/star/models/meta-llama/Llama-3.1-8B-Instruct')
    if not (base / 'tokenizer.json').is_file():
        pytest.skip('Official Llama metadata/tokenizer is unavailable.')
    architecture = model_inventory(base)
    assert architecture['projection_count'] == 224
    assert selected_projection_names(architecture, '31', 'down_proj') == ('model.layers.31.mlp.down_proj',)
    with pytest.raises(ValueError):
        selected_projection_names(architecture, '32', 'down_proj')


def test_actual_metadata_changes_only_selected_width_and_type(source):
    meta, architecture = source
    before = deepcopy(meta)
    specs, added = widened_specs(meta, architecture, (TARGET,))
    assert added == 21248 and meta == before
    expected = deepcopy(meta['converted_modules'])
    for spec in expected:
        if spec['name'] == TARGET:
            for field in ('decoders', 'stage_decoders'):
                for decoder in spec[field]:
                    decoder.update(hidden_dim=256, decoder_type='asymmetric')
    assert specs == expected
    bad = deepcopy(meta)
    next(s for s in bad['converted_modules'] if s['name'] == TARGET)['stage_decoders'][0]['hidden_dim'] = 256
    with pytest.raises(ValueError, match='H128/R0'):
        widened_specs(bad, architecture, (TARGET,))


def test_real_source_layout_and_decoder_only_storage_plan(source):
    meta, architecture = source
    original = torch.load(SOURCE / 'pytorch_model.bin', weights_only=True, mmap=True, map_location='cpu')
    specs, added = widened_specs(meta, architecture, (TARGET,))
    expected, keys = expected_widened_state(original, meta, (TARGET,))
    assert len(keys) == 6 and len(expected) == len(original) == 3423
    assert all(expected[key] is value for key, value in original.items() if key not in keys)
    assert sum(expected[key].numel() - original[key].numel() for key in keys) == added
    before = audit_state_dict(meta, original, max_bpw=2.5, max_lora_rank=0)
    after = audit_state_dict(dict(meta, converted_modules=specs), expected, max_bpw=2.5, max_lora_rank=0)
    _compare_budgets(before, after, 339968)
    assert after['original_linear_weights'] == architecture['original_linear_weights']
    assert after['user_linear_bits'] == before['user_linear_bits'] + 339968
    bad = deepcopy(after)
    bad['components']['rotation']['storage_bits'] += 8
    with pytest.raises(ValueError, match='rotation'):
        _compare_budgets(before, bad, 339968)
