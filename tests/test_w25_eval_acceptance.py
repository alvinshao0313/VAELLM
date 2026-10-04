"""Protocol/math tests plus actual local HF config/tokenizer shape discovery."""
import json
import math
from pathlib import Path
import sys

import pytest

from experiments.w25_eval.acceptance import _check_goal_scope, collect_results, leaf_metrics
from experiments.w25_eval.model_inventory import model_inventory
from experiments.qwen3_w25.summarize_results import (
    DATASET_CONFIGS, MMLU_SAMPLE_COUNTS, SAMPLE_COUNTS, TASK_METRICS,
)


def leaf_fixture(mmlu):
    sizes = MMLU_SAMPLE_COUNTS if mmlu else SAMPLE_COUNTS
    raw, counts, shots, configs = {}, {}, {}, {}
    for index, (task, count) in enumerate(sizes.items()):
        metric = 'acc,none' if mmlu else TASK_METRICS[task]
        raw[task] = {metric: (index + 1) / (len(sizes) + 1)}
        counts[task] = {'original': count, 'effective': count}
        shots[task] = 0
        path, name, split = (('hails/mmlu_no_train', task[5:], 'test') if mmlu else DATASET_CONFIGS[task])
        configs[task] = {'dataset_path': path, 'dataset_name': name, 'test_split': split,
                         'output_type': 'multiple_choice'}
    return raw, counts, shots, configs


def test_relative_paired_goal_cannot_be_applied_to_qwen():
    _check_goal_scope('llama', True)
    _check_goal_scope('qwen3', False)
    with pytest.raises(ValueError, match='Llama-only'):
        _check_goal_scope('qwen3', True)


def test_full_mmlu_weighting_and_reject_missing_subject():
    args = leaf_fixture(True)
    scores, subjects, batch = leaf_metrics(*args)
    expected = math.fsum(args[0][t]['acc,none'] * n for t, n in MMLU_SAMPLE_COUNTS.items()) / 14042
    assert scores['mmlu'] == expected and batch == 64 and len(subjects) == 57
    assert not math.isclose(expected, math.fsum(subjects.values()) / 57, abs_tol=1e-5)
    del args[1][next(iter(args[1]))]
    with pytest.raises(ValueError, match='exactly'):
        leaf_metrics(*args)


@pytest.mark.parametrize('change', ['effective', 'fewshot', 'dataset', 'nonfinite'])
def test_seven_protocol_rejects_incomplete_or_incompatible(change):
    args = leaf_fixture(False)
    if change == 'effective':
        args[1]['boolq']['effective'] -= 1
    elif change == 'fewshot':
        args[2]['rte'] = 5
    elif change == 'dataset':
        args[3]['piqa']['test_split'] = 'train'
    else:
        args[0]['arc_easy']['acc_norm,none'] = float('nan')
    with pytest.raises(ValueError):
        leaf_metrics(*args)


def test_hf_cli_raw_normalization_and_fixed_batch(tmp_path):
    inventory = {'base_model_path': str(tmp_path), 'model_type': 'llama', 'dense_parameter_elements': 100,
                 'tokenizer_bos_token_id': 1, 'tokenizer_eos_token_id': 2}
    paths = []
    for mmlu in (True, False):
        raw, counts, shots, configs = leaf_fixture(mmlu)
        metrics, _, batch = leaf_metrics(raw, counts, shots, configs)
        if mmlu:
            raw['mmlu'] = {'acc,none': metrics['mmlu']}
        payload = {'config': {'model': 'hf', 'model_args':
                    f'pretrained={tmp_path},dtype=bfloat16,use_fast_tokenizer=True,add_bos_token=False,use_cache=False,attn_implementation=sdpa',
                    'model_dtype': 'torch.bfloat16', 'model_num_parameters': 100,
                    'limit': None, 'batch_size': batch},
                   'results': raw, 'n-samples': counts, 'n-shot': shots, 'configs': configs,
                   'versions': {t: 1 for t in counts}, 'chat_template': None, 'chat_template_sha': None,
                   'system_instruction': None, 'system_instruction_sha': None, 'fewshot_as_multiturn': False,
                   'tokenizer_bos_token': ['bos', '1'], 'tokenizer_eos_token': ['eos', '2']}
        path = tmp_path / ('mmlu.json' if mmlu else 'seven.json')
        path.write_text(json.dumps(payload))
        paths.append(path)
    result = collect_results(paths, inventory)
    assert result['leaf_count'] == 64 and result['sample_count'] == 24742
    assert result['mean_percent'] == math.fsum(result['task_metrics'].values()) * 100 / 8
    original_args = payload['config']['model_args']
    payload['config']['model_args'] = original_args.replace('sdpa', 'flash_attention_2')
    paths[1].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='attention backend'):
        collect_results(paths, inventory)
    payload['config']['model_args'] = original_args
    payload['config']['batch_size'] = 'auto'
    paths[1].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='batch'):
        collect_results(paths, inventory)


@pytest.mark.parametrize('symlink', [False, True])
def test_cli_refuses_existing_report_before_input_loading(tmp_path, monkeypatch, symlink):
    from experiments.w25_eval.summarize import main
    output = tmp_path / 'report.json'
    if symlink:
        output.symlink_to(tmp_path / 'missing_target')
    else:
        output.write_text('keep existing evidence')
    monkeypatch.setattr(sys, 'argv', ['summarize', '--base-model-path', 'unused',
        '--source-manifest', 'unused', '--baseline-results', 'unused1', 'unused2', '--output', str(output)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert output.is_symlink() if symlink else output.read_text() == 'keep existing evidence'


@pytest.mark.parametrize('base,count,denominator', [
    ('/home/star/models/Qwen/Qwen3-8B', 252, 6945767424),
    ('/home/star/models/meta-llama/Llama-3.1-8B-Instruct', 224, 6979321856),
])
def test_actual_local_architecture_meta_only(base, count, denominator):
    if not (Path(base) / 'tokenizer.json').exists():
        pytest.skip('Official local metadata/tokenizer is not installed.')
    result = model_inventory(base)
    assert result['projection_count'] == count
    assert result['original_linear_weights'] == denominator
    assert result['projection_count'] == result['num_hidden_layers'] * 7
    assert result['weights_loaded_for_inventory'] is False
