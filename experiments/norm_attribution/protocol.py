"""Reuse the completed norm candidate and existing paired diagnostic inputs."""
import importlib.metadata
import json
from pathlib import Path

from experiments.longbench_v1.common import check_hashes, sha256
from experiments.longbench_candidate_probe.prepare import check_candidate_preparation
from experiments.mmlu_attribution.protocol import check_preparation as check_mmlu, require

ARMS = ('norm400', 'restore_qk', 'restore_other')


def norm_groups(shapes):
    qk = sorted(key for key in shapes if '.self_attn.q_norm.' in key or '.self_attn.k_norm.' in key)
    other = sorted(set(shapes) - set(qk))
    require(len(qk) == 72 and len(other) == 73 and len(shapes) == 145
            and 'model.norm.weight' in other, 'Require the actual Qwen3 36-layer norm partition.')
    require(all(shapes[key] == [128] for key in qk)
            and all(shapes[key] == [4096] for key in other), 'Norm geometry changed.')
    require(all(key == 'model.norm.weight' or key.endswith(('.input_layernorm.weight',
            '.post_attention_layernorm.weight')) for key in other), 'Unexpected non-QK norm.')
    return {'norm400': [], 'restore_qk': qk, 'restore_other': other}


def check_preparation(path):
    record = json.loads(Path(path).read_text())
    require(record['schema'] == 'norm_attribution_v1' and record['status'] == 'prepared_not_scheduled'
            and record['queued'] is False and record['arms'] == list(ARMS), 'Invalid diagnostic preparation.')
    check_hashes(record['source_sha256'])
    check_hashes(record['evidence_sha256'])
    norm, retrieval, manifest = check_candidate_preparation(record['norm_preparation'])
    mmlu, cases = check_mmlu(record['mmlu_preparation'])
    require(norm['config']['model']['checkpoint_id'] == record['candidate_id']
            and mmlu['source_checkpoint_id'] == record['restoration_source_id']
            and record['selected_norms'] == norm_groups(record['norm_shapes']), 'Norm/source identity changed.')
    require(norm['config']['model_policy'] == mmlu['model_policy']
            and norm['config']['model']['hf_path'] == mmlu['hf_path'], 'Native model policies differ.')
    for name, expected in record['package_versions'].items():
        require(importlib.metadata.version(name) == expected, 'Evaluation dependency version changed.')
    return record, norm, retrieval, manifest, mmlu, cases


def require_mmlu_pairing(observation, rows, control_directory):
    control_directory = Path(control_directory)
    native = json.loads((control_directory / 'native_inputs.json').read_text())
    expected = json.loads((control_directory / 'paired_choices.json').read_text())['cases']
    keys = ('task', 'doc_id', 'doc_sha256', 'arguments_sha256', 'target_index', 'actual_pair_encodings')
    require(len(rows) == len(expected) == 1824 and all(
        {k: left[k] for k in keys} == {k: right[k] for k in keys}
        for left, right in zip(rows, expected)), 'Actual MMLU case/token identities differ.')
    for key in ('hf_policy', 'batches', 'pairs'):
        require(observation[key] == native[key], 'Native MMLU policy/batching differs: ' + key)
    order = lambda values: sorted(values, key=lambda row: (row['task'], row['doc_id'], row['choice_index']))
    require(order(observation['request_order']) == order(native['request_order']), 'MMLU request inventory changed.')


def check_control(directory, preparation_sha256, *, retrieval=False):
    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text())
    require(report['status'] == 'diagnostic_complete' and report['exit_code'] == 0
            and report['preparation_sha256'] == preparation_sha256
            and report['sample_count'] == (16 if retrieval else 1824), 'Historical control is incomplete.')
    files = [('predictions.jsonl', 'predictions_sha256')] if retrieval else [
        ('paired_choices.json', 'paired_choices_sha256'), ('native_inputs.json', 'native_inputs_sha256'),
        ('metrics.json', 'metrics_sha256')]
    for name, key in files:
        require(sha256(directory / name) == report[key], 'Historical output changed: ' + name)
    return report, [directory / 'report.json', *(directory / name for name, _ in files)]
