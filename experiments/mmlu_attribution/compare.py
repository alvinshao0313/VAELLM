"""Compare all seven complete paired subsets without treating them as candidates."""
import argparse
import json
import math
from pathlib import Path

from experiments.longbench_v1.common import sha256, write_json
from experiments.mmlu_attribution.protocol import ARMS, check_preparation, require


def compare(preparation, results_root, output):
    record, _cases = check_preparation(preparation)
    results_root, output = Path(results_root).resolve(), Path(output).resolve()
    require(not output.exists(), 'Do not replace a prior paired comparison.')
    arms, native, evidence = {}, {}, {}
    for arm in ARMS:
        directory = results_root / arm
        report = json.loads((directory / 'report.json').read_text())
        require(report['status'] == 'diagnostic_complete' and report['exit_code'] == 0
                and report['sample_count'] == 1824 and report['protocol'] == record['protocol']
                and report['preparation_sha256'] == sha256(preparation)
                and report['non_target_state_and_identity_unchanged'], 'A paired arm is incomplete or changed protocol.')
        for name, key in (('paired_choices.json', 'paired_choices_sha256'), ('native_inputs.json', 'native_inputs_sha256'),
                          ('metrics.json', 'metrics_sha256')):
            require(sha256(directory / name) == report[key], 'A native paired output changed after completion.')
        arms[arm] = json.loads((directory / 'paired_choices.json').read_text())['cases']
        native[arm] = json.loads((directory / 'native_inputs.json').read_text())
        evidence[arm] = {'report': str(directory / 'report.json'), 'sha256': sha256(directory / 'report.json')}
    reference = arms['compressed']
    identity_fields = ('task', 'doc_id', 'doc_sha256', 'arguments_sha256', 'target_index', 'actual_pair_encodings')
    for arm, rows in arms.items():
        require(len(rows) == 1824 and all({k: x[k] for k in identity_fields} == {k: y[k] for k in identity_fields}
                for x, y in zip(rows, reference)), 'Case identities or native pair encodings differ across arms.')
        for key in ('hf_policy', 'batches', 'pairs'):
            require(native[arm][key] == native['compressed'][key], 'Native policy/input batching differs across arms: ' + key)
        # Task traversal order is recorded; content pairing remains keyed by task/doc/choice.
        ordered = lambda value: sorted(value, key=lambda row: (row['task'], row['doc_id'], row['choice_index']))
        require(ordered(native[arm]['request_order']) == ordered(native['compressed']['request_order']),
                'Native likelihood request inventory differs across arms.')
    dense = arms['dense']
    lost = [i for i, (full, compressed) in enumerate(zip(dense, reference)) if full['correct'] and not compressed['correct']]
    summary = {}
    for arm, rows in arms.items():
        correct = sum(row['correct'] for row in rows)
        rescued = sum(not source['correct'] and row['correct'] for source, row in zip(reference, rows))
        harmed = sum(source['correct'] and not row['correct'] for source, row in zip(reference, rows))
        summary[arm] = {'correct': correct, 'samples': 1824, 'accuracy_percent': correct * 100 / 1824,
            'rescued_compressed_errors': rescued, 'damaged_compressed_correct': harmed, 'net_recovered': rescued - harmed,
            'recovered_dense_correct_compressed_wrong': sum(rows[i]['correct'] for i in lost),
            'mean_gold_margin_delta': math.fsum(row['gold_margin'] - source['gold_margin']
                                              for source, row in zip(reference, rows)) / 1824}
    result = {'status': 'paired_diagnostic_complete', 'protocol': record['protocol'], 'arms': summary,
        'dense_correct_compressed_wrong_count': len(lost), 'input_batches_and_pairs_identical': True,
        'evidence': evidence, 'goal_acceptance': False, 'model_exported': False,
        'interpretation': 'Conditional original-BF16 restoration in the bound compressed source. Effects are not additive and do not predict decoder-recovery gains; not full MMLU or a bit-budget-compliant candidate.'}
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', required=True, type=Path)
    parser.add_argument('--results_root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    compare(args.preparation, args.results_root, args.output)
