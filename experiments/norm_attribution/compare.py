"""Compare complete paired norm diagnostics, with no full-benchmark claim."""
import argparse
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

from experiments.longbench_v1.common import sha256, write_json
from experiments.longbench_v1.protocol import generation_options
from experiments.longbench_v1.scoring import score_example
from experiments.mmlu_attribution.protocol import require
from .protocol import ARMS, check_preparation, require_mmlu_pairing


def paired_summary(rows, reference):
    correct = sum(row['correct'] for row in rows)
    rescued = sum(not old['correct'] and new['correct'] for old, new in zip(reference, rows))
    harmed = sum(old['correct'] and not new['correct'] for old, new in zip(reference, rows))
    return {'correct': correct, 'samples': 1824, 'accuracy_percent': correct * 100 / 1824,
        'rescued': rescued, 'harmed': harmed, 'net_correct': rescued - harmed,
        'mean_gold_margin_delta': math.fsum(new['gold_margin'] - old['gold_margin']
                                          for old, new in zip(reference, rows)) / 1824}


def compare(preparation, output):
    preparation, output = Path(preparation).resolve(), Path(output).resolve()
    record, norm, reference, manifest, _mmlu, _cases = check_preparation(preparation)
    require(not output.exists(), 'Do not replace a previous comparison.')
    tokenizer = AutoTokenizer.from_pretrained(manifest['tokenizer']['path'], use_fast=True, local_files_only=True)
    choices = {kind: json.loads((Path(directory) / 'paired_choices.json').read_text())['cases']
               for kind, directory in record['mmlu_controls'].items()}
    old_norm = Path(record['norm_retrieval_control']) / 'predictions.jsonl'
    retrieval = {'norm400': [json.loads(line) for line in old_norm.read_text().splitlines()]}
    evidence = {'norm400_retrieval': {'path': str(old_norm), 'sha256': sha256(old_norm)}}
    for arm in ARMS:
        directory = preparation.parent / 'arms' / arm
        report = json.loads((directory / 'report.json').read_text())
        require(report['status'] == 'diagnostic_complete' and report['exit_code'] == 0
                and report['arm'] == arm and report['candidate_id'] == record['candidate_id']
                and report['preparation_sha256'] == sha256(preparation)
                and report['non_target_state_and_identity_unchanged'] and report['restored_norms_unchanged_after']
                and report['input_pins_unchanged_after'] and not report['model_exported'], 'An arm is incomplete.')
        require(set(report['restored_norms']) == set(record['selected_norms'][arm]), 'Restoration selection changed.')
        for name, digest in report['mmlu']['files_sha256'].items():
            require(sha256(directory / name) == digest, 'Completed MMLU output changed.')
        choices[arm] = json.loads((directory / 'paired_choices.json').read_text())['cases']
        observation = json.loads((directory / 'native_inputs.json').read_text())
        require_mmlu_pairing(observation, choices[arm], record['mmlu_controls']['compressed'])
        require(report['mmlu']['correct'] == sum(row['correct'] for row in choices[arm]), 'MMLU report count differs.')
        if arm != 'norm400':
            require(sha256(directory / 'retrieval.jsonl') == report['retrieval']['predictions_sha256'], 'Retrieval output changed.')
            retrieval[arm] = [json.loads(line) for line in (directory / 'retrieval.jsonl').read_text().splitlines()]
        evidence[arm] = {'path': str(directory / 'report.json'), 'sha256': sha256(directory / 'report.json')}
    summaries = {}
    for arm, rows in retrieval.items():
        require(len(rows) == 16, 'Missing fixed retrieval cases.')
        for predicted, row in zip(rows, norm['cases']):
            for key in ('dataset', 'row_index', '_id', 'answers', 'all_classes', 'length', 'input_ids_sha256', 'input_tokens'):
                require(predicted[key] == row[key], 'Retrieval input identity changed: ' + key)
            options = generation_options(tokenizer, row['dataset'], row['input_tokens'], row['max_new_tokens'],
                                         protocol=norm['config']['protocol'])
            require(predicted['generation_options'] == predicted['effective_generation_options'] == options
                    and predicted['use_model_defaults'] is False and predicted['diagnostic'] is True
                    and predicted['generated_tokens'] == len(predicted['output_token_ids']) <= row['max_new_tokens']
                    and tokenizer.decode(predicted['output_token_ids'], skip_special_tokens=True) == predicted['pred'],
                    'Retrieval generation/decode differs from the native protocol.')
            require(predicted['official_score'] == score_example(row['dataset'], predicted['pred'], row['answers'], row['all_classes']),
                    'Official retrieval score changed.')
        scores = [row['official_score'] for row in rows]
        control = [row['official_score'] for row in retrieval['norm400']]
        summaries[arm] = {'correct': sum(scores), 'samples': 16,
            'en_correct': sum(scores[:8]), 'zh_correct': sum(scores[8:]),
            'rescued_vs_norm400': [[row['dataset'], row['row_index']] for row, old in zip(rows, control) if row['official_score'] > old],
            'harmed_vs_norm400': [[row['dataset'], row['row_index']] for row, old in zip(rows, control) if row['official_score'] < old]}
    for kind, rows in reference['controls'].items():
        scores = [row['official_score'] for row in rows]
        summaries[kind] = {'correct': sum(scores), 'samples': 16, 'en_correct': sum(scores[:8]), 'zh_correct': sum(scores[8:])}
    result = {'status': 'paired_diagnostic_complete', 'candidate_id': record['candidate_id'],
        'restoration_source_id': record['restoration_source_id'], 'evidence': evidence,
        'mmlu': {arm: {'versus_f927': paired_summary(rows, choices['compressed']),
                       'versus_norm400': paired_summary(rows, choices['norm400'])} for arm, rows in choices.items()},
        'retrieval': summaries, 'actual_mmlu_policy_batches_and_encodings_identical': True,
        'fixed_retrieval_inputs_and_official_scoring_verified': True, 'goal_acceptance': False, 'model_exported': False,
        'interpretation': 'Conditional removal of learned norm groups; final norm belongs to other73. Effects can interact and are not additive. No subgroup training efficacy, full MMLU, full LongBench or nine-term acceptance claim.'}
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    compare(**vars(parser.parse_args()))
