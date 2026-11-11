"""Audit completed paired replays against their actual files, weights and predictions."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    checked = {}

    def bound(reference):
        path = Path(reference['path'])
        actual = checked.setdefault(str(path), sha(path))
        assert actual == reference['sha256'], str(path)
        return read(path)

    terminal = read(root / 'controller.exit.json')
    assert terminal['status'] == 'completed' and terminal['exit_code'] == 0 and not terminal['errors']
    plan, pairs = read(root / 'plan.json'), read(root / 'paired_results.json')['categories']
    states = {arm: bound(terminal['chains'][arm]) for arm in ('a', 'b')}
    order = ['k_proj', 'o_proj', 'gate_proj', 'q_proj', 'v_proj', 'down_proj', 'up_proj']
    categories, finals, predictions = {}, {}, {}
    gpu_hours = 0.
    for arm, state in states.items():
        assert state['status'] == 'completed' and state['exit_code'] == 0
        assert [row['category_id'] for row in state['replay_stages']] == order
        assert [row['selected_step'] for row in state['replay_stages']] == [250, 125, 500, 375, 1000, 875, 375]
        assert [row['schedule_max'] for row in state['replay_stages']] == [500, 500, 500, 500, 1000, 1000, 500]
        budget = read(root / arm / 'budget.json')
        assert len(budget['tasks']) == 30 and budget['total_actual_GPU_hours'] <= 25
        assert all(row['status'] == 'completed' and row['exit_code'] == 0 for row in budget['tasks'].values())
        gpu_hours += budget['total_actual_GPU_hours']
        for row in state['replay_stages']:
            category = row['category_id']
            raw, deployment, boundary = (bound(row[key]) for key in ('raw', 'deployment', 'boundary'))
            train = bound(row['actual_training'])
            assert train['selected_stop'] == row['selected_step']
            assert boundary['uuid'] == deployment['uuid'] and boundary['completed_prefix'] == order[:order.index(category)+1]
            proof = bound(deployment['fresh_proof_reference'])
            assert proof['state_exact'] and proof['forward_executed'] and proof['finite_logits'] and proof['exit_code'] == 0
            score = bound(pairs[category][arm]['score'])
            assert score['evidence']['checkpoint_id'] == deployment['uuid']
            assert score['evidence']['validated_complete_summary'] and score['evidence']['sample_count'] == 24742 and score['evidence']['leaf_count'] == 64
            for path, digest in score['evidence']['artifacts'].items():
                assert sha(path) == digest, path
                checked[path] = digest
            categories.setdefault(category, {})[arm] = dict(
                state_digest=deployment['export_evidence']['exported_state_digest'], fit_report=raw['fit_report'],
                task_scores=score['task_scores'], mean_percent=score['mean_percent'],
                raw=row['raw'], deployment=row['deployment'], score=pairs[category][arm]['score'])
        nine = bound(state['replay_nine_task'])
        assert nine['exit_code'] == 0 and nine['longbench']['status'] == 'complete'
        assert nine['longbench']['sample_count_per_model'] == 4750 and len(nine['longbench']['tasks']) == 21
        for reference in nine['source'].values():
            bound(reference)
        payload = Path(deployment['path']) / read(Path(deployment['path']) / 'checkpoint_meta.json')['state_dict_file']
        payload_sha = sha(payload)
        assert payload_sha == deployment['digest']
        predictions[arm] = {}
        for index in range(2):
            part = read(root / arm / 'replay/final_longbench' / f'part_{index}' / 'partition_result.json')
            assert part['exit_code'] == 0 and part['checkpoint_id'] == nine['checkpoint_id']
            bound(part['native_report'])
            for path, digest in part['prediction_sha256'].items():
                assert sha(path) == digest
                checked[path] = digest
                predictions[arm][Path(path).stem] = [json.loads(line) for line in Path(path).read_text().splitlines()]
        finals[arm] = dict(nine_task=state['replay_nine_task'], uuid=nine['checkpoint_id'], payload=str(payload),
            payload_sha256=payload_sha, nine_score=nine['compressed_nine_task_score'],
            S8=nine['full8_mean_percent'], LongBench=nine['longbench_task_macro_mean'],
            MMLU=categories['up_proj'][arm]['task_scores']['mmlu'] * 100,
            BF16_nine=nine['baseline_nine_task_score'], drop_pp=nine['mean_drop_points'],
            logical_bpw=nine['actual_ledger']['logical_bits'] / nine['actual_ledger']['N'],
            packed_bpw=nine['actual_ledger']['packed_bits'] / nine['actual_ledger']['N'],
            tensor_bytes=nine['actual_ledger']['tensor_bytes'],
            quality_passed=nine['research_goal_passed'], GPU_hours=budget['total_actual_GPU_hours'],
            completed_at=datetime.fromtimestamp(state['ended_epoch'], ZoneInfo('Asia/Shanghai')).isoformat())
    assert gpu_hours == terminal['actual_GPU_hours'] and gpu_hours <= plan['total_GPU_hours']
    for category, rows in categories.items():
        a, b = rows['a'], rows['b']
        assert a['state_digest'] == b['state_digest'] and a['task_scores'] == b['task_scores']
        normalized = [deepcopy(row['fit_report']) for row in (a, b)]
        identity_differences = []
        for field in ('candidate_id', 'artifact_id'):
            if normalized[0][field] != normalized[1][field]:
                identity_differences.append(field)
            for report in normalized:
                report.pop(field)
        if normalized[0]['parent_identity']['parent_uuid'] != normalized[1]['parent_identity']['parent_uuid']:
            identity_differences.append('parent_identity.parent_uuid')
        for report in normalized:
            report['parent_identity'].pop('parent_uuid')
        assert normalized[0] == normalized[1], category
        rows['comparison'] = dict(tensor_digest_equal=True, full8_equal=True,
            fit_numerics_equal=True, differing_identity_fields=identity_differences)
        for row in rows.values():
            row.pop('fit_report', None)
    assert predictions['a'].keys() == predictions['b'].keys() and len(predictions['a']) == 21
    fields = ('dataset', 'row_index', '_id', 'answers', 'all_classes', 'input_ids_sha256', 'pred',
        'output_token_ids', 'generated_tokens', 'max_new_tokens', 'hit_generation_limit',
        'generation_options', 'effective_generation_options', 'input_tokens', 'removed_tokens')
    samples = 0
    for name, a in predictions['a'].items():
        b = predictions['b'][name]
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert all(x[field] == y[field] for field in fields), (name, x['row_index'])
        samples += len(a)
    assert samples == 4750
    for key in ('payload_sha256', 'nine_score', 'S8', 'LongBench', 'MMLU', 'logical_bpw', 'packed_bpw'):
        assert finals['a'][key] == finals['b'][key], key
    result = dict(status='verified', reproducibility_passed=True, quality_target_passed=finals['a']['quality_passed'],
        scope='Two independent frozen Qwen3-8B BF16-to-seven-category chains in this environment; not a cross-model/environment guarantee.',
        checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), actual_GPU_hours=gpu_hours,
        categories=categories, finals=finals, LongBench_matched_samples=samples,
        LongBench_compared_fields=fields, verified_source_files=checked,
        verifier=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)))
    output = root / 'completion_verification.json'
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], reproducibility_passed=True,
        quality_target_passed=result['quality_target_passed'], actual_GPU_hours=gpu_hours,
        finals=finals, matched_LongBench_samples=samples), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
