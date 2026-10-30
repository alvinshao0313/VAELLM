"""CPU screen of the unchanged first M32 cache; no model or synthetic logits."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'result/qwen3_8b_mmlu65_20261004'
HOT = RESULT / 'qwen_hotpot_recovery_20261012'
DIRECT = Path('/data/star/VAELLM/g09_recovery_20261029/direct_opd32')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def checked(binding):
    path = Path(binding['path'])
    assert path.stat().st_size == binding['bytes']
    assert digest(path) == binding['sha256'], str(path)
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def tensor_record(value):
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}


def margin(logp, index):
    other = logp.clone()
    other[index] = -torch.inf
    return float(logp[index] - torch.logsumexp(other, -1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = read(HOT / 'teacher_targets.json')
    dataset = torch.load(checked(manifest['dataset_binding']), map_location='cpu', weights_only=True)
    component = manifest['components'][0]
    cache = read(checked(component['cache_binding']))
    payload = torch.load(checked(component['payload_binding']), map_location='cpu', weights_only=True, mmap=True)
    assert cache['payload'] == component['payload_binding']
    assert payload['teacher_kind'] == cache['teacher_kind'] == 'full7'
    assert payload['source_original_state'] == cache['source_original_state'] == manifest['source_original_state']
    helper_path = RESULT / 'qwen_mmlu_target_mix32/targets.py'
    spec = importlib.util.spec_from_file_location('_unchanged_M_mixture', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    group = dataset['groups'][0]
    cases = dataset['training_cases'][:32]
    assert group['group_index'] == 0 and group['case_ids'][:32] == [case['case_id'] for case in cases]
    records = {row['case_id']: row for row in cache['target_records']}
    rows = []
    for case in cases:
        key = case['case_id']
        assert case['domain'] == 'mmlu' and case['source']['official_split'] == 'train'
        assert key.startswith('cais_mmlu/auxiliary_train/train/') and group['weights'][key] == .5 / 32
        item, old = payload['targets'][key], records[key]
        teacher = item['target_log_probs']
        assert teacher.dtype == torch.float32 and teacher.shape == (1, 4)
        assert tensor_record(teacher) == old['target_log_probs']
        assert item['input_signature'] == old['input_signature'] == case['input_signature']
        mixed = helper.target_log_distribution(teacher, case)
        gold = helper.check_mmlu_case(case)
        p_teacher, p_mixed = teacher.double().exp()[0], mixed.double().exp()[0]
        ideal_mix = .5 * p_teacher
        ideal_mix[gold] += .5
        rows.append({'case_id': key, 'input_signature': case['input_signature'],
            'gold_index': gold, 'native_continuation_ids': case['native_continuation_ids'].tolist(),
            'teacher_logp': teacher[0].tolist(), 'mixed_logp': mixed[0].tolist(),
            'teacher_mass': float(p_teacher.sum()), 'mixed_mass': float(p_mixed.sum()),
            'teacher_gold_margin_from_cache': margin(teacher.double()[0], gold),
            'mixed_gold_margin_from_cache': margin(mixed.double()[0], gold),
            'mixture_representation_error_l2': float((p_mixed-ideal_mix).norm()),
            'mixture_representation_error_max': float((p_mixed-ideal_mix).abs().max())})
    # Read only a fixed completed prefix; concurrent log growth is not a trajectory audit.
    selected_events = []
    with (DIRECT / 'run.log').open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'):
                break
            if not line.startswith('{"event": "nativegreedy_OPD_hotpot_update",'):
                continue
            row = json.loads(line)
            if row.get('event') == 'nativegreedy_OPD_hotpot_update' and row['step'] <= 2:
                selected_events.append({key: row[key] for key in
                    ('step', 'group_index', 'domain_losses', 'weighted_training_objective',
                     'gradient_norm_before_clip', 'lr', 'adam_step')})
                if row['step'] == 1:
                    assert row['M_case_ids'] == group['case_ids'][:32]
                selected_events[-1]['source_line_sha256'] = hashlib.sha256(line.encode()).hexdigest()
                if row['step'] == 2:
                    break
    margins = [row['mixed_gold_margin_from_cache'] for row in rows]
    report = {'status': 'CPU_cache_screen_complete_no_student_capture',
        'python': sys.executable, 'torch': torch.__version__, 'cuda_initialized': torch.cuda.is_initialized(),
        'source': {'dataset': manifest['dataset_binding'], 'teacher_cache': component['cache_binding'],
            'teacher_payload': component['payload_binding'], 'mixture_code': str(helper_path),
            'completed_update_log': str(DIRECT / 'run.log'),
            'mixture_code_sha256': digest(helper_path), 'diagnostic_code_sha256': digest(__file__)},
        'window': {'group_index': 0, 'documents': 32, 'selection': 'unchanged first formal M32 group',
            'weight_per_document': .5/32, 'support': 4, 'temperature': 1},
        'summary': {'mixed_margin_min': min(margins), 'mixed_margin_max': max(margins),
            'mixed_margin_above_5_5': sum(value > 5.5 for value in margins),
            'mixed_margin_above_16_6': sum(value > 16.6 for value in margins),
            'teacher_mass_max_abs_error': max(abs(row['teacher_mass']-1) for row in rows),
            'mixed_mass_max_abs_error': max(abs(row['mixed_mass']-1) for row in rows),
            'mixture_representation_error_max': max(row['mixture_representation_error_max'] for row in rows)},
        'first_two_completed_updates': selected_events, 'rows': rows,
        'limitations': ['Target margins are implied by stored logp, not observed student logits.',
            'Mixture representation error excludes upstream teacher logits and log-softmax error.',
            'No model forward, student gradient, parameter Jacobian, optimizer step, or quality measured.']}
    assert report['cuda_initialized'] is False and len(selected_events) == 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'summary': report['summary'],
        'first_two_completed_updates': selected_events}), flush=True)


if __name__ == '__main__':
    main()
