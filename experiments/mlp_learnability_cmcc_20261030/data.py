"""Same frozen H96 teacher-forcing inputs and targets for both parameterizations."""
from collections import Counter
import hashlib
import json

# Select N before legacy data helpers alter the package search order.
from . import bindings as runtime
from experiments.muon_compact_cmcc_20261030 import bindings as legacy
from experiments.muon_compact_cmcc_20261030 import inputs as original


def exposure(dataset, recovery, indices):
    domains = Counter()
    inputs = Counter()
    answers = Counter()
    groups = []
    for index in indices:
        group = dataset['groups'][index]
        cases = [recovery[key]['case'] for key in group['case_ids']]
        legacy.require(group['group_index'] == index and len(cases) == 40
            and all(case['domain'] == 'mmlu' for case in cases[:32])
            and all(case['domain'] != 'mmlu' for case in cases[32:]), 'Keep each actual complete M32/QA8 group.')
        for case in cases:
            domain = case['domain']
            domains[domain] += 1
            inputs[domain] += len(case['input_ids'])
            answers[domain] += recovery[case['case_id']]['target_log_probs'].shape[0]
            legacy.require(group['weights'][case['case_id']] == .5/(32 if domain == 'mmlu' else 8),
                'Keep the original equal domain and per-document weights.')
        groups.append(dict(group_index=index, M_batch_size=32,
            M_padded_tokens=32 * max(len(case['input_ids']) for case in cases[:32]),
            M_max_length=max(len(case['input_ids']) for case in cases[:32]),
            QA_max_length=max(len(case['input_ids']) for case in cases[32:]),
            QA_domains=sorted({case['domain'] for case in cases[32:]})))
    return dict(group_indices=list(indices), presentations=dict(domains), input_tokens=dict(inputs),
        answer_positions=dict(answers), total_input_tokens=sum(inputs.values()),
        total_answer_positions=sum(answers.values()), groups=groups)


def load():
    # This existing reader verifies all frozen source/cache identities and mixes
    # M exactly once. Its online-OPD interface deliberately drops QA targets.
    dataset, recovery, teacher, mixed = original.load_inputs()
    manifest_path = original.HOT / 'teacher_targets.json'
    manifest = legacy.read(manifest_path)
    bindings = {str(manifest_path): legacy.binding(manifest_path)}
    for component in manifest['components']:
        cache_path = original.checked(component['cache_binding'])
        payload_path = original.checked(component['payload_binding'])
        bindings[str(cache_path)] = legacy.binding(cache_path)
        bindings[str(payload_path)] = legacy.binding(payload_path)
        payload = legacy.torch.load(payload_path, map_location='cpu', weights_only=True, mmap=True)
        for key in component['case_ids']:
            case = recovery[key]['case']
            if case['domain'] == 'mmlu':
                continue
            target = payload['targets'][key]
            legacy.require(target['input_signature'] == case['input_signature']
                and target['domain'] == case['domain'] and 'target_log_probs' not in recovery[key],
                'Restore only the unchanged full-F teacher-forcing QA target, once.')
            recovery[key] = dict(case=case, target_log_probs=target['target_log_probs'])

    records = {}
    for key, item in recovery.items():
        case, target = item['case'], item['target_log_probs']
        legacy.require(case['source']['official_split'] == 'train'
            and target.dtype == legacy.torch.float32 and target.device.type == 'cpu'
            and not target.requires_grad, 'Use the verified train-only frozen FP32 targets.')
        records[key] = dict(input_signature=case['input_signature'], domain=case['domain'],
            actual_target=legacy.core.tensor_record(target))
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    legacy.require(digest == mixed['actual_input_target_records_sha256'],
        'The restored static TF pool must exactly equal the original once-mixed F_A_HOT inputs/targets.')
    cases = list(dataset['training_cases'])
    longest_m = max((case for case in cases if case['domain'] == 'mmlu'), key=lambda case: len(case['input_ids']))
    longest_qa = max((case for case in cases if case['domain'] != 'mmlu'), key=lambda case: len(case['input_ids']))
    selected = sorted({next(group['group_index'] for group in dataset['groups'] if case['case_id'] in group['case_ids'])
        for case in (longest_m, longest_qa)})
    legacy.require(len(selected) == 2, 'This fixed pool must provide two distinct actual peak groups.')
    full = exposure(dataset, recovery, range(32))
    short = exposure(dataset, recovery, selected)
    legacy.require(short['group_indices'] == [2, 10]
        and max(group['M_max_length'] for group in short['groups']) == 958
        and max(group['QA_max_length'] for group in short['groups']) == 10593,
        'The fixed real H96 resource shapes changed; review rather than silently truncate.')
    evidence = dict(schema='mlp_learnability_static_TF_data_v1',
        status='verified_original_train_inputs_and_static_full_F_targets',
        teacher=teacher, mixed_target_evidence=mixed, bindings=bindings,
        actual_input_target_records_sha256=digest, full_exposure=full, short_exposure=short,
        short_selection='First complete groups containing the longest actual M and QA inputs; no quality selection.',
        source='Same original H96 F_A_HOT cache, M mixture once; original QA teacher forcing, no OPD trajectories.',
        scope='Data and target identity only; no optimization, knowledge recovery or complete-nine result.')
    return dataset, recovery, evidence


def mmlu_log_distribution(model, cases):
    """Preserve the existing M32 math while gathering on the actual head device."""
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    inputs, options = legacy.mmlu.batch_inputs(cases, pad)
    device = model.get_input_embeddings().weight.device
    result = model(**{key: value.to(device) for key, value in inputs.items()}, use_cache=False, logits_to_keep=1)
    legacy.require(result.past_key_values is None
        and result.logits.shape == (32, 1, model.config.vocab_size)
        and result.logits.dtype == legacy.torch.bfloat16, 'Keep the original BF16 M32 last-query head.')
    selected = result.logits[:, 0].gather(1, options.to(result.logits.device))
    return legacy.torch.nn.functional.log_softmax(selected.float(), dim=-1)
