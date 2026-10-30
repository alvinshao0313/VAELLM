"""Reuse exact coverage/N32 inputs and full-F M targets without loading old R logits."""
from collections import Counter
from copy import deepcopy
import bindings as b

COVERAGE = b.R / 'qwen_open_mcqa_coverage64_20261014'
N32 = b.R / 'qwen_narrativeqa_opd_s32_20261013'
SHARED = b.R / 'qwen_opd_retention_qaopd_20261013/shared'
coverage = b.module(COVERAGE / 'coverage_inputs.py', '_aligned_protection_coverage_inputs')
targets = b.module(COVERAGE / 'cov64_targets.py', '_aligned_protection_cov64_targets')
shared = b.module(SHARED / 'shared_recovery.py', '_aligned_protection_shared_inputs')
retention = b.module(SHARED / 'shared_retention.py', '_aligned_protection_R_inputs')
loss = b.module(b.R / 'qwen_narrativeqa_opd_short_20261013/nqa_retention.py', '_aligned_protection_R_loss')


def check_binding(item):
    b.require(b.binding(item['path']) == item, 'Changed original material: ' + item['path'])


def load_materials():
    original = b.read(N32 / 'prepared.json')
    dataset, shared_record = shared.load_shared()
    n_binding = original['narrativeqa_payload']
    check_binding(n_binding)
    narrative = b.torch.load(n_binding['path'], map_location='cpu', weights_only=True)
    b.require(list(narrative) == original['narrativeqa_case_ids'] and len(narrative) == 32,
        'Fixed original Narrative32 order changed; do not rebuild selection.')
    for key, case in narrative.items():
        ids = case['prompt_input_ids']
        b.require(case['domain'] == 'narrativeqa' and case['source']['official_split'] == 'train'
            and ids.dtype == b.torch.long and ids.device.type == 'cpu' and 16384 <= len(ids) <= 32640
            and not set(case) & {'input_ids', 'target_ids', 'answer_query_positions', 'answer_target_positions'}
            and case['input_signature'] == b.core.mapping_digest(dict(domain='narrativeqa', case_id=key,
                prompt=b.core.tensor_record(ids), source=case['source'])), 'Narrative prompt-only identity changed.')
    selection = original['N_selection']
    b.require(selection['hash_namespace'] == 'qwen_narrativeqa_opd_short_20261013/v1/seed137'
        and not selection['offline_N_target_exists'], 'Changed original Narrative question/story namespace.')
    for item in selection['raw_story_bindings'].values():
        check_binding(item)
    for key in ('inventory_report', 'source_manifest'):
        check_binding(selection[key])
    for path, item in selection['original_metadata_bindings'].items():
        check_binding(dict(path=path, **item))
    groups_q = original['groups']
    by_id = {case['case_id']: case for case in dataset['training_cases']}
    by_id.update(narrative)
    n_seen = []
    for old, new in zip(dataset['groups'], groups_q, strict=True):
        i = old['group_index']
        b.require(i == new['group_index'] and old['case_ids'][:36] == new['case_ids'][:36], 'Original M/Q first36 slots changed.')
        if i in (0, 1, 2, 3, 16, 17, 18, 19):
            b.require(all(key in narrative for key in new['case_ids'][36:]), 'Only fixed last4 Q slots may become Narrative.')
            n_seen.extend(new['case_ids'][36:])
        else:
            b.require(new == old, 'Unselected QA group changed.')
        b.require(all(new['weights'][key] == (.5/32 if slot < 32 else .5/8)
            for slot, key in enumerate(new['case_ids'])), 'Original domain/doc weights changed.')
    b.require(n_seen == list(narrative), 'Fixed Narrative slots/order changed.')
    cov_record = b.read(COVERAGE / 'training_prepared.json')
    m_targets, m_evidence = targets.load_M_targets(cov_record)
    _, coverage_groups, _ = coverage.load_inputs('B_med64')
    groups_scale = cov_record['arm_groups']['B_med64'][:32]
    for index, group in enumerate(groups_scale):
        original_M = coverage_groups[index]
        b.require(group['group_index'] == original_M['group_index'] == index
            and original_M['step_1based'] == index + 1
            and original_M['original_N32_QA_group_index'] == original_M['original_N32_R_group_index'] == index
            and group['case_ids'][:32] == original_M['M_case_ids']
            and group['case_ids'][32:] == groups_q[index]['case_ids'][32:]
            and group['case_ids'][:16] == groups_q[index]['case_ids'][:16]
            and all(key.startswith('medmcqa/train/') for key in group['case_ids'][16:32]), 'Changed exact B_med64 first32 composition.')
    for key in original['groups_M_ids']:
        coverage.require_same(m_targets[key]['case'], by_id[key])
    # Only M has offline targets. Old QA TF answers never enter the objective.
    recovery = {key: {'case': value} for key, value in by_id.items() if value['domain'] != 'mmlu'}
    recovery.update(m_targets)
    archive, r_record = retention.load_retention()
    b.require(r_record['total_continuation_tokens'] == 15462 and r_record['vocab_size'] == 151936
        and r_record['cache_FP32_payload_bytes'] == 9396937728,
        'Original complete R896/allbranch continuation budget changed.')
    groups = dict(scale32=groups_scale, q64=groups_q)
    exposure = {}
    for arm, selected in groups.items():
        keys = [key for group in selected for key in group['case_ids']]
        counts = Counter(recovery[key]['case']['domain'] for key in keys)
        b.require(len(keys) == len(set(keys)) == 1280 and counts == dict(mmlu=1024, qasper=32, narrativeqa=32, hotpotqa=192),
            'Each original32 QA/M presentation must occur exactly once per pass.')
        exposure[arm] = dict(counts)
    materials = dict(groups=groups, N_selection=selection, narrativeqa_payload=n_binding,
        original_N32_prepared=b.binding(N32 / 'prepared.json'), shared_dataset=shared_record['dataset_binding'],
        coverage_prepared=b.binding(COVERAGE / 'training_prepared.json'), target_binding=cov_record['target_binding'],
        input_manifest_binding=cov_record['input_manifest_binding'], M_target_evidence=m_evidence,
        retention_prepared=b.binding(SHARED / 'retention_prepared.json'), retention_dataset=r_record['dataset_binding'],
        retention_source_inputs=dict(documents=896, continuation_tokens=15462, vocab_size=151936,
            raw_FP32_probability_bytes=9396937728, groups=r_record['groups'], batch_shapes=r_record['batch_shapes']),
        first_pass_exposure=exposure, scale_M_sources=dict(aux=512, Med=512),
        q64_group_order=list(range(32))*2, scale32_group_order=list(range(32)),
        QA_TF_targets_used=False, old_R_probabilities_loaded=False)
    return recovery, archive, materials


def record_for(prepared, arm, phase, run_dir):
    choice = b.ARMS[arm]
    order = [0, 19] if phase == 'short2' else list(range(32)) * (choice['steps']//32)
    r_order = [17, 28] if phase == 'short2' else order
    source = prepared['records'][arm]
    return {**deepcopy(source), 'phase': phase, 'arm': arm, 'group_indices': order,
        'R_group_indices': r_order, 'steps': len(order), 'planned_steps': len(order),
        'groups': prepared['materials']['groups'][arm], 'run_dir': str(run_dir),
        'prepared_sha256': b.sha256(b.OUTPUT / 'prepared.json')}
