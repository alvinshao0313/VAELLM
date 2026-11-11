"""Preserve each original auxiliary half and QA slot; fill SciQ from official train."""
from collections import Counter
from copy import deepcopy
import hashlib

from . import bindings as b


def rank(case_id):
    return hashlib.sha256(('causal_output_fit/137/' + case_id).encode()).hexdigest()


def helpers():
    native = b.mature.module(b.TOPIC / 'qwen_mmlu_causal_readout/prepare_data.py', '_SciQ_existing_PairTokenizer')
    common = b.mature.aligned.reference.data
    return native, common


def build():
    import pyarrow.parquet as pq
    from e2e_common.data import _stable_sciq_answer_index
    from lm_eval.evaluator_utils import get_task_list
    from lm_eval.tasks import TaskManager, get_task_dict

    old, recovery, cache, target_evidence = b.original.inputs.load_inputs()
    capture, inventory = b.read(b.DATA_ROOT / 'capture.json'), b.read(b.DATA_ROOT / 'design_inventory.json')
    b.require(capture['repository'] == 'allenai/sciq' and capture['revision'] == b.REVISION
        and capture['split'] == 'train' and capture['license'] == 'cc-by-nc-3.0', 'Require pinned official SciQ train.')
    for item in capture['files'].values():
        b.require(b.binding(item['path']) == {key:item[key] for key in ('path','bytes','sha256')}, 'Changed official SciQ source.')
    path = b.DATA_ROOT / 'data/train-00000-of-00001.parquet'
    b.require(b.mature.sha256(path) == '19644360954006d06e9ad3df07bddb34f8535c081b831d48f604603c713ac167', 'SciQ LFS SHA differs.')
    raw = pq.read_table(path).to_pylist()
    b.require(len(raw) == 11679, 'SciQ official train row count differs.')
    selected = sorted((rank(f'allenai_sciq/train/{index}'), f'allenai_sciq/train/{index}', index)
        for index in range(len(raw)))[:512]
    b.require([key for _,key,_ in selected] == [row['case_id'] for row in inventory['selection']], 'Fixed hash selection changed.')
    native, common = helpers()
    tokenizer = b.core.build_tokenizer(str(b.HF))
    codec = native.PairTokenizer(tokenizer)
    manager = TaskManager()
    leaves = get_task_list(get_task_dict(['mmlu_abstract_algebra'], manager))
    b.require(len(leaves) == 1, 'Require unchanged template only.')
    task = leaves[0].task
    template = '{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:'
    b.require(task.get_config('doc_to_text') == template
        and task.get_config('doc_to_choice') == ['A','B','C','D']
        and task.get_config('doc_to_target') == 'answer', 'Native no-subject four-choice template changed.')
    sciq, selection = [], []
    for selection_index, (digest, key, index) in enumerate(selected):
        row = raw[index]
        b.require(set(row) == {'question','distractor1','distractor2','distractor3','correct_answer','support'}
            and all(isinstance(row[name], str) and row[name].strip() for name in
                ('question','distractor1','distractor2','distractor3','correct_answer')), 'Invalid selected source; do not skip or replace.')
        question = row['question'].strip()
        gold = _stable_sciq_answer_index(question)
        choices = [row[name].strip() for name in ('distractor1','distractor2','distractor3')]
        choices.insert(gold, row['correct_answer'].strip())
        doc = dict(question=question, choices=choices, answer=gold, subject='')
        context = task.doc_to_text(doc)
        expected = question + '\n' + '\n'.join(f'{chr(65+i)}. {text}' for i,text in enumerate(choices)) + '\nAnswer:'
        b.require(context == expected, 'Prompt must contain exactly question/options/Answer, without support or subject.')
        variant = native.native_variant(task, doc, context, codec, 'allenai_sciq_native_no_subject', index)
        pairs = variant['pairs']
        b.require([pair['arguments'][1] for pair in pairs] == [' A',' B',' C',' D']
            and all(pair['arguments'][0] == expected and pair['context_ids'] == pairs[0]['context_ids'] for pair in pairs)
            and [pair['continuation_ids'] for pair in pairs] == [[362],[425],[356],[422]], 'Native complete-pair token boundaries changed.')
        ids = common.native.cpu_ids(pairs[0]['context_ids'])
        query = common.native.cpu_ids([len(ids)-1])
        case = common.native.finish_case(dict(domain='mmlu', case_id=key, split='calibration',
            input_ids=ids, prompt_input_ids=ids,
            native_continuation_ids=common.native.cpu_ids([pair['continuation_ids'] for pair in pairs]),
            native_arguments=[pair['arguments'] for pair in pairs], gold_index=gold,
            target_ids=common.native.cpu_ids(pairs[gold]['continuation_ids']),
            answer_query_positions=query, answer_target_positions=query+1,
            source=dict(data_path=str(path), source_identity=['allenai/sciq','default','train',index],
                repository='allenai/sciq', revision=b.REVISION, config='default', source_split='train', official_split='train',
                source_row_0based=index, subject='', doc_sha256=common.native.digest(doc),
                raw_record_sha256=common.native.digest(row), variant='native_no_subject',
                pair_tokens_sha256=[pair['pair_tokens_sha256'] for pair in pairs]), evidence_location=None))
        common.verify_case(case)
        b.mature.mmlu.targets.check_mmlu_case(case)
        b.require(len(ids) == inventory['selection'][selection_index]['tokens']
            and gold == inventory['selection'][selection_index]['gold']
            and common.native.digest(row) == inventory['selection'][selection_index]['doc_sha256'], 'Actual native SciQ differs from design inventory.')
        sciq.append(case)
        selection.append(dict(case_id=key, source_row_0based=index, hash=digest, gold_index=gold,
            input_tokens=len(ids), input_signature=case['input_signature'], raw_record_sha256=common.native.digest(row)))
    by_id = {case['case_id']:case for case in old['training_cases']}
    groups, kept, replaced, qa_ids = deepcopy(old['groups']), [], [], []
    for index, group in enumerate(groups):
        original = old['groups'][index]
        kept.extend(original['case_ids'][:16])
        replaced.extend(original['case_ids'][16:32])
        qa_ids.extend(original['case_ids'][32:])
        cases = [by_id[key] for key in original['case_ids'][:16]] + sciq[index*16:(index+1)*16]
        cases += [by_id[key] for key in original['case_ids'][32:]]
        group['case_ids'] = [case['case_id'] for case in cases]
        group['input_signatures'] = [case['input_signature'] for case in cases]
        group['weights'] = {case['case_id']:.5/(32 if case['domain'] == 'mmlu' else 8) for case in cases}
        b.require(group['case_ids'][:16] == original['case_ids'][:16]
            and group['case_ids'][32:] == original['case_ids'][32:]
            and group['input_signatures'][:16] == original['input_signatures'][:16]
            and group['input_signatures'][32:] == original['input_signatures'][32:]
            and all(group[name] == original[name] for name in group if name not in ('case_ids','input_signatures','weights'))
            and all(group['weights'][key] == original['weights'][key] for key in group['case_ids'][:16]+group['case_ids'][32:]),
            'Only last16 M slots may change; preserve all group metadata and original retained/QA weights.')
    m_cases = [case for index in range(32) for case in
        ([by_id[key] for key in old['groups'][index]['case_ids'][:16]] + sciq[index*16:(index+1)*16])]
    qa_cases = [case for case in old['training_cases'] if case['domain'] != 'mmlu']
    cases = deepcopy(m_cases + qa_cases)
    new_by_id = {case['case_id']:case for case in cases}
    for key in kept + list(dict.fromkeys(qa_ids)):
        common.require_same(new_by_id[key], by_id[key])
    exposure = Counter(key for group in groups for key in group['case_ids'])
    b.require(len(cases) == len(new_by_id) == 1152 and len(kept) == len(set(kept)) == 512
        and len(set(qa_ids)) == 128 and len(qa_ids) == 256
        and all(exposure[key] == (1 if case['domain'] == 'mmlu' else 2) for key,case in new_by_id.items()), 'Changed counts or original QA schedule.')
    tokens = sum(len(case['input_ids']) for case in m_cases)
    b.require(tokens == 201912 and sum(len(by_id[key]['input_ids']) for key in kept) == 182935
        and sum(len(case['input_ids']) for case in sciq) == 18977
        and max(len(case['input_ids']) for case in m_cases) == 958, 'Final native token counts differ.')
    evidence = dict(schema='C1_SciQ512_aux512_native_inputs_v1', capture=b.binding(b.DATA_ROOT/'capture.json'),
        source_files={name:{key:item[key] for key in ('path','bytes','sha256')} for name,item in capture['files'].items()},
        selection=selection, source_rows=11679, sciq_gold_counts=dict(Counter(case['gold_index'] for case in sciq)),
        official_split='train', revision=b.REVISION, license='CC-BY-NC-3.0', support_in_input=False,
        fixed_hash_rule='sha256(causal_output_fit/137/+case_id), tie case_id; first512; no filtering',
        original_cache=cache, original_target_evidence=target_evidence,
        retained_case_ids=kept, replaced_case_ids=replaced, QA_presentation_order=qa_ids,
        retained_aux_and_QA_values_order_weights_exact=True,
        presentations=dict(aux=512,sciq=512,qasper=64,hotpotqa=192),
        M_input_tokens=dict(control=364534,retained_aux=182935,replaced_aux=181599,sciq=18977,candidate=tokens),
        M_target_positions=1024, M_maximum_input_tokens=958, paired_equal_tokens=False,
        QA_prompt_tokens={domain:sum(len(new_by_id[key]['prompt_input_ids']) for key in qa_ids
            if new_by_id[key]['domain'] == domain) for domain in ('qasper','hotpotqa')},
        readout_case_ids=dict(sciq=[case['case_id'] for case in sciq[:32]],aux=kept[:32]),
        readout_selection='Fixed first32 selected SciQ plus first32 retained auxiliary; no model score used.',
        CUDA_initialized=b.torch.cuda.is_initialized())
    return dict(training_cases=cases,groups=groups,train_ids=[case['case_id'] for case in cases],evidence=evidence,
        retained_mixed_targets={key:recovery[key]['target_log_probs'] for key in kept})


def read_dataset():
    manifest = b.read(b.OUTPUT / 'dataset.json')
    b.require(b.binding(manifest['payload']['path']) == manifest['payload'], 'Changed SciQ dataset payload.')
    return b.torch.load(manifest['payload']['path'], map_location='cpu', weights_only=True)


def load_inputs():
    dataset = read_dataset()
    manifest = b.read(b.OUTPUT / 'teacher/cache.json')
    b.require(manifest['status'] == 'verified_serial_HF_SciQ512_once_mixed' and manifest['teacher_fully_released']
        and manifest['dataset'] == b.binding(b.OUTPUT/'dataset.pt')
        and b.binding(manifest['payload']['path']) == manifest['payload'], 'Changed verified SciQ target cache.')
    payload = b.torch.load(manifest['payload']['path'], map_location='cpu', weights_only=True)
    recovery = {}
    for case in dataset['training_cases']:
        key = case['case_id']
        recovery[key] = dict(case=case)
        if case['domain'] == 'mmlu':
            target = dataset['retained_mixed_targets'].get(key)
            if target is None:
                item = payload['targets'][key]
                b.require(item['input_signature'] == case['input_signature'], 'SciQ target/input binding differs.')
                target = item['target_log_probs']
                recomputed = b.mature.aligned.reference.mixed.helper.target_log_distribution(item['pure_teacher_log_probs'], case)
                b.require(b.torch.equal(target,recomputed), 'SciQ target must be mixed once by the original helper.')
            b.require(target.shape == (1,4) and target.dtype == b.torch.float32
                and b.torch.isfinite(target).all() and not target.requires_grad, 'Invalid frozen M target.')
            recovery[key]['target_log_probs'] = target
    b.require(len(recovery) == 1152 and sum('target_log_probs' in item for item in recovery.values()) == 1024,
        'Keep M1024 targets and discard all old QA trajectories.')
    evidence = dict(schema='C1_SciQ512_aux512_once_mixed_targets_v1', retained_aux512_exact=True,
        new_SciQ512_serial_B1=True, mixture_applications=1, cache=b.binding(b.OUTPUT/'teacher/cache.json'),
        input_target_sha256=b.core.mapping_digest({key:dict(input_signature=item['case']['input_signature'],
            target=b.core.tensor_record(item['target_log_probs'])) for key,item in recovery.items() if 'target_log_probs' in item}))
    return dataset,recovery,dict(manifest=b.binding(b.OUTPUT/'teacher/cache.json'),source='Retained original mixed aux512 plus new serial SciQ512'),evidence
