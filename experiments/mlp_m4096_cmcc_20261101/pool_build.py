"""The same official auxiliary_train native builder, extended to fixed hash-prefix4096."""
from experiments.muon_compact_cmcc_20261030.bindings import R, module
original = module(R / 'qwen_mmlu_aux_train_data_20261010/aux_core.py', '_coverage_existing_aux_builder')
require, native, data, fit_data = original.require, original.native, original.data, original.fit_data

def build(tokenizer, auxiliary, source):
    from lm_eval.evaluator_utils import get_task_list
    from lm_eval.tasks import TaskManager, get_task_dict

    # This existing task supplies only the common native MMLU text/choice methods.
    # Its subject description is never included and its evaluation docs are not selected.
    manager = TaskManager()
    leaves = get_task_list(get_task_dict(['mmlu_abstract_algebra'], manager))
    require(len(leaves) == 1, 'Require one unchanged common MMLU template.')
    task = leaves[0].task
    require(task.get_config('doc_to_text') == '{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:'
        and task.get_config('doc_to_choice') == ['A', 'B', 'C', 'D'] and task.get_config('doc_to_target') == 'answer', 'Native no-subject template changed.')
    codec = native.PairTokenizer(tokenizer)
    ranked = sorted((fit_data.rank(f'cais_mmlu/auxiliary_train/train/{index}'),
        f'cais_mmlu/auxiliary_train/train/{index}', index) for index in range(len(auxiliary)))
    selected = ranked[:4096]
    m_cases, selection = [], []
    for rank, (hash_value, case_id, index) in enumerate(selected):
        doc = dict(auxiliary[index]['train'])
        require(set(doc) == {'question', 'choices', 'answer', 'subject'} and doc['subject'] == ''
            and isinstance(doc['question'], str) and len(doc['choices']) == 4
            and isinstance(doc['answer'], int) and 0 <= doc['answer'] < 4, 'Invalid fixed selected auxiliary row; do not skip or replace it.')
        context = task.doc_to_text(doc)
        variant = native.native_variant(task, doc, context, codec, 'cais_mmlu_auxiliary_train_native_no_subject', index)
        pairs = variant['pairs']
        require(all(pair['arguments'][0] == context and pair['context_ids'] == pairs[0]['context_ids'] for pair in pairs)
            and [pair['continuation_ids'] for pair in pairs] == [[362], [425], [356], [422]], 'Actual native context/continuation boundary differs.')
        input_ids = data.cpu_ids(pairs[0]['context_ids']); query = data.cpu_ids([len(input_ids) - 1])
        gold = doc['answer']
        case = data.finish_case({'domain': 'mmlu', 'case_id': case_id, 'split': 'calibration',
            'input_ids': input_ids, 'prompt_input_ids': input_ids,
            'native_continuation_ids': data.cpu_ids([pair['continuation_ids'] for pair in pairs]),
            'native_arguments': [pair['arguments'] for pair in pairs], 'gold_index': gold,
            'target_ids': data.cpu_ids(pairs[gold]['continuation_ids']),
            'answer_query_positions': query, 'answer_target_positions': query + 1,
            'source': {'data_path': source['arrow_path'], 'source_identity': [source['repo'], source['config'], source['split'], index],
                'repository': source['repo'], 'revision': source['revision'], 'config': source['config'],
                'source_split': source['split'], 'official_split': source['split'], 'source_row_0based': index,
                'subject': '', 'doc_sha256': data.digest(doc), 'variant': 'native_no_subject',
                'pair_tokens_sha256': [pair['pair_tokens_sha256'] for pair in pairs]}, 'evidence_location': None})
        m_cases.append(case)
        selection.append({'selection_rank_0based': rank, 'source_row_0based': index, 'case_id': case_id,
            'hash': hash_value, 'doc_sha256': data.digest(doc), 'gold_index': gold,
            'prompt_tokens': len(input_ids), 'input_signature': case['input_signature']})
    return m_cases, selection, {
        'task_yaml_path': manager.task_index['mmlu_abstract_algebra']['yaml_path'],
        'native_template': task.get_config('doc_to_text'), 'task_description_used': False,
        'formal_test_documents_sampled': 0, 'formal_test_comparison_performed': False,
        'template_only_reuse': 'Existing common MMLU task methods; actual source is the selected flat auxiliary_train document.'}
