"""Validate unchanged train materials; bind formal OPD only after actual 01 audit."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import bindings as b
from rollout import replay_case


def materials():
    dataset, recovery, cache, targets = b.load_inputs()
    tokenizer = b.core.build_tokenizer(b.teacher_recipe()['model']['hf_path'])
    previous = b.read(b.R / 'qwen_hotpot_train_cpu_20261011/generation_prepared.json')
    hot_options = {row['case_id']: row['generation_options'] for row in previous['cases']}
    config = b.teacher_recipe()
    b.qa.check_config(config)
    eos, rows = config['protocol']['eos_token_id'], []
    for case in dataset['training_cases'][1024:]:
        limit = b.LIMITS[case['domain']]
        options = b.qa.generation_options(tokenizer, case['domain'], len(case['prompt_input_ids']), limit,
            protocol=config['protocol'])
        expected = case['qa_case']['generation_options'] if case['domain'] == 'qasper' else hot_options[case['case_id']]
        b.require(options == expected and case['source']['official_split'] == 'train', 'Changed original train QA options.')
        for tokens in ([eos[0]], [0]*(limit-1)+[eos[0]], [0]*limit):
            replay = replay_case(case, {'output_token_ids': tokens}, eos)
            b.require(b.torch.equal(replay['input_ids'], b.torch.cat((case['prompt_input_ids'],
                b.torch.tensor(tokens[:-1], dtype=b.torch.long))))
                and replay['target_ids'].tolist() == tokens
                and replay['answer_query_positions'].tolist() == list(range(len(case['prompt_input_ids'])-1,
                    len(case['prompt_input_ids'])+len(tokens)-1)), 'Changed complete-prefix replay or EOS semantics.')
        rows.append(dict(case_id=case['case_id'], domain=case['domain'], input_signature=case['input_signature'],
            prompt_tokens=len(case['prompt_input_ids']), maximum_replay_tokens=len(case['prompt_input_ids'])+limit-1,
            generation_options=options))
    by_id, short = {row['case_id']: row for row in rows}, []
    for index in b.STEPS['short2']:
        group = dataset['groups'][index]
        qa_rows = [by_id[key] for key in group['case_ids'][32:]]
        short.append(dict(group_index=index, QA_documents=qa_rows,
            M_batch_shape=[32, max(len(recovery[key]['case']['input_ids']) for key in group['case_ids'][:32])],
            longest_QA=max(qa_rows, key=lambda row: row['maximum_replay_tokens'])))
    b.require(short[0]['longest_QA']['domain'] == 'qasper'
        and short[0]['longest_QA']['maximum_replay_tokens'] == 10713
        and short[1]['longest_QA']['domain'] == 'hotpotqa'
        and short[1]['longest_QA']['maximum_replay_tokens'] == 2584, 'Changed representative original groups2/14.')
    validation = dict(fixed_cases=len(dataset['training_cases']), M1024_Q32_H96=True,
        all_actual_sources_train_only=True, M_once_QA_twice=True, M_teacher50_gold50_once=True,
        fullF_target_components=2, original_QA_options_cases=128, EOS_limit_full_prefix_replays=384,
        no_old_student_logits_or_retention_targets=True, QA_teacher_forcing_targets_removed=True,
        CUDA_initialized=b.torch.cuda.is_initialized(), FW=0, training=False, export=False)
    return dataset, recovery, cache, targets, rows, short, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materials-only', action='store_true', help='Verify source-independent materials without binding a student.')
    args = parser.parse_args()
    started = time.monotonic()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
              and not b.torch.cuda.is_initialized(), 'Preparation is CPU-only in caller-activated bitvae.')
    b.torch.set_num_threads(4)
    # Fail before reading large caches if an unready student was requested.
    student = None if args.materials_only else b.source()
    if not args.materials_only:
        b.require(not b.OUTPUT.exists(), 'Use a new, isolated aligned OPD output.')
    dataset, recovery, cache, targets, rows, short, validation = materials()
    import run
    import export
    import audit01
    files = list(b.HERE.glob('*.py')) + list(b.HERE.glob('*.sh'))
    # Reuse the exact immutable reference helper/data/teacher pins, without changing their globals.
    files += [Path(item['path']) for item in b.read(b.REFERENCE / 'prepared.json')['bindings'].values()]
    files += [b.REFERENCE / 'prepared.json']
    bindings = {str(path.resolve()): b.binding(path) for path in files}
    common = dict(actual_target_contract=b.CONTRACT, dataset_binding=b.read(b.reference.HOT / 'dataset_prepared.json')['dataset_binding'],
        data_contract=dataset['evidence'], teacher_cache=cache, unchanged_M_target_evidence=targets,
        QA_generation_bindings=rows, short2_representative_inputs=short, CPU_validation=validation,
        bindings=bindings, elapsed_seconds=time.monotonic()-started)
    if args.materials_only:
        path = b.PARENT.parent / 'opd_cpu_materials.json'
        b.require(not path.exists(), 'Do not overwrite the source-independent material evidence.')
        result = dict(schema='aligned_OPD_source_independent_materials_v1', status='materials_verified_student_unbound',
            student_source_bound=False, GPU_bridge_validated=False, **common)
    else:
        directory, meta, audit = student
        state = b.load_state(directory, meta)
        names = list(b.hot.student_projection_names(meta['base_model_path'], 'all'))
        declared, frozen = b.hot.state_inventory(state, names)
        b.require(len(state) == 3423 and len(names) == 252 and len(declared) == 1512
            and frozen['tensors'] == 1911 and set(names) == set(meta['compressed_targets'])
            and all(item['residual_stages'] == 2 and item['parallel_parts'] == 1
                and all(d['num_res_blocks'] == 0 and d['param_dtype'] == 'bfloat16' for d in item['stage_decoders'])
                for item in meta['converted_modules']), 'Actual aligned native decoder topology differs.')
        record = dict(source_checkpoint_id=meta['checkpoint_id'], model_config=b.model_config(directory, meta),
            student_scope='all', student_projection_names=names, decoder_parameters=declared,
            trainable_elements=sum(item['elements'] for item in declared.values()), validated_frozen_state=frozen,
            teacher_projection_names=names, teacher_kind='full7', teacher_identity=b.teacher_identity(state, meta, names, directory),
            groups=dataset['groups'], train_ids=dataset['train_ids'], retention_lambda=0., steps=32,
            mmlu_execution_batch_size=32, dataset_binding=common['dataset_binding'])
        # Source geometry is retained in full; K34 gate and fullRHT Q/K38 need no old-shape surrogate.
        result = dict(schema='CG_aligned_online_OPD32_v1', status='CPU_prepared_GPU_unvalidated', profile=b.PROFILE,
            source_checkpoint=str(directory), source_checkpoint_id=meta['checkpoint_id'], source_audit=b.binding(b.SOURCE_AUDIT),
            source_weight_stat=b.file_stat(directory / meta['state_dict_file']), source_bit_budget=audit['bit_budget'],
            decoder_parameters=declared, actual_inventory=dict(projections=len(names), tensors=len(declared),
                elements=record['trainable_elements'], frozen_tensors=frozen['tensors'], source_state_tensors=len(state)),
            source_geometry={item['name']: item for item in meta['converted_modules']}, record=record,
            optimizer=dict(name='AdamW', betas=[.9, .95], eps=1e-8, weight_decay=0, learning_rate=1e-6,
                schedule='1e-6*(1-(step-1)/500)', clip_norm=1.5, seed=137),
            phases=b.STEPS, limitations='CPU bindings only; actual source native greedy/FP32 master/Adam/cache bridge remains GPU-unvalidated.',
            **common)
        path = b.OUTPUT / 'prepared.json'
        b.OUTPUT.mkdir()
    b.require(not b.torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    b.core.write(path, result)
    print(json.dumps(dict(status=result['status'], path=str(path), CPU_validation=validation,
        source_checkpoint_id=result.get('source_checkpoint_id'), elapsed_seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    main()
