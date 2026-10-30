"""CPU-only verification of the one-domain coverage change and actual runtime wiring."""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import bindings as b


def reused_evidence():
    root = b.parent.OUTPUT
    old, _ = b.parent.read_prepared()
    paths = [root / 'short2/report.json', root / 'short2/native_audit.json',
        root / 'formal32/report.json', root / 'native_audit.json']
    short, short_audit, formal, audit = [b.read(path) for path in paths]
    b.require(short['steps'] == 2 and short['prepared_sha256'] == b.sha256(root / 'prepared.json')
        and short['source_restored_exact'] and short['saved_decoder_matches_final_FP32_cast_exact']
        and short_audit['status'] == audit['status'] == 'verified'
        and short_audit['serialized_tensors_compared'] == short_audit['fresh_strict_tensors_compared'] == 3423
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
        and formal['steps'] == formal['training']['steps'] == 32
        and formal['training']['group_indices'] == list(range(32))
        and formal['source_checkpoint_id'] == audit['source_checkpoint_id'] == b.SOURCE_ID
        and formal['prepared_sha256'] == b.sha256(root / 'prepared.json')
        and all((root / item).read_text().strip() == '0' for item in
            ('short2/run.exit', 'short2.exit', 'formal32/run.exit', 'formal32.exit')),
        'The actual same-source Muon two-update/export/formal evidence is incomplete.')
    b.require(old['source_weight_stat'] == b.file_stat(Path(old['source_checkpoint']) /
        b.read(Path(old['source_checkpoint']) / 'checkpoint_meta.json')['state_dict_file']),
        'The protected G09 source differs from the mature evidence.')
    return dict(evidence={str(path):b.binding(path) for path in paths},
        same_G09_source_and_matrix_layout=True, unchanged_mature_GPU_math_and_lifecycle=True,
        repeated_GPU_short=False, old_short_kappa_not_reused=True,
        scope='Input/group/coverage-only change; real stage-Muon2 and strict3423 export evidence reused, fresh first-real-gradient kappa required.')


def build_inputs(old_dataset, old_record):
    b.require(b.sha256(b.POOL) == b.POOL_SHA256, 'Changed pinned existing QA512 train dataset.')
    pool_manifest = b.read(b.POOL_MANIFEST)
    b.require(pool_manifest['dataset_binding'] == b.binding(b.POOL), 'QA512 pool and manifest binding differ.')
    pool = b.torch.load(b.POOL, map_location='cpu', weights_only=True)
    original = b.data_contract
    old_cases = old_dataset['training_cases']
    h = [case for case in pool['training_cases'] if case['domain'] == 'hotpotqa']
    b.require(len(h) == 384 and len(old_cases) == 1152, 'Require original M1024/Q32/H96 and existing H384 pool.')
    original.require_same(h[:96], old_cases[1056:])
    manifest = b.read(b.TRAIN_MANIFEST)
    source = manifest['source']
    b.require(source['upstream_release'] == 'v1.1'
        and source['immutable_revision'] == '1908d6afbbead072334abe2965f91bd2709910ab'
        and source['config'] == 'distractor' and source['split'] == 'train' and manifest['rows'] == 90447,
        'Use the pinned official HotpotQA v1.1 distractor/train release.')
    import pyarrow.parquet as pq
    ranked = []
    source_rows = {}
    selected = {case['raw_record']['id'] for case in h[:192]}
    for artifact in manifest['artifacts']:
        b.require(b.binding(artifact['local_path']) == dict(path=artifact['local_path'],
            bytes=artifact['bytes'], sha256=artifact['sha256']), 'Changed raw train shard.')
        table = pq.read_table(artifact['local_path'])
        b.require(table.num_rows == artifact['rows'], 'Raw train shard row count differs.')
        for row in table.to_pylist():
            key = 'hotpotqa/train/' + row['id']
            ranked.append((hashlib.sha256(('causal_output_fit/137/'+key).encode()).hexdigest(), key))
            if row['id'] in selected:
                source_rows[key] = row
        del table
    ranked.sort()
    b.require(len(ranked) == len({key for _,key in ranked}) == 90447
        and [key for _,key in ranked[:192]] == [case['case_id'] for case in h[:192]],
        'H192 must be the original complete-train hash prefix without filtering or replacement.')
    for case in h[:192]:
        original.verify_case(case)
        original.require_same(case['raw_record'], source_rows[case['case_id']])
        b.require(case['source']['official_split'] == 'train'
            and case['source']['hash_rank_key'] == hashlib.sha256(('causal_output_fit/137/'+case['case_id']).encode()).hexdigest()
            and original.native.digest(case['raw_record']) == case['source']['raw_record_sha256']
            and original.native.digest(case['raw_record']['context']) == case['source']['raw_context_structure_sha256']
            and case['encoding']['removed_tokens'] == 0, 'Changed Hotpot raw train content or frozen native tokenization.')
    cases = deepcopy(old_cases) + deepcopy(h[96:192])
    by_id = {case['case_id']:case for case in cases}
    groups = deepcopy(old_record['groups'])
    for index in range(20, 32):
        group = groups[index]
        batch = [by_id[key] for key in group['case_ids'][:32]] + h[96+(index-20)*8:104+(index-20)*8]
        group['case_ids'] = [case['case_id'] for case in batch]
        group['input_signatures'] = [case['input_signature'] for case in batch]
        group['weights'] = {case['case_id']:.5/(32 if case['domain'] == 'mmlu' else 8) for case in batch}
    b.require(groups[:20] == old_record['groups'][:20], 'Groups0..19 must remain completely unchanged.')
    original.require_same(cases[:1152], old_cases)
    exposure = Counter(key for group in groups for key in group['case_ids'])
    for index, group in enumerate(groups):
        old = old_record['groups'][index]
        b.require(group['group_index'] == index and group['qa_group_index'] == index%16
            and len(group['case_ids']) == 40 and sum(group['weights'].values()) == 1.
            and group['case_ids'][:32] == old['case_ids'][:32]
            and group['input_signatures'][:32] == old['input_signatures'][:32]
            and {key:group['weights'][key] for key in group['case_ids'][:32]}
                == {key:old['weights'][key] for key in old['case_ids'][:32]}, 'Changed M order/values/weights or original domain slot.')
        domains = [by_id[key]['domain'] for key in group['case_ids'][32:]]
        b.require(domains == [('qasper' if index%16 < 4 else 'hotpotqa')]*8,
            'The original Q4/H12 domain timing must remain unchanged.')
        if index < 20:
            b.require(group == old, 'Changed an untouched complete group.')
        else:
            b.require(group['case_ids'][32:] == [case['case_id'] for case in h[96+(index-20)*8:104+(index-20)*8]]
                and set(group)-{'case_ids','input_signatures','weights'} == set(old)-{'case_ids','input_signatures','weights'}
                and all(group[key] == old[key] for key in group if key not in ('case_ids','input_signatures','weights')),
                'Only the second Hotpot slot may change.')
    b.require(len(cases) == len(by_id) == 1248 and sum(exposure.values()) == 1280
        and all(exposure[case['case_id']] == (2 if case['domain'] == 'qasper' else 1) for case in cases),
        'Require M1024 once, Q32 twice, H192 once, and unchanged1280 document presentations.')
    prompt_tokens = {domain:sum(len(by_id[key]['prompt_input_ids'])*count for key,count in exposure.items()
        if by_id[key]['domain'] == domain) for domain in ('mmlu','qasper','hotpotqa')}
    old_by_id = {case['case_id']:case for case in old_cases}
    old_exposure = Counter(key for group in old_record['groups'] for key in group['case_ids'])
    old_prompt = {domain:sum(len(old_by_id[key]['prompt_input_ids'])*count for key,count in old_exposure.items()
        if old_by_id[key]['domain'] == domain) for domain in ('mmlu','qasper','hotpotqa')}
    b.require(prompt_tokens['hotpotqa'] == 277934 and old_prompt['hotpotqa'] == 278818
        and prompt_tokens['qasper'] == old_prompt['qasper'] == 408348
        and prompt_tokens['mmlu'] == old_prompt['mmlu'] == 364534, 'Actual prompt-token exposure differs from the declared design.')
    evidence = dict(schema='G09_Hotpot192_inputs_v1', parent_dataset=b.parent.read_prepared()[0]['dataset_binding'],
        QA512_pool=b.binding(b.POOL), QA512_manifest=b.binding(b.POOL_MANIFEST),
        official_train_manifest=b.binding(b.TRAIN_MANIFEST), source=source,
        license=dict(name='CC BY-SA 4.0', official_url='https://hotpotqa.github.io/'),
        selection=dict(rule='sha256(causal_output_fit/137/+case_id), tie case_id; complete90447 train hash prefix',
            pool_rows=90447, selected=192, original_H96_exact_prefix=True, new_indices=[96,192],
            filtering=False, test_based_selection=False),
        input_change='Only groups20..31 Hotpot slots; groups0..19 and every M/Q value/signature/order/weight exact.',
        qa_group_index='Original modulo16 slot retained; actual QA group presentations32, distinct QA batches28, unique QA documents224.',
        presentations=dict(mmlu=1024,qasper=64,hotpotqa=192,total=1280),
        unique_documents=dict(mmlu=1024,qasper=32,hotpotqa=192,total=1248),
        prompt_tokens=prompt_tokens, control_prompt_tokens=old_prompt, Hotpot_prompt_token_delta=-884,
        max_prompt_tokens={domain:max(len(case['prompt_input_ids']) for case in cases if case['domain'] == domain)
            for domain in ('mmlu','qasper','hotpotqa')},
        max_TF_input_tokens={domain:max(len(case['input_ids']) for case in cases if case['domain'] == domain)
            for domain in ('mmlu','qasper','hotpotqa')},
        generated_supervision='Observed from every real rollout/QA target in formal report and core log; no equality assumption.',
        CUDA_initialized=False)
    return dict(training_cases=cases, groups=groups, train_ids=[case['case_id'] for case in cases], evidence=evidence)


def main():
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and not b.torch.cuda.is_initialized(), 'CPU preparation requires activated bitvae and no visible CUDA.')
    b.require(not (b.OUTPUT / 'prepared.json').exists() and not (b.OUTPUT / 'dataset.pt').exists(), 'Do not overwrite preparation.')
    b.torch.set_num_threads(4)
    started = time.monotonic()
    parent, record = b.parent.read_prepared()
    old_dataset, old_recovery, _, _ = b.parent.load_inputs()
    record = deepcopy(record)
    dataset = build_inputs(old_dataset, record)
    record['groups'], record['train_ids'] = dataset['groups'], dataset['train_ids']
    record['coverage_contract'] = dataset['evidence']
    b.OUTPUT.mkdir(parents=True, exist_ok=True)
    b.torch.save(dataset, b.OUTPUT / 'dataset.pt')
    saved = b.torch.load(b.OUTPUT / 'dataset.pt', map_location='cpu', weights_only=True)
    b.data_contract.require_same(dataset, saved)
    import run
    import export
    b.require(run.fit.b is b and run.fit.opt.b is b and run.fit.motion.b is b and run.fit.rollout.b is b
        and run.runtime.fit is run.fit and run.runtime.rollout is run.fit.rollout
        and run.fit.rollout.original.LIMITS == b.LIMITS
        and export.original.b is b, 'Actual fit/optimizer/rollout/load_pair/export must bind this H192 contract.')
    inspect.signature(export.export).bind(b.OUTPUT / 'formal32/report.json', b.OUTPUT / 'native_final/final_model')
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for path in b.HERE.glob('*.sh'):
        subprocess.run(['bash','-n',str(path)], check=True)
    help_result = subprocess.run([str(b.PYTHON),'-B',str(b.HERE / 'evaluate.py'),'--help'],
        cwd=b.ROOT, env={**os.environ,'PYTHONPATH':str(b.ROOT),'CUDA_VISIBLE_DEVICES':''}, capture_output=True, text=True)
    b.require(help_result.returncode == 0 and '--gpus' in help_result.stdout, 'Actual full-nine entry/CLI import failed: '+help_result.stderr)
    files = list(b.HERE.glob('*.py')) + list(b.HERE.glob('*.sh'))
    files += [Path(item['path']) for item in parent['bindings'].values()]
    files += [b.MUON / 'bindings.py', b.MUON / 'fit.py', b.MUON / 'optimizer.py', b.MUON / 'rollout.py',
        b.MUON / 'motion.py', b.MUON / 'export.py', b.POOL, b.POOL_MANIFEST, b.TRAIN_MANIFEST,
        b.ROOT / 'experiments/overnight_goal_20261029/evaluate_muon.py']
    reused = reused_evidence()
    files += [Path(item) for item in reused['evidence']]
    result = dict(schema='G09_Muon_OPD_prepared_v1', experiment_schema='G09_Hotpot192_coverage_v1',
        status='CPU_verified_reused_Muon_GPU_evidence', source_checkpoint=parent['source_checkpoint'],
        source_checkpoint_id=b.SOURCE_ID, source_audit=parent['source_audit'], source_weight_stat=parent['source_weight_stat'],
        source_bit_budget=parent['source_bit_budget'], source_geometry=parent['source_geometry'],
        control_prepared=b.binding(b.parent.OUTPUT / 'prepared.json'), contract=b.CONTRACT, phases=b.STEPS,
        record=record, decoder_parameters=parent['decoder_parameters'], dataset_binding=b.binding(b.OUTPUT / 'dataset.pt'),
        bindings={str(path.resolve()):b.binding(path) for path in files},
        GPU_evidence_reuse=reused, CPU_validation=dict(actual_dataset_roundtrip_exact=True, sources_groups_exposure_verified=True,
            runtime_fit_optimizer_rollout_export_bound_exact=True, complete_nine_CLI_import_verified=True, CUDA_initialized=False),
        input_evidence=dataset['evidence'], elapsed_seconds=time.monotonic()-started)
    b.core.write(b.OUTPUT / 'prepared.json', result)
    loaded, _ = b.read_prepared()
    b.require(loaded == result, 'Prepared serialization/read contract differs.')
    actual_dataset, recovery, _, _ = b.load_inputs()
    b.data_contract.require_same(actual_dataset, dataset)
    for key,value in old_recovery.items():
        b.data_contract.require_same(recovery[key]['case'], value['case'])
        if value['case']['domain'] == 'mmlu':
            b.require(b.torch.equal(recovery[key]['target_log_probs'], value['target_log_probs']), 'Changed M mixed target.')
    b.require(not b.torch.cuda.is_initialized(), 'CPU verification initialized CUDA.')
    print(json.dumps(dict(status=result['status'], dataset_binding=result['dataset_binding'],
        input_evidence=dataset['evidence'], CPU_validation=result['CPU_validation'],
        prepared_sha256=b.sha256(b.OUTPUT / 'prepared.json'), elapsed_seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    main()
