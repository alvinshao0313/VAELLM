"""Bind the completed base and existing train sources; never start GPU work."""
import argparse
from collections import deque
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time

import pyarrow.parquet as pq
import torch

from e2e_common.data import DATASET_MIX_SOURCE_PRESETS, record_to_mcqa_example


ROOT = Path(__file__).resolve().parents[2]
FIRST = ROOT / 'artifacts/sc_cat/qwen--qwen3-8b/2026-11-05_184620_s31'
FINAL = FIRST / 'live_gate_up_20261107/gpu0_3/repair_20261108/after_fresh_fix/formal'
DATA = ROOT / 'result/qwen3_8b_mmlu65_20261004/cat_auto_protocol_cmcc_20261104/full_data_v0'
AUX = Path('/data1/cmcc/VAELLM/data/cais_mmlu_auxiliary_train_c30699e8356da336a370243923dbaf21066bb9fe')
WEBMED = Path('/data1/cmcc/VAELLM/data/s4_webmed_sources_20261105/sources.json')
KNOWLEDGE = {'A': [100, 0, 0], 'B': [50, 50, 0], 'C': [50, 0, 50],
             'D': [0, 50, 50], 'E': [50, 25, 25]}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    before = path.stat()
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), f'Changed while reading: {path}')
    return value.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def checked(reference):
    require(sha(reference['path']) == reference['sha256'], f'Binding mismatch: {reference["path"]}')
    return read(reference['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preparation requires hidden CUDA.')
    require(not output.exists(), 'Use a new preparation file.')
    began = time.monotonic()
    selection = read(FINAL / 'terminal/selection.json')['selected']
    boundary = read(FINAL / 'after_up.json')
    deployment = checked(selection['deployment']['deployment_report'])
    nine = checked(selection['nine_task'])
    checked(selection['deployment']['score'])
    model = Path(boundary['path'])
    meta = read(model / 'checkpoint_meta.json')
    weight = model / meta['state_dict_file']
    require(boundary['uuid'] == selection['checkpoint_id'] == nine['checkpoint_id'] == meta['checkpoint_id'],
            'Base, selection, deployment and metrics must refer to one artifact.')
    require(len(meta['compressed_targets']) == 252 and not meta['pending_dense_targets']
            and len(boundary['completed_prefix']) == 7, 'Second stage requires the complete compressed base.')
    weight_sha = sha(weight)
    require(weight_sha == boundary['digest'] == deployment['digest'], 'Actual base weights changed.')
    protected = [m['name'] for m in meta['converted_modules']
                 if m.get('protected_input_qvalues') and m.get('protected_input_scales')]
    require(protected and nine['budget_passed'] and nine['exit_code'] == 0,
            'Need existing protection and valid complete evaluation; quality pass is reported separately.')
    print(json.dumps({'event': 'complete_base_verified', 'uuid': boundary['uuid'],
                      'compressed': 252, 'protected': len(protected)}, ensure_ascii=False), flush=True)

    report = read(DATA / 'report.json')
    require(sha(DATA / 'data_plan.pt') == report['data_plan_sha256'], 'Original data-plan identity changed.')
    plan = torch.load(DATA / 'data_plan.pt', map_location='cpu', weights_only=True)
    registry = plan['local_sources_registry']
    original = []
    for source in registry['sources']:
        path = ROOT / DATASET_MIX_SOURCE_PRESETS[source['alias']].path
        require(path.is_file(), f'Missing original train artifact: {path}')
        original.append(dict(source, path=str(path.resolve()), bytes=path.stat().st_size,
                             identity_verification='reuse original bound source audit; not rehashed here'))
    capture = read(AUX / 'capture.json')
    require(capture['config'] == 'auxiliary_train' and capture['split'] == 'train', 'Wrong auxiliary split.')
    for file in capture['files'].values():
        require(sha(file['path']) == file['sha256'], 'Auxiliary source/card hash changed.')
    aux = pq.read_table(AUX / capture['selected_parquet'][0], use_threads=False).to_pylist()
    require(len(aux) == 99842, 'Unexpected auxiliary train row count.')
    task = next(row for row in original if row['alias'] == 'vaellm_eval_task')
    require(sha(task['path']) == task['sha256'], 'Original task mixture changed.')
    # Verify every source row, rather than infer source labels from text or row offsets alone.
    tail = deque(maxlen=len(aux))
    with Path(task['path']).open(encoding='utf-8') as stream:
        count = 0
        for count, line in enumerate(stream, start=1):
            tail.append(line)
    require(count == task['rows'], 'Original task mixture row count changed.')
    for index, (record, line) in enumerate(zip(aux, tail)):
        rendered = record_to_mcqa_example(record, text_format='mmlu_mcqa')
        require(rendered is not None, f'Unrenderable auxiliary row {index}.')
        expected = {'messages': [{'role': 'user', 'content': str(rendered['prompt'])},
                                {'role': 'assistant', 'content': ' ' + chr(65 + rendered['answer_index'])}]}
        require(json.loads(line) == expected, f'Auxiliary-to-original task identity mismatch at row {index}.')
    task['identity_verification'] = 'SHA256 and all auxiliary rows rendered and matched exactly'
    print(json.dumps({'event': 'original_knowledge_subpool_verified', 'matched': len(aux),
                      'task_rows': count, 'other_task_rows': count - len(aux)}), flush=True)

    additions = read(WEBMED)
    for name, source in additions.items():
        require(source['official_split'] == 'train' and not source['test_or_validation_downloaded'],
                'Only pinned official train additions are permitted.')
        require(sha(source['path']) == source['sha256'] and sha(source['card_path']) == source['card_sha256'],
                f'{name} source or card changed.')
        require(pq.ParquetFile(source['path']).metadata.num_rows == source['rows'], f'{name} row count changed.')

    task_share = Fraction(str(plan['manifest']['source_weights']['vaellm_eval_task']))
    knowledge_share = task_share * Fraction(len(aux), count)
    candidates = {key: dict(knowledge_percent=dict(zip(('auxiliary', 'web', 'med'), values)),
                           whole_stream_share={name: str(knowledge_share * Fraction(value, 100))
                                               for name, value in zip(('auxiliary', 'web', 'med'), values)},
                           stage='knowledge_data', parent_checkpoint_id=boundary['uuid'])
                  for key, values in KNOWLEDGE.items()}
    require(all(sum(v) == 100 for v in KNOWLEDGE.values()), 'Knowledge fractions must conserve the subpool.')
    require(not torch.cuda.is_initialized(), 'CPU preparation unexpectedly initialized CUDA.')
    result = dict(status='base_and_source_CPU_audit_passed_execution_settings_pending',
                  GPU_training_started=False, cuda_initialized=False,
                  authorization_scope='User selected existing complete SC-CAT research base and protocol preparation.',
                  base=dict(boundary=binding(FINAL / 'after_up.json'), selection=binding(FINAL / 'terminal/selection.json'),
                            uuid=boundary['uuid'], path=str(model.resolve()),
                            weight=dict(path=str(weight.resolve()), sha256=weight_sha),
                            metadata=binding(model / 'checkpoint_meta.json'), protected_projections=len(protected),
                            complete_nine=selection['nine_task'], S8=nine['full8_mean_percent'],
                            S9=nine['compressed_nine_task_score'], LongBench=nine['longbench_task_macro_mean'],
                            quality_passed=nine['quality_passed'], ledger=boundary['actual_ledger']),
                  original_data=dict(plan=binding(DATA / 'data_plan.pt'), sources=original,
                                     source_weights=plan['manifest']['source_weights'],
                                     knowledge_rows=len(aux), task_rows=count, knowledge_total_share=str(knowledge_share),
                                     auxiliary_start_row=count-len(aux), auxiliary_rows_exactly_matched=True),
                  auxiliary_capture=binding(AUX / 'capture.json'), new_sources=additions,
                  knowledge_candidates=candidates,
                  sequence=['scope_D_and_DP_on_A', 'knowledge_A_B_C_D_E_on_selected_scope',
                            'remaining_short_text_search_if_confirmed', 'selected_short_checkpoint_full_nine'],
                  fresh_training_count=6, original_control_reused_from_scope_winner=True,
                  unresolved=['optimizer and learning rates', 'training schedule and exposure budget',
                              'batch/accumulation and actual length handling', 'evaluation nodes and ties',
                              'GPU cost proposal after the concrete execution settings'],
                  implementation_scope='Input/identity preparation only; no second-stage trainer or GPU validation claimed.',
                  source_pins={str(path): sha(path) for path in [Path(__file__), ROOT/'e2e_common/data.py']},
                  protocol=binding(ROOT / 'docs/sc_cat_protocol.md'), elapsed_CPU_seconds=time.monotonic()-began)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'output': str(output),
                      'elapsed_CPU_seconds': result['elapsed_CPU_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
