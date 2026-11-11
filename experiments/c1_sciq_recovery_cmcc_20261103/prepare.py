"""CPU-only source/layout reuse and complete native data/target binding."""
from copy import deepcopy
import json
import os
import sys
import time

from . import bindings as b


def main():
    b.original.executor.cpu_environment()
    b.configure()
    b.require(not b.OUTPUT.exists(), 'Never overwrite this single data comparison.')
    started = time.monotonic()
    b.torch.set_num_threads(4)
    old = b.original.read_prepared(b.ORIGINAL_PREPARED)
    b.require(old['source_binding']['checkpoint_id'] == b.PARENT_ID, 'Use original C1 raw only.')
    dataset = b.inputs.build()
    b.require(not b.torch.cuda.is_initialized(), 'CPU preparation must not initialize CUDA.')
    b.OUTPUT.mkdir(parents=True)
    b.EVIDENCE.mkdir(parents=True)
    b.torch.save(dataset, b.OUTPUT/'dataset.pt')
    saved = b.torch.load(b.OUTPUT/'dataset.pt', map_location='cpu', weights_only=True)
    _native, common = b.inputs.helpers()
    common.require_same(saved,dataset)
    manifest = dict(schema='C1_SciQ_data_native_CPU_v1', status='verified_native_data_and_original_C1',
        payload=b.binding(b.OUTPUT/'dataset.pt'), evidence=dataset['evidence'],
        original_prepared=b.binding(b.ORIGINAL_PREPARED), source_binding=old['source_binding'],
        source_budget=old['source_budget'], Rice_lifecycle_evidence=b.verified_Rice_short(b.PREPARED),
        source_geometry_and_frozen_record_reused=True, no_new_GPU_short=True,
        bindings={**b.code_bindings(), **{str(path):b.binding(path) for path in
            (b.ORIGINAL_PREPARED,b.DATA_ROOT/'capture.json',b.DATA_ROOT/'design_inventory.json')}},
        CUDA_initialized=False, elapsed_seconds=time.monotonic()-started)
    b.core.write(b.OUTPUT/'dataset.json',manifest)
    print(json.dumps(dict(status=manifest['status'],payload=manifest['payload'],
        M_input_tokens=dataset['evidence']['M_input_tokens'], sciq_gold_counts=dataset['evidence']['sciq_gold_counts'],
        seconds=manifest['elapsed_seconds'])),flush=True)


def finalize():
    b.original.executor.cpu_environment()
    b.configure()
    b.require(not b.PREPARED.exists(), 'Never overwrite finalized preparation.')
    old = b.original.read_prepared(b.ORIGINAL_PREPARED)
    manifest = b.read(b.OUTPUT/'dataset.json')
    for item in manifest['bindings'].values():
        b.require(b.binding(item['path']) == item, 'CPU preparation source changed.')
    dataset,recovery,cache,evidence = b.inputs.load_inputs()
    old_dataset,old_recovery,_cache,_evidence = b.original.inputs.load_inputs()
    _native, common = b.inputs.helpers()
    old_by_id = {case['case_id']:case for case in old_dataset['training_cases']}
    for case in dataset['training_cases']:
        if not case['case_id'].startswith('allenai_sciq/'):
            common.require_same(case,old_by_id[case['case_id']])
            if case['domain'] == 'mmlu':
                b.require(b.torch.equal(recovery[case['case_id']]['target_log_probs'],
                    old_recovery[case['case_id']]['target_log_probs']), 'Retained auxiliary target changed.')
    teacher = b.read(b.OUTPUT/'teacher/cache.json')
    b.require(teacher['teacher_identity'] == old['record']['teacher_identity'] and teacher['teacher_fully_released'],
        'Serial SciQ teacher must be the exact original full-HF identity and fully released.')
    record = deepcopy(old['record'])
    record.update(groups=dataset['groups'],train_ids=dataset['train_ids'])
    bound = b.code_bindings()
    for path in (b.ORIGINAL_PREPARED,b.OUTPUT/'dataset.json',b.OUTPUT/'dataset.pt',
        b.OUTPUT/'teacher/cache.json',b.OUTPUT/'teacher/targets.pt',b.DATA_ROOT/'capture.json',b.DATA_ROOT/'design_inventory.json'):
        bound[str(path)] = b.binding(path)
    value = dict(schema='C1_SciQ_data_Muon_prepared_v1',status='CPU_source_data_and_serial_HF_cache_verified',
        source_binding=old['source_binding'],source_budget=old['source_budget'],source_geometry=old['source_geometry'],
        record=record,contract=b.CONTRACT,group_indices=b.GROUPS,original_prepared=b.binding(b.ORIGINAL_PREPARED),
        data=dataset['evidence'],M_target_evidence=evidence,teacher_cache=b.binding(b.OUTPUT/'teacher/cache.json'),
        Rice_lifecycle_evidence=b.verified_Rice_short(b.PREPARED),bindings=bound,CUDA_initialized=False,
        boundary='Data-only variable; completed original Rice/Muon two-update/save/fresh engineering evidence reused; no new GPU short/exact-resume.')
    b.core.write(b.PREPARED,value)
    b.read_prepared(b.PREPARED)
    print(json.dumps(dict(status=value['status'],prepared=str(b.PREPARED),source=b.PARENT_ID)),flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--finalize']:
        finalize()
    else:
        b.require(not sys.argv[1:], 'No optional configuration for this fixed data comparison.')
        main()
