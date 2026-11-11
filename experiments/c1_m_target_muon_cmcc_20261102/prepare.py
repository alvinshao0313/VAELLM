"""Bind the completed raw C1 and its already verified Rice/Muon lifecycle."""
import os
from pathlib import Path
import sys
import time

from . import bindings as b


def main():
    began = time.monotonic()
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not b.torch.cuda.is_initialized(), 'CPU-only bitvae preparation required.')
    b.configure()
    b.require(not b.PREPARED.exists(), 'Never overwrite parallel target-study preparation.')
    old = b.read(b.CPU_SOURCE_PREPARED)
    b.require(old['status'] == 'CPU_prepared_reused_Rice_lifecycle'
        and old['original_prepared'] == b.binding(b.ORIGINAL_PREPARED), 'Reuse the completed CPU source preparation.')
    for name, item in old['bindings'].items():
        if not Path(name).is_relative_to(b.HERE):
            b.require(b.binding(item['path']) == item, 'Original shared source/input changed: ' + item['path'])
    parent, _ = b.original.source(old['source_binding']['checkpoint_dir'], old['source_binding']['pipeline_report'])
    b.require(parent == old['source_binding'], 'The already verified raw C1 source changed.')
    dataset, recovery, cache, evidence = b.inputs.load_inputs()
    b.require(dataset['groups'] == old['record']['groups'] and dataset['train_ids'] == old['record']['train_ids']
        and evidence == old['original_M_target_evidence'], 'The original M1024/Q32/H96 input or once-mixed target changed.')
    proof = b.original.verified_Rice_short(b.ORIGINAL_PREPARED)
    value = dict(schema='C1_M_target_Muon_prepared_v1', status='CPU_prepared_reused_Rice_lifecycle',
        source_binding=old['source_binding'], source_budget=old['source_budget'], record=old['record'],
        original_prepared=b.binding(b.ORIGINAL_PREPARED), group_indices=b.GROUPS, contract=b.CONTRACT,
        CPU_source_preparation=b.binding(b.CPU_SOURCE_PREPARED),
        Rice_lifecycle_evidence=proof, original_M_target_evidence=evidence, old_teacher_cache=cache,
        bindings=b.code_bindings(), elapsed_seconds=time.monotonic() - began,
        boundary='No new lifecycle short-chain. New full-M target cache and fixed first-M32 real loss/backward probe remain required.')
    b.RUN_OUTPUT.mkdir(parents=True, exist_ok=True)
    b.core.write(b.PREPARED, value)
    print(dict(status=value['status'], seconds=value['elapsed_seconds'], source=old['record']['source_checkpoint_id']), flush=True)


if __name__ == '__main__':
    main()
