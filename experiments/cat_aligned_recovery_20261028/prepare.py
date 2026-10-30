"""CPU-bind G09 and the exact existing FineWeb Q/K-only2000 recipe."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

import common as c
from callback import AlignedQKCallback
from experiments.longbench_v1.common import file_stat
from experiments.w25_eval.validate_decoder_subset_training import expected_inventory


def main():
    started = time.monotonic()
    c.require(Path(sys.executable).resolve() == c.PYTHON.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
              and not c.torch.cuda.is_initialized() and not c.OUTPUT.exists(), 'Fresh CPU-only preparation required.')
    old = c.read(c.REFERENCE / 'launch.prepared.json')
    completed = c.read(c.REFERENCE / 'training.json')
    c.require(completed['status'] == 'completed' and completed['training_result']['global_step'] == 2000
              and completed['callback_completed_steps'] == [400, 800, 1200, 1600, 2000],
              'Require the actually completed old QK-only2000 path.')
    directory, meta, audit = c.source()
    inventory = expected_inventory(directory, old['architecture'], old['selected_targets'])
    c.require(inventory == old['expected_inventory'], 'Actual QK38 inventory differs from the old432-leaf recipe.')
    state = c.torch.load(directory / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    banks = {key: value for key, value in state.items() if '.vq_weight' in key}
    c.require(len(banks) == 504 and all(value.dtype == c.torch.uint8 for value in banks.values()), 'Freeze all504 packed banks.')
    bank_hashes = {key: hashlib.sha256(value.contiguous().reshape(-1).view(c.torch.uint8).numpy()).hexdigest()
                   for key, value in banks.items()}
    del banks, state
    code = {path: digest for path, digest in old['source_sha256'].items()
            if not Path(path).is_relative_to(c.REFERENCE)}
    code.update({str(path): c.sha256(path) for path in c.HERE.glob('*.py')})
    code.update({str(path): c.sha256(path) for path in c.HERE.glob('*.sh')})
    inputs = {path: digest for path, digest in old['input_evidence_sha256'].items()
              if '/data/fineweb_edu_' in path}
    for path in [c.REFERENCE / 'launch.prepared.json', c.REFERENCE / 'training.json', c.SOURCE_AUDIT,
                 directory / 'checkpoint_meta.json', directory / 'config.json']:
        inputs[str(path)] = c.sha256(path)
    record = dict(schema='CG09_aligned_QK_FineWeb2000_v1', status='CPU_prepared', kind='formal_s2000', profile=c.PROFILE, gpu=c.GPU,
        automatic_retry=False, queued=False, cwd=str(c.SNAPSHOT), python_executable=str(c.PYTHON),
        output_dir=str(c.OUTPUT), source_checkpoint=str(directory), source_checkpoint_id=c.SOURCE_ID,
        source_weight_stat=file_stat(directory / meta['state_dict_file']),
        source_weight_sha256=c.sha256(directory / meta['state_dict_file']), source_code_bank_sha256=bank_hashes,
        source_bit_budget=audit['bit_budget'], architecture=deepcopy(old['architecture']),
        expected_inventory=inventory, selected_targets=old['selected_targets'],
        selected_decoder_geometry={item['name']: item['stage_decoders'] for item in meta['converted_modules']
                                   if item['name'] in old['selected_targets']},
        rotation_blocks={item['name']: item['weight_rotation']['block_size'] for item in meta['converted_modules']},
        hf_source=old['hf_source'], input_data_stat=old['input_data_stat'], data=old['data'],
        source_sha256=code, input_evidence_sha256=inputs, training_argv=c.training_arguments(),
        environment={**old['environment'], 'VAELLM_ALIGNED_PROFILE': c.PROFILE},
        validation_reuse='Actual mature QK38/fullRHT/group8/QK-only2000 runtime; current G09 strict native audit. Only source/output and non-training external evaluation callbacks differ.',
        evaluation='No per400 external full8; keep native save400/limit1. Terminal model requires source/frozen-state audit before a later stage.',
        next_stage='all252 decoder online OPD32; not started automatically.',
        CUDA_initialized=False, disk_free_bytes=shutil.disk_usage(c.OUTPUT.parent.parent.parent).free)
    cfg, hf, args = c.configuration(record)
    callback = AlignedQKCallback(record=record, source_dir=directory, run_root_dir=c.OUTPUT)
    c.require(callback.source_dir == directory and callback.run_root_dir == c.OUTPUT
              and not callback.completed_steps, 'CPU callback construction changed source/output.')
    record['CPU_callback_initialization_verified'] = True
    record['cpu_configuration'] = dict(canonical_config=c.to_runtime_jsonable(cfg),
        hf_args=c.to_runtime_jsonable(hf), training_args=c.to_runtime_jsonable(args),
        canonical_difference_from_old=['student_checkpoint_dir', 'run_root_dir'])
    c.check_inputs(record)
    c.require(not c.torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['CPU_elapsed_seconds'] = time.monotonic() - started
    c.OUTPUT.mkdir()
    c.write_json(c.OUTPUT / 'prepared.json', record)
    print(dict(status=record['status'], source_checkpoint_id=c.SOURCE_ID, inventory=inventory,
               prepared=str(c.OUTPUT / 'prepared.json'), elapsed_seconds=record['CPU_elapsed_seconds']), flush=True)


if __name__ == '__main__':
    main()
