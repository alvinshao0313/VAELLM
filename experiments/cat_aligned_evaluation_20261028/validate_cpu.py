"""Read-only protocol/material checks; old09 is never relabeled as aligned01 or reevaluated."""
import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import bindings as b
import evaluate as e
from experiments.longbench_v1.models import check_config
from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS


def validate(output):
    b.cpu_environment()
    import torch
    started = time.monotonic()
    for path in b.HERE.glob('*.py'):
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    # Compare the inherited partition constant directly to the real mature source.
    origin = b.R / 'qwen_b32_s3_gain_calibration_20261024/evaluation_adapter/full9.py'
    tree = ast.parse(origin.read_text(encoding='utf-8'))
    original_groups = next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'LB_GROUPS' for target in node.targets))
    b.require(e.LB_GROUPS == original_groups, 'Partition policy drifted from the mature full9 source.')
    groups = {}
    for count in range(1, 9):
        devices = e.gpu_list(','.join(map(str, range(count))))
        partitions = e.task_groups(devices)
        groups[str(count)] = [dict(tasks=tasks, count=sum(EXPECTED_COUNTS[task] for task in tasks)) for tasks in partitions]
    invalid_gpus = []
    for value in ('', '0,0', '8', '0,1,2,3,4,5,6,7,8'):
        try:
            e.gpu_list(value)
        except ValueError:
            invalid_gpus.append(value)
        else:
            raise ValueError('Invalid GPU list was accepted: ' + value)
    cli = {}
    for name in ('evaluate.py', 'predict.py', 'summarize.py'):
        result = subprocess.run([str(b.PYTHON), '-B', str(b.HERE / name), '--help'],
            env=b.helpers().environment(b.ROOT, ''), stdin=subprocess.DEVNULL, capture_output=True, text=True)
        b.require(result.returncode == 0, name + ' --help failed: ' + result.stderr)
        cli[name] = dict(exit_code=result.returncode)
    old_root = b.R / 'qwen_recompression_followup_20261026/qwen_recompression_recovery_tail_20261027'
    materials = {}
    for profile in ('C', 'G'):
        old = old_root / profile / '09_joint1200'
        audit_path = old / 'native_audit.json'
        audit = b.read(audit_path)
        checkpoint = Path(audit['checkpoint_dir'])
        meta = b.read(checkpoint / 'checkpoint_meta.json')
        config = b.read(old / 'evaluation_full9/longbench_config.json')
        check_config(config)
        b.check_hashes(config['model']['recompression']['source_sha256'])
        b.require(audit['checkpoint_id'] == meta['checkpoint_id'] == config['model']['checkpoint_id'] == b.SOURCE_IDS[profile],
            'Old09 validation material identity changed.')
        pins = b.source_pins(dict(evidence_sha256={str(audit_path): b.sha256(audit_path)}))
        b.check_hashes(pins)
        actual_meta, actual_budget = b.recovery().checkpoint(checkpoint)
        b.require(actual_meta == meta and all(actual_budget[key] == audit['bit_budget'][key] for key in b.BUDGET_FIELDS),
            'Actual material storage and the2.6/rank0 helper disagree.')
        e.full8_pins(old / 'evaluation_full9', pins)
        material = dict(checkpoint_dir=str(checkpoint), bit_budget=audit['bit_budget'])
        new_config = b.make_config(material, meta, {str(audit_path): b.sha256(audit_path)})
        with tempfile.TemporaryDirectory(prefix='aligned_eval_cpu_') as temporary:
            scratch = Path(temporary)
            config_path = scratch / 'config.json'
            b.write_json(config_path, new_config)
            plan = e.make_plan(new_config, config_path, checkpoint, scratch, ['2', '3'], {})
            b.require(plan['checkpoint_id'] == meta['checkpoint_id'] and plan['expected_samples'] == 4750
                and set(plan['task_file_sources']) == set(DATASETS)
                and all(unit['argv'][3] == str(b.HERE / 'predict.py') and unit['native_runtime_seconds'] == 86400
                        for unit in plan['execution_units'])
                and plan['nine_task_argv'][1] == str(b.HERE / 'summarize.py')
                and plan['acceptance']['max_linear_bpw'] == 2.6 and plan['acceptance']['max_lora_rank'] == 0,
                'New partition/summarize/native2.6 plan wiring changed.')
        materials[profile] = dict(checkpoint_id=meta['checkpoint_id'], existing_config=str(old / 'evaluation_full9/longbench_config.json'),
            existing_config_sha256=b.sha256(old / 'evaluation_full9/longbench_config.json'),
            logical_bpw=audit['bit_budget']['user_linear_bpw'], packed_bpw=audit['bit_budget']['packed_linear_bpw'],
            existing_config_and_recompression_pins_checked=True,
            native2_6_storage_and_full8_64_leaf_24742_artifact_pins_checked=True,
            boundary='Existing09 material only: no aligned01 claim, model load, inference, rescoring or source substitution.')
    refused = {}
    for profile in ('C', 'G'):
        path = b.DATA / profile / '01_qk_fineweb2000/native_audit.json'
        if not path.exists():
            try:
                b.source(profile, '01')
            except ValueError as error:
                refused[profile] = str(error)
            else:
                raise ValueError('Unaudited actual01 source was accepted.')
        else:
            refused[profile] = 'Actual01 audit now exists; unaudited-source rejection was not exercised.'
    formula = b.formula()
    import predict
    b.require(Path(predict.load_model.__code__.co_filename).resolve() == b.ROOT / 'experiments/cat_recompression/longbench.py'
        and predict.load_model.__name__ == 'load_model' and callable(formula.summarize) and not torch.cuda.is_initialized(),
        'Thin wrappers must retain the actual mature native loader/formula and remain on CPU.')
    result = dict(status='CPU_material_and_entrypoints_verified_not_model_evaluated',
        checked_unix=time.time(), elapsed_seconds=time.monotonic()-started, CUDA_initialized=torch.cuda.is_initialized(),
        source_sha256={str(path): b.sha256(path) for path in b.HERE.glob('*.py')},
        inherited_partition_source=str(origin), inherited_partition_source_sha256=b.sha256(origin),
        partitions_1_to_8=groups, invalid_gpu_lists_rejected=invalid_gpus, cli=cli,
        old09_material_only=materials, actual01_unready_refusal=refused,
        boundary='No GPU/model execution; real aligned01 audits remain required before evaluation.')
    output.parent.mkdir(parents=True, exist_ok=True)
    b.require(not output.exists(), 'Do not overwrite CPU verification evidence.')
    b.write_json(output, result)
    print(dict(status=result['status'], output=str(output), elapsed_seconds=result['elapsed_seconds']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=b.OUTPUT / 'cpu_validation.json')
    validate(parser.parse_args().output)
