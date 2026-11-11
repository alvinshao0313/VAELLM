"""Check actual new-machine HF, train data, parsed recipe and real save boundaries."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import time
import traceback

from bindings import ARTIFACTS, FRESH, HF, original_cpu, recipe, s, source_pins, validate_raw2_meta


def execute(output):
    import sys
    import torch
    s.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Hide CUDA for CPU validation.')
    s.require(Path(sys.executable).resolve() == s.PYTHON.resolve() and not output.exists(), 'Use caller bitvae and a new evidence directory.')
    output.mkdir(parents=True)
    report = dict(status='running', started_unix=time.time(), command=[sys.executable, *sys.argv])
    s.write(output / 'validation.json', report)
    try:
        report['hf_identity'] = s.hf_identity(HF)
        report['data_identity'] = original_cpu.calibration_and_data_identity(HF)
        plan = recipe.plan('B', HF, output / 'not_executed')
        original = s.read(ARTIFACTS / 'cat_structure_full_20261029/prepared.json')['arms']['B']
        parsed = recipe.parse('B', original['native_argv'])
        s.require({name: asdict(cfg) for name, cfg in parsed[4].items()} == plan['category_configs']
            and {name: asdict(parsed[0].resolve_after_category_config(name)) for name in s.CATEGORIES} == plan['recovery_configs'],
            'All seven resolved specs and recovery settings must match the original actual B recipe.')
        report['plan'] = plan
        after_o_meta = ARTIFACTS / 'qwen_recompression_20261026/C/initial_after_o/checkpoint_meta.json'
        raw_meta = ARTIFACTS / 'adaptive_native_model_20261029/cat_gate_prefixes/B/raw_checkpoint_meta.json'
        s.validate_after_o_meta(s.read(after_o_meta), '/home/star/models/Qwen/Qwen3-8B')
        validate_raw2_meta(s.read(raw_meta), '/home/star/models/Qwen/Qwen3-8B')
        report['real_metadata_evidence'] = {str(path): s.sha(path) for path in (after_o_meta, raw_meta)}
        original_cpu.EVENT_META = after_o_meta
        import pytest
        code = pytest.main([str(FRESH / 'test_contract.py'), '-q', '-p', 'no:cacheprovider',
            '-k', 'not seven_resolved_categories_original_recovery_and_exposure'])
        s.require(code == 0 and not torch.cuda.is_initialized(), 'Existing applicable CPU contracts failed.')
        report.update(status='cpu_verified_gpu_pending', exit_code=0, source_sha256=source_pins(),
            applicable_existing_contract_cases=14, real_B_resolved_config_comparison=True,
            original_template_not_migrated=True, historical_16case_evidence='See unchanged fresh README; not claimed rerun.',
            cuda_initialized=torch.cuda.is_initialized())
    except BaseException as error:
        report.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.time() - report['started_unix']
        s.write(output / 'validation.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output.resolve())
