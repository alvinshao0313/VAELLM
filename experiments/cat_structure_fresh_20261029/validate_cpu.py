"""Actual HF/config/data evidence, with no model construction or GPU work."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import shlex
import sys
import time
import traceback

import fresh_support as s
import recipe

# Offline comparison evidence only; neither path is imported by the runtime.
TEMPLATE = s.SNAPSHOT.parent / 'formal_k38_gate_s4_seq500/run.sh'
EVENT_META = Path('/data/star/VAELLM/qwen_recompression_20261026/C/initial_after_o/checkpoint_meta.json')


def template_argv():
    tokens = shlex.split(TEMPLATE.read_text(encoding='utf-8').replace('\\\n', ''), comments=True)
    return tokens[tokens.index('tools/cat_train.py') + 1:]


def option_map(values):
    result = {}
    index = 0
    while index < len(values):
        key = values[index]
        s.require(key.startswith('--') and key not in result, 'Invalid or repeated template option: ' + key)
        if index + 1 < len(values) and not values[index + 1].startswith('--'):
            result[key] = values[index + 1]
            index += 2
        else:
            result[key] = True
            index += 1
    return result


def compare_template(arm, plan):
    original = template_argv()
    before, after = option_map(original), option_map(plan['native_argv'])
    changes = {key: dict(template=before.get(key), fresh=after.get(key))
               for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    allowed = {'--model_path', '--output_dir', '--codebook_bits', '--codebook_dim', '--residual_stages',
               '--decoder_type', '--decoder_base_ch', '--decoder_num_res_blocks'}
    s.require(set(changes) <= allowed, 'Unexpected difference from the original fresh N recipe: ' + str(changes))
    original_parsed = recipe.parse(arm, original)
    original_cfgs = original_parsed[4]
    for name in s.CATEGORIES:
        if name != 'gate_proj':
            s.require(asdict(original_cfgs[name]) == plan['category_configs'][name], 'Changed non-gate spec: ' + name)
        s.require(asdict(original_parsed[0].resolve_after_category_config(name)) == plan['recovery_configs'][name],
                  'Changed original recovery/data scope: ' + name)
    return dict(source=str(TEMPLATE), source_sha256=s.sha(TEMPLATE), actual_cli_changes=changes,
                unchanged='All non-gate resolved specs and all seven recovery/data configurations exactly match.')


def calibration_and_data_identity(base):
    s.enable_snapshot()
    from e2e_common import data, lazy_datasets
    from train_utils.activation_utils import _build_calibration_cache
    observed = []
    original = lazy_datasets._load_preset_raw_datasets

    def observe(preset):
        train, evaluation = original(preset)
        observed.append(dict(preset=asdict(preset), rows=len(train), fingerprint=train._fingerprint,
            license_in_dataset_info=train.info.license,
            cache_files=[s.stat(item['filename']) for item in train.cache_files]))
        return train, evaluation

    lazy_datasets._load_preset_raw_datasets = observe
    try:
        cache = _build_calibration_cache(dataset='alpaca=1', model_path=str(base), access_token=None,
                                         nsamples=512, seqlen=256, seed=31)
    finally:
        lazy_datasets._load_preset_raw_datasets = original
    s.require(len(observed) == 1 and observed[0]['preset']['alias'] == 'alpaca'
              and observed[0]['preset']['train_split'] == 'train'
              and len(cache.input_ids) == 512 and all(tuple(tokens.shape) == (1, 256) for tokens in cache.input_ids),
              'Actual original Alpaca train512×256 calibration token generation failed.')
    digest = hashlib.sha256()
    for tokens in cache.input_ids:
        digest.update(tokens.contiguous().numpy().tobytes())
    sources, weights, normalized = data.normalize_dataset_mix_spec(recipe.MIX)
    recovery_sources = []
    for alias, weight in zip(sources, weights):
        preset = data.DATASET_MIX_SOURCE_PRESETS[alias]
        s.require(preset.train_split == 'train' and preset.eval_split is None, 'Recovery must use existing train-only sources.')
        recovery_sources.append(dict(preset=asdict(preset), probability=weight,
                                     actual_file=s.stat(data._resolve_dataset_path(preset.path))))
    manifest_path = s.ROOT / 'result/qwen3_8b_w25_20260930/data/task_manifest.json'
    manifest = s.read(manifest_path)
    return dict(calibration=dict(actual_native_token_generation=True, model_forward=False,
        dataset=cache.dataset, samples=512, seqlen=256, seed=31, token_count=131072,
        input_ids_sha256=digest.hexdigest(), actual_source=observed[0]),
        recovery=dict(mix=normalized, sources=recovery_sources, manifest_path=str(manifest_path),
            manifest_sha256=s.sha(manifest_path), existing_task_manifest=manifest,
            verification='Actual N alias resolution/file stat; reuse existing provenance. No repeated full 22GB data digest or recovery sampling.'))


def execute(base_model_path, output_dir):
    import os
    s.require(Path(sys.executable).resolve() == s.PYTHON.resolve()
              and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU validation requires caller bitvae and CUDA_VISIBLE_DEVICES empty.')
    output = Path(output_dir).resolve()
    s.require(not output.exists(), 'Never overwrite CPU evidence.')
    output.mkdir(parents=True)
    report_path = output / 'validation.json'
    report = dict(status='running', started_unix=time.time(), command=[sys.executable, *sys.argv],
                  scope='CPU only. No fresh prefix/save or suffix execution, model construction, GPU forward, or quality claim.')
    s.write(report_path, report)
    forbidden = []

    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0]))
            if path.name in {'pytorch_model.bin', 'cat_runtime_state.pt', 'training_model_state.pt'}:
                forbidden.append(str(path))
                raise ValueError('CPU proof must not read historical model/runtime tensors: ' + str(path))

    sys.addaudithook(audit)
    try:
        report['hf_identity'] = s.hf_identity(base_model_path)
        plans = {arm: recipe.plan(arm, base_model_path, output / ('planned_' + arm)) for arm in ('A', 'B')}
        report['plans'] = plans
        report['template_comparison'] = {arm: compare_template(arm, plan) for arm, plan in plans.items()}
        report['data_identity'] = calibration_and_data_identity(base_model_path)
        s.validate_after_o_meta(s.read(EVENT_META), base_model_path)
        report['existing_save_event_evidence'] = dict(path=str(EVENT_META), sha256=s.sha(EVENT_META),
            proof='Read real prior metadata solely to check boundary predicate; not a fresh save or a runtime input.')
        import pytest

        class Results:
            def __init__(self):
                self.passed = self.failed = 0

            def pytest_runtest_logreport(self, report):
                if report.when == 'call':
                    self.passed += int(report.passed)
                    self.failed += int(report.failed)

        results = Results()
        code = pytest.main([str(s.HERE / 'test_contract.py'), '-q', '-p', 'no:cacheprovider'], plugins=[results])
        s.require(code == 0, 'Fresh protocol/recipe CPU tests failed.')
        import torch
        s.require(not torch.cuda.is_initialized() and not forbidden, 'CPU validation unexpectedly touched CUDA/history weights.')
        report.update(status='cpu_verified_gpu_unvalidated', exit_code=0,
            tests=dict(passed=results.passed, failed=results.failed), source_sha256=s.source_pins(),
            cuda_initialized=torch.cuda.is_initialized(), forbidden_history_weight_opens=forbidden,
            pending='Actual new BF16 prefix calibration-stat forward/save, two-process suffix reload, native updates/save and final storage/full9 remain unvalidated.')
    except BaseException as error:
        report.update(status='failed', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.time() - report['started_unix']
        s.write(report_path, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_model_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    arguments = parser.parse_args()
    execute(arguments.base_model_path.resolve(strict=True), arguments.output_dir)
