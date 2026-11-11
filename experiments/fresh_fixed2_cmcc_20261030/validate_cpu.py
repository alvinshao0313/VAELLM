"""Validate the actual formal entry imports, raw2 recipe and runtime without GPU."""
import argparse
import ast
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Cover the same helper imports and ordering as the real formal process.
from . import run_pipeline as pipeline

s = pipeline.s

import torch

from experiments.cmcc_runtime_20261030.paths import ARTIFACTS


def execute(output):
    s.c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
        and Path(sys.executable).resolve() == s.common.s.PYTHON.resolve() and not output.exists(),
        'Use caller bitvae, hidden CUDA and a new CPU evidence directory.')
    output.mkdir(parents=True)
    record = dict(status='running', command=[sys.executable, *sys.argv], started_unix=time.time())
    try:
        common = ARTIFACTS / 'fresh_common_cmcc_20261030'
        binding = s.bind(common / 'formal')
        cat, hf, training, vae, cfgs, runtime = s.recipe(output / 'not_executed', binding)
        activation, _plan, restored_identity = pipeline.restore_cat_runtime_state(runtime, access_token=hf.access_token)
        cache = activation['cache']
        s.c.require(restored_identity == binding['source_runtime_identity']
            and len(cache.input_ids) == cache.nsamples == 512 and cache.seqlen == 256
            and all(tuple(value.shape) in ((256,), (1, 256)) and value.dtype == torch.int64
                and value.device.type == 'cpu' for value in cache.input_ids)
            and Path(cache.model_path).resolve() == s.HF.resolve(),
            'Restore the actual formal parent identity and complete original512x256 CPU calibration.')
        previous = s.c.read(s.ROOT / 'result/qwen3_8b_mmlu65_20261004/fresh_common_cmcc_20261030/cpu/validation.json')
        identity = restored_identity
        s.c.require(previous['status'] == 'cpu_verified_gpu_pending' and previous['exit_code'] == 0
            and json.loads(json.dumps({name: asdict(cfg) for name, cfg in cfgs.items()})) == previous['plan']['category_configs']
            and json.loads(json.dumps({name: asdict(cat.resolve_after_category_config(name)) for name in s.recovery.CATEGORIES}))
                == previous['plan']['recovery_configs']
            and identity == previous['plan']['runtime_identity'], 'C0 must reuse all seven actual fresh B category/recovery settings.')
        request = s.c.read(common / 'formal/request.json')
        s.c.require(identity['experiment_runtime'] == request['effective_contract']
            and cat.seed == training.seed == 31 and vae.model_path == str(s.HF), 'Actual formal common recipe differs.')
        rejected = {}
        for name in ('short_v2',):
            try:
                s.bind(common / name)
            except ValueError as error:
                rejected[name] = str(error)
            else:
                raise ValueError('An ineligible actual source was accepted: ' + name)
        preserved = s.c.read(common / 'short_v2/run.json')
        raw = Path(preserved['gate_raw2']['path'])
        after_o = Path(preserved['after_o']['path'])
        s.common.validate_raw2_meta(s.c.read(raw / 'checkpoint_meta.json'), s.HF)
        s.common.s.validate_after_o_meta(s.c.read(after_o / 'checkpoint_meta.json'), s.HF)
        rows = []
        for category, shuffle in (('up_proj', 500031), ('down_proj', 600031)):
            cfg = cfgs[category]
            s.c.require(cfg.residual_stages == 2 and cfg.steps == 10000 and cfg.codebook_dim == 32
                and identity['experiment_runtime']['category_shuffle_seeds'][category] == shuffle
                and identity['experiment_runtime']['category_vector_batches'][category] == 8192,
                'The own live suffix must keep its exact two-stage10K/batch/shuffle contract.')
            rows.append(dict(category=category, stages=2, updates_per_stage=10000, vector_batch=8192, shuffle=shuffle, seed=31))
        syntax = []
        for path in s.HERE.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            syntax.append(str(path))
        s.common.s.check_pins(request['source_sha256'])
        s.common.s.enable_snapshot()
        namespace_paths = {name: list(sys.modules[name].__path__)
            for name in ('train_utils', 'litebsq', 'rotation', 'e2e_common')}
        s.c.require(all(paths == [str(s.common.s.SNAPSHOT / name)]
            for name, paths in namespace_paths.items()), 'Keep every selected namespace closed to N after the real formal helper imports.')
        module_paths = {name: str(Path(module.__file__).resolve()) for name, module in list(sys.modules.items())
            if name.split('.')[0] in namespace_paths and getattr(module, '__file__', None)}
        s.c.require(all(Path(path).is_relative_to(s.common.s.SNAPSHOT) for path in module_paths.values()),
            'The actual formal entry imported a project module outside N.')
        record.update(status='verified_actual_formal_entry_recipe_runtime_GPU_parent_pending', exit_code=0,
            real_common_request=str(common / 'formal/request.json'), common_driver=s.c.read(common / 'formal/run.json')['pid'],
            seven_category_configs_equal=True, seven_recovery_configs_equal=True, runtime_identity_equal=True,
            source_binding=binding, formal_entry=str(Path(pipeline.__file__).resolve()),
            formal_entry_helper_imports_all_N=True, namespace_paths=namespace_paths,
            actual_module_sha256={path: s.c.sha(path) for path in sorted(set(module_paths.values()))},
            residual_protection_module=module_paths['train_utils.cat_train_residual_protection'],
            formal_runtime_restored=True, calibration_samples=512, calibration_tokens_per_sample=256,
            extra_training_updates=0,
            actual_ineligible_sources_rejected=rejected, suffix_contract=rows, syntax_checked=syntax,
            preserved_short_metadata_only=True, short_weights_not_loaded=True,
            source_sha256=s.source_pins(), cuda_initialized=torch.cuda.is_initialized(),
            boundary='Real formal raw2 binding, complete formal-entry imports, unchanged seven-category recipe and restored512x256 runtime only. No model weights loaded or copied; the new parent GPU finite/save/fresh/RNG check runs in the formal entry before its first recovery update.')
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds'] = time.time() - record['started_unix']
        s.c.write(output / 'validation.json', record)
    print(dict(status=record['status'], output=str(output)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output.resolve())
