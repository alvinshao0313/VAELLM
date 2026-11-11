"""Actual common/runtime/entry CPU evidence; new Rice GPU checks remain pending."""
import argparse
import ast
from dataclasses import asdict
import os
from pathlib import Path
import sys
import time
import traceback

# Import the complete real formal entry before testing the selected namespace.
from . import run_pipeline as pipeline
from . import evaluate, sequence

import torch

s, c = pipeline.s, pipeline.c


def execute(output):
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
        and Path(sys.executable).resolve() == s.common.s.PYTHON.resolve() and not output.exists(),
        'Use caller bitvae, hidden CUDA and a new CPU evidence directory.')
    output.mkdir(parents=True)
    record = dict(status='running', command=[sys.executable, *sys.argv], started_unix=time.time())
    try:
        binding = s.bind(s.COMMON)
        cat, hf, training, vae, cfgs, runtime = s.recipe(output / 'not_executed', binding)
        activation, _plan, identity = pipeline.restore_cat_runtime_state(runtime, access_token=hf.access_token)
        cache = activation['cache']
        c.require(identity == binding['source_runtime_identity'] and len(cache.input_ids) == cache.nsamples == 512
            and cache.seqlen == 256 and all(tuple(value.shape) in ((256,), (1, 256)) and value.dtype == torch.int64
                and value.device.type == 'cpu' for value in cache.input_ids)
            and Path(cache.model_path).resolve() == s.HF.resolve(), 'Restore the complete real512x256 formal runtime.')
        previous = c.read(s.ROOT / 'result/qwen3_8b_mmlu65_20261004/fresh_common_cmcc_20261030/cpu/validation.json')
        import json
        normalized = lambda value: json.loads(json.dumps(value))
        c.require(previous['status'] == 'cpu_verified_gpu_pending' and previous['exit_code'] == 0
            and normalized({name: asdict(cfg) for name, cfg in cfgs.items()}) == previous['plan']['category_configs']
            and normalized({name: asdict(cat.resolve_after_category_config(name)) for name in s.recovery.CATEGORIES})
                == previous['plan']['recovery_configs'] and identity == previous['plan']['runtime_identity'],
            'Fresh Rice must reuse the real common/C0 seven-category recipe and runtime.')
        raw_meta = c.read(Path(binding['source_raw2']['path']) / 'checkpoint_meta.json')
        gates = [item for item in raw_meta['converted_modules'] if item['name'].endswith('.gate_proj')]
        c.require({item['name'] for item in gates} == {f'model.layers.{i}.mlp.gate_proj' for i in range(36)}
            and all(item['residual_stages'] == 2 and item['stage_codebook_dims'] == [64, 64]
                and item['protected_channel_quant_format'] == 'int8' and item['weight_rotation'] for item in gates),
            'Real raw2 must contain the complete36-layer two-stage INT8/rotation gate recipe.')
        evaluate.setup(sequence.SUPPORT)
        inventory = evaluate.b.model_inventory(s.HF)
        c.require(evaluate.b.BASE.resolve(strict=True) == s.HF.resolve(strict=True)
            and len(inventory['projection_shapes']) == 252
            and all(inventory['projection_shapes'][item['name']] == [item['out_features'], item['in_features']]
                for item in raw_meta['converted_modules']),
            'The positive cmcc reader must see the real local HF252 projection inventory.')
        dispatch = (evaluate.b.source, evaluate.b.source_pins, evaluate.b.command)
        reader_binding = evaluate.configure_readers()
        c.require(dispatch == (evaluate.b.source, evaluate.b.source_pins, evaluate.b.command)
            and evaluate.b.source is evaluate.source
            and evaluate.b.command('predict', '--config', 'not_executed.json')[4]
                == 'experiments.fresh_rice_cmcc_20261031.evaluate',
            'Positive repeated source-reader initialization reset the C1 source/pins/command dispatch.')
        suffix = []
        for category, shuffle in (('up_proj', 500031), ('down_proj', 600031)):
            cfg = cfgs[category]
            c.require(cfg.residual_stages == 2 and cfg.steps == 10000 and cfg.codebook_dim == 32
                and identity['experiment_runtime']['category_shuffle_seeds'][category] == shuffle
                and identity['experiment_runtime']['category_vector_batches'][category] == 8192,
                'Use the own-live S2x10K/batch8192/seed31/shuffle contract.')
            suffix.append(dict(category=category, stages=2, updates_per_stage=10000, vector_batch=8192, shuffle=shuffle, seed=31))
        rejected_sources = {}
        for label, path in (('actual_common_short_v2', s.COMMON.parent / 'short_v2'),
            ('old_archive_source', s.ROOT / 'experiments/adaptive_class_20261030'),
            ('different_incomplete_common_path', output / 'not_a_completed_common')):
            try:
                s.bind(path)
            except (ValueError, FileNotFoundError) as error:
                rejected_sources[label] = dict(path=str(path), error=str(error))
            else:
                raise ValueError('Ineligible source accepted: ' + label)
        # Negative headers exercise the real acceptance function without inventing
        # a successful GPU report, payload, checkpoint or dependency.
        invalid_headers = (
            dict(schema='fresh_C1_Rice_pipeline_v1', arm=s.ARM, status='running', phase='formal'),
            dict(schema='fresh_C1_Rice_pipeline_v1', arm=s.ARM, status='completed_fresh_C1_Rice_pipeline', exit_code=0, phase='short'),
            dict(schema='fresh_C0_fixed2_pipeline_v1', arm='C0_fixed2', status='completed_fresh_C0_fixed2_pipeline', exit_code=0, phase='formal'),
        )
        rejected_headers = []
        for header in invalid_headers:
            try:
                evaluate.validate_report(header)
            except ValueError as error:
                rejected_headers.append(dict(header=header, error=str(error)))
            else:
                raise ValueError('Invalid final-report header accepted.')
        # Call the actual public source entry on the real common report, which is
        # a valid shared source but never a completed C1 endpoint.
        try:
            evaluate.b.source(s.COMMON, s.COMMON / 'run.json')
        except ValueError as error:
            source_endpoint_rejection = str(error)
        else:
            raise ValueError('A common prerequisite masqueraded as the C1 final endpoint.')
        c.require(dispatch == (evaluate.b.source, evaluate.b.source_pins, evaluate.b.command),
            'The public source call reset the installed evaluation dispatch.')
        namespaces = {name: list(sys.modules[name].__path__) for name in ('train_utils', 'litebsq', 'rotation', 'e2e_common')}
        c.require(all(paths == [str(s.common.s.SNAPSHOT / name)] for name, paths in namespaces.items()),
            'Keep every selected namespace closed to N after actual pipeline/evaluation/sequence imports.')
        modules = {name: str(Path(module.__file__).resolve()) for name, module in list(sys.modules.items())
            if name.split('.')[0] in namespaces and getattr(module, '__file__', None)}
        c.require(all(Path(path).is_relative_to(s.common.s.SNAPSHOT) for path in modules.values()),
            'The actual C1 entry imported a runtime module outside N.')
        c.require(pipeline.control.ENTRY == pipeline.ENTRY
            and Path(s.sparse.__file__).resolve() == s.ROOT / 'experiments/adaptive_class_20261030/sparse.py',
            'Use the verified Rice consumer and rebind fresh subprocess entry only in this new process.')
        syntax = []
        for path in s.HERE.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            syntax.append(str(path))
        evidence = c.read(output.parent / 'protected_pins_before.json')
        protected = {}
        for path, expected in evidence.items():
            c.require(c.sha(path) == expected['plan_sha256'], 'An active plan changed during isolated C1 work.')
            s.common.s.check_pins(expected['source_sha256'])
            protected[path] = dict(source_count=expected['source_count'], all_exact=True, plan_sha256=expected['plan_sha256'])
        record.update(status='verified_actual_fresh_Rice_source_recipe_runtime_GPU_pending', exit_code=0,
            source_binding=binding, seven_category_configs_equal=True, seven_recovery_configs_equal=True,
            runtime_identity_equal=True, real_gate_specs=len(gates), formal_runtime_restored=True,
            local_HF_projection_count=252, evaluation_BASE=str(evaluate.b.BASE),
            positive_local_reader_binding=reader_binding, source_call_preserves_dispatch=True,
            actual_common_not_C1_endpoint_rejected=source_endpoint_rejection,
            calibration_samples=512, calibration_tokens_per_sample=256, extra_training_updates=0,
            suffix_contract=suffix, formal_entry=str(Path(pipeline.__file__).resolve()),
            generic_fresh_subprocess_entry=pipeline.control.ENTRY, namespace_paths=namespaces,
            formal_entry_helper_imports_all_N=True,
            actual_module_sha256={path: c.sha(path) for path in sorted(set(modules.values()))},
            actual_ineligible_sources_rejected=rejected_sources, invalid_report_headers_rejected=rejected_headers,
            syntax_checked=syntax, protected_active_source_pins=protected, source_sha256=s.source_pins(),
            cuda_initialized=torch.cuda.is_initialized(), real_model_weights_loaded=False,
            boundary='CPU verifies actual new common/36gate metadata, full formal entry namespaces and original recipe/runtime only. Rice36 construction and zero-update BF16 finite/save/independent-fresh/CPU-CUDA RNG check must pass in the formal GPU entry before any original500 recovery; no GPU pass claimed here.')
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds'] = time.time() - record['started_unix']
        c.write(output / 'validation.json', record)
    print(dict(status=record['status'], output=str(output)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output.resolve())
