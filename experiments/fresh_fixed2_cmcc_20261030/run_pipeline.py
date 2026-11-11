"""C0 fixed2 gate500 -> own live up/down S2x10K+500, from a fresh raw2."""
import argparse
from dataclasses import asdict
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import sources as s

import torch

from experiments.adaptive_class_20261030 import continuation as flow
from experiments.adaptive_class_20261030.run_pipeline import fresh, logits, warm
from experiments.adaptive_native_model_20261029 import checkpoint, recovery
from experiments.adaptive_vector_rvq import common as c
from train_utils.cat_runtime_state_v6 import restore_cat_runtime_state

ENTRY = 'experiments.fresh_fixed2_cmcc_20261030.run_pipeline'


def reload_subprocess(directory, evidence):
    subprocess.run([sys.executable, '-B', '-u', '-m', ENTRY,
        '--fresh', str(directory), '--evidence', str(evidence)], cwd=s.ROOT, check=True)
    result = c.read(directory.parent / 'strict_reload.json')
    c.require(result['status'] == 'strict_full_state_and_BF16_logits_verified'
        and result['exit_code'] == 0 and result['same_input_logits_exact'], 'The independent fresh consumer did not verify.')
    return result


def initial_parent_check(model, runtime, round_base, meta, output):
    """Cover the new real parent loader/recipe/export wiring before any recovery."""
    cache = runtime['activation_runtime']['cache']['input_ids']
    c.require(len(cache) == 512 and tuple(cache[0].shape) in ((256,), (1, 256)), 'Use this parent\'s real512x256 calibration input.')
    inputs = dict(input_ids=cache[0].reshape(1, 256).clone())
    prepared = warm(model, 144)
    expected = logits(model, inputs)
    payload = torch.load(round_base / checkpoint.STATE_FILE, map_location='cpu', weights_only=True, mmap=True)
    digest = c.digest(payload)
    c.require(c.digest(model.state_dict()) == digest, 'Initial deployed fields differ from the actual fixed2 round_base save.')
    evidence = output / 'raw2_same_input.pt'
    torch.save(dict(checkpoint_id=meta['checkpoint_id'], deployment_digest=digest, inputs=inputs, logits=expected), evidence)
    model.to('cpu')
    for module in model.modules():
        clear = getattr(module, 'clear_decoded_weight_cache', None)
        if callable(clear):
            clear()
    del payload, expected
    gc.collect()
    torch.cuda.empty_cache()
    strict = reload_subprocess(round_base, evidence)
    strict_path = output / 'raw2_strict_reload.json'
    (output / 'strict_reload.json').rename(strict_path)
    return dict(status='verified_actual_fresh_raw2_fixed2_parent', checkpoint_id=meta['checkpoint_id'],
        native_modules=144, compact_modules=36, pending_dense_modules=72, full_state_and_same_input_BF16_logits_exact=True,
        actual_input_tokens=256, real_parent_runtime_restored=True, strict_reload_report=str(strict_path),
        deployment_digest=digest, warmup=prepared, extra_training_updates=0,
        scope='Only the new parent/source/runtime/ordinary fixed2 loading gap; original recovery/live fitter evidence reused.')


def execute(args):
    c.require(Path(sys.executable).resolve() == s.common.s.PYTHON.resolve()
        and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') and ',' not in os.environ['CUDA_VISIBLE_DEVICES']
        and not args.output.exists(), 'Use caller bitvae, the assigned single GPU and a fresh C0 output.')
    binding = s.bind(args.common)
    pins = s.source_pins()
    cat, hf, training, vae, cfgs, runtime = s.recipe(args.output, binding)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = dict(schema='fresh_C0_fixed2_pipeline_v1', status='running', arm='C0_fixed2', phase='formal',
        pid=os.getpid(), start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        command=[sys.executable, *sys.argv], started_unix=time.time(), gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        pipeline_source_sha256=pins, source_binding=binding,
        source_after_o=binding['source_after_o']['checkpoint_id'], source_raw2=binding['source_raw2']['checkpoint_id'],
        planned_recovery_steps_per_category=500, actual_recovery_steps_per_category=500,
        planned_native_fit_updates_per_stage=10000, actual_native_fit_updates_per_stage=10000,
        additional_gate_fit_updates=0, third_stage_fit_updates=0, rice_sparse_fields=0, stages=[])
    c.write(args.output / 'report.json', report)
    try:
        recovery.native.configure_deterministic_mode(cat.deterministic)
        activation_runtime, plan, identity = restore_cat_runtime_state(runtime, access_token=hf.access_token)
        c.require(identity == binding['source_runtime_identity'], 'Restoring the new parent runtime changed identity.')
        model, report['raw_source'] = s.build_model(binding)
        # Match the comparison pipeline: model reconstruction precedes seeding.
        recovery.native.set_seed(cat.seed)
        for index, category in enumerate(('gate_proj', 'up_proj', 'down_proj'), start=4):
            output = args.output / f'{index:02d}_{category}'
            output.mkdir()
            stage = recovery.canonical.resolve_cat_after_category_stage(cat, training, category=category, round_idx=index)
            provenance = dict(schema='fresh_C0_fixed2_provenance_v1', arm='C0_fixed2', phase='formal',
                source_binding=binding, source_after_o=report['source_after_o'], source_raw2=report['source_raw2'],
                actual_category=category, actual_recovery_steps=500, planned_recovery_schedule=500,
                additional_gate_fit_updates=0, third_stage_fit_updates=0, rice_sparse_fields=0,
                suffix_fit_stages=2, suffix_fit_updates_per_stage=10000, future_VAE_seed=31)
            meta, budget, state = checkpoint.save(model, output / 'round_base', provenance=provenance)
            del state
            row = dict(category=category, original_stage=asdict(stage), round_base_checkpoint_id=meta['checkpoint_id'],
                round_base_budget=budget, actual_started_unix=time.time())
            report['stages'].append(row)
            report['current_stage'] = category
            c.write(args.output / 'report.json', report)
            if category == 'gate_proj':
                # Readout/export reconstruction must not change training initialization.
                before_cpu_rng = torch.get_rng_state()
                before_cuda_rng = torch.cuda.get_rng_state()
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    report['parent_loading_validation'] = initial_parent_check(model, runtime, output / 'round_base', meta, output)
                c.require(torch.equal(before_cpu_rng, torch.get_rng_state())
                    and torch.equal(before_cuda_rng, torch.cuda.get_rng_state()),
                    'Readout changed the training CPU/CUDA random state.')
                report['parent_loading_validation']['CPU_CUDA_RNG_preserved'] = True
                c.write(args.output / 'report.json', report)
            model, evidence, probe = flow.recover(model, stage, vae, hf, output, output / 'round_base', meta, short=False)
            row['recovery'] = evidence
            row['deployment_warmup'] = warm(model, len(checkpoint.inventories(model)['native']))
            expected = logits(model, probe)
            destination = args.output / 'final_model' if category == 'down_proj' else output / 'after_recovery'
            saved, measured, payload = checkpoint.save(model, destination, provenance=provenance)
            row.update(after_recovery_checkpoint_id=saved['checkpoint_id'], after_recovery_budget=measured,
                deployment_digest=c.digest(payload), deployment_path=str(destination))
            evidence_path = output / 'same_input_deployment.pt'
            torch.save(dict(checkpoint_id=saved['checkpoint_id'], deployment_digest=row['deployment_digest'],
                inputs=probe, logits=expected), evidence_path)
            del model, payload, probe, expected
            gc.collect()
            torch.cuda.empty_cache()
            strict = reload_subprocess(destination, evidence_path)
            c.require(strict['checkpoint_id'] == saved['checkpoint_id'], 'Fresh check loaded another C0 boundary.')
            row['strict_reload_report'] = str(destination.parent / 'strict_reload.json')
            c.write(args.output / 'report.json', report)
            if category == 'down_proj':
                c.require(not saved['pending_dense_modules'] and len(saved['native_modules']) == 216
                    and len(saved['compact_modules']) == 36 and measured['budget_passed'], 'Final C0 must satisfy216/36/0 and real<=2.6.')
                report.update(final_checkpoint_id=saved['checkpoint_id'], final_model=str(destination), final_budget=measured,
                    final_strict_reload_report=row['strict_reload_report'])
            else:
                model, loaded_meta, loaded, _budget = checkpoint.load(destination)
                c.require(loaded_meta['checkpoint_id'] == saved['checkpoint_id']
                    and not loaded.missing_keys and not loaded.unexpected_keys, 'Fit only this C0 arm\'s own fresh merged dense state.')
                following = 'up_proj' if category == 'gate_proj' else 'down_proj'
                activation_runtime, plan, row['next_live_category_fit'] = flow.fit_next(model, cat, hf, training, vae,
                    cfgs, activation_runtime, plan, previous=category, category=following, output=output / 'next_category_fit', short=False)
                c.write(args.output / 'report.json', report)
        common_binding = s.bind(args.common)
        c.require(common_binding == binding, 'The common fresh source changed during C0 execution.')
        s.common.s.check_pins(pins)
        report.update(status='completed_fresh_C0_fixed2_pipeline', exit_code=0)
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.monotonic() - started)
        c.write(args.output / 'report.json', report)
    print(dict(status=report['status'], elapsed_seconds=report['elapsed_seconds']), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fresh', type=Path)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.fresh:
        c.require(args.evidence is not None, 'Fresh verification requires actual saved input/output evidence.')
        fresh(args.fresh, args.evidence)
    else:
        c.require(args.common is not None and args.output is not None, 'Bind the actual common prerequisite and a new C0 output.')
        args.output = args.output.resolve()
        execute(args)


if __name__ == '__main__':
    main()
