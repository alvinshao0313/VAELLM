"""Fresh C1 Rice: new source -> original gate500 -> own live up/down S2x10K+500."""
import argparse
from dataclasses import asdict
import gc
import os
from pathlib import Path
import sys
import time
import traceback

from . import sources as s
from experiments.fresh_fixed2_cmcc_20261030 import run_pipeline as control

import torch

from experiments.adaptive_class_20261030 import continuation as flow
from train_utils.cat_runtime_state_v6 import restore_cat_runtime_state

c, checkpoint, recovery = s.c, s.checkpoint, s.recovery
ENTRY = 'experiments.fresh_rice_cmcc_20261031.run_pipeline'
# Only this process's generic fresh-check command is rebound; C0 files are immutable.
control.ENTRY = ENTRY
warm, logits, fresh = control.warm, control.logits, control.fresh


def execute(args):
    c.require(Path(sys.executable).resolve() == s.common.s.PYTHON.resolve()
        and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') and ',' not in os.environ['CUDA_VISIBLE_DEVICES']
        and not args.output.exists(), 'Use caller bitvae, assigned single GPU and a new C1 Rice output.')
    binding = s.bind(args.common)
    pins = s.source_pins()
    cat, hf, training, vae, cfgs, runtime = s.recipe(args.output, binding)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = dict(schema='fresh_C1_Rice_pipeline_v1', status='running', arm=s.ARM, phase='formal',
        pid=os.getpid(), start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        command=[sys.executable, *sys.argv], started_unix=time.time(), gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        pipeline_source_sha256=pins, source_binding=binding,
        source_after_o=s.AFTER_O_ID, source_raw2=s.RAW2_ID,
        planned_recovery_steps_per_category=500, actual_recovery_steps_per_category=500,
        planned_native_fit_updates_per_stage=10000, actual_native_fit_updates_per_stage=10000,
        additional_gate_fit_updates=0, third_stage_fit_updates=0, rice_gate_count=36, stages=[])
    c.write(args.output / 'report.json', report)
    try:
        recovery.native.configure_deterministic_mode(cat.deterministic)
        activation_runtime, plan, identity = restore_cat_runtime_state(runtime, access_token=hf.access_token)
        c.require(identity == binding['source_runtime_identity'], 'Restored fresh runtime identity changed.')
        model, report['raw_source'] = s.build_model(binding)
        # Match C0: complete model/payload construction precedes training seed31.
        recovery.native.set_seed(cat.seed)
        for index, category in enumerate(('gate_proj', 'up_proj', 'down_proj'), start=4):
            output = args.output / f'{index:02d}_{category}'
            output.mkdir()
            stage = recovery.canonical.resolve_cat_after_category_stage(cat, training, category=category, round_idx=index)
            provenance = dict(schema='fresh_C1_Rice_provenance_v1', arm=s.ARM, phase='formal',
                source_binding=binding, source_after_o=s.AFTER_O_ID, source_raw2=s.RAW2_ID,
                actual_category=category, actual_recovery_steps=500, planned_recovery_schedule=500,
                additional_gate_fit_updates=0, third_stage_fit_updates=0, rice_gate_count=36,
                rice_recipe=report['raw_source']['rice_recipe'],
                suffix_fit_stages=2, suffix_fit_updates_per_stage=10000, future_VAE_seed=31)
            meta, budget, state = checkpoint.save(model, output / 'round_base', provenance=provenance)
            del state
            row = dict(category=category, original_stage=asdict(stage), round_base_checkpoint_id=meta['checkpoint_id'],
                round_base_budget=budget, actual_started_unix=time.time())
            report['stages'].append(row)
            report['current_stage'] = category
            c.write(args.output / 'report.json', report)
            if category == 'gate_proj':
                before_cpu, before_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    checked = control.initial_parent_check(model, runtime, output / 'round_base', meta, output)
                c.require(torch.equal(before_cpu, torch.get_rng_state()) and torch.equal(before_cuda, torch.cuda.get_rng_state()),
                    'Fresh Rice finite/save/fresh readout changed training CPU/CUDA RNG.')
                checked.update(status='verified_actual_fresh_raw2_Rice_parent', CPU_CUDA_RNG_preserved=True,
                    scope='New actual source/runtime/Rice wiring only; existing Rice frozen/save/fresh/live-suffix lifecycle reused.')
                report['parent_loading_validation'] = checked
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
            strict = control.reload_subprocess(destination, evidence_path)
            c.require(strict['checkpoint_id'] == saved['checkpoint_id'], 'Fresh consumer loaded another C1 boundary.')
            row['strict_reload_report'] = str(destination.parent / 'strict_reload.json')
            c.write(args.output / 'report.json', report)
            if category == 'down_proj':
                c.require(not saved['pending_dense_modules'] and len(saved['native_modules']) == 216
                    and len(saved['compact_modules']) == 36 and measured['budget_passed'], 'Final C1 needs216/36/0 and true logical<=2.6.')
                report.update(final_checkpoint_id=saved['checkpoint_id'], final_model=str(destination), final_budget=measured,
                    final_strict_reload_report=row['strict_reload_report'])
            else:
                model, loaded_meta, loaded, _budget = checkpoint.load(destination)
                c.require(loaded_meta['checkpoint_id'] == saved['checkpoint_id']
                    and not loaded.missing_keys and not loaded.unexpected_keys, 'Fit only this Rice arm\'s own fresh live dense suffix.')
                following = 'up_proj' if category == 'gate_proj' else 'down_proj'
                activation_runtime, plan, row['next_live_category_fit'] = flow.fit_next(model, cat, hf, training, vae,
                    cfgs, activation_runtime, plan, previous=category, category=following, output=output / 'next_category_fit', short=False)
                c.write(args.output / 'report.json', report)
        c.require(s.bind(args.common) == binding, 'Actual common source changed during C1.')
        s.common.s.check_pins(pins)
        report.update(status='completed_fresh_C1_Rice_pipeline', exit_code=0)
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.monotonic() - started)
        c.write(args.output / 'report.json', report)
    print(dict(status=report['status'], elapsed_seconds=report['elapsed_seconds']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fresh', type=Path)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.fresh:
        c.require(args.evidence is not None, 'Fresh verification requires real saved evidence.')
        fresh(args.fresh, args.evidence)
    else:
        c.require(args.common is not None and args.output is not None, 'Bind this actual completed common and a fresh C1 output.')
        args.output = args.output.resolve()
        execute(args)


if __name__ == '__main__':
    main()
