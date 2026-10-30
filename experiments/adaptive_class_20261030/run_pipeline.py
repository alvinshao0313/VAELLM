"""Recover each actual arm, then fit/recover its own live up/down sequentially."""
import argparse
from dataclasses import asdict
import gc
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from experiments.adaptive_native_model_20261029 import checkpoint, recovery
from experiments.adaptive_vector_rvq import common as c
from tools.cat_eval import _prepare_model_for_eval
from train_utils.cat_runtime_state_v6 import restore_cat_runtime_state
from . import continuation as flow, sources as s


def warm(model, count):
    result = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    c.require(result['total'] == result['warmed'] == count and not result['failed'], 'Warm the actual native inventory before deployed comparison.')
    return result


@torch.no_grad()
def logits(model, inputs):
    result = model(**{name: value.to('cuda:0') for name, value in inputs.items()}, use_cache=False, logits_to_keep=1).logits
    c.require(result.dtype == torch.bfloat16 and torch.isfinite(result).all().item(), 'Use finite actual BF16 deployed logits.')
    return result.detach().cpu().contiguous()


def fresh(directory, evidence_path):
    expected = torch.load(evidence_path, map_location='cpu', weights_only=True)
    model, meta, loaded, budget = checkpoint.load(directory)
    c.require(not loaded.missing_keys and not loaded.unexpected_keys and meta['checkpoint_id'] == expected['checkpoint_id']
        and c.digest(model.state_dict()) == expected['deployment_digest'], 'Independent full mixed strict reload changed actual fields.')
    prepared = warm(model, len(meta['native_modules']))
    actual = logits(model, expected['inputs'])
    c.require(torch.equal(actual, expected['logits']), 'Independent new consumer deployment changed same-input BF16 logits.')
    c.write(directory.parent / 'strict_reload.json', dict(status='strict_full_state_and_BF16_logits_verified', exit_code=0,
        pid=os.getpid(), checkpoint_id=meta['checkpoint_id'], persistent_fields=len(model.state_dict()),
        deployment_digest=expected['deployment_digest'], same_input_logits_exact=True,
        actual_budget=budget, warmup=prepared, effective_weight_files=[str(directory / checkpoint.STATE_FILE)],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def execute(args):
    c.require(not args.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')
        and ',' not in os.environ['CUDA_VISIBLE_DEVICES'], 'Use the assigned single GPU and a fresh branch output.')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    pins = {**s.code_identity(), **{str(path.resolve()): c.sha(path) for path in
        (Path(__file__), Path(flow.__file__), Path(recovery.__file__), Path(recovery.__file__).with_name('training_checkpoint.py'))}}
    report = dict(status='running', arm=args.arm, phase='short' if args.short else 'formal', pid=os.getpid(),
        command=[sys.executable, *sys.argv], started_unix=time.time(), gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        pipeline_source_sha256=pins, source_raw_manifest=str(s.ARCHIVE / 'manifest.json'),
        source_raw_manifest_sha256=c.sha(s.ARCHIVE / 'manifest.json'), source_after_o=s.BASE_ID,
        artifact_source_reports=[dict(path=str(root / 'report.json'), sha256=c.sha(root / 'report.json')) for root in args.artifacts],
        planned_recovery_steps_per_category=500, actual_recovery_steps_per_category=2 if args.short else 500,
        planned_native_fit_updates_per_stage=10000, actual_native_fit_updates_per_stage=2 if args.short else 10000,
        stages=[])
    c.write(args.output / 'report.json', report)
    try:
        paths = flow.layer_artifacts(args.artifacts, args.arm, short=args.short)
        model, source = flow.build_model(paths)
        report['raw_source'] = source
        cat, hf, training, vae, cfgs, raw_runtime, _stage = recovery.recipe(args.output)
        recovery.native.configure_deterministic_mode(cat.deterministic)
        recovery.native.set_seed(cat.seed)
        activation_runtime, plan, _identity = restore_cat_runtime_state(raw_runtime, access_token=hf.access_token)
        for index, category in enumerate(('gate_proj', 'up_proj', 'down_proj'), start=4):
            stage_output = args.output / f'{index:02d}_{category}'
            stage_output.mkdir(parents=True)
            stage = recovery.canonical.resolve_cat_after_category_stage(cat, training, category=category, round_idx=index)
            provenance = dict(arm=args.arm, phase=report['phase'], source_raw_manifest=report['source_raw_manifest'],
                source_raw_manifest_sha256=report['source_raw_manifest_sha256'], source_after_o=s.BASE_ID,
                actual_category=category, actual_recovery_steps=2 if args.short else 500,
                stage3_fit_actual_updates=2 if args.short else 10000, planned_recovery_schedule=500)
            meta, budget, state = checkpoint.save(model, stage_output / 'round_base', provenance=provenance)
            del state
            stage_row = dict(category=category, original_stage=asdict(stage), round_base_checkpoint_id=meta['checkpoint_id'],
                round_base_budget=budget, actual_started_unix=time.time())
            report['stages'].append(stage_row)
            report['current_stage'] = category
            c.write(args.output / 'report.json', report)
            model, evidence, probe = flow.recover(model, stage, vae, hf, stage_output,
                stage_output / 'round_base', meta, short=args.short)
            stage_row['recovery'] = evidence
            stage_row['deployment_warmup'] = warm(model, len(checkpoint.inventories(model)['native']))
            expected_logits = logits(model, probe)
            destination = args.output / 'final_model' if category == 'down_proj' else stage_output / 'after_recovery'
            saved, actual_budget, payload = checkpoint.save(model, destination, provenance=provenance)
            stage_row.update(after_recovery_checkpoint_id=saved['checkpoint_id'], after_recovery_budget=actual_budget,
                deployment_digest=c.digest(payload), deployment_path=str(destination))
            torch.save(dict(checkpoint_id=saved['checkpoint_id'], deployment_digest=stage_row['deployment_digest'],
                inputs=probe, logits=expected_logits), stage_output / 'same_input_deployment.pt')
            # The final path lives one level above a stage; put its reload evidence
            # alongside the final model so evaluation has a single stable location.
            evidence_path = stage_output / 'same_input_deployment.pt'
            del model, payload, probe, expected_logits
            gc.collect()
            torch.cuda.empty_cache()
            subprocess.run([sys.executable, '-B', '-u', '-m', 'experiments.adaptive_class_20261030.run_pipeline',
                '--fresh', str(destination), '--evidence', str(evidence_path)], cwd=s.ROOT, check=True)
            reload_path = destination.parent / 'strict_reload.json'
            reloaded = c.read(reload_path)
            c.require(reloaded['checkpoint_id'] == saved['checkpoint_id'] and reloaded['same_input_logits_exact'], 'Independent deployed consumer reload did not verify the actual model.')
            stage_row['strict_reload_report'] = str(reload_path)
            c.write(args.output / 'report.json', report)
            if category == 'down_proj':
                c.require(not saved['pending_dense_modules'] and len(saved['native_modules']) == 216 and len(saved['compact_modules']) == 36
                    and actual_budget['budget_passed'], 'Final whole-class output is incomplete or exceeds its true logical budget.')
                report.update(final_checkpoint_id=saved['checkpoint_id'], final_model=str(destination), final_budget=actual_budget,
                    final_strict_reload_report=str(reload_path))
            else:
                model, fresh_meta, strict, _budget = checkpoint.load(destination)
                c.require(fresh_meta['checkpoint_id'] == saved['checkpoint_id'] and not strict.missing_keys and not strict.unexpected_keys,
                    'Fit the next actual category only from this arm\'s own verified fresh merged boundary.')
                following = 'up_proj' if category == 'gate_proj' else 'down_proj'
                activation_runtime, plan, fit_report = flow.fit_next(model, cat, hf, training, vae, cfgs, activation_runtime, plan,
                    previous=category, category=following, output=stage_output / 'next_category_fit', short=args.short)
                stage_row['next_live_category_fit'] = fit_report
                c.write(args.output / 'report.json', report)
        c.require(all(c.sha(path) == digest for path, digest in pins.items()), 'A consumed pipeline source changed during this branch.')
        report.update(status='completed_short_whole_class_pipeline' if args.short else 'completed_whole_class_pipeline', exit_code=0)
    except BaseException:
        report.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.monotonic() - started)
        c.write(args.output / 'report.json', report)
    print(dict(status=report['status'], elapsed_seconds=report['elapsed_seconds']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('dynamic', 'rice'))
    parser.add_argument('--artifacts', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--fresh', type=Path)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.fresh:
        c.require(args.evidence is not None, 'Fresh verification requires the actual pre-save input/output evidence.')
        fresh(args.fresh, args.evidence)
    else:
        c.require(args.arm and args.artifacts and args.output, 'Provide the complete branch, completed artifact roots and new output.')
        execute(args)


if __name__ == '__main__':
    main()
