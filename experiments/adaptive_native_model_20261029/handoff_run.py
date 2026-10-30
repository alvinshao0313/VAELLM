"""Execute the bounded real mixed recovery/save/merge/fresh/next-up handoff."""
from dataclasses import asdict
import gc
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from experiments.adaptive_vector_rvq import common as c
from tools.cat_eval import _prepare_model_for_eval
from . import checkpoint, recovery
from .handoff_validation import HandoffCheck
from .next_category import fit_live_up


def prepared(path):
    record = c.read(path / 'report.json')
    c.require(record['status'] == 'CPU_verified_waiting_team_review' and record['exit_code'] == 0
              and record['independent_fresh']['no_old_parent_weight_read'], 'Reuse the actual completed CPU preparation.')
    for filename, expected in record['source_sha256'].items():
        c.require(c.sha(filename) == expected, 'CPU-verified source changed: ' + filename)
    return record


def sources():
    return {str(path.resolve()): c.sha(path) for path in Path(__file__).parent.glob('*.py')}


def probe_output(model, inputs):
    with torch.no_grad():
        output = model(**{name: tensor.to('cuda:0') for name, tensor in inputs.items()}, use_cache=False, logits_to_keep=1)
    c.require(output.logits.dtype == torch.bfloat16 and torch.isfinite(output.logits).all().item(),
              'Use the same actual BF16 last-token model output, without teacher or loss changes.')
    return output.logits.detach().cpu().contiguous()


def warm(model):
    result = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    c.require(result['total'] == result['warmed'] == 144 and not result['failed'], 'Warm the same144 native projections on both sides of deployment saving.')
    c.require(model.config._attn_implementation == 'flash_attention_2'
        and all(tensor.dtype == torch.float32 for name, tensor in model.named_buffers() if 'inv_freq' in name),
        'Keep original FA2/FP32 RoPE and all real model geometry.')
    return result


def check_entry(cpu_dir):
    record = prepared(cpu_dir)
    archive_manifest = c.read(recovery.ARCHIVE / 'manifest.json')
    c.require(c.sha(recovery.ARCHIVE / 'manifest.json') == record['source']['archive_manifest_sha256']
        and c.sha(recovery.ARCHIVE / 'cat_runtime_state.pt') == archive_manifest['files']['cat_runtime_state.pt']['sha256'],
        'The actual archived raw B progress/calibration material changed after CPU preparation.')
    cat, hf, training, vae, cfgs, runtime, stage = recovery.recipe(cpu_dir)
    c.require(not stage.config.runtime.distill_hif4_act, 'This narrow actual stage has HIF4 disabled in its original configuration.')
    return dict(status='entry_contract_verified_existing_CPU_reused', saved_checkpoint_id=record['saved_checkpoint_id'],
        source_fitter=str(Path(recovery.native.__file__).resolve()), max_steps=stage.config.opt.steps,
        batch=stage.config.opt.batch_size, max_length=stage.config.data.model_max_length,
        regular_save_steps=stage.save_steps, original_teacher_model_offload=stage.config.runtime.teacher_model_offload,
        inventories=record['independent_fresh']['actual_FP32_selection'], current_source_sha256=sources(),
        raw_runtime_sha256=archive_manifest['files']['cat_runtime_state.pt']['sha256'])


def execute(cpu_dir, output, resources):
    evidence = check_entry(cpu_dir)
    c.require(not output.exists(), 'Never overwrite a handoff run or earlier result.')
    output.mkdir(parents=True)
    report = dict(status='running', phase='loading_actual_mixed_round_base', pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        command=[sys.executable, *sys.argv], started_unix=time.time(), resource=resources,
        CPU_report_sha256=c.sha(cpu_dir / 'report.json'), entry_contract=evidence,
        team_review='Astra core narrow review found no material blocker; root authorized this GPU0 short chain',
        validation_only=True, training={}, no_third_stage_or_routing_or_quality=True)
    c.write(output / 'report.json', report)
    trainer = None
    try:
        torch.set_num_threads(8)
        cat, hf, training, vae, cfgs, runtime, stage = recovery.recipe(output)
        recovery.native.configure_deterministic_mode(cat.deterministic)
        recovery.native.set_seed(cat.seed)
        torch.cuda.reset_peak_memory_stats()
        model, meta, loaded, budget = checkpoint.load(cpu_dir / 'mixed_round_base')
        c.require(meta['checkpoint_id'] == evidence['saved_checkpoint_id'] and not loaded.missing_keys and not loaded.unexpected_keys,
                  'Load only the actual complete CPU-verified mixed boundary.')
        observer = HandoffCheck(report['training'])
        trainer, teacher, native_targets, compact_targets, inventory = recovery.make_trainer(model, stage, vae, hf, output,
            cpu_dir / 'mixed_round_base', meta, callbacks=[observer], validation_save_steps=1)
        observer.trainer = trainer
        report.update(phase='actual_original500_schedule_two_updates', actual_trainable_inventory=inventory,
                      original_stage=asdict(stage), initial_bit_budget=budget,
                      initialization='Fresh base seed31 and original Trainer stage seed35; no CAT global RNG/optimizer continuation claimed')
        c.write(output / 'report.json', report)
        train_output = trainer.train()
        c.require(trainer.state.global_step == 2, 'The bounded validation did not finish exactly two actual updates.')
        report['training']['save_audits'] = list(trainer.save_audits)
        report['training']['trainer_metrics'] = train_output.metrics
        report['training']['actual_log_history'] = trainer.state.log_history
        probe = observer.probe
        # Use the original release helper and offload the real teacher before
        # dense merging/prewarming; no stale Trainer/Adam graphs stay resident.
        recovery.canonical._release_cat_trainer_training_state(trainer, logger=recovery.log)
        teacher.offload_for_eval()
        trainer.teacher_runtime = None
        report['phase'] = 'merging_actual72_dense_and_BF16_deployment'
        c.write(output / 'report.json', report)
        model, merge = recovery.finalize(trainer, stage, native_targets)
        report['merge'] = merge
        report['deployment_warmup'] = warm(model)
        expected = probe_output(model, probe)
        saved, saved_budget, payload = checkpoint.save(model, output / 'after_recovery', provenance=dict(
            CPU_checkpoint_id=meta['checkpoint_id'], CPU_report_sha256=report['CPU_report_sha256'],
            actual_updates=2, schedule_max_steps=500, raw_B_fixture_only=True,
            purpose='Bounded mixed recovery/save/retrain/merge/fresh handoff; not a500-update or quality endpoint'))
        report.update(saved_checkpoint_id=saved['checkpoint_id'], deployment_tensor_digest=c.digest(payload),
            deployment_tensor_count=len(payload), saved_bit_budget=saved_budget,
            expected_output_digest=c.digest({'logits': expected}), expected_output_shape=list(expected.shape),
            parent_peak_allocated_bytes=torch.cuda.max_memory_allocated(), parent_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            phase='deployment_saved_fresh_pending')
        torch.save(dict(inputs=probe, expected_logits=expected), output / 'same_actual_train_input.pt')
        c.write(output / 'report.json', report)
        observer.trainer = None
        del trainer, teacher, model, native_targets, compact_targets, observer, payload
        trainer = None
        gc.collect()
        torch.cuda.empty_cache()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).with_name('run_handoff.py')), '--prepared', str(cpu_dir),
            '--output', str(output), '--fresh'], cwd=recovery.ROOT, check=True)
        verified = c.read(output / 'fresh_handoff.json')
        c.require(verified['status'] == 'fresh_mixed_then_real_next_up_verified', 'Independent fresh/next-up handoff did not complete.')
        report.update(status='mixed_recovery_save_merge_fresh_next_up_verified', exit_code=0,
            phase='completed_bounded_chain', independent_fresh=verified, actual_source_sha256=sources(),
            completed_scope='Actual2/500 recovery, step1 mixed save/retrain, actual72 dense merge, stable save and independent fresh, actualM36 up S2x2 fit/replacement only')
    except BaseException:
        if trainer is not None:
            report['training']['save_audits'] = list(trainer.save_audits)
        report.update(status='failed_preserved_partial_evidence', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        c.write(output / 'report.json', report)
    print(json.dumps(dict(status=report['status'], seconds=report['elapsed_seconds'])), flush=True)


def fresh(cpu_dir, output):
    parent = c.read(output / 'report.json')
    c.require(parent['phase'] == 'deployment_saved_fresh_pending' and os.getppid() == parent['pid'],
              'Fresh GPU execution is only the independent child of this same assigned short handoff.')
    result = dict(status='running', phase='fresh_loading_own_mixed_deployment', pid=os.getpid(), started_unix=time.time())
    c.write(output / 'fresh_handoff.json', result)
    try:
        torch.set_num_threads(8)
        torch.cuda.reset_peak_memory_stats()
        original_load, reads = torch.load, []

        def only_deployment(path, *args, **kwargs):
            actual = Path(path).resolve()
            c.require(actual == output / 'after_recovery' / checkpoint.STATE_FILE, 'Fresh mixed loader attempted an old parent payload read.')
            reads.append(str(actual))
            return original_load(path, *args, **kwargs)
        torch.load = only_deployment
        try:
            model, meta, loaded, budget = checkpoint.load(output / 'after_recovery')
        finally:
            torch.load = original_load
        c.require(meta['checkpoint_id'] == parent['saved_checkpoint_id'] and not loaded.missing_keys and not loaded.unexpected_keys
            and c.digest(model.state_dict()) == parent['deployment_tensor_digest'] and all(not parameter.requires_grad for parameter in model.parameters()),
            'Independent fresh complete deployment state/freeze differs from the actual saved BF16 state.')
        result.update(phase='fresh_same_actual_input_output', exact_complete_deployment_state=True, deployment_fields=len(model.state_dict()),
                      effective_deployment_weight_reads=reads, old_parent_weights_read=False, actual_bit_budget=budget, warmup=warm(model))
        probe = torch.load(output / 'same_actual_train_input.pt', map_location='cpu', weights_only=True)
        actual = probe_output(model, probe['inputs'])
        c.require(torch.equal(actual, probe['expected_logits']), 'Independent fresh deployment changed the same actual input output.')
        result.update(same_input_logits_exact=True, max_abs_difference=float((actual.float() - probe['expected_logits'].float()).abs().max()),
                      output_digest=c.digest({'logits': actual}), phase='real_N_live_up_fit_and_native_replacement')
        c.write(output / 'fresh_handoff.json', result)
        cat, hf, training, vae, cfgs, runtime, stage = recovery.recipe(output)
        c.require(c.sha(recovery.ARCHIVE / 'cat_runtime_state.pt') == parent['entry_contract']['raw_runtime_sha256'],
                  'Fresh next-up must use the same preserved actual raw progress/calibration material.')
        result['next_up'] = fit_live_up(model, cat, hf, training, vae, cfgs, runtime, output / 'next_up')
        result.update(status='fresh_mixed_then_real_next_up_verified', exit_code=0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            no_full500_or_third_stage_or_routing_or_quality_claim=True)
    except BaseException:
        result.update(status='failed_preserved_fresh_evidence', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        result.update(finished_unix=time.time(), elapsed_seconds=time.time() - result['started_unix'])
        c.write(output / 'fresh_handoff.json', result)
    print(json.dumps(dict(status=result['status'], seconds=result['elapsed_seconds'])), flush=True)
