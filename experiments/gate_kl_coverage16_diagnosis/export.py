"""Export the existing sixteen-group choices on unchanged f927 decoders, then full8."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from transformers import AutoTokenizer

from experiments.full_weight_bits_probe.core import popcounts
from experiments.gate_down_student_context_evaluation.run import compare_control
from experiments.gate_kl_coverage16_diagnosis.common import (
    BASE, BANKS, GROUPS, ROOT, SOURCE, SOURCE_ID, TARGET, VECTORS, bound, check, read)
from experiments.gate_kl_coverage16_probe.common import choice_inventory, replay_codes
from experiments.gate_student_context_evaluation.deploy import decoded_gate, native_policy
from experiments.gate_student_context_evaluation.prepare import budget_equal, require_state
from experiments.longbench_v1.common import check_hashes, file_stat, sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen3_w25.step_export_callback import run_logged_stage
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.w25_eval.acceptance import collect_results, validate_checkpoint
from litebsq.vae_linear import VAELinear
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.v6_model_loader import load_v6_model_checkpoint


def cpu_plan(output_dir):
    """Bind small completed evidence; defer source tensor replay to actual deployment."""
    require(not torch.cuda.is_initialized(), 'Prepare without CUDA or loading model tensors.')
    output = Path(output_dir).resolve()
    previous = BASE / 'gate_kl_coverage16_decoder500'
    template = read(previous / 'launch.prepared.json')['evaluation']
    report_path = previous / 'run/report.json'; report = read(report_path)
    proposal = report['proposal']; choices = previous / 'run/choices.pt'
    require(report['status'] == 'completed' and report['source_checkpoint_id'] == SOURCE_ID
            and proposal['source_checkpoint_id'] == SOURCE_ID and proposal['all13_original_bytes_restored']
            and proposal['frozen_complement_count'] == 3410
            and report['stage_xor'] == proposal['stage_xor'] == [62024, 2488]
            and Path(report['choices_path']) == choices and sha256(choices) == report['choices_sha256']
            and [row['group'] for row in proposal['choice_inventory']] == list(range(GROUPS)),
            'Require the completed existing mixed128 sixteen-group proposal, without fitting again.')
    keys = ('source_checkpoint', 'source_checkpoint_id', 'source_weight_stat', 'source_native_policy',
            'source_bit_budget', 'hf_weight_stats', 'architecture')
    plan = {key: deepcopy(template[key]) for key in keys}
    require(Path(plan['source_checkpoint']) == SOURCE and plan['source_checkpoint_id'] == SOURCE_ID
            and read(SOURCE / 'checkpoint_meta.json')['checkpoint_id'] == SOURCE_ID,
            'Only fresh f927 supplies all decoders and the complete complement.')
    controls = {'f927': deepcopy(template['controls']['f927'])}
    specs = {
        'four_group_codes_only': ('gate_kl_calibration_full8/alpaca64_fineweb64',
            'c7f5f6bd-cf10-4dd9-bfe0-92a91fe455f5'),
        'four_group_decoder500': ('gate_kl_calibration_decoder_full8/alpaca64_fineweb64',
            '55a3aeb5-9b37-43ae-9da9-6964c9d6abac'),
        'sixteen_group_decoder500': ('gate_kl_coverage16_decoder500',
            '834bdf15-1e93-4088-8da3-02215d7e8f81'),
    }
    evidence = [previous / 'launch.prepared.json', report_path, choices, SOURCE / 'checkpoint_meta.json']
    for name, (relative, identity) in specs.items():
        directory = BASE / relative; summary_path = directory / 'full8_summary.json'; summary = read(summary_path)
        require(summary['status'] == 'full8_evaluation_verified' and summary['checkpoint_id'] == identity
                and summary['source_checkpoint_id'] == SOURCE_ID, 'Wrong completed control: ' + name)
        controls[name] = {'checkpoint': str(directory / 'final_model'), 'checkpoint_id': identity,
            'results': [str(directory / (part + '.json')) for part in ('mmlu', 'seven')],
            'mean_percent': summary['mean_percent']}
        evidence.append(summary_path)
    for name, control in controls.items():
        directory = Path(control['checkpoint']); meta_path = directory / 'checkpoint_meta.json'; meta = read(meta_path)
        require(meta['checkpoint_id'] == control['checkpoint_id'], 'Control UUID changed: ' + name)
        score = collect_results(control['results'], plan['architecture'], checkpoint=(directory, meta))
        require(score['leaf_count'] == 64 and score['sample_count'] == 24742
                and score['mean_percent'] == control['mean_percent'], 'Incomplete or changed raw control: ' + name)
        evidence.extend([meta_path, *map(Path, control['results'])])
    commands = [{'name': 'deploy', 'argv': [sys.executable, '-m', __name__, '--preparation',
        str(output / 'launch.bound.json'), '--phase', 'deploy']}]
    for name, tasks, batch in (('mmlu', 'mmlu', '64'),
            ('seven', 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa', '32')):
        commands.append({'name': name, 'argv': [sys.executable, 'tools/cat_eval.py', '--checkpoint_dir',
            str(output / 'final_model'), '--strict', '--eval_lm_eval', '--eval_device', 'cuda:0',
            '--tasks', tasks, '--num_fewshot', '0', '--lm_batch_size', batch, '--tokenizer_use_fast', 'true',
            '--prewarm_group_size', '8', '--eval_log_dir', str(output / (name + '_logs')),
            '--output_json', str(output / (name + '.json'))]})
    plan.update(output_dir=str(output), controls=controls, commands=commands,
        choices_path=str(choices), choices_sha256=report['choices_sha256'], proposal_report=str(report_path),
        proposal_report_sha256=sha256(report_path), initial_state=proposal['initial_state'],
        applied_state=proposal['applied_state'], stage_xor=proposal['stage_xor'],
        choice_inventory=proposal['choice_inventory'], input_sha256={str(p): sha256(p) for p in evidence},
        primary_control='four_group_codes_only', protocol={'full8_leaves': 64, 'full8_samples': 24742,
            'num_fewshot': 0, 'mmlu_batch': 64, 'seven_batch': 32, 'limit': None,
            'groups': list(range(GROUPS)), 'vectors': VECTORS, 'new_training': False,
            'mutable_decoder_tensors': 0, 'mutable_packed_banks': 2,
            'boundary': 'Replay the existing mixed128 groups0..15 on fresh f927; every decoder remains original. '
                'No new search or fit. Full8 does not substitute for complete LongBench/nine acceptance.'})
    return plan


def runtime_inputs(path):
    path = Path(path).resolve(); record = read(path); bound(record)
    plan = record['evaluation']; output = Path(record['output_dir']).resolve()
    require(path == output / 'launch.bound.json' and output == Path(plan['output_dir'])
            and Path.cwd().resolve() == ROOT and Path(plan['source_checkpoint']) == SOURCE
            and plan['source_checkpoint_id'] == record['source_checkpoint_id'] == SOURCE_ID
            and GROUPS == 16 and VECTORS == 64512, 'Prepared source, output or scope changed.')
    check(record); check_hashes(plan['input_sha256'])
    require(sha256(plan['choices_path']) == plan['choices_sha256']
            and sha256(plan['proposal_report']) == plan['proposal_report_sha256'], 'Existing proposal artifacts changed.')
    return record, plan, output


def patched_state(original, choices, plan):
    require(len(original) == 3423 and choice_inventory(choices) == plan['choice_inventory'],
            'Wrong full source inventory or sixteen-group choices.')
    local = {key[len(TARGET) + 1:]: value for key, value in original.items() if key.startswith(TARGET + '.')}
    require({key: tensor_record(value) for key, value in local.items()} == plan['initial_state'],
            'Fresh f927 gate differs from the actual original proposal state.')
    endpoint, replay_xor = replay_codes(local, choices)
    require({key: tensor_record(value) for key, value in endpoint.items()} == plan['applied_state'],
            'Exact bank replay differs from the recorded pre-fit proposal.')
    decoder = 0
    for key, value in local.items():
        if key in BANKS:
            require_state({key: endpoint[key][VECTORS:]}, {key: value[VECTORS:]}, 'Frozen groups16..383')
        else:
            require_state({key: endpoint[key]}, {key: value}, 'All local decoders, protection and rotation are frozen')
            decoder += int(key.startswith('_parallel_stage_decoder.'))
    xor = [popcounts(local[key], endpoint[key]) for key in BANKS]
    require(len(local) == 13 and decoder == 6 and xor == replay_xor == plan['stage_xor'] == [62024, 2488],
            'Code-only replay must preserve six decoder leaves and the actual XOR.')
    expected = {**original, **{TARGET + '.' + key: value for key, value in endpoint.items()}}
    outside = [key for key in original if not key.startswith(TARGET + '.')]
    banks = [key for key in original if key.rsplit('.', 1)[-1] in BANKS]
    require(len(outside) == 3410 and all(expected[key] is original[key] for key in outside)
            and len(banks) == 504 and sum(key in outside for key in banks) == 502, 'Another tensor was replaced.')
    patch = {'patched_local_tensor_count': 13, 'frozen_outside_target_tensors': 3410,
        'selected_decoder_tensors': 0, 'actual_changed_decoder_tensors': 0, 'all_decoder_tensors_exact': True,
        'frozen_local_nonmutable_tensors': 11, 'frozen_local_nonmutable_exact': True,
        'frozen_complete_state_tensors': 3421, 'stage_changed_bits': xor, 'changed_code_bits': sum(xor),
        'groups': list(range(GROUPS)), 'vectors': VECTORS, 'exact_choices_replay': True,
        'codes_outside_selected_groups_exact': True, 'source_packed_bank_count': 504,
        'frozen_outside_target_packed_bank_count': 502, 'frozen_outside_target_packed_banks_exact': True}
    return expected, endpoint, patch


@torch.no_grad()
def deploy(path):
    record, plan, output = runtime_inputs(path)
    require(not (output / 'final_model').exists() and not (output / 'audit.json').exists(), 'Fresh deployment only.')
    audit = {'status': 'deploying', 'started_unix': time.time()}
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(0)
        require(torch.cuda.device_count() == 1, 'Use only the bound GPU.')
        meta = read(SOURCE / 'checkpoint_meta.json'); require(meta['checkpoint_id'] == SOURCE_ID, 'Source UUID changed.')
        original = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
        choices = torch.load(plan['choices_path'], map_location='cpu', weights_only=True)
        expected, endpoint, patch = patched_state(original, choices, plan)
        budget_equal(audit_state_dict(meta, original, max_bpw=2.5, max_lora_rank=0), plan['source_bit_budget'])
        model, loaded_meta, loaded = load_v6_model_checkpoint(str(SOURCE), strict=True, map_location='cpu', expected_kind='final_model')
        require(loaded_meta == meta and not loaded.missing_keys and not loaded.unexpected_keys, 'Source strict load failed.')
        model.eval().requires_grad_(False); require_state(model.state_dict(), original, 'Fresh complete f927')
        rope = native_policy(model, plan['source_native_policy']); gate = model.get_submodule(TARGET)
        require(isinstance(gate, VAELinear), 'The target must remain a native VAELinear.')
        loaded = gate.load_state_dict(endpoint, strict=True)
        require(not loaded.missing_keys and not loaded.unexpected_keys, 'Local strict load failed.')
        gate.clear_decoded_weight_cache(); gate._build_parallel_stage_decode_plan()
        require_state(model.state_dict(), expected, 'Patched complete model'); reference = decoded_gate(model)
        require_state(model.state_dict(), expected, 'Native decode preserves persistent state')
        tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], use_fast=True, local_files_only=True)
        require(tokenizer.is_fast, 'Use the original fast evaluation tokenizer.')
        provenance = {'source_checkpoint_id': SOURCE_ID, 'target': TARGET, 'choices_sha256': plan['choices_sha256'],
            'proposal_report_sha256': plan['proposal_report_sha256'], 'patch_contract': patch,
            'purpose': 'Existing mixed128 sixteen-group code-only endpoint; no decoder fitting or optimizer state.'}
        saved = save_v6_full_checkpoint(model, str(output / 'final_model'), checkpoint_kind='final_model',
            compressed_targets=meta['compressed_targets'], pending_dense_targets=meta['pending_dense_targets'],
            skip_targets=meta['skip_targets'], legacy_original_only_sources=meta.get('legacy_original_only_sources', []),
            train_mode='none', norm_train_mode='none', lm_head_train_mode='none', lora_config=None,
            completed_categories=meta.get('completed_categories', []), compression_categories=meta.get('compression_categories', []),
            target_layers=[18], target_modules=['gate_proj'],
            finalized_status={'decoder_finalized': True, 'runtime_clean': True, 'isolated_endpoint_persisted': True},
            runtime_audit=provenance, base_model_path=meta['base_model_path'], tokenizer=tokenizer, save_config=True,
            extra_meta={'isolated_gate_endpoint': provenance})
        final_meta = saved['meta_payload']
        require(final_meta['checkpoint_id'] != SOURCE_ID and final_meta['converted_modules'] == meta['converted_modules'],
                'Require a fresh UUID and unchanged compressed module specifications.')
        persisted = torch.load(saved['state_dict'], map_location='cpu', weights_only=True, mmap=True)
        serialized = require_state(persisted, expected, 'Serialized complete code-only model')
        budget = audit_state_dict(final_meta, persisted, max_bpw=2.5, max_lora_rank=0)
        budget_equal(budget, plan['source_bit_budget'])
        del model, gate, tokenizer; gc.collect(); torch.cuda.empty_cache()
        fresh, fresh_meta, loaded = load_v6_model_checkpoint(str(output / 'final_model'), strict=True,
            map_location='cpu', expected_kind='final_model')
        require(fresh_meta == final_meta and not loaded.missing_keys and not loaded.unexpected_keys, 'Fresh strict reload failed.')
        fresh.eval().requires_grad_(False); count = require_state(fresh.state_dict(), expected, 'Independent complete reconstruction')
        require(native_policy(fresh, plan['source_native_policy']) == rope, 'Reload changed native FP32 RoPE.')
        actual = decoded_gate(fresh); require(torch.equal(actual, reference), 'Native BF16 decoded gate changed on reload.')
        require_state(fresh.state_dict(), expected, 'Reload decode preserves all persistent state')
        check(record); check_hashes(plan['input_sha256'])
        audit.update(status='verified', source_checkpoint_id=SOURCE_ID, checkpoint_id=final_meta['checkpoint_id'],
            choices_sha256=plan['choices_sha256'], proposal_report_sha256=plan['proposal_report_sha256'], **patch,
            serialized_tensors_compared=serialized, reloaded_tensors_compared=count,
            fresh_strict_reload_state_exact=True, endpoint_state_exact=True, frozen_outside_target_exact=True,
            native_full_weight_reload_exact=True, integer_budget_unchanged=True, rank_zero=True,
            native_full_weight=tensor_record(actual), native_rope_buffers=rope, bit_budget=budget,
            checkpoint_meta_sha256=sha256(saved['meta']), checkpoint_state_stat=file_stat(saved['state_dict']),
            maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
            maximum_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0))
    except BaseException as error:
        audit.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()); raise
    finally:
        audit['finished_unix'] = time.time(); write_json(output / 'audit.json', audit)


def run(path):
    record, plan, output = runtime_inputs(path)
    require(not any((output / name).exists() for name in ('evaluation.json', 'final_model', 'audit.json',
            'mmlu.json', 'seven.json', 'full8_summary.json', 'evaluation.exit')), 'Fresh evaluation only.')
    progress = {'status': 'running', 'started_unix': time.time(), 'stages': {}, 'preparation': str(path)}; code = 1
    try:
        require([row['name'] for row in plan['commands']] == ['deploy', 'mmlu', 'seven'], 'Wrong stage inventory.')
        for row in plan['commands']:
            name, command = row['name'], row['argv']
            require(Path(command[0]).resolve() == Path(sys.executable).resolve()
                    and '--lm_limit' not in command and '--limit' not in command, 'Keep the complete fixed-protocol evaluation.')
            progress['stages'][name] = {'argv': command, 'started_unix': time.time()}; write_json(output / 'evaluation.json', progress)
            try:
                status = run_logged_stage(command, name=name, output_dir=output, env=os.environ.copy(), allowed_exit_codes=(0,))
            finally:
                exit_path = output / (name + '.exit')
                if exit_path.exists(): progress['stages'][name]['exit_code'] = int(exit_path.read_text().strip())
                progress['stages'][name]['finished_unix'] = time.time(); write_json(output / 'evaluation.json', progress)
            require(status == 0, 'Native deployment/evaluation stage failed.')
        final, meta, budget = validate_checkpoint(output / 'final_model', plan['architecture']); audit = read(output / 'audit.json')
        require(audit['status'] == 'verified' and audit['checkpoint_id'] == meta['checkpoint_id']
                and audit['source_checkpoint_id'] == SOURCE_ID and audit['choices_sha256'] == plan['choices_sha256']
                and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 3423
                and audit['frozen_complete_state_tensors'] == 3421 and audit['stage_changed_bits'] == plan['stage_xor']
                and audit['selected_decoder_tensors'] == audit['actual_changed_decoder_tensors'] == 0
                and audit['all_decoder_tensors_exact'] and audit['exact_choices_replay']
                and audit['codes_outside_selected_groups_exact'] and audit['fresh_strict_reload_state_exact']
                and audit['native_full_weight_reload_exact'], 'Code-only deployment proof is incomplete or mismatched.')
        budget_equal(budget, plan['source_bit_budget']); paths = [output / (name + '.json') for name in ('mmlu', 'seven')]
        current = collect_results(paths, plan['architecture'], checkpoint=(final, meta))
        require(current['leaf_count'] == 64 and current['sample_count'] == 24742, 'Incomplete eight-task result.')
        comparisons = {}
        for name, control in plan['controls'].items():
            directory = Path(control['checkpoint']); control_meta = read(directory / 'checkpoint_meta.json')
            require(control_meta['checkpoint_id'] == control['checkpoint_id'], 'Control UUID changed: ' + name)
            previous = compare_control(paths, control['results'], plan['architecture'], checkpoint=(directory, control_meta))
            require(previous['mean_percent'] == control['mean_percent'], 'Raw control score changed: ' + name)
            comparisons[name] = {'checkpoint_id': control['checkpoint_id'], 'mean_percent': previous['mean_percent'],
                'mean_delta_pp': current['mean_percent'] - previous['mean_percent'],
                'task_delta_pp': {key: 100 * (value - previous['task_metrics'][key]) for key, value in current['task_metrics'].items()}}
        check(record); check_hashes(plan['input_sha256'])
        summary = {'status': 'full8_evaluation_verified', 'checkpoint_id': meta['checkpoint_id'],
            'checkpoint_dir': str(final), 'source_checkpoint_id': SOURCE_ID, **current, 'bit_budget': budget,
            'stage_changed_bits': audit['stage_changed_bits'], 'changed_code_bits': audit['changed_code_bits'],
            'decoder_fit_steps': 0, 'all_decoder_tensors_exact': True, 'comparisons': comparisons,
            'primary_control': plan['primary_control'], 'protocol': plan['protocol'],
            'choices_sha256': plan['choices_sha256'], 'proposal_report_sha256': plan['proposal_report_sha256'],
            'audit_sha256': sha256(output / 'audit.json'), 'longbench_status': 'not_evaluated',
            'current_nine_item_goal_status': 'not_evaluated', 'goal_acceptance': False}
        write_json(output / 'full8_summary.json', summary)
        progress.update(status='completed', checkpoint_id=meta['checkpoint_id'], mean_percent=current['mean_percent'],
            full8_summary_sha256=sha256(output / 'full8_summary.json')); code = 0
    except BaseException as error:
        progress.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(progress['traceback'], flush=True)
    finally:
        progress.update(exit_code=code, finished_unix=time.time()); write_json(output / 'evaluation.json', progress)
        (output / 'evaluation.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--phase', choices=('run', 'deploy'), default='run')
    options = parser.parse_args()
    if options.phase == 'deploy': deploy(options.preparation)
    else: sys.exit(run(options.preparation))
