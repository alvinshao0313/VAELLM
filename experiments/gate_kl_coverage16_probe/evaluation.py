"""Native deployment and complete eight-task evaluation of the fixed sixteen groups."""
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
from experiments.gate_group_accept_probe.core import apply_choices
from experiments.gate_kl_coverage16_probe.common import (
    BASE, BANKS, GROUPS, ROOT, SOURCE, SOURCE_ID, TARGET, VECTORS, check, read)
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
    """Read the completed four-group protocol and small control evidence only."""
    require(not torch.cuda.is_initialized(), 'Prepare the evaluation plan without CUDA.')
    output = Path(output_dir).resolve()
    previous = BASE / 'gate_kl_calibration_decoder_full8/alpaca64_fineweb64'
    template = read(previous / 'launch.json'); summary = read(previous / 'full8_summary.json')
    require(template['status'] == 'completed' and template['exit_code'] == 0
            and summary['status'] == 'full8_evaluation_verified'
            and summary['checkpoint_id'] == '55a3aeb5-9b37-43ae-9da9-6964c9d6abac'
            and template['source_checkpoint_id'] == SOURCE_ID,
            'Use the completed same-source four-group decoder500 protocol.')
    keys = ('source_checkpoint', 'source_checkpoint_id', 'source_weight_stat', 'source_native_policy',
            'source_bit_budget', 'hf_weight_stats', 'architecture')
    plan = {key: deepcopy(template[key]) for key in keys}
    require(Path(plan['source_checkpoint']) == SOURCE
            and read(SOURCE / 'checkpoint_meta.json')['checkpoint_id'] == SOURCE_ID,
            'Only original f927 supplies the deployment complement.')
    specs = {
        'f927': (SOURCE, SOURCE_ID, template['control_result_paths']),
        'original_codes500': (Path(template['gate_only_control_checkpoint']),
            template['gate_only_control_checkpoint_id'], template['gate_only_control_results']),
        'four_group_decoder500': (previous / 'final_model', summary['checkpoint_id'],
            [str(previous / (name + '.json')) for name in ('mmlu', 'seven')]),
    }
    evidence = [previous / name for name in ('launch.json', 'full8_summary.json', 'review.json')]
    controls = {}
    for name, (checkpoint, identity, paths) in specs.items():
        meta_path = checkpoint / 'checkpoint_meta.json'; meta = read(meta_path)
        require(meta['checkpoint_id'] == identity, 'Control metadata identity changed: ' + name)
        scored = collect_results(paths, plan['architecture'], checkpoint=(checkpoint, meta))
        require(scored['leaf_count'] == 64 and scored['sample_count'] == 24742, 'Incomplete existing control: ' + name)
        controls[name] = {'checkpoint': str(checkpoint), 'checkpoint_id': identity,
            'results': list(map(str, paths)), 'mean_percent': scored['mean_percent']}
        evidence.extend([meta_path, *map(Path, paths)])
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
        input_sha256={str(path): sha256(path) for path in evidence},
        primary_control='four_group_decoder500', protocol={'full8_leaves': 64, 'full8_samples': 24742,
            'num_fewshot': 0, 'mmlu_batch': 64, 'seven_batch': 32, 'limit': None,
            'boundary': 'Fresh f927 mixed128 groups0..15 then conditional-MLP decoder500; compare completed '
                'four-group500, original-code500 and f927. Full8 does not substitute for full21/nine acceptance.'})
    return plan


def runtime_inputs(path):
    path = Path(path).resolve(); record = read(path); plan = record['evaluation']
    output = Path(record['output_dir']).resolve()
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path.cwd().resolve() == ROOT and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and not record['queued'] and not record['automatic_retry']
            and all(os.environ.get(k) == v for k, v in record['environment'].items()),
            'Use the bound single GPU and prepared native environment without retry.')
    require(output == Path(plan['output_dir']) and path == output / 'launch.bound.json'
            and plan['source_checkpoint_id'] == record['source_checkpoint_id'] == SOURCE_ID
            and Path(plan['source_checkpoint']) == SOURCE and GROUPS == 16 and VECTORS == 64512,
            'Prepared source, sixteen-group scope or output differs.')
    check(record); check_hashes(plan['input_sha256'])
    report_path = output / 'run/report.json'; report = read(report_path)
    require(report['status'] == 'completed' and report['source_checkpoint_id'] == SOURCE_ID
            and report['preparation_sha256'] == record['preparation_sha256']
            and report['steps'] == 500 and report['frozen_codes_exact']
            and report['training']['status'] == 'completed'
            and report['training']['steps'] == report['training']['schedule_steps'] == 500
            and report['training']['frozen_state_exact'] and report['training']['frozen_objective_buffers_exact'],
            'Require the completed original-initialization proposal and fixed-code native500 fit.')
    require(Path(report['endpoint_path']).resolve() == output / 'run/gate_state.pt'
            and Path(report['choices_path']).resolve().is_relative_to(output / 'run')
            and sha256(report['endpoint_path']) == report['endpoint_sha256']
            and sha256(report['choices_path']) == report['choices_sha256']
            and len(report['endpoint_state']) == 13, 'The actual local endpoint or choices changed.')
    return record, plan, output, report, report_path


def patched_state(original, endpoint, choices, report):
    require(len(original) == 3423 and choices.dtype == torch.uint8 and tuple(choices.shape) == (VECTORS,)
            and int(choices.max()) <= 64, 'Wrong complete source or sixteen-group compact choices.')
    local = {key[len(TARGET) + 1:]: value for key, value in original.items() if key.startswith(TARGET + '.')}
    require(set(local) == set(endpoint) and len(local) == 13, 'Expected exactly thirteen native gate tensors.')
    initial, changes, frozen = dict(local), {}, 0
    for key, value in endpoint.items():
        old = local[key]
        require(value.shape == old.shape and value.dtype == old.dtype, 'Changed native topology/dtype: ' + key)
        if key in BANKS:
            require(tuple(old.shape) == (1548288, 1, 4) and old.dtype == torch.uint8, 'Wrong packed geometry.')
            initial[key] = old.clone(); apply_choices(initial[key][:VECTORS], choices, BANKS.index(key))
            require_state({key: value}, {key: initial[key]}, 'Saved codes versus exact sixteen-group replay')
            require_state({key: value[VECTORS:]}, {key: old[VECTORS:]}, 'Frozen groups16..383')
        elif key.startswith('_parallel_stage_decoder.'):
            require(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()), 'Invalid deployed decoder.')
            changes[key] = int(torch.count_nonzero(value != old))
        else:
            require_state({key: value}, {key: old}, 'Frozen local protection and rotation'); frozen += 1
    require(len(changes) == 6 and frozen == 5 and sum(changes.values()) > 0,
            'Require six native decoder leaves, five frozen local tensors and a real deployed update.')
    require({k: tensor_record(v) for k, v in initial.items()} == report['training']['initial_payload'],
            'Decoder500 did not start from original f927 plus these exact choices.')
    xor = [popcounts(local[key], endpoint[key]) for key in BANKS]
    expected_xor = [int(((choices > stage * 32) & (choices <= (stage + 1) * 32)).sum()) for stage in range(2)]
    require(xor == expected_xor == report['stage_xor'], 'Actual packed changes differ from saved choices/report.')
    expected = {**original, **{TARGET + '.' + k: v for k, v in endpoint.items()}}
    outside = [k for k in original if not k.startswith(TARGET + '.')]
    banks = [k for k in original if k.rsplit('.', 1)[-1] in BANKS]
    require(len(outside) == 3410 and all(expected[k] is original[k] for k in outside)
            and len(banks) == 504 and sum(k in outside for k in banks) == 502, 'Unexpected complement mutation.')
    return expected, {'patched_local_tensor_count': 13, 'frozen_outside_target_tensors': 3410,
        'selected_decoder_tensors': 6, 'changed_decoder_elements': changes,
        'actual_changed_decoder_tensors': sum(n > 0 for n in changes.values()),
        'stage_changed_bits': xor, 'changed_code_bits': sum(xor), 'groups': list(range(GROUPS)),
        'vectors': VECTORS, 'exact_choices_replay': True, 'codes_outside_selected_groups_exact': True,
        'source_packed_bank_count': 504, 'frozen_outside_target_packed_bank_count': 502,
        'frozen_outside_target_packed_banks_exact': True,
        'frozen_local_nonmutable_tensors': frozen, 'frozen_local_nonmutable_exact': True}


@torch.no_grad()
def deploy(path):
    record, plan, output, report, report_path = runtime_inputs(path)
    require(not (output / 'final_model').exists() and not (output / 'audit.json').exists(), 'Fresh deployment only.')
    audit = {'status': 'deploying', 'started_unix': time.time()}
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(0)
        require(torch.cuda.device_count() == 1, 'The deployment must use only its bound GPU.')
        meta = read(SOURCE / 'checkpoint_meta.json'); require(meta['checkpoint_id'] == SOURCE_ID, 'Source UUID changed.')
        original = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
        endpoint = torch.load(report['endpoint_path'], map_location='cpu', weights_only=True, mmap=True)
        choices = torch.load(report['choices_path'], map_location='cpu', weights_only=True)
        require({k: tensor_record(v) for k, v in endpoint.items()} == report['endpoint_state']
                == report['training']['final_payload'], 'Saved endpoint differs from the completed fit.')
        expected, patch = patched_state(original, endpoint, choices, report)
        budget_equal(audit_state_dict(meta, original, max_bpw=2.5, max_lora_rank=0), plan['source_bit_budget'])
        model, loaded_meta, loaded = load_v6_model_checkpoint(str(SOURCE), strict=True, map_location='cpu', expected_kind='final_model')
        require(loaded_meta == meta and not loaded.missing_keys and not loaded.unexpected_keys, 'Source strict load failed.')
        model.eval().requires_grad_(False); require_state(model.state_dict(), original, 'Fresh complete f927')
        rope = native_policy(model, plan['source_native_policy']); gate = model.get_submodule(TARGET)
        require(isinstance(gate, VAELinear), 'Target must stay a native VAELinear.')
        loaded = gate.load_state_dict(endpoint, strict=True)
        require(not loaded.missing_keys and not loaded.unexpected_keys, 'Local strict load failed.')
        gate.clear_decoded_weight_cache(); gate._build_parallel_stage_decode_plan()
        require_state(model.state_dict(), expected, 'Patched complete model'); reference = decoded_gate(model)
        require_state(model.state_dict(), expected, 'Native decode preserves persistent state')
        tokenizer = AutoTokenizer.from_pretrained(meta['base_model_path'], use_fast=True, local_files_only=True)
        provenance = {'source_checkpoint_id': SOURCE_ID, 'target': TARGET, 'endpoint_sha256': report['endpoint_sha256'],
            'choices_sha256': report['choices_sha256'], 'fit_report_sha256': sha256(report_path), 'patch_contract': patch,
            'purpose': 'Fresh sixteen-group mixed128/conditional-MLP500 quality endpoint; not a resume checkpoint.'}
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
                'Require a fresh UUID with every original module specification preserved.')
        persisted = torch.load(saved['state_dict'], map_location='cpu', weights_only=True, mmap=True)
        serialized = require_state(persisted, expected, 'Serialized complete endpoint')
        budget = audit_state_dict(final_meta, persisted, max_bpw=2.5, max_lora_rank=0)
        budget_equal(budget, plan['source_bit_budget'])
        del model, gate, tokenizer; gc.collect(); torch.cuda.empty_cache()
        fresh, fresh_meta, loaded = load_v6_model_checkpoint(str(output / 'final_model'), strict=True,
            map_location='cpu', expected_kind='final_model')
        require(fresh_meta == final_meta and not loaded.missing_keys and not loaded.unexpected_keys, 'Fresh strict reload failed.')
        fresh.eval().requires_grad_(False); count = require_state(fresh.state_dict(), expected, 'Independent complete reconstruction')
        require(native_policy(fresh, plan['source_native_policy']) == rope, 'Reload changed native FP32 RoPE.')
        actual = decoded_gate(fresh); require(torch.equal(actual, reference), 'Native BF16 decoded gate changed on reload.')
        require_state(fresh.state_dict(), expected, 'Reload native decode preserves persistent state')
        check(record); check_hashes(plan['input_sha256'])
        audit.update(status='verified', source_checkpoint_id=SOURCE_ID, checkpoint_id=final_meta['checkpoint_id'],
            endpoint_sha256=report['endpoint_sha256'], choices_sha256=report['choices_sha256'], **patch,
            serialized_tensors_compared=serialized, reloaded_tensors_compared=count,
            fresh_strict_reload_state_exact=True, endpoint_state_exact=True, frozen_outside_target_exact=True,
            native_full_weight_reload_exact=True, integer_budget_unchanged=True, rank_zero=True,
            native_full_weight=tensor_record(actual), native_rope_buffers=rope, bit_budget=budget,
            checkpoint_meta_sha256=sha256(saved['meta']), checkpoint_state_stat=file_stat(saved['state_dict']),
            maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
            maximum_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0), fit_report_sha256=sha256(report_path))
    except BaseException as error:
        audit.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()); raise
    finally:
        audit['finished_unix'] = time.time(); write_json(output / 'audit.json', audit)


def run(path):
    record, plan, output, report, report_path = runtime_inputs(path)
    require(not any((output / name).exists() for name in ('evaluation.json', 'final_model', 'audit.json',
            'mmlu.json', 'seven.json', 'full8_summary.json', 'evaluation.exit')), 'Fresh evaluation only.')
    progress = {'status': 'running', 'started_unix': time.time(), 'stages': {}, 'preparation': str(path)}; code = 1
    try:
        require([r['name'] for r in plan['commands']] == ['deploy', 'mmlu', 'seven'], 'Wrong native stage inventory.')
        for row in plan['commands']:
            name, command = row['name'], row['argv']
            require(Path(command[0]).resolve() == Path(sys.executable).resolve()
                    and '--lm_limit' not in command and '--limit' not in command, 'Keep complete fixed-protocol evaluation.')
            progress['stages'][name] = {'argv': command, 'started_unix': time.time()}
            write_json(output / 'evaluation.json', progress)
            try:
                status = run_logged_stage(command, name=name, output_dir=output, env=os.environ.copy(), allowed_exit_codes=(0,))
            finally:
                exit_path = output / (name + '.exit')
                if exit_path.exists(): progress['stages'][name]['exit_code'] = int(exit_path.read_text().strip())
                progress['stages'][name]['finished_unix'] = time.time(); write_json(output / 'evaluation.json', progress)
            require(status == 0, 'Native deployment/evaluation stage failed.')
        final, meta, budget = validate_checkpoint(output / 'final_model', plan['architecture']); audit = read(output / 'audit.json')
        require(audit['status'] == 'verified' and audit['checkpoint_id'] == meta['checkpoint_id']
                and audit['source_checkpoint_id'] == SOURCE_ID and audit['endpoint_sha256'] == report['endpoint_sha256']
                and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 3423
                and audit['frozen_outside_target_tensors'] == 3410 and audit['stage_changed_bits'] == report['stage_xor']
                and audit['exact_choices_replay'] and audit['codes_outside_selected_groups_exact']
                and audit['fresh_strict_reload_state_exact'] and audit['native_full_weight_reload_exact'],
                'Deployment proof is incomplete or belongs to another endpoint.')
        budget_equal(budget, plan['source_bit_budget']); paths = [output / (n + '.json') for n in ('mmlu', 'seven')]
        current = collect_results(paths, plan['architecture'], checkpoint=(final, meta))
        require(current['leaf_count'] == 64 and current['sample_count'] == 24742, 'Incomplete eight-task result.')
        comparisons = {}
        for name, control in plan['controls'].items():
            directory = Path(control['checkpoint']); control_meta = read(directory / 'checkpoint_meta.json')
            require(control_meta['checkpoint_id'] == control['checkpoint_id'], 'Control UUID changed: ' + name)
            previous = compare_control(paths, control['results'], plan['architecture'], checkpoint=(directory, control_meta))
            require(previous['mean_percent'] == control['mean_percent'], 'Control raw score changed: ' + name)
            comparisons[name] = {'checkpoint_id': control['checkpoint_id'], 'mean_percent': previous['mean_percent'],
                'mean_delta_pp': current['mean_percent'] - previous['mean_percent'],
                'task_delta_pp': {k: 100 * (v - previous['task_metrics'][k]) for k, v in current['task_metrics'].items()}}
        check(record); check_hashes(plan['input_sha256'])
        summary = {'status': 'full8_evaluation_verified', 'checkpoint_id': meta['checkpoint_id'],
            'checkpoint_dir': str(final), 'source_checkpoint_id': SOURCE_ID, **current, 'bit_budget': budget,
            'stage_changed_bits': audit['stage_changed_bits'], 'changed_code_bits': audit['changed_code_bits'],
            'comparisons': comparisons, 'primary_control': plan['primary_control'], 'protocol': plan['protocol'],
            'endpoint_sha256': report['endpoint_sha256'], 'fit_report_sha256': sha256(report_path),
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
