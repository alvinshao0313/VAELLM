"""CPU-only preparation; this entry never prepares or launches a formal recovery."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.longbench_v1.common import check_hashes, file_stat, sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.llama_k34_lm_validation.config import (
    CONTROL, OUTPUT, PUBLIC, ROOT, R2, SOURCE, SOURCE_ID, check_inputs, configuration, read, training_arguments)
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.validate_decoder_subset_training import expected_inventory
from train_utils.runtime_snapshot import to_runtime_jsonable


def prepare():
    require(Path.cwd() == ROOT and Path(__file__).resolve().is_relative_to(ROOT)
            and os.environ.get('PYTHONPATH') == str(ROOT) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized(), 'Prepare with CPU-only bitvae and actual S cwd/PYTHONPATH.')
    require(not OUTPUT.exists(), 'Never overwrite an existing preparation.')
    control = read(CONTROL)
    audit_path = SOURCE.parent / 'audit.json'
    audit, meta = read(audit_path), read(SOURCE / 'checkpoint_meta.json')
    terminal = read(SOURCE.parent / 'launch.json')
    review = read(SOURCE.parent / 'independent_review.json')
    summary = read(SOURCE.parent / 'full8_summary.json')
    packing_proof_path = R2 / 'llama_gate_small_rate_pair/validate_k34_s2/verification.json'
    packing_proof = read(packing_proof_path)
    require(control['status'] == 'completed' and control['source_checkpoint_id'] == 'e767c232-56fd-4b03-a4ae-dd0781b6e047'
            and audit['status'] == 'verified' and audit['checkpoint_id'] == meta['checkpoint_id'] == SOURCE_ID
            and terminal['status'] == 'completed' and terminal['exit_code'] == 0
            and review['status'] == 'independent_cpu_verified'
            and terminal['checkpoint_id'] == review['checkpoint_id'] == summary['checkpoint_id'] == SOURCE_ID
            and audit['source_union_save_fresh_tensors'] == 2979 and audit['replaced_projection_count'] == 32
            and audit['preserved_projection_count'] == 192 and audit['actual_gate_transpose'] is True
            and audit['zero_padding_banks'] == 64
            and packing_proof['status'] == 'verified_native_short'
            and packing_proof['zero_padding_and_pack_replay_banks'] == 64,
            'Require actual completed K34/True, native union and existing packing evidence.')
    require(all((SOURCE.parent / (stage + '.exit')).read_text().strip() == '0'
                for stage in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer'))
            and terminal['full8_summary_sha256'] == sha256(SOURCE.parent / 'full8_summary.json')
            and summary['audit_sha256'] == sha256(audit_path)
            and audit['assembly_report_sha256'] == sha256(SOURCE / 'assembly_report.json'),
            'Source terminal metadata chain is incomplete.')
    sources, evidence = {}, {str(CONTROL): sha256(CONTROL)}
    for raw, digest in control['source_sha256'].items():
        original = Path(raw)
        mapped = ROOT / original.relative_to(PUBLIC) if original.is_relative_to(PUBLIC) else original
        # Missing old preparation-only sources are evidence, never imported via public fallback.
        if mapped.exists():
            require(sha256(mapped) == digest, 'Original LM runtime differs in S: ' + str(mapped))
            sources[str(mapped)] = digest
        else:
            require(sha256(original) == digest, 'Original source evidence drifted: ' + str(original))
            evidence[str(original)] = digest
    seeds = [*Path(__file__).parent.glob('*.py'), ROOT / 'experiments/w25_eval/subset_export.py',
        ROOT / 'experiments/w25_eval/subset_audit.py', ROOT / 'rotation/hadamard_utils.py',
        ROOT / 'rotation/hadamard_extended.py']
    for path in Path(__file__).parent.glob('*.py'):
        require(sha256(path) == sha256(PUBLIC / path.relative_to(ROOT)), 'New public/S entry differs.')
    pending, seen = list(seeds), set()
    # Pin local direct imports of the thin entries as well as the original complete native inventory.
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text())
        sources[str(path)] = sha256(path)
        for node in ast.walk(tree):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else
                       [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                local = ROOT / (module.replace('.', '/') + '.py')
                if local.is_file():
                    pending.append(local)
    inputs = [audit_path, SOURCE / 'checkpoint_meta.json', SOURCE / 'config.json',
        SOURCE / 'assembly_report.json', SOURCE.parent / 'independent_review.json',
        SOURCE.parent / 'launch.json', SOURCE.parent / 'full8_summary.json', packing_proof_path,
        R2 / 'qwen_fullqk_lm_validate_s4/verification.json',
        *[SOURCE.parent / (stage + '.exit') for stage in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')],
        Path(control['data']['manifest']), Path(control['data']['canonical_statistics'])]
    evidence.update({str(path): sha256(path) for path in inputs})
    check_hashes(sources)
    check_hashes(evidence)
    architecture = model_inventory(meta['base_model_path'])
    inventory = expected_inventory(SOURCE, architecture, tuple(architecture['projection_shapes']))
    expected = deepcopy(control['expected_inventory'])
    expected['selected_decoder_elements'] += 16384
    require(inventory == expected and inventory['serialized_tensors'] == 2979
            and inventory['all_decoder_tensors'] == 1344 and inventory['frozen_tensors'] == 1635
            and inventory['selected_decoder_elements'] == 3806208,
            'Actual source metadata inventory differs from the complete native LM scope.')
    budget = audit['bit_budget']
    require(budget == summary['bit_budget'] == review['bit_budget']
            and budget['user_linear_bits'] == 17326653440 and budget['packed_linear_bits'] == 18020286464
            and budget['code_padding_bits'] == 693633024
            and budget['original_linear_weights'] == 6979321856 and budget['budget_passed'], 'Wrong native source budget.')
    environment = deepcopy(control['environment'])
    environment['PYTHONPATH'] = str(ROOT)
    free = shutil.disk_usage(OUTPUT.parent).free
    require(free >= 20_000_000_000, 'Reserve20GB for native final, independent export and the retained step4 state.')
    record = {'schema': 'llama_k34true_all_decoder_lm_s4_v1', 'kind': 'validate_s4', 'status': 'prepared_unbound',
        'gpu': None, 'queued': False, 'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable,
        'output_dir': str(OUTPUT), 'source_checkpoint': str(SOURCE), 'source_checkpoint_id': SOURCE_ID,
        'source_weight_stat': file_stat(SOURCE / 'pytorch_model.bin'), 'source_bit_budget': budget,
        'control_launch': str(CONTROL), 'architecture': architecture, 'expected_inventory': inventory,
        'selected_targets': list(architecture['projection_shapes']), 'hf_source': deepcopy(control['hf_source']),
        'input_data_stat': deepcopy(control['input_data_stat']), 'data': deepcopy(control['data']),
        'source_sha256': sources, 'input_evidence_sha256': evidence,
        'training_argv': training_arguments(OUTPUT),
        'argv': [sys.executable, '-u', '-m', 'experiments.llama_k34_lm_validation.train',
                 '--launch_record', str(OUTPUT / 'launch.json')],
        'dispatch_argv': [sys.executable, '-u', '-m', 'experiments.llama_k34_lm_validation.run',
                 '--launch_record', str(OUTPUT / 'launch.json')], 'environment': environment,
        'schedule': {'max_steps': 2000, 'warmup_steps': 20, 'save_steps': 2, 'save_total_limit': 1,
            'actual_stop_step': 4, 'independent_export_steps': [4], 'full8_steps': []},
        'resource_budget_seconds': 1200, 'termination_grace_seconds': 20,
        'resource_cap_enforcement': 'Root must wrap dispatch_argv with the recorded timeout; no internal watchdog/retry.',
        'estimate': {'elapsed_seconds': [180, 400], 'hard_seconds': 1200,
            'training_peak_gib_estimate': [35, 55], 'disk_bytes_reserved': 20000000000, 'disk_free_at_prepare': free,
            'boundary': 'K34 FP32 packed backward has not been measured. Same real Llama all-D B4/max1024; extra16384 decoder parameters and115638272 source tensor bytes versus e767. Export starts only after training exits.'},
        'readiness': {'cpu': 'pending', 'gpu': 'not_scheduled', 'formal': 'not_prepared'},
        'scientific_boundary': '98a8 replaces e767 gate with fresh K34/transposeTrue CAT. It changes gate codes/decoder/geometry and recovery history. Four updates establish execution, not quality. Six padding bits per K34 vector remain immutable; logical and physical budgets are separate.',
        'scope': 'Existing FineWeb all-decoder LM recipe; new observation only. No new algorithm, dataset, dependency, or modifications to pre-existing public/S sources.',
        'validation_reuse': {'existing_k34_native_forward': str(packing_proof_path),
            'existing_llama_e2e_recipe': str(CONTROL),
            'existing_one_gpu_train_exit_export_audit': str(R2 / 'qwen_fullqk_lm_validate_s4/verification.json'),
            'scope': 'Existing native save2/4 and periodic callback reused. Only K34/True64-stage FP32 packed backward/update and actual all-D resource coverage are new. No repeated CAT/group8 inference, full8 or two-GPU callback test.'}}
    cfg, hf, args, names, _ = configuration(record)
    record['selected_targets'] = list(names)
    record['cpu_configuration'] = {'canonical_config': to_runtime_jsonable(cfg), 'hf_args': to_runtime_jsonable(hf),
        'training_args': to_runtime_jsonable(args), 'canonical_difference_from_control': ['student_checkpoint_dir', 'run_root_dir', 'remaining_argv/save_steps'],
        'native_data_config_matches_control': to_runtime_jsonable(cfg.data) == control['cpu_configuration']['canonical_config']['data']}
    check_inputs(record)
    require(not torch.cuda.is_initialized(), 'CPU preparation unexpectedly initialized CUDA.')
    record.update(cuda_initialized=False)
    record['readiness']['cpu'] = 'passed'
    OUTPUT.mkdir()
    write_json(OUTPUT / 'launch.prepared.json', record)
    print({'status': 'prepared_unbound', 'path': str(OUTPUT / 'launch.prepared.json'),
        'sha256': sha256(OUTPUT / 'launch.prepared.json'), 'sources': len(sources), 'evidence': len(evidence),
        'inventory': inventory, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    prepare()
