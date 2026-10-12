"""CPU prepare only the four-update13bd compatibility/resource validation."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.longbench_v1.common import check_hashes, file_stat, sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.llama_s4_lm_validation.config import CONTROL, OUTPUT, check_inputs, configuration, read, training_arguments
from experiments.llama_s4_lm_validation.state import PUBLIC, R2, ROOT, SOURCE, SOURCE_ID, TOTAL_BITS, expected_inventory
from experiments.w25_eval.model_inventory import model_inventory
from train_utils.runtime_snapshot import to_runtime_jsonable


def prepare():
    require(Path.cwd() == ROOT and Path(__file__).resolve().is_relative_to(ROOT)
            and os.environ.get('PYTHONPATH') == str(ROOT) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized() and not OUTPUT.exists(), 'Fresh CPU-only bitvae preparation in S.')
    control = read(CONTROL); meta = read(SOURCE / 'checkpoint_meta.json')
    audit_path = SOURCE.parent / 'audit.json'; audit = read(audit_path)
    terminal = read(SOURCE.parent / 'launch.json'); review = read(SOURCE.parent / 'independent_review.json')
    summary = read(SOURCE.parent / 'full8_summary.json')
    regression = PUBLIC / 'result/qwen3_8b_w25_20260930/validation/s4_decoder_regression/launch.json'
    cpu = PUBLIC / 'result/qwen3_8b_w25_20260930/validation/s4_packed_v6_cpu/launch.json'
    gpu_proof, cpu_proof = read(regression)['gpu_stage'], read(cpu)
    require(control['status'] == 'completed' and control['source_checkpoint_id'] == 'e767c232-56fd-4b03-a4ae-dd0781b6e047'
            and meta['checkpoint_id'] == audit['checkpoint_id'] == terminal['checkpoint_id'] == review['checkpoint_id']
            == summary['checkpoint_id'] == SOURCE_ID and audit['status'] == 'verified'
            and review['status'] == 'independent_cpu_verified' and terminal['status'] == 'completed' and terminal['exit_code'] == 0
            and audit['source_union_save_fresh_tensors'] == 3043 and audit['full_code_bank_count'] == 512
            and audit['gate_code_bank_count'] == 128 and audit['replaced_projection_count'] == 32
            and audit['preserved_projection_count'] == 192 and gpu_proof['status'] == cpu_proof['status'] == 'passed'
            and gpu_proof['exit_code'] == cpu_proof['exit_code'] == 0, 'Require completed S4 native and numeric evidence.')
    require(all((SOURCE.parent / (s + '.exit')).read_text().strip() == '0'
                for s in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer'))
            and terminal['full8_summary_sha256'] == sha256(SOURCE.parent / 'full8_summary.json')
            and summary['audit_sha256'] == sha256(audit_path)
            and audit['assembly_report_sha256'] == sha256(SOURCE / 'assembly_report.json'), 'Source terminal chain differs.')
    sources, evidence = {}, {str(CONTROL): sha256(CONTROL)}
    for raw, digest in control['source_sha256'].items():
        original = Path(raw); mapped = ROOT / original.relative_to(PUBLIC) if original.is_relative_to(PUBLIC) else original
        if mapped.exists():
            require(sha256(mapped) == digest, 'Original native LM runtime differs in S: ' + str(mapped))
            sources[str(mapped)] = digest
        else:
            require(sha256(original) == digest, 'Original preparation evidence drifted: ' + str(original))
            evidence[str(original)] = digest
    pending, seen = list(Path(__file__).parent.glob('*.py')), set()
    for path in pending:
        require(sha256(path) == sha256(PUBLIC / path.relative_to(ROOT)), 'New public/S entry differs.')
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        seen.add(path); tree = ast.parse(path.read_text()); sources[str(path)] = sha256(path)
        for node in ast.walk(tree):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else
                       [i.name for i in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                local = ROOT / (module.replace('.', '/') + '.py')
                if local.is_file():
                    pending.append(local)
    inputs = [audit_path, SOURCE / 'checkpoint_meta.json', SOURCE / 'config.json', SOURCE / 'assembly_report.json',
        SOURCE.parent / 'independent_review.json', SOURCE.parent / 'launch.json', SOURCE.parent / 'full8_summary.json',
        regression, cpu, regression.parent / 'gpu.exit', cpu.parent / 'run.exit',
        R2 / 'llama_k34_lm_validate_s4/verification.json',
        *[SOURCE.parent / (s + '.exit') for s in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')],
        Path(control['data']['manifest']), Path(control['data']['canonical_statistics'])]
    evidence.update({str(p): sha256(p) for p in inputs})
    check_hashes(sources); check_hashes(evidence)
    architecture = model_inventory(meta['base_model_path']); inventory = expected_inventory(architecture)
    budget = audit['bit_budget']
    require(budget == summary['bit_budget'] == review['bit_budget'] and budget['budget_passed']
            and budget['user_linear_bits'] == budget['packed_linear_bits'] == TOTAL_BITS
            and budget['code_padding_bits'] == 0 and budget['original_linear_weights'] == 6979321856
            and budget['components']['decoder']['budget_bits'] == inventory['selected_decoder_elements'] * 16,
            'Actual source inventory and storage differ.')
    environment = deepcopy(control['environment']); environment['PYTHONPATH'] = str(ROOT)
    free = shutil.disk_usage(OUTPUT.parent).free
    require(free >= 20_000_000_000, 'Reserve20GB for native final, independent export and retained step4 state.')
    record = {'schema': 'llama_gate_s4_all_decoder_lm_four_updates_v1', 'kind': 'validate_s4', 'status': 'prepared_unbound',
        'gpu': None, 'queued': False, 'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable,
        'output_dir': str(OUTPUT), 'source_checkpoint': str(SOURCE), 'source_checkpoint_id': SOURCE_ID,
        'source_weight_stat': file_stat(SOURCE / meta['state_dict_file']), 'source_bit_budget': budget,
        'control_launch': str(CONTROL), 'architecture': architecture, 'expected_inventory': inventory,
        'selected_targets': list(architecture['projection_shapes']), 'hf_source': deepcopy(control['hf_source']),
        'input_data_stat': deepcopy(control['input_data_stat']), 'data': deepcopy(control['data']),
        'source_sha256': sources, 'input_evidence_sha256': evidence, 'training_argv': training_arguments(OUTPUT),
        'argv': [sys.executable, '-u', '-m', 'experiments.llama_s4_lm_validation.train', '--launch_record', str(OUTPUT / 'launch.json')],
        'dispatch_argv': [sys.executable, '-u', '-m', 'experiments.llama_s4_lm_validation.run', '--launch_record', str(OUTPUT / 'launch.json')],
        'environment': environment, 'schedule': {'max_steps': 2000, 'warmup_steps': 20, 'scheduler': 'cosine',
            'save_steps': 2, 'save_total_limit': 1, 'actual_stop_step': 4, 'independent_export_steps': [4], 'full8_steps': []},
        'resource_budget_seconds': 1200, 'termination_grace_seconds': 20,
        'resource_cap_enforcement': 'Root wraps dispatch_argv with1200+20; no internal watchdog/retry.',
        'estimate': {'elapsed_seconds': [200, 500], 'hard_seconds': 1200, 'training_peak_gib_estimate': [38, 65],
            'disk_bytes_reserved': 20000000000, 'disk_free_at_prepare': free,
            'boundary': 'Mixed32S4/192S2 FP32 packed Adam resource has not been measured. S4 hidden stage scratch doubles on gate; extra decoder parameter bytes alone cannot bound peak. Export follows training process exit.'},
        'readiness': {'cpu': 'pending', 'gpu': 'not_scheduled', 'formal': 'not_prepared'},
        'scientific_boundary': 'Only source/output/save2 differ from original e767 FineWeb recipe. Stop at4 retains max2000/warm20. Existing native mixed-stage algorithm, no active code modifications; compatibility/resource evidence only, not quality or exact resume.',
        'validation_reuse': {'s4_bf16_grad_group8': str(regression), 's4_cpu_sgd_save_reload': str(cpu),
            'actual_s4_cat_full8': str(SOURCE.parent / 'independent_review.json'), 'original_fineweb_recipe': str(CONTROL),
            'single_gpu_exit_before_export': str(R2 / 'llama_k34_lm_validate_s4/verification.json'),
            'scope': 'No repeat CAT, S4 local parity or full8. New real path is mixed-stage FP32 packed Adam plus native selected-step independent export.'}}
    cfg, hf, args, names, _ = configuration(record); record['selected_targets'] = list(names)
    record['cpu_configuration'] = {'canonical_config': to_runtime_jsonable(cfg), 'hf_args': to_runtime_jsonable(hf),
        'training_args': to_runtime_jsonable(args), 'canonical_difference_from_control': ['student_checkpoint_dir', 'run_root_dir', 'remaining_argv/save_steps'],
        'native_data_config_matches_control': to_runtime_jsonable(cfg.data) == control['cpu_configuration']['canonical_config']['data']}
    check_inputs(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['readiness']['cpu'] = 'passed'; record['cuda_initialized'] = False
    OUTPUT.mkdir(); write_json(OUTPUT / 'launch.prepared.json', record)
    print({'status': 'prepared_unbound', 'sha256': sha256(OUTPUT / 'launch.prepared.json'),
        'sources': len(sources), 'evidence': len(evidence), 'inventory': inventory, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    prepare()
