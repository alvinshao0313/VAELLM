"""CPU-only preparation of two fresh 2000-step general-LM decoder recoveries."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.general_lm_recovery.train import ROOT, SAVE_STEPS, check_inputs, configuration, read, training_arguments
from experiments.llama_ce_reproduction.callback import stage_commands
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.acceptance import collect_results, validate_checkpoint
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.validate_decoder_subset_training import expected_inventory
from train_utils.runtime_snapshot import to_runtime_jsonable

R1 = ROOT / 'result/qwen3_8b_w25_20260930'
R2 = ROOT / 'result/qwen3_8b_mmlu65_20261004'
RL = ROOT / 'result/llama31_8b_w25_20261004'
DATA = ROOT / 'data/fineweb_edu_10bt_20261005'
LL_CONTROL = RL / 'final_all_decoder_task50_s400'
Q_RESOURCE = R1 / 'e2e_mlpbest_ce50_s8000'
LL_PROOF = RL / 'exact_ce2000_reproduction/validate_ce10_s4'
SOURCES = {
    'qwen3': (R1 / 'e2e_best_combo_down0_joint_s2000/step_1200/final_model',
              'f927ffda-dd5b-4d00-b63a-5e839a182d70',
              R1 / 'e2e_best_combo_down0_joint_s2000/step_1200'),
    'llama': (RL / 'remaining_prefix_block32_s500/home_star_models_meta-llama_Llama-3.1-8B-Instruct_20261004_104649/final_model',
              '35776715-2707-45a9-9fb4-2da1efef3f0e', RL / 'remaining_prefix_block32_s500_full8'),
}


def data_contract():
    manifest_path, stats_path = DATA / 'manifest.json', DATA / 'canonical_statistics.json'
    manifest, stats = read(manifest_path), read(stats_path)
    require(manifest['status'] == 'ready_for_canonical_cpu_statistics'
            and manifest['repo'] == 'HuggingFaceFW/fineweb-edu' and manifest['config'] == 'sample-10BT'
            and manifest['split'] == 'train' and manifest['original_shard'] == 'sample/10BT/000_00000.parquet'
            and manifest['parquet_sha256'] == manifest['expected_sha256']
            and not manifest['packing'] and not manifest['local_filtering']
            and manifest['counts']['rows'] >= 8000
            and all(value == 0 for key, value in manifest['counts'].items() if key != 'rows'),
            'Require the complete, unfiltered official original shard and valid text in every row.')
    require(stats['status'] == 'verified' and not stats['cuda_initialized']
            and stats['manifest_sha256'] == sha256(manifest_path)
            and stats['jsonl_sha256'] == manifest['jsonl_sha256']
            and stats['same_raw_sampler_order_across_tokenizers']
            and stats['sampler']['sample_count'] == 8000 and stats['sampler']['data_seed'] == 0
            and stats['sampler']['epoch'] == 0 and stats['sampler']['world_size'] == 1
            and not stats['sampler']['group_by_length'] and stats['sampler']['native_trainer_use_seedable_sampler'],
            'Require the actual canonical two-tokenizer/8000-document seeded CPU observation.')
    train_file = Path(manifest['jsonl_path']).resolve()
    require(train_file == DATA / 'train.jsonl' and train_file.stat().st_size == manifest['jsonl_bytes'],
            'JSONL location or size differs from the completed conversion.')
    check_hashes(stats['source_sha256'])
    for model_type, item in stats['models'].items():
        require(model_type in SOURCES and item['source_rows'] == manifest['counts']['rows']
                and item['sampler_first8000']['summary']['documents'] == 8000
                and item['sampler_first8000']['summary']['batches'] == 2000
                and set(item['checkpoint_windows']) == set(map(str, SAVE_STEPS)),
                'The full fixed training budget was not covered by canonical CPU data statistics.')
        check_hashes(item['tokenizer_files_sha256'])
    require(set(stats['models']) == set(SOURCES), 'Both native model tokenizers are required.')
    return manifest, stats, train_file


def prepare(output_root):
    output_root = Path(output_root).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and os.environ.get('ACCELERATE_GRADIENT_ACCUMULATION_STEPS', '1') == '1', 'Prepare on CPU with CUDA hidden.')
    require(not output_root.exists(), 'Use a fresh preparation directory.')
    usage = shutil.disk_usage(output_root.parent)
    require(usage.free >= 60_000_000_000, 'Reserve at least 30GB per formal branch.')
    manifest, stats, train_file = data_contract()
    control, q_resource = read(LL_CONTROL / 'launch.json'), read(Q_RESOURCE / 'launch.json')
    proof, proof_review = read(LL_PROOF / 'training.json'), read(LL_PROOF / 'independent_review.json')
    require(control['status'] == q_resource['status'] == 'completed'
            and proof['status'] == proof_review['status'] == 'verified'
            and proof['callback_completed_steps'] == [2, 4] and proof['teacher_output_offload'] == 'none'
            and proof['source_checkpoint_id'] == SOURCES['llama'][1], 'Reusable native decoder/callback evidence is incomplete.')
    source_pins = {str((ROOT / p).resolve()): digest for p, digest in control['source_sha256'].items()
                   if not (ROOT / p).resolve().is_relative_to(ROOT / 'result')}
    for path, digest in stats['source_sha256'].items():
        require(path not in source_pins or source_pins[path] == digest, 'Data proof and validated runtime source differ: ' + path)
        source_pins[path] = digest
    additional = ['experiments/llama_ce_reproduction/train.py', 'experiments/llama_ce_reproduction/callback.py',
        'experiments/qwen3_w25/reproduction_recipe.py', 'experiments/longbench_v1/common.py',
        'experiments/long_context_recovery/contract.py', 'train_utils/runtime_snapshot.py',
        'train_utils/distill_losses.py']
    for path in [*(ROOT / p for p in additional), *sorted(Path(__file__).parent.glob('*.py'))]:
        ast.parse(path.read_text())
        value = sha256(path)
        require(str(path) not in source_pins or source_pins[str(path)] == value, 'Validated native source drifted: ' + str(path))
        source_pins[str(path)] = value
    evidence_paths = [LL_CONTROL / 'launch.json', LL_CONTROL / 'step_0400/audit.json',
        Q_RESOURCE / 'launch.json', LL_PROOF / 'training.json', LL_PROOF / 'independent_review.json',
        DATA / 'manifest.json', DATA / 'canonical_statistics.json']
    records = {}
    for model_type, (source, checkpoint_id, result_dir) in SOURCES.items():
        output = output_root / model_type
        architecture = model_inventory(read(source / 'checkpoint_meta.json')['base_model_path'])
        source, meta, budget = validate_checkpoint(source, architecture)
        require(architecture['model_type'] == model_type and meta['checkpoint_id'] == checkpoint_id
                and budget['budget_passed'] and budget['user_linear_bpw'] <= 2.5, 'Original source identity or budget differs.')
        inventory = expected_inventory(source, architecture, tuple(architecture['projection_shapes']))
        require(inventory['selected_projections'] == inventory['deployed_projections'] == architecture['projection_count']
                and inventory['selected_decoder_tensors'] == inventory['all_decoder_tensors']
                and inventory['selected_code_banks'] == inventory['active_score_elements'] == 0,
                'Only all native decoders may be selected.')
        protocol_paths = [result_dir / (name + '.json') for name in ('mmlu', 'seven')]
        original = collect_results(protocol_paths, architecture, checkpoint=(source, meta))
        hf_source = deepcopy((q_resource if model_type == 'qwen3' else control)['hf_source'])
        # Old Qwen records also carry the prior SHA inside each stat dictionary.
        # Keep that provenance separately; the reused checker accepts exact stat fields.
        original_weight_records = hf_source['weight_files']
        hf_source['weight_files'] = [{key: item[key] for key in ('path', 'size', 'mtime_ns', 'inode')}
                                     for item in original_weight_records]
        for item in hf_source['weight_files']:
            check_stat(item)
        hf_source['prior_weight_identity_records'] = original_weight_records
        require(Path(hf_source['path']).resolve() == Path(architecture['base_model_path'])
                and stats['models'][model_type]['tokenizer_path'] == architecture['base_model_path'],
                'Native teacher/tokenizer/HF model identity differs.')
        inputs = {str(path): sha256(path) for path in [*evidence_paths, source / 'checkpoint_meta.json', *protocol_paths]}
        record = {'schema': 'general_lm_all_decoder_2000_v1', 'status': 'prepared_unbound', 'kind': 'formal_s2000',
            'gpu': None, 'export_gpu': None, 'queued': False, 'automatic_retry': False,
            'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
            'source_checkpoint': str(source), 'source_checkpoint_id': checkpoint_id,
            'source_weight_stat': file_stat(source / 'pytorch_model.bin'), 'source_bit_budget': budget,
            'architecture': architecture, 'expected_inventory': inventory, 'hf_source': hf_source,
            'input_data_stat': [file_stat(train_file)], 'source_sha256': source_pins, 'input_evidence_sha256': inputs,
            'data': {'train_file': str(train_file), 'manifest': str(DATA / 'manifest.json'),
                'canonical_statistics': str(DATA / 'canonical_statistics.json'), 'jsonl_sha256': manifest['jsonl_sha256'],
                'rows': manifest['counts']['rows'], 'encoding': stats['encoding'], 'sampler': stats['sampler'],
                'expected_exposure': stats['models'][model_type]['sampler_first8000']['summary'],
                'checkpoint_windows': stats['models'][model_type]['checkpoint_windows'],
                'boundary': 'Native single-document plain LM; no packing/chat formatting/filtering. CPU sampler replay is not observed training exposure.'},
            'control_result_paths': [str(path) for path in protocol_paths], 'control_mean_percent': original['mean_percent'],
            'training_argv': training_arguments(source, architecture['base_model_path'], output, train_file),
            'argv': [sys.executable, '-u', '-m', 'experiments.general_lm_recovery.train', '--launch_record', str(output / 'launch.json')],
            'environment': {'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONPATH': str(ROOT),
                'PYTHONHASHSEED': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'WANDB_MODE': 'offline',
                'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'HF_HUB_OFFLINE': '1',
                'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'ACCELERATE_GRADIENT_ACCUMULATION_STEPS': '1'},
            'schedule': {'max_steps': 2000, 'warmup_steps': 20, 'scheduler': 'cosine', 'save_steps': 400,
                         'actual_stop_step': 2000, 'full8_steps': SAVE_STEPS},
            'resource_budget_seconds': 64800, 'termination_grace_seconds': 20,
            'validation_reuse': {'qwen_all_decoder_teacher_none': str(Q_RESOURCE / 'launch.json'),
                'llama_all_decoder_teacher_none_callback': str(LL_PROOF / 'independent_review.json'),
                'llama_full_export_audit': str(LL_CONTROL / 'step_0400/audit.json'),
                'scope': 'Existing all-decoder B4/max1024/DP/teacher-none and synchronous independent export/audit/evaluate paths; no repeated GPU short test.'},
            'readiness': {'cpu': 'pending', 'gpu': 'not_scheduled'},
            'protocol': {'mmlu_batch': 64, 'seven_batch': 32, 'num_fewshot': 0, 'limit': None,
                'leaf_count': 64, 'sample_count': 24742, 'longbench': 'not_evaluated', 'nine_item_goal': 'not_evaluated'},
            'estimate': {'storage_bytes_per_branch': 30_000_000_000, 'disk_free_at_prepare': usage.free,
                'training_hours_estimate': [10, 15], 'new_plain_lm_throughput': 'unmeasured'},
            'experience': {'reference': 'docs/experience/lessons/kd_and_data.md',
                'difference': 'General web text and LR3e-6; not prior task50/ChatML or MMLU multiple-choice LM. Full2000 avoids treating first400 as convergence.',
                'claim': 'No quality improvement is assumed; frozen code/norm/head and unchanged persistent bit budget.'}}
        cfg, hf, args, selected, actual_architecture = configuration(record)
        canonical = to_runtime_jsonable(cfg)
        require(canonical['data'] == stats['data_config'], 'Actual canonical CLI data config differs from the CPU sampler observation.')
        record['cpu_configuration'] = {'canonical_config': canonical, 'hf_args': to_runtime_jsonable(hf),
            'training_args': to_runtime_jsonable(args), 'all_projection_count': len(selected),
            'model_type': actual_architecture['model_type'], 'native_data_config_matches_statistics': True}
        checkpoint = output / 'NATIVE_RUN/trainer_state/checkpoint-400'
        record['stage_templates'] = [{'name': name, 'argv': argv, 'allowed_exit_codes': allowed}
            for name, argv, allowed in stage_commands(checkpoint=checkpoint, source_dir=source, output_dir=output / 'step_0400')]
        check_inputs(record)
        require(not torch.cuda.is_initialized(), 'CPU preparation unexpectedly initialized CUDA.')
        record.update(cuda_initialized=False)
        record['readiness']['cpu'] = 'passed'
        records[model_type] = record
    output_root.mkdir(parents=True)
    for name, record in records.items():
        (output_root / name).mkdir()
        write_json(output_root / name / 'launch.json', record)
    summary = {'status': 'cpu_prepared_unbound', 'cuda_initialized': False, 'data_rows': manifest['counts']['rows'],
        'data_manifest_sha256': sha256(DATA / 'manifest.json'), 'data_statistics_sha256': sha256(DATA / 'canonical_statistics.json'),
        'source_pin_count': len(source_pins), 'launch_sha256': {name: sha256(output_root / name / 'launch.json') for name in records},
        'source_bpw': {name: record['source_bit_budget']['user_linear_bpw'] for name, record in records.items()},
        'disk_free_bytes': usage.free, 'scope': 'Full2000/every400 complete full8; existing native method, no GPU execution.'}
    write_json(output_root / 'preparation.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=R2 / 'general_lm_recovery_s2000')
    prepare(parser.parse_args().output_root)
