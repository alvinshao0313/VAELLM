"""CPU preparation of the two exact CE2000 recipes and one save2/stop4 diagnostic."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_ce_reproduction.callback import stage_commands
from experiments.llama_ce_reproduction.train import ROOT, check_inputs, configuration, read, training_arguments
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.acceptance import collect_results
from experiments.w25_eval.validate_decoder_subset_training import expected_inventory
from train_utils.runtime_snapshot import to_runtime_jsonable

RL = ROOT / 'result/llama31_8b_w25_20261004'
QROOT = ROOT / 'result/qwen3_8b_w25_20260930'
CONTROL = RL / 'final_all_decoder_task50_s400'
Q_RUNS = {'ce10': 'e2e_decoder_remaining_prefix_ce10_lr1e5',
          'ce50': 'e2e_decoder_remaining_prefix_ce50_task100_resp_lr1e5'}


def science_equal(candidate, original):
    """Compare the saved scientific sections; textual .500/.5 ratios are equivalent."""
    for name in ('loss', 'opt', 'runtime', 'aux', 'lora'):
        require(candidate[name] == original[name], 'Original successful recipe differs in ' + name)
    left, right = deepcopy(candidate['data']), deepcopy(original['data'])
    left.pop('dataset_mix'); right.pop('dataset_mix')
    require(left == right, 'Actual normalized sampling/tokenization recipe differs.')
    require(candidate['train_mode'] == original['train_mode'] == 'decoder', 'Wrong original trainable mode.')


def prepare(output_root):
    output_root = Path(output_root).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('ACCELERATE_GRADIENT_ACCUMULATION_STEPS', '1') == '1',
            'CPU-only GAS1 preparation is required.')
    require(not output_root.exists(), 'Do not overwrite a prepared run root.')
    usage = shutil.disk_usage(output_root.parent)
    require(usage.free >= 2 * 30_000_000_000, 'Reserve at least 30GB per formal branch.')
    control = read(CONTROL / 'launch.json')
    require(control['status'] == 'completed' and control['exit_code'] == 0
            and control['source_checkpoint_id'] == '35776715-2707-45a9-9fb4-2da1efef3f0e',
            'Require the completed native Llama decoder control from original CAT3577.')
    source = Path(control['source_checkpoint'])
    source_meta = read(source / 'checkpoint_meta.json')
    architecture = control['architecture']
    names = tuple(architecture['projection_shapes'])
    inventory = expected_inventory(source, architecture, names)
    audit = read(CONTROL / 'step_0400/audit.json')
    budget = audit['source_bit_budget']
    require(audit['status'] == 'verified' and budget['budget_passed'] and budget['user_linear_bpw'] <= 2.5
            and inventory['deployed_projections'] == 224 and inventory['all_decoder_tensors'] == 1344
            and inventory['all_code_banks'] == 448 and inventory['serialized_tensors'] == 2979,
            'Actual Llama source inventory or previous generic export evidence differs.')
    protocol_paths = [RL / 'remaining_prefix_block32_s500_full8' / (name + '.json') for name in ('mmlu', 'seven')]
    original = collect_results(protocol_paths, architecture, checkpoint=(source, source_meta))
    sources = {str((ROOT / p).resolve()): h for p, h in control['source_sha256'].items()}
    additional = ['experiments/qwen3_w25/reproduction_recipe.py', 'experiments/qwen3_w25/step_export_callback.py',
        'experiments/qwen3_w25/aux_step_export_callback.py', 'experiments/qwen3_w25/subset_step_export_callback.py',
        'experiments/w25_eval/validate_decoder_subset_training.py', 'experiments/w25_eval/validate_wide_subset_training.py',
        'experiments/w25_eval/acceptance.py', 'experiments/w25_eval/subset_state.py',
        'experiments/w25_eval/subset_export.py', 'experiments/w25_eval/subset_audit.py',
        'experiments/longbench_v1/common.py', 'experiments/long_context_recovery/contract.py']
    for path in [*(ROOT / p for p in additional), *sorted(Path(__file__).parent.glob('*.py'))]:
        ast.parse(path.read_text())
        value = sha256(path)
        require(str(path) not in sources or sources[str(path)] == value, 'Previously validated source drifted: ' + str(path))
        sources[str(path)] = value
    inputs = {str((ROOT / p).resolve()): h for p, h in control['input_evidence_sha256'].items()}
    proof = RL / 'final_all_decoder_validate_s4/verification.json'
    for path in [CONTROL / 'launch.json', CONTROL / 'step_0400/audit.json', proof, *protocol_paths]:
        inputs[str(path)] = sha256(path)
    history = {}
    for recipe, name in Q_RUNS.items():
        directory = QROOT / name
        snapshots = list(directory.glob('*/normalized_e2e_runtime_args.json'))
        require(len(snapshots) == 1, 'Expected one actual original recipe snapshot.')
        snapshot = read(snapshots[0])
        launch = read(directory / 'launch.json')
        require(launch['status'] == 'completed', 'Original recipe did not finish.')
        history[recipe] = {'directory': str(directory), 'snapshot': str(snapshots[0]),
                          'canonical_config': snapshot['canonical_config']}
        for path in (snapshots[0], directory / 'launch.json'):
            inputs[str(path)] = sha256(path)
    records = {}
    for name, recipe, diagnostic in [('ce10_task50', 'ce10', False), ('ce50_task100_prompt0', 'ce50', False),
                                     ('validate_ce10_s4', 'ce10', True)]:
        output = output_root / name
        record = {'schema': 'model_adaptive_exact_ce2000_v1', 'status': 'prepared_unbound',
            'kind': 'validate_s4' if diagnostic else 'formal_s2000', 'recipe': recipe,
            'gpu': None, 'export_gpu': None, 'queued': False, 'automatic_retry': False,
            'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
            'source_checkpoint': str(source), 'source_checkpoint_id': source_meta['checkpoint_id'],
            'source_weight_stat': control['source_weight_stat'], 'source_bit_budget': budget,
            'architecture': architecture, 'expected_inventory': inventory,
            'hf_source': control['hf_source'], 'input_data_stat': control['input_data_stat'],
            'source_sha256': sources, 'input_evidence_sha256': inputs,
            'control_result_paths': [str(p) for p in protocol_paths], 'control_mean_percent': original['mean_percent'],
            'training_argv': training_arguments(recipe, source, architecture['base_model_path'], output, diagnostic=diagnostic),
            'argv': [sys.executable, '-u', '-m', 'experiments.llama_ce_reproduction.train', '--launch_record', str(output / 'launch.json')],
            'environment': {'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONPATH': str(ROOT),
                'PYTHONHASHSEED': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'WANDB_MODE': 'offline',
                'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'HF_HUB_OFFLINE': '1',
                'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'ACCELERATE_GRADIENT_ACCUMULATION_STEPS': '1'},
            'schedule': {'max_steps': 2000, 'warmup_steps': 20, 'scheduler': 'cosine',
                         'save_steps': 2 if diagnostic else 400, 'actual_stop_step': 4 if diagnostic else 2000},
            'original_qwen_recipe': {'directory': history[recipe]['directory'], 'snapshot': history[recipe]['snapshot']},
            'validation_reuse': {'native_llama_export_proof': str(proof), 'sha256': sha256(proof),
                'scope': 'Existing actual decoder/frozen/storage/native-final path; new callback pause/resume and teacher_output_none resource remain to validate.'},
            'readiness': {'cpu': 'pending', 'gpu': 'not_scheduled', 'short_validation':
                          {'path': str(output_root / 'validate_ce10_s4/training.json'), 'sha256': None}},
            'protocol': {'mmlu_batch': 64, 'seven_batch': 32, 'num_fewshot': 0, 'limit': None,
                'diagnostic_full8': False, 'longbench': 'not_evaluated', 'nine_item_goal': 'not_evaluated'},
            'estimate': {'formal_storage_bytes': 30_000_000_000, 'disk_free_at_prepare': usage.free,
                'old_offload_cpu_400_train_minutes': 103, 'five_full8_postprocess_minutes': 40,
                'formal_hours_old_throughput_estimate': 9.3, 'actual_none_task100_throughput': 'unmeasured'}}
        cfg, hf, args, selected, actual_architecture = configuration(record)
        canonical = to_runtime_jsonable(cfg)
        science_equal(canonical, history[recipe]['canonical_config'])
        require(actual_architecture == architecture and tuple(selected) == names, 'Actual parser/model selection differs.')
        record['cpu_configuration'] = {'canonical_config': canonical, 'hf_args': to_runtime_jsonable(hf),
            'training_args': to_runtime_jsonable(args), 'original_scientific_sections_match': True,
            'allowed_model_adaptations': ['student/teacher/output paths', 'target_layers all resolves from native model inventory'],
            'diagnostic_adaptations': ['save2; observer stop4, original2000 horizon'] if diagnostic else []}
        checkpoint = output / 'NATIVE_RUN/trainer_state' / ('checkpoint-2' if diagnostic else 'checkpoint-400')
        record['stage_templates'] = [{'name': name, 'argv': argv, 'allowed_exit_codes': allowed}
            for name, argv, allowed in stage_commands(checkpoint=checkpoint, source_dir=source,
                output_dir=output / ('step_0002' if diagnostic else 'step_0400'), diagnostic=diagnostic)]
        check_inputs(record)
        record['readiness']['cpu'] = 'passed'
        record['cuda_initialized'] = torch.cuda.is_initialized()
        require(not record['cuda_initialized'], 'CPU preparation must not initialize CUDA.')
        records[name] = record
    output_root.mkdir(parents=True)
    for name, record in records.items():
        output = output_root / name
        output.mkdir()
        write_json(output / 'launch.json', record)
    summary = {'status': 'cpu_prepared_unbound', 'cuda_initialized': False,
        'source_checkpoint_id': source_meta['checkpoint_id'], 'source_bpw': budget['user_linear_bpw'],
        'expected_inventory': inventory, 'source_pin_count': len(sources), 'input_pin_count': len(inputs),
        'launch_sha256': {name: sha256(output_root / name / 'launch.json') for name in records},
        'disk_free_bytes': usage.free, 'dual_formal_requested_storage_bytes': 60_000_000_000,
        'scope': 'Existing methods; exact original CE2000 recipes with model-adaptive export, no GPU execution.'}
    write_json(output_root / 'preparation.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=RL / 'exact_ce2000_reproduction')
    prepare(parser.parse_args().output_root)
