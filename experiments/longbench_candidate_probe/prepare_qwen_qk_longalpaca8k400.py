"""Bind the completed QK-only LongAlpaca400 endpoint to official EN/ZH retrieval400."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import torch

from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config, make_config
from experiments.w25_eval.acceptance import TASK_METRICS, collect_results

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / 'result/qwen3_8b_mmlu65_20261004'
TRAIN = R2 / 'qwen_qk_longalpaca8k_s400'
OV = R2 / 'best1200_ov72_longalpaca8k_s400'
ROTATION = R2 / 'qwen_qk_rotation_retrieval400/full'
OUTPUT = R2 / 'qwen_qk_longalpaca8k_retrieval400'
SOURCE_ID = 'f927ffda-dd5b-4d00-b63a-5e839a182d70'
TASKS = ('passage_retrieval_en', 'passage_retrieval_zh')


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def retired(pid, ticks):
    path = Path(f'/proc/{pid}/stat')
    return not path.exists() or int(path.read_text().rsplit(')', 1)[1].split()[19]) != ticks


def full8(paths, checkpoint, architecture):
    meta = read(checkpoint / 'checkpoint_meta.json')
    result = collect_results(paths, architecture, checkpoint=(checkpoint, meta))
    counts = {}
    for path in paths:
        raw = read(path)['evals']['lm_eval']
        for task, sizes in raw['n_samples'].items():
            metric = 'acc,none' if task.startswith('mmlu_') else TASK_METRICS[task]
            total = sizes['effective']; correct = raw['raw_results'][task][metric] * total
            require(math.isclose(correct, round(correct), rel_tol=0, abs_tol=1e-8), 'Noninteger correct count.')
            counts[task] = {'correct': round(correct), 'total': total}
    subjects = [name for name in counts if name.startswith('mmlu_')]
    require(len(counts) == 64 and len(subjects) == 57 and sum(x['total'] for x in counts.values()) == 24742,
            'Require all64 leaves/24742 examples.')
    counts['mmlu'] = {'correct': sum(counts[name]['correct'] for name in subjects), 'total': 14042}
    result.update(checkpoint_id=meta['checkpoint_id'], integer_counts=counts)
    return result


def completed_candidate():
    launch = read(TRAIN / 'launch.json'); process = read(TRAIN / 'process.json')
    require(launch['status'] == 'completed' and launch['dispatcher_exit_code'] == 0
            and launch['source_checkpoint_id'] == SOURCE_ID and launch['mode'] == 'formal',
            'Wait for natural native training, export and full8 completion; no partial preparation.')
    exits = {name: int((TRAIN / (name + '.exit')).read_text()) for name in ('training', 'dispatcher', 'outer')}
    step = TRAIN / 'step_0400'; checkpoint = step / 'final_model'
    exits.update({name: int((step / (name + '.exit')).read_text())
                  for name in ('export', 'audit', 'mmlu', 'seven', 'summary')})
    require(all(exits[name] == 0 for name in exits if name != 'summary') and exits['summary'] in (0, 1)
            and all(launch['stages'][name]['exit_code'] == exits[name]
                    for name in ('export', 'audit', 'mmlu', 'seven', 'summary')),
            'Summary1 is a valid below-target score; all computational stages must exit0.')
    require(retired(process['outer_pid'], process['outer_start_ticks'])
            and retired(launch['worker_pid'], launch['worker_start_ticks'])
            and retired(launch['training_pid'], launch['training_start_ticks'])
            and process['bound_sha256'] == sha256(TRAIN / 'launch.bound.json')
            and process['script_sha256'] == sha256(TRAIN / 'run.sh'), 'Source process identity or terminal state differs.')
    audit, acceptance, trained = (read(path) for path in (step / 'audit.json', step / 'acceptance.json', TRAIN / 'training.json'))
    targets = {f'model.layers.{layer}.self_attn.{category}' for layer in range(36) for category in ('q_proj', 'k_proj')}
    require(trained['status'] == 'trained' and trained['mode'] == 'formal'
            and trained['training_result']['global_step'] == 400
            and trained['training_result']['round_base_checkpoint_id'] == SOURCE_ID
            and trained['actual_control_exposure_exact']
            and set(trained['selected_targets']) == set(launch['selected_targets']) == set(audit['selected_modules']) == targets,
            'Require actual400 with only72 Q/K projections and matched original OV exposure.')
    require(audit['status'] == 'verified' and audit['source_checkpoint_id'] == SOURCE_ID
            and audit['checkpoint_id'] == acceptance['checkpoint_id'] == launch['checkpoint_id']
            and audit['selected_global_step'] == 400 and audit['selected_module_count'] == 72
            and audit['selected_decoder_tensor_count'] == 432 and audit['frozen_tensors_compared'] == 2991
            and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 3423
            and audit['code_bank_count'] == 504 and audit['changed_code_bits'] == 0
            and audit['fresh_strict_reload_state_exact'] and audit['frozen_bytes_unchanged']
            and audit['selected_decoder_matches_fp32_cast_exactly'] and audit['selected_decoder_update_survived_bf16']
            and audit['native_decoder_dtype'] == 'torch.bfloat16' and audit['rank_zero']
            and sha256(step / 'audit.json') == launch['audit_sha256']
            and sha256(step / 'acceptance.json') == launch['acceptance_sha256'], 'Reuse the completed native subset audit.')
    budget = audit['bit_budget']
    require(budget == acceptance['bit_budget'] and budget['budget_passed'] and not budget['contract_violations']
            and budget['user_linear_bits'] == budget['packed_linear_bits'] == 17065672704
            and budget['original_linear_weights'] == 6945767424 and budget['code_padding_bits'] == 0
            and all(budget['components'].get(k, {}).get('storage_bits', 0) == 0 for k in ('lora', 'residual_lora', 'dense_weights')),
            'Require original actual2.45698879076bpw/rank0 deployment.')
    candidate = full8([step / 'mmlu.json', step / 'seven.json'], checkpoint, launch['architecture'])
    require(acceptance['status'] in ('passed', 'not_met') and acceptance['budget_passed']
            and acceptance['task_metrics'] == candidate['task_metrics']
            and math.isclose(acceptance['mean_percent'], candidate['mean_percent'], rel_tol=0, abs_tol=1e-12),
            'Actual complete raw scores differ from native summary.')
    recovery = OV / 'evaluation_recovery'; terminal = read(recovery / 'terminal.json')
    require(sha256(recovery / 'terminal.json') == '5c15d34dff32cf53e9b02b501a092ba8771dfcf794e84991034c1abeee33800f'
            and terminal['status'] == 'completed' and terminal['exit_code'] == 0
            and terminal['summary_sha256'] == sha256(recovery / 'summary.json'), 'Use the actual recovered OV400 full8.')
    ov = full8([OV / 'step_0400/mmlu.json', recovery / 'seven.json'], OV / 'step_0400/final_model', launch['architecture'])
    source = Path(launch['source_checkpoint'])
    original = full8([source.parent / 'mmlu.json', source.parent / 'seven.json'], source, launch['architecture'])
    require(ov['task_metrics'] == read(recovery / 'summary.json')['task_metrics']
            and original['checkpoint_id'] == SOURCE_ID, 'Historical full8 controls changed.')
    comparisons = {name: {'checkpoint_id': control['checkpoint_id'], 'mean_percent': control['mean_percent'],
        'task_metrics': control['task_metrics'], 'integer_counts': control['integer_counts'],
        'candidate_minus_control_pp': candidate['mean_percent'] - control['mean_percent'],
        'task_delta_pp': {task: 100 * (candidate['task_metrics'][task] - control['task_metrics'][task]) for task in TASK_METRICS},
        'mmlu_correct_delta': candidate['integer_counts']['mmlu']['correct'] - control['integer_counts']['mmlu']['correct']}
        for name, control in (('ov8k400', ov), ('f927', original))}
    paths = [TRAIN / name for name in ('launch.prepared.json', 'launch.bound.json', 'run.sh', 'training.json',
             'exposure.json', 'training.exit', 'dispatcher.exit', 'outer.exit')]
    paths += [step / name for name in ('audit.json', 'acceptance.json', 'mmlu.json', 'seven.json',
              'export.exit', 'audit.exit', 'mmlu.exit', 'seven.exit', 'summary.exit', 'final_model/checkpoint_meta.json')]
    paths += [recovery / name for name in ('terminal.json', 'summary.json', 'seven.json')]
    paths += [OV / 'step_0400/mmlu.json', OV / 'step_0400/final_model/checkpoint_meta.json',
              source / 'checkpoint_meta.json', source.parent / 'mmlu.json', source.parent / 'seven.json']
    inputs = {**launch['input_evidence_sha256'], **{str(p): sha256(p) for p in paths}}
    check_hashes(launch['source_sha256']); check_hashes(inputs)
    for stat in (*launch['hf_weight_stats'], launch['source_weight_stat']): check_stat(stat)
    return launch, checkpoint, budget, candidate, comparisons, inputs, exits, process


def prepare():
    require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized() and not OUTPUT.exists(), 'Use CPU bitvae/public source and a new output.')
    launch, checkpoint, budget, candidate, comparisons, inputs, exits, process = completed_candidate()
    old, config0, review, scores = (read(ROTATION / name) for name in ('preparation.json', 'config.json', 'review.json', 'scores.json'))
    require(sha256(ROTATION / 'review.json') == '4818ddab48c11bb3115c958f1997d8982ee8e4a9966a61429260c099aed1c95e'
            and review['status'] == 'verified' and review['samples'] == 400 and review['execution_exit_code'] == 0
            and review['paired_input_generation_options_exact'] and review['decoded_output_tokens_exact']
            and review['checkpoint_id'] == scores['checkpoint_id'] == '43b0c5a2-4ef7-4a00-a6d1-ed887be18d01'
            and old['config_sha256'] == scores['config_sha256'] == sha256(ROTATION / 'config.json')
            and review['evidence_sha256'][str(ROTATION / 'scores.json')] == sha256(ROTATION / 'scores.json'),
            'Reuse the reviewed original-HF QK full rotation400 and its original protocol.')
    controls = {'f927': old['controls']['f927'], 'qk_hf_full': {'directory': str(ROTATION),
        'checkpoint_id': review['checkpoint_id'], 'model_reload_required': False,
        'tasks': {task: {'samples': 200, 'score_percent': review['task_results'][task]['score_percent'],
            'prediction_path': str(ROTATION / 'run/predictions' / (task + '.jsonl'))} for task in TASKS}}}
    config = make_config(config0['archive']['path'], kind='compressed', hf_path=launch['architecture']['base_model_path'],
                         checkpoint=checkpoint, mode='full')
    require(config['model']['checkpoint_id'] == candidate['checkpoint_id']
            and all(config[k] == config0[k] for k in ('schema', 'mode', 'protocol', 'archive', 'package_versions',
                                                     'model_policy', 'prewarm_group_size', 'device')),
            'Preserve original Qwen FA2/FP32RoPE/32K/nonthinking native400 policy.')
    check_config(config)
    for path in (ROTATION / 'preparation.json', ROTATION / 'config.json', ROTATION / 'review.json', ROTATION / 'scores.json'):
        inputs[str(path)] = sha256(path)
    for control in controls.values():
        directory = Path(control['directory']); historical = read(directory / 'config.json')
        require(all(historical[k] == config[k] for k in ('protocol', 'archive', 'package_versions')),
                'Historical400 controls differ in protocol.')
        inputs[str(directory / 'config.json')] = sha256(directory / 'config.json')
        for task in TASKS:
            path = control['tasks'][task]['prediction_path']
            require(scores['prediction_sha256'][path] == sha256(path), 'Reviewed400 prediction changed.')
            inputs[path] = sha256(path)
    _, index = load_archive(config['archive']['path'], check_token_file=False)
    rows = {task: [r for r in index if r['dataset'] == task] for task in TASKS}
    require(all([r['row_index'] for r in value] == list(range(200)) for value in rows.values()),
            'Use original archived200+200 without selection.')
    sources = dict(launch['source_sha256'])
    for path, digest in config['source_sha256'].items():
        require(path not in sources or sources[path] == digest, 'Training/generation source identity drifted: ' + path)
        sources[path] = digest
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    sources[str(ROOT / 'experiments/w25_eval/acceptance.py')] = sha256(ROOT / 'experiments/w25_eval/acceptance.py')
    check_hashes(sources); check_hashes(inputs)
    record = {'schema': 'qwen_qk_longalpaca8k_retrieval400_v1', 'status': 'prepared_unbound', 'gpu': None,
        'source_root': str(ROOT), 'python_executable': sys.executable, 'checkpoint_id': candidate['checkpoint_id'],
        'config_path': str(OUTPUT / 'config.json'), 'output_dir': str(OUTPUT / 'run'), 'datasets': list(TASKS),
        'sample_count': 400, 'task_samples': {task: 200 for task in TASKS}, 'source_sha256': sources,
        'evidence_sha256': inputs, 'source_bit_budget': budget, 'completed_candidate': {
            'directory': str(TRAIN), 'checkpoint_dir': str(checkpoint), 'full8': candidate, 'exits': exits,
            'source_checkpoint_id': SOURCE_ID, 'source_process': process, 'all_source_processes_retired': True,
            'serialized_tensors': 3423, 'trainable_decoder_tensors': 432, 'frozen_tensors': 2991,
            'frozen_code_banks': 504, 'actual400_exposure_matches_ov': True},
        'full8_controls': comparisons, 'controls': controls,
        'environment': {**old['environment'], 'PYTHONPATH': str(ROOT)},
        'input_lengths': {task: {'min': min(r['input_tokens'] for r in value), 'max': max(r['input_tokens'] for r in value)}
                          for task, value in rows.items()},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.longbench_v1.predict', '--config', str(OUTPUT / 'config.json'),
                          '--output', str(OUTPUT / 'run'), '--datasets', *TASKS],
        'timeout_seconds': 900, 'termination_grace_seconds': 20, 'queued': False, 'automatic_retry': False, 'resume': False,
        'resource_evidence': {'same_protocol_qk_full400_seconds': review['duration_seconds'],
            'wall_seconds_estimate': [420, 600], 'new_disk_budget_bytes': 100 * 1024**2,
            'boundary': 'One native Qwen400 inference; no export, training or further short test.'},
        'selection': 'Original archived EN/ZH rows0..199 and tokens; no filtering or resampling.', 'scoring': old['scoring'],
        'comparison_boundary': 'Full8 primary: matched OV8K400, secondary f927. Retrieval400: f927 baseline and QK HF/full '
            'rotation reference; rotation and distillation are different interventions. OV has only16 retrieval cases, not400. '
            'Training max8192 is a cap, not a claim that all400 batches contain8192 tokens.',
        'validation_reuse': 'Completed native3423 save/fresh,432 FP32-to-BF16 decoder cast,2991 frozen/504 banks unchanged; '
            'only raw full8 recomputed on CPU. Native predictor still checks fresh strict loading and actual storage. '
            'Original f927 config predates the model_policy field; its actual400 protocol was verified by the QK full review. '
            'Current policy is compared explicitly with QK full, with all three archive/protocol/package records equal.',
        'scope': 'Only2/21 complete official retrieval tasks; no fullLongBench or nine-item acceptance claim.',
        'full_longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated',
        'model_mutations': False, 'model_exported': False, 'cuda_initialized': False}
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    OUTPUT.mkdir(); write_json(OUTPUT / 'config.json', config)
    record['config_sha256'] = sha256(OUTPUT / 'config.json'); write_json(OUTPUT / 'preparation.json', record)
    print({'status': record['status'], 'checkpoint_id': candidate['checkpoint_id'], 'source_count': len(sources),
           'input_count': len(inputs), 'preparation_sha256': sha256(OUTPUT / 'preparation.json'),
           'mean_percent': candidate['mean_percent'], 'mmlu_correct': candidate['integer_counts']['mmlu']['correct'],
           'full8_delta_pp': {k: v['candidate_minus_control_pp'] for k, v in comparisons.items()}}, flush=True)


if __name__ == '__main__':
    prepare()
