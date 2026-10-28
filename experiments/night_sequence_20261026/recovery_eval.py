"""The existing complete native-v6 full8/LongBench21 protocol for fixed endpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from recovery_core import (HERE, ROOT, SNAPSHOT, TOPIC, checkpoint, environment, module_from,
                           own_sources, read, require, sha, stat, write)


def run_stage(command, *, name, output, gpu='', seconds=3600, foreground=False):
    argv = ['timeout'] + (['--foreground'] if foreground else [])
    argv += ['--signal=TERM', '--kill-after=20s', str(seconds) + 's', *command]
    output = Path(output)
    with (output / (name + '.log')).open('x', encoding='utf-8') as log:
        log.write(json.dumps(dict(command=argv, cwd=str(ROOT), gpu=gpu)) + '\n')
        log.flush()
        result = subprocess.run(argv, cwd=ROOT, env=environment(gpu), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
    (output / (name + '.exit')).write_text(str(result.returncode) + '\n', encoding='utf-8')
    require(result.returncode == 0, f'{name} exited{result.returncode}; see {output / (name + ".log")}.')
    return result.returncode


def full8_summary(checkpoint_dir, output):
    from experiments.w25_eval.acceptance import collect_results
    from experiments.w25_eval.model_inventory import model_inventory
    meta = read(Path(checkpoint_dir) / 'checkpoint_meta.json')
    api = module_from(TOPIC / 'qwen_aux_retention32_batch_20261011/evaluate.py', '_night_full8_helpers')
    template = read(api.TEMPLATE)['nine_task_argv']
    index = template.index('--baseline_results')
    baseline_paths = template[index + 1:index + 3]
    paths = [Path(output) / 'mmlu.json', Path(output) / 'seven.json']
    inventory = model_inventory(meta['base_model_path'])
    result = collect_results(paths, inventory, checkpoint=(Path(checkpoint_dir), meta))
    baseline = collect_results(baseline_paths, inventory)
    require(result['versions'] == baseline['versions'] and result['sample_count'] == 24742,
            'Complete full8 versions and24742 examples must match BF16.')
    for path, reference, count in zip(paths, baseline_paths, (14042, 10700), strict=True):
        lm = read(path)['evals']['lm_eval']
        require(lm['configs'] == read(reference)['configs'] and lm['sample_artifacts']['total_count'] == count,
                'Full8 configurations/raw counts differ from the original protocol.')
        for row in lm['sample_artifacts']['tasks'].values():
            artifact = Path(row['path'])
            require(artifact.is_relative_to(Path(output)) and sha(artifact) == row['sha256'],
                    'The raw full8 sample binding changed.')
    write(Path(output) / 'full8_summary.json', dict(checkpoint_id=meta['checkpoint_id'], **result))
    return result, baseline_paths, template


def summarize_nine(checkpoint_dir, output, config_path, baseline_paths, template):
    from experiments.longbench_v1.scoring import score_directory
    config = read(config_path)
    scores_path = Path(output) / 'longbench/scores.json'
    scores = score_directory(config['archive']['path'], Path(output) / 'longbench', scores_path)
    require(scores['sample_count'] == 4750 and len(scores['tasks']) == 21,
            'Use all21 canonical LongBench tasks and4750 examples.')
    formula_dir = TOPIC / 'qwen_b32_mlp_s3_20261023'
    sys.path.insert(0, str(formula_dir))
    formula = module_from(formula_dir / 'summarize_nine.py', '_night_original_nine_formula')
    meta = read(Path(checkpoint_dir) / 'checkpoint_meta.json')
    nine = formula.summarize(base_model_path=meta['base_model_path'], checkpoint_dir=str(checkpoint_dir),
        baseline_results=baseline_paths,
        compressed_results=[str(Path(output) / 'mmlu.json'), str(Path(output) / 'seven.json')],
        longbench_manifest=config['archive']['path'],
        baseline_longbench_scores=template[template.index('--baseline_longbench_scores') + 1],
        compressed_longbench_scores=str(scores_path))
    require(nine['quality_target_evaluated'] and nine['checkpoint_id'] == meta['checkpoint_id']
            and abs(nine['baseline_nine_task_score'] - 68.04714491790848) < 1e-12,
            'The actual model identity/original nine-term reference changed.')
    write(Path(output) / 'nine_task_acceptance.json', nine)
    return nine


def evaluate(checkpoint_dir, output, gpu, *, full9=False, evidence=None, foreground=False):
    from experiments.longbench_v1.models import make_config
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Evaluation driver must remain on CPU.')
    checkpoint_dir, output = Path(checkpoint_dir).resolve(), Path(output).resolve()
    require(not output.exists(), 'Never overwrite an endpoint evaluation.')
    meta, budget = checkpoint(checkpoint_dir)
    if evidence:
        audit = read(evidence)
        require(audit['status'] == 'verified' and audit['checkpoint_id'] == meta['checkpoint_id'],
                'Use the actual successful endpoint audit.')
    output.mkdir(parents=True)
    record = dict(status='running', checkpoint_dir=str(checkpoint_dir), checkpoint_id=meta['checkpoint_id'],
                  gpu=gpu, full9=full9, started_unix=time.time(), bit_budget=budget,
                  evidence=str(evidence) if evidence else None, stages={},
                  inner_timeout_foreground=foreground)
    report_path = output / 'evaluation.json'
    write(report_path, record)
    api = module_from(TOPIC / 'qwen_aux_retention32_batch_20261011/evaluate.py', '_night_native_evaluation_helpers')
    pins = {str(path): sha(path) for path in [*own_sources(), *api.source_paths(),
            checkpoint_dir / 'checkpoint_meta.json', ROOT / 'experiments/cat_recompression/longbench.py']}
    if evidence:
        pins[str(Path(evidence).resolve())] = sha(evidence)
    weight_stat = stat(checkpoint_dir / meta['state_dict_file'])
    try:
        for name, tasks, batch in (('mmlu', 'mmlu', '64'), ('seven', api.SEVEN, '32')):
            argv = [sys.executable, '-B', '-u', str(api.SAMPLE_EVAL), '--checkpoint_dir', str(checkpoint_dir),
                '--strict', '--eval_lm_eval', '--eval_device', 'cuda:0', '--tasks', tasks, '--num_fewshot', '0',
                '--lm_batch_size', batch, '--tokenizer_use_fast', 'true', '--prewarm_group_size', '8',
                '--eval_log_dir', str(output / (name + '_logs')), '--output_json', str(output / (name + '.json'))]
            record['stages'][name] = dict(command=argv, started_unix=time.time())
            write(report_path, record)
            run_stage(argv, name=name, output=output, gpu=str(gpu), foreground=foreground)
            record['stages'][name].update(exit_code=0, finished_unix=time.time())
            write(report_path, record)
        full8, baseline_paths, template = full8_summary(checkpoint_dir, output)
        record['full8_mean_percent'] = full8['mean_percent']
        if full9:
            reference = read(api.REFERENCE_CONFIG)
            config = make_config(reference['archive']['path'], kind='compressed', hf_path=meta['base_model_path'],
                                 checkpoint=checkpoint_dir, mode='full')
            require(all(config[key] == reference[key] for key in reference if key != 'model'),
                    'Keep the original archive,32K model policy and complete generation protocol.')
            config['model']['recompression'] = dict(max_linear_bpw=2.6, max_lora_rank=0,
                                                    bit_budget=budget, source_sha256=pins)
            config_path = output / 'longbench_config.json'
            write(config_path, config)
            argv = [sys.executable, '-B', '-u', str(ROOT / 'experiments/cat_recompression/longbench.py'),
                    '--config', str(config_path), '--output', str(output / 'longbench')]
            record['stages']['longbench'] = dict(command=argv, started_unix=time.time())
            write(report_path, record)
            run_stage(argv, name='longbench', output=output, gpu=str(gpu), seconds=28800, foreground=foreground)
            record['stages']['longbench'].update(exit_code=0, finished_unix=time.time())
            nine = summarize_nine(checkpoint_dir, output, config_path, baseline_paths, template)
            record.update(nine_task_score=nine['compressed_nine_task_score'], passed=nine['passed'])
        for path, expected in pins.items():
            require(sha(path) == expected, 'Evaluation source binding changed: ' + path)
        require(stat(checkpoint_dir / meta['state_dict_file']) == weight_stat, 'Evaluation changed native weights.')
        record['status'] = 'complete_full8_full21_nine' if full9 else 'complete_full8'
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        write(report_path, record)
    return record
