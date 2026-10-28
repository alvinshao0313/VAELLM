"""One fixed endpoint: complete eight tasks and full LongBench on its assigned GPU."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import export as native
from experiments.longbench_v1.models import make_config, check_config
from experiments.longbench_v1.common import check_hashes
from experiments.longbench_v1.scoring import score_directory

HERE = Path(__file__).resolve().parent
api = native.module_from(native.RESULT / 'qwen_aux_retention32_batch_20261011/evaluate.py', '_deosc_evaluation_helpers')
HELPER_DIR = native.RESULT / 'qwen_b32_s3_gain_calibration_20261024/evaluation_adapter'
sys.path.insert(0, str(HELPER_DIR))
mature = native.module_from(HELPER_DIR / 'full9.py', '_deosc_full8_helpers')
mature.e = SimpleNamespace(helpers=lambda: api, read=native.read, require=native.require, PYTHON=Path(sys.executable))
stages = native.module_from(native.SNAPSHOT / 'experiments/qwen3_w25/step_export_callback.py', '_deosc_logged_stages')


def evaluate(endpoint, gpu):
    native.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU evaluation driver; only children use the assigned GPU.')
    endpoint = endpoint.resolve()
    audit_path = endpoint / 'native_audit.json'
    audit = native.read(audit_path)
    checkpoint = Path(audit['checkpoint_dir']).resolve()
    meta = native.read(checkpoint / 'checkpoint_meta.json')
    native.require(audit['status'] == 'verified' and audit['sources']['global_step'] == 256
                   and audit['checkpoint_id'] == meta['checkpoint_id'], 'Only the fixed actual terminal256 is benchmarked.')
    output = endpoint / 'evaluation_full9'
    native.require(not output.exists(), 'Never overwrite or resume an evaluation.')
    resources = mature.check_resources([str(gpu)])
    output.mkdir()
    reference = native.read(api.REFERENCE_CONFIG)
    config = make_config(reference['archive']['path'], kind='compressed', hf_path=meta['base_model_path'],
                         checkpoint=checkpoint, mode='full')
    native.require(all(config[key] == reference[key] for key in reference if key != 'model'),
                   'Keep the original complete LongBench archive and generation protocol.')
    paths = [HERE / name for name in ('export.py', 'inference.py', 'evaluate_full9.py')]
    paths += [HELPER_DIR / 'full9.py', native.CPU_LOADER,
              native.RESULT / 'qwen_b32_mlp_s3_20261023/summarize_nine.py',
              audit_path, checkpoint / 'checkpoint_meta.json', endpoint / 'endpoint.json',
              Path(audit['sources']['payload']['path']), *api.source_paths()]
    pins = {str(path): native.binding(path)['sha256'] for path in paths}
    config['model']['deoscillation_audit'] = str(audit_path)
    config['model']['deoscillation_source_sha256'] = pins
    check_config(config)
    report = {'status': 'running', 'checkpoint_id': meta['checkpoint_id'], 'arm': audit['sources']['arm'],
              'global_step': 256, 'started_unix': time.time(), 'gpu': gpu, 'resources': resources,
              'automatic_retry': False, 'partial_scoring': False, 'stages': {}}

    def stage(name, argv, *, visible='', seconds=3600):
        command = ['timeout', '--signal=TERM', '--kill-after=20s', str(seconds) + 's', *argv]
        report['stages'][name] = {'command': command, 'started_unix': time.time()}
        native.write(output / 'evaluation.json', report)
        try:
            stages.run_logged_stage(command, name=name, output_dir=output,
                                    env=api.environment(native.SNAPSHOT, visible))
        finally:
            row = report['stages'][name]
            row['finished_unix'] = time.time()
            exit_path = output / (name + '.exit')
            row['exit_code'] = exit_path.read_text().strip() if exit_path.exists() else None
            native.write(output / 'evaluation.json', report)

    try:
        for name, argv in mature.full8_commands(api, checkpoint, output):
            stage(name, argv, visible=str(gpu))
        full8 = mature.validate_full8(output, checkpoint, meta, config, pins)
        config_path = output / 'longbench_config.json'
        native.write(config_path, config)
        stage('longbench', [sys.executable, '-B', '-u', str(HERE / 'inference.py'), 'predict',
              '--config', str(config_path), '--output', str(output / 'longbench')], visible=str(gpu), seconds=86580)
        scores_path = output / 'longbench/scores.json'
        score_directory(config['archive']['path'], output / 'longbench', scores_path)
        template = native.read(api.TEMPLATE)['nine_task_argv']
        baseline_index = template.index('--baseline_results')
        sys.path.insert(0, str(native.RESULT / 'qwen_b32_mlp_s3_20261023'))
        formula = native.module_from(native.RESULT / 'qwen_b32_mlp_s3_20261023/summarize_nine.py', '_deosc_nine_formula')
        nine = formula.summarize(base_model_path=meta['base_model_path'], checkpoint_dir=str(checkpoint),
            longbench_manifest=config['archive']['path'],
            baseline_longbench_scores=template[template.index('--baseline_longbench_scores') + 1],
            compressed_longbench_scores=str(scores_path), baseline_results=template[baseline_index + 1:baseline_index + 3],
            compressed_results=[str(output / 'mmlu.json'), str(output / 'seven.json')])
        native.require(nine['quality_target_evaluated'] and nine['checkpoint_id'] == meta['checkpoint_id']
                       and abs(nine['baseline_nine_task_score'] - 68.04714491790848) < 1e-12,
                       'Original nine-term baseline or actual endpoint identity changed.')
        native.write(output / 'nine_task_acceptance.json', nine)
        check_hashes(pins)
        report.update(status='complete_full8_full21_nine', full8_mean_percent=full8['mean_percent'],
                      nine_task_score=nine['compressed_nine_task_score'], passed=nine['passed'])
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['finished_unix'] = time.time()
        native.write(output / 'evaluation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    args = parser.parse_args()
    print(evaluate(args.endpoint, args.gpu), flush=True)
