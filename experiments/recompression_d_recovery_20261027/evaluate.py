"""Unchanged native full8/LongBench21 protocol, with a24-hour LongBench bound."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import traceback

import core as d
from recovery_eval import full8_summary, summarize_nine


def evaluate(checkpoint, output, gpu, evidence):
    from experiments.longbench_v1.models import make_config
    d.check()
    d.require(d.os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not output.exists(), 'One fresh full9 driven from CPU.')
    meta, budget = d.native.checkpoint(checkpoint)
    audit = d.read(evidence)
    d.require(audit['status'] == 'verified' and audit['checkpoint_id'] == meta['checkpoint_id'], 'Use the actual successful D05 audit.')
    output.mkdir(parents=True)
    record = dict(status='running', checkpoint_id=meta['checkpoint_id'], checkpoint_dir=str(checkpoint),
        gpu=gpu, started_unix=time.time(), bit_budget=budget, evidence=str(evidence), stages={},
        full8_samples=24742, longbench_samples=4750, longbench_timeout_seconds=86400)
    path = output / 'evaluation.json'
    d.write(path, record)
    api = d.native.module(d.TOPIC / 'qwen_aux_retention32_batch_20261011/evaluate.py', '_D_recovery_full9_protocol')
    pins = {str(item): d.sha(item) for item in [*d.HERE.glob('*.py'), *api.source_paths(), evidence,
        checkpoint / 'checkpoint_meta.json', d.ROOT / 'experiments/cat_recompression/longbench.py']}
    original_stat = d.stat(checkpoint / meta['state_dict_file'])
    try:
        for name, tasks, batch in (('mmlu', 'mmlu', '64'), ('seven', api.SEVEN, '32')):
            command = [sys.executable, '-B', '-u', str(api.SAMPLE_EVAL), '--checkpoint_dir', str(checkpoint),
                '--strict', '--eval_lm_eval', '--eval_device', 'cuda:0', '--tasks', tasks, '--num_fewshot', '0',
                '--lm_batch_size', batch, '--tokenizer_use_fast', 'true', '--prewarm_group_size', '8',
                '--eval_log_dir', str(output / (name + '_logs')), '--output_json', str(output / (name + '.json'))]
            record['stages'][name] = d.stages.stage(name, command, output, gpu=gpu, seconds=7200)
            d.write(path, record)
        full8, baseline_paths, template = full8_summary(checkpoint, output)
        record['full8_mean_percent'] = full8['mean_percent']
        reference = d.read(api.REFERENCE_CONFIG)
        config = make_config(reference['archive']['path'], kind='compressed', hf_path=meta['base_model_path'], checkpoint=checkpoint, mode='full')
        d.require(all(config[key] == reference[key] for key in reference if key != 'model'), 'Keep the original archive,32K model policy and complete21 generation protocol.')
        config['model']['recompression'] = dict(max_linear_bpw=2.6, max_lora_rank=0, bit_budget=budget, source_sha256=pins)
        config_path = output / 'longbench_config.json'
        d.write(config_path, config)
        command = [sys.executable, '-B', '-u', str(d.HERE / 'evaluate.py'), '--config', str(config_path), '--output', str(output / 'longbench')]
        record['stages']['longbench'] = d.stages.stage('longbench', command, output, gpu=gpu, seconds=86400)
        nine = summarize_nine(checkpoint, output, config_path, baseline_paths, template)
        for filename, digest in pins.items():
            d.require(d.sha(filename) == digest, 'D05 evaluation source changed: ' + filename)
        d.require(d.stat(checkpoint / meta['state_dict_file']) == original_stat, 'Evaluation modified D05 native weights.')
        record.update(status='complete_full8_full21_nine', nine_task_score=nine['compressed_nine_task_score'], passed=nine['passed'])
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        d.write(path, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    d.check()
    from experiments.longbench_v1 import runner
    from experiments.longbench_v1.protocol import DATASETS
    consumer = d.native.module(d.ROOT / 'experiments/cat_recompression/longbench.py', '_D_recovery_native_LB_loader')
    runner.load_model = consumer.load_model
    d.require(d.read(args.config)['mode'] == 'full' and not args.output.exists(), 'One fresh full21 run without partial scoring or resume.')
    result = runner.run(args.config, args.output, datasets=list(DATASETS), resume=False)
    d.require(result['status'] == 'partition_complete' and result['exit_code'] == 0
        and result['new_predictions'] == 4750 and result['already_present'] == 0, 'All21 official tasks/4750 predictions are required.')
