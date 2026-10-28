"""Finite C/G tail: parallel06/07, independent07 full9, then08/09/final9."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import common as c


def process_start(pid):
    return Path(f'/proc/{pid}/stat').read_text().split()[21]


def stage(name, command, output, *, gpu='', seconds=3600, format_recovery=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    argv = ['timeout', '--signal=TERM', '--kill-after=60s', str(seconds) + 's', *command]
    item = dict(command=argv, gpu=gpu, started_unix=time.time(), timeout_seconds=seconds,
                format_recovery=format_recovery, cwd=str(c.SNAPSHOT))
    with (output / (name + '.log')).open('x', encoding='utf-8') as log:
        log.write(json.dumps(item) + '\n')
        log.flush()
        process = subprocess.Popen(argv, cwd=c.SNAPSHOT, env=c.environment(gpu, format_recovery=format_recovery),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        item.update(pid=process.pid, start_ticks=process_start(process.pid))
        c.write(output / (name + '_process.json'), item)
        code = process.wait()
    item.update(exit_code=code, finished_unix=time.time())
    c.write(output / (name + '_process.json'), item)
    (output / (name + '.exit')).write_text(str(code) + '\n', encoding='utf-8')
    c.require(code == 0, f'{name} exited{code}; see {output / (name + ".log")}.')
    return item


def worker(profile, kind):
    c.check_prepared()
    stage_name = '06_decoder_kl' if kind == 'decoder' else '07_chatml_joint256'
    output = c.OUTPUT / profile / stage_name
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'pipeline.json'
    c.require(not path.exists(), 'No retry or duplicate recovery arm.')
    gpu = c.GPUS[profile][kind]
    record = dict(status='running', profile=profile, stage=stage_name, gpu=gpu,
        source_checkpoint_id=c.SOURCE_IDS[profile], started_unix=time.time(), stages={})
    c.write(path, record)
    try:
        c.check_gpu(gpu)
        if kind == 'decoder':
            command = [sys.executable, '-B', '-u', str(c.HERE / 'train.py'), '--profile', profile, '--recipe', 'decoder']
            record['stages']['train'] = stage('train', command, output, gpu=gpu, seconds=50400)
            command = [sys.executable, '-B', '-u', str(c.HERE / 'endpoints.py'), 'audit', '--profile', profile, '--recipe', 'decoder']
            record['stages']['audit'] = stage('audit', command, output, seconds=3600)
        else:
            command = [sys.executable, '-B', '-u', '-m', 'experiments.qwen3_w25.train_format_recovery',
                       '--launch_record', str(output / 'format_launch.json')]
            record['stages']['train'] = stage('train', command, output, gpu=gpu, seconds=5400, format_recovery=True)
            command = [sys.executable, '-B', '-u', str(c.HERE / 'endpoints.py'), 'chatml', '--profile', profile]
            record['stages']['export'] = stage('export', command, output, seconds=3600)
            command = [sys.executable, '-B', '-u', str(c.HERE / 'endpoints.py'), 'consumer', '--profile', profile]
            record['stages']['consumer'] = stage('consumer', command, output, gpu=gpu, seconds=1800)
        audit = c.read(output / 'native_audit.json')
        c.require(audit['status'] == 'verified', 'Only an independently audited actual native donor may be used.')
        record.update(status='complete_native', checkpoint_id=audit['checkpoint_id'], final_model=audit['checkpoint_dir'],
                      native_audit=str(output / 'native_audit.json'))
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        c.write(path, record)


def evaluate(profile, which):
    c.check_prepared()
    sys.path.insert(0, str(c.ROOT / 'experiments/night_sequence_20261026'))
    from recovery_eval import full8_summary, summarize_nine
    from experiments.longbench_v1.models import make_config
    stage_name = '07_chatml_joint256' if which == 'chatml' else '09_joint1200'
    root = c.OUTPUT / profile / stage_name
    evidence = root / 'native_audit.json'
    audit = c.read(evidence)
    c.require(audit['status'] == 'verified', 'Evaluate the actual verified native endpoint.')
    if which == 'chatml':
        proof = c.read(root / 'gpu_validation.json')
        c.require(proof['status'] == 'verified' and proof['checkpoint_id'] == audit['checkpoint_id'], 'Actual ChatML saved consumer must pass before full9.')
    checkpoint = Path(audit['checkpoint_dir'])
    meta, budget = c.checkpoint(checkpoint)
    gpu = c.GPUS[profile][which if which == 'chatml' else 'decoder']
    output = root / 'evaluation_full9'
    c.require(not output.exists(), 'One complete full9 per fixed endpoint; no retry.')
    c.check_gpu(gpu)
    output.mkdir()
    record = dict(status='running', profile=profile, which=which, checkpoint_id=meta['checkpoint_id'],
        checkpoint_dir=str(checkpoint), gpu=gpu, started_unix=time.time(), bit_budget=budget, stages={},
        full8_sample_count=24742, longbench_sample_count=4750, longbench_timeout_seconds=86400,
        quality_does_not_select_or_block_fixed_main_chain=True)
    path = output / 'evaluation.json'
    c.write(path, record)
    api = c.module(c.TOPIC / 'qwen_aux_retention32_batch_20261011/evaluate.py', '_recompression_tail_evaluation_helpers')
    pins = {str(item): c.sha(item) for item in [*c.HERE.glob('*.py'), *api.source_paths(), evidence,
        checkpoint / 'checkpoint_meta.json', c.ROOT / 'experiments/cat_recompression/longbench.py']}
    before = c.stat(checkpoint / meta['state_dict_file'])
    try:
        for name, tasks, batch in (('mmlu', 'mmlu', '64'), ('seven', api.SEVEN, '32')):
            command = [sys.executable, '-B', '-u', str(api.SAMPLE_EVAL), '--checkpoint_dir', str(checkpoint),
                '--strict', '--eval_lm_eval', '--eval_device', 'cuda:0', '--tasks', tasks, '--num_fewshot', '0',
                '--lm_batch_size', batch, '--tokenizer_use_fast', 'true', '--prewarm_group_size', '8',
                '--eval_log_dir', str(output / (name + '_logs')), '--output_json', str(output / (name + '.json'))]
            record['stages'][name] = stage(name, command, output, gpu=gpu, seconds=7200)
            c.write(path, record)
        full8, baseline_paths, template = full8_summary(checkpoint, output)
        record['full8_mean_percent'] = full8['mean_percent']
        reference = c.read(api.REFERENCE_CONFIG)
        config = make_config(reference['archive']['path'], kind='compressed', hf_path=meta['base_model_path'], checkpoint=checkpoint, mode='full')
        c.require(all(config[key] == reference[key] for key in reference if key != 'model'), 'Keep the original archive,32K BF16 policy and complete generation protocol.')
        config['model']['recompression'] = dict(max_linear_bpw=2.6, max_lora_rank=0, bit_budget=budget, source_sha256=pins)
        config_path = output / 'longbench_config.json'
        c.write(config_path, config)
        command = [sys.executable, '-B', '-u', str(c.HERE / 'run.py'), 'longbench', '--config', str(config_path), '--output', str(output / 'longbench')]
        record['stages']['longbench'] = stage('longbench', command, output, gpu=gpu, seconds=86400)
        nine = summarize_nine(checkpoint, output, config_path, baseline_paths, template)
        for filename, digest in pins.items():
            c.require(c.sha(filename) == digest, 'Evaluation pinned source changed: ' + filename)
        c.require(c.stat(checkpoint / meta['state_dict_file']) == before, 'Formal evaluation modified native source weights.')
        record.update(status='complete_full8_full21_nine', nine_task_score=nine['compressed_nine_task_score'], passed=nine['passed'])
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        c.write(path, record)
    return record


def spawn(command, output, name):
    output.mkdir(parents=True, exist_ok=True)
    with (output / (name + '.log')).open('x', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=c.ROOT, env=c.environment(''), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(pid=child.pid, start_ticks=process_start(child.pid), command=command, log=str(output / (name + '.log')), started_unix=time.time())
    return child, record


def profile_run(profile):
    c.check_prepared()
    root = c.OUTPUT / profile
    path = root / 'pipeline.json'
    c.require(not path.exists(), 'One finite recovery-tail execution per source.')
    record = dict(status='running_parallel_06_07', profile=profile, started_unix=time.time(), pid=c.os.getpid(),
                  source_checkpoint_id=c.SOURCE_IDS[profile], automatic_retry=False, stages={})
    c.write(path, record)
    chatml_eval = None
    try:
        entry = [sys.executable, '-B', '-u', str(c.HERE / 'run.py')]
        decoder, record['decoder_process'] = spawn([*entry, 'decoder', '--profile', profile], root, 'decoder_worker')
        chatml, record['chatml_process'] = spawn([*entry, 'chatml', '--profile', profile], root, 'chatml_worker')
        c.write(path, record)
        # Wait on the actual one-shot producer, then launch its independent quality consumer.
        chatml_code = chatml.wait()
        record['chatml_process']['exit_code'] = chatml_code
        c.require(chatml_code == 0 and c.read(root / '07_chatml_joint256/pipeline.json')['status'] == 'complete_native', 'Actual ChatML producer failed; fixed assembly cannot proceed.')
        chatml_eval, record['chatml_full9_process'] = spawn([*entry, 'evaluate', '--profile', profile, '--which', 'chatml'], root, 'chatml_full9_worker')
        record['status'] = 'waiting_06_with_07_full9_independent'
        c.write(path, record)
        decoder_code = decoder.wait()
        record['decoder_process']['exit_code'] = decoder_code
        c.require(decoder_code == 0 and c.read(root / '06_decoder_kl/pipeline.json')['status'] == 'complete_native', 'Actual KL decoder producer failed; fixed attention donor unavailable.')
        gpu = c.GPUS[profile]['decoder']
        c.check_gpu(gpu)
        record['status'] = 'running_08_09'
        c.write(path, record)
        record['stages']['attention'] = stage('attention', [sys.executable, '-B', '-u', str(c.HERE / 'endpoints.py'), 'attention', '--profile', profile], root, seconds=3600)
        source = c.DATA / profile / '08_joint_mlp_decoder_attention/final_model'
        record['stages']['last_train'] = stage('last_train', [sys.executable, '-B', '-u', str(c.HERE / 'train.py'), '--profile', profile, '--recipe', 'last_joint', '--source', str(source)], root, gpu=gpu, seconds=50400)
        record['stages']['last_audit'] = stage('last_audit', [sys.executable, '-B', '-u', str(c.HERE / 'endpoints.py'), 'audit', '--profile', profile, '--recipe', 'last_joint'], root, seconds=3600)
        record['status'] = 'running_final_full9'
        c.write(path, record)
        final = evaluate(profile, 'final')
        record.update(final_evaluation=str(root / '09_joint1200/evaluation_full9/evaluation.json'), final_nine_task_score=final['nine_task_score'])
        # Its execution/quality result never changes08/09 or cancels either independent evaluation.
        record['chatml_full9_process']['exit_code'] = chatml_eval.wait()
        chatml_report = c.read(root / '07_chatml_joint256/evaluation_full9/evaluation.json')
        record.update(status='complete' if record['chatml_full9_process']['exit_code'] == 0 else 'complete_main_with_chatml_evaluation_failure',
                      chatml_evaluation_status=chatml_report['status'], chatml_nine_task_score=chatml_report.get('nine_task_score'))
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        c.write(path, record)


def launch():
    c.check_prepared()
    c.require(not (c.OUTPUT / 'launch.json').exists(), 'Launch the fixed two-source tail only once.')
    for gpu in (1, 3, 6, 7):
        c.check_gpu(gpu)
    record = dict(status='launching', started_unix=time.time(), prepared_sha256=c.sha(c.PREPARED), workers={}, automatic_retry=False)
    c.write(c.OUTPUT / 'launch.json', record)
    for profile in ('C', 'G'):
        _child, row = spawn([sys.executable, '-B', '-u', str(c.HERE / 'run.py'), 'profile', '--profile', profile], c.OUTPUT, profile + '_tail')
        record['workers'][profile] = row
        c.write(c.OUTPUT / 'launch.json', record)
    record['status'] = 'two_fixed_profiles_running'
    c.write(c.OUTPUT / 'launch.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['launch', 'profile', 'decoder', 'chatml', 'evaluate', 'longbench'])
    parser.add_argument('--profile', choices=['C', 'G'])
    parser.add_argument('--which', choices=['chatml', 'final'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'launch':
        launch()
    elif args.command == 'profile':
        profile_run(args.profile)
    elif args.command in ('decoder', 'chatml'):
        worker(args.profile, args.command)
    elif args.command == 'evaluate':
        evaluate(args.profile, args.which)
    else:
        c.check_prepared()
        from experiments.longbench_v1 import runner
        from experiments.longbench_v1.protocol import DATASETS
        consumer = c.module(c.ROOT / 'experiments/cat_recompression/longbench.py', '_recompression_tail_longbench_loader')
        runner.load_model = consumer.load_model
        c.require(c.read(args.config)['mode'] == 'full' and not args.output.exists(), 'One fresh complete21-task run; no partial/resume.')
        result = runner.run(args.config, args.output, datasets=list(DATASETS), resume=False)
        c.require(result['status'] == 'partition_complete' and result['exit_code'] == 0
            and result['new_predictions'] == 4750 and result['already_present'] == 0, 'All21 official tasks/4750 samples must finish.')


if __name__ == '__main__':
    main()
