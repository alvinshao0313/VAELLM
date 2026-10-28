"""Repair only the GPU-consumer entry and reattach the unchanged finite tail."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/recompression_recovery_tail_20261027'))

import common as c
import run as original

REPAIR = c.OUTPUT / 'consumer_repair'
PREPARED = REPAIR / 'prepared.json'
processes = c.module(c.ROOT / 'experiments/night_sequence_20261026/queue_wait.py', '_tail_repair_process_wait')


def check():
    c.check_prepared()
    record = c.read(PREPARED)
    c.require(record['status'] == 'cpu_verified' and c.sha(HERE / 'repair.py') == record['repair_sha256'], 'Use the frozen verified one-shot consumer repair.')
    c.require(c.sha(c.PREPARED) == record['original_prepared_sha256'], 'The original fixed five-source preparation changed.')
    for path, digest in record['evidence_sha256'].items():
        c.require(c.sha(path) == digest, 'Bound completed endpoint/failure changed: ' + path)
    return record


def prepare():
    c.check_prepared()
    c.require(not PREPARED.exists(), 'Prepare this confirmed consumer-only repair once.')
    wait_source = c.ROOT / 'experiments/night_sequence_20261026/queue_wait.py'
    evidence, donors = {str(wait_source): c.sha(wait_source)}, {}
    for profile in ('C', 'G'):
        root = c.OUTPUT / profile
        failure = root / '07_chatml_joint256/consumer.log'
        log = failure.read_text(encoding='utf-8')
        c.require('CPU endpoint work must remain on CPU.' in log and 'in consume_chatml' not in log,
                  'Only the confirmed pre-forward GPU/CPU guard failure is repaired.')
        c.require(c.read(root / 'pipeline.json')['status'] == 'failed'
            and c.read(root / '07_chatml_joint256/pipeline.json')['status'] == 'failed'
            and c.read(root / '07_chatml_joint256/native_audit.json')['status'] == 'verified'
            and c.read(root / '07_chatml_joint256/format_observations/diagnostic.json')['status'] == 'complete'
            and not (root / '07_chatml_joint256/gpu_validation.json').exists()
            and not (root / '07_chatml_joint256/evaluation_full9').exists(), 'The actual trained/exported07 must be intact and unevaluated.')
        donor = c.read(root / 'pipeline.json')['decoder_process']
        c.require(processes.process_start(donor['pid']) == int(donor['start_ticks']), 'The original independent KL worker identity changed.')
        donors[profile] = donor
        for path in (failure, root / 'pipeline.json', root / '07_chatml_joint256/pipeline.json',
                     root / '07_chatml_joint256/native_audit.json', root / '07_chatml_joint256/format_observations/diagnostic.json'):
            evidence[str(path)] = c.sha(path)
    record = dict(status='cpu_verified', schema='pre_forward_consumer_guard_repair_v1', created_unix=time.time(),
        repair_sha256=c.sha(HERE / 'repair.py'), original_prepared_sha256=c.sha(c.PREPARED),
        evidence_sha256=evidence, actual_original_KL_donors=donors,
        change='Use the existing GPU-visible native strict loader for the GPU consumer; no CPU-only constructor/guard.',
        no_training_replayed=True, trained_payload_and_native_export_unchanged=True,
        subsequent_fixed_recipes_unchanged=True, process_wait='existing Linux pidfd exit event; no polling')
    c.write(PREPARED, record)
    return record


def consume(profile):
    import torch
    from torch.nn import functional as F
    from tools.cat_eval import _prepare_model_for_eval
    from train_utils.v6_model_loader import load_v6_model_checkpoint
    check()
    torch.set_num_threads(4)
    root = c.OUTPUT / profile / '07_chatml_joint256'
    c.require(not (root / 'gpu_validation.json').exists(), 'One real consumer repair; no overwrite.')
    audit = c.read(root / 'native_audit.json')
    model, meta, loaded = load_v6_model_checkpoint(audit['checkpoint_dir'], map_location='cpu', strict=True, expected_kind='final_model')
    c.require(not loaded.missing_keys and not loaded.unexpected_keys and meta['checkpoint_id'] == audit['checkpoint_id'], 'Actual exported07 GPU-native strict load failed.')
    model.eval().requires_grad_(False)
    warmup = _prepare_model_for_eval(model, 'cuda:0', 8, logging.getLogger(__name__), {})
    c.require(warmup['total'] == warmup['warmed'] == 252 and not warmup['failed'] and not warmup['skipped'], 'Warm every actual native projection.')
    target, calls = model.get_submodule(c.TARGET), []
    def observe(layer, args, output):
        c.require(layer.cache_decoded_weight and layer.temporary and not layer.trainable_decode
            and torch.equal(output, F.linear(args[0], layer._cached_weight, layer.bias)), 'Real saved L0down native-cache consumption differs.')
        calls.append(1)
    handle = target.register_forward_hook(observe)
    try:
        archive = torch.load(c.ARCHIVE / 'batches.pt', map_location='cpu', weights_only=True)
        batch = archive['formats']['chatml']['batches'][0]
        with torch.no_grad():
            output = model(input_ids=batch['input_ids'].to('cuda:0'), attention_mask=batch['attention_mask'].to('cuda:0'), use_cache=False, logits_to_keep=1)
        c.require(calls == [1] and output.logits.dtype == torch.bfloat16 and tuple(output.logits.shape) == (4, 1, 151936)
            and torch.isfinite(output.logits).all().item() and output.past_key_values is None, 'The original train-only batch4 consumer must execute a complete finite native forward.')
    finally:
        handle.remove()
    proof = dict(status='verified', checkpoint_id=meta['checkpoint_id'], whole_model_forwards=1, batch_size=4,
        actual_native_L0_down_Flinear=True, warmup=warmup, archive_sha256=c.sha(c.ARCHIVE / 'batches.pt'),
        native_audit_sha256=c.sha(root / 'native_audit.json'), quality_selection=False,
        GPU_native_loader=True, repair_prepared_sha256=c.sha(PREPARED))
    c.write(root / 'gpu_validation.json', proof)
    return proof


def profile_run(profile):
    prepared = check()
    root = REPAIR / profile
    root.mkdir(parents=True, exist_ok=False)
    path = root / 'pipeline.json'
    record = dict(status='repairing_consumer', profile=profile, started_unix=time.time(), stages={}, automatic_retry=False)
    c.write(path, record)
    try:
        entry = [sys.executable, '-B', '-u', str(HERE / 'repair.py')]
        gpu = c.GPUS[profile]['chatml']
        c.check_gpu(gpu)
        record['stages']['consumer'] = original.stage('consumer', [*entry, 'consumer', '--profile', profile], root, gpu=gpu, seconds=1800)
        evaluation, record['chatml_full9_process'] = original.spawn([*entry, 'evaluate', '--profile', profile], root, 'chatml_full9_worker')
        record['status'] = 'waiting_original_06_with_07_full9_independent'
        c.write(path, record)
        donor = prepared['actual_original_KL_donors'][profile]
        processes.wait_for_process(donor['pid'], int(donor['start_ticks']))
        c.require(c.read(c.OUTPUT / profile / '06_decoder_kl/pipeline.json')['status'] == 'complete_native', 'Actual original06 must finish its native audit; no replacement training.')
        gpu = c.GPUS[profile]['decoder']
        c.check_gpu(gpu)
        record['status'] = 'running_08_09'
        c.write(path, record)
        endpoints = [sys.executable, '-B', '-u', str(c.HERE / 'endpoints.py')]
        record['stages']['attention'] = original.stage('attention', [*endpoints, 'attention', '--profile', profile], root, seconds=3600)
        source = c.DATA / profile / '08_joint_mlp_decoder_attention/final_model'
        record['stages']['last_train'] = original.stage('last_train', [sys.executable, '-B', '-u', str(c.HERE / 'train.py'), '--profile', profile, '--recipe', 'last_joint', '--source', str(source)], root, gpu=gpu, seconds=50400)
        record['stages']['last_audit'] = original.stage('last_audit', [*endpoints, 'audit', '--profile', profile, '--recipe', 'last_joint'], root, seconds=3600)
        record['status'] = 'running_final_full9'
        c.write(path, record)
        final = original.evaluate(profile, 'final')
        record.update(final_nine_task_score=final['nine_task_score'], final_evaluation=str(c.OUTPUT / profile / '09_joint1200/evaluation_full9/evaluation.json'))
        record['chatml_full9_process']['exit_code'] = evaluation.wait()
        report = c.read(c.OUTPUT / profile / '07_chatml_joint256/evaluation_full9/evaluation.json')
        record.update(status='complete' if evaluation.returncode == 0 else 'complete_main_with_chatml_evaluation_failure',
            chatml_evaluation_status=report['status'], chatml_nine_task_score=report.get('nine_task_score'))
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        c.write(path, record)


def launch():
    check()
    c.require(not (REPAIR / 'launch.json').exists(), 'Launch the confirmed consumer repair only once.')
    for gpu in (6, 7):
        c.check_gpu(gpu)
    record = dict(status='launching', prepared_sha256=c.sha(PREPARED), started_unix=time.time(), workers={})
    c.write(REPAIR / 'launch.json', record)
    for profile in ('C', 'G'):
        _child, row = original.spawn([sys.executable, '-B', '-u', str(HERE / 'repair.py'), 'profile', '--profile', profile], REPAIR, profile + '_tail')
        record['workers'][profile] = row
        c.write(REPAIR / 'launch.json', record)
    record['status'] = 'two_fixed_consumer_repairs_running'
    c.write(REPAIR / 'launch.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'launch', 'profile', 'consumer', 'evaluate'])
    parser.add_argument('--profile', choices=['C', 'G'])
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'launch':
        launch()
    elif args.command == 'profile':
        profile_run(args.profile)
    elif args.command == 'consumer':
        consume(args.profile)
    else:
        check()
        original.evaluate(args.profile, 'chatml')
