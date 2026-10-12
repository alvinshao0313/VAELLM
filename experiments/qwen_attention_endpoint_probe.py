"""One existing-weight endpoint reconstruction probe; no training or full checkpoint export."""
import argparse
from copy import deepcopy
import gc
import importlib.metadata
import logging
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from transformers import AutoTokenizer

from experiments.qwen_attention_decoded_target_pair.common import ROOT, PUBLIC, RESULT, NAMES, PROTECTION, read, check
from experiments.qwen_attention_decoded_target_pair.endpoint import decoded_records, save_and_reconstruct
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _replace_modules, _require
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from train_utils.utils import configure_deterministic_mode, set_seed
from train_utils.v6_model_loader import load_v6_model_checkpoint


def check_probe(record):
    _require(record['schema'] == 'existing_qwen_attention_endpoint_probe_v1' and record['gpu'] == 0
             and not record['training'] and not record['queued'] and not record['automatic_retry'], 'Unexpected probe scope.')
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    for item in record['protected_stats']:
        check_stat(item)


def prepare(output):
    _require(ROOT != PUBLIC and Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
             and not output.exists(), 'Prepare one fresh CPU record in S.')
    formal_path = RESULT / 'qwen_attention_decoded_target_pair/hf/launch.prepared.json'
    formal = read(formal_path); check(formal)
    candidate_path = RESULT / 'qwen_attention_int4_pair/candidate/launch.json'; candidate = read(candidate_path)
    donor = Path(candidate['donor_checkpoint_dir']); meta = read(donor / 'checkpoint_meta.json')
    _require(candidate['status'] == 'completed' and candidate['exit_code'] == 0
             and len(meta['converted_modules']) == 144 and meta['checkpoint_id'] == candidate['donor_checkpoint_id'],
             'Require the actual completed K+8 donor.')
    source = Path(__file__).resolve()
    _require(sha256(source) == sha256(PUBLIC / source.relative_to(ROOT)), 'Probe public/S differs.')
    inputs = dict(formal['input_sha256'])
    for path in (formal_path, candidate_path, donor / 'checkpoint_meta.json'):
        inputs[str(path)] = sha256(path)
    record = {'schema': 'existing_qwen_attention_endpoint_probe_v1', 'status': 'bound_ready', 'gpu': 0,
        'training': False, 'queued': False, 'automatic_retry': False, 'cwd': str(ROOT),
        'python_executable': sys.executable, 'formal_record': str(formal_path), 'output_dir': str(output),
        'donor_checkpoint_dir': str(donor), 'donor_checkpoint_id': meta['checkpoint_id'],
        'source_sha256': {**formal['source_sha256'], str(source): sha256(source)}, 'input_sha256': inputs,
        'protected_stats': [*formal['protected_stats'], file_stat(donor / meta['state_dict_file'])],
        'environment': formal['environment'], 'timeout_seconds': 600, 'termination_grace_seconds': 20,
        'boundary': 'Existing candidate144 attention plus f9271551 complement. Validate endpoint geometry/save/rebuild/group8/full forward only; candidate vL4/vL26 differ from formal fixed plans. No new training or fixed-plan training validation.'}
    record['launcher_argv'] = ['timeout', '--signal=TERM', '--kill-after=20', '600', sys.executable, '-u',
        '-m', 'experiments.qwen_attention_endpoint_probe', '--launch-record', str(output / 'launch.json')]
    check_probe(record); _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'prepared_sha256': sha256(output / 'launch.prepared.json')}, flush=True)


def run(path):
    record = read(path); output = Path(record['output_dir']); code = 1
    _require(record['status'] == 'bound_ready' and Path.cwd() == ROOT
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
             and all(os.environ.get(k) == v for k, v in record['environment'].items())
             and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
             and not (output / 'endpoint').exists(), 'Require the unique explicitly bound probe.')
    check_probe(record)
    report = {'status': 'starting', 'training': False, 'started_unix': time.time(), 'boundary': record['boundary']}
    try:
        _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Require one actual GPU0.')
        formal = read(record['formal_record']); check(formal)
        for name, version in formal['package_versions'].items():
            _require(importlib.metadata.version(name) == version, 'Package changed: ' + name)
        configure_deterministic_mode(True); set_seed(31); torch.cuda.reset_peak_memory_stats()
        record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
        write_json(path, record)
        donor, meta, loaded = load_v6_model_checkpoint(record['donor_checkpoint_dir'], map_location='cpu',
                                                      strict=True, expected_kind='final_model')
        _require(meta['checkpoint_id'] == record['donor_checkpoint_id']
                 and not loaded.missing_keys and not loaded.unexpected_keys, 'Actual native donor identity failed.')
        donor.eval().requires_grad_(False)
        reference = decoded_records(donor, formal, torch.device('cuda:0'))
        print('Actual original donor144 group8 reference captured.', flush=True)
        prefixes = tuple(name + '.' for name in NAMES)
        selected = {key: value for key, value in donor.state_dict().items() if key.startswith(prefixes)}
        training_context = {'materialization': {'targets': {name: {
            field: tensor_record(selected[name + '.' + field]) for field in PROTECTION} for name in NAMES}}}
        base, base_meta, base_loaded = load_v6_model_checkpoint(formal['base_dir'], map_location='cpu', strict=True,
                                                              expected_kind='final_model')
        _require(base_meta['checkpoint_id'] == formal['base_checkpoint_id']
                 and not base_loaded.missing_keys and not base_loaded.unexpected_keys, 'Actual source load failed.')
        base_state = torch.load(Path(formal['base_dir']) / base_meta['state_dict_file'],
                                map_location='cpu', mmap=True, weights_only=True)
        outside = {key: value for key, value in base_state.items() if not key.startswith(prefixes)}
        _require(len(selected) == 1872 and len(outside) == 1551, 'Wrong actual source union inventories.')
        _replace_modules(base, donor, NAMES)
        _assert_state_equal(base.state_dict(), {**outside, **selected}, 'Actual donor1872 plus f9271551')
        del donor, outside; gc.collect(); torch.cuda.empty_cache()
        differences = {name: {'f927_only': sorted(set(base_state[name + '.protected_input_indices'].tolist())
                            - set(selected[name + '.protected_input_indices'].tolist())),
                'candidate_only': sorted(set(selected[name + '.protected_input_indices'].tolist())
                            - set(base_state[name + '.protected_input_indices'].tolist()))}
            for name in NAMES if not torch.equal(selected[name + '.protected_input_indices'],
                                                 base_state[name + '.protected_input_indices'])}
        _require(differences == {
            'model.layers.4.self_attn.v_proj': {'f927_only': [2178], 'candidate_only': [1984]},
            'model.layers.26.self_attn.v_proj': {'f927_only': [1773], 'candidate_only': [3327]}},
            'Actual candidate/f927 known plan differences changed.')
        endpoint_record = deepcopy(formal)
        endpoint_record.update(output_dir=str(output / 'endpoint'), target_kind='existing_candidate_endpoint_probe')
        (output / 'endpoint').mkdir()
        fresh, _meta, audit = save_and_reconstruct(base, base_state, endpoint_record, training_context,
                                                  torch.device('cuda:0'))
        _require(reference == audit['native_group8_weight_records'], 'New endpoint differs from original native donor group8.')
        write_json(output / 'endpoint_audit.json', audit)
        del base, base_state, selected, training_context; gc.collect(); torch.cuda.empty_cache()
        print('Actual3423 strict endpoint reconstruction and original donor144 group8 equality passed.', flush=True)
        from tools.cat_eval import _prepare_model_for_eval
        warmup = _prepare_model_for_eval(fresh, 'cuda:0', 8, logging.getLogger(__name__), {})
        _require(warmup['total'] == 252 and warmup['warmed'] == 252 and warmup['failed'] == 0, 'Incomplete native252 cache.')
        for name in NAMES:
            _require(tensor_record(fresh.get_submodule(name)._cached_weight) == reference[name], 'Actual native evaluation cache differs.')
        tokenizer = AutoTokenizer.from_pretrained(formal['hf_path'], use_fast=True, local_files_only=True)
        ids = tokenizer('A short test sentence.', return_tensors='pt', add_special_tokens=False)['input_ids'][:, :4]
        _require(ids.shape == (1, 4), 'Four-token integration input differs.')
        with torch.inference_mode(): logits = fresh(input_ids=ids.to('cuda:0'), use_cache=False).logits
        policy = formal['model_policy']
        _require(logits.dtype == torch.bfloat16 and bool(torch.isfinite(logits).all())
                 and logits.shape == (1, 4, fresh.config.vocab_size)
                 and fresh.config._attn_implementation == policy['attn_implementation']
                 and fresh.config.model_type == policy['model_type'], 'Native complete model forward differs.')
        report.update(status='verified_existing_endpoint_gpu', original_donor_group8_exact_projections=144,
            endpoint_union_tensors=3423, actual_plan_differences=differences, fixed_plan_training_validated=False,
            native_warmup=warmup, four_token_logits=tensor_record(logits), model_policy=policy,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            endpoint_audit_sha256=sha256(output / 'endpoint_audit.json'), native_full_checkpoint_exported=False)
        check_probe(record); record['status'] = 'completed'; code = 0
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        record.update(status='failed', error=report['error']); raise
    finally:
        report.update(exit_code=code, elapsed_seconds=time.time() - report['started_unix'])
        write_json(output / 'report.json', report)
        record.update(exit_code=code, finished_unix=time.time(), report_sha256=sha256(output / 'report.json'))
        write_json(path, record); (output / 'run.exit').write_text(str(code) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path); mode.add_argument('--launch-record', type=Path)
    args = parser.parse_args()
    prepare(args.prepare.resolve()) if args.prepare else run(args.launch_record.resolve())
