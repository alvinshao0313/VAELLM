"""Evaluate true native7200 parameters with one-time packed BF16 weight caches."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import torch

from experiments.fp32_decoder_export.export import (
    REPO, SOURCE_FILES as EXPORT_SOURCES, inspect_scope, precision_inventory, rope_record,
)
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.export_e2e_selected import restore_decoder_state
from experiments.w25_eval.acceptance import TASK_METRICS, _group_agrees, collect_results, leaf_metrics
from experiments.w25_eval.subset_state import persistent_inventory

SOURCE_FILES = tuple(dict.fromkeys((*EXPORT_SOURCES,
    'experiments/packed_decode_evaluation/prepare.py', 'experiments/packed_decode_evaluation/run.py',
    'experiments/liftquant_recovery/recovery_runtime.py', 'train_utils/eval_utils.py',
    'train_utils/checkpoint_bpw.py', 'experiments/w25_eval/acceptance.py',
    'experiments/qwen3_w25/summarize_results.py', 'experiments/longbench_v1/common.py',
    'litebsq/packed_bit_linear.py', 'litebsq/parallel_layers.py')))
SOURCE_ID = '552282f2-868a-4105-90df-46f3c6f6a818'
NATIVE_ID = 'ff8ef13e-a975-4a85-86f2-d26417a3e416'
CONTROL_ID = '9439e164-04a0-481d-8320-7570b3af9c55'
SHARDS = (('mmlu', 'mmlu', 64),
          ('seven', 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa', 32))


def read(path):
    return json.loads(Path(path).read_text())


def check(record):
    _require({str(REPO / name) for name in SOURCE_FILES} <= set(record['source_sha256']),
             'Pin every new entry and its shared dependencies.')
    check_hashes(record['source_sha256']); check_hashes(record['input_evidence_sha256'])
    for row in record['protected_weight_stats']:
        check_stat(row)


def control_results(record):
    directory = Path(record['control_checkpoint_dir']).resolve()
    meta = read(directory / 'checkpoint_meta.json')
    audit = read(Path(record['control_result_paths'][0]).parent / 'audit.json')
    _require(meta['checkpoint_id'] == record['control_checkpoint_id'] == CONTROL_ID
             and audit['selected_checkpoint_id'] == record['native_checkpoint_id'] == NATIVE_ID
             and audit['source_checkpoint_id'] == record['source_checkpoint_id'] == SOURCE_ID
             and audit['status'] == 'verified' and audit['fp32_native_decoder_tensors_exact'] == 1512
             and audit['frozen_source_tensors_exact'] == 1911 and audit['frozen_code_banks'] == 504
             and audit['source_global_step'] == 7200
             and audit['native_training_model_state_sha256'] == sha256(
                 Path(record['training_checkpoint_dir']) / 'training_model_state.pt'),
             'Require the reviewed same-native FP32 wholefused control, not BF16 storage.')
    # The retired control weights are unnecessary: this only validates original raw results.
    result = collect_results(record['control_result_paths'], record['architecture'], checkpoint=(directory, meta))
    return meta, audit, result


def collect_memory_results(paths, record):
    scores, subjects, inputs = {}, {}, []
    for path, (name, tasks, batch) in zip(paths, SHARDS, strict=True):
        payload = read(path)
        _require(payload['checkpoint_loader'] == 'v6_source_plus_native_mutable_in_memory'
                 and payload['checkpoint_id'] == record['native_checkpoint_id']
                 and payload['checkpoint_dir'] == record['training_checkpoint_dir']
                 and payload['strict_source_load'] and payload['native_mutable_restore_exact']
                 and payload['source_checkpoint_id'] == record['source_checkpoint_id']
                 and payload['base_model_path'] == record['architecture']['base_model_path']
                 and payload['tokenizer_class'] == record['architecture']['tokenizer_class']
                 and payload['tokenizer_is_fast'] and payload['eval_hif4_act'] is False,
                 'Wrong in-memory source/native identity or tokenizer.')
        result = payload['evals']['lm_eval']
        _require(result['tasks'] == tasks.split(',') and result['batch_size'] == batch
                 and result['num_fewshot'] == 0 and result['limit'] is None,
                 'Require complete zero-shot M64/seven32 evaluation.')
        current, sub, actual_batch = leaf_metrics(result['raw_results'], result['n_samples'], result['n_shot'], result['configs'])
        _group_agrees(current, result['raw_results'], result['group_results'])
        _require(actual_batch == batch and set(result['task_metrics']) == set(current)
                 and set(result['task_metric_keys']) == set(current) and not set(scores) & set(current),
                 'Wrong or duplicate result inventory.')
        for task, value in current.items():
            _require(result['task_metric_keys'][task] == TASK_METRICS[task]
                     and math.isclose(result['task_metrics'][task], value, abs_tol=1e-12, rel_tol=0),
                     'Summary differs from actual raw metrics.')
        scores.update(current); subjects.update(sub)
        inputs.append({'path': str(path), 'sha256': sha256(path), 'batch_size': batch,
                       'leaf_count': len(result['n_samples']),
                       'sample_count': sum(row['effective'] for row in result['n_samples'].values())})
    _require(set(scores) == set(TASK_METRICS) and len(subjects) == 57
             and sum(row['leaf_count'] for row in inputs) == 64
             and sum(row['sample_count'] for row in inputs) == 24742, 'Incomplete eight-task evaluation.')
    return {'task_metrics': scores, 'task_metric_keys': TASK_METRICS, 'mmlu_subject_metrics': subjects,
            'mean_percent': math.fsum(scores.values()) * 100 / 8, 'input_results': inputs,
            'leaf_count': 64, 'sample_count': 24742}


def cache_inventory(selected):
    result = {}
    for name, module in selected:
        value = module._cached_weight
        _require(not module.trainable_decode and module.cache_decoded_weight
                 and value is not None and value.dtype == torch.bfloat16 and value.device.type == 'cuda'
                 and tuple(value.shape) == (module.out_features, module.in_features)
                 and not value.requires_grad, 'Invalid materialized training-path cache: ' + name)
        result[name] = {'pointer': value.data_ptr(), 'shape': list(value.shape), 'dtype': str(value.dtype),
                        'bytes': value.numel() * value.element_size()}
    return result


def run(record_path):
    from compressed_e2e_fintuning.runtime_v6 import _build_v6_tokenizer, _finalize_decoders, _sync_model_padding_config
    from compressed_e2e_fintuning.runtime_v6_pipeline import (
        _assert_finalization_probe_close, _build_finalization_probe_inputs, _run_finalization_probe,
    )
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from experiments.liftquant_recovery.recovery_runtime import prime_packed_cache
    from litebsq.vae_linear import VAELinear, clear_model_vae_linear_cache
    from train_utils.checkpoint_bpw import audit_state_dict
    from train_utils.checkpoint_v6 import load_v6_training_model_state
    from train_utils.config.targets import collect_e2e_compressed_targets
    from train_utils.eval_utils import run_lm_eval
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    record_path = Path(record_path).resolve(); record = read(record_path)
    output = Path(record['output_dir']).resolve(); code = 1

    @contextmanager
    def phase(name):
        record['status'] = name
        record['phases'][name] = {'started_unix': time.time(), 'same_process_pid': os.getpid()}
        write_json(record_path, record)
        value = 1
        try:
            yield
            value = 0
        finally:
            (output / (name + '.exit')).write_text(str(value) + '\n')
            record['phases'][name].update(exit_code=value, finished_unix=time.time())
            write_json(record_path, record)

    try:
        _require(record['status'] == 'bound_ready' and type(record['gpu']) is int
                 and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                 and Path.cwd().resolve() == Path(record['cwd']).resolve() == REPO
                 and not record['queued'] and not record['automatic_retry'], 'Use one root-bound invocation.')
        _require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                 and all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'))
                 and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                 'Use the pinned offline and normal numerical environment.')
        _require(output == record_path.parent and not any((output / name).exists() for name in
                 ('audit.json', 'mmlu.json', 'seven.json', 'full8_summary.json', 'dispatcher.exit')),
                 'Never overwrite a previous evaluation.')
        check(record)
        _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose one bound CUDA device.')
        torch.set_num_threads(4); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(0)
        record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(), phases={},
                      worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
                      actual_environment={key: os.environ.get(key) for key in record['environment']})
        write_json(record_path, record)
        with phase('restore'):
            native = Path(record['training_checkpoint_dir'])
            meta, state, snapshot, source, base, architecture = inspect_scope(native)
            _require(meta['checkpoint_id'] == NATIVE_ID and base['checkpoint_id'] == SOURCE_ID
                     and str(source) == record['source_checkpoint_dir'] and architecture == record['architecture'],
                     'Actual native7200 and original source must match preparation.')
            control_meta, control_audit, control = control_results(record)
            training = SimpleNamespace(**snapshot['training_args'])
            configure_e2e_determinism(bool(training.full_determinism)); set_e2e_seed(training.seed)
            if training.tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = bool(training.tf32)
                torch.backends.cudnn.allow_tf32 = bool(training.tf32)
            model, loaded_meta, loaded = load_v6_model_checkpoint(str(source), map_location='cpu', strict=True, expected_kind='final_model')
            _require(loaded_meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys,
                     'Original full source strict load failed.')
            original = _load_state(source)
            inventory, banks = persistent_inventory(model, architecture)
            keys = {key for values in inventory.values() for key in values}
            selected = collect_e2e_compressed_targets(model, target_layers=tuple(range(36)),
                target_modules=tuple(meta['target_modules']), num_layers=36)
            _require(len(selected) == 252 and len(keys) == 1512 and sum(map(len, banks.values())) == 504,
                     'Unexpected actual inventory.')
            rope = rope_record(model)
            _require(restore_decoder_state(model, selected, checkpoint=native, snapshot=snapshot) == 1512,
                     'Incomplete true FP32 decoder restore.')
            mutable, manifest = load_v6_training_model_state(str(native), map_location='cpu')
            _require(set(mutable) == keys and len(manifest) == 1512, 'Native mutable set differs.')
            expected = dict(original); expected.update(mutable)
            _require(_assert_state_equal(model.state_dict(), expected, 'In-memory source plus true native7200') == 3423,
                     'Complete in-memory persistent state differs.')
            precision = precision_inventory(model, keys)
            _require(precision['rope'] == rope, 'FP32 native RoPE changed.')
            budget = audit_state_dict(control_meta, model.state_dict(), max_bpw=2.5, max_lora_rank=0)
            budget_keys = ('original_linear_weights', 'user_linear_bits', 'packed_linear_bits', 'components',
                           'outside_linear_bits', 'full_model_tensor_bytes')
            _require(budget['budget_passed'] and not budget['contract_violations']
                     and all(budget[k] == control_audit['bit_budget'][k] for k in budget_keys),
                     'Actual runtime storage differs from the reviewed same-native FP32 state.')
            tokenizer = _build_v6_tokenizer(str(base['base_model_path']), SimpleNamespace(**snapshot['hf_args']))
            _sync_model_padding_config(model, tokenizer); model.config.use_cache = False
            model.eval().to('cuda:0')
            _require(type(tokenizer).__name__ == architecture['tokenizer_class'] and tokenizer.is_fast
                     and model.config._attn_implementation == 'flash_attention_2', 'Native model/tokenizer policy changed.')
        with phase('cache'):
            probe_inputs = _build_finalization_probe_inputs(tokenizer)
            VAELinear.reset_fuse_stats()
            trained, dtype = _run_finalization_probe(model, probe_inputs)
            training_stats = VAELinear.get_fuse_stats()
            record['training_probe_stats'] = training_stats
            record['training_probe_wholefused_hits_inferred'] = training_stats['hit'] - training_stats['packed_u8_linear_hit']
            write_json(record_path, record)
            # Both successful packed and whole-decoder paths increment `hit`.
            _require(dtype == torch.bfloat16 and training_stats['packed_u8_linear_hit'] == 252
                     and training_stats['hit'] == 252 and training_stats['miss'] == 0,
                     'The before-finalization probe must actually use training packed decode.')
            _require(_finalize_decoders(selected) == 252, 'Incomplete decoder freeze.')
            model.requires_grad_(False)
            VAELinear.reset_fuse_stats()
            prime_packed_cache(model)
            materialization_stats = VAELinear.get_fuse_stats()
            record['materialization_stats'] = materialization_stats
            record['materialization_wholefused_hits_inferred'] = materialization_stats['hit'] - materialization_stats['packed_u8_linear_hit']
            write_json(record_path, record)
            _require(materialization_stats['packed_u8_linear_hit'] == 252
                     and materialization_stats['hit'] == 252 and materialization_stats['miss'] == 0,
                     'Materialization fell back to the deployment wholefused path.')
            caches = cache_inventory(selected)
            _require(sum(v['bytes'] for v in caches.values()) == 13891534848, 'Wrong complete BF16 Linear cache size.')
            cached, cached_dtype = _run_finalization_probe(model, probe_inputs)
            _require(cached_dtype == dtype and cache_inventory(selected) == caches
                     and VAELinear.get_fuse_stats() == materialization_stats, 'Cached probe regenerated decoder weights.')
            parity = _assert_finalization_probe_close(trained, cached, output_dtype=dtype,
                                                       label='Actual training decode versus packed materialized cache')
            audit = {'status': 'cache_verified', 'checkpoint_loader': 'v6_source_plus_native_mutable_in_memory',
                'source_checkpoint_dir': str(source), 'source_checkpoint_id': SOURCE_ID,
                'training_checkpoint_dir': str(native), 'native_checkpoint_id': NATIVE_ID,
                'control_metadata_reference': record['control_checkpoint_dir'], 'control_checkpoint_id': CONTROL_ID,
                'strict_source_load': True, 'native_mutable_restore_exact': True, 'persistent_tensors_exact': 3423,
                'native_fp32_decoder_tensors': 1512, 'frozen_source_tensors': 1911, 'frozen_code_banks': 504,
                'precision': precision, 'bit_budget_from_actual_runtime_state': budget,
                'training_probe_stats': training_stats, 'materialization_stats': materialization_stats,
                'training_probe_wholefused_hits_inferred': record['training_probe_wholefused_hits_inferred'],
                'materialization_wholefused_hits_inferred': record['materialization_wholefused_hits_inferred'],
                'training_to_cached_probe_parity': parity, 'cache_count': len(caches),
                'cache_bytes': sum(v['bytes'] for v in caches.values()), 'cache': caches,
                'probe_input_tokens': probe_inputs['input_ids'].tolist(), 'cache_reused_by_probe': True,
                'optimizer_steps': 0, 'new_checkpoint_or_cache_file_saved': False,
                'scope': 'Actual native FP32 values; BF16 packed decoder arithmetic, cached only in process RAM. No independent final-model load from the retired control directory.'}
            write_json(output / 'audit.json', audit)
        paths = []
        for name, tasks, batch in SHARDS:
            with phase(name):
                _require(cache_inventory(selected) == caches and VAELinear.get_fuse_stats() == materialization_stats,
                         'Cache was replaced before evaluation.')
                result = run_lm_eval(model, tokenizer, SimpleNamespace(model_path=str(base['base_model_path']),
                    tasks=tasks, num_fewshot=0, batch_size=batch, lm_limit=None, mmlu_debug_samples=0))
                payload = {'checkpoint_dir': str(native), 'checkpoint_id': NATIVE_ID,
                    'source_checkpoint_dir': str(source), 'source_checkpoint_id': SOURCE_ID,
                    'base_model_path': str(base['base_model_path']),
                    'checkpoint_loader': 'v6_source_plus_native_mutable_in_memory',
                    'strict_source_load': True, 'native_mutable_restore_exact': True,
                    'adapter_dir': None, 'eval_hif4_act': False,
                    'tokenizer_class': type(tokenizer).__name__, 'tokenizer_is_fast': bool(tokenizer.is_fast),
                    'decode_policy': 'one-time native packed BF16 cache, real FP32 native decoder values',
                    'evals': {'lm_eval': result}}
                path = output / (name + '.json'); write_json(path, payload); paths.append(path)
                _require(cache_inventory(selected) == caches and VAELinear.get_fuse_stats() == materialization_stats,
                         'Full evaluation replaced caches or executed a decoder again.')
                reference = read(record['control_result_paths'][len(paths)-1])['evals']['lm_eval']
                for field in ('configs', 'versions', 'n_shot', 'n_samples'):
                    _require(result[field] == reference[field], name + ': control protocol differs: ' + field)
                audit[name + '_cache_unchanged'] = True; write_json(output / 'audit.json', audit)
        with phase('summary'):
            actual = collect_memory_results(paths, record)
            _require(rope_record(model) == rope, 'Evaluation changed native FP32 RoPE.')
            clear_model_vae_linear_cache(model); model.to('cpu')
            _require(_assert_state_equal(model.state_dict(), expected, 'All persistent state after evaluation') == 3423,
                     'Inference changed persistent parameters or codes.')
            _require(precision_inventory(model, keys) == precision, 'Inference changed persistent precision.')
            check(record)
            audit.update(status='verified', persistent_state_after_full8_exact=True,
                source_and_input_pins_unchanged=True, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0))
            write_json(output / 'audit.json', audit)
            summary = {'status': 'full8_evaluation_verified', 'native_checkpoint_id': NATIVE_ID,
                'source_checkpoint_id': SOURCE_ID, 'control_checkpoint_id': CONTROL_ID, **actual,
                'control_mean_percent': control['mean_percent'],
                'mean_delta_percentage_points': actual['mean_percent'] - control['mean_percent'],
                'task_delta_percentage_points': {task: 100*(value-control['task_metrics'][task])
                                                 for task, value in actual['task_metrics'].items()},
                'control_results': control['input_results'], 'audit_sha256': sha256(output/'audit.json'),
                'actual_configs_versions_shots_counts_equal_control': True,
                'user_linear_bpw': budget['user_linear_bpw'], 'mmlu_65_passed': actual['task_metrics']['mmlu'] >= .65,
                'longbench_status': 'not_evaluated', 'current_nine_item_goal_status': 'not_evaluated',
                'goal_acceptance': False, 'scope': audit['scope']}
            write_json(output / 'full8_summary.json', summary)
            record.update(full8_summary_sha256=sha256(output/'full8_summary.json'),
                          mean_percent=actual['mean_percent'], mmlu_percent=100*actual['task_metrics']['mmlu'],
                          user_linear_bpw=budget['user_linear_bpw'])
        record['status'] = 'completed'; code = 0
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        record.update(exit_code=code, finished_unix=time.time())
        write_json(record_path, record); (output/'dispatcher.exit').write_text(str(code)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(args.launch_record)
