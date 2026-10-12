"""Four real Llama all-norm updates, using the existing native observation/export path."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state, _require
from experiments.qwen3_w25.export_e2e_selected import _sha256
from experiments.w25_eval.norm_state import validate_runtime
from experiments.w25_eval.validate_norm_only_training import NormObservations, expected_inventory

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / 'result/llama31_8b_w25_20261004/final_all_decoder_task50_s400'


def read(path):
    return json.loads(Path(path).read_text())


def training_arguments(output, *, diagnostic):
    argv = list(read(CONTROL / 'launch.json')['argv'][3:])
    changes = {'--run_root_dir': str(Path(output).resolve()), '--train_mode': 'none',
               '--norm_train_mode': 'all', '--distill_fp32_components': 'norm'}
    if diagnostic:
        changes.update({'--save_steps': '2', '--logging_steps': '1'})
    for key, value in changes.items():
        _require(argv.count(key) == 1, 'Ambiguous control argument: ' + key)
        argv[argv.index(key) + 1] = value
    return argv


def configuration(record):
    diagnostic = record['kind'] == 'validate_s4'
    _require(record['kind'] in ('validate_s4', 'formal_s400'), 'Unknown bounded norm run.')
    expected = training_arguments(record['output_dir'], diagnostic=diagnostic)
    _require(record['training_argv'] == expected, 'Only the declared trainable selection/output may change.')
    cfg, hf, args = parse_args(expected)
    meta, architecture = validate_runtime(cfg, args)
    _require(architecture['model_type'] == 'llama' and architecture == record['architecture']
             and meta['checkpoint_id'] == record['source_checkpoint_id']
             and Path(cfg.student_checkpoint_dir).resolve() == Path(record['source_checkpoint']).resolve(),
             'Native Llama architecture/source differs.')
    _require(expected_inventory(cfg.student_checkpoint_dir, architecture) == record['expected_inventory'],
             'Actual norm/frozen inventory differs.')
    _require(args.max_steps == 2000 and args.save_steps == (2 if diagnostic else 400)
             and args.save_total_limit == 1 and args.get_warmup_steps(2000) == 20,
             'Retain the original 2000-step schedule and 20 warmup updates.')
    _require(cfg.opt.batch_size == args.per_device_train_batch_size == 4
             and cfg.opt.gradient_accumulation_steps == args.gradient_accumulation_steps == 1
             and cfg.data.model_max_length == 1024 and args.gradient_checkpointing
             and args.gradient_checkpointing_kwargs == {'use_reentrant': False}, 'Real B4/max1024/GAS1 required.')
    _require(cfg.loss.loss_type == 'kl_top_mass' and cfg.loss.alpha == 0
             and cfg.loss.top_k == 100 and cfg.loss.temperature == 1 and cfg.loss.prompt_loss_weight == .3
             and cfg.opt.learning_rate == 1e-5 and cfg.aux.norm_lr is None
             and cfg.runtime.teacher_output_offload == 'cpu',
             'Retain pure mass-KL/task50 and the existing norm LR fallback to 1e-5.')
    _require(cfg.data.dataset_task == 'sft' and cfg.data.dataset_mix
             == 'edgerazor_ii_7m=0.341,edgerazor_ii_gen=0.067,edgerazor_tulu=0.028,edgerazor_am=0.064,vaellm_eval_task=0.5',
             'The five-source task50 mixture differs.')
    return cfg, hf, args, architecture


def train(record):
    _require(record['kind'] == 'validate_s4', 'This entry runs only the genuine four-update validation.')
    cfg, hf, args, architecture = configuration(record)
    callback = NormObservations(architecture)
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(int(args.seed))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = run(cfg, hf, args, extra_callbacks=(callback,))
        observed, expected = callback.diagnostic, record['expected_inventory']
        active = observed['trainable_inventory']
        _require(result['global_step'] == 4 and result['round_base_checkpoint_id'] == record['source_checkpoint_id'],
                 'Four updates did not finish from the original CAT source.')
        _require(active['mutable_tensors'] == expected['mutable_tensors']
                 and active['norm_elements'] == expected['norm_elements']
                 and active['norm_shapes'] == expected['norm_shapes']
                 and active['frozen_decoder_tensors'] == expected['all_decoder_tensors']
                 and active['frozen_code_banks'] == expected['all_code_banks'], 'Actual trainables escaped the native norms.')
        _require([x['step'] for x in observed['optimizer_steps']] == [1, 2, 3, 4]
                 and [x['step'] for x in observed['native_saves']] == [2, 4]
                 and {x['step'] for x in observed['batches']} == {1, 2, 3, 4}
                 and all(x['shape'][0] == 4 and x['shape'][1] <= 1024 for x in observed['batches'])
                 and all(x['main_moment_dtypes'] == ['torch.float32'] for x in observed['optimizer_steps']),
                 'Real B4 updates/FP32 Adam/save2+4 evidence is incomplete.')
        report = {'status': 'trained', 'training_result': result, 'observations': observed,
                  'expected_inventory': expected, 'architecture': architecture,
                  'elapsed_seconds': time.perf_counter() - started,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'diagnostic_not_quality': True,
                  'resource_coverage': 'Four genuine dynamic batches, not a full-data worst-case bound.'}
        (Path(record['output_dir']) / 'training.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    finally:
        callback.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def verify(record):
    configuration(record)
    output, expected = Path(record['output_dir']), record['expected_inventory']
    trained = read(output / 'training.json')
    audit = read(output / 'step_0004/audit.json')
    exported = read(output / 'step_0004/final_model/export_record.json')
    _require(audit['status'] == 'verified' and audit['source_checkpoint_id'] == record['source_checkpoint_id']
             and audit['train_mode'] == 'none' and audit['norm_train_mode'] == 'all'
             and audit['lm_head_train_mode'] == 'none'
             and audit['mutable_tensor_count'] == expected['mutable_tensors']
             and audit['norm_elements'] == expected['norm_elements']
             and audit['frozen_tensors_compared'] == expected['frozen_tensors']
             and audit['code_bank_count'] == expected['all_code_banks']
             and audit['decoder_tensor_count'] == expected['all_decoder_tensors']
             and audit['changed_code_bits'] == audit['changed_decoder_elements'] == 0
             and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == expected['serialized_tensors']
             and audit['changed_norm_fp32_elements'] > 0 and audit['norm_matches_selected_fp32_cast_exactly']
             and audit['fresh_strict_reload_state_exact'] and audit['rank_zero']
             and audit['actual_tensor_storage_unchanged'], 'Independent norm state audit failed.')
    _require(exported['source_global_step'] == 4 and exported['source_max_steps'] == 2000
             and exported['mutable_restore']['component_counts'] == {'norm': expected['mutable_tensors']}
             and exported['optimizer_steps_executed'] == 0
             and not exported['decoder_finalization_performed'] and not exported['head_fusion_performed'],
             'Independent export changed training or frozen decoder/head state.')
    native = _load_state(Path(trained['training_result']['saved_model_dir']))
    independent = _load_state(output / 'step_0004/final_model')
    count = _assert_state_equal(native, independent, 'Native final versus independent norm step4')
    _require(count == expected['serialized_tensors'], 'Native/independent state inventory differs.')
    del native, independent
    gc.collect()
    report = {'status': 'verified', 'source_checkpoint_id': record['source_checkpoint_id'],
              'checkpoint_id': audit['checkpoint_id'], 'expected_inventory': expected,
              'native_vs_step4_tensors_compared': count,
              'training_report_sha256': _sha256(output / 'training.json'),
              'audit_sha256': _sha256(output / 'step_0004/audit.json'),
              'export_record_sha256': _sha256(output / 'step_0004/final_model/export_record.json'),
              'changed_norm_fp32_elements': audit['changed_norm_fp32_elements'],
              'changed_norm_bf16_elements': audit['changed_norm_bf16_elements'],
              'norm_update_survived_bf16': audit['norm_update_survived_bf16'], 'changed_code_bits': 0,
              'bit_budget': audit['bit_budget'], 'diagnostic_not_quality': True,
              'export_peak_allocated_bytes': exported['peak_cuda_allocated_bytes'],
              'export_peak_reserved_bytes': exported['peak_cuda_reserved_bytes']}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=('train', 'verify'))
    args = parser.parse_args()
    {'train': train, 'verify': verify}[args.phase](read(args.launch_record))


if __name__ == '__main__':
    main()
