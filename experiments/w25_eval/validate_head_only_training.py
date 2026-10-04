"""Four native head updates/save2+4, then same-card independent step4 export verification.

The existing training observations collect real batches and Adam dtypes. No
mid-training export, alternate backward, decoder finalization or resume is added.
"""
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
from experiments.qwen3_w25.aux_step_export_callback import AuxStepExportCallback
from experiments.qwen3_w25.export_e2e_selected import _sha256
from experiments.w25_eval.head_state import HEAD_KEY, load_head_state
from experiments.w25_eval.subset_state import persistent_inventory
from experiments.w25_eval.train_head_only import validate_runtime
from experiments.w25_eval.validate_wide_subset_training import TrainingObservations


def expected_inventory(source_dir, architecture):
    state = _load_state(Path(source_dir))
    decoders = {name: [key for key in state if key.startswith(name + '._parallel_stage_decoder.')]
                for name in architecture['projection_shapes']}
    banks = [name + suffix for name in decoders for suffix in ('.vq_weight', '.vq_weight_s1')]
    _require(all(len(keys) == 6 for keys in decoders.values())
             and all(key in state and state[key].dtype == torch.uint8 for key in banks),
             'Source must contain every native six-leaf/two-bank compressed projection.')
    head = state[HEAD_KEY]
    _require(head.ndim == 2 and head.dtype == torch.bfloat16, 'The source head must be BF16.')
    return {'deployed_projections': len(decoders), 'serialized_tensors': len(state),
        'all_decoder_tensors': sum(map(len, decoders.values())), 'all_code_banks': len(banks),
        'mutable_tensors': 1, 'head_elements': head.numel(), 'head_shape': list(head.shape),
        'frozen_tensors': len(state) - 1}


def configuration(record):
    cfg, hf, args = parse_args(record['training_argv'])
    meta, architecture = validate_runtime(cfg, args)
    _require(Path(cfg.student_checkpoint_dir).resolve() == Path(record['source_checkpoint']).resolve()
             and meta['checkpoint_id'] == record['source_checkpoint_id']
             and Path(cfg.run_root_dir).resolve() == Path(record['output_dir']).resolve(), 'Source/output binding differs.')
    _require(expected_inventory(cfg.student_checkpoint_dir, architecture) == record['expected_inventory'],
             'Prepared head/frozen inventory differs from actual source storage.')
    _require(args.max_steps == 2000 and args.save_steps == 2 and args.save_total_limit == 1
             and args.get_warmup_steps(2000) == 20, 'Keep the native2000/warmup20 schedule and save2.')
    _require(cfg.opt.batch_size == args.per_device_train_batch_size == 4
             and cfg.opt.gradient_accumulation_steps == args.gradient_accumulation_steps == 1
             and cfg.data.model_max_length == 1024 and args.gradient_checkpointing
             and args.gradient_checkpointing_kwargs == {'use_reentrant': False}, 'Keep real B4/max1024/nonreentrant checkpointing.')
    _require(cfg.loss.loss_type == 'kl_top_mass' and cfg.loss.top_k == 100
             and cfg.loss.temperature == 1 and cfg.loss.prompt_loss_weight == .3
             and cfg.aux.lm_head_lr == 3e-5 and cfg.runtime.teacher_output_offload == 'cpu',
             'Short validation must retain the specified headLR/task50 KL/CPU teacher-output recipe.')
    _require(cfg.data.dataset_task == 'sft' and cfg.data.dataset_mix
             == 'edgerazor_ii_7m=0.341,edgerazor_ii_gen=0.067,edgerazor_tulu=0.028,edgerazor_am=0.064,vaellm_eval_task=0.5',
             'Keep the original five-source task50 mixture.')
    return cfg, hf, args, architecture


class HeadObservations(TrainingObservations):
    def __init__(self, architecture):
        super().__init__(())
        self.architecture = architecture

    def on_train_begin(self, args, state, control, **kwargs):
        AuxStepExportCallback.on_train_begin(self, args, state, control, **kwargs)
        _require(state.global_step == 0 and args.max_steps == 2000 and args.save_steps == 2
                 and args.get_warmup_steps(2000) == 20, 'Require a fresh four-step native validation.')
        model = kwargs['model']
        inventory, banks = persistent_inventory(model, self.architecture)
        active = {name: p for name, p in model.named_parameters() if p.requires_grad}
        _require(set(active) == {HEAD_KEY} and active[HEAD_KEY].dtype == torch.float32
                 and not hasattr(model, 'sparse_bit_tuning'), 'Only the FP32 head may be trainable.')
        optimizer = kwargs['optimizer']
        while hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer
        _require(not hasattr(optimizer, 'bit_optimizer')
                 and {id(p) for group in optimizer.param_groups for p in group['params']} == {id(active[HEAD_KEY])},
                 'The optimizer must contain only the true head parameter.')
        self.diagnostic['trainable_inventory'] = {'keys': [HEAD_KEY], 'mutable_tensors': 1,
            'head_elements': active[HEAD_KEY].numel(), 'head_shape': list(active[HEAD_KEY].shape),
            'frozen_decoder_tensors': sum(map(len, inventory.values())),
            'frozen_code_banks': sum(map(len, banks.values())), 'sparse_runtime_present': False}
        self.diagnostic['input_gradient_note'] = (
            'Native runtime unconditionally calls enable_input_require_grads; '
            'frozen backbone parameters do not imply absence of activation backward.')

        def observe_inputs(_module, positional, values):
            inputs = values.get('input_ids')
            if inputs is None and positional:
                inputs = positional[0]
            if inputs is not None:
                mask = values.get('attention_mask')
                self.diagnostic['batches'].append({'step': self._observed_step, 'shape': list(inputs.shape),
                    'padded_tokens': inputs.numel(), 'attention_tokens': None if mask is None else int(mask.sum().item())})
        self._input_hook = model.register_forward_pre_hook(observe_inputs, with_kwargs=True)
        return control

    def on_save(self, args, state, control, **kwargs):
        result = super().on_save(args, state, control, **kwargs)
        checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        mutable, _manifest = load_head_state(checkpoint)
        _require(not (checkpoint / 'sparse_bit_tuning').exists(), 'Head-only native save contains score state.')
        self.diagnostic['native_saves'][-1].update(mutable_keys=list(mutable), mutable_dtype=str(mutable[HEAD_KEY].dtype))
        return result


def train(record):
    cfg, hf, args, architecture = configuration(record)
    callback = HeadObservations(architecture)
    configure_e2e_determinism(bool(args.full_determinism))
    set_e2e_seed(int(args.seed))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = run(cfg, hf, args, extra_callbacks=(callback,))
        observed = callback.diagnostic
        expected = record['expected_inventory']
        _require(result['global_step'] == 4 and result['round_base_checkpoint_id'] == record['source_checkpoint_id'],
                 'Native training did not complete four updates from the specified fresh source.')
        _require(observed['trainable_inventory']['head_elements'] == expected['head_elements']
                 and observed['trainable_inventory']['frozen_decoder_tensors'] == expected['all_decoder_tensors']
                 and observed['trainable_inventory']['frozen_code_banks'] == expected['all_code_banks'], 'Training inventory differs.')
        _require([x['step'] for x in observed['optimizer_steps']] == [1, 2, 3, 4]
                 and [x['step'] for x in observed['native_saves']] == [2, 4]
                 and {x['step'] for x in observed['batches']} == {1, 2, 3, 4}
                 and all(x['shape'][0] == 4 and x['shape'][1] <= 1024 for x in observed['batches'])
                 and all(x['main_moment_dtypes'] == ['torch.float32'] for x in observed['optimizer_steps']),
                 'Real B4 forward/FP32 Adam/save2+4 evidence is incomplete.')
        report = {'status': 'trained', 'training_result': result, 'observations': observed,
            'expected_inventory': expected, 'architecture': architecture,
            'elapsed_seconds': time.perf_counter() - started,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'diagnostic_not_quality': True, 'resource_coverage': 'Four actual dynamic batches, not a full-data worst-case bound.'}
        (Path(record['output_dir']) / 'training.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    finally:
        callback.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def verify(record):
    configuration(record)
    expected, output = record['expected_inventory'], Path(record['output_dir'])
    trained = json.loads((output / 'training.json').read_text())
    audit = json.loads((output / 'step_0004/audit.json').read_text())
    exported = json.loads((output / 'step_0004/final_model/export_record.json').read_text())
    _require(audit['status'] == 'verified' and audit['source_checkpoint_id'] == record['source_checkpoint_id']
             and audit['train_mode'] == 'none' and audit['lm_head_train_mode'] == 'full'
             and audit['frozen_tensors_compared'] == expected['frozen_tensors']
             and audit['code_bank_count'] == expected['all_code_banks']
             and audit['decoder_tensor_count'] == expected['all_decoder_tensors']
             and audit['changed_code_bits'] == audit['changed_decoder_elements'] == 0
             and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == expected['serialized_tensors']
             and audit['changed_head_fp32_elements'] > 0 and audit['head_matches_selected_fp32_cast_exactly']
             and audit['fresh_strict_reload_state_exact'] and audit['rank_zero'] and audit['actual_tensor_storage_unchanged'],
             'Independent head-only export audit failed.')
    _require(exported['source_global_step'] == 4 and exported['source_max_steps'] == 2000
             and exported['mutable_restore']['component_counts'] == {'lm_head': 1}
             and exported['optimizer_steps_executed'] == 0
             and not exported['decoder_finalization_performed'] and not exported['head_fusion_performed'],
             'Export must restore exactly the head without training or decoder mutation.')
    native = _load_state(Path(trained['training_result']['saved_model_dir']))
    independent = _load_state(output / 'step_0004/final_model')
    compared = _assert_state_equal(native, independent, 'Native final versus independently restored head step4')
    _require(compared == expected['serialized_tensors'], 'Native/fresh inventory differs.')
    del native, independent
    gc.collect()
    report = {'status': 'verified', 'source_checkpoint_id': record['source_checkpoint_id'],
        'checkpoint_id': audit['checkpoint_id'], 'expected_inventory': expected,
        'native_vs_step4_tensors_compared': compared,
        'training_report_sha256': _sha256(output / 'training.json'), 'audit_sha256': _sha256(output / 'step_0004/audit.json'),
        'export_record_sha256': _sha256(output / 'step_0004/final_model/export_record.json'),
        'changed_head_fp32_elements': audit['changed_head_fp32_elements'],
        'changed_head_bf16_elements': audit['changed_head_bf16_elements'], 'changed_code_bits': 0,
        'bit_budget': audit['bit_budget'], 'export_peak_allocated_bytes': exported['peak_cuda_allocated_bytes'],
        'export_peak_reserved_bytes': exported['peak_cuda_reserved_bytes'], 'diagnostic_not_quality': True}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=('train', 'verify'))
    args = parser.parse_args()
    {'train': train, 'verify': verify}[args.phase](json.loads(args.launch_record.read_text()))


if __name__ == '__main__':
    main()
