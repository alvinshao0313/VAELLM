"""Native FineWeb all-decoder/full dense-head recovery and four-update observation."""
from __future__ import annotations
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from compressed_e2e_fintuning.args import parse_args
from compressed_e2e_fintuning.runtime_v6 import run as native_run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from experiments.general_lm_recovery.train import ROOT, SAVE_STEPS, training_arguments as control_arguments
from experiments.general_lm_full_kl_probe.train import FullKLObservations
from experiments.general_lm_head_recovery.callback import UnionStepCallback
from experiments.general_lm_head_recovery.state import HEAD_KEY
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen3_w25.aux_step_export_callback import AuxStepExportCallback
from experiments.w25_eval.model_inventory import model_inventory
from experiments.w25_eval.norm_state import norm_parameters
from experiments.w25_eval.subset_state import persistent_inventory, selected_names, validate_recipe
from train_utils.runtime_snapshot import to_runtime_jsonable


def training_arguments(record):
    argv = control_arguments(record['source_checkpoint'], record['architecture']['base_model_path'],
                             record['output_dir'], record['data']['train_file'])
    argv[argv.index('--distill_fp32_components') + 1] = 'decoder,lm_head'
    argv[argv.index('--lm_head_train_mode') + 1] = 'full'
    return argv + ['--lm_head_lr', '3e-6']


def configuration(record):
    require(record['kind'] in ('validate_s4', 'formal_s2000'), 'Unknown head-recovery run kind.')
    require(record['training_argv'] == training_arguments(record), 'Unexpected dense-head training arguments.')
    cfg, hf, args = parse_args(record['training_argv'])
    expected = deepcopy(read(record['control_normalized'])['canonical_config'])
    expected['run_root_dir'] = record['output_dir']
    expected['opt']['distill_fp32_components'] = ['decoder', 'lm_head']
    expected['aux'].update(lm_head_train_mode='full', lm_head_lr=3e-6)
    expected['explicit_cli_flags'] = sorted([*expected['explicit_cli_flags'], '--lm_head_lr'])
    require(to_runtime_jsonable(cfg) == expected, 'Only output and declared full-head settings may differ from the native FineWeb control.')
    meta = read(Path(cfg.student_checkpoint_dir) / 'checkpoint_meta.json')
    architecture = model_inventory(meta['base_model_path'])
    validate_recipe(meta, architecture)
    names = selected_names(cfg.target_layers, cfg.target_modules, architecture)
    require(architecture == record['architecture'] and architecture['model_type'] == 'llama'
            and set(names) == set(architecture['projection_shapes'])
            and meta['checkpoint_id'] == record['source_checkpoint_id']
            and cfg.resume_from_checkpoint is None and cfg.stop_after_step is None,
            'Require the fresh complete e767 Llama source, including its actual D16 down specification.')
    require(args.max_steps == 2000 and args.save_steps == 400 and args.get_warmup_steps(2000) == 20
            and args.bf16 and not args.fp16 and cfg.aux.lm_head_lr == cfg.opt.decoder_lr == 3e-6
            and cfg.aux.norm_train_mode == cfg.aux.residual_lora_mode == 'none'
            and cfg.loss.loss_type == 'kl_top_mass' and cfg.loss.top_k == 100 and cfg.loss.temperature == 1,
            'Keep the native2000/warm20/BF16/top100-mass recipe with frozen norms and no LoRA.')
    return cfg, hf, args, names, architecture


class HeadObservations(FullKLObservations):
    """Observe the new head path without retaining another full FP32 head."""
    def on_train_begin(self, args, state, control, **kwargs):
        AuxStepExportCallback.on_train_begin(self, args, state, control, **kwargs)
        require(state.global_step == 0 and args.max_steps == 2000 and args.save_steps == 400
                and args.get_warmup_steps(2000) == 20, 'Require the fresh unchanged native schedule.')
        model = kwargs['model']
        inventory, banks = persistent_inventory(model, self.architecture)
        decoder_keys = {key for name in self.selected_targets for key in inventory[name]}
        active = {key: p for key, p in model.named_parameters() if p.requires_grad}
        require(set(active) == decoder_keys | {HEAD_KEY}
                and all(p.dtype == torch.float32 for p in active.values())
                and type(model.lm_head) is torch.nn.Linear and not model.config.tie_word_embeddings
                and model.lm_head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr()
                and not hasattr(model, 'sparse_bit_tuning'), 'Only FP32 decoders and the original untied dense head may train.')
        norms = norm_parameters(model)
        require(all(not p.requires_grad for p in norms.values()), 'Backbone norms must stay frozen.')
        optimizer = kwargs['optimizer']
        while hasattr(optimizer, 'optimizer'): optimizer = optimizer.optimizer
        require({id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in active.values()}
                and not hasattr(optimizer, 'bit_optimizer'), 'Optimizer union differs from actual trainables.')
        self.active = {key: active[key] for key in active if key != HEAD_KEY}
        self.head = active[HEAD_KEY]
        self.previous = torch.cat([p.detach().reshape(-1) for p in self.active.values()]).clone()
        self.head_indices = torch.arange(1024, device=self.head.device) * (self.head.numel() - 1) // 1023
        self.previous_head = self.head.detach().view(-1)[self.head_indices].clone()
        rope = {name: str(value.dtype) for name, value in model.named_buffers() if name.endswith('inv_freq')}
        require(rope and set(rope.values()) == {'torch.float32'}, 'Native RoPE frequencies must remain FP32.')
        self.diagnostic['head_updates'] = []
        self.diagnostic['trainable_inventory'] = {'decoder_tensors': len(self.active),
            'decoder_elements': self.previous.numel(), 'decoder_keys': sorted(self.active),
            'head_tensors': 1, 'head_shape': list(self.head.shape), 'head_elements': self.head.numel(),
            'frozen_norm_tensors': len(norms), 'selected_projection_code_banks': sum(map(len, banks.values())),
            'score_elements': 0, 'student_rope_dtypes': rope,
            'observation_previous_parameter_bytes': self.previous.numel() * 4 + self.previous_head.numel() * 4,
            'head_update_observation': 'Fixed 1024 scalar sample; full selected-head cast/update checked after training.'}
        def inputs(_module, positional, values):
            ids = values.get('input_ids', positional[0] if positional else None)
            if ids is not None:
                mask = values.get('attention_mask')
                self.diagnostic['batches'].append({'step': self._observed_step, 'shape': list(ids.shape),
                    'padded_tokens': ids.numel(), 'attention_tokens': int(mask.sum()) if mask is not None else None,
                    'causal_tokens': int(mask[:, 1:].sum()) if mask is not None else None})
        self._input_hook = model.register_forward_pre_hook(inputs, with_kwargs=True)
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        super().on_pre_optimizer_step(args, state, control, **kwargs)
        grad = self.head.grad
        require(grad is not None and grad.dtype == torch.float32, 'Missing actual FP32 head gradient.')
        norm = float(grad.detach().norm())
        require(math.isfinite(norm) and norm > 0, 'Head gradient must be finite and nonzero.')
        self.diagnostic['gradients'][-1].update(head_gradient_norm=norm, head_gradient_tensors=1)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        current = self.head.detach().view(-1)[self.head_indices]
        delta = float((current - self.previous_head).norm())
        require(math.isfinite(delta), 'Nonfinite sampled head update.')
        self.previous_head.copy_(current)
        optimizer = kwargs['optimizer']
        while hasattr(optimizer, 'optimizer'): optimizer = optimizer.optimizer
        moments = optimizer.state[self.head]
        require(all(key in moments and moments[key].dtype == torch.float32 and moments[key].shape == self.head.shape
                    for key in ('exp_avg', 'exp_avg_sq')), 'Actual head Adam slots must have full FP32 shape.')
        self.diagnostic['head_updates'].append({'step': int(state.global_step), 'sampled_fp32_delta_l2': delta,
            'adam_moment_dtypes': {key: str(moments[key].dtype) for key in ('exp_avg', 'exp_avg_sq')}})
        return control

    def close(self):
        super().close()
        self.head = self.head_indices = self.previous_head = None


def train_short(record):
    require(record['kind'] == 'validate_s4' and record['status'] == 'training'
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu']), 'Short training is launched only by its bound same-GPU dispatcher.')
    check_inputs(record)
    cfg, hf, args, names, architecture = configuration(record)
    callback = HeadObservations(names, architecture)
    configure_e2e_determinism(bool(args.full_determinism)); set_e2e_seed(int(args.seed))
    started = time.time(); torch.cuda.reset_peak_memory_stats()
    report = {'status': 'failed', 'diagnostic_not_quality': True}
    try:
        result = to_runtime_jsonable(native_run(cfg, hf, args, extra_callbacks=(callback,)))
        observed, expected = callback.diagnostic, record['expected_inventory']
        require(result['global_step'] == 4 and result['round_base_checkpoint_id'] == record['source_checkpoint_id'], 'Native training must finish at actual step4.')
        for key in ('optimizer_steps', 'gradients', 'parameter_updates', 'head_updates', 'losses'):
            require([x['step'] for x in observed[key]] == [1, 2, 3, 4], 'Incomplete native observations: ' + key)
        require([x['step'] for x in observed['native_saves']] == [4]
                and {x['step'] for x in observed['batches']} == {1, 2, 3, 4}
                and all(x['shape'][0] == 4 and x['shape'][1] <= 1024 for x in observed['batches'])
                and all(x['main_moment_dtypes'] == ['torch.float32'] for x in observed['optimizer_steps'])
                and sum(x['fp32_delta_l2'] for x in observed['parameter_updates']) > 0, 'Native updates, data or save boundary differs.')
        actual = observed['trainable_inventory']
        require(actual['decoder_tensors'] == expected['selected_decoder_tensors']
                and actual['decoder_elements'] == expected['selected_decoder_elements']
                and actual['head_shape'] == expected['head_shape'] and actual['head_elements'] == expected['head_elements']
                and actual['frozen_norm_tensors'] == expected['norm_tensors'], 'Actual inventory differs from the source-derived inventory.')
        check_inputs(record)
        report.update(status='trained', training_result=result, observations=observed,
            resource_coverage='Four native dynamic B4<=1024 batches; not a full-data worst-case bound.', full8_and_longbench='not_evaluated')
    finally:
        report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      elapsed_seconds=time.time() - started)
        write_json(Path(record['output_dir']) / 'training.json', report)
        callback.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


def run_formal(path):
    path = Path(path).resolve(); record = read(path); output = path.parent; code = 1
    try:
        require(record['kind'] == 'formal_s2000' and record['status'] == 'bound_ready'
                and type(record['gpu']) is int and type(record['export_gpu']) is int and record['gpu'] != record['export_gpu']
                and not record['queued'] and not record['automatic_retry']
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                and Path.cwd().resolve() == Path(record['cwd']).resolve(), 'Bind the fresh formal training and dedicated export GPUs explicitly.')
        require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                'The actual native environment differs.')
        require(not list(output.glob('home_*')) and not list(output.glob('step_*')) and not (output / 'training.json').exists(), 'No resume or overwrite.')
        check_inputs(record)
        proof = read(record['validation_proof'])
        require(proof['status'] == 'verified' and proof['source_checkpoint_id'] == record['source_checkpoint_id']
                and proof['expected_inventory'] == record['expected_inventory']
                and proof['source_sha256'] == record['source_sha256'], 'Require the successful same-source head short validation.')
        cfg, hf, args, _names, _architecture = configuration(record)
        callback = UnionStepCallback(record=record, source_dir=cfg.student_checkpoint_dir,
            run_root_dir=cfg.run_root_dir, export_cuda_visible_devices=str(record['export_gpu']))
        record.update(status='training', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]), started_unix=time.time())
        write_json(path, record)
        configure_e2e_determinism(bool(args.full_determinism)); set_e2e_seed(int(args.seed)); torch.cuda.reset_peak_memory_stats()
        result = to_runtime_jsonable(native_run(cfg, hf, args, extra_callbacks=(callback,)))
        require(result['global_step'] == 2000 and callback.completed_steps == SAVE_STEPS
                and result['round_base_checkpoint_id'] == record['source_checkpoint_id'], 'Incomplete native trajectory or full8 callback.')
        check_inputs(record)
        report = {'status': 'completed', 'training_result': result, 'callback_completed_steps': callback.completed_steps,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'source_checkpoint_id': record['source_checkpoint_id'], 'expected_inventory': record['expected_inventory'],
            'teacher_output_offload': 'none', 'longbench_status': 'not_evaluated', 'current_nine_item_goal_status': 'not_evaluated'}
        write_json(output / 'training.json', report)
        record.update(status='completed', training_result=result, completed_steps=callback.completed_steps,
                      training_report_sha256=sha256(output / 'training.json')); code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-record', type=Path, required=True)
    parser.add_argument('--short-training-child', action='store_true')
    cli = parser.parse_args()
    if cli.short_training_child: train_short(read(cli.launch_record))
    else: sys.exit(run_formal(cli.launch_record))
