"""Native joint updates; only the four-step diagnostic adds component observations."""
import argparse
from pathlib import Path
import math
import time

import torch
from transformers import TrainerCallback

from compressed_e2e_fintuning.runtime_v6 import run as native_run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from e2e_common.full_lora import iter_named_full_compressed_peft_proxies, iter_named_peft_lora_layers, collect_exact_peft_lora_config
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen_decoder_lora_fallback.callback import JointStepCallback
from experiments.qwen_decoder_lora_fallback.config import check_inputs, configuration, read, require
from experiments.qwen_decoder_lora_fallback.state import inspect_step, partition
from train_utils.checkpoint_v6 import load_v6_training_model_state
from train_utils.runtime_snapshot import to_runtime_jsonable


class JointObservations(TrainerCallback):
    def __init__(self, diagnostic, architecture):
        self.short, self.architecture = diagnostic, architecture
        self.report = {'batches': [], 'gradients': [], 'updates': [], 'native_saves': []}
        self.hook, self.previous, self.groups = None, {}, {}

    def on_train_begin(self, args, state, control, **kwargs):
        require(state.global_step == 0 and args.max_steps == 2000 and args.save_steps == (2 if self.short else 400)
                and args.get_warmup_steps(2000) == 20, 'Fresh native schedule differs.')
        model = kwargs['model']
        proxies = dict(iter_named_full_compressed_peft_proxies(model))
        require(set(proxies) == set(self.architecture['projection_shapes']), 'All252 native LoRA proxies required.')
        expected = {'decoder': [p for proxy in proxies.values() for p in proxy.base_layer._parallel_stage_decoder.parameters()],
                    'A': [m.lora_A['default'].weight for _, m in iter_named_peft_lora_layers(model)],
                    'B': [m.lora_B['default'].weight for _, m in iter_named_peft_lora_layers(model)]}
        active = {name: p for name, p in model.named_parameters() if p.requires_grad}
        identities = [id(p) for group in expected.values() for p in group]
        require(len(set(identities)) == len(identities) == len(active) == 2016
                and set(identities) == {id(p) for p in active.values()}
                and all(p.dtype == torch.float32 for p in active.values()), 'Trainables escaped joint FP32 scope.')
        require(len(expected['decoder']) == 1512 and len(expected['A']) == len(expected['B']) == 252
                and sum(p.numel() for p in expected['decoder']) == 4633344
                and sum(p.numel() for p in expected['A'] + expected['B']) == 5455872,
                'Actual joint trainable geometry differs.')
        topology = collect_exact_peft_lora_config(model, default_rank=2, alpha=4, dropout=0)
        require(topology == {'rank': 2, 'alpha': 4.0, 'dropout': 0.0, 'rank_pattern': {},
                            'target_modules': sorted(proxies)}, 'Actual rank topology differs.')
        optimizer = kwargs['optimizer']
        while hasattr(optimizer, 'optimizer'): optimizer = optimizer.optimizer
        require({id(p) for g in optimizer.param_groups for p in g['params']} == set(identities)
                and not hasattr(optimizer, 'bit_optimizer'), 'Optimizer scope differs.')
        self.report['trainable_inventory'] = {'decoder_tensors': 1512, 'decoder_elements': 4633344,
            'lora_tensors': 504, 'lora_elements': 5455872, 'code_banks': 504, 'frozen_source_tensors': 1911}
        if self.short:
            self.groups = expected
            self.previous = {k: torch.cat([p.detach().reshape(-1) for p in v]).clone() for k, v in expected.items()}
            def inputs(_module, positional, values):
                ids = values.get('input_ids', positional[0] if positional else None)
                if ids is not None:
                    mask = values.get('attention_mask')
                    self.report['batches'].append({'step': self.step, 'shape': list(ids.shape),
                        'attention_tokens': int(mask.sum()) if mask is not None else None})
            self.hook = model.register_forward_pre_hook(inputs, with_kwargs=True)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self.step = int(state.global_step) + 1
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if not self.short: return control
        values = {}
        for kind, parameters in self.groups.items():
            require(all(p.grad is not None and p.grad.dtype == torch.float32 for p in parameters), 'Missing component gradient: ' + kind)
            norm = float(torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in parameters])))
            require(math.isfinite(norm) and (norm > 0 or kind == 'A' and self.step <= 2),
                    'Nonfinite/zero component gradient beyond native zero-B initialization: ' + kind)
            values[kind] = norm
        self.report['gradients'].append({'step': self.step, 'post_clip_norm': values})
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not self.short: return control
        values = {}
        for kind, parameters in self.groups.items():
            now = torch.cat([p.detach().reshape(-1) for p in parameters])
            delta = float((now - self.previous[kind]).norm()); self.previous[kind].copy_(now)
            require(math.isfinite(delta) and (delta > 0 or self.step == 1 or kind == 'A' and self.step == 2),
                    'Missing component update beyond warmup/zero-B initialization: ' + kind)
            values[kind] = delta
        optimizer = kwargs['optimizer']
        while hasattr(optimizer, 'optimizer'): optimizer = optimizer.optimizer
        moments = sorted({str(v.dtype) for item in optimizer.state.values()
                          for k, v in item.items() if k in ('exp_avg', 'exp_avg_sq')})
        require(moments == ['torch.float32'], 'Joint Adam moments must remain FP32.')
        self.report['updates'].append({'step': self.step, 'delta_l2': values, 'moment_dtypes': moments})
        if int(state.global_step) == 4: control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        meta, saved, _, _, _, architecture = inspect_step(checkpoint)
        mutable, manifest = load_v6_training_model_state(str(checkpoint), map_location='cpu')
        partition(mutable, manifest, architecture['projection_shapes'])
        require(saved['global_step'] == int(state.global_step), 'Native saved step differs.')
        self.report['native_saves'].append({'step': int(state.global_step), 'checkpoint_id': meta['checkpoint_id'],
            'meta_sha256': sha256(checkpoint / 'checkpoint_meta.json'), 'mutable_tensors': len(mutable)})
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.hook is not None: self.hook.remove()
        self.previous.clear(); self.groups.clear()
        return control


def train(path):
    record = read(path); output = Path(record['output_dir'])
    require(not (output / 'training.json').exists() and torch.cuda.device_count() == 1, 'Fresh one-GPU training required.')
    check_inputs(record)
    cfg, hf, args, names, architecture = configuration(record)
    short = record['kind'] == 'validate_s4'
    observer = JointObservations(short, architecture); callbacks = [observer]
    if not short:
        proof = record['readiness']['short_validation']
        require(proof['sha256'] and sha256(proof['path']) == proof['sha256'] and read(proof['path'])['status'] == 'verified',
                'A genuine completed joint short proof is required before formal training.')
        callbacks.append(JointStepCallback(record=record, source_dir=cfg.student_checkpoint_dir,
            run_root_dir=cfg.run_root_dir, export_cuda_visible_devices=str(record['export_gpu'])))
    configure_e2e_determinism(bool(args.full_determinism)); set_e2e_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(); started = time.time()
    report = {'status': 'failed', 'source_checkpoint_id': record['source_checkpoint_id'], 'diagnostic_not_quality': short}
    try:
        result = to_runtime_jsonable(native_run(cfg, hf, args, extra_callbacks=tuple(callbacks)))
        require(result['global_step'] == (4 if short else 2000)
                and result['round_base_checkpoint_id'] == record['source_checkpoint_id'], 'Final native source/step differs.')
        require([v['step'] for v in observer.report['native_saves']] == ([2, 4] if short else [400, 800, 1200, 1600, 2000]),
                'Native save boundaries differ.')
        if short:
            require(all([v['step'] for v in observer.report[key]] == [1, 2, 3, 4] for key in ('gradients', 'updates'))
                    and {v['step'] for v in observer.report['batches']} == {1, 2, 3, 4}
                    and all(v['shape'][0] == 4 and v['shape'][1] <= 1024 for v in observer.report['batches']),
                    'Four real B4 update observations are incomplete.')
        else: require(callbacks[-1].completed_steps == [400, 800, 1200, 1600, 2000], 'Missing complete eight-task callback.')
        check_inputs(record)
        report.update(status='trained' if short else 'completed', training_result=result,
                      callback_completed_steps=[] if short else callbacks[-1].completed_steps)
    finally:
        report.update(observations=observer.report, elapsed_seconds=time.time() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(output / 'training.json', report)
        if torch.distributed.is_available() and torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--launch_record', required=True, type=Path)
    train(parser.parse_args().launch_record)
