"""Native four-step joint prefix with model-derived scope; gradient observer reused unchanged."""
import argparse
from pathlib import Path
import time
import torch
from compressed_e2e_fintuning.runtime_v6 import run as native_run
from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
from e2e_common.full_lora import iter_named_full_compressed_peft_proxies, iter_named_peft_lora_layers, collect_exact_peft_lora_config
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen_decoder_lora_fallback.train import JointObservations as NativeObservations
from experiments.llama_decoder_lora_validation.binding import check_inputs, configuration, read, require
from experiments.llama_decoder_lora_validation.state import inspect_step, partition
from train_utils.checkpoint_v6 import load_v6_training_model_state
from train_utils.runtime_snapshot import to_runtime_jsonable


class JointObservations(NativeObservations):
    def __init__(self, record):
        super().__init__(True, record['architecture'])
        self.geometry, self.frozen_count = record['geometry'], record['expected_inventory']['frozen_tensors']

    def on_train_begin(self, args, state, control, **kwargs):
        require(state.global_step == 0 and args.max_steps == 2000 and args.save_steps == (2 if self.short else 400)
                and args.get_warmup_steps(2000) == 20, 'Fresh native schedule differs.')
        g = self.geometry
        model = kwargs['model']
        proxies = dict(iter_named_full_compressed_peft_proxies(model))
        require(set(proxies) == set(self.architecture['projection_shapes']), 'All actual native LoRA proxies required.')
        expected = {'decoder': [p for proxy in proxies.values() for p in proxy.base_layer._parallel_stage_decoder.parameters()],
                    'A': [m.lora_A['default'].weight for _, m in iter_named_peft_lora_layers(model)],
                    'B': [m.lora_B['default'].weight for _, m in iter_named_peft_lora_layers(model)]}
        active = {name: p for name, p in model.named_parameters() if p.requires_grad}
        identities = [id(p) for group in expected.values() for p in group]
        require(len(set(identities)) == len(identities) == len(active) == g['decoder_tensors'] + g['lora_tensors']
                and set(identities) == {id(p) for p in active.values()}
                and all(p.dtype == torch.float32 for p in active.values()), 'Trainables escaped joint FP32 scope.')
        require(len(expected['decoder']) == g['decoder_tensors'] and len(expected['A']) == len(expected['B']) == g['projections']
                and sum(p.numel() for p in expected['decoder']) == g['decoder_elements']
                and sum(p.numel() for p in expected['A'] + expected['B']) == g['lora_elements'],
                'Actual joint trainable geometry differs.')
        topology = collect_exact_peft_lora_config(model, default_rank=2, alpha=4, dropout=0)
        require(topology == {'rank': 2, 'alpha': 4.0, 'dropout': 0.0, 'rank_pattern': {},
                            'target_modules': sorted(proxies)}, 'Actual rank topology differs.')
        optimizer = kwargs['optimizer']
        while hasattr(optimizer, 'optimizer'): optimizer = optimizer.optimizer
        require({id(p) for g in optimizer.param_groups for p in g['params']} == set(identities)
                and not hasattr(optimizer, 'bit_optimizer'), 'Optimizer scope differs.')
        self.report['trainable_inventory'] = {**g, 'frozen_source_tensors': self.frozen_count}
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

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        meta, saved, _, _, _, architecture = inspect_step(checkpoint)
        mutable, manifest = load_v6_training_model_state(str(checkpoint), map_location='cpu')
        partition(mutable, manifest, architecture['projection_shapes'], self.geometry['decoder_tensors'])
        require(saved['global_step'] == int(state.global_step), 'Native saved step differs.')
        self.report['native_saves'].append({'step': int(state.global_step), 'checkpoint_id': meta['checkpoint_id'],
            'meta_sha256': sha256(checkpoint / 'checkpoint_meta.json'), 'mutable_tensors': len(mutable)})
        return control


def train(path):
    record = read(path); output = Path(record['output_dir'])
    require(record['kind'] == 'validate_s4' and not (output / 'training.json').exists()
            and torch.cuda.device_count() == 1, 'Fresh one-GPU four-step validation required.')
    check_inputs(record); cfg, hf, args = configuration(record)
    observer = JointObservations(record)
    configure_e2e_determinism(bool(args.full_determinism)); set_e2e_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(); started = time.time()
    report = {'status': 'failed', 'source_checkpoint_id': record['source_checkpoint_id'], 'diagnostic_not_quality': True}
    try:
        result = to_runtime_jsonable(native_run(cfg, hf, args, extra_callbacks=(observer,)))
        require(result['global_step'] == 4 and result['round_base_checkpoint_id'] == record['source_checkpoint_id'],
                'Actual native step/source differs.')
        require([v['step'] for v in observer.report['native_saves']] == [2, 4]
                and all([v['step'] for v in observer.report[key]] == [1, 2, 3, 4] for key in ('gradients', 'updates'))
                and {v['step'] for v in observer.report['batches']} == {1, 2, 3, 4}
                and all(v['shape'][0] == 4 and v['shape'][1] <= 1024 for v in observer.report['batches']),
                'Actual saves/full-shape updates are incomplete.')
        check_inputs(record); report.update(status='trained', training_result=result)
    finally:
        report.update(observations=observer.report, elapsed_seconds=time.time() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(output / 'training.json', report)
        if torch.distributed.is_available() and torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--launch_record', required=True, type=Path)
    train(parser.parse_args().launch_record)
