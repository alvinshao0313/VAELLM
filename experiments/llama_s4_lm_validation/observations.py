"""Observe native grouped S4 gradients/updates while reusing save/stop observations."""
from pathlib import Path

import torch

from experiments.long_context_recovery.contract import require
from experiments.qwen3_w25.aux_step_export_callback import AuxStepExportCallback
from experiments.w25_eval.validate_wide_subset_training import TrainingObservations
from experiments.llama_s4_lm_validation.state import inspect_step, persistent_inventory


class GateStageObservations(TrainingObservations):
    def __init__(self, names, architecture):
        super().__init__(names)
        self.architecture, self.groups, self.before = architecture, {}, {}
        self.diagnostic.update(gate_gradients=[], gate_updates=[], gate_layout={})

    def on_train_begin(self, args, state, control, **kwargs):
        AuxStepExportCallback.on_train_begin(self, args, state, control, **kwargs)
        require(state.global_step == 0 and args.max_steps == 2000 and args.save_steps == 2
                and args.get_warmup_steps(2000) == 20, 'Preserve the original2000/warm20 prefix.')
        model = kwargs['model']; inventory, banks = persistent_inventory(model, self.architecture)
        expected = {key for name in self.selected_targets for key in inventory[name]}
        active = {name: p for name, p in model.named_parameters() if p.requires_grad}
        require(set(active) == expected and len(active) == 1344
                and all(p.dtype == torch.float32 for p in active.values())
                and not hasattr(model, 'sparse_bit_tuning'), 'Only1344 grouped decoder leaves may train.')
        optimizer = kwargs['optimizer']
        while hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer
        parameters = [p for group in optimizer.param_groups for p in group['params']]
        require(not hasattr(optimizer, 'bit_optimizer') and len(parameters) == len({id(p) for p in parameters})
                and {id(p) for p in parameters} == {id(p) for p in active.values()}, 'Optimizer inventory differs.')
        self.diagnostic['trainable_inventory'] = {'decoder_keys': sorted(active), 'decoder_tensors': len(active),
            'decoder_elements': sum(p.numel() for p in active.values()), 'score_keys': [], 'score_elements': 0,
            'selected_projection_code_banks': sum(len(banks[n]) for n in self.selected_targets)}

        def observe_inputs(_module, positional, values):
            inputs = values.get('input_ids')
            if inputs is None and positional:
                inputs = positional[0]
            if inputs is not None:
                mask = values.get('attention_mask')
                self.diagnostic['batches'].append({'step': self._observed_step, 'shape': list(inputs.shape),
                    'padded_tokens': inputs.numel(), 'attention_tokens': None if mask is None else int(mask.sum())})
        self._input_hook = model.register_forward_pre_hook(observe_inputs, with_kwargs=True)
        modules = dict(model.named_modules())
        for name in self.selected_targets:
            if not name.endswith('.gate_proj'):
                continue
            module = modules[name]; decoder = module._parallel_stage_decoder
            require(list(module._parallel_stage_layout) == [(s, 0) for s in range(4)]
                    and decoder.num_models == 4 and decoder.in_dim == 16 and decoder.out_dim == 32
                    and module.trainable_decode and not module.cache_decoded_weight
                    and module.weight_rotation.block_size == 32 and not module.transpose
                    and module.packed_vq_decoder_linear
                    and tuple(module._parallel_stage_grouped_vq_logical_shape) == (1806336, 4, 16),
                    'Wrong original differentiable packed S4 gate: ' + name)
            parameters = dict(decoder.named_parameters())
            require(len(parameters) == 6 and all(p.requires_grad and p.dtype == torch.float32
                    and p.numel() % 4 == 0 for p in parameters.values()), 'Wrong grouped decoder parameters.')
            self.groups[name] = parameters
            self.before[name] = {key: p.detach().clone() for key, p in parameters.items()}
            self.diagnostic['gate_layout'][name] = {'stage_part_layout': list(module._parallel_stage_layout),
                'parameter_shapes': {k: list(p.shape) for k, p in parameters.items()},
                'logical_shape': list(module._parallel_stage_grouped_vq_logical_shape),
                'trainable_decode': True, 'decoded_weight_cache': False}
        require(len(self.groups) == 32, 'Observe all32 S4 gates.')
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        item = {'step': int(state.global_step) + 1, 'stages': {}}
        for name, parameters in self.groups.items():
            require(all(p.grad is not None and p.grad.dtype == torch.float32 for p in parameters.values()),
                    'Missing FP32 decoder gradient: ' + name)
            for stage in range(4):
                gradients = [p.grad.detach().reshape(4, -1)[stage] for p in parameters.values()]
                finite = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
                square = float(torch.stack([g.double().square().sum() for g in gradients]).sum())
                require(finite and square > 0, f'Nonfinite or zero gradient: {name}/stage{stage}')
                item['stages'][f'{name}/stage{stage}'] = {'gradient_tensors': 6, 'finite': finite, 'l2': square ** .5}
        self.diagnostic['gate_gradients'].append(item)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        result = super().on_step_end(args, state, control, **kwargs)
        item = {'step': int(state.global_step), 'stages': {}}
        for name, parameters in self.groups.items():
            for stage in range(4):
                changed = int(torch.stack([torch.count_nonzero(p.detach().reshape(4, -1)[stage]
                    != self.before[name][key].reshape(4, -1)[stage]) for key, p in parameters.items()]).sum())
                require(int(state.global_step) == 1 or changed > 0, f'No stage update: {name}/stage{stage}')
                item['stages'][f'{name}/stage{stage}'] = {'changed_fp32_elements_this_step': changed}
            for key, p in parameters.items():
                self.before[name][key].copy_(p.detach())
        self.diagnostic['gate_updates'].append(item)
        return result

    def on_save(self, args, state, control, **kwargs):
        result = super().on_save(args, state, control, **kwargs)
        inspect_step(Path(args.output_dir) / f'checkpoint-{state.global_step}')
        return result

    def close(self):
        super().close()
        self.before.clear(); self.groups.clear()
