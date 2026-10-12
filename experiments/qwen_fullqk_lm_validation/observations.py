"""Observe actual Q/K stage gradients and updates without changing training."""
import torch

from experiments.long_context_recovery.contract import require
from experiments.w25_eval.validate_decoder_subset_training import DecoderObservations


class QKStageObservations(DecoderObservations):
    def __init__(self, names, architecture):
        super().__init__(names, architecture)
        self.groups, self.before = {}, {}
        self.diagnostic.update(qk_gradients=[], qk_updates=[], qk_layout={})

    def on_train_begin(self, args, state, control, **kwargs):
        result = super().on_train_begin(args, state, control, **kwargs)
        modules = dict(kwargs['model'].named_modules())
        for name in self.selected_targets:
            if not name.endswith(('.q_proj', '.k_proj')):
                continue
            module = modules[name]
            decoder = module._parallel_stage_decoder
            layout = list(module._parallel_stage_layout)
            require(layout == [(0, 0), (1, 0)] and decoder.num_models == 2
                    and module.trainable_decode and not module.cache_decoded_weight
                    and module.weight_rotation.block_size == 0,
                    'Expected two-stage differentiable full-rotation decode: ' + name)
            parameters = dict(decoder.named_parameters())
            require(len(parameters) == 6 and all(p.requires_grad and p.dtype == torch.float32
                    and p.numel() % 2 == 0 for p in parameters.values()), 'Wrong packed stage parameter layout.')
            self.groups[name] = parameters
            self.before[name] = {key: p.detach().clone() for key, p in parameters.items()}
            self.diagnostic['qk_layout'][name] = {'stage_part_layout': layout,
                'parameter_shapes': {key: list(p.shape) for key, p in parameters.items()},
                'rotation_block_size': 0, 'trainable_decode': True, 'decoded_weight_cache': False}
        require(len(self.groups) == 72, 'Expected all36 Q plus all36 K projections.')
        return result

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        item = {'step': int(state.global_step) + 1, 'stages': {}}
        for name, parameters in self.groups.items():
            require(all(p.grad is not None and p.grad.dtype == torch.float32 for p in parameters.values()),
                    'Missing native FP32 Q/K gradient: ' + name)
            for stage in range(2):
                gradients = [p.grad.detach().reshape(2, -1)[stage] for p in parameters.values()]
                finite = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
                square = float(torch.stack([g.double().square().sum() for g in gradients]).sum())
                require(finite and square > 0, f'Nonfinite or zero native gradient: {name}/stage{stage}')
                item['stages'][f'{name}/stage{stage}'] = {'gradient_tensors': 6, 'finite': finite, 'l2': square ** .5}
        self.diagnostic['qk_gradients'].append(item)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        result = super().on_step_end(args, state, control, **kwargs)
        item = {'step': int(state.global_step), 'stages': {}}
        for name, parameters in self.groups.items():
            for stage in range(2):
                changed = int(torch.stack([torch.count_nonzero(p.detach().reshape(2, -1)[stage]
                    != self.before[name][key].reshape(2, -1)[stage]) for key, p in parameters.items()]).sum())
                # Native warmup's first update uses zero LR; subsequent updates must be real.
                require(int(state.global_step) == 1 or changed > 0, f'No stage update: {name}/stage{stage}')
                item['stages'][f'{name}/stage{stage}'] = {'changed_fp32_elements_this_step': changed}
            for key, p in parameters.items():
                self.before[name][key].copy_(p.detach())
        self.diagnostic['qk_updates'].append(item)
        return result

    def close(self):
        super().close()
        self.before.clear()
        self.groups.clear()
