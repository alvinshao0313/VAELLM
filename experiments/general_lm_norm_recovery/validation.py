"""Four native updates observing both mutable classes and synchronous saves."""
from pathlib import Path
import torch
from experiments.long_context_recovery.contract import require
from experiments.qwen3_w25.aux_step_export_callback import AuxStepExportCallback
from experiments.w25_eval.validate_wide_subset_training import TrainingObservations
from experiments.general_lm_norm_recovery.state import persistent_inventory, norm_parameters, load_union_state


class UnionObservations(TrainingObservations):
    def __init__(self, architecture):
        super().__init__(tuple(architecture['projection_shapes']))
        self.architecture = architecture
        self.groups = {}
        self.before = {}
        self.diagnostic['gradients'] = []
        self.diagnostic['updates'] = []

    def on_train_begin(self, args, state, control, **kwargs):
        AuxStepExportCallback.on_train_begin(self, args, state, control, **kwargs)
        require(state.global_step == 0 and args.max_steps == 2000 and args.save_steps == 2
                and args.get_warmup_steps(2000) == 20, 'Require fresh native2000 with save2 and stop4.')
        model = kwargs['model']
        inventory, banks = persistent_inventory(model, self.architecture)
        active = {key: p for key, p in model.named_parameters() if p.requires_grad}
        norms = norm_parameters(model)
        decoders = {key for keys in inventory.values() for key in keys}
        require(not decoders & norms.keys() and set(active) == decoders | norms.keys()
                and all(p.dtype == torch.float32 for p in active.values())
                and not hasattr(model, 'sparse_bit_tuning'), 'Only the native FP32 decoder/norm union may train.')
        self.groups = {'decoder': {key: active[key] for key in sorted(decoders)}, 'norm': norms}
        self.before = {key: p.detach().clone() for key, p in active.items()}
        optimizer = kwargs['optimizer']
        while hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer
        require(not hasattr(optimizer, 'bit_optimizer')
                and {id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in active.values()},
                'Optimizer inventory differs from the mutable union.')
        rates = {}
        for kind, parameters in self.groups.items():
            ids = {id(p) for p in parameters.values()}
            rates[kind] = sorted({float(group.get('initial_lr', group['lr'])) for group in optimizer.param_groups
                                 if any(id(p) in ids for p in group['params'])})
        require(rates == {'decoder': [3e-6], 'norm': [1e-5]}, 'Native optimizer component learning rates differ.')
        self.diagnostic['trainable_inventory'] = {'component_counts': {k: len(v) for k, v in self.groups.items()},
            'component_elements': {k: sum(p.numel() for p in v.values()) for k, v in self.groups.items()},
            'keys': sorted(active), 'code_banks': sum(map(len, banks.values())), 'initial_learning_rates': rates,
            'mutable_shapes': {key: list(p.shape) for key, p in active.items()},
            'rope_dtypes': sorted({str(b.dtype) for name, b in model.named_buffers() if name.endswith('inv_freq')})}
        def inputs(_module, positional, values):
            ids = values.get('input_ids', positional[0] if positional else None)
            if ids is not None:
                mask = values.get('attention_mask')
                self.diagnostic['batches'].append({'step': self._observed_step, 'shape': list(ids.shape),
                    'padded_tokens': ids.numel(), 'attention_tokens': None if mask is None else int(mask.sum())})
        self._input_hook = model.register_forward_pre_hook(inputs, with_kwargs=True)
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        item = {'step': int(state.global_step) + 1, 'components': {}}
        for kind, parameters in self.groups.items():
            grads = [p.grad for p in parameters.values() if p.grad is not None]
            require(grads and all(g.dtype == torch.float32 for g in grads), 'Missing actual FP32 gradient: ' + kind)
            finite = bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
            square = float(torch.stack([g.detach().double().square().sum() for g in grads]).sum())
            require(finite and square > 0, 'Nonfinite or zero component gradient: ' + kind)
            item['components'][kind] = {'gradient_tensors': len(grads), 'fp32_gradients': True, 'finite': finite, 'l2': square ** .5}
        self.diagnostic['gradients'].append(item)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        result = super().on_step_end(args, state, control, **kwargs)
        item = {'step': int(state.global_step), 'components': {}}
        for kind, parameters in self.groups.items():
            changed = int(torch.stack([torch.count_nonzero(p.detach() != self.before[key]) for key, p in parameters.items()]).sum())
            item['components'][kind] = {'changed_fp32_elements_this_step': changed}
            # Native cosine warmup starts at zero; step1 may legitimately be unchanged.
            if int(state.global_step) > 1:
                require(changed > 0, 'No real post-warmup component update: ' + kind)
            for key, p in parameters.items():
                self.before[key].copy_(p.detach())
        self.diagnostic['updates'].append(item)
        return result

    def on_save(self, args, state, control, **kwargs):
        result = super().on_save(args, state, control, **kwargs)
        mutable, _ = load_union_state(Path(args.output_dir) / f'checkpoint-{state.global_step}',
            {key: list(p.shape) for key, p in self.groups['decoder'].items()},
            {key: list(p.shape) for key, p in self.groups['norm'].items()})
        self.diagnostic['native_saves'][-1]['mutable_tensors'] = len(mutable)
        return result

    def close(self):
        super().close()
        self.before.clear()
        self.groups.clear()
