"""Observe two real updates on the original 500-step recovery schedule."""
from transformers import TrainerCallback
import torch
import time

from bindings import s


class RecoveryCheck(TrainerCallback):
    def __init__(self, evidence, category):
        self.evidence, self.category = evidence, category
        self.active, self.initial = {}, {}
        self.handle = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        index = s.CATEGORIES.index(self.category)
        s.require(args.max_steps == 500 and args.per_device_train_batch_size == 4
            and args.gradient_accumulation_steps == 1 and args.max_grad_norm == 1.5,
            'Validation keeps the original recovery batch, length and 500-step schedule.')
        self.active = {name: value for name, value in model.named_parameters() if value.requires_grad}
        lora = {name: value for name, value in self.active.items() if 'lora_' in name}
        decoder = {name: value for name, value in self.active.items() if 'lora_' not in name}
        s.require(len(lora) == (6 - index) * 36 * 2 and len(decoder) == (index + 1) * 36 * 6
            and all('decoder' in name for name in decoder)
            and all(value.dtype == torch.float32 for value in self.active.values()),
            'Observe every current prefix decoder and every remaining dense rank12 LoRA.')
        self.initial = {name: value.detach().cpu().clone() for name, value in self.active.items()}
        self.evidence.update(category=self.category, planned_steps=500, actual_steps=0,
            decoder_leaves=len(decoder), lora_leaves=len(lora), updates=[], input_shapes=[])
        self.step_started = time.time()

        def inputs(_module, _args, values):
            ids, mask = values.get('input_ids'), values.get('attention_mask')
            if ids is None:
                return
            s.require(ids.ndim == 2 and ids.shape[0] == 4 and 0 < ids.shape[1] <= 1024
                and mask is not None and mask.shape == ids.shape and bool((mask.sum(-1) > 0).all()),
                'Preserve real dynamic-padded B4/max1024 student inputs.')
            self.evidence['input_shapes'].append(list(ids.shape))
        self.handle = model.register_forward_pre_hook(inputs, with_kwargs=True)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        s.require(all(value.grad is not None and bool(torch.isfinite(value.grad).all())
            for value in self.active.values()), 'Missing or nonfinite real decoder/LoRA gradient.')

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        rows = [optimizer.state[value] for value in self.active.values()]
        s.require(all(int(row['step']) == state.global_step and row['exp_avg'].dtype == torch.float32
            and row['exp_avg_sq'].dtype == torch.float32 and bool(torch.isfinite(row['exp_avg']).all())
            and bool(torch.isfinite(row['exp_avg_sq']).all()) for row in rows), 'Invalid real FP32 Adam state.')
        changed = sum(int((value.detach().cpu() != self.initial[name]).sum())
            for name, value in self.active.items() if 'decoder' in name)
        self.evidence['updates'].append(dict(step=state.global_step, changed_decoder_elements=changed,
            elapsed_seconds=time.time() - self.step_started, lr=[group['lr'] for group in optimizer.param_groups]))
        self.step_started = time.time()
        if state.global_step == 2:
            s.require(changed > 0, 'The second real update must change a decoder master.')
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        s.require(state.global_step == 2, 'Validate exactly two updates without changing the schedule.')
        self.evidence.update(actual_steps=2, peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved())
        self.handle.remove()
        self.active.clear()
        self.initial.clear()
