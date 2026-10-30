"""Observe the true B4/1024 D64 recovery path, stopping at two of500 updates."""
import common as c
from transformers import TrainerCallback
from litebsq.vae_linear import VAELinear


class RecoveryCheck(TrainerCallback):
    def __init__(self, evidence):
        self.evidence = evidence
        self.active = {}
        self.initial = {}
        self.handle = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        torch = c.fork.torch
        c.require(args.max_steps == 500 and args.per_device_train_batch_size == 4
            and args.gradient_accumulation_steps == 1 and args.max_grad_norm == 1.5,
            'Short validation must keep the original real recovery resource/schedule conditions.')
        modules = [(name, module) for name, module in model.named_modules() if isinstance(module, VAELinear)]
        gates = [(name, module) for name, module in modules if '.mlp.gate_proj' in name]
        c.require(len(modules) == 180 and len(gates) == 36
            and all(module.codebook_dim == 64 and module.residual_stages == 2 for _, module in gates),
            'Require all144 real attention and all36 real D64 gate codecs.')
        self.active = {name: p for name, p in model.named_parameters() if p.requires_grad}
        lora = {name: p for name, p in self.active.items() if 'lora_' in name}
        decoder = {name: p for name, p in self.active.items() if 'lora_' not in name}
        c.require(len(lora) == 144 and all(('.mlp.up_proj.' in name or '.mlp.down_proj.' in name)
            and (p.shape[0] == 12 if 'lora_A' in name else p.shape[1] == 12) for name, p in lora.items()),
            'Require exactly72 future dense rank12 LoRA targets.')
        c.require(decoder and all('decoder' in name for name in decoder)
            and all(p.dtype == torch.float32 for p in self.active.values()), 'Require the complete real FP32 prefix-decoder/LoRA selection.')
        self.initial = {name: p.detach().cpu().clone() for name, p in self.active.items()}
        self.evidence.update(planned_steps=500, batch=4, maximum_length=1024,
            compressed_prefix=180, gate_codecs=36, future_dense_lora_targets=72,
            decoder_leaves=len(decoder), lora_leaves=len(lora), update_rows=[], input_shapes=[], input_masks=[])

        def inputs(_module, _args, kwargs):
            ids = kwargs.get('input_ids')
            if ids is not None:
                shape = list(ids.shape)
                self.evidence['input_shapes'].append(shape)
                mask = kwargs.get('attention_mask')
                c.require(len(shape) == 2 and shape[0] == 4 and 0 < shape[1] <= 1024
                    and shape[1] % 8 == 0 and mask is not None and tuple(mask.shape) == tuple(ids.shape)
                    and ((mask == 0) | (mask == 1)).all().item() and (mask.sum(-1) > 0).all().item(),
                    f'Changed original B4/dynamic-padding/max1024 student inputs: {shape}.')
                self.evidence['input_masks'].append(dict(valid_tokens=mask.sum(-1).tolist(),
                    tensors=c.fork.state_digest({'input_ids': ids.detach().cpu(), 'attention_mask': mask.detach().cpu()})))
                print({'short_student_inputs': shape, 'valid_tokens': mask.sum(-1).tolist()}, flush=True)
        self.handle = model.register_forward_pre_hook(inputs, with_kwargs=True)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        torch = c.fork.torch
        c.require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in self.active.values()),
            'Missing/nonfinite real prefix decoder or dense-LoRA gradients.')
        gates = [p for name, p in self.active.items() if '.mlp.gate_proj.' in name and 'decoder' in name]
        c.require(gates and any(torch.count_nonzero(p.grad).item() for p in gates), 'D64 packed decoder received no actual gradient.')

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        torch = c.fork.torch
        rows = [optimizer.state[p] for p in self.active.values()]
        c.require(all(int(row['step']) == state.global_step
            and row['exp_avg'].dtype == row['exp_avg_sq'].dtype == torch.float32
            and torch.isfinite(row['exp_avg']).all().item() and torch.isfinite(row['exp_avg_sq']).all().item()
            for row in rows), 'Continuous FP32 Adam state is invalid.')
        changed = sum(int((p.detach().cpu() != self.initial[name]).sum()) for name, p in self.active.items())
        gate_changed = sum(int((p.detach().cpu() != self.initial[name]).sum()) for name, p in self.active.items()
            if '.mlp.gate_proj.' in name and 'decoder' in name)
        self.evidence['update_rows'].append(dict(step=state.global_step, changed_FP32_elements=changed,
            gate_changed_FP32_elements=gate_changed, optimizer_initialized_leaves=len(rows),
            lr=[group['lr'] for group in optimizer.param_groups]))
        print({'short_actual_update': self.evidence['update_rows'][-1]}, flush=True)
        if state.global_step == 2:
            c.require(gate_changed > 0 and max(shape[1] for shape in self.evidence['input_shapes']) == 1024,
                'Require real D64 decoder updates and a true original1024-token batch within the two updates.')
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        c.require(state.global_step == 2, 'Expected exactly two actual updates on the unchanged500-step schedule.')
        self.evidence['actual_steps'] = 2
        self.evidence['peak_bytes'] = dict(allocated=c.fork.torch.cuda.max_memory_allocated(),
                                           reserved=c.fork.torch.cuda.max_memory_reserved())
        if self.handle is not None:
            self.handle.remove()
        self.active.clear(); self.initial.clear()
