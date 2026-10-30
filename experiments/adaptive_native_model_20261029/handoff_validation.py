"""Observe real two-update/save/retrain behavior without changing the500 schedule."""
import json

import torch
from transformers import TrainerCallback

from experiments.adaptive_vector_rvq import common as c


def frozen_digest(model, trainable_ids):
    return c.digest({name: value for name, value in model.state_dict(keep_vars=True).items() if id(value) not in trainable_ids})


class HandoffCheck(TrainerCallback):
    def __init__(self, evidence):
        self.evidence = evidence
        self.active, self.initial = {}, {}
        self.trainer = None
        self.hook = None
        self.probe = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        c.require(args.max_steps == 500 and args.per_device_train_batch_size == 4 and args.gradient_accumulation_steps == 1
                  and args.gradient_checkpointing and args.max_grad_norm == 1.5 and args.save_steps == 1,
                  'Preserve original500/B4/accumulation/checkpointing; only the short save trigger is1.')
        self.active = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        c.require(self.active and all(parameter.dtype == torch.float32 for parameter in self.active.values()),
                  'All actual native/compact D and dense LoRA training leaves must remain FP32.')
        self.initial = {name: parameter.detach().cpu().clone() for name, parameter in self.active.items()}
        self.trainable_ids = {id(parameter) for parameter in self.active.values()}
        self.frozen = frozen_digest(model, self.trainable_ids)
        selected = self.trainer.model_level_trainable_selection
        c.require(self.trainable_ids == {id(parameter) for parameter in (*selected.decoder_parameters.values(), *selected.lora_parameters.values())},
                  'Actual trainable parameters do not exactly match the combined optimizer inventories.')
        self.evidence.update(schedule_max_steps=500, actual_save_interval=1, inputs=[], updates=[],
                             actual_trainable_leaves=len(self.active), frozen_before=self.frozen)

        def inputs(_module, _args, fields):
            ids, mask = fields.get('input_ids'), fields.get('attention_mask')
            if ids is None:
                return
            c.require(ids.ndim == 2 and ids.shape[0] == 4 and 0 < ids.shape[1] <= 1024 and ids.shape[1] % 8 == 0
                and mask is not None and tuple(mask.shape) == tuple(ids.shape) and ((mask == 0) | (mask == 1)).all().item()
                and (mask.sum(-1) > 0).all().item(), 'The real B4/dynamic-padding/max1024 input contract changed.')
            row = dict(shape=list(ids.shape), valid_tokens=mask.sum(-1).tolist(),
                       digest=c.digest({'input_ids': ids, 'attention_mask': mask}))
            self.evidence['inputs'].append(row)
            if self.probe is None:
                self.probe = {name: fields[name].detach().cpu().clone() for name in ('input_ids', 'attention_mask', 'position_ids') if name in fields}
            print(json.dumps(dict(event='mixed_actual_student_inputs', **row)), flush=True)
        self.hook = model.register_forward_pre_hook(inputs, with_kwargs=True)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        c.require(all(parameter.grad is not None and parameter.grad.dtype == torch.float32
                      and torch.isfinite(parameter.grad).all().item() for parameter in self.active.values()),
                  'A real decoder or dense-LoRA gradient is missing/nonfinite/not FP32.')
        compact = [parameter for _name, module in self.trainer.selected_compact for parameter in module.parameters()]
        c.require(any(torch.count_nonzero(parameter.grad).item() for parameter in compact), 'The true training objective did not reach compact gate decoders.')

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        states = [optimizer.state[parameter] for parameter in self.active.values()]
        c.require(all(int(value['step']) == state.global_step and value['exp_avg'].dtype == value['exp_avg_sq'].dtype == torch.float32
            and torch.isfinite(value['exp_avg']).all().item() and torch.isfinite(value['exp_avg_sq']).all().item() for value in states),
            'All actual Adam state must be finite FP32 and advance continuously on both updates.')
        changed = {name: int(torch.count_nonzero(parameter.detach().cpu() != self.initial[name])) for name, parameter in self.active.items()}
        compact_changes = sum(value for name, value in changed.items() if '.mlp.gate_proj.' in name)
        lora_ids = {id(parameter) for parameter in self.trainer.model_level_trainable_selection.lora_parameters.values()}
        lora_changes = sum(changed[name] for name, parameter in self.active.items() if id(parameter) in lora_ids)
        row = dict(step=int(state.global_step), master_changed_elements=sum(changed.values()), compact_master_changed_elements=compact_changes,
            dense_lora_master_changed_elements=lora_changes, Adam_initialized_leaves=len(states),
            learning_rates=[group['lr'] for group in optimizer.param_groups],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        self.evidence['updates'].append(row)
        print(json.dumps(dict(event='mixed_actual_update', **row)), flush=True)
        if state.global_step == 2:
            c.require(compact_changes > 0 and lora_changes > 0 and max(row['shape'][1] for row in self.evidence['inputs']) == 1024,
                      'Two real updates must change compact and dense-LoRA masters and cover a real1024-token batch.')
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        c.require(state.global_step == 2 and self.trainer.save_audits and self.trainer.save_audits[0]['actual_step'] == 1,
                  'Need actual step1 mixed state save and continuation through actual step2.')
        c.require(frozen_digest(model, self.trainable_ids) == self.frozen, 'A frozen native/compact code/route/protection/backbone/dense-base field changed.')
        self.evidence.update(actual_steps=2, frozen_persistent_fields_exact=True, save_audits=self.trainer.save_audits)
        self.hook.remove()
        self.hook = None
        self.active.clear()
        self.initial.clear()
