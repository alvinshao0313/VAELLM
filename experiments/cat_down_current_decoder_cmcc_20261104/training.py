"""Only scope selection changes; keep the original Trainer, KD, save and finalize."""
import gc
import json

from . import bindings as b
from experiments.adaptive_class_20261030 import continuation as flow
from experiments.adaptive_native_model_20261029.training_checkpoint import MixedCatTrainer


def selection(model, stage):
    r, cfg = b.recovery, stage.config
    groups = b.checkpoint.inventories(model)
    names = tuple(name for name in groups['native'] if name.endswith('.mlp.down_proj'))
    b.require([len(groups[key]) for key in ('native', 'compact', 'pending_dense')] == [216, 36, 0]
        and set(names) == {f'model.layers.{i}.mlp.down_proj' for i in range(36)},
        'Select exactly all36 real native down decoders from the original raw.')
    native = tuple((name, model.get_submodule(name)) for name in names)
    selected = r.build_model_level_trainable_selection(model, aux=cfg.aux,
        fp32_components=cfg.opt.distill_fp32_components, compressed_modules=(), dense_target_modules=(),
        decoder_modules=native, rank=cfg.lora.rank, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout,
        rank_explicit=cfg.lora.rank_explicit, train_decoder=True, train_lora=False, freeze=True)
    r.configure_distill_precision(selected, components=cfg.opt.distill_fp32_components, training_args=stage, logger=r.log)
    lr = r.ModelLevelOptimizerLRConfig(learning_rate=cfg.opt.learning_rate, decoder_lr=cfg.opt.resolved_decoder_lr(),
        weight_decay=cfg.opt.weight_decay, norm_lr=cfg.aux.norm_lr, lm_head_lr=cfg.aux.lm_head_lr,
        residual_lora_lr=cfg.aux.residual_lora_lr)
    active = {name: p for name, p in model.named_parameters() if p.requires_grad}
    b.require(len(active) == len(selected.decoder_parameters) == 216 and sum(p.numel() for p in active.values()) == 767232
        and {id(p) for p in active.values()} == {id(p) for p in selected.decoder_parameters.values()}
        and all(p.dtype == b.torch.float32 and any(name.startswith(prefix+'.') for prefix in names) for name,p in active.items())
        and not any((selected.lora_parameters, selected.norm_parameters, selected.lm_head_parameters, selected.residual_lora_parameters)),
        'The actual trainables must be exclusively216/767232 FP32 down master leaves, LoRA0.')
    b.require(all(not getattr(model.get_submodule(name), 'trainable_decode', False)
        for name in groups['native'] if name not in names), 'Frozen native modules must retain their original decode mode.')
    optimizer_groups = r.build_model_level_param_groups(selected, lr_config=lr, model=model)
    info = dict(effective_mode='current_decoder', reference_stage_mode=stage.mode,
        native_modules=36, compact_modules=0, dense_lora_targets=0, decoder_leaves=216,
        trainable_elements=767232, actual_trainable_leaves=len(active), selected_native_names=list(names),
        frozen_native_modules=180, frozen_compact_modules=36,
        optimizer_groups=[dict(name=g['group_name'], leaves=len(g['params']), lr=g['lr'], weight_decay=g['weight_decay']) for g in optimizer_groups])
    return selected, native, (), lr, info


class PeakMixedCatTrainer(MixedCatTrainer):
    """Technical-only replay of an actual maximum-padding original-stream B4 batch."""
    def get_train_dataloader(self):
        batch = b.torch.load(b.DATA_FIXTURE, map_location='cpu', weights_only=True)['peak_batch']
        rows = [{key: value[i] for key,value in batch.items()} for _ in range(2) for i in range(4)]
        loader = b.torch.utils.data.DataLoader(rows, batch_size=4, shuffle=False,
            collate_fn=self.data_collator, num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory)
        return self.accelerator.prepare(loader)


def make_trainer(model, stage, vae, hf, output, round_base, round_meta, callbacks, *, short):
    r, cfg = b.recovery, stage.config
    tokenizer, data = r.canonical._prepare_after_category_dataset(model=model, vae_args=vae, cfg=cfg)
    previous_use_cache = bool(model.config.use_cache)
    model.config.use_cache = False
    r.install_precision_runtime(model, b.torch.bfloat16)
    selected, native, compact, lr, info = selection(model, stage)
    selected.peft_model.enable_input_require_grads()
    selected.peft_model.to(stage.train_device).train()
    args = r.canonical._build_training_arguments(stage, is_iterable=data.is_iterable, group_by_length=data.group_by_length)
    args.output_dir = str(output / 'trainer_state')
    regular_save = dict(strategy=stage.save_strategy, save_steps=stage.save_steps, save_only_model=stage.save_only_model)
    if short:
        args.save_steps = 1
    teacher = r.native.DistillTeacherRuntime(model_path=vae.model_path, access_token=hf.access_token,
        forward_device=stage.train_device, dtype=r.native.resolve_distill_teacher_dtype(args, model),
        model_offload=cfg.runtime.teacher_model_offload, logger=r.log)
    cls = PeakMixedCatTrainer if short else MixedCatTrainer
    trainer = cls(model=selected.peft_model, train_dataset=data.train_dataset, eval_dataset=data.eval_dataset,
        args=args, callbacks=callbacks, processing_class=tokenizer,
        data_collator=r.canonical.build_distill_data_collator(tokenizer, model_max_length=cfg.data.model_max_length,
            dynamic_padding=cfg.data.dynamic_padding), loss_config=cfg.loss, teacher_runtime=teacher,
        teacher_output_offload=cfg.runtime.teacher_output_offload,
        teacher_output_pin_memory=cfg.runtime.teacher_output_pin_memory,
        teacher_output_chunk_tokens=cfg.runtime.teacher_output_chunk_tokens,
        mixed_round_base=round_base, mixed_round_meta=round_meta, selected_native=native,
        selected_compact=compact, regular_save=regular_save)
    r.attach_model_level_optimizer_contract(trainer, selection=selected, lr_config=lr)
    trainer.mixed_original_use_cache = previous_use_cache
    return trainer, teacher, native, compact, info


class ObserveDown(flow.ObserveRecovery):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        super().on_train_begin(args, state, control, model=model, **kwargs)
        b.require(len(self.parameters) == 216 and sum(p.numel() for p in self.parameters.values()) == 767232
            and args.dataloader_num_workers == 16 and not args.group_by_length,
            'Keep the exact down selection and original iterable loader settings.')
        self.evidence['inputs'], self.evidence['gradient_layers'] = [], []
        fixture = b.torch.load(b.DATA_FIXTURE,map_location='cpu',weights_only=True)
        self.expected_first = fixture['peak_batch' if self.short else 'first_batch']
        self.hook.remove()
        def inputs(_module, _args, fields):
            if fields.get('input_ids') is None:
                return
            batch = {key: fields[key].detach().cpu().clone() for key in ('input_ids','attention_mask','position_ids') if key in fields}
            shape = list(batch['input_ids'].shape)
            b.require(shape[0] == 4 and 0 < shape[1] <= 1024, 'Require actual originalB4/cap1024.')
            if self.probe is None:
                b.require(all(key in batch and b.torch.equal(batch[key],self.expected_first[key])
                    for key in ('input_ids','attention_mask'))
                    and ('position_ids' not in self.expected_first or b.torch.equal(batch['position_ids'],self.expected_first['position_ids'])),
                    'Actual GPU first input must equal the CPU actual peak replay or formal original first batch.')
                self.evidence['first_input_CPU_identity_exact'] = True
                self.expected_first = None
                self.probe = batch
            if state.global_step < 2:
                self.evidence['inputs'].append(dict(step=int(state.global_step), shape=shape, digest=b.c.digest(batch)))
        self.hook = model.register_forward_pre_hook(inputs, with_kwargs=True)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        super().on_pre_optimizer_step(args, state, control, **kwargs)
        if state.global_step < 2:
            layers = [name for name,module in self.trainer.selected_native
                if any(p.grad is not None and bool(b.torch.count_nonzero(p.grad)) for p in module.parameters() if p.requires_grad)]
            b.require(len(layers) == 36, 'Loss must reach every down decoder, including earlier layers through frozen modules.')
            self.evidence['gradient_layers'].append(dict(step=int(state.global_step)+1, nonzero_down_layers=layers))

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        result = super().on_step_end(args, state, control, optimizer=optimizer, **kwargs)
        if state.global_step <= 2:
            changed = {name: sum(int(b.torch.count_nonzero(p.detach().cpu()!=self.initial[key]))
                for key,p in self.parameters.items() if key.startswith(name+'.')) for name,_ in self.trainer.selected_native}
            row = dict(step=int(state.global_step), changed_elements_by_down_layer=changed,
                BF16_changed_elements=sum(int(b.torch.count_nonzero(p.detach().cpu().bfloat16()!=self.initial[name].bfloat16())) for name,p in self.parameters.items()))
            b.require(all(int(optimizer.state[p]['step']) == state.global_step
                and optimizer.state[p]['exp_avg'].dtype == optimizer.state[p]['exp_avg_sq'].dtype == b.torch.float32 for p in self.parameters.values()),
                'Original Adam state must advance with finite FP32 masters.')
            if state.global_step == 1:
                b.require(not any(changed.values()), 'Original warmupLR0 first step must not move masters.')
            else:
                b.require(all(value > 0 for value in changed.values()) and row['BF16_changed_elements'] > 0,
                    'Second scheduled update must change all36 actual down decoders and survive BF16 deployment.')
            self.evidence.setdefault('scope_updates', []).append(row)
            print(json.dumps(dict(event='current_down_scope_update', **row)), flush=True)
        return result


def recover(model, stage, vae, hf, output, round_base, round_meta, *, short):
    evidence = {}
    observer = ObserveDown(evidence, short=short)
    trainer, teacher, native, _, info = make_trainer(model, stage, vae, hf, output, round_base, round_meta, [observer], short=short)
    observer.trainer = trainer
    try:
        result = trainer.train()
        evidence.update(actual_inventory=info, trainer_metrics=result.metrics,
            actual_history=trainer.state.log_history, schedule_max_steps=500,
            short_data_replay='Original500 stream actual max-padding batch repeated twice for technical peak coverage only.' if short else None)
        probe = observer.probe
        b.recovery.canonical._release_cat_trainer_training_state(trainer, logger=b.recovery.log)
        teacher.offload_for_eval()
        trainer.teacher_runtime = None
        model, merged = b.recovery.finalize(trainer, stage, native)
        evidence['merge'] = merged
        observer.trainer = None
        del teacher, trainer
        gc.collect()
        b.torch.cuda.empty_cache()
        return model, evidence, probe
    finally:
        observer.trainer = None
