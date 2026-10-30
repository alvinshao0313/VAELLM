"""Original500 recovery and real live up/down through the unchanged N fitter."""
from dataclasses import replace
import gc
import logging
from pathlib import Path

import torch
from transformers import TrainerCallback

from experiments.adaptive_native_model_20261029 import checkpoint, recovery
from experiments.adaptive_native_model_20261029.handoff_validation import frozen_digest
from experiments.adaptive_vector_rvq import common as c
from train_utils.cat_runtime_state_v6 import restore_cat_runtime_state
from train_utils.cat_data_prep import LinearPrepRef, build_outlier_channel_index_plan
from . import artifact, sources as s

CATEGORIES = recovery.CATEGORIES
log = logging.getLogger(__name__)


def layer_artifacts(roots, arm, *, short):
    found = {}
    for root in roots:
        report = c.read(root / 'report.json')
        expected = 'completed_short_actual_group_paths' if short else 'completed_formal_stage3_and_once_routes'
        c.require(report['status'] == expected and report['exit_code'] == 0 and report['short_validation'] == short
            and report['live_after_o'] == s.BASE_ID, 'Use only the actual completed same-parent group endpoints.')
        for group in report['groups']:
            c.require(group['training']['actual_steps'] == (2 if short else 10000)
                and group['training']['schedule_max_steps'] == 10000, 'The actual stage3 exposure/schedule is not the fixed comparison.')
            for endpoint in group['endpoints']:
                name = endpoint['name']
                item = endpoint['dynamic' if arm == 'dynamic' else 'control']
                path = Path(item['path'])
                c.require(name not in found and c.sha(path) == item['sha256']
                    and path.stat().st_size <= endpoint['control']['actual_complete_bytes'], 'Actual whole-class field identity/cap changed or layer repeated.')
                found[name] = path
    expected_names = {f'model.layers.{layer}.mlp.gate_proj' for layer in range(36)}
    c.require(set(found) == expected_names, 'One whole36gate class is required, without borrowing a missing layer.')
    return found


@torch.no_grad()
def build_model(paths):
    model, source = recovery.build_raw_mixed()
    before = c.digest({name: value for name, value in model.state_dict().items() if '.mlp.gate_proj.' not in name})
    for name, path in paths.items():
        module, _saved = artifact.load(path)
        c.require(module.resolved['geometry']['name'] == name, 'Layer artifact name differs from its target.')
        checkpoint.native.set_module_by_name(model, name, module)
    c.require(c.digest({name: value for name, value in model.state_dict().items() if '.mlp.gate_proj.' not in name}) == before,
              'Building a comparison arm changed a non-gate parent field.')
    return model.eval().requires_grad_(False), source


class ObserveRecovery(TrainerCallback):
    def __init__(self, evidence, *, short):
        self.evidence, self.short = evidence, short
        self.trainer, self.probe, self.hook = None, None, None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        c.require(args.max_steps == 500 and args.per_device_train_batch_size == 4 and args.gradient_accumulation_steps == 1
            and args.gradient_checkpointing and args.save_steps == (1 if self.short else 250), 'Keep the original500 recovery and regular save contract.')
        self.parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.ids = {id(value) for value in self.parameters.values()}
        self.initial = {name: value.detach().cpu().clone() for name, value in self.parameters.items()}
        self.frozen = frozen_digest(model, self.ids)
        self.evidence.update(actual_trainable_leaves=len(self.parameters), frozen_before=self.frozen, updates=[])

        def first_input(_module, _args, fields):
            if self.probe is None and fields.get('input_ids') is not None:
                self.probe = {key: fields[key].detach().cpu().clone() for key in ('input_ids', 'attention_mask', 'position_ids') if key in fields}
        self.hook = model.register_forward_pre_hook(first_input, with_kwargs=True)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if state.global_step < 2 or state.global_step + 1 in (250, 500):
            c.require(all(value.grad is not None and value.grad.dtype == torch.float32 and torch.isfinite(value.grad).all().item()
                for value in self.parameters.values()), 'Actual whole-class recovery has a missing/nonfinite FP32 decoder/LoRA gradient.')

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        if state.global_step <= 2 or state.global_step in (250, 500):
            c.require(all(int(optimizer.state[value]['step']) == state.global_step for value in self.parameters.values()),
                'Actual Adam inventory/step count differs from the whole-class recovery.')
            changed = sum(int(torch.count_nonzero(value.detach().cpu() != self.initial[name])) for name, value in self.parameters.items())
            row = dict(step=int(state.global_step), changed_master_elements=changed,
                learning_rates=[group['lr'] for group in optimizer.param_groups], peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            self.evidence['updates'].append(row)
            print(dict(event='whole_class_recovery_update', **row), flush=True)
            if state.global_step == 2:
                c.require(changed > 0, 'The second original500 recovery update must change real masters after warmup.')
                if self.short:
                    control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        c.require(state.global_step == (2 if self.short else 500) and frozen_digest(model, self.ids) == self.frozen,
            'Whole-class recovery did not finish the declared budget or changed a frozen payload.')
        c.require(self.trainer.save_audits and self.trainer.save_audits[0]['actual_step'] == (1 if self.short else 250),
            'The regular mixed training save and return to training were not exercised.')
        self.evidence.update(actual_steps=int(state.global_step), frozen_persistent_fields_exact=True, save_audits=self.trainer.save_audits)
        self.hook.remove()
        self.hook = None
        self.parameters.clear()
        self.initial.clear()


def recover(model, stage, vae, hf, output, round_base, round_meta, *, short):
    evidence = {}
    observer = ObserveRecovery(evidence, short=short)
    trainer, teacher, native_targets, _compact, inventory = recovery.make_trainer(model, stage, vae, hf,
        output, round_base, round_meta, callbacks=[observer], validation_save_steps=1 if short else None)
    observer.trainer = trainer
    try:
        result = trainer.train()
        evidence.update(actual_inventory=inventory, trainer_metrics=result.metrics,
            actual_history=trainer.state.log_history, schedule_max_steps=500)
        probe = observer.probe
        recovery.canonical._release_cat_trainer_training_state(trainer, logger=recovery.log)
        teacher.offload_for_eval()
        trainer.teacher_runtime = None
        model, merged = recovery.finalize(trainer, stage, native_targets)
        evidence['merge'] = merged
        observer.trainer = None
        del teacher, trainer
        gc.collect()
        torch.cuda.empty_cache()
        return model, evidence, probe
    finally:
        observer.trainer = None


def fit_next(model, cat, hf, training, vae, cfgs, activation_runtime, plan, *, previous, category, output, short):
    native = recovery.native
    previous_index, index = CATEGORIES.index(previous), CATEGORIES.index(category)
    c.require(index == previous_index + 1 and category in ('up_proj', 'down_proj') and cat.channel_scope == 'layer',
        'Follow only the actual sequential live up/down with the existing layer protection.')
    pairs, missing = native._collect_sorted_category_refs(model, category=category,
        transpose_modules=cat.transpose_modules.split(','), only_decoder_projections=True, compression_categories=CATEGORIES)
    refs = [ref for _layer, ref in pairs]
    c.require(not missing and len(refs) == 36 and all(type(ref.module) is torch.nn.Linear for ref in refs), 'Fit the complete actual returned live dense category.')
    targets = {ref.name: ref.module.weight for ref in refs}
    source_digest = c.digest(targets)
    gates = c.digest({name: tensor for name, tensor in model.state_dict().items() if '.mlp.gate_proj.' in name})
    activation_runtime, plan, _budget = native._refresh_channel_protection_after_category(model=model,
        current_category=previous, current_category_idx=previous_index, active_categories=CATEGORIES, cat_args=cat,
        resolved_category_cfgs=cfgs, transpose_modules=cat.transpose_modules.split(','), only_decoder_projections=True,
        compression_categories=CATEGORIES, target_layers='all', skip_layer_keys=set(), activation_runtime=activation_runtime,
        global_adaptive_plan=plan, global_raw_budget_value=0, resolved_channel_mode=cat.channel_protect_mode,
        resolved_channel_rank_metric=cat.channel_rank_metric, resolved_channel_mlp_rank_metric=cat.channel_mlp_rank_metric,
        channel_axis=cat.channel_axis, category_channel_protect_count={name: cfg.channel_protect_count for name, cfg in cfgs.items()},
        linear_group_size=36, plan_is_main=True, plan_world_size=1, run_output_dir=str(output), logger=native.log)
    cfg = cfgs[category]
    actmax, actmean = native._activation_views_for_refs(refs, activation_runtime, category=category, rank_metric=cat.channel_rank_metric)
    protection = build_outlier_channel_index_plan(group_refs=[LinearPrepRef(name=ref.name, weight=ref.module.weight,
        in_features=ref.module.in_features, out_features=ref.module.out_features, transpose=ref.transpose) for ref in refs],
        activation_weight_by_linear=actmax, activation_abs_mean_by_linear=actmean, channel_protect_count=cfg.channel_protect_count,
        channel_axis=cat.channel_axis, channel_scope='layer', channel_rank_metric=cat.channel_rank_metric, channel_min_per_layer=cat.channel_min_per_layer)
    model.to('cpu')
    for module in model.modules():
        clear = getattr(module, 'clear_decoded_weight_cache', None)
        if callable(clear):
            clear()
    gc.collect()
    torch.cuda.empty_cache()
    c.require(c.digest(targets) == source_digest, 'Statistics refresh/offload changed the actual live fitter source.')
    # A three-decoder fresh consumer consumes more global initialization RNG
    # than fixed2. Start each arm's next VAE from the same explicit fresh seed.
    native.set_seed(cat.seed)
    payload = native.train_group_vae_payload(model=model, group_refs=refs, group_tag=f'whole_class.{category}.L0-35',
        runtime_cfg=replace(cfg, steps=2) if short else cfg, vae_args=vae, training_args=training,
        train_device=cat.train_device, convert_device=cat.convert_device, do_convert=True, batch_size=8192,
        log_every=cat.log_every, eval_every=cat.eval_every, eval_blocks=cat.eval_blocks, gpu_resident_data=cat.gpu_resident_data,
        activation_runtime=activation_runtime, channel_protect_mode=cat.channel_protect_mode, channel_plan=protection,
        channel_scope=cat.channel_scope, channel_rank_metric=cat.channel_rank_metric, channel_axis=cat.channel_axis,
        channel_quant=cat.channel_quant, channel_min_per_layer=cat.channel_min_per_layer,
        deterministic=cat.deterministic, shuffle_seed=index * 100000 + 31)
    c.require(c.digest(targets) == source_digest, 'Fitting modified the source before native replacement.')
    native.apply_group_vae_payload(model=model, group_refs=refs, payload=payload,
        group_tag=f'whole_class.{category}.L0-35', convert_device=cat.convert_device)
    c.require(c.digest({name: tensor for name, tensor in model.state_dict().items() if '.mlp.gate_proj.' in name}) == gates,
        'Actual live next-category fitting/replacement changed compact gate fields.')
    counts = [len(checkpoint.inventories(model)[key]) for key in ('native', 'compact', 'pending_dense')]
    c.require(counts == ([180, 36, 36] if category == 'up_proj' else [216, 36, 0]), 'Incorrect whole-class sequential mixed topology.')
    return activation_runtime, plan, dict(category=category, source_target_digest=source_digest, raw_live_source='Actual own preceding recovery/fresh merged dense weights',
        actual_stages=2, actual_updates_per_stage=2 if short else 10000, schedule_steps=2 if short else 10000,
        vector_batch_per_layer=8192, shuffle_seed=index * 100000 + 31, original_M36_pooled_BSQ=True,
        fresh_VAE_initialization_seed=int(cat.seed),
        gate_fields_exact=True, inventories=checkpoint.inventories(model), category_protection=cfg.channel_protect_count,
        category_rotation_block=cfg.weight_rotation_block_size)
