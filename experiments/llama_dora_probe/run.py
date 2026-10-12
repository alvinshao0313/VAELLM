"""One fresh arm of the isolated64-update after-gate LoRA/DoRA comparison."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
from pathlib import Path
import time
import traceback

import torch
from torch.utils.data import Dataset, SequentialSampler
from transformers import AutoTokenizer, TrainerCallback, set_seed

from experiments.llama_dora_probe.prepare import check, configuration, read, SOURCE_ID
from experiments.llama_dora_probe.state import (compare_states, merge_adapter, output_difference,
    require, select_adapter, state_records, tensor_sha)
from experiments.longbench_v1.common import sha256, write_json
from litebsq.vae_linear import clear_model_vae_linear_cache
from train_utils.cat_after_category_common import (CanonicalCatSFTTrainer, _build_training_arguments,
                                                  _release_cat_trainer_training_state)
from train_utils.checkpoint_v6 import _collect_vae_linear_specs, save_v6_full_checkpoint
from train_utils.distill_data import build_distill_data_collator
from train_utils.distill_decoder import NamedMainDecoderTarget, finalize_main_decoder_targets
from train_utils.distill_precision import configure_distill_precision
from train_utils.distill_teacher import DistillTeacherRuntime
from train_utils.model_level_optimizer import ModelLevelOptimizerLRConfig, attach_model_level_optimizer_contract
from train_utils.runtime_snapshot import to_runtime_jsonable
from train_utils.v6_model_loader import load_v6_model_checkpoint


class FixedOrderTrainer(CanonicalCatSFTTrainer):
    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)


class FixedDocuments(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return dict(self.samples[index])


class Observations(TrainerCallback):
    def __init__(self, selection):
        self.groups = {'lora': selection.lora_parameters, 'decoder': selection.decoder_parameters}
        self.gradients, self.steps = [], []
        self.started = None

    def on_train_begin(self, args, state, control, **kwargs):
        expected = {id(p) for params in self.groups.values() for p in params.values()}
        optimizer = kwargs['optimizer']
        require({id(p) for group in optimizer.param_groups for p in group['params']} == expected,
                'Optimizer omitted magnitude/decoder parameters or included frozen weights.')

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize(); self.started = time.monotonic()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        row = {'step': state.global_step + 1}
        for name, params in self.groups.items():
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values()),
                    'Missing/nonfinite gradients: ' + name)
            row[name + '_gradient_l2'] = math.sqrt(math.fsum(float(p.grad.double().square().sum()) for p in params.values()))
        mags = [p for name, p in self.groups['lora'].items() if '.lora_magnitude_vector.' in name]
        row['magnitude_gradient_l2'] = math.sqrt(math.fsum(float(p.grad.double().square().sum()) for p in mags))
        self.gradients.append(row)

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.steps.append({'step': state.global_step, 'seconds': time.monotonic() - self.started})


@torch.no_grad()
def held_loss(trainer, model, batches):
    model.eval()
    values = []
    for batch in batches:
        with trainer.compute_loss_context_manager():
            loss = trainer.compute_loss(model, trainer._prepare_inputs(dict(batch)))
        require(torch.isfinite(loss), 'Nonfinite heldout native loss.')
        values.append(float(loss))
    return {'batch_losses': values, 'mean_native_kl_top_mass': math.fsum(values) / len(values),
            'aggregation': 'Equal mean of16 fixed B4 native losses, with original prompt/response weighting.'}


@torch.no_grad()
def probe_output(model, sample):
    model.eval()
    ids = sample['input_ids'].unsqueeze(0).to('cuda:0')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       logits_to_keep=1).logits.float().cpu()
    return result


def run(preparation, arm):
    record = read(preparation); output = Path(record['output_dir']) / arm
    require(arm in record['arms'] and record['status'] == 'prepared' and not output.exists(),
            'Use one fresh prepared arm; no resume/overwrite.')
    require(os.environ.get('CUDA_VISIBLE_DEVICES', '').isdigit() and torch.cuda.device_count() == 1,
            'Explicitly assign one available GPU after source review.')
    check(record)
    stage = configuration(output); cfg = stage.config
    require(to_runtime_jsonable(cfg) == record['configuration'], 'Prepared native configuration changed.')
    output.mkdir(); report = {'status': 'running', 'arm': arm, 'source_checkpoint_id': SOURCE_ID,
        'preparation_sha256': sha256(preparation), 'pid': os.getpid(), 'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'worker_start_ticks': int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'exact_resume_supported': False, 'quality_evaluation': 'Fixed heldout native KL only; no downstream acceptance.'}
    write_json(output / 'report.json', report)
    logger = logging.getLogger('llama_dora_probe'); logging.basicConfig(level=logging.INFO)
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    try:
        samples = torch.load(record['samples_path'], map_location='cpu', weights_only=True)
        tokenizer = AutoTokenizer.from_pretrained(record['base_model_path'], local_files_only=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        collator = build_distill_data_collator(tokenizer, model_max_length=1024, dynamic_padding=True)
        held = [collator([dict(x) for x in samples['heldout'][i:i + 4]]) for i in range(0, 64, 4)]
        for group in ('train', 'heldout'):
            actual = [{'begin_document': i + (256 if group == 'heldout' else 0),
                'shape': list((batch := collator([dict(x) for x in samples[group][i:i + 4]]))['input_ids'].shape),
                'tensor_sha256': {key: tensor_sha(value) for key, value in batch.items()}}
                for i in range(0, len(samples[group]), 4)]
            require(actual == record['batch_records'][group], 'Fixed canonical batches changed: ' + group)
        model, meta, loaded = load_v6_model_checkpoint(record['source_checkpoint'], map_location='cpu', strict=True)
        require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys,
                'Fresh source did not load strictly.')
        before = state_records(model); specs = _collect_vae_linear_specs(model)
        targets = tuple((name, model.get_submodule(name)) for name in record['decoder_names'])
        mutable = {name + '.weight' for name in record['dense_names']}
        for name, module in targets:
            mutable.update(name + '._parallel_stage_decoder.' + key for key, _ in module._parallel_stage_decoder.named_parameters())
        require(mutable <= before.keys() and len(targets) == 160, 'Mutable state inventory differs.')
        set_seed(31)
        selection = select_adapter(model, record['dense_names'], targets, use_dora=arm == 'dora')
        model = selection.peft_model; original_use_cache = model.config.use_cache
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        model.enable_input_require_grads(); model.to('cuda:0')
        args = _build_training_arguments(stage, is_iterable=False, group_by_length=False)
        args.dataloader_num_workers = 0; args.per_device_eval_batch_size = 4
        configure_distill_precision(selection, components=cfg.opt.distill_fp32_components, training_args=args, logger=logger)
        adapter_before = {name: tensor_sha(p) for name, p in selection.lora_parameters.items()}
        observer = Observations(selection)
        teacher = DistillTeacherRuntime(model_path=record['base_model_path'], access_token=None,
            forward_device='cuda:0', dtype=torch.bfloat16, model_offload=cfg.runtime.teacher_model_offload, logger=logger)
        trainer = FixedOrderTrainer(model=model, train_dataset=FixedDocuments(samples['train']), args=args,
            processing_class=tokenizer, data_collator=collator, callbacks=[observer],
            loss_config=cfg.loss, teacher_runtime=teacher, distill_hif4_act_controller=None,
            teacher_output_offload=cfg.runtime.teacher_output_offload,
            teacher_output_pin_memory=cfg.runtime.teacher_output_pin_memory,
            teacher_output_chunk_tokens=cfg.runtime.teacher_output_chunk_tokens)
        attach_model_level_optimizer_contract(trainer, selection=selection,
            lr_config=ModelLevelOptimizerLRConfig(learning_rate=cfg.opt.learning_rate,
                decoder_lr=cfg.opt.decoder_lr, weight_decay=cfg.opt.weight_decay))
        report['heldout_before'] = held_loss(trainer, model, held)
        observed_batches = []
        def inputs_hook(module, args, kwargs):
            if module.training:
                observed_batches.append({key: tensor_sha(kwargs[key]) for key in ('input_ids', 'attention_mask')})
        handle = model.register_forward_pre_hook(inputs_hook, with_kwargs=True)
        try:
            trained = trainer.train()
        finally:
            handle.remove()
        require(trained.global_step == 64 and len(observer.steps) == len(observer.gradients) == 64,
                'The arm did not complete64 real updates.')
        expected = [{key: b['tensor_sha256'][key] for key in ('input_ids', 'attention_mask')}
                    for b in record['batch_records']['train']]
        require(observed_batches == expected, 'Actual model batches differ from the shared fixed order.')
        report.update(training=trained.metrics, gradients=observer.gradients, steps=observer.steps,
            actual_batches_exact=True, adapter_sha256_before=adapter_before,
            adapter_sha256_after={name: tensor_sha(p) for name, p in selection.lora_parameters.items()})
        report['heldout_unmerged'] = held_loss(trainer, model, held)
        unmerged_output = probe_output(model, samples['heldout'][0])
        _release_cat_trainer_training_state(trainer, logger=logger)
        finalized = finalize_main_decoder_targets(tuple(NamedMainDecoderTarget(name, module) for name, module in targets))
        require(finalized == 160, 'Prefix decoder finalization missed targets.')
        model = merge_adapter(model, stage); model.requires_grad_(False)
        clear_model_vae_linear_cache(model)
        report['heldout_merged_bf16'] = held_loss(trainer, model, held)
        merged_output = probe_output(model, samples['heldout'][0])
        report['merge_output_difference'] = output_difference(unmerged_output, merged_output)
        report['merge_numeric_boundary'] = 'Separate BF16 low-rank/DoRA forwards and merged dense BF16 need not be bitwise equal; compare held losses and output relativeL2 before making a quality claim.'
        model.config.use_cache = original_use_cache; model.to('cpu'); clear_model_vae_linear_cache(model)
        after = state_records(model); report['frozen_and_storage'] = compare_states(before, after, mutable)
        require(_collect_vae_linear_specs(model) == specs, 'Compressed inference specs changed.')
        require(any(k.endswith('.weight') and k in {n + '.weight' for n in record['dense_names']}
                    for k in report['frozen_and_storage']['changed_mutable_tensors']), 'No dense merge survived.')
        saved = save_v6_full_checkpoint(model, str(output / 'checkpoint'), checkpoint_kind='category_boundary',
            compressed_targets=meta['compressed_targets'], pending_dense_targets=meta['pending_dense_targets'],
            skip_targets=meta['skip_targets'], completed_categories=meta['completed_categories'],
            compression_categories=meta['compression_categories'], target_layers=meta['target_layers'],
            target_modules=meta['target_modules'], base_model_path=record['base_model_path'], tokenizer=tokenizer,
            after_category_mode='remaining_lora_prefix_decoder', runtime_audit={'probe': arm, 'exact_resume_supported': False},
            extra_meta={'stage': 'after_category', 'category': 'gate_proj', 'source_checkpoint_id': SOURCE_ID})
        serialized = torch.load(saved['state_dict'], map_location='cpu', weights_only=True, mmap=True)
        require(state_records(serialized) == after, 'Serialized state differs from finalized state.')
        del serialized, model, trainer, selection, observer, targets, teacher
        gc.collect(); torch.cuda.empty_cache()
        fresh, fresh_meta, fresh_load = load_v6_model_checkpoint(saved['output_dir'], map_location='cpu', strict=True)
        require(not fresh_load.missing_keys and not fresh_load.unexpected_keys and state_records(fresh) == after
                and _collect_vae_linear_specs(fresh) == specs, 'Fresh strict reload differs.')
        require(fresh_meta['checkpoint_id'] == saved['checkpoint_id'] != SOURCE_ID and fresh_meta['lora_config'] is None,
                'Saved checkpoint retained adapters or source UUID.')
        check(record)
        report.update(status='completed', exit_code=0, checkpoint_id=saved['checkpoint_id'],
            checkpoint_dir=saved['output_dir'], source_state=before, final_state=after,
            serialized_and_fresh_reload_exact=True, inference_specs_exact=True, frozen_code_bits_changed=0)
    except BaseException as error:
        report.update(status='failed', exit_code=1, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report.update(seconds=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(output / 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--arm', choices=('lora', 'dora'), required=True)
    args = parser.parse_args()
    run(args.preparation, args.arm)
