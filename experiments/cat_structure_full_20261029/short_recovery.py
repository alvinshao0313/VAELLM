"""Fresh zero-to-two recovery from the real pre-update validation round_base."""
from pathlib import Path
import gc

import common as c
from train_utils.v6_model_loader import load_v6_model_checkpoint
from transformers import AutoTokenizer


def recover(path, *, native, cat, hf, training, vae, record, save):
    path = Path(path).resolve()
    failed = c.read(c.OUTPUT / 'B/short2.json')
    meta = c.fork.load_v6_meta(str(path))
    c.require(failed['status'] == 'failed' and failed['phase'] == 'short2'
        and failed['short_recovery']['update_rows'] == []
        and failed['actual_groups'][0]['steps_per_stage'] == 2
        and path.is_relative_to(c.OUTPUT / 'B/short2_native')
        and path.parts[-3:] == ('training_rounds', '0004_gate_proj', 'round_base')
        and meta['checkpoint_kind'] == 'round_base'
        and meta['checkpoint_id'] == 'dd46f947-5ca0-41ac-b4f1-29af4821f574'
        and meta['converted_module_count'] == 180 and len(meta['pending_dense_targets']) == 72
        and meta['completed_categories'] == list(c.CATEGORIES[:4])
        and len(meta['extra_meta']['distill_stage_history']) == 4
        and meta['extra_meta']['active_category'] == 'gate_proj' and meta['lora_config'] is None,
        'Only the actual unupdated full36 B short-fit round_base may supply this fresh recovery check.')
    source_runtime = c.fork.load_v6_cat_runtime_state(str(path), map_location='cpu')
    c.fork.validate_cat_runtime_identity(source_runtime['runtime_identity'], record['expected_runtime_identity'])
    record['short_round_base'] = dict(path=str(path), checkpoint_id=meta['checkpoint_id'],
        metadata_sha256=c.sha(path / 'checkpoint_meta.json'), runtime_sha256=c.sha(path / 'cat_runtime_state.pt'),
        weight_stat=c.fork.stat(path / meta['state_dict_file']), prior_optimizer_updates=0,
        failed_fit_report=str(c.OUTPUT / 'B/short2.json'), failed_fit_report_sha256=c.sha(c.OUTPUT / 'B/short2.json'))
    record['short_round_base'].update(global_rng_restored=False, base_seed=31, trainer_stage_seed=35,
        semantics='Fresh recovery of the actual unupdated fitted model; not Trainer step0 or optimizer exact-resume.')
    cat.output_dir = native._build_run_output_dir(cat.output_dir, vae.model_path)
    native.configure_deterministic_mode(True)
    native.set_seed(31)
    model, loaded_meta, loaded = load_v6_model_checkpoint(str(path), base_model_path=vae.model_path,
        map_location='cpu', strict=True, expected_kind='round_base')
    c.require(loaded_meta == meta and not loaded.missing_keys and not loaded.unexpected_keys,
        'The real validation round_base failed canonical strict loading.')
    teacher = native.DistillTeacherRuntime(model_path=vae.model_path, access_token=hf.access_token,
        forward_device=native.resolve_distill_train_device(cat.train_device),
        dtype=native.resolve_distill_teacher_dtype(training, model),
        model_offload=training.distill_teacher_model_offload, logger=native.log)
    try:
        result = native.run_after_category_distill(model=model, category='gate_proj', cat_args=cat,
            vae_args=vae, training_args=training, logger=native.log, lora_round_idx=4,
            transpose_modules=cat.transpose_modules.split(','), only_decoder_projections=True,
            compression_categories=c.CATEGORIES, teacher_runtime=teacher, newly_compressed_target_count=36,
            current_category_target_names=tuple(f'model.layers.{i}.mlp.gate_proj' for i in range(36)),
            online_cat=True)
        history = list(meta['extra_meta']['distill_stage_history'])
        history.append(dict(result.distill_meta))
        tokenizer = AutoTokenizer.from_pretrained(vae.model_path, use_fast=True, local_files_only=True)
        save(result.model, str(Path(cat.output_dir) / 'after_gate_proj'), checkpoint_kind='category_boundary',
            category='gate_proj', completed_categories=c.CATEGORIES[:5], compression_categories=c.CATEGORIES,
            cat_args=cat, vae_args=vae, training_args=training, tokenizer=tokenizer,
            base_model_path=vae.model_path, distill_stage_meta=result.distill_meta,
            distill_stage_history=history, round_idx=result.next_lora_round_idx, cat_runtime_state=source_runtime)
    finally:
        del teacher
        gc.collect()
        c.fork.torch.cuda.empty_cache()
