"""Bind the fixed existing recipes/archive to the two actual assembly sources."""
from __future__ import annotations

import ast
import gc
from pathlib import Path
import time

import common as c


def prepare():
    import torch
    from compressed_e2e_fintuning.args import parse_args
    from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _load_state
    from experiments.qwen3_w25.audit_e2e_decoder import persistent_inventory
    from experiments.qwen3_w25.reproduction_inputs import prepare_format_launch
    from experiments.qwen3_w25.reproduction_recipe import DATA_FILES, format_protocol, training_command
    from experiments.qwen3_w25.train_reproduction_stage import validate_configuration
    c.require(not torch.cuda.is_initialized() and c.os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preparation must hide CUDA.')
    c.require(not c.PREPARED.exists() and not c.DATA.exists(), 'Use the single fresh fixed recovery-tail output.')
    torch.set_num_threads(4)
    started = time.monotonic()
    context = c.native_context()
    old_diagnostic = c.ROOT / 'result/qwen3_8b_w25_20260930/format_ce50_chatml_joint_s256/observations/diagnostic.json'
    old = c.read(old_diagnostic)
    c.require(old['status'] == 'complete' and old['steps'] == 256 and old['joint']
        and old['bit_registration']['active_scores'] == 147849216
        and old['runtime']['trainable_decoder_elements'] == 21312 and old['runtime']['frozen_tensor_count'] == 3415,
        'The existing fixed256 reference must contain the complete production-path evidence.')
    old_meta_path = Path(old['source_dir']) / 'checkpoint_meta.json'
    old_spec = next(row for row in c.read(old_meta_path)['converted_modules'] if row['name'] == c.TARGET)
    data_proof = c.read(c.PREVIOUS / 'recovery/data_path_repair.json')
    c.require(data_proof['status'] == 'verified' and data_proof['all_five_actual_preset_loaders_passed']
        and len(data_proof['presets']) == 5, 'Reuse the verified five actual snapshot train data loaders.')
    for relative in DATA_FILES:
        c.require((c.SNAPSHOT / relative).resolve() == (c.ROOT / relative).resolve(), 'Snapshot data preset resolves to another dataset: ' + relative)
    evidence, commands, launches = {}, {}, {}
    paths = {old_diagnostic, old_meta_path, c.PREVIOUS / 'recovery/data_path_repair.json',
        c.ARCHIVE / 'preparation.json', c.ARCHIVE / 'batches.pt',
        c.ROOT / 'result/qwen3_8b_w25_20260930/data/task_manifest.json',
        c.ROOT / 'result/qwen3_8b_w25_20260930/data/sources.json',
        c.TOPIC / 'qwen_c1_dual_down_pv32_20261019/pv_cpu_loader.py'}
    stats = []
    for profile, source in c.SOURCES.items():
        meta, budget = c.checkpoint(source)
        audit_path = c.PREVIOUS / 'repaired_recovery' / profile / 'assembly/native_audit.json'
        previous = c.read(audit_path)
        c.require(meta['checkpoint_id'] == c.SOURCE_IDS[profile] == previous['checkpoint_id']
            and previous['status'] == 'verified' and previous['fresh_strict_tensors_compared'] == 3423, 'Use the actual independently verified current same-source assembly.')
        model, loaded_meta, loaded = c.load_cpu(source)
        c.require(loaded_meta == meta and not loaded.missing_keys and not loaded.unexpected_keys, 'Actual source native CPU strict load failed.')
        count = _assert_state_equal(model.state_dict(), _load_state(source), 'Current assembly native fresh strict')
        inventory, banks = persistent_inventory(model)
        target = model.get_submodule(c.TARGET)
        spec = next(row for row in meta['converted_modules'] if row['name'] == c.TARGET)
        shapes = [tuple(target.get_stage_part_vq_storage(stage, 0).shape) for stage in range(2)]
        logical = [tuple(target.get_stage_part_vq_spec(stage, 0)['logical_shape']) for stage in range(2)]
        decoder_shapes = {key.removeprefix(c.TARGET + '.'): list(model.state_dict()[key].shape) for key in inventory[c.TARGET]}
        c.require(count == 3423 and sum(map(len, inventory.values())) == 1512 and sum(map(len, banks.values())) == 504
            and shapes == [(1540096, 1, 6)] * 2 and logical == [(1540096, 1, 48)] * 2
            and sum(model.state_dict()[key].numel() for key in inventory[c.TARGET]) == 21312,
            'Existing fixed256/last-joint native target shape or full inventory is incompatible.')
        for key in ('codebook_dim', 'residual_stages', 'parallel_parts', 'transpose', 'protected_input_indices', 'stage_codebook_dims'):
            c.require(spec.get(key) == old_spec.get(key), 'Actual current L0down differs from measured fixed256: ' + key)
        # Compare actual decoder dimensions, rather than inferring shape from an element count.
        c.require(spec.get('stage_decoders') == old_spec.get('stage_decoders') and spec.get('decoders') == old_spec.get('decoders'),
                  'Actual packed native decoder specifications differ from the measured fixed256 source.')
        evidence[profile] = dict(source_dir=str(source), checkpoint_id=meta['checkpoint_id'], fresh_strict_tensors=count,
            bit_budget=budget, actual_L0_down_spec=spec, decoder_shapes=decoder_shapes, packed_shapes=shapes,
            logical_shapes=logical, old_target_spec_exact=True, all_native_decoder_tensors=1512, code_banks=504,
            previous_native_audit=str(audit_path), source_quality_not_used_for_selection=True)
        for recipe in ('decoder', 'last_joint'):
            future_source = source if recipe == 'decoder' else c.DATA / profile / '08_joint_mlp_decoder_attention/final_model'
            argv = c.training_argv(profile, recipe, future_source)
            cfg, _hf, args = parse_args(argv)
            validate_configuration(cfg, args, 2000 if recipe == 'decoder' else 1200)
            c.require(args.per_device_train_batch_size == 4 and args.gradient_accumulation_steps == 1
                and cfg.data.model_max_length == 1024 and args.seed == args.data_seed == 0
                and cfg.opt.decoder_lr == 1e-5 and args.max_grad_norm == 1.5
                and cfg.loss.loss_type == ('kl_top_mass' if recipe == 'decoder' else 'kd_top_mass')
                and cfg.loss.prompt_loss_weight == (.3 if recipe == 'decoder' else 0.)
                and cfg.loss.top_k == 100 and cfg.loss.temperature == 1, 'Original fixed recipe math changed.')
            commands[profile + '_' + recipe] = argv
        root = c.OUTPUT / profile / '07_chatml_joint256'
        root.mkdir(parents=True)
        formal = training_command('formal_reference', source_dir=source, hf_model=c.HF,
            output_dir=c.DATA / profile / '07_configuration_only', export_gpu=c.GPUS[profile]['chatml'])
        launch = prepare_format_launch(root=root, source_dir=source, hf_model=c.HF, archive_dir=c.ARCHIVE,
            environment=c.environment(c.GPUS[profile]['chatml'], format_recovery=True), formal_command=formal)
        launches[profile] = str(launch)
        paths.update([audit_path, source / 'checkpoint_meta.json', source / 'config.json', launch, root / 'format_reference.json'])
        stats.append(c.stat(source / meta['state_dict_file']))
        del model
        gc.collect()
    relevant = ('experiments/qwen3_w25/reproduction_recipe.py', 'experiments/qwen3_w25/reproduction_inputs.py',
        'experiments/qwen3_w25/train_reproduction_stage.py', 'experiments/qwen3_w25/format_recovery_diagnostic.py',
        'experiments/qwen3_w25/kl_gradient_diagnostic.py', 'experiments/qwen3_w25/kl_joint_export.py',
        'experiments/qwen3_w25/cpu_bit_export.py', 'experiments/qwen3_w25/e2e_subset_audit.py',
        'experiments/qwen3_w25/e2e_subset_export.py', 'experiments/qwen3_w25/audit_e2e_decoder.py',
        'experiments/qwen3_w25/assemble_categories.py', 'experiments/qwen3_w25/assemble_runtime_categories.py',
        'experiments/cat_recompression/longbench.py')
    paths.update(c.ROOT / name for name in relevant)
    paths.update(c.SNAPSHOT / name for name in relevant if (c.SNAPSHOT / name).is_file())
    paths.update(Path(path) for path in context.values())
    paths.update(c.HERE.glob('*.py'))
    paths.update(c.ROOT / 'experiments/night_sequence_20261026' / name for name in ('recovery_core.py', 'recovery_eval.py'))
    for path in c.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    stats.extend(c.stat(c.ROOT / relative) for relative in DATA_FILES)
    record = dict(schema='fixed_CG_same_source_recovery_tail_v1', status='cpu_verified', created_unix=time.time(),
        sources=evidence, fixed_gpus=c.GPUS, training_argv=commands, format_launch=launches,
        protocol=dict(stages=['06_KL2000', '07_ChatML_joint256', '08_07base_06attention', '09_joint1200_of_cosine2000'],
            original_schedule_steps=2000, warmup_steps=20, decoder_finish=2000, last_joint_finish=1200,
            last_joint_save_events=[400, 800, 1200], last_joint_save_total_limit=1,
            last_joint_checkpoint_retention='Original save_total_limit=1; events are not three retained checkpoints.',
            format=format_protocol(), complete_longbench_samples=4750, full8_samples=24742, max_logical_bpw=2.6,
            rank=0, automatic_retry=False, automatic_resume=False, intermediate_quality_selection=False,
            chatml_full9_independent_of_main_tail=True, longbench_timeout_seconds=86400),
        reused_evidence=dict(fixed256_diagnostic=str(old_diagnostic), fixed256_source_metadata=str(old_meta_path),
            exact_L0_decoder_and_packed_geometry=True, snapshot_data_path_repair=str(c.PREVIOUS / 'recovery/data_path_repair.json'),
            no_duplicate_GPU_smoke=True, CPU_evidence_is_not_quality_or_logit_parity=True),
        native_context=context, source_sha256={str(path): c.sha(path) for path in sorted(paths)}, input_stats=stats,
        CUDA_initialized=False, GPU_training_executed=False, elapsed_seconds=time.monotonic() - started)
    c.write(c.PREPARED, record)
    c.require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    return record


if __name__ == '__main__':
    result = prepare()
    print(dict(status=result['status'], elapsed_seconds=result['elapsed_seconds']), flush=True)
