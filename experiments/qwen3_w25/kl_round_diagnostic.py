"""Isolated full-model KL observation of the existing pure-bit round lifecycle."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

from experiments.qwen3_w25.kl_gradient_diagnostic import SOURCE_ID, batch_manifest, read_formal_config

TARGET = "model.layers.0.mlp.down_proj"
SEED = 0
OBSERVED_PER_STAGE = 128
TRAIN_BATCHES = 256
HELDOUT_BATCHES = 8


def logical_ordinals(state, logical_indices):
    """Track fixed physical bits across full-coverage affine permutations."""
    if state.n_active != state.n_bits or state.cursor != 0:
        raise ValueError("This observer requires complete coverage at cursor zero.")
    if any(not 0 <= int(index) < state.n_bits for index in logical_indices):
        raise ValueError("Logical bit index outside the bank.")
    result = [((int(index) - state.offset) * state.inverse) % state.n_bits for index in logical_indices]
    if any(state.logical_index(ordinal) != index for ordinal, index in zip(result, logical_indices)):
        raise RuntimeError("Affine observer failed its inverse identity.")
    return result


def validate_batch_archive(archive):
    batches = archive["batches"]
    if archive.get("format") != "qwen3_formal_kl_batches_v1" or len(batches) != TRAIN_BATCHES + HELDOUT_BATCHES:
        raise ValueError("Expected the fixed 256 training + 8 heldout canonical batches.")
    manifests = []
    for index, batch in enumerate(batches):
        if set(batch) != {"input_ids", "attention_mask", "labels"}:
            raise ValueError("Unexpected canonical batch fields.")
        if any(value.dtype != torch.int64 or value.device.type != "cpu" for value in batch.values()):
            raise ValueError("The immutable batch archive must contain CPU int64 tensors.")
        if batch["input_ids"].shape[0] != 4 or not 2 <= batch["input_ids"].shape[1] <= 1024:
            raise ValueError("The data archive must retain B4/max1024.")
        manifests.append(batch_manifest(batch, index, sample_offset=index * 4))
    if manifests != archive["manifests"]:
        raise ValueError("Token/label hashes or batch masks differ from the stored manifest.")
    training = {s["input_and_label_sha256"] for b in manifests[:TRAIN_BATCHES] for s in b["samples"]}
    heldout = {s["input_and_label_sha256"] for b in manifests[TRAIN_BATCHES:] for s in b["samples"]}
    if training & heldout:
        raise ValueError("Canonical training and heldout token/label identities overlap; do not resample silently.")
    return {"training_sequences": TRAIN_BATCHES * 4, "training_unique_hashes": len(training),
            "heldout_sequences": HELDOUT_BATCHES * 4, "heldout_unique_hashes": len(heldout),
            "train_heldout_hash_intersection": 0,
            "independence_scope": "disjoint observed token+label hashes; raw document IDs are unavailable",
            "max_training_valid_tokens": max(sum(s["input_tokens"] for s in b["samples"]) for b in manifests[:TRAIN_BATCHES]),
            "manifests": manifests}


def timing_summary(rows):
    steady = [row["training_step_seconds"] for row in rows[1:]]
    if not steady:
        raise ValueError("At least two genuine optimizer steps are needed for timing.")
    median, maximum = statistics.median(steady), max(steady)
    return {"first_step_seconds": rows[0]["training_step_seconds"], "steady_steps": len(steady),
            "steady_median_seconds": median, "steady_max_seconds": maximum,
            "one_256_step_arm_median_minutes": 256 * median / 60,
            "two_256_step_arms_median_minutes": 512 * median / 60,
            "one_256_step_arm_max_reference_minutes": 256 * maximum / 60,
            "scope": "synchronized forward/backward + existing Composite optimizer step; excludes observation, initialization, evaluation; extrapolation is not a runtime guarantee"}


def _sample_state(manager, fixed_indices):
    result = {}
    for spec in manager.bank_specs:
        state = manager.sampler_states[spec.canonical_key]
        ordinals = logical_ordinals(state, fixed_indices[spec.canonical_key])
        index = torch.tensor(ordinals, device=spec.device, dtype=torch.long) + spec.score_start
        score = manager.score_module.score_chunks[spec.chunk_id]
        moments = manager.bit_optimizer._state[spec.chunk_id]
        scale = manager.score_module.coordinate_scales[spec.canonical_key]
        sample = score.detach()[index].float().cpu()
        result[spec.canonical_key] = {
            "sampler": state.to_metadata(), "logical_indices": fixed_indices[spec.canonical_key],
            "active_ordinals": ordinals, "scale": scale, "radius": scale / 2,
            "scores": sample.tolist(), "absolute_score_over_radius": (sample.abs() / (scale / 2)).tolist(),
            "exp_avg": None if moments.exp_avg is None else moments.exp_avg[index].detach().float().cpu().tolist(),
            "exp_avg_sq": None if moments.exp_avg_sq is None else moments.exp_avg_sq[index].detach().float().cpu().tolist()}
    return result


def _heldout(trainer, model, batches):
    values = []
    model.eval()
    try:
        for index, batch in enumerate(batches):
            prepared = trainer._prepare_inputs(batch)
            try:
                with torch.no_grad(), trainer.compute_loss_context_manager():
                    loss = trainer.compute_loss(model, prepared)
                manifest = batch_manifest(batch, index, sample_offset=(TRAIN_BATCHES + index) * 4)
                values.append({"loss": float(loss), "denominator": manifest["loss_denominator_fp32"]})
            finally:
                trainer._release_active_teacher_targets()
    finally:
        model.train()
    if not all(torch.isfinite(torch.tensor(row["loss"])) for row in values):
        raise FloatingPointError("Nonfinite heldout KL.")
    denominator = sum(row["denominator"] for row in values)
    return {"batches": values, "equal_batch_loss": statistics.mean(row["loss"] for row in values),
            "loss_denominator_weighted": sum(row["loss"] * row["denominator"] for row in values) / denominator}


def run_kl_round_diagnostic(formal_launch, output_dir, *, steps=8, round_steps=128, data_cache=None,
                            save_final_codes=False):
    import logging
    from transformers.trainer_pt_utils import AcceleratorConfig
    from compressed_e2e_fintuning.runtime_v6 import _build_v6_tokenizer, _load_teacher, _sync_model_padding_config
    from compressed_e2e_fintuning.runtime_v6_pipeline import _enable_internal_decode_runtime
    from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from e2e_common.lazy_datasets import default_dataloader_num_workers
    from experiments.liftquant_recovery.recovery_runtime import prime_packed_cache
    from experiments.qwen3_w25.local_bits import _changed_bits, _code_banks, _digest
    from litebsq.vae_linear import VAELinear
    from sparse_bit_tuning.config import SparseBitTuningConfig
    from sparse_bit_tuning.manager import SparseBitTuningManager
    from train_utils.decoder_execution import enable_vae_linear_by_execution_plan
    from train_utils.distill_data import build_distill_dataset, build_distill_data_collator
    from train_utils.distill_precision import configure_distill_precision
    from train_utils.model_level_trainables import build_model_level_trainable_selection
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    if steps not in (8, 256) or round_steps not in (128, 512):
        raise ValueError("Only the 8-step profile and the predefined 256-step R128/R512 arms are supported.")
    if steps == 256 and data_cache is None:
        raise ValueError("Long arms must reuse the profile's exact batch archive.")
    if save_final_codes and (steps != 256 or os.environ.get("FLASH_ATTENTION_DETERMINISTIC") != "1"):
        raise ValueError("Final-code replay requires 256 steps and deterministic FlashAttention.")
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory.")
    cfg, hf_args, args = read_formal_config(formal_launch, output)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Bind exactly one authorized CUDA device before interpreter startup.")
    source = Path(cfg.student_checkpoint_dir).resolve()
    if output.is_relative_to(source):
        raise ValueError("Output may not be inside the protected source checkpoint.")
    formal = json.loads(Path(formal_launch).read_text())
    weight_path = Path(formal["source_weight_stat"]["path"])
    def source_fingerprint():
        stat = weight_path.stat()
        return {"weight_stat": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino},
                "metadata_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                    for name in ("checkpoint_meta.json", "config.json")}}
    fingerprint = source_fingerprint()
    if fingerprint["weight_stat"] != {key: formal["source_weight_stat"][key] for key in ("size", "mtime_ns", "inode")}:
        raise ValueError("The protected source weight stat changed.")
    if fingerprint["metadata_sha256"]["checkpoint_meta.json"] != formal["source_checkpoint_meta_sha256"]:
        raise ValueError("The protected source metadata changed.")
    pins = dict(formal["source_sha256"])
    repo = Path(__file__).resolve().parents[2]
    for relative in ("experiments/qwen3_w25/kl_round_diagnostic.py", "experiments/qwen3_w25/diagnose_kl_rounds.py",
                     "experiments/qwen3_w25/kl_gradient_diagnostic.py", "experiments/qwen3_w25/local_bits.py",
                     "experiments/liftquant_recovery/recovery_runtime.py"):
        path = repo / relative
        pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if save_final_codes:
        helper = repo / "experiments/qwen3_w25/kl_round_export.py"
        pins[str(helper)] = hashlib.sha256(helper.read_bytes()).hexdigest()
    for path, expected in pins.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source pin changed: {path}")
    output.mkdir(parents=True)
    report = {"status": "preparing", "source_checkpoint_id": SOURCE_ID, "source_dir": str(source),
              "source_fingerprint": fingerprint, "worker_pid": os.getpid(),
              "worker_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
              "started_unix_seconds": time.time(),
              "code_sha256": pins, "steps": steps, "round_steps": round_steps, "target": TARGET,
              "mode": "profile" if steps == 8 else "round_arm", "effective_train_mode": "sparse_bit",
              "formal_reference": str(Path(formal_launch).resolve()),
              "configuration": {"batch_size": 4, "max_length": 1024, "canonical_max_steps": 2000,
                  "loss": asdict(cfg.loss), "data": asdict(cfg.data), "active_ratio": 1.0, "bit_optimizer": "adam",
                  "bit_lr": 4e-5, "bit_weight_decay": 0.0, "adam_eps": 1e-8, "seed": SEED,
                  "all_decoder_storage": "FP32 frozen", "compute_dtype": "BF16", "gradient_clipping": False},
              "limitations": ["Compares production round reset including proxy, moments, and affine permutation; not moments alone.",
                  "Existing early termination after ten stable post-flip steps remains enabled.",
                  "An eight-step profile can validly have zero hard flips; it cannot establish long-run quality.",
                  "The frozen hash starts after FP32 parameter promotion; it does not compare against original BF16 storage.",
                  "No downstream accuracy inference is made from full-model heldout KL."], "steps_observed": [], "heldout": {},
              "round_reset_events": []}
    path = output / "diagnostic.json"
    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    save()
    started, trainer, manager = time.perf_counter(), None, None
    device = torch.device("cuda:0")
    try:
        configure_e2e_determinism(bool(args.full_determinism))
        set_e2e_seed(SEED)
        model, meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True)
        if meta["checkpoint_id"] != SOURCE_ID:
            raise ValueError("Unexpected source checkpoint identity.")
        modules = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, VAELinear)]
        if len(modules) != 252:
            raise ValueError("Expected 252 VAELinear modules.")
        _enable_internal_decode_runtime(modules, decoder_enabled=True, sparse_enabled=True, vae_decoder_checkpoint=True)
        selection = build_model_level_trainable_selection(model, aux=cfg.aux, compressed_modules=modules,
            rank=cfg.lora.rank, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout, train_decoder=True, train_lora=False,
            decoder_execution_mode="decoder_sparse_bit", freeze=True, fp32_components=cfg.opt.distill_fp32_components)
        model = selection.peft_model or model
        configure_distill_precision(selection, components=cfg.opt.distill_fp32_components, training_args=args)
        model.requires_grad_(False)
        for _, layer in modules:
            layer.disable_trainable_decode()
        target = model.get_submodule(TARGET)
        enable_vae_linear_by_execution_plan(target, mode="sparse_bit")
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(args.gradient_checkpointing_kwargs))
        model.to(device)
        # The GPU plan must own the changing packed bytes; no-grad heldout clears derived views.
        target._build_parallel_stage_decode_plan()
        if target._parallel_stage_grouped_vq_packed.device != device:
            raise RuntimeError("The live target packed plan must reside on the training GPU.")
        for name, layer in modules:
            if name != TARGET:
                prime_packed_cache(layer)
        cached = [(name, layer, layer._cached_weight.data_ptr()) for name, layer in modules if name != TARGET]
        if any(layer.trainable_decode or not layer.cache_decoded_weight or layer._cached_weight.dtype != torch.bfloat16 for _, layer, _ in cached):
            raise RuntimeError("Frozen native BF16 caches are not usable by forward.")
        report["frozen_cache"] = {"modules": len(cached), "bytes": sum(layer._cached_weight.numel()*2 for _, layer, _ in cached),
                                  "decode": "existing prime_packed_cache, native packed-u8 BF16"}
        tokenizer = _build_v6_tokenizer(str(meta["base_model_path"]), hf_args)
        _sync_model_padding_config(model, tokenizer)
        banks = _code_banks(target)
        excluded = {f"{TARGET}.{name}" for name, _ in banks}
        frozen_before = _digest(model.state_dict(), excluded)
        schema_before = {name: (tuple(value.shape), str(value.dtype)) for name, value in model.state_dict().items()}
        original_codes = {stage: target.get_stage_part_vq_storage(stage, 0).detach().cpu().clone() for stage in range(2)}
        manager = SparseBitTuningManager(root_model=model, targets=[(TARGET, target)], target_devices={TARGET: device},
            training_seed=SEED, streaming=False, calibration_dtype=torch.bfloat16,
            config=SparseBitTuningConfig(enabled=True, active_ratio=1.0, optimizer="adam", bit_lr=4e-5,
                weight_decay=0.0, round_steps=round_steps, proxy_coordinates="decoder_sensitivity"))
        bundle = build_distill_dataset(cfg.data, tokenizer)
        if not bundle.is_iterable:
            raise ValueError("Expected the canonical iterable mixture.")
        args.output_dir = str(output / "unused_trainer_output")
        args.dataloader_num_workers = default_dataloader_num_workers()
        args.dataloader_pin_memory = True
        args.group_by_length = False
        args.accelerator_config = AcceleratorConfig(dispatch_batches=False, split_batches=False)
        args.remove_unused_columns = False
        args.report_to = []
        teacher, identity = _load_teacher(cfg, hf_args, args, model, base_model_path=str(meta["base_model_path"]),
                                         dp_device=device, log=logging.getLogger(__name__))
        trainer = VAEDecoderE2ETrainer(model=model, args=args, tokenizer=tokenizer, train_dataset=bundle.train_dataset,
            eval_dataset=None, data_collator=build_distill_data_collator(tokenizer, model_max_length=1024, dynamic_padding=True),
            loss_config=cfg.loss, teacher_model=teacher, teacher_output_offload="cpu", teacher_model_offload="none",
            teacher_output_pin_memory=True, teacher_output_chunk_tokens=8, sparse_bit_manager=manager, callbacks=[])
        trainer.create_optimizer()
        trainer.create_scheduler(num_training_steps=2000)
        if trainer._sparse_bit_main_optimizer is not None or trainer._sparse_bit_main_parameters:
            raise RuntimeError("Pure-bit observation must not create a continuous optimizer.")
        model_for_step = trainer.accelerator.prepare_model(model)
        if {id(p) for p in model.parameters() if p.requires_grad} != manager.score_module.bit_parameter_ids():
            raise RuntimeError("Only the target score tensors may be trainable.")
        fixed_indices = {s.canonical_key: [manager.sampler_states[s.canonical_key].logical_index(i) for i in range(OBSERVED_PER_STAGE)] for s in manager.bank_specs}
        finish_round = manager._finish_nonstreaming_round
        def observe_existing_round_reset():
            event = {"global_step": len(report["steps_observed"]) + 1,
                     "completed_round": manager.global_bit_round, "completed_round_step": manager.bit_round_step,
                     "after_update_before_reset": _sample_state(manager, fixed_indices)}
            finish_round()
            event["after_existing_reset"] = _sample_state(manager, fixed_indices)
            report["round_reset_events"].append(event)
        # Observe this isolated instance; the original production transition still executes exactly once.
        manager._finish_nonstreaming_round = observe_existing_round_reset
        report["registration"] = {"total_bits": sum(s.n_bits for s in manager.bank_specs), "active_scores": sum(s.n_active for s in manager.bank_specs),
            "banks": [{"key": s.canonical_key, "total_bits": s.n_bits, "active_scores": s.n_active} for s in manager.bank_specs],
            "observed_logical_bits_per_stage": OBSERVED_PER_STAGE, "stable_steps": manager.stable_steps,
            "initial_sample": _sample_state(manager, fixed_indices)}
        if data_cache is None:
            iterator = iter(trainer.get_train_dataloader())
            batches = [{key: value.detach().cpu().clone() for key, value in next(iterator).items()} for _ in range(TRAIN_BATCHES + HELDOUT_BATCHES)]
            archive = {"format": "qwen3_formal_kl_batches_v1", "data": asdict(cfg.data),
                       "batches": batches, "manifests": [batch_manifest(b, i, sample_offset=i*4) for i, b in enumerate(batches)]}
            del iterator
            cache = output / "batches.pt"
            torch.save(archive, cache)
        else:
            cache = Path(data_cache).resolve()
            archive = torch.load(cache, map_location="cpu", weights_only=True)
            if archive["data"] != asdict(cfg.data):
                raise ValueError("Prepared batches use a different canonical data configuration.")
            batches = archive["batches"]
        report["data_archive"] = {"path": str(cache), "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(), **validate_batch_archive(archive)}
        if max(b["input_ids"].shape[1] for b in batches[:8]) != 1024:
            raise ValueError("The eight-step profile must include genuine B4/max1024 batches.")
        report["runtime"] = {"teacher_identity": identity, "dataloader_workers": args.dataloader_num_workers,
            "mixed_precision": trainer.accelerator.mixed_precision, "decoder_parameter_tensors": len(selection.decoder_parameters),
            "decoder_elements": sum(p.numel() for p in selection.decoder_parameters.values()),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32, "full_determinism": args.full_determinism}
        report["setup_seconds"] = time.perf_counter() - started
        report["setup_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        if steps == 256:
            report["heldout"]["0"] = _heldout(trainer, model_for_step, batches[TRAIN_BATCHES:])
        report["status"] = "running"
        save()
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(steps):
            before = _sample_state(manager, fixed_indices)
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            loss = trainer.training_step(model_for_step, batches[index])
            torch.cuda.synchronize(device)
            backward_seconds = time.perf_counter() - tick
            manager.bit_optimizer.validate_gradients()
            gradients = {}
            for spec in manager.bank_specs:
                state = manager.sampler_states[spec.canonical_key]
                ordinals = logical_ordinals(state, fixed_indices[spec.canonical_key])
                indices = torch.tensor(ordinals, device=device, dtype=torch.long) + spec.score_start
                gradient = manager.score_module.score_chunks[spec.chunk_id].grad
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("A full-coverage score gradient is nonfinite.")
                gradients[spec.canonical_key] = gradient[indices].detach().float().cpu().tolist()
            if not torch.isfinite(loss) or any(p.grad is not None for p in selection.decoder_parameters.values()):
                raise RuntimeError("Nonfinite loss or an unfrozen decoder gradient.")
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
            torch.cuda.synchronize(device)
            optimizer_seconds = time.perf_counter() - tick
            elapsed = backward_seconds + optimizer_seconds
            after = _sample_state(manager, fixed_indices)
            row = {"step": index + 1, "loss": float(loss), "training_step_seconds": elapsed,
                   "forward_backward_seconds": backward_seconds, "optimizer_seconds": optimizer_seconds,
                   "batch_manifest_index": index,
                   "telemetry": dict(trainer._last_sparse_bit_telemetry)}
            if steps == 8 or index + 1 in (1, 2, 8, 127, 128, 129, 256) or row["telemetry"]["bit/round_ended"]:
                row.update(before=before, gradient=gradients, after=after)
            report["steps_observed"].append(row)
            report["elapsed_seconds"] = time.perf_counter() - started
            print(json.dumps({"step": index+1, "loss": float(loss), "seconds": elapsed, "telemetry": row["telemetry"]}), flush=True)
            save()
            if steps == 256 and index + 1 in (128, 256):
                # final_commit persists current bits without resetting scores, moments, or sampler.
                manager.final_commit()
                report["heldout"][str(index+1)] = _heldout(trainer, model_for_step, batches[TRAIN_BATCHES:])
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        snapshot = manager.checkpoint_packed_snapshot()
        manager.final_commit()
        report["final_bits"] = {}
        for spec in manager.bank_specs:
            stored = target.get_stage_part_vq_storage(spec.stage_idx, spec.part_idx).detach().cpu()
            if not torch.equal(stored, snapshot[spec.canonical_key]):
                raise RuntimeError("Existing final_commit did not preserve the live hard code state.")
            report["final_bits"][spec.canonical_key] = {"xor_vs_source": _changed_bits(original_codes[spec.stage_idx], stored, spec.logical_shape)}
        report["moments_allocated"] = all(s.exp_avg is not None and s.exp_avg_sq is not None for s in manager.bit_optimizer._state.values())
        report["cumulative_flip_events_all_steps"] = sum(row["telemetry"]["bit/step_flip_count"] for row in report["steps_observed"])
        report["moment_update_observed"] = any(any(value != 0 for value in bank["exp_avg"])
            for bank in report["steps_observed"][0]["after"].values())
        report["score_update_observed"] = any(before_bank["scores"] != report["steps_observed"][0]["after"][key]["scores"]
            for key, before_bank in report["steps_observed"][0]["before"].items())
        if not report["moments_allocated"] or not report["moment_update_observed"] or not report["score_update_observed"]:
            raise RuntimeError("The profile did not demonstrate genuine score and Adam-state updates.")
        report["frozen_cache"]["same_pointers_after_training"] = all(layer._cached_weight is not None and layer._cached_weight.data_ptr()==pointer for _, layer, pointer in cached)
        if not report["frozen_cache"]["same_pointers_after_training"]:
            raise RuntimeError("A frozen native weight cache was discarded during training.")
        manager.detach_runtime()
        frozen_after = _digest(model.state_dict(), excluded)
        report["frozen_state"] = {"before": frozen_before, "after": frozen_after, "unchanged": frozen_before == frozen_after,
            "excluded_persistent_keys": sorted(excluded), "reference": "after FP32 decoder promotion, before score registration"}
        if frozen_before != frozen_after or schema_before != {name: (tuple(value.shape), str(value.dtype)) for name, value in model.state_dict().items()}:
            raise RuntimeError("Frozen model bytes or persistent tensor schema changed.")
        report["timing"] = timing_summary(report["steps_observed"])
        if source_fingerprint() != fingerprint:
            raise RuntimeError("Protected source fingerprint changed during the diagnostic.")
        if any(hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected for name, expected in pins.items()):
            raise RuntimeError("A pinned source file changed during the diagnostic.")
        if hashlib.sha256(cache.read_bytes()).hexdigest() != report["data_archive"]["sha256"]:
            raise RuntimeError("The immutable batch archive changed during the diagnostic.")
        report["source_pins_data_unchanged"] = True
        if save_final_codes:
            from experiments.qwen3_w25.kl_round_export import save_code_payload
            report["final_codes"] = save_code_payload(
                output / "final_codes.pt",
                {f"{TARGET}.{name}": target.state_dict()[name] for name, _ in banks}, report)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if trainer is not None:
            trainer._release_active_teacher_targets()
        if manager is not None:
            manager.detach_runtime()
        report["elapsed_seconds"] = time.perf_counter() - started
        save()
