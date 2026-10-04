"""Isolated L0-down decoder-only versus decoder/bit joint KL bridge."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

from experiments.qwen3_w25.kl_gradient_diagnostic import SOURCE_ID, read_formal_config
from experiments.qwen3_w25.kl_round_diagnostic import TARGET, _heldout, _sample_state, validate_batch_archive

ARCHIVE_SHA = "1b481a4748c2735be951b45db9998fe8bec8e43f69410965cf720c5b3a6c2d53"
CODE_NAMES = ("vq_weight", "vq_weight_s1")


def snapshot(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


@torch.no_grad()
def restore_phase(layer, decoder, codes):
    """Copy persistent values without replacing Parameters, then rebuild actual grouped banks."""
    from train_utils.checkpoint_v6 import refresh_vae_linear_runtime_after_state_load

    if getattr(layer, "_sparse_bit_binding", None) is not None:
        raise ValueError("Detach the live bit manager before crossing terminal states.")
    state = layer.state_dict()
    decoder_names = {name for name in state if name.startswith("_parallel_stage_decoder.")}
    if set(decoder) != decoder_names or len(decoder_names) != 6 or set(codes) != set(CODE_NAMES):
        raise ValueError("Expected six packed decoder tensors and exactly two persistent code banks.")
    selected = {**decoder, **codes}
    for name, value in selected.items():
        if state[name].shape != value.shape or state[name].dtype != value.dtype:
            raise ValueError(f"Crossed tensor shape/dtype mismatch: {name}")
    for name, value in selected.items():
        state[name].copy_(value)
    refresh_vae_linear_runtime_after_state_load(layer)
    for name, value in selected.items():
        if not torch.equal(layer.state_dict()[name].detach().cpu(), value.detach().cpu()):
            raise RuntimeError(f"Persistent phase restore failed: {name}")
    grouped = torch.cat([layer.get_stage_part_vq_storage(stage, part)
                         for stage, part in layer._parallel_stage_layout], dim=1).contiguous()
    if not torch.equal(layer._parallel_stage_grouped_vq_packed, grouped) or layer._cached_weight is not None:
        raise RuntimeError("The rebuilt grouped packed state/cache does not represent the requested phase.")


def interaction_summary(phases):
    values = {name: float(phases[name]["loss_denominator_weighted"])
              for name in ("D0C0", "D1C0", "D0C1", "D1C1")}
    return {"weighted_losses": values,
            "code_benefit_at_trained_decoder": values["D1C0"] - values["D1C1"],
            "decoder_benefit_at_trained_codes": values["D0C1"] - values["D1C1"],
            "interaction_loss": values["D1C1"] - values["D1C0"] - values["D0C1"] + values["D0C0"],
            "meaning": "Positive benefit means lower KL. Negative interaction means better than additive loss reductions at these fixed states; not a causal training attribution."}


def run_joint_diagnostic(formal_launch, output_dir, data_cache, pure_bit_reference, *, joint, steps=256):
    import logging
    from transformers.trainer_pt_utils import AcceleratorConfig
    from compressed_e2e_fintuning.runtime_v6 import _build_v6_tokenizer, _load_teacher, _sync_model_padding_config
    from compressed_e2e_fintuning.runtime_v6_pipeline import _enable_internal_decode_runtime
    from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from experiments.liftquant_recovery.recovery_runtime import prime_packed_cache
    from experiments.qwen3_w25.local_bits import _changed_bits, _digest
    from litebsq.vae_linear import VAELinear
    from sparse_bit_tuning.config import SparseBitTuningConfig
    from sparse_bit_tuning.manager import SparseBitTuningManager
    from train_utils.distill_precision import configure_distill_precision
    from train_utils.model_level_optimizer import ModelLevelOptimizerLRConfig, attach_model_level_optimizer_contract
    from train_utils.model_level_trainables import build_model_level_trainable_selection
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    if steps not in (8, 256):
        raise ValueError("Only an eight-step resource profile or the paired 256-step bridge is supported.")
    for name, value in {"FLASH_ATTENTION_DETERMINISTIC": "1", "PYTHONHASHSEED": "0",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}.items():
        if os.environ.get(name) != value:
            raise ValueError(f"Set {name}={value} before interpreter startup.")
    output, archive_path, reference_path = [Path(p).resolve() for p in (output_dir, data_cache, pure_bit_reference)]
    if output.exists():
        raise ValueError("Use a fresh output directory.")
    cfg, hf_args, args = read_formal_config(formal_launch, output)
    if (cfg.opt.resolved_decoder_lr() != 1e-5 or args.max_grad_norm != 1.5 or args.weight_decay != 0
            or args.get_warmup_steps(2000) != 20 or args.full_determinism or args.tf32
            or str(getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type)) != "cosine"):
        raise ValueError("The formal decoder optimizer/scheduler or precision changed.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Bind exactly one authorized GPU before startup.")
    source = Path(cfg.student_checkpoint_dir).resolve()
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the protected source.")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    formal = json.loads(Path(formal_launch).read_text())
    repo = Path(__file__).resolve().parents[2]
    pins = {str((repo / path).resolve()): digest for path, digest in formal["source_sha256"].items()}
    for name in ("kl_joint_diagnostic.py", "diagnose_kl_joint.py", "kl_round_diagnostic.py", "kl_gradient_diagnostic.py", "local_bits.py"):
        path = Path(__file__).parent / name
        pins[str(path)] = sha(path)
    cache_helper = repo / "experiments/liftquant_recovery/recovery_runtime.py"
    pins[str(cache_helper)] = sha(cache_helper)
    reference_hash = sha(reference_path)
    metadata = {str(source / "checkpoint_meta.json"): formal["source_checkpoint_meta_sha256"],
                str(source / "config.json"): sha(source / "config.json"), **formal["hf_source"]["metadata_sha256"]}
    input_stats = [formal["source_weight_stat"], *formal["input_data_stat"], *formal["hf_source"]["weight_files"]]
    def check_inputs():
        for path, digest in {**pins, **metadata, str(archive_path): ARCHIVE_SHA, str(reference_path): reference_hash}.items():
            if sha(path) != digest:
                raise ValueError(f"Pinned input changed: {path}")
        for item in input_stats:
            stat = Path(item["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != tuple(item[k] for k in ("size", "mtime_ns", "inode")):
                raise ValueError(f"Protected input stat changed: {item['path']}")
    check_inputs()
    reference = json.loads(reference_path.read_text())
    if (reference["status"] != "complete" or reference["source_checkpoint_id"] != SOURCE_ID
            or reference["round_steps"] != 512 or reference["steps"] != 256
            or reference["data_archive"]["sha256"] != ARCHIVE_SHA):
        raise ValueError("Expected the completed same-source pure-bit R512 reference.")
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    if archive["data"] != asdict(cfg.data):
        raise ValueError("Canonical archive data recipe changed.")
    archive_info = validate_batch_archive(archive)
    batches = archive["batches"]
    output.mkdir(parents=True)
    report = {"status": "preparing", "source_checkpoint_id": SOURCE_ID, "source_dir": str(source),
              "worker_pid": os.getpid(), "worker_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
              "started_unix_seconds": time.time(), "steps": steps, "joint": joint, "target": TARGET,
              "source_sha256": pins, "input_metadata_sha256": metadata, "input_stats": input_stats,
              "pure_bit_reference": {"path": str(reference_path), "sha256": reference_hash},
              "configuration": {"batch_size": 4, "max_length": 1024, "seed": 0, "data": asdict(cfg.data),
                  "loss": asdict(cfg.loss), "decoder_lr": 1e-5, "decoder_weight_decay": 0., "decoder_clip": 1.5,
                  "canonical_scheduler_steps": 2000, "warmup_steps": 20, "bit_lr": 4e-5 if joint else None,
                  "bit_optimizer": "adam" if joint else None, "active_ratio": 1. if joint else 0.,
                  "round_steps": 512 if joint else None, "all_decoder_storage": "FP32", "compute_dtype": "BF16"},
              "data_archive": {"path": str(archive_path), "sha256": ARCHIVE_SHA, **archive_info},
              "limitations": ["Only L0down decoder and optionally its two banks train; not all252 joint training.",
                  "Four-way crossing measures fixed-state FP32-decoder/BF16-compute KL, not causal optimizer contributions or deployment accuracy.",
                  "Eight fixed heldout batches have disjoint observed token/label hashes; raw-document independence is not certified.",
                  "The optional eight-step profile measures updates/resources and does not run heldout/crossed phases."],
              "steps_observed": [], "heldout": {}, "round_reset_events": []}
    report_path = output / "diagnostic.json"
    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    started, trainer, manager = time.perf_counter(), None, None
    device = torch.device("cuda:0")
    save()
    try:
        configure_e2e_determinism(False)
        set_e2e_seed(0)
        model, meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True)
        if meta["checkpoint_id"] != SOURCE_ID:
            raise ValueError("Unexpected source identity.")
        modules = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, VAELinear)]
        if len(modules) != 252:
            raise ValueError("Expected252 VAELinear modules.")
        _enable_internal_decode_runtime(modules, decoder_enabled=True, sparse_enabled=True, vae_decoder_checkpoint=True)
        promotion = build_model_level_trainable_selection(model, aux=cfg.aux, compressed_modules=modules,
            rank=0, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout, train_decoder=True, train_lora=False,
            decoder_execution_mode="decoder_sparse_bit", freeze=True, fp32_components=("decoder",))
        configure_distill_precision(promotion, components=("decoder",), training_args=args)
        model.requires_grad_(False)
        for _, layer in modules:
            layer.disable_trainable_decode()
        target = model.get_submodule(TARGET)
        selected = [(TARGET, target)]
        mode = "decoder_sparse_bit" if joint else "trainable_decoder"
        selection = build_model_level_trainable_selection(model, aux=cfg.aux, compressed_modules=selected,
            rank=0, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout, train_decoder=True, train_lora=False,
            decoder_execution_mode=mode, freeze=True, fp32_components=("decoder",))
        configure_distill_precision(selection, components=("decoder",), training_args=args)
        parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        decoder_names = {TARGET + "." + name for name in target.state_dict() if name.startswith("_parallel_stage_decoder.")}
        code_names = {TARGET + "." + name for name in CODE_NAMES}
        if len(parameters) != 6 or set(parameters) != decoder_names or any(p.dtype != torch.float32 for p in parameters.values()):
            raise ValueError("Expected exactly six selected packed decoder leaves.")
        initial_decoder = snapshot({name: value for name, value in target.state_dict().items() if name.startswith("_parallel_stage_decoder.")})
        original_codes = snapshot({name: target.state_dict()[name] for name in CODE_NAMES})
        excluded = decoder_names | (code_names if joint else set())
        frozen_before = _digest(model.state_dict(), excluded)
        if _digest(model.state_dict(), code_names) != reference["frozen_state"]["before"]:
            raise ValueError("Promoted initialization differs from the pure-bit source state.")
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(args.gradient_checkpointing_kwargs))
        model.to(device)
        target._build_parallel_stage_decode_plan()
        for name, layer in modules:
            if name != TARGET:
                prime_packed_cache(layer)
        cached = [(layer, layer._cached_weight.data_ptr()) for name, layer in modules if name != TARGET]
        if any(layer.trainable_decode or not layer.cache_decoded_weight or layer._cached_weight.dtype != torch.bfloat16 for layer, _ in cached):
            raise RuntimeError("The251 frozen native BF16 caches are not active.")
        if joint:
            manager = SparseBitTuningManager(root_model=model, targets=selected, target_devices={TARGET: device},
                training_seed=0, streaming=False, calibration_dtype=torch.bfloat16,
                config=SparseBitTuningConfig(enabled=True, active_ratio=1., optimizer="adam", bit_lr=4e-5,
                                            weight_decay=0., round_steps=512, proxy_coordinates="decoder_sensitivity"))
        tokenizer = _build_v6_tokenizer(str(meta["base_model_path"]), hf_args)
        _sync_model_padding_config(model, tokenizer)
        args.output_dir = str(output / "unused_trainer_output")
        args.accelerator_config = AcceleratorConfig(dispatch_batches=False, split_batches=False)
        args.remove_unused_columns, args.report_to = False, []
        teacher, identity = _load_teacher(cfg, hf_args, args, model, base_model_path=str(meta["base_model_path"]),
                                         dp_device=device, log=logging.getLogger(__name__))
        trainer = VAEDecoderE2ETrainer(model=model, args=args, tokenizer=tokenizer, train_dataset=None,
            eval_dataset=None, loss_config=cfg.loss, teacher_model=teacher, teacher_output_offload="cpu",
            teacher_model_offload="none", teacher_output_pin_memory=True, teacher_output_chunk_tokens=8,
            sparse_bit_manager=manager, callbacks=[])
        attach_model_level_optimizer_contract(trainer, selection=selection,
            lr_config=ModelLevelOptimizerLRConfig(learning_rate=1e-5, weight_decay=0., decoder_lr=1e-5))
        trainer.create_optimizer()
        trainer.create_scheduler(num_training_steps=2000)
        model_for_step = trainer.accelerator.prepare_model(model)
        main_optimizer = trainer._sparse_bit_main_optimizer if joint else trainer.optimizer
        if {id(p) for group in main_optimizer.param_groups for p in group["params"]} != {id(p) for p in parameters.values()}:
            raise RuntimeError("Continuous optimizer inventory differs from the six decoder leaves.")
        expected_trainable = {id(p) for p in parameters.values()} | (manager.score_module.bit_parameter_ids() if joint else set())
        if {id(p) for p in model.parameters() if p.requires_grad} != expected_trainable:
            raise RuntimeError("Unexpected trainable component.")
        if joint:
            fixed = {s.canonical_key: [manager.sampler_states[s.canonical_key].logical_index(i) for i in range(128)] for s in manager.bank_specs}
            report["bit_registration"] = {"active_scores": sum(s.n_active for s in manager.bank_specs), "initial_sample": _sample_state(manager, fixed)}
            if report["bit_registration"]["active_scores"] != 147849216:
                raise ValueError("Unexpected full-bit inventory.")
            if report["bit_registration"]["initial_sample"] != reference["registration"]["initial_sample"]:
                raise ValueError("Joint score initialization differs from the matched pure-bit reference.")
            finish = manager._finish_nonstreaming_round
            def observe_reset():
                event = {"step": len(report["steps_observed"]) + 1, "before": _sample_state(manager, fixed)}
                finish()
                event["after"] = _sample_state(manager, fixed)
                report["round_reset_events"].append(event)
            manager._finish_nonstreaming_round = observe_reset
        report["runtime"] = {"teacher_identity": identity, "mixed_precision": trainer.accelerator.mixed_precision,
            "promoted_decoder_tensors": len(promotion.decoder_parameters), "trainable_decoder_tensors": len(parameters),
            "trainable_decoder_elements": sum(p.numel() for p in parameters.values()), "frozen_native_cache_modules": len(cached),
            "frozen_tensor_count": len(model.state_dict()) - len(excluded) - (len(manager.score_module.score_chunks) if joint else 0),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32, "full_determinism": args.full_determinism}
        if steps == 256:
            report["heldout"]["0"] = _heldout(trainer, model_for_step, batches[256:])
            report["initial_reference"] = {"heldout_exact_equal": report["heldout"]["0"] == reference["heldout"]["0"],
                "per_batch_loss_delta": [a["loss"] - b["loss"] for a, b in zip(report["heldout"]["0"]["batches"], reference["heldout"]["0"]["batches"])]}
        report["setup_seconds"], report["status"] = time.perf_counter() - started, "running"
        save()
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(steps):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            loss = trainer.training_step(model_for_step, batches[index])
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters.values()):
                raise FloatingPointError("Nonfinite loss/decoder gradient or disconnected decoder.")
            if joint:
                manager.bit_optimizer.validate_gradients()
                if any(not torch.isfinite(p.grad).all() for p in manager.score_module.score_chunks):
                    raise FloatingPointError("Nonfinite score gradient.")
            # Explicit decoder inventory keeps the production bit gradients out of clipping.
            norm = trainer.accelerator.clip_grad_norm_(parameters.values(), 1.5)
            lrs = [group["lr"] for group in main_optimizer.param_groups]
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
            trainer.state.global_step = index + 1
            torch.cuda.synchronize(device)
            row = {"step": index + 1, "batch_manifest_index": index, "loss": float(loss),
                   "seconds": time.perf_counter() - tick, "decoder_gradient_norm_before_clipping": float(norm),
                   "decoder_learning_rates_used": lrs}
            if joint:
                row["telemetry"] = dict(trainer._last_sparse_bit_telemetry)
                if index + 1 in (1, 2, 8, 127, 128, 129, 256):
                    row["score_sample_after"] = _sample_state(manager, fixed)
            report["steps_observed"].append(row)
            print(json.dumps(row), flush=True)
            save()
            if steps == 256 and index + 1 in (128, 256):
                if joint:
                    manager.final_commit()
                report["heldout"][str(index + 1)] = _heldout(trainer, model_for_step, batches[256:])
        report["first_loss_delta_vs_pure_bit"] = report["steps_observed"][0]["loss"] - reference["steps_observed"][0]["loss"]
        if joint:
            packed = manager.checkpoint_packed_snapshot()
            manager.final_commit()
            for spec in manager.bank_specs:
                if not torch.equal(target.get_stage_part_vq_storage(spec.stage_idx, spec.part_idx).detach().cpu(), packed[spec.canonical_key]):
                    raise RuntimeError("Final persistent codes differ from live grouped codes.")
            manager.detach_runtime()
        final_decoder = snapshot({name: value for name, value in target.state_dict().items() if name.startswith("_parallel_stage_decoder.")})
        final_codes = snapshot({name: target.state_dict()[name] for name in CODE_NAMES})
        changed = {name: int((final_decoder[name] != value).sum()) for name, value in initial_decoder.items()}
        if not any(changed.values()):
            raise RuntimeError("No genuine decoder update.")
        report["decoder_changed_elements"] = changed
        report["decoder_moment_dtypes"] = {name: {k: str(main_optimizer.state[p][k].dtype) for k in ("exp_avg", "exp_avg_sq")} for name, p in parameters.items()}
        if any(dtype != "torch.float32" for state in report["decoder_moment_dtypes"].values() for dtype in state.values()):
            raise RuntimeError("Decoder Adam moments did not retain FP32 storage.")
        if joint:
            report["cumulative_flip_events_all_steps"] = sum(row["telemetry"]["bit/step_flip_count"] for row in report["steps_observed"])
        report["net_xor_by_stage"] = [_changed_bits(original_codes[name], final_codes[name], tuple(target.get_stage_part_vq_spec(stage, 0)["logical_shape"])) for stage, name in enumerate(CODE_NAMES)]
        if not joint and any(report["net_xor_by_stage"]):
            raise RuntimeError("Decoder-only arm modified codes.")
        if steps == 256:
            phases = {}
            try:
                for name, dec, codes in (("D0C0", initial_decoder, original_codes), ("D1C0", final_decoder, original_codes),
                                         ("D0C1", initial_decoder, final_codes), ("D1C1", final_decoder, final_codes)):
                    restore_phase(target, dec, codes)
                    phases[name] = _heldout(trainer, model_for_step, batches[256:])
            finally:
                restore_phase(target, final_decoder, final_codes)
            report["crossed_heldout"] = phases
            report["interaction"] = interaction_summary(phases)
            report["phase_equivalence"] = {"D0C0_equals_initial": phases["D0C0"] == report["heldout"]["0"],
                "D1C1_equals_terminal_before_detach": phases["D1C1"] == report["heldout"]["256"],
                "scope": "actual native repeated heldout values; discrepancies remain explicit, not silently discarded"}
        frozen_after = _digest(model.state_dict(), excluded)
        report["frozen_state"] = {"before": frozen_before, "after": frozen_after, "unchanged": frozen_before == frozen_after,
                                   "excluded_keys": sorted(excluded), "reference": "after all252 FP32 decoder promotion; original persistent codes"}
        report["frozen_cache_pointers_unchanged"] = all(layer._cached_weight is not None and layer._cached_weight.data_ptr() == pointer for layer, pointer in cached)
        if frozen_before != frozen_after or not report["frozen_cache_pointers_unchanged"]:
            raise RuntimeError("Frozen persistent state or native caches changed.")
        payload = {"format": "qwen3_kl_joint_bridge_v1", "source_checkpoint_id": SOURCE_ID, "joint": joint,
                   "steps": steps, "target": TARGET, "decoder_fp32": final_decoder, "codes": final_codes}
        payload_path = output / "selected_state.pt"
        torch.save(payload, payload_path)
        loaded = torch.load(payload_path, map_location="cpu", weights_only=True)
        if any(not torch.equal(value, loaded[part][name]) for part in ("decoder_fp32", "codes") for name, value in payload[part].items()):
            raise RuntimeError("Selected terminal payload reload differs.")
        report["selected_payload"] = {"path": str(payload_path), "sha256": sha(payload_path), "reloaded_exact": True, "is_full_checkpoint": False}
        report["peak_cuda_allocated_bytes"], report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
        report["timing"] = {"steady_median_seconds": statistics.median(row["seconds"] for row in report["steps_observed"][1:]),
                           "steady_max_seconds": max(row["seconds"] for row in report["steps_observed"][1:])}
        check_inputs()
        report["source_pins_data_unchanged"], report["status"] = True, "complete"
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
