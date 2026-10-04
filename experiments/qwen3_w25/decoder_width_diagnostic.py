"""Isolated, paired H128/H256 decoder-only optimization under the existing KL loss.

Only L0 gate/down decoders may change. The canonical 256+8 batch archive is an
input, never regenerated here. Eight steps measure the new backward path and
resources; 256 steps measure a local capacity proxy, not downstream accuracy.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch

from experiments.qwen3_w25.kl_gradient_diagnostic import SOURCE_ID, TARGETS, batch_manifest, read_formal_config
from experiments.qwen3_w25.widen_decoder import widen_packed_decoder_h128_to_h256_

ARCHIVE_SHA256 = "1b481a4748c2735be951b45db9998fe8bec8e43f69410965cf720c5b3a6c2d53"
TRAIN_BATCHES = 256


def summarize_losses(rows):
    if not rows or any(not math.isfinite(r["loss"]) or not math.isfinite(r["denominator"])
                       or r["denominator"] <= 0 for r in rows):
        raise ValueError("Need finite losses and positive effective token denominators.")
    denominator = math.fsum(row["denominator"] for row in rows)
    return {"batches": rows, "equal_batch_loss": statistics.mean(row["loss"] for row in rows),
            "loss_denominator_sum": denominator,
            "loss_denominator_weighted": math.fsum(row["loss"] * row["denominator"] for row in rows) / denominator}


def decoder_snapshot(parameters):
    return {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}


def parameter_changes(before, after):
    if before.keys() != after.keys():
        raise ValueError("Decoder parameter inventory changed.")
    result = {}
    for name, original in before.items():
        value = after[name].detach().cpu()
        if value.shape != original.shape or value.dtype != original.dtype:
            raise ValueError(f"Decoder shape/dtype changed: {name}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite decoder: {name}")
        result[name] = {"elements": value.numel(), "changed_elements": int((value != original).sum()),
                        "max_abs_change": float((value.double() - original.double()).abs().max())}
    return {"parameters": result, "changed_elements": sum(row["changed_elements"] for row in result.values()),
            "changed_parameter_tensors": sum(row["changed_elements"] > 0 for row in result.values())}


@torch.no_grad()
def project_selected_bf16_(parameters):
    """Terminal projection of selected leaves only; caller retains the FP32 payload."""
    for parameter in parameters.values():
        if parameter.dtype != torch.float32:
            raise ValueError("BF16 projection expects FP32 training parameters.")
        parameter.data = parameter.data.to(dtype=torch.bfloat16)
    return decoder_snapshot(parameters)


def width_activity(decoder, *, gradients=False):
    def value(parameter):
        result = parameter.grad if gradients else parameter.detach()
        if result is None:
            raise RuntimeError("Selected decoder has no gradient.")
        return result.detach()
    incoming = value(decoder.linear_in.conv.weight).reshape(2, decoder.hidden_dim, decoder.in_dim, 1)
    outgoing = value(decoder.linear_out.conv.weight).reshape(2, decoder.out_dim, decoder.hidden_dim, 1)
    result = {"hidden_dim": decoder.hidden_dim, "incoming_nonzero": int(torch.count_nonzero(incoming)),
              "outgoing_nonzero": int(torch.count_nonzero(outgoing))}
    if decoder.hidden_dim == 256:
        result.update(new_output_half_nonzero=int(torch.count_nonzero(outgoing[:, :, 128:])),
                      hidden_copy_different_elements=int(torch.count_nonzero(incoming[:, :128] != incoming[:, 128:])),
                      hidden_copy_max_abs_difference=float((incoming[:, :128] - incoming[:, 128:]).abs().max()))
    return result


def _heldout(trainer, model, batches):
    rows = []
    was_training = model.training
    model.eval()
    try:
        for index, batch in enumerate(batches):
            try:
                with torch.no_grad(), trainer.compute_loss_context_manager():
                    loss = trainer.compute_loss(model, trainer._prepare_inputs(batch))
                manifest = batch_manifest(batch, index, sample_offset=(TRAIN_BATCHES + index) * 4)
                rows.append({"archive_batch_index": TRAIN_BATCHES + index, "loss": float(loss),
                             "denominator": manifest["loss_denominator_fp32"]})
            finally:
                trainer._release_active_teacher_targets()
    finally:
        model.train(was_training)
    return summarize_losses(rows)


def run_decoder_width_diagnostic(formal_launch, output_dir, *, hidden_dim, data_cache, steps=8):
    import logging
    from transformers.trainer_pt_utils import AcceleratorConfig
    from compressed_e2e_fintuning.runtime_v6 import _build_v6_tokenizer, _load_teacher, _sync_model_padding_config
    from compressed_e2e_fintuning.runtime_v6_pipeline import _enable_internal_decode_runtime
    from compressed_e2e_fintuning.trainer import VAEDecoderE2ETrainer
    from e2e_common.determinism import configure_e2e_determinism, set_e2e_seed
    from experiments.liftquant_recovery.recovery_runtime import prime_packed_cache
    from experiments.qwen3_w25.expand_decoder import validate_packed_r0_decoder
    from experiments.qwen3_w25.kl_round_diagnostic import validate_batch_archive
    from experiments.qwen3_w25.local_bits import _digest
    from litebsq.vae_linear import VAELinear
    from train_utils.distill_precision import configure_distill_precision
    from train_utils.model_level_optimizer import ModelLevelOptimizerLRConfig, attach_model_level_optimizer_contract
    from train_utils.model_level_trainables import build_model_level_trainable_selection
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    if hidden_dim not in (128, 256) or steps not in (8, 256):
        raise ValueError("Only the matched H128/H256 8-step profile and 256-step arms are supported.")
    if (os.environ.get("FLASH_ATTENTION_DETERMINISTIC") != "1" or os.environ.get("PYTHONHASHSEED") != "0"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise ValueError("Set FA2 determinism, PYTHONHASHSEED=0 and CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup.")
    output, formal_path, archive_path = map(lambda p: Path(p).resolve(), (output_dir, formal_launch, data_cache))
    if output.exists():
        raise ValueError("Use a fresh diagnostic output directory.")
    cfg, hf_args, args = read_formal_config(formal_path, output)
    if (cfg.opt.resolved_decoder_lr() != 1e-5 or args.max_grad_norm != 1.5 or args.weight_decay != 0
            or str(getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type)) != "cosine"
            or args.get_warmup_steps(2000) != 20 or args.full_determinism or args.tf32):
        raise ValueError("The canonical LR, clipping, scheduler or precision changed.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Bind exactly one authorized CUDA device before interpreter startup.")
    formal = json.loads(formal_path.read_text())
    source = Path(cfg.student_checkpoint_dir).resolve()
    if output.is_relative_to(source):
        raise ValueError("Output must not be inside the protected source checkpoint.")
    repo = Path(__file__).resolve().parents[2]
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    pins = {str((repo / path).resolve()): digest for path, digest in formal["source_sha256"].items()}
    for relative in ("decoder_width_diagnostic.py", "diagnose_decoder_width.py", "widen_decoder.py",
                     "expand_decoder.py", "kl_gradient_diagnostic.py", "kl_round_diagnostic.py", "local_bits.py"):
        file = Path(__file__).parent / relative
        pins[str(file)] = sha(file)
    file = repo / "experiments/liftquant_recovery/recovery_runtime.py"
    pins[str(file)] = sha(file)
    input_metadata = {str(source / "checkpoint_meta.json"): formal["source_checkpoint_meta_sha256"],
                      str(source / "config.json"): sha(source / "config.json"), **formal["hf_source"]["metadata_sha256"]}
    input_stats = [formal["source_weight_stat"], *formal["input_data_stat"], *formal["hf_source"]["weight_files"]]
    def check_inputs():
        for path, expected in {**pins, **input_metadata, str(archive_path): ARCHIVE_SHA256}.items():
            if sha(path) != expected:
                raise ValueError(f"Pinned input changed: {path}")
        for item in input_stats:
            stat = Path(item["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != tuple(item[k] for k in ("size", "mtime_ns", "inode")):
                raise ValueError(f"Input stat changed: {item['path']}")
    check_inputs()
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    if archive["data"] != asdict(cfg.data):
        raise ValueError("Canonical data configuration differs from the immutable archive.")
    archive_info = validate_batch_archive(archive)
    batches = archive["batches"]
    if max(batch["input_ids"].shape[1] for batch in batches[:8]) != 1024:
        raise ValueError("The profile must include genuine B4/max1024 inputs.")
    output.mkdir(parents=True)
    report = {"status": "preparing", "source_checkpoint_id": SOURCE_ID, "source_dir": str(source),
        "formal_reference": str(formal_path), "formal_reference_sha256": sha(formal_path),
        "source_sha256": pins, "input_metadata_sha256": input_metadata, "input_stats": input_stats,
        "worker_pid": os.getpid(), "worker_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]),
        "started_unix_seconds": time.time(), "mode": "profile" if steps == 8 else "width_arm",
        "hidden_dim": hidden_dim, "steps": steps, "targets": list(TARGETS), "effective_train_mode": "decoder",
        "configuration": {"batch_size": 4, "max_length": 1024, "gradient_accumulation_steps": 1,
            "canonical_max_steps": 2000, "warmup_steps": 20, "decoder_lr": 1e-5,
            "max_grad_norm": 1.5, "loss": asdict(cfg.loss), "data": asdict(cfg.data), "seed": 0,
            "trainable_parameter_dtype": "FP32", "compute_dtype": "BF16", "code_training": False,
            "residual_blocks": 0, "external_lora": False, "FLASH_ATTENTION_DETERMINISTIC": "1"},
        "data_archive": {"path": str(archive_path), "sha256": ARCHIVE_SHA256, **archive_info},
        "limitations": ["Only L0 gate/down decoders train; the result is not full-model decoder capacity or downstream accuracy.",
            "Eight steps measure backward correctness/resources, not quality; first optimizer step has LR zero.",
            "Both arms use native packed BF16 frozen caches and selected FP32 leaves; initial KL differences remain explicit.",
            "Heldout uses eight fixed batches, disjoint observed token/label hashes, not certified raw-document independence.",
            "Terminal BF16 projection is an in-memory observation, not a full-checkpoint export/reload."],
        "steps_observed": [], "heldout": {}}
    report_path = output / "diagnostic.json"
    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    save()
    started, trainer = time.perf_counter(), None
    device = torch.device("cuda:0")
    try:
        configure_e2e_determinism(False)
        set_e2e_seed(0)
        model, meta, _ = load_v6_model_checkpoint(str(source), map_location="cpu", strict=True)
        if meta["checkpoint_id"] != SOURCE_ID:
            raise ValueError("Unexpected source checkpoint identity.")
        modules = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, VAELinear)]
        selected = [(name, model.get_submodule(name)) for name in TARGETS]
        if len(modules) != 252:
            raise ValueError("Expected 252 real VAELinear modules.")
        excluded = {name for name in model.state_dict() if any(name.startswith(target + "._parallel_stage_decoder.") for target in TARGETS)}
        if len(excluded) != 12:
            raise ValueError("Expected exactly twelve selected packed decoder parameter keys.")
        frozen_before = _digest(model.state_dict(), excluded)
        report["widening"] = {}
        for name, layer in selected:
            validate_packed_r0_decoder(layer)
            if hidden_dim == 256:
                report["widening"][name] = widen_packed_decoder_h128_to_h256_(layer)
        for _, layer in modules:
            layer.disable_trainable_decode()
        _enable_internal_decode_runtime(selected, decoder_enabled=True, sparse_enabled=False, vae_decoder_checkpoint=True)
        selection = build_model_level_trainable_selection(model, aux=cfg.aux, compressed_modules=selected,
            rank=cfg.lora.rank, alpha=cfg.lora.alpha, dropout=cfg.lora.dropout, train_decoder=True, train_lora=False,
            decoder_execution_mode="trainable_decoder", freeze=True, fp32_components=("decoder",))
        model = selection.peft_model or model
        configure_distill_precision(selection, components=("decoder",), training_args=args)
        parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        if set(parameters) != excluded or {id(p) for p in parameters.values()} != {id(p) for p in selection.decoder_parameters.values()}:
            raise RuntimeError("Only the twelve selected decoder leaves may train.")
        initial = decoder_snapshot(parameters)
        if _digest(model.state_dict(), excluded) != frozen_before:
            raise RuntimeError("Decoder initialization changed frozen source bytes.")
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(args.gradient_checkpointing_kwargs))
        model.to(device)
        for name, layer in modules:
            if name not in TARGETS:
                prime_packed_cache(layer)
        cached = [(layer, layer._cached_weight.data_ptr()) for name, layer in modules if name not in TARGETS]
        if any(layer.trainable_decode or not layer.cache_decoded_weight or layer._cached_weight.dtype != torch.bfloat16 for layer, _ in cached):
            raise RuntimeError("Frozen native BF16 cache is not active.")
        tokenizer = _build_v6_tokenizer(str(meta["base_model_path"]), hf_args)
        _sync_model_padding_config(model, tokenizer)
        args.output_dir = str(output / "unused_trainer_output")
        args.accelerator_config = AcceleratorConfig(dispatch_batches=False, split_batches=False)
        args.remove_unused_columns, args.report_to = False, []
        teacher, identity = _load_teacher(cfg, hf_args, args, model, base_model_path=str(meta["base_model_path"]),
                                         dp_device=device, log=logging.getLogger(__name__))
        trainer = VAEDecoderE2ETrainer(model=model, args=args, tokenizer=tokenizer, train_dataset=None,
            eval_dataset=None, loss_config=cfg.loss, teacher_model=teacher, teacher_output_offload="cpu",
            teacher_model_offload="none", teacher_output_pin_memory=True, teacher_output_chunk_tokens=8, callbacks=[])
        attach_model_level_optimizer_contract(trainer, selection=selection,
            lr_config=ModelLevelOptimizerLRConfig(learning_rate=cfg.opt.learning_rate,
                weight_decay=cfg.opt.weight_decay, decoder_lr=cfg.opt.decoder_lr))
        trainer.create_optimizer()
        trainer.create_scheduler(num_training_steps=2000)
        model_for_step = trainer.accelerator.prepare_model(model)
        if {id(p) for group in trainer.optimizer.param_groups for p in group["params"]} != {id(p) for p in parameters.values()}:
            raise RuntimeError("Optimizer contains another component.")
        report["runtime"] = {"teacher_identity": identity, "mixed_precision": trainer.accelerator.mixed_precision,
            "selected_tensors": len(parameters), "selected_elements": sum(p.numel() for p in parameters.values()),
            "frozen_tensor_count": len(model.state_dict()) - len(excluded),
            "code_bank_count": sum(t.dtype == torch.uint8 and (name.endswith("vq_weight") or name.endswith("vq_weight_s1")) for name, t in model.state_dict().items()),
            "frozen_native_cache_modules": len(cached), "frozen_cache_bytes": sum(layer._cached_weight.numel() * 2 for layer, _ in cached),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32, "full_determinism": args.full_determinism,
            "optimizer": type(trainer.optimizer).__name__, "optimizer_defaults": trainer.optimizer.defaults}
        if report["runtime"]["code_bank_count"] != 504 or report["runtime"]["allow_tf32"]:
            raise RuntimeError("Code inventory or TF32 configuration differs from the matched design.")
        report["setup_seconds"] = time.perf_counter() - started
        report["setup_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        if steps == 256:
            report["heldout"]["0"] = _heldout(trainer, model_for_step, batches[TRAIN_BATCHES:])
        set_e2e_seed(0)
        report["status"] = "running"
        save()
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(steps):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            loss = trainer.training_step(model_for_step, batches[index])
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters.values()):
                raise FloatingPointError("Nonfinite loss/gradient or a selected decoder without a gradient.")
            gradient_norm = trainer.accelerator.clip_grad_norm_(parameters.values(), args.max_grad_norm)
            learning_rates = [group["lr"] for group in trainer.optimizer.param_groups]
            activity = {name: width_activity(layer._parallel_stage_decoder, gradients=True) for name, layer in selected} if index in (0, steps - 1) else None
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
            trainer.state.global_step = index + 1
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - tick
            row = {"step": index + 1, "archive_batch_index": index, "loss": float(loss),
                   "training_step_seconds": elapsed, "gradient_norm_before_clipping": float(gradient_norm),
                   "learning_rates_used": learning_rates}
            if activity is not None:
                row["gradient_activity_after_clipping"] = activity
            report["steps_observed"].append(row)
            print(json.dumps(row), flush=True)
            save()
            if steps == 256 and index + 1 in (128, 256):
                report["heldout"][str(index + 1)] = _heldout(trainer, model_for_step, batches[TRAIN_BATCHES:])
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        final_fp32 = decoder_snapshot(parameters)
        report["fp32_update"] = parameter_changes(initial, final_fp32)
        report["final_activity"] = {name: width_activity(layer._parallel_stage_decoder) for name, layer in selected}
        if not report["fp32_update"]["changed_elements"]:
            raise RuntimeError("No genuine decoder parameter update was observed.")
        report["fp32_adam_moments"] = {name: {key: str(trainer.optimizer.state[p][key].dtype) for key in ("exp_avg", "exp_avg_sq")}
                                        for name, p in parameters.items()}
        if any(dtype != "torch.float32" for state in report["fp32_adam_moments"].values() for dtype in state.values()):
            raise RuntimeError("Expected genuine FP32 Adam moments.")
        projected = project_selected_bf16_(parameters)
        if any(not torch.equal(projected[name], final_fp32[name].bfloat16()) for name in projected):
            raise RuntimeError("Terminal selected parameter projection differs from exact BF16 cast.")
        report["bf16_update"] = parameter_changes({name: p.bfloat16() for name, p in initial.items()}, projected)
        for _, layer in selected:
            layer.clear_decoded_weight_cache()
        if steps == 256:
            report["heldout"]["256_bf16_projection"] = _heldout(trainer, model_for_step, batches[TRAIN_BATCHES:])
        frozen_after = _digest(model.state_dict(), excluded)
        report["frozen_state"] = {"before": frozen_before, "after": frozen_after, "unchanged": frozen_before == frozen_after,
                                  "excluded_keys": sorted(excluded), "reference": "original BF16 source before widening or FP32 promotion"}
        report["frozen_cache_pointers_unchanged"] = all(layer._cached_weight is not None and layer._cached_weight.data_ptr() == pointer for layer, pointer in cached)
        if frozen_before != frozen_after or not report["frozen_cache_pointers_unchanged"]:
            raise RuntimeError("Frozen source bytes or native caches changed.")
        if any(p.grad is not None for name, p in model.named_parameters() if name not in excluded):
            raise RuntimeError("A frozen parameter acquired a gradient.")
        payload_path = output / "selected_decoders.pt"
        payload = {"format": "qwen3_decoder_width_probe_v1", "source_checkpoint_id": SOURCE_ID,
                   "hidden_dim": hidden_dim, "targets": list(TARGETS), "steps": steps,
                   "fp32": final_fp32, "bf16": projected}
        torch.save(payload, payload_path)
        reloaded = torch.load(payload_path, map_location="cpu", weights_only=True)
        for precision in ("fp32", "bf16"):
            if reloaded[precision].keys() != payload[precision].keys() or any(not torch.equal(value, reloaded[precision][name]) for name, value in payload[precision].items()):
                raise RuntimeError("Selected decoder payload did not reload exactly.")
        report["selected_payload"] = {"path": str(payload_path), "sha256": sha(payload_path), "reloaded_exact": True,
                                      "is_full_checkpoint": False, "bf16_selected_tensor_bytes": sum(p.numel()*2 for p in projected.values())}
        steady = [row["training_step_seconds"] for row in report["steps_observed"][1:]]
        report["timing"] = {"first_step_seconds": report["steps_observed"][0]["training_step_seconds"],
                            "steady_median_seconds": statistics.median(steady), "steady_max_seconds": max(steady),
                            "one_256_step_arm_median_minutes": statistics.median(steady) * 256 / 60,
                            "scope": "synchronized training_step + finite checks + clipping + Adam + scheduler; excludes setup/heldout/save"}
        check_inputs()
        report["source_pins_data_unchanged"], report["status"] = True, "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if trainer is not None:
            trainer._release_active_teacher_targets()
        report["elapsed_seconds"] = time.perf_counter() - started
        save()
