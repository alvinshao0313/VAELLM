"""Task-9 canonical E2E v6 training pipeline."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers.trainer_callback import TrainerState

from compressed_e2e_fintuning.device_map import apply_boundary_device_map, apply_layer_device_map
from compressed_e2e_fintuning.mid_eval import EvalAfterSaveCallback
from compressed_e2e_fintuning.offload import (
    unwrap_streaming_offload_layers,
    wrap_model_layers_for_streaming_offload,
)
from compressed_e2e_fintuning.runtime_v6 import (
    _assert_final_runtime_clean,
    _barrier,
    _build_eval_args,
    _build_v6_tokenizer,
    _cleanup_runtime,
    _collect_existing_full_low_rank,
    _finalize_decoders,
    _install_run_file_logger,
    _is_main_process,
    _load_teacher,
    _load_v6_student,
    _module_suffixes,
    _place_student_model,
    _resolve_run_output_dir,
    _resolve_train_components,
    _sync_model_padding_config,
)
from compressed_e2e_fintuning.trainer import (
    E2EDistillTokenStatsCallback,
    E2ETrainerLogCallback,
    VAEDecoderE2ETrainer,
    replace_progress_log_callback,
)
from compressed_e2e_fintuning.v6_runtime_state import build_e2e_immutable_resume_contract
from e2e_common.dense_loss import get_output_logits
from e2e_common.full_lora import (
    collect_exact_peft_lora_config,
    finalize_model_level_lora,
    iter_named_peft_lora_layers,
)
from e2e_common.lazy_datasets import default_dataloader_num_workers
from e2e_common.runtime_utils import eval_final_ppl
from litebsq.vae_linear import VAELinear, clear_model_vae_linear_cache
from litebsq.vae_linear_prewarm import NamedVAELinearTarget
from rotation.model_utils import get_layers
from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from train_utils.config.configs import LoRAConfig, validate_lora_against_checkpoint
from train_utils.config.targets import collect_e2e_compressed_targets, resolve_target_layers
from train_utils.decoder_execution import (
    enable_vae_linear_by_execution_plan,
    prime_named_vae_linear_cache_with_group_fallback,
)
from train_utils.distill_data import build_distill_data_collator, build_distill_dataset
from train_utils.hif4_act import build_hif4_act_controller, register_hif4_act_hooks
from train_utils.model_level_optimizer import ModelLevelOptimizerLRConfig, attach_model_level_optimizer_contract
from train_utils.model_level_trainables import (
    build_model_level_trainable_selection,
    finalize_lm_head_linear_if_needed,
)
from train_utils.utils import get_logger


def _checkpoint_lora_config(meta: Optional[dict]) -> Optional[LoRAConfig]:
    if not isinstance(meta, dict):
        return None
    raw = meta.get("lora_config")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(f"checkpoint lora_config must be dict, got {type(raw)}.")
    missing = [key for key in ("rank", "alpha", "dropout") if key not in raw]
    if missing:
        raise ValueError(f"checkpoint lora_config is incomplete; missing={missing}.")
    resolved = LoRAConfig(
        rank=int(raw["rank"]),
        alpha=float(raw["alpha"]),
        dropout=float(raw["dropout"]),
    )
    resolved.validate()
    return resolved


def _resolve_effective_lora_config(cfg, *, structural_meta: Optional[dict], lora_active: bool) -> LoRAConfig:
    cfg.lora.validate()
    if not bool(lora_active):
        return cfg.lora
    checkpoint_cfg = _checkpoint_lora_config(structural_meta)
    if checkpoint_cfg is None:
        return cfg.lora
    return validate_lora_against_checkpoint(cfg.lora, checkpoint_cfg)


def _validate_resume_topology_request(
    step_meta: Optional[dict],
    *,
    cfg,
    resolved_target_layers: Sequence[int],
    resolved_target_modules: Sequence[str],
) -> None:
    if step_meta is None:
        return
    expected = {
        "train_mode": str(cfg.train_mode),
        "norm_train_mode": str(cfg.aux.norm_train_mode),
        "lm_head_train_mode": str(cfg.aux.lm_head_train_mode),
        "target_layers": [int(v) for v in resolved_target_layers],
        "target_modules": [str(v) for v in resolved_target_modules],
    }
    for key, current in expected.items():
        saved = step_meta.get(key)
        if saved != current:
            raise ValueError(
                f"E2E resume topology request mismatch for {key}: checkpoint={saved!r} current={current!r}."
            )


def _enable_internal_decode_runtime(
    selected: Sequence[Tuple[str, VAELinear]],
    *,
    decoder_enabled: bool,
    sparse_enabled: bool,
    vae_decoder_checkpoint: bool,
) -> None:
    if not (decoder_enabled or sparse_enabled):
        return
    for _name, module in selected:
        module.packed_vq_decoder_linear = True
        packed = getattr(module, "_parallel_stage_decoder", None)
        if isinstance(packed, nn.Module):
            if hasattr(packed, "use_checkpoint"):
                packed.use_checkpoint = bool(vae_decoder_checkpoint)
            continue
        seen: set[int] = set()
        for stage_idx in range(int(module.residual_stages)):
            for part_idx in range(int(module.parallel_parts)):
                decoder = module.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)
                if id(decoder) in seen:
                    continue
                seen.add(id(decoder))
                decoder.use_checkpoint = bool(vae_decoder_checkpoint)


def _selection_has_continuous(selection) -> bool:
    return any(
        bool(getattr(selection, name))
        for name in (
            "decoder_parameters",
            "lora_parameters",
            "norm_parameters",
            "lm_head_parameters",
        )
    )


def _resolved_learning_rates(cfg) -> dict:
    return {
        "learning_rate": float(cfg.opt.learning_rate),
        "decoder_lr": float(cfg.opt.resolved_decoder_lr()),
        "norm_lr": float(cfg.aux.norm_lr) if cfg.aux.norm_lr is not None else float(cfg.opt.learning_rate),
        "lm_head_lr": (
            float(cfg.aux.lm_head_lr)
            if cfg.aux.lm_head_lr is not None
            else float(cfg.opt.learning_rate)
        ),
        "weight_decay": float(cfg.opt.weight_decay),
    }


def _validate_v6_step_training_args(training_args) -> None:
    if bool(getattr(training_args, "save_only_model", False)):
        raise ValueError("v6 exact-step checkpointing requires save_only_model=false.")
    if bool(getattr(training_args, "ignore_data_skip", False)):
        raise ValueError("v6 exact-step resume requires ignore_data_skip=false.")
    if bool(getattr(training_args, "load_best_model_at_end", False)):
        raise ValueError("v6 lightweight step checkpoints do not support load_best_model_at_end.")


def _validate_sparse_trainer_modes(training_args) -> None:
    unsupported = []
    if getattr(training_args, "deepspeed", None):
        unsupported.append("DeepSpeed")
    fsdp = getattr(training_args, "fsdp", None)
    if fsdp and str(fsdp).strip().lower() not in {"", "[]", "none"}:
        unsupported.append("FSDP")
    if int(getattr(training_args, "tp_size", 1) or 1) > 1:
        unsupported.append("HF tensor parallel (tp_size>1)")
    if bool(getattr(training_args, "torch_compile", False)):
        unsupported.append("torch_compile")
    if bool(getattr(training_args, "auto_find_batch_size", False)):
        unsupported.append("auto_find_batch_size")
    if unsupported:
        raise ValueError(
            "Sparse Bit exact training does not support Trainer modes that rewrite model/optimizer lifecycle: "
            + ", ".join(unsupported)
        )


def _release_trainer_training_state(trainer, *, log) -> None:
    import gc

    released = []
    for attr in ("optimizer", "lr_scheduler", "scaler"):
        if getattr(trainer, attr, None) is not None:
            setattr(trainer, attr, None)
            released.append(attr)
    if released:
        log.info("Released Trainer training state before finalization: %s", ", ".join(released))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prewarm_sparse_residual_cache(model: nn.Module, training_args, *, log) -> dict:
    dtype = None
    if bool(getattr(training_args, "bf16", False)):
        dtype = torch.bfloat16
    elif bool(getattr(training_args, "fp16", False)):
        dtype = torch.float16
    stats = {"total": 0, "warmed": 0, "skipped": 0, "failed": 0}
    root = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    for _name, module in root.named_modules():
        if not isinstance(module, VAELinear):
            continue
        stats["total"] += 1
        has_sparse = getattr(module, "has_sparse_residual", None)
        if not (callable(has_sparse) and bool(has_sparse())):
            stats["skipped"] += 1
            continue
        prime = getattr(module, "prime_sparse_residual_cache", None)
        if not callable(prime):
            stats["failed"] += 1
            raise RuntimeError("VAELinear with sparse residual has no prime_sparse_residual_cache().")
        if bool(prime(dtype=dtype)):
            stats["warmed"] += 1
        else:
            stats["skipped"] += 1
    log.info("Sparse residual prewarm: %s", stats)
    return stats


def _snapshot_peft_lora_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    snapshot: Dict[str, torch.Tensor] = {}
    for module_name, layer in iter_named_peft_lora_layers(model):
        for rel_name, param in layer.named_parameters(recurse=True):
            if "lora_" not in str(rel_name):
                continue
            snapshot[f"{module_name}.{rel_name}"] = param.detach().to("cpu").clone()
    return snapshot


def _assert_tensor_snapshot_equal(
    before: Dict[str, torch.Tensor],
    after: Dict[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if set(before) != set(after):
        raise RuntimeError(
            f"{label} key mismatch: missing={sorted(set(before) - set(after))} "
            f"extra={sorted(set(after) - set(before))}"
        )
    for name in before:
        if not torch.equal(before[name], after[name]):
            raise RuntimeError(f"{label} changed unexpectedly at {name}.")


def _snapshot_packed_payloads(selected: Sequence[Tuple[str, VAELinear]]) -> dict:
    payload = {}
    for name, module in selected:
        banks = {}
        for stage_idx in range(int(module.residual_stages)):
            for part_idx in range(int(module.parallel_parts)):
                storage = module.get_stage_part_vq_storage(stage_idx=stage_idx, part_idx=part_idx)
                banks[(int(stage_idx), int(part_idx))] = (
                    storage.detach().to(device="cpu", dtype=torch.uint8).clone().contiguous()
                )
        payload[str(name)] = banks
    return payload


def _assert_packed_payloads_equal(before: dict, after: dict, *, label: str) -> None:
    if set(before) != set(after):
        raise RuntimeError(f"{label} module-set mismatch.")
    for module_name, banks in before.items():
        if set(banks) != set(after[module_name]):
            raise RuntimeError(f"{label} bank-set mismatch for {module_name}.")
        for bank_key, tensor in banks.items():
            if not torch.equal(tensor, after[module_name][bank_key]):
                raise RuntimeError(f"{label} changed at {module_name} bank={bank_key}.")


def _build_finalization_probe_inputs(tokenizer) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        "VAELLM finalization parity",
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.numel() < 1:
        raise RuntimeError("Tokenizer produced no input_ids for finalization parity probe.")
    max_tokens = min(4, int(input_ids.shape[-1]))
    probe = {"input_ids": input_ids[..., :max_tokens].contiguous()}
    attention_mask = encoded.get("attention_mask")
    if torch.is_tensor(attention_mask):
        probe["attention_mask"] = attention_mask[..., :max_tokens].contiguous()
    return probe


def _model_input_device(model: nn.Module) -> torch.device:
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        embeddings = get_embeddings()
        if isinstance(embeddings, nn.Module):
            for parameter in embeddings.parameters():
                return parameter.device
            for buffer in embeddings.buffers():
                return buffer.device
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cpu")


def _unwrap_model_for_finalization(trainer) -> nn.Module:
    """Remove distributed and mixed-precision forward wrappers before structural finalization."""
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is None:
        return trainer.model
    return accelerator.unwrap_model(
        trainer.model,
        keep_fp32_wrapper=False,
    )


def _load_completed_resume_state(
    resume_from_checkpoint: Optional[str],
    *,
    max_steps: int,
) -> Optional[TrainerState]:
    if not resume_from_checkpoint:
        return None
    state_path = os.path.join(os.path.abspath(str(resume_from_checkpoint)), "trainer_state.json")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Resume checkpoint is missing trainer_state.json: {state_path}")
    state = TrainerState.load_from_json(state_path)
    if int(state.global_step) < int(max_steps):
        return None
    if int(state.global_step) > int(max_steps):
        raise ValueError(
            "Resume checkpoint global_step exceeds configured max_steps: "
            f"global_step={int(state.global_step)} max_steps={int(max_steps)}."
        )
    return state


@torch.no_grad()
def _run_finalization_probe(
    model: nn.Module,
    probe_inputs: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.dtype]:
    device = _model_input_device(model)
    inputs = {name: tensor.to(device=device) for name, tensor in probe_inputs.items()}
    outputs = model(**inputs)
    logits = get_output_logits(outputs)
    if not torch.is_tensor(logits):
        raise TypeError(f"Finalization parity probe expected tensor logits, got {type(logits)}.")
    if not bool(torch.isfinite(logits.float()).all()):
        raise RuntimeError("Finalization parity probe produced non-finite logits.")
    return logits.detach().float().cpu(), logits.dtype


def _assert_finalization_probe_close(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    label: str = "Finalization parity",
    ulp_multiplier: float = 2.0,
    rtol_override: Optional[float] = None,
    atol_override: Optional[float] = None,
    relative_l2_limit: Optional[float] = None,
) -> dict:
    if before.shape != after.shape:
        raise RuntimeError(
            f"Finalization parity logits shape changed: before={tuple(before.shape)} after={tuple(after.shape)}."
        )
    if output_dtype.is_floating_point:
        tolerance = max(1e-5, float(ulp_multiplier) * float(torch.finfo(output_dtype).eps))
    else:
        tolerance = 1e-5
    rtol = tolerance if rtol_override is None else float(rtol_override)
    atol = tolerance if atol_override is None else float(atol_override)
    try:
        torch.testing.assert_close(before, after, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise AssertionError(f"{label} failed: {exc}") from exc
    diff = (after - before).float()
    denom = torch.linalg.vector_norm(before.float()).clamp_min(1e-12)
    relative_l2 = float((torch.linalg.vector_norm(diff) / denom).item()) if diff.numel() else 0.0
    if relative_l2_limit is not None and relative_l2 > float(relative_l2_limit):
        raise AssertionError(
            f"{label} relative_l2={relative_l2} exceeds limit={float(relative_l2_limit)}."
        )
    return {
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "relative_l2": relative_l2,
        "rtol": rtol,
        "atol": atol,
        "output_dtype": str(output_dtype),
    }


def _install_post_finalize_probe_runtime(
    model: nn.Module,
    *,
    cfg,
    layer_device_map,
):
    if str(cfg.runtime.parallel_mode) != "layer_mp":
        return [], None
    if layer_device_map is None:
        raise RuntimeError("layer_mp finalization probe requires resolved layer_device_map.")
    if layer_device_map and all(torch.device(device).type == "cpu" for device in layer_device_map.values()):
        return [], None
    if str(cfg.runtime.offload_mode) == "streaming":
        handles, _boundary_map = apply_boundary_device_map(
            model,
            layer_device_map=layer_device_map,
        )
        manager, _streaming_map = wrap_model_layers_for_streaming_offload(
            model,
            layer_devices=layer_device_map,
            prefetch_distance=int(cfg.runtime.offload_prefetch_distance),
            checkpoint_layers=False,
        )
        return handles, manager
    handles, _hf_device_map = apply_layer_device_map(
        model,
        layer_device_map=layer_device_map,
    )
    return handles, None


def _remove_post_finalize_probe_runtime(model: nn.Module, handles, manager) -> None:
    if manager is not None:
        manager.offload_all(synchronize=True)
        unwrap_streaming_offload_layers(model)
    for handle in handles:
        handle.remove()
    clear_model_vae_linear_cache(model)


def _prewarm_final_eval(model: nn.Module, *, group_size: int, log) -> dict:
    targets = [
        NamedVAELinearTarget(name=str(name), base_layer=module)
        for name, module in model.named_modules()
        if isinstance(module, VAELinear)
    ]
    stats, resolved = prime_named_vae_linear_cache_with_group_fallback(
        targets,
        clear_existing=True,
        compute_device=None,
        initial_group_size=int(group_size),
        logger=log,
    )
    if int(stats.get("failed", 0)) > 0:
        raise RuntimeError(f"Final eval prewarm failed: {stats}")
    return {
        **{str(k): int(v) for k, v in stats.items()},
        "group_size": int(resolved.group_size),
    }


def _run_final_lm_eval(*, model, tokenizer, cfg, base_model_path: str, output_dir: str, log):
    from compressed_e2e_fintuning.mid_eval import run_e2e_lm_eval

    eval_args = _build_eval_args(cfg)
    if not bool(str(eval_args.eval_tasks or "").strip()):
        return None
    return run_e2e_lm_eval(
        model=model,
        tokenizer=tokenizer,
        args=eval_args,
        base_model_path=str(base_model_path),
        output_dir=str(output_dir),
        log=log,
        eval_tag="final",
        move_to_device=str(cfg.runtime.parallel_mode) == "dp" or (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ),
    )


def run_pipeline(cfg, hf_args, training_args) -> Dict[str, object]:
    log = get_logger("compressed_e2e_fintuning")
    model, round_base_dir, round_base_meta, step_meta, base_model_path = _load_v6_student(
        cfg, hf_args, log=log
    )
    run_output_dir = _resolve_run_output_dir(cfg, base_model_path=base_model_path)
    _install_run_file_logger(log, run_output_dir)
    log.info("Run output directory: %s", run_output_dir)

    _validate_v6_step_training_args(training_args)
    train_decoder, train_lora, train_sparse = _resolve_train_components(cfg.train_mode)
    if (
        str(cfg.train_mode) == "none"
        and str(cfg.aux.norm_train_mode) == "none"
        and str(cfg.aux.lm_head_train_mode) == "none"
    ):
        raise ValueError("train_mode=none requires norm_train_mode or lm_head_train_mode to be enabled.")
    if train_sparse:
        _validate_sparse_trainer_modes(training_args)

    root = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    layer_count = len(list(get_layers(root)))
    resolved_target_layers = resolve_target_layers(cfg.target_layers, num_layers=layer_count)
    selected = collect_e2e_compressed_targets(
        root,
        target_layers=cfg.target_layers,
        target_modules=cfg.target_modules,
        num_layers=layer_count,
    )
    resolved_target_modules = _module_suffixes(selected)
    if (train_decoder or train_lora or train_sparse) and not selected:
        raise ValueError("The requested E2E train_mode requires at least one compressed VAELinear target.")
    _validate_resume_topology_request(
        step_meta,
        cfg=cfg,
        resolved_target_layers=resolved_target_layers,
        resolved_target_modules=resolved_target_modules,
    )

    _enable_internal_decode_runtime(
        selected,
        decoder_enabled=train_decoder,
        sparse_enabled=train_sparse,
        vae_decoder_checkpoint=bool(cfg.runtime.vae_decoder_checkpoint),
    )
    initial_low_rank_payloads = _collect_existing_full_low_rank(selected) if train_lora else None
    lora_active = bool(train_lora or str(cfg.aux.lm_head_train_mode) == "lora")
    structural_meta = step_meta if step_meta is not None else round_base_meta
    effective_lora = _resolve_effective_lora_config(
        cfg,
        structural_meta=structural_meta,
        lora_active=lora_active,
    )
    decoder_execution_mode = "decoder_sparse_bit" if train_decoder and train_sparse else "trainable_decoder"
    selection = build_model_level_trainable_selection(
        model,
        aux=cfg.aux,
        compressed_modules=selected,
        dense_target_modules=(),
        rank=int(effective_lora.rank),
        alpha=float(effective_lora.alpha),
        dropout=float(effective_lora.dropout),
        rank_explicit=bool(cfg.lora.rank_explicit),
        initial_low_rank_payloads=initial_low_rank_payloads,
        train_decoder=bool(train_decoder),
        train_lora=bool(train_lora),
        decoder_execution_mode=decoder_execution_mode,
        freeze=True,
    )
    model = selection.peft_model or model
    exact_lora_config = collect_exact_peft_lora_config(
        model,
        default_rank=int(effective_lora.rank),
        alpha=float(effective_lora.alpha),
        dropout=float(effective_lora.dropout),
    )
    if train_sparse and not train_decoder:
        for _name, module in selected:
            enable_vae_linear_by_execution_plan(module, mode="sparse_bit")

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(getattr(model, "config", None), "use_cache"):
        model.config.use_cache = False

    model, hook_handles, streaming_manager, saved_tensor_offload, dp_device, layer_device_map = _place_student_model(
        model, cfg, training_args, log=log
    )

    sparse_bit_manager = None
    if train_sparse:
        from sparse_bit_tuning.config import SparseBitTuningConfig
        from sparse_bit_tuning.manager import SparseBitTuningManager
        from sparse_bit_tuning.runtime_integration import resolve_target_devices

        target_devices = resolve_target_devices(
            selected,
            parallel_mode=str(cfg.runtime.parallel_mode),
            dp_local_device=dp_device,
            offload_mode=str(cfg.runtime.offload_mode),
            layer_device_map=layer_device_map,
        )
        sparse_cfg = SparseBitTuningConfig(
            enabled=True,
            active_ratio=float(cfg.bit_active_ratio),
            optimizer=str(cfg.bit_optimizer),
            bit_lr=cfg.bit_lr,
            weight_decay=float(cfg.bit_weight_decay),
            round_steps=cfg.bit_round_steps,
        ).normalized()
        sparse_bit_manager = SparseBitTuningManager(
            root_model=model,
            targets=selected,
            target_devices=target_devices,
            training_seed=int(training_args.seed),
            config=sparse_cfg,
            streaming=str(cfg.runtime.offload_mode) == "streaming",
        )

    sparse_residual_prewarm = _prewarm_sparse_residual_cache(model, training_args, log=log)

    tokenizer = _build_v6_tokenizer(base_model_path, hf_args)
    _sync_model_padding_config(model, tokenizer)
    data_bundle = build_distill_dataset(cfg.data, tokenizer)
    if int(getattr(training_args, "dataloader_num_workers", 0) or 0) <= 0:
        training_args.dataloader_num_workers = int(default_dataloader_num_workers())
    training_args.dataloader_pin_memory = True
    training_args.group_by_length = bool(data_bundle.group_by_length)
    if bool(data_bundle.is_iterable):
        from transformers.trainer_pt_utils import AcceleratorConfig

        training_args.group_by_length = False
        training_args.accelerator_config = AcceleratorConfig(
            dispatch_batches=False,
            split_batches=False,
        )
    data_collator = build_distill_data_collator(
        tokenizer,
        model_max_length=int(cfg.data.model_max_length),
        dynamic_padding=bool(cfg.data.dynamic_padding),
    )

    teacher_model, teacher_identity = _load_teacher(
        cfg,
        hf_args,
        training_args,
        model,
        base_model_path=base_model_path,
        dp_device=dp_device,
        log=log,
    )
    immutable_contract = build_e2e_immutable_resume_contract(
        cfg=cfg,
        training_args=training_args,
        tokenizer=tokenizer,
        input_checkpoint_id=str(round_base_meta["checkpoint_id"]),
        resolved_target_layers=resolved_target_layers,
        resolved_target_modules=resolved_target_modules,
        teacher_identity=teacher_identity,
    )
    immutable_contract["lora"] = exact_lora_config

    training_args.output_dir = os.path.join(run_output_dir, "trainer_state")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False
    training_args.save_safetensors = False

    eval_args = _build_eval_args(cfg)
    callbacks = [E2ETrainerLogCallback(logger=log)]
    eval_after_save_callback = None
    if bool(eval_args.eval_after_save):
        eval_after_save_callback = EvalAfterSaveCallback(
            e2e_args=eval_args,
            tokenizer=tokenizer,
            base_model_path=base_model_path,
            run_output_dir=run_output_dir,
            log=log,
            parallel_mode=str(cfg.runtime.parallel_mode),
        )
        callbacks.append(eval_after_save_callback)

    hif4_controller = build_hif4_act_controller(bool(cfg.runtime.distill_hif4_act))
    hif4_handles = []
    if hif4_controller is not None:
        hif4_handles = register_hif4_act_hooks(model, hif4_controller)
        if not hif4_handles:
            raise RuntimeError("distill_hif4_act=true but no student linear modules were hookable.")

    trainer = VAEDecoderE2ETrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=data_bundle.train_dataset,
        eval_dataset=data_bundle.eval_dataset,
        data_collator=data_collator,
        loss_config=cfg.loss,
        teacher_model=teacher_model,
        saved_tensor_offload=saved_tensor_offload,
        streaming_offload_manager=streaming_manager,
        teacher_output_offload=str(cfg.runtime.teacher_output_offload),
        teacher_model_offload=str(cfg.runtime.teacher_model_offload),
        teacher_output_pin_memory=bool(cfg.runtime.teacher_output_pin_memory),
        teacher_output_chunk_tokens=int(cfg.runtime.teacher_output_chunk_tokens),
        sparse_bit_manager=sparse_bit_manager,
        distill_hif4_act_controller=hif4_controller,
        callbacks=callbacks,
    )
    if _selection_has_continuous(selection):
        lr_config = ModelLevelOptimizerLRConfig(
            learning_rate=float(cfg.opt.learning_rate),
            weight_decay=float(cfg.opt.weight_decay),
            decoder_lr=cfg.opt.decoder_lr,
            norm_lr=cfg.aux.norm_lr,
            lm_head_lr=cfg.aux.lm_head_lr,
        )
        attach_model_level_optimizer_contract(
            trainer,
            selection=selection,
            lr_config=lr_config,
        )
    replace_progress_log_callback(trainer)
    trainer.add_callback(E2EDistillTokenStatsCallback(trainer=trainer, logger=log))
    if eval_after_save_callback is not None:
        eval_after_save_callback.bind_trainer(trainer)

    checkpoint_context = {
        "round_base_dir": str(round_base_dir),
        "round_base_checkpoint_id": str(round_base_meta["checkpoint_id"]),
        "train_mode": str(cfg.train_mode),
        "compressed_targets": tuple(round_base_meta.get("compressed_targets") or ()),
        "pending_dense_targets": tuple(round_base_meta.get("pending_dense_targets") or ()),
        "skip_targets": tuple(round_base_meta.get("skip_targets") or ()),
        "legacy_original_only_sources": tuple(
            round_base_meta.get("legacy_original_only_sources") or ()
        ),
        "norm_train_mode": str(cfg.aux.norm_train_mode),
        "lm_head_train_mode": str(cfg.aux.lm_head_train_mode),
        "lora_config": exact_lora_config,
        "resolved_learning_rates": _resolved_learning_rates(cfg),
        "compression_categories": tuple(round_base_meta.get("compression_categories") or ()),
        "target_layers": tuple(int(v) for v in resolved_target_layers),
        "target_modules": tuple(str(v) for v in resolved_target_modules),
        "immutable_resume_contract": immutable_contract,
        "base_model_path": str(base_model_path),
        "runtime_audit": {
            "runtime": "compressed_e2e_fintuning.runtime_v6",
            "distill_hif4_act": bool(cfg.runtime.distill_hif4_act),
            "selected_target_count": len(selected),
            "recovery_lora_config": exact_lora_config,
        },
        "hf_artifact_refs": {},
    }
    trainer.configure_v6_step_checkpoint(
        context=checkpoint_context,
        selected_vae_modules=selected,
    )

    try:
        completed_resume_state = _load_completed_resume_state(
            cfg.resume_from_checkpoint or None,
            max_steps=int(training_args.max_steps),
        )
        if completed_resume_state is None:
            trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint or None)
        else:
            trainer._load_from_checkpoint(str(cfg.resume_from_checkpoint))
            trainer.state = completed_resume_state
            log.info(
                "Resume checkpoint already reached max_steps; loaded exact state and skipped training: "
                "global_step=%d max_steps=%d",
                int(trainer.state.global_step),
                int(training_args.max_steps),
            )
    except Exception:
        _cleanup_runtime(
            model,
            hook_handles=hook_handles,
            streaming_manager=streaming_manager,
            hif4_handles=hif4_handles,
        )
        raise

    final_model = _unwrap_model_for_finalization(trainer)
    final_model.eval()
    if hasattr(trainer, "offload_teacher_to_cpu"):
        trainer.offload_teacher_to_cpu()
    elif teacher_model is not None:
        teacher_model.to("cpu")
    _release_trainer_training_state(trainer, log=log)

    lora_before_sparse = (
        _snapshot_peft_lora_parameters(final_model)
        if sparse_bit_manager is not None
        else {}
    )
    packed_after_commit = None
    if sparse_bit_manager is not None:
        sparse_bit_manager.final_commit()
        if lora_before_sparse:
            _assert_tensor_snapshot_equal(
                lora_before_sparse,
                _snapshot_peft_lora_parameters(final_model),
                label="LoRA parameters across Sparse Bit final_commit",
            )
        packed_after_commit = _snapshot_packed_payloads(selected)
        sparse_bit_manager.detach_runtime()

    finalization_probe_inputs = None
    finalization_probe_before = None
    finalization_probe_dtype = None
    if _is_main_process():
        finalization_probe_inputs = _build_finalization_probe_inputs(tokenizer)
        finalization_probe_before, finalization_probe_dtype = _run_finalization_probe(
            final_model,
            finalization_probe_inputs,
        )

    if train_decoder or train_sparse:
        _finalize_decoders(selected)

    compressed_proxy_names = [str(name) for name, _module in selected] if train_lora else []
    if list(iter_named_peft_lora_layers(final_model)):
        final_model = finalize_model_level_lora(
            final_model,
            compressed_proxy_names=compressed_proxy_names or None,
        )

    if packed_after_commit is not None and lora_active:
        _assert_packed_payloads_equal(
            packed_after_commit,
            _snapshot_packed_payloads(selected),
            label="Sparse Bit hard payload across LoRA finalization",
        )

    core_structural_parity = None
    core_structural_probe = None
    if _is_main_process():
        core_structural_probe, core_structural_dtype = _run_finalization_probe(
            final_model,
            finalization_probe_inputs,
        )
        if core_structural_dtype != finalization_probe_dtype:
            raise RuntimeError(
                "Core structural finalization output dtype changed: "
                f"before={finalization_probe_dtype} after={core_structural_dtype}."
            )
        core_structural_parity = _assert_finalization_probe_close(
            finalization_probe_before,
            core_structural_probe,
            output_dtype=finalization_probe_dtype,
            label="Core structural finalization parity",
        )

    lm_head_fused = finalize_lm_head_linear_if_needed(
        final_model,
        lm_head_train_mode=str(cfg.aux.lm_head_train_mode),
    )

    structural_probe_after = None
    structural_parity = None
    if _is_main_process():
        structural_probe_after, structural_probe_dtype = _run_finalization_probe(
            final_model,
            finalization_probe_inputs,
        )
        if structural_probe_dtype != finalization_probe_dtype:
            raise RuntimeError(
                "Structural finalization output dtype changed: "
                f"before={finalization_probe_dtype} after={structural_probe_dtype}."
            )
        structural_parity = _assert_finalization_probe_close(
            finalization_probe_before,
            structural_probe_after,
            output_dtype=finalization_probe_dtype,
            label="Structural finalization parity",
            # Reassociating two large linear operators into one is mathematically
            # exact but changes floating-point accumulation. Core decoder/LoRA parity was
            # already checked above with the strict default tolerance.
            ulp_multiplier=32.0 if lm_head_fused else 2.0,
            rtol_override=1e-3 if lm_head_fused else None,
            atol_override=0.25 if lm_head_fused else None,
            relative_l2_limit=5e-3 if lm_head_fused else None,
        )
        if lm_head_fused:
            checkpoint_context["runtime_audit"]["lm_head_fusion_forward_parity"] = (
                _assert_finalization_probe_close(
                    core_structural_probe,
                    structural_probe_after,
                    output_dtype=finalization_probe_dtype,
                    label="LM-head fusion forward parity",
                    ulp_multiplier=32.0,
                    rtol_override=1e-3,
                    atol_override=0.25,
                    relative_l2_limit=5e-3,
                )
            )

    _cleanup_runtime(
        final_model,
        hook_handles=hook_handles,
        streaming_manager=streaming_manager,
        hif4_handles=hif4_handles,
    )
    _assert_final_runtime_clean(final_model)

    finalization_parity = None
    runtime_cleanup_parity = None
    if _is_main_process():
        probe_handles, probe_streaming_manager = _install_post_finalize_probe_runtime(
            final_model,
            cfg=cfg,
            layer_device_map=layer_device_map,
        )
        try:
            finalization_probe_after, finalization_probe_after_dtype = _run_finalization_probe(
                final_model,
                finalization_probe_inputs,
            )
            if finalization_probe_after_dtype != finalization_probe_dtype:
                raise RuntimeError(
                    "Finalization parity output dtype changed: "
                    f"before={finalization_probe_dtype} after={finalization_probe_after_dtype}."
                )
            runtime_cleanup_parity = _assert_finalization_probe_close(
                structural_probe_after,
                finalization_probe_after,
                output_dtype=finalization_probe_dtype,
                label="Runtime cleanup/reinstall parity",
            )
            finalization_parity = _assert_finalization_probe_close(
                finalization_probe_before,
                finalization_probe_after,
                output_dtype=finalization_probe_dtype,
                label="End-to-end finalization parity",
                ulp_multiplier=32.0 if lm_head_fused else 2.0,
                rtol_override=1e-3 if lm_head_fused else None,
                atol_override=0.25 if lm_head_fused else None,
                relative_l2_limit=5e-3 if lm_head_fused else None,
            )
        finally:
            _remove_post_finalize_probe_runtime(
                final_model,
                probe_handles,
                probe_streaming_manager,
            )
    dist_ready = bool(torch.distributed.is_available() and torch.distributed.is_initialized())
    parity_payload = (
        {
            "structural": structural_parity,
            "runtime_cleanup": runtime_cleanup_parity,
            "end_to_end": finalization_parity,
        }
        if _is_main_process()
        else None
    )
    if dist_ready:
        payload = [parity_payload]
        torch.distributed.broadcast_object_list(payload, src=0)
        parity_payload = payload[0]
    if not isinstance(parity_payload, dict):
        raise RuntimeError("Finalization forward parity result was not resolved on all ranks.")
    for key in ("structural", "runtime_cleanup", "end_to_end"):
        if not isinstance(parity_payload.get(key), dict):
            raise RuntimeError(f"Finalization parity component {key!r} was not resolved on all ranks.")
    structural_parity = dict(parity_payload["structural"])
    runtime_cleanup_parity = dict(parity_payload["runtime_cleanup"])
    finalization_parity = dict(parity_payload["end_to_end"])
    checkpoint_context["runtime_audit"]["structural_finalization_forward_parity"] = structural_parity
    checkpoint_context["runtime_audit"]["core_structural_finalization_forward_parity"] = (
        core_structural_parity
    )
    checkpoint_context["runtime_audit"]["runtime_cleanup_forward_parity"] = runtime_cleanup_parity
    checkpoint_context["runtime_audit"]["finalization_forward_parity"] = finalization_parity

    final_lm_eval = None
    has_eval_tasks = bool(str(cfg.runtime.evaluation.eval_tasks or "").strip())
    if has_eval_tasks and (str(cfg.runtime.parallel_mode) == "dp" or dist_ready):
        final_lm_eval = _run_final_lm_eval(
            model=final_model,
            tokenizer=tokenizer,
            cfg=cfg,
            base_model_path=base_model_path,
            output_dir=run_output_dir,
            log=log,
        )
        clear_model_vae_linear_cache(final_model)

    final_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finalized_status = {
        "sparse_bit_committed": bool(train_sparse),
        "decoder_finalized": bool(train_decoder or train_sparse),
        "lora_finalized": bool(lora_active),
        "aux_finalized": True,
        "runtime_clean": True,
        "inference_forward_parity": True,
    }
    final_dir = os.path.join(run_output_dir, "final_model")
    save_result = save_v6_full_checkpoint(
        final_model,
        final_dir,
        checkpoint_kind="final_model",
        compressed_targets=tuple(round_base_meta.get("compressed_targets") or ()),
        pending_dense_targets=tuple(round_base_meta.get("pending_dense_targets") or ()),
        skip_targets=tuple(round_base_meta.get("skip_targets") or ()),
        legacy_original_only_sources=tuple(
            round_base_meta.get("legacy_original_only_sources") or ()
        ),
        train_mode=str(cfg.train_mode),
        norm_train_mode=str(cfg.aux.norm_train_mode),
        lm_head_train_mode=str(cfg.aux.lm_head_train_mode),
        lora_config=None,
        resolved_learning_rates=checkpoint_context["resolved_learning_rates"],
        completed_categories=tuple(round_base_meta.get("completed_categories") or ()),
        compression_categories=tuple(round_base_meta.get("compression_categories") or ()),
        target_layers=tuple(int(v) for v in resolved_target_layers),
        target_modules=tuple(str(v) for v in resolved_target_modules),
        immutable_resume_contract=immutable_contract,
        finalized_status=finalized_status,
        runtime_audit=checkpoint_context["runtime_audit"],
        base_model_path=str(base_model_path),
        tokenizer=tokenizer if bool(cfg.save_tokenizer) else None,
        save_config=True,
        is_main_process=_is_main_process(),
        distributed_barrier=_barrier if dist_ready else None,
    )

    ppl_eval = None
    eval_prewarm = None
    if _is_main_process():
        need_ppl = not bool(cfg.runtime.evaluation.skip_ppl_eval)
        need_post_save_lm = bool(has_eval_tasks and final_lm_eval is None)
        if need_ppl or need_post_save_lm:
            final_model.to(torch.device(str(cfg.runtime.evaluation.eval_device)))
            eval_prewarm = _prewarm_final_eval(
                final_model,
                group_size=int(cfg.runtime.evaluation.eval_prewarm_group_size),
                log=log,
            )
            if need_ppl:
                ppl_eval = eval_final_ppl(
                    model=final_model,
                    args=eval_args,
                    model_path=str(base_model_path),
                    output_dir=run_output_dir,
                    log=log,
                )
            if need_post_save_lm:
                final_lm_eval = _run_final_lm_eval(
                    model=final_model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    base_model_path=base_model_path,
                    output_dir=run_output_dir,
                    log=log,
                )
            clear_model_vae_linear_cache(final_model)
            final_model.to("cpu")

        run_meta_path = os.path.join(run_output_dir, "run_meta.json")
        with open(run_meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "runtime": "compressed_e2e_fintuning.runtime_v6",
                    "round_base_checkpoint_id": str(round_base_meta["checkpoint_id"]),
                    "final_checkpoint_id": str(save_result["checkpoint_id"]),
                    "global_step": int(getattr(trainer.state, "global_step", 0)),
                    "train_mode": str(cfg.train_mode),
                    "data_sources": data_bundle.source_stats,
                    "sparse_residual_prewarm": sparse_residual_prewarm,
                    "finalization_forward_parity": finalization_parity,
                },
                handle,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            handle.write("\n")
    else:
        run_meta_path = None

    return {
        "run_output_dir": run_output_dir,
        "saved_model_dir": str(save_result["output_dir"]),
        "checkpoint_id": str(save_result["checkpoint_id"]),
        "round_base_checkpoint_id": str(round_base_meta["checkpoint_id"]),
        "global_step": int(getattr(trainer.state, "global_step", 0)),
        "final_ppl": None if ppl_eval is None else ppl_eval["result"],
        "final_ppl_path": None if ppl_eval is None else ppl_eval["path"],
        "final_lm_eval_path": None if final_lm_eval is None else final_lm_eval.get("json_path"),
        "final_lm_eval_summary_path": (
            None if final_lm_eval is None else final_lm_eval.get("summary_path")
        ),
        "final_eval_prewarm": eval_prewarm,
        "sparse_residual_prewarm": sparse_residual_prewarm,
        "finalization_forward_parity": finalization_parity,
        "run_meta_path": run_meta_path,
    }


__all__ = ["run_pipeline"]
