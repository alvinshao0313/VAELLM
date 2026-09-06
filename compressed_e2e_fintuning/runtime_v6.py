"""Canonical v6 runtime for compressed E2E recovery training."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from compressed_e2e_fintuning.device_map import (
    apply_boundary_device_map,
    apply_layer_device_map,
    resolve_layer_device_map,
)
from compressed_e2e_fintuning.offload import (
    SavedTensorOffloadContext,
    unwrap_streaming_offload_layers,
    validate_streaming_layer_devices,
    wrap_model_layers_for_streaming_offload,
)
from e2e_common.data import build_tokenizer
from e2e_common.full_lora import iter_named_peft_lora_layers
from litebsq.vae_linear import VAELinear, clear_model_vae_linear_cache
from rotation.model_utils import get_layers, get_model
from train_utils.base_reference import load_frozen_base_reference_model_distributed_from_hf_args
from train_utils.checkpoint_v6 import (
    FULL_MODEL_KINDS,
    load_v6_full_checkpoint_into_model,
    load_v6_meta,
    load_v6_training_step_meta,
    resolve_training_step_round_base_ref,
    resolve_v6_checkpoint_dir,
)
from train_utils.distill_teacher import resolve_distill_teacher_dtype, resolve_distill_teacher_required
from train_utils.distributed_guard import distributed_guarded_main
from train_utils.hif4_act import remove_hif4_act_hooks


_TRAIN_MODE_COMPONENTS = {
    "none": (False, False, False),
    "decoder": (True, False, False),
    "lora": (False, True, False),
    "sparse_bit": (False, False, True),
    "decoder_lora": (True, True, False),
    "decoder_sparse_bit": (True, False, True),
    "lora_sparse_bit": (False, True, True),
    "decoder_lora_sparse_bit": (True, True, True),
}


def _dist_ready() -> bool:
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized())


def _barrier() -> None:
    if _dist_ready():
        torch.distributed.barrier()


def _resolve_hf_credential(hf_args):
    for key, value in vars(hf_args).items():
        if str(key).startswith("access_"):
            return value
    return None


def _build_v6_tokenizer(base_model_path: str, hf_args):
    return build_tokenizer(str(base_model_path), _resolve_hf_credential(hf_args))


def _sync_model_padding_config(model: nn.Module, tokenizer) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return
    field_name = "pad_" + "token_id"
    if getattr(config, field_name, None) is None:
        setattr(config, field_name, getattr(tokenizer, field_name, None))


def _world_rank() -> int:
    return int(torch.distributed.get_rank()) if _dist_ready() else 0


def _is_main_process() -> bool:
    return _world_rank() == 0


def _safe_path_token(value: str) -> str:
    text = str(value or "").strip().replace("\\\\", "/")
    text = re.sub(r"[^A-Za-z0-9._/-]+", "_", text)
    text = text.replace("/", "__")
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "unknown_model"


def _create_run_output_dir(root_output_dir: str, model_path: str) -> str:
    root = os.path.abspath(str(root_output_dir))

    def _create() -> str:
        os.makedirs(root, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        base = os.path.join(root, f"{_safe_path_token(model_path)}_{ts}")
        run_dir = base
        suffix = 1
        while os.path.exists(run_dir):
            run_dir = f"{base}_{suffix}"
            suffix += 1
        os.makedirs(run_dir, exist_ok=False)
        return run_dir

    run_dir = distributed_guarded_main(_create, barrier=True) if _dist_ready() else _create()
    if not isinstance(run_dir, str) or not run_dir:
        raise RuntimeError(f"Failed to resolve run output directory: {run_dir!r}")
    os.makedirs(run_dir, exist_ok=True)
    return os.path.abspath(run_dir)


def _resolve_run_output_dir(cfg, *, base_model_path: str) -> str:
    if cfg.resume_from_checkpoint:
        step_dir = os.path.abspath(str(cfg.resume_from_checkpoint))
        parent = os.path.dirname(step_dir)
        if os.path.basename(parent) == "trainer_state":
            return os.path.dirname(parent)
    return _create_run_output_dir(str(cfg.run_root_dir), str(base_model_path))


def _install_run_file_logger(log: logging.Logger, run_output_dir: str) -> None:
    if not _is_main_process():
        return
    path = os.path.join(run_output_dir, "compressed_e2e_fintuning.log")
    for handler in log.handlers:
        if isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == os.path.abspath(path):
            return
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    log.addHandler(handler)


def _resolve_round_base(cfg) -> Tuple[str, dict, Optional[dict]]:
    configured_dir = resolve_v6_checkpoint_dir(str(cfg.student_checkpoint_dir))
    configured_meta = load_v6_meta(configured_dir)
    if configured_meta.get("checkpoint_kind") not in FULL_MODEL_KINDS:
        raise ValueError(
            "--student_checkpoint_dir must be a full v6 checkpoint; "
            f"got kind={configured_meta.get('checkpoint_kind')!r}."
        )
    if not cfg.resume_from_checkpoint:
        return os.path.abspath(configured_dir), dict(configured_meta), None
    step_dir = os.path.abspath(str(cfg.resume_from_checkpoint))
    step_meta = load_v6_training_step_meta(step_dir)
    round_base_dir, round_base_meta = resolve_training_step_round_base_ref(step_dir, step_meta)
    if str(configured_meta["checkpoint_id"]) != str(round_base_meta["checkpoint_id"]):
        raise ValueError(
            "student checkpoint and resume checkpoint resolve to different round bases: "
            f"student={configured_meta['checkpoint_id']!r} resume={round_base_meta['checkpoint_id']!r}."
        )
    return os.path.abspath(round_base_dir), dict(round_base_meta), dict(step_meta)


def _resolve_train_components(train_mode: str) -> Tuple[bool, bool, bool]:
    mode = str(train_mode).strip().lower()
    try:
        return _TRAIN_MODE_COMPONENTS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported canonical train_mode={train_mode!r}.") from exc


def _load_v6_student(cfg, hf_args, *, log):
    round_base_dir, round_base_meta, step_meta = _resolve_round_base(cfg)
    base_model_path = str(round_base_meta.get("base_model_path") or "").strip()
    if not base_model_path:
        raise ValueError("v6 round base is missing non-empty base_model_path.")
    model = get_model(base_model_path, _resolve_hf_credential(hf_args))
    model, loaded_meta, _load_result = load_v6_full_checkpoint_into_model(
        model,
        round_base_dir,
        expected_kind=str(round_base_meta["checkpoint_kind"]),
        strict=True,
    )
    if str(loaded_meta["checkpoint_id"]) != str(round_base_meta["checkpoint_id"]):
        raise RuntimeError("v6 round-base checkpoint_id changed during load.")
    if hasattr(getattr(model, "config", None), "use_cache"):
        model.config.use_cache = False
    log.info(
        "Loaded v6 round base: dir=%s checkpoint_id=%s kind=%s base_model=%s",
        round_base_dir,
        round_base_meta["checkpoint_id"],
        round_base_meta["checkpoint_kind"],
        base_model_path,
    )
    return model, round_base_dir, round_base_meta, step_meta, base_model_path


def _unwrap_peft_root(model: nn.Module) -> nn.Module:
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        base = getter()
        if isinstance(base, nn.Module):
            return base
    return model


def _module_suffixes(selected: Sequence[Tuple[str, VAELinear]]) -> Tuple[str, ...]:
    out: List[str] = []
    seen = set()
    for name, _module in selected:
        suffix = str(name).rsplit(".", 1)[-1]
        if suffix not in seen:
            seen.add(suffix)
            out.append(suffix)
    return tuple(out)


def _set_selected_decoder_checkpoint(selected: Sequence[Tuple[str, VAELinear]], *, enabled: bool) -> None:
    seen: set[int] = set()
    for _name, module in selected:
        packed = getattr(module, "_parallel_stage_decoder", None)
        if isinstance(packed, nn.Module):
            if id(packed) not in seen:
                seen.add(id(packed))
                if hasattr(packed, "use_checkpoint"):
                    packed.use_checkpoint = bool(enabled)
            continue
        for stage_idx in range(int(module.residual_stages)):
            for part_idx in range(int(module.parallel_parts)):
                decoder = module.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)
                if id(decoder) in seen:
                    continue
                seen.add(id(decoder))
                decoder.use_checkpoint = bool(enabled)


def _collect_existing_full_low_rank(
    selected: Sequence[Tuple[str, VAELinear]],
) -> Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, module in selected:
        low_rank_a = getattr(module, "low_rank_a", None)
        low_rank_b = getattr(module, "low_rank_b", None)
        if (low_rank_a is None) != (low_rank_b is None):
            raise RuntimeError(f"{name}: partial VAELinear low-rank payload is invalid.")
        if low_rank_a is None:
            continue
        module._validate_low_rank_payload_tensors(low_rank_a, low_rank_b)
        payloads[str(name)] = (
            low_rank_a.detach().to("cpu").clone().contiguous(),
            low_rank_b.detach().to("cpu").clone().contiguous(),
        )
    return payloads or None


def _resolve_dp_local_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def _place_student_model(model: nn.Module, cfg, training_args, *, log):
    parallel_mode = str(cfg.runtime.parallel_mode)
    offload_mode = str(cfg.runtime.offload_mode)
    structure_model = _unwrap_peft_root(model)
    hook_handles = []
    streaming_manager = None
    saved_tensor_offload = None
    dp_local_device: Optional[torch.device] = None
    layer_device_map: Optional[Dict[int, torch.device]] = None
    if parallel_mode == "dp":
        dp_local_device = _resolve_dp_local_device()
        model.to(dp_local_device)
        if offload_mode == "saved_tensors":
            saved_tensor_offload = SavedTensorOffloadContext(
                enabled=True,
                min_tensor_bytes=int(cfg.runtime.offload_min_tensor_bytes),
                pin_memory=bool(cfg.runtime.offload_pin_memory),
            )
        return model, hook_handles, streaming_manager, saved_tensor_offload, dp_local_device, layer_device_map

    layers = list(get_layers(structure_model))
    layer_device_map = resolve_layer_device_map(str(cfg.runtime.layer_device_map), len(layers))
    if offload_mode == "streaming":
        validate_streaming_layer_devices(layer_device_map)
        if bool(training_args.gradient_checkpointing):
            log.info("streaming offload owns layer checkpointing; disabling HF gradient_checkpointing")
            training_args.gradient_checkpointing = False
        hook_handles, boundary_map = apply_boundary_device_map(
            structure_model, layer_device_map=layer_device_map
        )
        streaming_manager, streaming_map = wrap_model_layers_for_streaming_offload(
            structure_model,
            layer_devices=layer_device_map,
            prefetch_distance=int(cfg.runtime.offload_prefetch_distance),
            checkpoint_layers=bool(cfg.runtime.offload_checkpoint),
        )
        hf_device_map = {**boundary_map, **streaming_map}
    else:
        hook_handles, hf_device_map = apply_layer_device_map(
            structure_model, layer_device_map=layer_device_map
        )
    if model is not structure_model:
        setattr(model, "hf_device_map", dict(hf_device_map))
        setattr(model, "is_parallelizable", True)
        setattr(model, "model_parallel", True)
    if offload_mode in {"saved_tensors", "streaming"}:
        saved_tensor_offload = SavedTensorOffloadContext(
            enabled=True,
            min_tensor_bytes=int(cfg.runtime.offload_min_tensor_bytes),
            pin_memory=bool(cfg.runtime.offload_pin_memory),
        )
    return model, hook_handles, streaming_manager, saved_tensor_offload, dp_local_device, layer_device_map


def _model_identity(model: nn.Module, model_path: str) -> dict:
    config = getattr(model, "config", None)
    architectures = getattr(config, "architectures", None) if config is not None else None
    payload = {
        "model_type": getattr(config, "model_type", None) if config is not None else None,
        "architectures": list(architectures or []),
        "hidden_size": getattr(config, "hidden_size", None) if config is not None else None,
        "num_hidden_layers": getattr(config, "num_hidden_layers", None) if config is not None else None,
        "vocab_size": getattr(config, "vocab_size", None) if config is not None else None,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "model_path": str(model_path),
        "revision_hint": None if config is None else getattr(config, "_commit_hash", None),
        "config_digest": digest,
        "config": payload,
    }


def _load_teacher(cfg, hf_args, training_args, student_model, *, base_model_path: str, dp_device, log):
    required = resolve_distill_teacher_required(
        loss_type=str(cfg.loss.loss_type),
        hidden_loss_weight=float(cfg.loss.hidden_loss_weight),
        pre_mlp_hidden_loss_weight=float(cfg.loss.pre_mlp_hidden_loss_weight),
    )
    if not required:
        return None, None
    teacher_path = str(cfg.teacher_model_path or base_model_path)
    dtype = resolve_distill_teacher_dtype(training_args, student_model)
    if str(cfg.runtime.teacher_model_offload) == "none" and dp_device is not None and dp_device.type == "cuda":
        teacher = load_frozen_base_reference_model_distributed_from_hf_args(
            teacher_path,
            hf_args,
            device=dp_device,
            logger=log,
        )
        if next((p.dtype for p in teacher.parameters() if p.is_floating_point()), dtype) != dtype:
            teacher.to(dtype=dtype)
    else:
        teacher = get_model(teacher_path, _resolve_hf_credential(hf_args))
        teacher.to(device="cpu", dtype=dtype)
    teacher.requires_grad_(False)
    teacher.eval()
    if hasattr(getattr(teacher, "config", None), "use_cache"):
        teacher.config.use_cache = False
    identity = _model_identity(teacher, teacher_path)
    log.info(
        "Teacher ready: path=%s model_offload=%s output_offload=%s",
        teacher_path,
        cfg.runtime.teacher_model_offload,
        cfg.runtime.teacher_output_offload,
    )
    return teacher, identity


def _build_eval_args(cfg) -> SimpleNamespace:
    ev = cfg.runtime.evaluation
    return SimpleNamespace(
        eval_after_save=bool(ev.eval_after_save),
        eval_tasks=ev.eval_tasks,
        eval_num_fewshot=int(ev.eval_num_fewshot),
        eval_lm_batch_size=str(ev.eval_batch_size),
        eval_lm_limit=ev.eval_limit,
        eval_device=str(ev.eval_device),
        eval_hif4_act=bool(ev.eval_hif4_act),
        skip_ppl_eval=bool(ev.skip_ppl_eval),
        ppl_seqlen=int(ev.ppl_seqlen),
        ppl_limit=int(ev.ppl_limit),
        eval_prewarm_group_size=int(ev.eval_prewarm_group_size),
    )


def _finalize_decoders(selected: Sequence[Tuple[str, VAELinear]]) -> int:
    count = 0
    seen = set()
    for _name, module in selected:
        if id(module) in seen:
            continue
        seen.add(id(module))
        module.disable_trainable_decode()
        packed_decoder = getattr(module, "_parallel_stage_decoder", None)
        if isinstance(packed_decoder, nn.Module):
            packed_decoder.requires_grad_(False)
        else:
            for stage_idx in range(int(module.residual_stages)):
                for part_idx in range(int(module.parallel_parts)):
                    module.get_stage_part_decoder(
                        stage_idx=stage_idx,
                        part_idx=part_idx,
                    ).requires_grad_(False)
        module.clear_decoded_weight_cache()
        count += 1
    return count


def _cleanup_runtime(model: nn.Module, *, hook_handles, streaming_manager, hif4_handles=()) -> None:
    remove_hif4_act_hooks(tuple(hif4_handles or ()))
    if streaming_manager is not None:
        streaming_manager.offload_all(synchronize=True)
        unwrap_streaming_offload_layers(_unwrap_peft_root(model))
    for handle in hook_handles:
        handle.remove()
    clear_model_vae_linear_cache(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _assert_final_runtime_clean(model: nn.Module) -> None:
    if list(iter_named_peft_lora_layers(model)):
        raise RuntimeError("Final v6 model still contains PEFT LoRA layers.")
    if hasattr(model, "sparse_bit_tuning"):
        raise RuntimeError("Final v6 model still contains sparse_bit_tuning runtime module.")
    for name, module in model.named_modules():
        if not isinstance(module, VAELinear):
            continue
        if getattr(module, "_sparse_bit_binding", None) is not None:
            raise RuntimeError(f"{name}: Sparse Bit binding survived finalization.")
        if bool(getattr(module, "trainable_decode", False)):
            raise RuntimeError(f"{name}: trainable_decode survived finalization.")


def run(cfg, hf_args, training_args) -> Dict[str, object]:
    # Lazy import avoids a module cycle: orchestration consumes the runtime
    # primitives above, while this remains the single public E2E v6 entrypoint.
    from compressed_e2e_fintuning.runtime_v6_pipeline import run_pipeline

    return run_pipeline(cfg, hf_args, training_args)


__all__ = ["run"]
