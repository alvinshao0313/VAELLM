import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer, default_data_collator

from e2e_common.compressed_checkpoint import (
    get_decode_device_diagnostics,
    load_compressed_student_checkpoint,
    resolve_base_model_path,
)
from e2e_common.data import build_tokenizer
from e2e_common.e2e_args import needs_teacher
from e2e_common.compressed_subspace_lora import (
    CompressedSubspacePeftProxy,
    PeftZeroLinearCarrier,
    extract_subspace_peft_low_rank_payloads,
    initialize_subspace_peft_lora_from_low_rank,
    iter_named_compressed_subspace_peft_proxies,
    wrap_vae_linears_with_compressed_subspace_peft_proxy,
)
from e2e_common.low_rank_lora import (
    extract_low_rank_payloads_from_lora,
    iter_lora_target_modules,
    write_low_rank_payloads_to_compressed_model,
)
from e2e_common.peft_proxy import detach_and_clear_vae_low_rank_payloads, is_peft_lora_linear
from e2e_common.post_norm_head import ensure_post_norm_head_linear
from e2e_common.lazy_datasets import build_edgerazor_data_collator, dataset_length_or_none, default_dataloader_num_workers
from e2e_common.runtime_utils import build_datasets_with_main_process_first, eval_final_ppl
from litebsq.low_rank_scope import (
    LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
    LOW_RANK_SCOPE_FULL,
    normalize_low_rank_scope,
)
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from litebsq.vae_linear_prewarm import NamedVAELinearDecodeTarget, decode_named_vae_linear_weights
from rotation.model_utils import get_layers, get_model
from train_utils.model_checkpoint_io import (
    STATE_DICT_FILENAME,
    _build_distributed_run_output_dir,
    save_model_checkpoint,
)
from train_utils.utils import get_logger
from compressed_e2e_fintuning.device_map import apply_boundary_device_map, apply_layer_device_map, resolve_layer_device_map
from compressed_e2e_fintuning.mid_eval import EvalAfterSaveCallback, run_e2e_lm_eval
from compressed_e2e_fintuning.offload import (
    SavedTensorOffloadContext,
    unwrap_streaming_offload_layers,
    validate_streaming_layer_devices,
    wrap_model_layers_for_streaming_offload,
)
from train_utils.lora_utils import ensure_distill_process_group_initialized, is_distill_distributed
from compressed_e2e_fintuning.trainables import (
    VAEDecoderTrainableSelection,
    collect_selected_vae_linears,
    resolve_target_layer_ids,
    select_vae_decoder_trainables,
    unpack_parallel_stage_decoders,
    validate_selected_low_rank_payloads,
    validate_selected_low_rank_scope,
)
from compressed_e2e_fintuning.trainer import (
    E2ETrainerLogCallback,
    VAEDecoderE2ETrainer,
    replace_progress_log_callback,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    return str(value)


def _namespace_to_dict(ns) -> Dict[str, Any]:
    if ns is None:
        return {}
    if is_dataclass(ns):
        return {str(k): _jsonable(v) for k, v in asdict(ns).items()}
    if hasattr(ns, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(ns).items()}
    return {"value": _jsonable(ns)}


def _selection_to_meta(selection) -> Dict[str, object]:
    data = asdict(selection)
    data["target_module_count"] = len(selection.target_modules)
    data["decoder_layer_count"] = len(selection.decoder_layer_ids)
    return data


def _iter_named_vae_linears(model: nn.Module) -> Iterable[Tuple[str, VAELinear]]:
    for name, module in model.named_modules():
        if isinstance(module, VAELinear):
            yield str(name), module


def _set_vae_decoder_checkpoint(model: nn.Module, enabled: bool) -> Tuple[int, int]:
    changed = 0
    total = 0
    for _name, module in _iter_named_vae_linears(model):
        for stage_idx in range(int(module.residual_stages)):
            for part_idx in range(int(module.parallel_parts)):
                decoder = module.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)
                total += 1
                if bool(getattr(decoder, "use_checkpoint", False)) != bool(enabled):
                    decoder.use_checkpoint = bool(enabled)
                    changed += 1
    return changed, total


def _resolve_reference_dtype(module: nn.Module) -> torch.dtype:
    for param in module.parameters():
        if param.is_floating_point():
            return param.dtype
    for buffer in module.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return torch.float32


def _copy_low_rank_payloads(
    selected_modules: Sequence[Tuple[str, VAELinear]],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, module in selected_modules:
        low_rank_a = getattr(module, "low_rank_a", None)
        low_rank_b = getattr(module, "low_rank_b", None)
        if low_rank_a is None or low_rank_b is None:
            raise ValueError(f"{name}: selected VAELinear has no complete low_rank_a/low_rank_b payload.")
        payloads[str(name)] = (
            low_rank_a.detach().to(device="cpu").contiguous(),
            low_rank_b.detach().to(device="cpu").contiguous(),
        )
    return payloads


@torch.no_grad()
def _materialize_selected_vae_linears_without_low_rank(
    model: nn.Module,
    selected_modules: Sequence[Tuple[str, VAELinear]],
    *,
    group_size: int,
    compute_device: str,
    log,
) -> int:
    decode_targets = [
        NamedVAELinearDecodeTarget(
            name=name,
            base_layer=module,
            target_dtype=_resolve_reference_dtype(module),
            include_low_rank=False,
        )
        for name, module in selected_modules
    ]
    if not decode_targets:
        raise ValueError("No selected VAELinear modules to materialize.")
    log.info(
        "Start low-rank dense init: selected=%d group_size=%d compute_device=%s include_low_rank=false",
        len(decode_targets),
        int(group_size),
        str(compute_device),
    )
    decoded_results = decode_named_vae_linear_weights(
        decode_targets,
        group_size=int(group_size),
        compute_device=compute_device,
        logger=log,
        respect_cache_policy=False,
    )
    decoded_by_name = {item.name: item for item in decoded_results}
    if len(decoded_by_name) != len(selected_modules):
        raise RuntimeError(
            f"Low-rank dense init decode count mismatch: decoded={len(decoded_by_name)} expected={len(selected_modules)}."
        )

    converted = 0
    for name, old_module in selected_modules:
        decoded = decoded_by_name[str(name)]
        dense_linear = nn.Linear(
            int(old_module.in_features),
            int(old_module.out_features),
            bias=old_module.bias is not None,
            device=decoded.decoded_weight.device,
            dtype=decoded.decoded_weight.dtype,
        )
        dense_linear.weight.copy_(decoded.decoded_weight)
        if dense_linear.bias is not None and old_module.bias is not None:
            dense_linear.bias.copy_(
                old_module.bias.detach().to(device=dense_linear.bias.device, dtype=dense_linear.bias.dtype)
            )
        dense_linear.train(old_module.training)
        dense_linear.to("cpu")
        set_module_by_name(model, str(name), dense_linear)
        converted += 1
    log.info("Finished low-rank dense init: converted=%d", converted)
    return int(converted)


def _initialize_lora_from_low_rank(
    peft_model: nn.Module,
    low_rank_payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
) -> int:
    initialized = 0
    for base_name, module in iter_lora_target_modules(peft_model):
        if base_name not in low_rank_payloads:
            continue
        low_rank_a, low_rank_b = low_rank_payloads[base_name]
        lora_a = module.lora_A["default"].weight
        lora_b = module.lora_B["default"].weight
        if tuple(lora_a.shape) != tuple(low_rank_b.shape):
            raise RuntimeError(f"{base_name}: lora_A shape {tuple(lora_a.shape)} != low_rank_b {tuple(low_rank_b.shape)}.")
        if tuple(lora_b.shape) != tuple(low_rank_a.shape):
            raise RuntimeError(f"{base_name}: lora_B shape {tuple(lora_b.shape)} != low_rank_a {tuple(low_rank_a.shape)}.")
        lora_a.data.copy_(low_rank_b.to(device=lora_a.device, dtype=lora_a.dtype))
        lora_b.data.copy_(low_rank_a.to(device=lora_b.device, dtype=lora_b.dtype))
        initialized += 1
    if initialized != len(low_rank_payloads):
        raise RuntimeError(
            f"LoRA init target count mismatch: initialized={initialized} expected={len(low_rank_payloads)}."
        )
    return int(initialized)


def _build_low_rank_peft_model(
    model: nn.Module,
    *,
    low_rank_payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    rank: int,
    decoder_layer_ids: Sequence[int],
    target_module_suffixes: Sequence[str],
    parallel_stage_decode: bool,
    log,
) -> Tuple[nn.Module, VAEDecoderTrainableSelection]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover
        raise ImportError("compressed_lora 训练需要 peft。请先安装：pip install peft") from exc

    target_modules = sorted(low_rank_payloads.keys())
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(rank),
        target_modules=target_modules,
        lora_alpha=float(rank),
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=True,
    )
    peft_model = get_peft_model(model, peft_config)
    initialized = _initialize_lora_from_low_rank(peft_model, low_rank_payloads)
    trainable_names = sorted(name for name, param in peft_model.named_parameters() if bool(param.requires_grad))
    trainable_count = int(
        sum(int(param.numel()) for _name, param in peft_model.named_parameters() if bool(param.requires_grad))
    )
    if trainable_count < 1:
        raise RuntimeError("No trainable low-rank LoRA parameters found.")
    log.info("Initialized low-rank LoRA targets from checkpoint payloads: initialized=%d rank=%d", initialized, int(rank))
    selection = VAEDecoderTrainableSelection(
        decoder_layer_ids=[int(idx) for idx in decoder_layer_ids],
        target_modules=target_modules,
        target_module_suffixes=list(target_module_suffixes),
        bias_modules=[],
        final_norm_modules=[],
        post_norm_head_modules=[],
        low_rank_modules=target_modules,
        trainable_parameter_names=trainable_names,
        trainable_parameter_count=trainable_count,
        parallel_stage_decode=bool(parallel_stage_decode),
        train_mode="compressed_lora",
    )
    return peft_model, selection


def _build_subspace_low_rank_peft_model(
    model: nn.Module,
    *,
    selected_modules: Sequence[Tuple[str, VAELinear]],
    low_rank_payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    rank: int,
    decoder_layer_ids: Sequence[int],
    target_module_suffixes: Sequence[str],
    parallel_stage_decode: bool,
    log,
) -> Tuple[nn.Module, VAEDecoderTrainableSelection]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover
        raise ImportError("compressed_lora 训练需要 peft。请先安装：pip install peft") from exc

    selected_module_names = [str(name) for name, _module in selected_modules]
    if sorted(selected_module_names) != sorted(low_rank_payloads.keys()):
        raise RuntimeError(
            "subspace low-rank payload keys must match selected modules: "
            f"payloads={sorted(low_rank_payloads.keys())} selected={sorted(selected_module_names)}."
        )
    for name, module in selected_modules:
        scope = normalize_low_rank_scope(getattr(module, "low_rank_scope", LOW_RANK_SCOPE_FULL))
        if scope != LOW_RANK_SCOPE_COMPRESSED_SUBSPACE:
            raise ValueError(
                f"{name}: subspace PEFT path requires low_rank_scope={LOW_RANK_SCOPE_COMPRESSED_SUBSPACE!r}, "
                f"got {scope!r}."
            )

    detach_and_clear_vae_low_rank_payloads(selected_modules)
    wrap_vae_linears_with_compressed_subspace_peft_proxy(model, selected_modules)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(rank),
        target_modules=[CompressedSubspacePeftProxy.CARRIER_NAME],
        lora_alpha=float(rank),
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=True,
    )
    peft_model = get_peft_model(model, peft_config)

    proxy_refs = list(iter_named_compressed_subspace_peft_proxies(peft_model))
    if len(proxy_refs) != len(selected_modules):
        raise RuntimeError(
            f"subspace PEFT proxy count mismatch: proxies={len(proxy_refs)} selected={len(selected_modules)}."
        )
    for module_name, proxy in proxy_refs:
        carrier = getattr(proxy, CompressedSubspacePeftProxy.CARRIER_NAME, None)
        if not is_peft_lora_linear(carrier):
            raise TypeError(
                f"{module_name}: expected PEFT plain LoRA Linear after get_peft_model, got {type(carrier)}."
            )
        base_layer = carrier.base_layer
        if not isinstance(base_layer, PeftZeroLinearCarrier):
            raise TypeError(
                f"{module_name}: PEFT base_layer must remain PeftZeroLinearCarrier, got {type(base_layer)}."
            )
        if int(base_layer.weight.numel()) != 1:
            raise RuntimeError(
                f"{module_name}: carrier sentinel weight must have numel==1, got {int(base_layer.weight.numel())}."
            )
        if not torch.equal(
            base_layer.weight.detach().cpu().reshape(-1),
            torch.zeros(1, dtype=base_layer.weight.dtype),
        ):
            raise RuntimeError(f"{module_name}: carrier sentinel weight must stay identically zero.")
        if bool(base_layer.weight.requires_grad):
            raise RuntimeError(f"{module_name}: carrier sentinel weight must remain frozen.")

    initialized = initialize_subspace_peft_lora_from_low_rank(
        peft_model,
        low_rank_payloads,
        module_names=selected_module_names,
    )
    trainable_names = sorted(
        name for name, param in peft_model.named_parameters() if bool(param.requires_grad)
    )
    trainable_count = int(
        sum(int(param.numel()) for _name, param in peft_model.named_parameters() if bool(param.requires_grad))
    )
    if trainable_count < 1:
        raise RuntimeError("No trainable subspace LoRA parameters found.")
    expected_trainable = 0
    for _name, (low_rank_a, low_rank_b) in low_rank_payloads.items():
        expected_trainable += int(low_rank_a.numel()) + int(low_rank_b.numel())
    if int(trainable_count) != int(expected_trainable):
        raise RuntimeError(
            f"subspace trainable param count mismatch: got={trainable_count} expected={expected_trainable}."
        )
    log.info(
        "Initialized subspace PEFT LoRA from checkpoint payloads: initialized=%d rank=%d "
        "compressed_lora_scope=%s trainable_params=%d",
        initialized,
        int(rank),
        LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
        int(trainable_count),
    )
    target_modules = sorted(selected_module_names)
    selection = VAEDecoderTrainableSelection(
        decoder_layer_ids=[int(idx) for idx in decoder_layer_ids],
        target_modules=target_modules,
        target_module_suffixes=list(target_module_suffixes),
        bias_modules=[],
        final_norm_modules=[],
        post_norm_head_modules=[],
        low_rank_modules=target_modules,
        trainable_parameter_names=trainable_names,
        trainable_parameter_count=trainable_count,
        parallel_stage_decode=bool(parallel_stage_decode),
        train_mode="compressed_lora",
    )
    return peft_model, selection


def _prepare_compressed_lora_train_model(
    model: nn.Module,
    *,
    selected_modules: Sequence[Tuple[str, VAELinear]],
    target_module_suffixes: Sequence[str],
    low_rank_scope: str,
    low_rank_rank: int,
    low_rank_payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    decoder_layer_ids: Sequence[int],
    parallel_stage_decode: bool,
    decode_group_size: int,
    decode_device: str,
    log,
) -> Tuple[nn.Module, VAEDecoderTrainableSelection]:
    """Route compressed_lora setup by resolved low-rank scope."""
    if low_rank_scope == LOW_RANK_SCOPE_FULL:
        decode_device_diag = get_decode_device_diagnostics(str(decode_device))
        resolved_decode_device = str(decode_device_diag["resolved_device"])
        log.info(
            "Low-rank dense init decode config: requested_device=%s resolved_device=%s group_size=%d",
            str(decode_device),
            resolved_decode_device,
            int(decode_group_size),
        )
        log.info(
            "Low-rank decode device diagnostics: LOCAL_RANK=%s CUDA_VISIBLE_DEVICES=%s visible_cuda_count=%d",
            str(decode_device_diag["local_rank"]),
            str(decode_device_diag["cuda_visible_devices"]),
            int(decode_device_diag["visible_cuda_count"]),
        )
        _materialize_selected_vae_linears_without_low_rank(
            model,
            selected_modules,
            group_size=int(decode_group_size),
            compute_device=resolved_decode_device,
            log=log,
        )
        return _build_low_rank_peft_model(
            model,
            low_rank_payloads=low_rank_payloads,
            rank=int(low_rank_rank),
            decoder_layer_ids=decoder_layer_ids,
            target_module_suffixes=target_module_suffixes,
            parallel_stage_decode=bool(parallel_stage_decode),
            log=log,
        )
    if low_rank_scope == LOW_RANK_SCOPE_COMPRESSED_SUBSPACE:
        return _build_subspace_low_rank_peft_model(
            model,
            selected_modules=selected_modules,
            low_rank_payloads=low_rank_payloads,
            rank=int(low_rank_rank),
            decoder_layer_ids=decoder_layer_ids,
            target_module_suffixes=target_module_suffixes,
            parallel_stage_decode=bool(parallel_stage_decode),
            log=log,
        )
    raise AssertionError(f"Unreachable low_rank_scope={low_rank_scope!r}")


def _peft_base_model(model: nn.Module) -> nn.Module:
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        return get_base_model()
    return model


def _resolve_dp_local_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _ensure_torch_distributed_initialized(log) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    from train_utils.lora_utils import _resolve_distill_process_group_timeout_sec

    already = torch.distributed.is_available() and torch.distributed.is_initialized()
    timeout_sec = _resolve_distill_process_group_timeout_sec()
    ensure_distill_process_group_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    log.info(
        "%s torch.distributed for DP: world_size=%d rank=%s local_rank=%d pg_timeout_sec=%d "
        "(DISTILL_NCCL_TIMEOUT_SEC)",
        "Reconfirmed" if already else "Initialized",
        world_size,
        str(os.environ.get("RANK", "0")),
        local_rank,
        int(timeout_sec),
    )


def _load_teacher(*, args, hf_args, base_model_path: str, log, device: Optional[torch.device] = None):
    if not needs_teacher(args.loss_type) and float(getattr(args, "hidden_loss_weight", 0.0)) <= 0.0:
        return None, "disabled"
    teacher_path = str(args.teacher_model_path or base_model_path)
    log.info("Loading teacher model from %s", teacher_path)
    teacher_model = get_model(teacher_path, hf_args.access_token)
    teacher_model.eval()
    if hasattr(teacher_model, "config"):
        teacher_model.config.use_cache = False
    for param in teacher_model.parameters():
        param.requires_grad = False
    if device is not None:
        teacher_model.to(device)
        log.info("Moved teacher model to %s", device)
    return teacher_model, "external_teacher"


def _resolve_eval_device(requested_device: str) -> str:
    device = str(requested_device or "").strip()
    if not device:
        raise ValueError("--eval_device cannot be empty.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested eval_device={device}, but CUDA is not available.")
    return device


def _disable_trainable_decode_for_eval(model: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        disable_fn = getattr(module, "disable_trainable_decode", None)
        if callable(disable_fn):
            disable_fn()
            count += 1
    return count


@torch.no_grad()
def _prewarm_sparse_residual_cache(model: torch.nn.Module, *, dtype: Optional[torch.dtype], log) -> Dict[str, int]:
    from e2e_common.proxy_trainables import iter_named_vae_module_refs

    total = 0
    warmed = 0
    skipped = 0
    failed = 0
    for ref in iter_named_vae_module_refs(model):
        total += 1
        module = ref.base_layer
        has_sparse_fn = getattr(module, "has_sparse_residual", None)
        has_sparse = bool(has_sparse_fn()) if callable(has_sparse_fn) else False
        if not has_sparse:
            skipped += 1
            continue
        prime_fn = getattr(module, "prime_sparse_residual_cache", None)
        if not callable(prime_fn):
            failed += 1
            raise RuntimeError(f"{ref.name}: VAELinear has no prime_sparse_residual_cache method.")
        try:
            did_warm = bool(prime_fn(dtype=dtype))
        except Exception as exc:
            failed += 1
            raise RuntimeError(f"Sparse residual prewarm failed for '{ref.name}': {exc}") from exc
        if did_warm:
            warmed += 1
        else:
            skipped += 1

    log.info(
        "[sparse_residual_prewarm] done: total=%d warmed=%d skipped=%d failed=%d dtype=%s",
        int(total),
        int(warmed),
        int(skipped),
        int(failed),
        str(dtype),
    )
    return {
        "total": int(total),
        "warmed": int(warmed),
        "skipped": int(skipped),
        "failed": int(failed),
    }


def _resolve_train_sparse_residual_dtype(training_args) -> Optional[torch.dtype]:
    if bool(getattr(training_args, "bf16", False)):
        return torch.bfloat16
    if bool(getattr(training_args, "fp16", False)):
        return torch.float16
    return None


def _clear_vae_linear_cache(model: torch.nn.Module, log, *, reason: str) -> int:
    from litebsq.vae_linear import clear_model_vae_linear_cache

    cleared = clear_model_vae_linear_cache(model)
    log.info("%s: cleared decoded cache for %d VAELinear modules.", reason, int(cleared))
    return int(cleared)


def _release_trainer_cuda_state(trainer, log) -> None:
    """Drop optimizer/scheduler CUDA tensors before final eval/prewarm."""
    import gc

    released = []
    for attr_name in ("optimizer", "lr_scheduler", "scaler"):
        if getattr(trainer, attr_name, None) is not None:
            setattr(trainer, attr_name, None)
            released.append(attr_name)
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        for attr_name in ("optimizer", "lr_scheduler", "scaler"):
            if getattr(accelerator, attr_name, None) is not None:
                setattr(accelerator, attr_name, None)
                released.append(f"accelerator.{attr_name}")
        free_memory = getattr(accelerator, "free_memory", None)
        if callable(free_memory):
            try:
                free_memory()
            except TypeError:
                # Some accelerate versions require explicit args; best-effort only.
                pass
    if released:
        log.info("Released trainer CUDA training state: %s", ", ".join(released))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _park_model_on_cpu(model: torch.nn.Module, log, *, reason: str) -> None:
    import gc

    model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("%s: parked model on CPU and cleared CUDA cache.", reason)


def _prewarm_final_eval_model(*, model: torch.nn.Module, args, log) -> Dict[str, int]:
    device = _resolve_eval_device(str(args.eval_device))
    group_size = int(getattr(args, "eval_prewarm_group_size", 8))
    log.info("[eval_prewarm] Moving final model to %s ...", device)
    model.to(device)

    from e2e_common.proxy_trainables import iter_named_vae_module_refs
    from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache

    named_targets = [
        NamedVAELinearTarget(name=ref.name, base_layer=ref.base_layer)
        for ref in iter_named_vae_module_refs(model)
    ]
    stats = prime_named_vae_linear_cache(
        named_targets,
        clear_existing=True,
        group_size=group_size,
        compute_device=device,
        logger=log,
    )
    if int(stats.get("failed", 0)) > 0:
        raise RuntimeError(f"Final eval prewarm failed: {stats}")
    log.info(
        "[eval_prewarm] done: total=%d warmed=%d skipped=%d failed=%d group_size=%d device=%s",
        int(stats.get("total", 0)),
        int(stats.get("warmed", 0)),
        int(stats.get("skipped", 0)),
        int(stats.get("failed", 0)),
        group_size,
        device,
    )
    return stats


def _run_final_lm_eval(
    *,
    model,
    tokenizer,
    args,
    base_model_path: str,
    output_dir: str,
    log,
    parallel_mode: str,
):
    return run_e2e_lm_eval(
        model=model,
        tokenizer=tokenizer,
        args=args,
        base_model_path=str(base_model_path),
        output_dir=str(output_dir),
        log=log,
        eval_tag="final",
        # layer_mp keeps the existing device map; dp / multi-process moves to local rank device.
        move_to_device=str(parallel_mode).strip().lower() == "dp" or is_distill_distributed(),
        parallel_stage_decode=bool(args.parallel_stage_decode),
    )


def _build_run_meta(
    *,
    args,
    hf_args,
    training_args,
    data_info: Dict[str, Any],
    trainable_info: Dict[str, Any],
    teacher_source: str,
    resolved_student_checkpoint_dir: str,
    base_model_path: str,
    layer_device_map: Dict[str, str],
    offload_mode: str,
    global_step: int,
) -> Dict[str, Any]:
    stage = str(getattr(args, "e2e_stage", "compressed_e2e_fintuning"))
    args_key = str(getattr(args, "e2e_args_key", "compressed_e2e_args"))
    meta = {
        "format": "vae_decoder_e2e_run_meta",
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "teacher_source": str(teacher_source),
        "student_checkpoint_dir": str(resolved_student_checkpoint_dir),
        "base_model_path": str(base_model_path),
        "source_checkpoint_state_dict_file": STATE_DICT_FILENAME,
        "parallel_mode": str(getattr(args, "parallel_mode", "layer_mp")),
        "layer_device_map": dict(layer_device_map),
        "offload_mode": str(offload_mode),
        "global_step": int(global_step),
        "hf_args": _namespace_to_dict(hf_args),
        "training_args": _namespace_to_dict(training_args),
        "dataset": _jsonable(data_info),
        "trainables": _jsonable(trainable_info),
    }
    if hasattr(args, "finetune_mode"):
        meta["finetune_mode"] = str(getattr(args, "finetune_mode"))
        meta["internal_vae_train_mode"] = str(getattr(args, "vae_train_mode"))
    meta[args_key] = _namespace_to_dict(args)
    return meta


def run(args, hf_args, training_args):
    stage = str(getattr(args, "e2e_stage", "compressed_e2e_fintuning"))
    parallel_mode = str(getattr(args, "parallel_mode", "layer_mp")).strip().lower()
    # Temporary logger before run dir exists; replaced after output dir is ready.
    log = get_logger(stage)
    if parallel_mode == "dp":
        _ensure_torch_distributed_initialized(log)
    run_output_dir = _build_distributed_run_output_dir(
        args.run_root_dir,
        os.path.basename(args.student_checkpoint_dir),
    )
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, f"{stage}.log")
    log = get_logger(stage)
    resume_from_checkpoint = None if args.resume_from_checkpoint is None else str(args.resume_from_checkpoint).strip()

    log.info("Run output directory: %s", run_output_dir)
    log.info("Parallel mode: %s", parallel_mode)
    log.info("Resolved student checkpoint directory: %s", args.student_checkpoint_dir)
    log.info("Input VAE decoder e2e args:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2))

    model, meta, resolved_student_checkpoint_dir = load_compressed_student_checkpoint(
        args.student_checkpoint_dir,
        access_token=hf_args.access_token,
        logger=log,
    )
    base_model_path = resolve_base_model_path(meta, args.teacher_model_path)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if bool(args.use_post_norm_head_linear):
        ensure_post_norm_head_linear(model)

    decoder_ckpt_changed, decoder_ckpt_total = _set_vae_decoder_checkpoint(
        model,
        enabled=bool(args.vae_decoder_checkpoint),
    )
    log.info(
        "Applied VAE decoder checkpoint config: enabled=%s changed=%d total=%d",
        str(bool(args.vae_decoder_checkpoint)).lower(),
        int(decoder_ckpt_changed),
        int(decoder_ckpt_total),
    )

    layers = list(get_layers(model))
    decoder_layer_ids = resolve_target_layer_ids(args.decoder_layer_ids, len(layers))
    train_mode = str(getattr(args, "vae_train_mode", "decoder")).strip().lower()
    low_rank_payloads_for_export: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
    low_rank_scope: Optional[str] = None
    low_rank_rank: Optional[int] = None
    if train_mode in {"compressed_lora", "both"}:
        selected_modules, target_module_suffixes = collect_selected_vae_linears(
            model,
            decoder_layer_ids=decoder_layer_ids,
            target_module_names=args.target_module_names,
        )
        if not selected_modules:
            raise ValueError("No eligible VAELinear modules found for requested decoder_layers / target_modules.")
        if train_mode == "compressed_lora":
            # Keep current compressed_lora uniform-rank requirement unchanged.
            low_rank_rank = validate_selected_low_rank_payloads(
                selected_modules,
                require_uniform_rank=True,
            )
            low_rank_scope = validate_selected_low_rank_scope(selected_modules)
        else:
            # both: only enforce scope consistency; do not add uniform-rank.
            low_rank_scope = validate_selected_low_rank_scope(selected_modules)
        log.info("Resolved compressed low-rank scope: %s", low_rank_scope)

    if train_mode == "compressed_lora":
        low_rank_payloads = _copy_low_rank_payloads(selected_modules)
        model, selection = _prepare_compressed_lora_train_model(
            model,
            selected_modules=selected_modules,
            target_module_suffixes=target_module_suffixes,
            low_rank_scope=str(low_rank_scope),
            low_rank_rank=int(low_rank_rank),
            low_rank_payloads=low_rank_payloads,
            decoder_layer_ids=decoder_layer_ids,
            parallel_stage_decode=bool(args.parallel_stage_decode),
            decode_group_size=int(args.decode_group_size),
            decode_device=str(getattr(args, "decode_device", "auto")),
            log=log,
        )
    else:
        selection = select_vae_decoder_trainables(
            model,
            decoder_layer_ids=decoder_layer_ids,
            target_module_names=args.target_module_names,
            parallel_stage_decode=bool(args.parallel_stage_decode),
            tune_final_norm=bool(args.tune_final_norm),
            use_post_norm_head_linear=bool(args.use_post_norm_head_linear),
            vae_tune_bias=bool(args.vae_tune_bias),
            train_mode=train_mode,
        )
    log.info(
        "Selected VAE trainables: mode=%s layers=%s targets=%d suffixes=%s bias_modules=%d low_rank_modules=%d final_norm=%s post_norm_head=%s trainable_tensors=%d trainable_params=%d parallel_stage_decode=%s",
        selection.train_mode,
        selection.decoder_layer_ids,
        len(selection.target_modules),
        selection.target_module_suffixes,
        len(selection.bias_modules),
        len(selection.low_rank_modules),
        selection.final_norm_modules,
        selection.post_norm_head_modules,
        len(selection.trainable_parameter_names),
        int(selection.trainable_parameter_count),
        str(selection.parallel_stage_decode).lower(),
    )

    offload_mode = str(args.offload_mode).strip().lower()
    streaming_manager = None
    saved_tensor_offload = None
    hook_handles = []
    dp_local_device: Optional[torch.device] = None
    if parallel_mode == "dp":
        if str(args.layer_device_map).strip().lower() not in {"", "auto"}:
            log.info(
                "Ignoring --layer_device_map=%s under --parallel_mode dp; placing full model on local rank device.",
                args.layer_device_map,
            )
        dp_local_device = _resolve_dp_local_device()
        if dp_local_device.type == "cuda":
            torch.cuda.set_device(dp_local_device)
        model.to(dp_local_device)
        # Do not set model_parallel / is_parallelizable / hf_device_map so HF Trainer can wrap DDP.
        hf_device_map = {"all": str(dp_local_device)}
        log.info(
            "Applied DP device placement: device=%s offload_mode=%s",
            dp_local_device,
            offload_mode,
        )
        if offload_mode == "saved_tensors":
            saved_tensor_offload = SavedTensorOffloadContext(
                enabled=True,
                min_tensor_bytes=int(args.offload_min_tensor_bytes),
                pin_memory=bool(args.offload_pin_memory),
            )
    else:
        resolved_layer_device_map = resolve_layer_device_map(args.layer_device_map, len(layers))
        device_map_model = _peft_base_model(model) if train_mode == "compressed_lora" else model
        if offload_mode == "streaming":
            validate_streaming_layer_devices(resolved_layer_device_map)
            if bool(getattr(training_args, "gradient_checkpointing", False)):
                log.info(
                    "offload_mode=streaming manages layer checkpointing itself; overriding HF gradient_checkpointing=false."
                )
                training_args.gradient_checkpointing = False
            hook_handles, boundary_map = apply_boundary_device_map(
                device_map_model, layer_device_map=resolved_layer_device_map
            )
            streaming_manager, streaming_map = wrap_model_layers_for_streaming_offload(
                device_map_model,
                layer_devices=resolved_layer_device_map,
                prefetch_distance=int(args.offload_prefetch_distance),
                checkpoint_layers=bool(args.offload_checkpoint),
            )
            hf_device_map = {**boundary_map, **streaming_map}
        else:
            hook_handles, hf_device_map = apply_layer_device_map(
                device_map_model, layer_device_map=resolved_layer_device_map
            )
        if train_mode == "compressed_lora":
            setattr(model, "hf_device_map", hf_device_map)
            setattr(model, "is_parallelizable", True)
            setattr(model, "model_parallel", True)

        if offload_mode in {"saved_tensors", "streaming"}:
            saved_tensor_offload = SavedTensorOffloadContext(
                enabled=True,
                min_tensor_bytes=int(args.offload_min_tensor_bytes),
                pin_memory=bool(args.offload_pin_memory),
            )

        log.info(
            "Applied layer device/offload config: offload_mode=%s offload_checkpoint=%s map=%s saved_tensor_min_bytes=%d pin_memory=%s",
            offload_mode,
            str(bool(args.offload_checkpoint)).lower(),
            json.dumps(hf_device_map, ensure_ascii=False, sort_keys=True),
            int(args.offload_min_tensor_bytes),
            str(bool(args.offload_pin_memory)).lower(),
        )
    sparse_residual_prewarm = _prewarm_sparse_residual_cache(
        model,
        dtype=_resolve_train_sparse_residual_dtype(training_args),
        log=log,
    )

    tokenizer = build_tokenizer(str(base_model_path), access_token=hf_args.access_token)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset, eval_dataset, data_info = build_datasets_with_main_process_first(
        args,
        training_args,
        tokenizer,
        log,
    )
    train_len = dataset_length_or_none(train_dataset)
    if train_len is not None and train_len < 1:
        raise ValueError("Lazy training dataset is empty. Increase input text volume or lower --model_max_length.")
    eval_len = dataset_length_or_none(eval_dataset) if eval_dataset is not None else None
    if eval_len is not None and eval_len < 1:
        eval_dataset = None
        eval_len = None
    if int(getattr(training_args, "dataloader_num_workers", 0)) <= 0:
        training_args.dataloader_num_workers = int(default_dataloader_num_workers())
    training_args.dataloader_pin_memory = True
    if bool(data_info.get("lazy_iterable", False)):
        training_args.group_by_length = False
    dataset_task = str(data_info.get("dataset_task", "lm")).strip().lower()
    if dataset_task == "mcqa":
        data_collator = default_data_collator
    else:
        data_collator = build_edgerazor_data_collator(
            tokenizer,
            max_seq_len=int(data_info["block_size"]),
        )
    log.info(
        "Prepared datasets: train=%s eval=%s block_size=%d lazy_iterable=%s dataloader_num_workers=%d",
        "unknown" if train_len is None else str(train_len),
        "none" if eval_len is None else str(eval_len),
        int(data_info["block_size"]),
        str(bool(data_info.get("lazy_iterable", False))).lower(),
        int(training_args.dataloader_num_workers),
    )

    teacher_model, teacher_source = _load_teacher(
        args=args,
        hf_args=hf_args,
        base_model_path=str(base_model_path),
        log=log,
        device=dp_local_device if parallel_mode == "dp" else None,
    )
    log.info("Teacher source: %s", teacher_source)
    log.info(
        "Hidden alignment config: weight=%.6f layer_weighting=%s",
        float(getattr(args, "hidden_loss_weight", 0.0)),
        str(getattr(args, "hidden_layer_weighting", "uniform")),
    )
    log.info(
        "Teacher output config: offload=%s pin_memory=%s chunk_tokens=%d "
        "teacher_weight_offload=false",
        str(args.teacher_output_offload),
        str(bool(args.teacher_output_pin_memory)).lower(),
        int(args.teacher_output_chunk_tokens),
    )

    training_args.output_dir = os.path.join(run_output_dir, "trainer_state")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False
    if bool(getattr(training_args, "save_safetensors", False)):
        log.info("Disabling safetensors for Trainer checkpoints; VAE state_dict may contain shared storage.")
        training_args.save_safetensors = False

    eval_after_save_callback = None
    trainer_callbacks = [E2ETrainerLogCallback(logger=log)]
    if bool(getattr(args, "eval_after_save", False)):
        eval_after_save_callback = EvalAfterSaveCallback(
            e2e_args=args,
            tokenizer=tokenizer,
            base_model_path=str(base_model_path),
            run_output_dir=str(run_output_dir),
            log=log,
            parallel_stage_decode=bool(args.parallel_stage_decode),
            parallel_mode=str(parallel_mode),
        )
        trainer_callbacks.append(eval_after_save_callback)
        log.info(
            "Enabled eval-after-save: save_steps=%s eval_tasks=%s",
            str(getattr(training_args, "save_steps", None)),
            str(args.eval_tasks),
        )

    # Custom token-mean distill loss; disable HF num_items_in_batch loss scaling.
    model.accepts_loss_kwargs = False
    trainer = VAEDecoderE2ETrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        loss_type=args.loss_type,
        teacher_model=teacher_model,
        distill_temperature=args.distill_temperature,
        distill_alpha=args.distill_alpha,
        hidden_loss_weight=float(args.hidden_loss_weight),
        hidden_layer_weighting=str(args.hidden_layer_weighting),
        teacher_output_offload=str(args.teacher_output_offload),
        teacher_output_pin_memory=bool(args.teacher_output_pin_memory),
        teacher_output_chunk_tokens=int(args.teacher_output_chunk_tokens),
        eakld_confidence_k=int(args.eakld_confidence_k),
        saved_tensor_offload=saved_tensor_offload,
        streaming_offload_manager=streaming_manager,
        callbacks=trainer_callbacks,
    )
    replace_progress_log_callback(trainer)
    if eval_after_save_callback is not None:
        eval_after_save_callback.bind_trainer(trainer)
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)
    except Exception:
        if streaming_manager is not None:
            streaming_manager.offload_all(synchronize=True)
        for handle in hook_handles:
            handle.remove()
        raise

    final_model = trainer.accelerator.unwrap_model(trainer.model) if getattr(trainer, "accelerator", None) else trainer.model
    final_model.eval()
    if hasattr(trainer, "offload_teacher_to_cpu"):
        previous_teacher_device = trainer.offload_teacher_to_cpu()
        if previous_teacher_device is not None and previous_teacher_device.type != "cpu":
            log.info("Offloaded teacher to CPU after training (was %s).", previous_teacher_device)
    elif teacher_model is not None:
        teacher_model.to("cpu")
    _release_trainer_cuda_state(trainer, log)

    for handle in hook_handles:
        handle.remove()
    if streaming_manager is not None:
        streaming_manager.offload_all(synchronize=True)
        unwrap_target = _peft_base_model(final_model) if train_mode == "compressed_lora" else final_model
        unwrapped_streaming = unwrap_streaming_offload_layers(unwrap_target)
        log.info("Unwrapped %d streaming offload layers before final save.", unwrapped_streaming)
    if train_mode == "compressed_lora":
        if low_rank_scope == LOW_RANK_SCOPE_FULL:
            low_rank_payloads_for_export = extract_low_rank_payloads_from_lora(
                final_model,
                selection.target_modules,
            )
        elif low_rank_scope == LOW_RANK_SCOPE_COMPRESSED_SUBSPACE:
            low_rank_payloads_for_export = extract_subspace_peft_low_rank_payloads(
                final_model,
                module_names=selection.target_modules,
            )
        else:
            raise AssertionError(f"Unreachable low_rank_scope={low_rank_scope!r}")
        export_model, _export_meta, _export_resolved_dir = load_compressed_student_checkpoint(
            args.student_checkpoint_dir,
            access_token=hf_args.access_token,
            logger=log,
        )
        written = write_low_rank_payloads_to_compressed_model(
            export_model,
            low_rank_payloads_for_export,
            expected_scope=low_rank_scope,
        )
        log.info(
            "Exported trained LoRA payloads back to compressed low-rank branches: written=%d scope=%s",
            written,
            low_rank_scope,
        )
        final_model = export_model
        if hasattr(final_model, "config"):
            final_model.config.use_cache = False
            if getattr(final_model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
                final_model.config.pad_token_id = tokenizer.pad_token_id
    else:
        unpacked = unpack_parallel_stage_decoders(final_model)
        log.info("Unpacked %d parallel stage decoder modules before final save.", unpacked)
        disabled_decode = _disable_trainable_decode_for_eval(final_model)
        log.info("Disabled trainable decode mode on %d VAELinear modules before final save/eval.", disabled_decode)
    _clear_vae_linear_cache(final_model, log, reason="Final save/eval prep")

    lm_eval = None
    has_eval_tasks = bool(str(getattr(args, "eval_tasks", None) or "").strip())
    need_ppl_eval = not bool(getattr(args, "skip_ppl_eval", False))
    # DP（含 NPROC=1）：在模型落到 CPU 前先做 lm-eval，避免后续 prewarm 与残留训练状态抢显存。
    # WORLD_SIZE>1 时各 rank 仍走 mid_eval 内的分卡聚合路径。
    if has_eval_tasks and (parallel_mode == "dp" or is_distill_distributed()):
        local_device = dp_local_device if dp_local_device is not None else _resolve_dp_local_device()
        final_model.to(local_device)
        lm_eval = _run_final_lm_eval(
            model=final_model,
            tokenizer=tokenizer,
            args=args,
            base_model_path=str(base_model_path),
            output_dir=run_output_dir,
            log=log,
            parallel_mode=parallel_mode,
        )
    _park_model_on_cpu(final_model, log, reason="Before final save")
    if getattr(trainer, "model", None) is not None and trainer.model is not final_model:
        trainer.model.to("cpu")

    model_out = None
    run_meta_path = None
    ppl_eval = None
    eval_prewarm = None
    if bool(getattr(training_args, "should_save", True)):
        model_out = os.path.join(run_output_dir, "final_model")
        tok = None
        if bool(args.save_tokenizer):
            tok = AutoTokenizer.from_pretrained(str(base_model_path), use_fast=True, token=hf_args.access_token)
        save_paths = save_model_checkpoint(
            final_model,
            model_out,
            base_model_path=str(base_model_path),
            tokenizer=tok,
            save_config=True,
            extra_meta={
                "stage": stage,
                "finetune_mode": None if not hasattr(args, "finetune_mode") else str(getattr(args, "finetune_mode")),
                "vae_train_mode": str(train_mode),
                "tune_final_norm": bool(args.tune_final_norm),
                "use_post_norm_head_linear": bool(args.use_post_norm_head_linear),
                "vae_tune_bias": bool(args.vae_tune_bias),
                "sparse_residual_prewarm": sparse_residual_prewarm,
            },
            unload_vae_original_weights=False,
        )
        run_meta = _build_run_meta(
            args=args,
            hf_args=hf_args,
            training_args=training_args,
            data_info=data_info,
            trainable_info=_selection_to_meta(selection),
            teacher_source=teacher_source,
            resolved_student_checkpoint_dir=str(resolved_student_checkpoint_dir),
            base_model_path=str(base_model_path),
            layer_device_map=hf_device_map,
            offload_mode=offload_mode,
            global_step=int(getattr(trainer.state, "global_step", 0)),
        )
        run_meta_path = os.path.join(run_output_dir, "run_meta.json")
        with open(run_meta_path, "w", encoding="utf-8") as handle:
            json.dump(run_meta, handle, ensure_ascii=False, indent=2)
        log.info("Saved final compressed model to %s", save_paths["output_dir"])
        need_post_save_lm_eval = lm_eval is None and has_eval_tasks
        if need_ppl_eval or need_post_save_lm_eval:
            eval_prewarm = _prewarm_final_eval_model(
                model=final_model,
                args=args,
                log=log,
            )
            ppl_eval = eval_final_ppl(
                model=final_model,
                args=args,
                model_path=str(base_model_path),
                output_dir=run_output_dir,
                log=log,
            )
            if need_post_save_lm_eval:
                lm_eval = _run_final_lm_eval(
                    model=final_model,
                    tokenizer=tokenizer,
                    args=args,
                    base_model_path=str(base_model_path),
                    output_dir=run_output_dir,
                    log=log,
                    parallel_mode=parallel_mode,
                )
        else:
            log.info("Skipping final eval prewarm because PPL and post-save lm-eval are both disabled.")
    else:
        log.info("Skipping final model save on this rank because should_save=false.")

    return {
        "run_output_dir": run_output_dir,
        "saved_model_dir": model_out,
        "teacher_source": teacher_source,
        "final_ppl": None if ppl_eval is None else ppl_eval["result"],
        "final_ppl_path": None if ppl_eval is None else ppl_eval["path"],
        "final_lm_eval_path": None if lm_eval is None else lm_eval["json_path"],
        "final_lm_eval_summary_path": None if lm_eval is None else lm_eval["summary_path"],
        "final_eval_prewarm": eval_prewarm,
        "sparse_residual_prewarm": sparse_residual_prewarm,
        "run_meta_path": run_meta_path,
    }
