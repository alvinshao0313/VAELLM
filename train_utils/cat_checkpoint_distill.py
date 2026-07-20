import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn

from e2e_common.peft_proxy import PeftVAELinearProxy, iter_named_peft_vae_proxies
from e2e_common.post_norm_head import fuse_post_norm_head_linear
from e2e_common.temporary_switch_linear import TemporarySwitchLinear
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from litebsq.vae_linear_prewarm import (
    NamedVAELinearTarget,
    clear_model_vae_linear_cache,
)
from train_utils.cat_after_category_distill import (
    category_targets_have_low_rank,
    collect_compressed_category_targets,
    run_after_category_distill,
)
from train_utils.cat_train_args import resolve_category_runtime_configs
from train_utils.cat_train_eval import eval_after_category as _eval_after_category
from train_utils.cat_train_runtime import save_normalized_cat_train_snapshot as _save_normalized_cat_train_snapshot
from train_utils.lora_utils import (
    distill_distributed_barrier,
    ensure_distill_process_group_initialized,
    is_distill_distributed,
    is_distill_main_process,
)
from train_utils.model_checkpoint_io import (
    META_FILENAME,
    _build_distributed_run_output_dir,
    load_model_checkpoint,
    resolve_checkpoint_dir,
    save_model_checkpoint,
)
from train_utils.utils import (
    configure_deterministic_mode,
    format_namespace as _format_namespace,
    get_logger,
    set_seed,
    split_csv as _split_csv,
)


_CHECKPOINT_DISTILL_MODES = {"compressed_lora", "decoder", "both"}


@dataclass(frozen=True)
class _NamedCategoryVAETarget:
    name: str
    category: str
    module: nn.Module
    base_layer: VAELinear


@dataclass
class _CheckpointDistillResidency:
    stashed_modules: Dict[str, nn.Module] = field(default_factory=dict)
    original_weight_bank: Dict[str, nn.Parameter] = field(default_factory=dict)
    managed_shapes: Dict[str, Tuple[int, int]] = field(default_factory=dict)


def _category_of_module_name(name: str) -> str:
    return str(name).rsplit(".", 1)[-1]


def _resolve_vae_base_layer(module: nn.Module) -> VAELinear:
    if isinstance(module, PeftVAELinearProxy):
        return module.base_layer
    if isinstance(module, VAELinear):
        return module
    raise TypeError(f"Expected VAELinear or PeftVAELinearProxy, got {type(module)}")


def _iter_named_vae_targets(model: nn.Module) -> List[_NamedCategoryVAETarget]:
    targets: List[_NamedCategoryVAETarget] = []
    skip_prefixes: List[str] = []
    for name, module in model.named_modules():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in skip_prefixes):
            continue
        if isinstance(module, PeftVAELinearProxy):
            skip_prefixes.append(f"{name}.base_layer")
            skip_prefixes.append(f"{name}.per_decoded_linear")
            base_layer = module.base_layer
        elif isinstance(module, VAELinear):
            base_layer = module
        else:
            continue
        targets.append(
            _NamedCategoryVAETarget(
                name=str(name),
                category=str(name).rsplit(".", 1)[-1],
                module=module,
                base_layer=base_layer,
            )
        )
    return targets


def _detach_original_weight(base_layer: VAELinear) -> None:
    """Drop original_weight from VAE so .cpu() cannot move shared bank Parameter."""
    if getattr(base_layer, "original_weight", None) is None:
        return
    base_layer.register_parameter("original_weight", None)


def _bind_original_weight(base_layer: VAELinear, weight: nn.Parameter) -> None:
    weight.requires_grad = False
    base_layer.original_weight = weight


def _ensure_bank_param(
    residency: _CheckpointDistillResidency,
    *,
    name: str,
    source: torch.Tensor,
    device: torch.device,
) -> nn.Parameter:
    existing = residency.original_weight_bank.get(name)
    if existing is not None:
        if existing.device != device:
            existing.data = existing.data.to(device=device, non_blocking=True)
        existing.requires_grad = False
        return existing
    data = source.detach().to(device=device)
    param = nn.Parameter(data, requires_grad=False)
    residency.original_weight_bank[name] = param
    residency.managed_shapes[name] = (int(param.shape[0]), int(param.shape[1]))
    return param


def _stash_vae_module_to_cpu(
    *,
    name: str,
    module: nn.Module,
    residency: _CheckpointDistillResidency,
) -> None:
    base_layer = _resolve_vae_base_layer(module)
    base_layer.clear_decoded_weight_cache()
    _detach_original_weight(base_layer)
    module.to("cpu")
    residency.stashed_modules[name] = module


def _make_frozen_linear_from_bank(
    *,
    name: str,
    bank_weight: nn.Parameter,
    bias: Optional[torch.Tensor],
    in_features: int,
    out_features: int,
) -> nn.Linear:
    if tuple(bank_weight.shape) != (int(out_features), int(in_features)):
        raise ValueError(
            f"{name}: bank weight shape {tuple(bank_weight.shape)} != ({int(out_features)}, {int(in_features)})"
        )
    linear = nn.Linear(
        int(in_features),
        int(out_features),
        bias=bias is not None,
        device=bank_weight.device,
        dtype=bank_weight.dtype,
    )
    linear.weight = bank_weight
    if bias is not None:
        if tuple(bias.shape) != (int(out_features),):
            raise ValueError(f"{name}: bias shape {tuple(bias.shape)} != ({int(out_features)},)")
        bias_param = bias if isinstance(bias, nn.Parameter) else nn.Parameter(bias.detach().clone())
        bias_param.requires_grad = False
        if bias_param.device != bank_weight.device or bias_param.dtype != bank_weight.dtype:
            bias_param = nn.Parameter(
                bias_param.detach().to(device=bank_weight.device, dtype=bank_weight.dtype),
                requires_grad=False,
            )
        linear.bias = bias_param
    linear.requires_grad_(False)
    linear.eval()
    return linear


def _make_switch_linear_from_vae(
    *,
    name: str,
    base_layer: VAELinear,
    bank_weight: nn.Parameter,
    dtype: torch.dtype,
) -> TemporarySwitchLinear:
    del name  # used by callers for error context only
    compute_device = bank_weight.device
    sample_param = next(base_layer.parameters(), None)
    if sample_param is None or sample_param.device != compute_device:
        base_layer.to(compute_device)
    decoded = base_layer._decode_weight(dtype=dtype).detach().to(device=compute_device, dtype=dtype)
    student = nn.Parameter(decoded, requires_grad=False)
    bias = getattr(base_layer, "bias", None)
    bias_tensor = None if bias is None else bias.detach().to(device=compute_device, dtype=dtype)
    return TemporarySwitchLinear(
        in_features=int(base_layer.in_features),
        out_features=int(base_layer.out_features),
        student_weight=student,
        teacher_weight=bank_weight,
        bias=bias_tensor,
    )


def _ensure_managed_name_inventory(
    *,
    model: nn.Module,
    residency: _CheckpointDistillResidency,
    device: torch.device,
) -> List[str]:
    """Populate bank from live VAE targets and return all managed module names."""
    for target in _iter_named_vae_targets(model):
        original = getattr(target.base_layer, "original_weight", None)
        if original is None and target.name not in residency.original_weight_bank:
            raise RuntimeError(
                f"{target.name}: original_weight is required for checkpoint-distill residency bank."
            )
        if original is not None:
            bank_param = _ensure_bank_param(
                residency,
                name=target.name,
                source=original,
                device=device,
            )
            _bind_original_weight(target.base_layer, bank_param)
            residency.managed_shapes[target.name] = (
                int(target.base_layer.out_features),
                int(target.base_layer.in_features),
            )
        elif target.name in residency.original_weight_bank:
            _bind_original_weight(target.base_layer, residency.original_weight_bank[target.name])

    for name, module in list(residency.stashed_modules.items()):
        base_layer = _resolve_vae_base_layer(module)
        residency.managed_shapes.setdefault(
            name,
            (int(base_layer.out_features), int(base_layer.in_features)),
        )
        if name not in residency.original_weight_bank:
            original = getattr(base_layer, "original_weight", None)
            if original is None:
                raise RuntimeError(f"{name}: stashed VAELinear missing original_weight and bank entry.")
            _ensure_bank_param(residency, name=name, source=original, device=device)
            _detach_original_weight(base_layer)

    names: Set[str] = set(residency.original_weight_bank.keys())
    names.update(residency.stashed_modules.keys())
    names.update(residency.managed_shapes.keys())
    for name, module in model.named_modules():
        if isinstance(module, (VAELinear, PeftVAELinearProxy, TemporarySwitchLinear)):
            names.add(str(name))
    return sorted(names)


def _collect_vae_targets_by_category(model: nn.Module) -> Dict[str, List[_NamedCategoryVAETarget]]:
    by_category: Dict[str, List[_NamedCategoryVAETarget]] = {}
    for target in _iter_named_vae_targets(model):
        by_category.setdefault(target.category, []).append(target)
    return by_category


def _iter_vae_decoder_checkpoint_targets(base_layer: VAELinear):
    seen = set()
    for attr_name in ("decoder", "_parallel_stage_decoder"):
        decoder = getattr(base_layer, attr_name, None)
        if decoder is None or id(decoder) in seen:
            continue
        seen.add(id(decoder))
        yield decoder

    decoders = getattr(base_layer, "decoders", None)
    if decoders is None:
        return
    for decoder in decoders:
        if decoder is None or id(decoder) in seen:
            continue
        seen.add(id(decoder))
        yield decoder


def _apply_vae_decoder_checkpoint_override(*, model: nn.Module, vae_args, logger, mode: str) -> int:
    override = getattr(vae_args, "vae_decoder_checkpoint", None)
    if override is None:
        return 0

    resolved_mode = str(mode).strip().lower()
    if resolved_mode == "compressed_lora":
        logger.info(
            "Checkpoint distill: mode=compressed_lora，忽略 --vae_decoder_checkpoint（前向不跑 decoder）。"
        )
        return 0

    enabled = bool(override)
    changed = 0
    for target in _iter_named_vae_targets(model):
        for decoder in _iter_vae_decoder_checkpoint_targets(target.base_layer):
            if not hasattr(decoder, "use_checkpoint"):
                continue
            decoder.use_checkpoint = enabled
            changed += 1

    logger.info(
        "Checkpoint distill VAE decoder checkpoint override: use_checkpoint=%s decoders=%d",
        str(enabled).lower(),
        int(changed),
    )
    return changed


def _get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    module: nn.Module = model
    try:
        for part in str(name).split("."):
            module = getattr(module, part)
    except AttributeError:
        return None
    return module


def _materialize_name_to_switch_linear(
    *,
    model: nn.Module,
    name: str,
    residency: _CheckpointDistillResidency,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    module = _get_module_by_name(model, name)
    bank_weight = residency.original_weight_bank[name]
    if bank_weight.device != device:
        bank_weight.data = bank_weight.data.to(device=device, non_blocking=True)

    if isinstance(module, TemporarySwitchLinear):
        if module.teacher_weight is not bank_weight:
            module.register_parameter("teacher_weight", bank_weight)
        module.set_temporary(True)
        module.to(device)
        return

    if name in residency.stashed_modules:
        vae_module = residency.stashed_modules.pop(name)
        set_module_by_name(model, name, vae_module)
        module = vae_module

    if module is None:
        raise RuntimeError(f"{name}: cannot materialize completed category; module missing.")
    if not isinstance(module, (VAELinear, PeftVAELinearProxy)):
        raise TypeError(f"{name}: expected VAELinear to materialize, got {type(module)}")

    base_layer = _resolve_vae_base_layer(module)
    if getattr(base_layer, "original_weight", None) is None:
        _bind_original_weight(base_layer, bank_weight)
    switch = _make_switch_linear_from_vae(
        name=name,
        base_layer=base_layer,
        bank_weight=bank_weight,
        dtype=dtype,
    )
    _stash_vae_module_to_cpu(name=name, module=module, residency=residency)
    set_module_by_name(model, name, switch)


def _ensure_current_vae_on_model(
    *,
    model: nn.Module,
    name: str,
    residency: _CheckpointDistillResidency,
    device: torch.device,
) -> None:
    module = _get_module_by_name(model, name)
    if isinstance(module, (VAELinear, PeftVAELinearProxy)):
        base_layer = _resolve_vae_base_layer(module)
        _bind_original_weight(base_layer, residency.original_weight_bank[name])
        if callable(getattr(module, "set_temporary", None)):
            module.set_temporary(True)
        else:
            base_layer.set_temporary(True)
        base_layer.clear_decoded_weight_cache()
        module.to(device)
        return

    if name not in residency.stashed_modules:
        raise RuntimeError(f"{name}: active non-completed category missing VAELinear stash.")
    vae_module = residency.stashed_modules.pop(name)
    base_layer = _resolve_vae_base_layer(vae_module)
    _bind_original_weight(base_layer, residency.original_weight_bank[name])
    if callable(getattr(vae_module, "set_temporary", None)):
        vae_module.set_temporary(True)
    else:
        base_layer.set_temporary(True)
    base_layer.clear_decoded_weight_cache()
    vae_module.to(device)
    set_module_by_name(model, name, vae_module)


def _ensure_future_linear_on_model(
    *,
    model: nn.Module,
    name: str,
    residency: _CheckpointDistillResidency,
    device: torch.device,
) -> None:
    bank_weight = residency.original_weight_bank[name]
    if bank_weight.device != device:
        bank_weight.data = bank_weight.data.to(device=device, non_blocking=True)

    module = _get_module_by_name(model, name)
    out_features, in_features = residency.managed_shapes.get(name, (int(bank_weight.shape[0]), int(bank_weight.shape[1])))
    bias = None
    if isinstance(module, (VAELinear, PeftVAELinearProxy)):
        base_layer = _resolve_vae_base_layer(module)
        bias = getattr(base_layer, "bias", None)
        _stash_vae_module_to_cpu(name=name, module=module, residency=residency)
    elif isinstance(module, TemporarySwitchLinear):
        bias = module.bias
        if name not in residency.stashed_modules:
            raise RuntimeError(
                f"{name}: TemporarySwitchLinear leaving active prefix without stashed VAELinear."
            )
    elif isinstance(module, nn.Linear) and module.weight is bank_weight:
        module.to(device)
        return

    linear = _make_frozen_linear_from_bank(
        name=name,
        bank_weight=bank_weight,
        bias=None if bias is None else bias.detach(),
        in_features=int(in_features),
        out_features=int(out_features),
    )
    set_module_by_name(model, name, linear)


def _apply_checkpoint_distill_residency(
    *,
    model: nn.Module,
    active_categories: Sequence[str],
    completed_categories: Sequence[str],
    residency: _CheckpointDistillResidency,
    device: torch.device,
    dtype: torch.dtype,
    logger,
) -> None:
    active_set = {str(category) for category in active_categories}
    completed_set = {str(category) for category in completed_categories}
    managed_names = _ensure_managed_name_inventory(model=model, residency=residency, device=device)

    dual_count = 0
    active_vae = 0
    future_count = 0
    for name in managed_names:
        category = _category_of_module_name(name)
        if category in active_set and category in completed_set:
            _materialize_name_to_switch_linear(
                model=model,
                name=name,
                residency=residency,
                device=device,
                dtype=dtype,
            )
            dual_count += 1
        elif category in active_set:
            _ensure_current_vae_on_model(
                model=model,
                name=name,
                residency=residency,
                device=device,
            )
            active_vae += 1
        else:
            _ensure_future_linear_on_model(
                model=model,
                name=name,
                residency=residency,
                device=device,
            )
            future_count += 1

    logger.info(
        "Checkpoint distill residency: active_categories=%s completed=%s "
        "switch_linear=%d active_vae=%d future_linear=%d total_stashed=%d bank=%d",
        ",".join(str(category) for category in active_categories),
        ",".join(str(category) for category in completed_categories) if completed_categories else "(none)",
        int(dual_count),
        int(active_vae),
        int(future_count),
        int(len(residency.stashed_modules)),
        int(len(residency.original_weight_bank)),
    )


def _restore_checkpoint_distill_residency(
    *,
    model: nn.Module,
    residency: _CheckpointDistillResidency,
    logger,
) -> None:
    restored = 0
    for name, module in list(residency.stashed_modules.items()):
        base_layer = _resolve_vae_base_layer(module)
        base_layer.clear_decoded_weight_cache()
        bank_weight = residency.original_weight_bank.get(name)
        if bank_weight is None:
            raise RuntimeError(f"{name}: missing original_weight_bank entry during residency restore.")
        _bind_original_weight(base_layer, bank_weight)
        set_module_by_name(model, name, module)
        del residency.stashed_modules[name]
        restored += 1
    logger.info("Checkpoint distill residency: restored stashed VAELinear modules=%d", int(restored))


def _resolve_residency_device(train_device: str) -> torch.device:
    device_text = str(train_device).strip() or "cuda"
    if device_text.startswith("cuda") and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device_text == "cuda" or device_text == "cuda:":
            return torch.device(f"cuda:{local_rank}")
        return torch.device(device_text)
    return torch.device(device_text)


def _resolve_residency_dtype(*, bf16: bool, fp16: bool) -> torch.dtype:
    if bool(bf16):
        return torch.bfloat16
    if bool(fp16):
        return torch.float16
    return torch.float32


def _set_active_vae_category_prefix(
    *,
    model: nn.Module,
    active_categories: Sequence[str],
    logger,
) -> List[NamedVAELinearTarget]:
    active_set = {str(category) for category in active_categories}
    prewarm_targets: List[NamedVAELinearTarget] = []
    missing_original: List[str] = []
    compressed_count = 0
    original_count = 0
    switch_count = 0

    for name, module in model.named_modules():
        if isinstance(module, TemporarySwitchLinear):
            module.set_temporary(True)
            switch_count += 1

    for target in _iter_named_vae_targets(model):
        active = target.category in active_set
        if callable(getattr(target.module, "set_temporary", None)):
            target.module.set_temporary(active)
        else:
            target.base_layer.set_temporary(active)
        target.base_layer.clear_decoded_weight_cache()

        use_original = bool(getattr(target.base_layer, "always_use_original", False)) or not bool(
            getattr(target.base_layer, "temporary", True)
        )
        if use_original:
            original_count += 1
            if getattr(target.base_layer, "original_weight", None) is None:
                missing_original.append(target.name)
        else:
            compressed_count += 1
        if active:
            prewarm_targets.append(NamedVAELinearTarget(name=target.name, base_layer=target.base_layer))

    if missing_original:
        raise RuntimeError(
            "Original-weight path requested but original_weight is missing for: "
            + ", ".join(missing_original)
        )
    logger.info(
        "Checkpoint distill active categories=%s compressed_vae=%d original_vae=%d switch_linear=%d",
        ",".join(str(category) for category in active_categories),
        int(compressed_count),
        int(original_count),
        int(switch_count),
    )
    return prewarm_targets


def _load_checkpoint_for_distill(*, cat_args, hf_args, vae_args, logger) -> nn.Module:
    checkpoint_dir = resolve_checkpoint_dir(str(cat_args.resume_from_checkpoint))
    meta_path = os.path.join(checkpoint_dir, META_FILENAME)
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    base_model_path = meta.get("base_model_path") or getattr(vae_args, "model_path", None)
    if not base_model_path:
        raise ValueError(
            f"Cannot determine base model path for checkpoint: {checkpoint_dir}. "
            "Please save checkpoints with base_model_path metadata or pass --model_path."
        )

    logger.info("Loading VAE checkpoint for distill: %s", checkpoint_dir)
    logger.info("Checkpoint distill base model path: %s", str(base_model_path))
    model, load_meta, load_result = load_model_checkpoint(
        checkpoint_dir,
        access_token=hf_args.access_token,
        base_model_path=str(base_model_path),
        map_location="cpu",
        strict=True,
        preserve_original_weights_from_base=True,
    )
    vae_args.model_path = str(load_meta.get("base_model_path") or base_model_path)
    logger.info(
        "Checkpoint loaded for distill. missing_keys=%d unexpected_keys=%d converted_module_count=%s",
        len(getattr(load_result, "missing_keys", [])),
        len(getattr(load_result, "unexpected_keys", [])),
        str(load_meta.get("converted_module_count")),
    )
    return model


def _load_completed_categories_from_checkpoint(checkpoint_dir: str) -> List[str]:
    meta_path = os.path.join(checkpoint_dir, META_FILENAME)
    if not os.path.exists(meta_path):
        return []
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    extra_meta = meta.get("extra_meta") if isinstance(meta, dict) else None
    if not isinstance(extra_meta, dict):
        return []
    raw = extra_meta.get("completed_categories")
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _save_distill_model(
    *,
    model: nn.Module,
    model_out: str,
    cat_args,
    hf_args,
    vae_args,
    logger,
    extra_meta: Optional[dict],
    unload_vae_original_weights: bool,
) -> None:
    if not bool(getattr(cat_args, "convert", False)):
        raise ValueError("--save_model requires --convert")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(vae_args.model_path, use_fast=True, token=hf_args.access_token)
    fused_post_norm_head = fuse_post_norm_head_linear(model)
    if fused_post_norm_head:
        logger.info("Save: fused post_norm_linear into lm_head.weight.")
    leftover_proxies = [name for name, _proxy in iter_named_peft_vae_proxies(model)]
    if leftover_proxies:
        raise RuntimeError(
            "Save found unexported PeftVAELinearProxy modules: "
            + ", ".join(leftover_proxies)
        )
    cleared = clear_model_vae_linear_cache(model)
    logger.info("Save: cleared decoded cache for %d VAELinear modules.", cleared)
    save_paths = save_model_checkpoint(
        model,
        model_out,
        base_model_path=vae_args.model_path,
        tokenizer=tok,
        save_config=True,
        extra_meta=extra_meta,
        unload_vae_original_weights=bool(unload_vae_original_weights),
    )
    logger.info("Saved model to %s", save_paths["output_dir"])


def _save_final_model(
    *,
    model: nn.Module,
    run_output_dir: str,
    cat_args,
    hf_args,
    vae_args,
    logger,
    completed_categories: Optional[Sequence[str]] = None,
) -> None:
    _save_distill_model(
        model=model,
        model_out=os.path.join(run_output_dir, "final_model"),
        cat_args=cat_args,
        hf_args=hf_args,
        vae_args=vae_args,
        logger=logger,
        extra_meta={
            "stage": "final",
            "completed_categories": [str(item) for item in (completed_categories or [])],
        },
        unload_vae_original_weights=bool(cat_args.unload_vae_original_weights_on_final_save),
    )


def _save_after_category_checkpoint(
    *,
    model: nn.Module,
    run_output_dir: str,
    category: str,
    completed_categories: Sequence[str],
    mode: str,
    active_categories: Sequence[str],
    residency: _CheckpointDistillResidency,
    cat_args,
    hf_args,
    vae_args,
    training_args,
    logger,
) -> None:
    # All ranks restore so DDP model graphs stay aligned; only rank0 writes.
    _restore_checkpoint_distill_residency(model=model, residency=residency, logger=logger)
    distill_distributed_barrier()
    try:
        if is_distill_main_process():
            _save_distill_model(
                model=model,
                model_out=os.path.join(run_output_dir, f"after_{category}"),
                cat_args=cat_args,
                hf_args=hf_args,
                vae_args=vae_args,
                logger=logger,
                extra_meta={
                    "stage": "after_category",
                    "category": str(category),
                    "completed_categories": [str(item) for item in completed_categories],
                    "distill_after_category": str(mode),
                },
                unload_vae_original_weights=False,
            )
    finally:
        distill_distributed_barrier()
        residency_device = _resolve_residency_device(str(getattr(cat_args, "train_device", "cuda")))
        residency_dtype = _resolve_residency_dtype(
            bf16=bool(getattr(training_args, "bf16", False)),
            fp16=bool(getattr(training_args, "fp16", False)),
        )
        _apply_checkpoint_distill_residency(
            model=model,
            active_categories=active_categories,
            completed_categories=completed_categories,
            residency=residency,
            device=residency_device,
            dtype=residency_dtype,
            logger=logger,
        )
        _set_active_vae_category_prefix(
            model=model,
            active_categories=active_categories,
            logger=logger,
        )


def run_cat_checkpoint_distill(*, cat_args, hf_args, training_args, vae_args) -> None:
    mode = str(getattr(cat_args, "distill_after_category", "none")).strip().lower()
    if mode not in _CHECKPOINT_DISTILL_MODES:
        raise ValueError(
            "cat checkpoint distill only supports --distill_after_category=compressed_lora, decoder, or both."
        )
    if not str(getattr(cat_args, "resume_from_checkpoint", "") or "").strip():
        raise ValueError("--resume_from_checkpoint is required for cat checkpoint distill.")
    if bool(getattr(training_args, "distill_hif4_act", False)) and mode == "none":
        raise ValueError("--distill_hif4_act 仅在每类后蒸馏阶段生效，因此必须设置 --distill_after_category。")
    if not bool(getattr(cat_args, "convert", False)):
        raise ValueError("--distill_after_category requires --convert，因为每类后蒸馏必须作用在已替换的压缩模型上。")

    configure_deterministic_mode(bool(getattr(cat_args, "deterministic", False)))
    set_seed(cat_args.seed)
    ensure_distill_process_group_initialized()
    if not is_distill_main_process():
        os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

    os.makedirs(cat_args.output_dir, exist_ok=True)
    run_output_dir = _build_distributed_run_output_dir(cat_args.output_dir, vae_args.model_path)
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "linear_by_category.log")
    logger = get_logger("linear_by_category")
    cat_args.output_dir = run_output_dir

    logger.info("Run output directory: %s", run_output_dir)
    if is_distill_distributed():
        logger.info(
            "Checkpoint distill distributed mode: world_size=%d local_rank=%s",
            int(os.environ.get("WORLD_SIZE", "1")),
            str(os.environ.get("LOCAL_RANK", "0")),
        )
    if bool(getattr(cat_args, "deterministic", False)):
        logger.info("Deterministic mode enabled: torch deterministic algorithms on, TF32 disabled.")
    logger.info(
        "Args:\nscript=%s\nvae=%s\ntraining=%s",
        _format_namespace(cat_args),
        _format_namespace(vae_args),
        _format_namespace(training_args),
    )

    model = _load_checkpoint_for_distill(cat_args=cat_args, hf_args=hf_args, vae_args=vae_args, logger=logger)
    _apply_vae_decoder_checkpoint_override(model=model, vae_args=vae_args, logger=logger, mode=mode)
    transpose_modules = _split_csv(cat_args.transpose_modules)
    target_categories = _split_csv(cat_args.target_categories)
    only_decoder_projections = not bool(cat_args.include_all_linears)
    eval_tasks_text = str(getattr(cat_args, "eval_tasks", "")).strip()
    run_task_eval = bool(eval_tasks_text)
    run_category_eval = bool(cat_args.eval_ppl) or run_task_eval
    if not run_category_eval:
        logger.info("跳过类别后评估：--eval_ppl=false 且 --eval_tasks 为空。")

    targets_by_category = _collect_vae_targets_by_category(model)
    missing_target_categories = [
        category for category in target_categories if category not in targets_by_category
    ]
    if missing_target_categories:
        raise ValueError(
            "target_categories contains categories without VAELinear in checkpoint: "
            + ",".join(missing_target_categories)
        )

    resume_checkpoint_dir = resolve_checkpoint_dir(str(cat_args.resume_from_checkpoint))
    completed_categories = _load_completed_categories_from_checkpoint(resume_checkpoint_dir)
    if bool(getattr(cat_args, "distill_reset_completed", False)):
        if completed_categories:
            logger.info(
                "Checkpoint distill: --distill_reset_completed=true，忽略 resume ckpt 中的 completed_categories=%s",
                ",".join(completed_categories),
            )
        else:
            logger.info("Checkpoint distill: --distill_reset_completed=true，resume ckpt 无 completed_categories。")
        completed_categories = []
    elif completed_categories:
        logger.info(
            "Checkpoint distill resume progress: completed_categories=%s",
            ",".join(completed_categories),
        )

    resolved_category_cfgs = resolve_category_runtime_configs(cat_args, vae_args, target_categories)
    snapshot_path = _save_normalized_cat_train_snapshot(
        run_output_dir=run_output_dir,
        cat_args=cat_args,
        vae_args=vae_args,
        training_args=training_args,
        resolved_category_cfgs=resolved_category_cfgs,
    )
    logger.info("Saved normalized parameter snapshot: %s", snapshot_path)

    eval_tokenizer = None
    if run_task_eval:
        from transformers import AutoTokenizer

        if is_distill_main_process():
            logger.info("加载类别后下游任务评估 tokenizer: %s", vae_args.model_path)
        eval_tokenizer = AutoTokenizer.from_pretrained(
            vae_args.model_path,
            use_fast=True,
            token=hf_args.access_token,
        )

    residency = _CheckpointDistillResidency()
    residency_device = _resolve_residency_device(str(getattr(cat_args, "train_device", "cuda")))
    residency_dtype = _resolve_residency_dtype(
        bf16=bool(getattr(training_args, "bf16", False)),
        fp16=bool(getattr(training_args, "fp16", False)),
    )
    lora_round_idx = 0
    active_categories: List[str] = []
    completed_categories = list(completed_categories)
    for category in target_categories:
        active_categories.append(str(category))
        _apply_checkpoint_distill_residency(
            model=model,
            active_categories=active_categories,
            completed_categories=completed_categories,
            residency=residency,
            device=residency_device,
            dtype=residency_dtype,
            logger=logger,
        )
        prewarm_targets = _set_active_vae_category_prefix(
            model=model,
            active_categories=active_categories,
            logger=logger,
        )
        del prewarm_targets  # outer prewarm removed (O5); inner path in after-category distill handles cache

        skip_from_progress = str(category) in set(completed_categories)
        if skip_from_progress:
            logger.info(
                "Checkpoint distill: category=%s 已在 completed_categories 中，跳过蒸馏（已物化为 TemporarySwitchLinear）。",
                str(category),
            )
            lora_round_idx = int(lora_round_idx) + 1
            continue

        if run_category_eval:
            if is_distill_main_process():
                logger.info("每类后蒸馏前评估...")
            _eval_after_category(
                model=model,
                vae_args=vae_args,
                ppl_limit=cat_args.ppl_limit,
                category=category,
                logger=logger,
                eval_device=cat_args.train_device,
                eval_hif4_act=cat_args.eval_hif4_act,
                eval_ppl=cat_args.eval_ppl,
                eval_tasks=eval_tasks_text,
                tokenizer=eval_tokenizer,
            )

        distill_result = run_after_category_distill(
            model=model,
            category=category,
            cat_args=cat_args,
            vae_args=vae_args,
            training_args=training_args,
            logger=logger,
            lora_round_idx=lora_round_idx,
            transpose_modules=transpose_modules,
            only_decoder_projections=only_decoder_projections,
            target_categories=target_categories,
        )
        model = distill_result.model
        lora_round_idx = int(distill_result.next_lora_round_idx)

        if int(distill_result.trained_target_count) > 0:
            if str(category) not in completed_categories:
                completed_categories.append(str(category))
            # Materialize this category to TemporarySwitchLinear before save/eval.
            _apply_checkpoint_distill_residency(
                model=model,
                active_categories=active_categories,
                completed_categories=completed_categories,
                residency=residency,
                device=residency_device,
                dtype=residency_dtype,
                logger=logger,
            )
            if bool(cat_args.save_model):
                _save_after_category_checkpoint(
                    model=model,
                    run_output_dir=run_output_dir,
                    category=str(category),
                    completed_categories=completed_categories,
                    mode=mode,
                    active_categories=active_categories,
                    residency=residency,
                    cat_args=cat_args,
                    hf_args=hf_args,
                    vae_args=vae_args,
                    training_args=training_args,
                    logger=logger,
                )
        elif mode in {"compressed_lora", "both"}:
            if category_targets_have_low_rank(collect_compressed_category_targets(model, category)):
                if str(category) not in completed_categories:
                    completed_categories.append(str(category))
                    logger.info(
                        "Checkpoint distill: category=%s 因已有 low_rank_a/b 记入 completed_categories。",
                        str(category),
                    )
                _apply_checkpoint_distill_residency(
                    model=model,
                    active_categories=active_categories,
                    completed_categories=completed_categories,
                    residency=residency,
                    device=residency_device,
                    dtype=residency_dtype,
                    logger=logger,
                )

        if run_category_eval:
            if is_distill_main_process():
                logger.info("每类后蒸馏后评估...")
            _eval_after_category(
                model=model,
                vae_args=vae_args,
                ppl_limit=cat_args.ppl_limit,
                category=category,
                logger=logger,
                eval_device=cat_args.train_device,
                eval_hif4_act=cat_args.eval_hif4_act,
                eval_ppl=cat_args.eval_ppl,
                eval_tasks=eval_tasks_text,
                tokenizer=eval_tokenizer,
            )

    if run_category_eval:
        if is_distill_main_process():
            logger.info("所有类别蒸馏完成后最终评估...")
        _eval_after_category(
            model=model,
            vae_args=vae_args,
            ppl_limit=cat_args.ppl_limit,
            category="none",
            logger=logger,
            eval_device=cat_args.train_device,
            eval_hif4_act=cat_args.eval_hif4_act,
            eval_ppl=cat_args.eval_ppl,
            eval_tasks=eval_tasks_text,
            tokenizer=eval_tokenizer,
        )
    if cat_args.save_model and is_distill_main_process():
        _restore_checkpoint_distill_residency(model=model, residency=residency, logger=logger)
        _save_final_model(
            model=model,
            run_output_dir=run_output_dir,
            cat_args=cat_args,
            hf_args=hf_args,
            vae_args=vae_args,
            logger=logger,
            completed_categories=completed_categories,
        )

    distill_distributed_barrier()
    logger.info("Done.")
