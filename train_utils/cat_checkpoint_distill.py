import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn

from e2e_common.peft_proxy import PeftVAELinearProxy, iter_named_peft_vae_proxies
from e2e_common.post_norm_head import fuse_post_norm_head_linear
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from litebsq.vae_linear_prewarm import (
    NamedVAELinearTarget,
    clear_model_vae_linear_cache,
)
from train_utils.cat_after_category_distill import (
    collect_compressed_category_targets,
    run_after_category_distill,
)
from train_utils.cat_arg_overrides import resolve_after_category_value
from train_utils.cat_train_args import resolve_category_runtime_configs
from train_utils.cat_train_eval import eval_after_category as _eval_after_category
from train_utils.cat_train_runtime import (
    normalize_cat_runtime_vae_original_state,
    save_normalized_cat_train_snapshot as _save_normalized_cat_train_snapshot,
)
from train_utils.base_reference import clone_frozen_linear_from_reference
from train_utils.distill_teacher import DistillTeacherRuntime, resolve_distill_teacher_dtype
from train_utils.lora_utils import (
    distill_distributed_barrier,
    ensure_distill_process_group_initialized,
    is_distill_distributed,
    is_distill_main_process,
    resolve_distill_train_device,
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
    stashed_vae_modules: Dict[str, nn.Module] = field(default_factory=dict)
    reference_dense_linears: Dict[str, nn.Linear] = field(default_factory=dict)
    managed_categories: Dict[str, str] = field(default_factory=dict)


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


def _stash_vae_module_to_cpu(
    *,
    name: str,
    module: nn.Module,
    residency: _CheckpointDistillResidency,
) -> None:
    base_layer = _resolve_vae_base_layer(module)
    base_layer.clear_decoded_weight_cache()
    module.to("cpu")
    residency.stashed_vae_modules[name] = module


def _ensure_managed_name_inventory(
    *,
    model: nn.Module,
    residency: _CheckpointDistillResidency,
) -> List[str]:
    for target in _iter_named_vae_targets(model):
        residency.managed_categories.setdefault(str(target.name), str(target.category))
    names = set(residency.managed_categories.keys())
    names.update(residency.stashed_vae_modules.keys())
    names.update(residency.reference_dense_linears.keys())
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


def _get_or_create_reference_linear(
    *,
    teacher_runtime,
    name: str,
    residency: _CheckpointDistillResidency,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Linear:
    existing = residency.reference_dense_linears.get(name)
    if existing is not None:
        existing.to(device=device, dtype=dtype)
        existing.requires_grad_(False)
        existing.eval()
        return existing
    if teacher_runtime is None:
        raise RuntimeError("checkpoint-distill residency requires independent teacher_runtime for reference clones.")
    reference_model = teacher_runtime.get_or_load()
    clone = clone_frozen_linear_from_reference(
        reference_model,
        name,
        device=device,
        dtype=dtype,
    )
    residency.reference_dense_linears[name] = clone
    return clone


def _ensure_active_vae(
    *,
    model: nn.Module,
    name: str,
    residency: _CheckpointDistillResidency,
    device: torch.device,
) -> None:
    module = _get_module_by_name(model, name)
    if isinstance(module, (VAELinear, PeftVAELinearProxy)):
        base_layer = _resolve_vae_base_layer(module)
        base_layer.clear_decoded_weight_cache()
        module.to(device)
        return

    if not isinstance(module, nn.Linear):
        raise TypeError(f"{name}: expected active VAELinear or reference nn.Linear, got {type(module)}.")
    if name not in residency.stashed_vae_modules:
        raise RuntimeError(f"{name}: active category missing stashed VAELinear.")
    vae_module = residency.stashed_vae_modules.pop(name)
    base_layer = _resolve_vae_base_layer(vae_module)
    base_layer.clear_decoded_weight_cache()
    vae_module.to(device)
    set_module_by_name(model, name, vae_module)
    module.to("cpu")


def _ensure_inactive_reference_linear(
    *,
    model: nn.Module,
    name: str,
    residency: _CheckpointDistillResidency,
    teacher_runtime,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    module = _get_module_by_name(model, name)
    if isinstance(module, (VAELinear, PeftVAELinearProxy)):
        _stash_vae_module_to_cpu(name=name, module=module, residency=residency)
    elif isinstance(module, nn.Linear):
        reference = _get_or_create_reference_linear(
            teacher_runtime=teacher_runtime,
            name=name,
            residency=residency,
            device=device,
            dtype=dtype,
        )
        if module is not reference:
            raise TypeError(f"{name}: live nn.Linear is not the managed reference clone.")
        reference.to(device=device, dtype=dtype)
        return
    else:
        raise TypeError(f"{name}: expected inactive VAELinear or managed reference nn.Linear, got {type(module)}.")

    linear = _get_or_create_reference_linear(
        teacher_runtime=teacher_runtime,
        name=name,
        residency=residency,
        device=device,
        dtype=dtype,
    )
    set_module_by_name(model, name, linear)


def _apply_checkpoint_distill_residency(
    *,
    model: nn.Module,
    active_categories: Sequence[str],
    residency: _CheckpointDistillResidency,
    teacher_runtime,
    device: torch.device,
    dtype: torch.dtype,
    logger,
) -> None:
    active_set = {str(category) for category in active_categories}
    managed_names = _ensure_managed_name_inventory(model=model, residency=residency)

    active_vae = 0
    inactive_reference_linear = 0
    for name in managed_names:
        category = _category_of_module_name(name)
        if category in active_set:
            _ensure_active_vae(
                model=model,
                name=name,
                residency=residency,
                device=device,
            )
            active_vae += 1
        else:
            _ensure_inactive_reference_linear(
                model=model,
                name=name,
                residency=residency,
                teacher_runtime=teacher_runtime,
                device=device,
                dtype=dtype,
            )
            inactive_reference_linear += 1

    logger.info(
        "Checkpoint distill residency: active_categories=%s active_vae=%d "
        "inactive_reference_linear=%d stashed_vae=%d reference_clone_cache_size=%d",
        ",".join(str(category) for category in active_categories),
        int(active_vae),
        int(inactive_reference_linear),
        int(len(residency.stashed_vae_modules)),
        int(len(residency.reference_dense_linears)),
    )


def _restore_checkpoint_distill_residency(
    *,
    model: nn.Module,
    residency: _CheckpointDistillResidency,
    logger,
) -> None:
    restored = 0
    for name, module in list(residency.stashed_vae_modules.items()):
        base_layer = _resolve_vae_base_layer(module)
        base_layer.clear_decoded_weight_cache()
        set_module_by_name(model, name, module)
        del residency.stashed_vae_modules[name]
        restored += 1
    removed_reference = 0
    for name, reference in list(residency.reference_dense_linears.items()):
        live = _get_module_by_name(model, name)
        if live is reference:
            raise RuntimeError(f"{name}: reference clone still live after restoring all VAELinear modules.")
        reference.to("cpu")
        removed_reference += 1
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


def _collect_active_vae_prewarm_targets(
    *,
    model: nn.Module,
    active_categories: Sequence[str],
    logger,
) -> List[NamedVAELinearTarget]:
    active_set = {str(category) for category in active_categories}
    prewarm_targets: List[NamedVAELinearTarget] = []

    for target in _iter_named_vae_targets(model):
        target.base_layer.clear_decoded_weight_cache()
        if target.category in active_set:
            prewarm_targets.append(NamedVAELinearTarget(name=target.name, base_layer=target.base_layer))

    logger.info(
        "Checkpoint distill active categories=%s active_prewarm_targets=%d",
        ",".join(str(category) for category in active_categories),
        int(len(prewarm_targets)),
    )
    return prewarm_targets


def _restore_final_vae_representation(
    *,
    model: nn.Module,
    residency: _CheckpointDistillResidency,
    completed_categories: Sequence[str],
    logger,
) -> List[NamedVAELinearTarget]:
    _restore_checkpoint_distill_residency(
        model=model,
        residency=residency,
        logger=logger,
    )
    return _collect_active_vae_prewarm_targets(
        model=model,
        active_categories=list(completed_categories),
        logger=logger,
    )


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
        preserve_original_weights_from_base=False,
    )
    stripped = normalize_cat_runtime_vae_original_state(model)
    vae_args.model_path = str(load_meta.get("base_model_path") or base_model_path)
    logger.info(
        "Checkpoint loaded for distill. missing_keys=%d unexpected_keys=%d converted_module_count=%s stripped_original_weight=%d",
        len(getattr(load_result, "missing_keys", [])),
        len(getattr(load_result, "unexpected_keys", [])),
        str(load_meta.get("converted_module_count")),
        int(stripped),
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
        unload_vae_original_weights=False,
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
        unload_vae_original_weights=False,
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
    teacher_runtime,
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
            residency=residency,
            teacher_runtime=teacher_runtime,
            device=residency_device,
            dtype=residency_dtype,
            logger=logger,
        )
        _collect_active_vae_prewarm_targets(
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
    teacher_runtime = DistillTeacherRuntime(
        model_path=str(vae_args.model_path),
        access_token=hf_args.access_token,
        forward_device=resolve_distill_train_device(cat_args.train_device),
        dtype=resolve_distill_teacher_dtype(training_args, model),
        model_offload=str(getattr(training_args, "distill_teacher_model_offload", "none")),
        logger=logger,
    )
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
                "Checkpoint distill: --distill_reset_completed=true，忽略 resume ckpt 中的 "
                "completed_categories=%s；已有 low_rank_a/b 的类将用其初始化 LoRA 再蒸并覆盖写回。",
                ",".join(completed_categories),
            )
        else:
            logger.info(
                "Checkpoint distill: --distill_reset_completed=true，resume ckpt 无 completed_categories；"
                "若类上已有 low_rank_a/b，将从其初始化 LoRA 续蒸。"
            )
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
    lora_round_idx = int(len(completed_categories))
    active_categories: List[str] = []
    completed_categories = list(completed_categories)
    independent_categories = bool(getattr(cat_args, "distill_independent_categories", False))
    if independent_categories:
        logger.info(
            "Checkpoint distill: --distill_independent_categories=true，"
            "每轮只激活当前类；已完成类恢复为未压缩 Linear。"
        )
    for category in target_categories:
        if independent_categories:
            round_active_categories = [str(category)]
        else:
            active_categories.append(str(category))
            round_active_categories = active_categories
        _apply_checkpoint_distill_residency(
            model=model,
            active_categories=round_active_categories,
            residency=residency,
            teacher_runtime=teacher_runtime,
            device=residency_device,
            dtype=residency_dtype,
            logger=logger,
        )
        prewarm_targets = _collect_active_vae_prewarm_targets(
            model=model,
            active_categories=round_active_categories,
            logger=logger,
        )
        del prewarm_targets  # outer prewarm removed (O5); inner path in after-category distill handles cache

        skip_from_progress = str(category) in set(completed_categories)
        if skip_from_progress:
            logger.info(
                "Checkpoint distill: category=%s 已在 completed_categories 中，跳过蒸馏。",
                str(category),
            )
            continue

        category_steps = int(resolve_after_category_value(cat_args.distill_steps, category))
        run_this_category_eval = bool(run_category_eval) and category_steps > 0
        if run_category_eval and not run_this_category_eval:
            logger.info(
                "Checkpoint distill: category=%s distill_steps=%d，跳过该类别评估。",
                str(category),
                category_steps,
            )

        if run_this_category_eval:
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
                run_output_dir=run_output_dir,
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
            teacher_runtime=teacher_runtime,
            newly_compressed_target_count=0,
        )
        model = distill_result.model
        lora_round_idx = int(distill_result.next_lora_round_idx)

        if bool(distill_result.did_train):
            if str(category) not in completed_categories:
                completed_categories.append(str(category))
            _apply_checkpoint_distill_residency(
                model=model,
                active_categories=round_active_categories,
                residency=residency,
                teacher_runtime=teacher_runtime,
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
                    active_categories=round_active_categories,
                    residency=residency,
                    cat_args=cat_args,
                    hf_args=hf_args,
                    vae_args=vae_args,
                    training_args=training_args,
                    teacher_runtime=teacher_runtime,
                    logger=logger,
                )

        if run_this_category_eval:
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
                run_output_dir=run_output_dir,
            )

    needs_final_vae_representation = bool(run_category_eval) or bool(cat_args.save_model)
    if needs_final_vae_representation:
        if is_distill_main_process():
            logger.info(
                "Finalizing checkpoint distill with VAELinear representation: completed_categories=%s",
                ",".join(completed_categories) if completed_categories else "(none)",
            )
        _restore_final_vae_representation(
            model=model,
            residency=residency,
            completed_categories=completed_categories,
            logger=logger,
        )
        distill_distributed_barrier()

    if run_category_eval:
        if is_distill_main_process():
            logger.info("所有类别蒸馏完成后最终评估（VAELinear 路径）...")
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
            run_output_dir=run_output_dir,
        )

    if cat_args.save_model and is_distill_main_process():
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
