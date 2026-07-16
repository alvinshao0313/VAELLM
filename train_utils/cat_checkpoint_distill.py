import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

from e2e_common.peft_proxy import PeftVAELinearProxy, iter_named_peft_vae_proxies
from e2e_common.post_norm_head import fuse_post_norm_head_linear
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from litebsq.vae_linear_prewarm import (
    NamedVAELinearTarget,
    clear_model_vae_linear_cache,
    prime_named_vae_linear_cache,
)
from train_utils.cat_after_category_distill import run_after_category_distill
from train_utils.cat_train_args import resolve_category_runtime_configs
from train_utils.cat_train_eval import eval_after_category as _eval_after_category
from train_utils.cat_train_runtime import save_normalized_cat_train_snapshot as _save_normalized_cat_train_snapshot
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
    stashed_modules: Dict[str, nn.Module] = field(default_factory=dict)


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


def _apply_vae_decoder_checkpoint_override(*, model: nn.Module, vae_args, logger) -> int:
    override = getattr(vae_args, "vae_decoder_checkpoint", None)
    if override is None:
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


def _make_frozen_linear_from_vae_original(*, name: str, base_layer: VAELinear) -> nn.Linear:
    original_weight = getattr(base_layer, "original_weight", None)
    if original_weight is None:
        raise RuntimeError(
            f"{name}: original_weight is required to replace inactive checkpoint-distill VAELinear."
        )
    if tuple(original_weight.shape) != (int(base_layer.out_features), int(base_layer.in_features)):
        raise ValueError(
            f"{name}: original_weight shape {tuple(original_weight.shape)} != "
            f"({int(base_layer.out_features)}, {int(base_layer.in_features)})"
        )
    bias = getattr(base_layer, "bias", None)
    linear = nn.Linear(
        int(base_layer.in_features),
        int(base_layer.out_features),
        bias=bias is not None,
        device=original_weight.device,
        dtype=original_weight.dtype,
    )
    linear.weight = nn.Parameter(original_weight.detach(), requires_grad=False)
    if bias is not None:
        if tuple(bias.shape) != (int(base_layer.out_features),):
            raise ValueError(
                f"{name}: bias shape {tuple(bias.shape)} != ({int(base_layer.out_features)},)"
            )
        linear.bias = nn.Parameter(bias.detach(), requires_grad=False)
    linear.requires_grad_(False)
    linear.eval()
    return linear


def _apply_checkpoint_distill_residency(
    *,
    model: nn.Module,
    active_categories: Sequence[str],
    residency: _CheckpointDistillResidency,
    logger,
) -> None:
    active_set = {str(category) for category in active_categories}
    restored = 0
    for name, module in list(residency.stashed_modules.items()):
        category = str(name).rsplit(".", 1)[-1]
        if category not in active_set:
            continue
        set_module_by_name(model, name, module)
        del residency.stashed_modules[name]
        restored += 1

    stashed = 0
    active = 0
    for target in list(_iter_named_vae_targets(model)):
        if target.category in active_set:
            if callable(getattr(target.module, "set_temporary", None)):
                target.module.set_temporary(True)
            else:
                target.base_layer.set_temporary(True)
            target.base_layer.clear_decoded_weight_cache()
            active += 1
            continue
        target.base_layer.clear_decoded_weight_cache()
        residency.stashed_modules[target.name] = target.module
        set_module_by_name(
            model,
            target.name,
            _make_frozen_linear_from_vae_original(name=target.name, base_layer=target.base_layer),
        )
        stashed += 1

    logger.info(
        "Checkpoint distill residency: active_categories=%s active_vae=%d newly_stashed=%d restored=%d total_stashed=%d",
        ",".join(str(category) for category in active_categories),
        int(active),
        int(stashed),
        int(restored),
        int(len(residency.stashed_modules)),
    )


def _restore_checkpoint_distill_residency(
    *,
    model: nn.Module,
    residency: _CheckpointDistillResidency,
    logger,
) -> None:
    restored = 0
    for name, module in list(residency.stashed_modules.items()):
        if isinstance(module, PeftVAELinearProxy):
            module.base_layer.clear_decoded_weight_cache()
        elif isinstance(module, VAELinear):
            module.clear_decoded_weight_cache()
        set_module_by_name(model, name, module)
        del residency.stashed_modules[name]
        restored += 1
    logger.info("Checkpoint distill residency: restored stashed VAELinear modules=%d", int(restored))


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
        "Checkpoint distill active categories=%s compressed_targets=%d original_targets=%d",
        ",".join(str(category) for category in active_categories),
        int(compressed_count),
        int(original_count),
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


def _save_final_model(*, model: nn.Module, run_output_dir: str, cat_args, hf_args, vae_args, logger) -> None:
    if not bool(getattr(cat_args, "convert", False)):
        raise ValueError("--save_model requires --convert")

    from transformers import AutoTokenizer

    model_out = os.path.join(run_output_dir, "final_model")
    tok = AutoTokenizer.from_pretrained(vae_args.model_path, use_fast=True, token=hf_args.access_token)
    fused_post_norm_head = fuse_post_norm_head_linear(model)
    if fused_post_norm_head:
        logger.info("Final save: fused post_norm_linear into lm_head.weight.")
    leftover_proxies = [name for name, _proxy in iter_named_peft_vae_proxies(model)]
    if leftover_proxies:
        raise RuntimeError(
            "Final save found unexported PeftVAELinearProxy modules: "
            + ", ".join(leftover_proxies)
        )
    cleared = clear_model_vae_linear_cache(model)
    logger.info("Final save: cleared decoded cache for %d VAELinear modules.", cleared)
    save_paths = save_model_checkpoint(
        model,
        model_out,
        base_model_path=vae_args.model_path,
        tokenizer=tok,
        save_config=True,
        extra_meta={"stage": "final"},
        unload_vae_original_weights=bool(cat_args.unload_vae_original_weights_on_final_save),
    )
    logger.info("Saved final model to %s", save_paths["output_dir"])


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
    _apply_vae_decoder_checkpoint_override(model=model, vae_args=vae_args, logger=logger)
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
    lora_round_idx = 0
    active_categories: List[str] = []
    distill_device = resolve_distill_train_device(str(cat_args.train_device))
    for category in target_categories:
        active_categories.append(str(category))
        _apply_checkpoint_distill_residency(
            model=model,
            active_categories=active_categories,
            residency=residency,
            logger=logger,
        )
        prewarm_targets = _set_active_vae_category_prefix(
            model=model,
            active_categories=active_categories,
            logger=logger,
        )
        if mode == "compressed_lora":
            prewarm_stats = prime_named_vae_linear_cache(
                prewarm_targets,
                clear_existing=True,
                group_size=8,
                compute_device=distill_device,
                logger=logger,
            )
            logger.info(
                "Checkpoint distill prewarm: category=%s active_categories=%s total=%d warmed=%d skipped=%d failed=%d",
                str(category),
                ",".join(active_categories),
                int(prewarm_stats.get("total", 0)),
                int(prewarm_stats.get("warmed", 0)),
                int(prewarm_stats.get("skipped", 0)),
                int(prewarm_stats.get("failed", 0)),
            )

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
        )

    distill_distributed_barrier()
    logger.info("Done.")
