from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from e2e_common.peft_proxy import (
    PeftVAELinearProxy,
    ensure_peft_vae_linear_proxy,
    ensure_peft_vae_proxy_adapter,
    export_peft_proxy_lora_to_low_rank,
    is_peft_proxy_adapter_linear,
    iter_named_peft_vae_proxies,
    materialize_peft_proxy_decoded_linears,
    unwrap_peft_vae_proxies,
)
from litebsq.vae_linear import VAELinear
from litebsq.vae_linear_prewarm import NamedVAELinearTarget, prime_model_vae_linear_cache, prime_named_vae_linear_cache
from train_utils.hif4_act import (
    build_hif4_act_controller,
    register_hif4_act_hooks,
    remove_hif4_act_hooks,
)
from e2e_common.lazy_datasets import dataset_length_or_none, is_iterable_training_dataset
from train_utils.lora_data import prepare_distill_datasets
from train_utils.lora_utils import (
    _build_lora_trainer,
    _build_sft_args,
    _ensure_lora_stack_available,
    _ensure_lora_tokenizer_ready,
    _freeze_model_for_lora,
    _log_lora_stage_start,
    _resolve_distill_stage_config,
    _restore_model_use_cache,
    distill_distributed_barrier,
    is_distill_distributed,
    is_distill_main_process,
    lora_finetune_remaining_categories,
)
from train_utils.utils import collect_linears as _collect_linears


_COMPRESSED_DISTILL_MODES = {"compressed_lora", "decoder", "both"}


@dataclass(frozen=True)
class AfterCategoryDistillResult:
    model: nn.Module
    next_lora_round_idx: int
    trained_target_count: int = 0


def _get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    current = model
    for token in str(module_name).split("."):
        if not hasattr(current, token):
            raise ValueError(f"Failed to resolve module {module_name!r}: missing {token!r}.")
        current = getattr(current, token)
    if not isinstance(current, nn.Module):
        raise TypeError(f"Resolved object at {module_name!r} is not an nn.Module: {type(current)}")
    return current


def _resolve_base_layer(module_name: str, module: nn.Module) -> VAELinear:
    if isinstance(module, PeftVAELinearProxy):
        return module.base_layer
    if isinstance(module, VAELinear):
        return module
    raise TypeError(f"{module_name}: expected VAELinear or PeftVAELinearProxy, got {type(module)}")


def _logger_warning(logger, message: str, *args) -> None:
    warn = getattr(logger, "warning", None)
    if callable(warn):
        warn(message, *args)
        return
    info = getattr(logger, "info", None)
    if callable(info):
        info(message, *args)


def _log_distill_dataset(logger, dataset_mix_spec, source_stats, train_len: Optional[int], train_is_iterable: bool) -> None:
    logger.info(
        "After-category distill: 训练混合数据集=%s lazy_iterable=%s dataset_len=%s eval_dataset=none",
        str(dataset_mix_spec),
        str(train_is_iterable).lower(),
        "unknown" if train_len is None else str(train_len),
    )
    for source_info in source_stats:
        logger.info(
            "After-category distill: 混合数据源 alias=%s weight=%.6f raw_rows=%d hf=%s config=%s train_split=%s lazy_iterable=%s",
            str(source_info["alias"]),
            float(source_info["weight"]),
            int(source_info["raw_rows"]),
            str(source_info["path"]),
            "none" if source_info["config"] is None else str(source_info["config"]),
            str(source_info["train_split"]),
            str(source_info.get("is_iterable", train_is_iterable)).lower(),
        )


def _log_vae_linear_cache_status(model: nn.Module, logger, prefix: str) -> None:
    total = 0
    low_rank = 0
    cached = 0
    cache_off = 0
    skip_prewarm = 0
    proxy_count = 0

    for module in model.modules():
        if isinstance(module, PeftVAELinearProxy):
            proxy_count += 1
        if isinstance(module, VAELinear):
            total += 1
            if module.has_low_rank_residual():
                low_rank += 1
            cached_weight = getattr(module, "_cached_weight", None)
            if isinstance(cached_weight, torch.Tensor):
                cached += 1
            if not bool(getattr(module, "cache_decoded_weight", True)):
                cache_off += 1
            if bool(getattr(module, "_skip_global_cache_prewarm", False)):
                skip_prewarm += 1

    logger.info(
        "%s VAELinear cache status: total=%d proxy=%d low_rank=%d cached=%d cache_off=%d skip_prewarm=%d",
        str(prefix),
        int(total),
        int(proxy_count),
        int(low_rank),
        int(cached),
        int(cache_off),
        int(skip_prewarm),
    )


def collect_compressed_category_targets(
    model: nn.Module,
    category: str,
) -> List[Tuple[str, VAELinear]]:
    category = str(category)
    targets: List[Tuple[str, VAELinear]] = []
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
        if name.rsplit(".", 1)[-1] != category:
            continue
        targets.append((str(name), base_layer))
    return targets


def category_targets_have_low_rank(targets: Sequence[Tuple[str, VAELinear]]) -> bool:
    if not targets:
        return False
    return all(base_layer.has_low_rank_residual() for _name, base_layer in targets)


def _wrap_targets_as_peft_proxies(
    model: nn.Module,
    targets: Sequence[Tuple[str, VAELinear]],
) -> List[str]:
    existing = [name for name, _proxy in iter_named_peft_vae_proxies(model)]
    if existing:
        raise RuntimeError(f"Refusing to start compressed LoRA distill with existing PeftVAELinearProxy modules: {existing}")
    wrapped: List[str] = []
    for name, module in targets:
        ensure_peft_vae_linear_proxy(model, str(name), module)
        wrapped.append(str(name))
    return wrapped


def _set_proxy_decoder_adapter_mode(model: nn.Module, module_names: Sequence[str], enabled: bool) -> None:
    for name in module_names:
        module = _get_module_by_name(model, str(name))
        if isinstance(module, PeftVAELinearProxy):
            module._train_decoder_with_adapter = bool(enabled)


def _enable_only_decoder_params(base_layer: VAELinear) -> List[nn.Parameter]:
    base_layer.enable_trainable_decode(parallel_stage_decode=True)
    trainable: List[nn.Parameter] = []
    for param in base_layer.parameters():
        if bool(param.requires_grad):
            trainable.append(param)
    return trainable


def _unwrap_peft_proxies_without_export(model: nn.Module, module_names: Sequence[str]) -> int:
    return unwrap_peft_vae_proxies(model, module_names=module_names)


def _enable_compressed_trainable_params(
    model: nn.Module,
    module_names: Sequence[str],
    *,
    mode: str,
) -> List[nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False

    trainable: List[nn.Parameter] = []
    for name in module_names:
        module = _get_module_by_name(model, str(name))
        if mode in {"decoder", "both"}:
            base_layer = _resolve_base_layer(str(name), module)
            trainable.extend(_enable_only_decoder_params(base_layer))
        if mode in {"compressed_lora", "both"}:
            if not isinstance(module, PeftVAELinearProxy):
                raise TypeError(f"{name}: expected PeftVAELinearProxy for compressed_lora, got {type(module)}")
            peft_linear = module.per_decoded_linear
            if not is_peft_proxy_adapter_linear(peft_linear):
                raise TypeError(f"{name}: expected PEFT adapter linear, got {type(peft_linear)}")
            for param_name, param in peft_linear.named_parameters():
                if param_name in {"base_layer.weight", "base_layer.bias"}:
                    continue
                param.requires_grad = True
                trainable.append(param)
    return trainable


def _finalize_decoder_trainables(model: nn.Module, module_names: Sequence[str]) -> int:
    finalized = 0
    for name in module_names:
        module = _get_module_by_name(model, str(name))
        base_layer = _resolve_base_layer(str(name), module)
        packed = getattr(base_layer, "_parallel_stage_decoder", None)
        if packed is not None:
            packed.requires_grad_(False)
        base_layer.disable_trainable_decode()
        base_layer.clear_decoded_weight_cache()
        finalized += 1
    return int(finalized)


def _train_without_merging_peft_adapters(
    *,
    trainer,
    hif4_act_controller,
    logger,
) -> nn.Module:
    hif4_act_handles: List[torch.utils.hooks.RemovableHandle] = []
    if hif4_act_controller is not None:
        if hasattr(trainer, "distill_hif4_act_controller"):
            trainer.distill_hif4_act_controller = hif4_act_controller
        hif4_act_handles = register_hif4_act_hooks(trainer.model, hif4_act_controller)
        if not hif4_act_handles:
            raise RuntimeError("启用 HiFloat4 激活量化失败：未找到可注册 hook 的逻辑线性层。")
        logger.info(
            "After-category distill: 已启用 HiFloat4 激活量化，student 前向量化类型=hifx4，hook 模块数=%d",
            len(hif4_act_handles),
        )

    if hif4_act_controller is not None:
        hif4_act_controller.enabled = True
    try:
        trainer.train()
    finally:
        if hif4_act_controller is not None:
            hif4_act_controller.enabled = False
        remove_hif4_act_hooks(hif4_act_handles)

    distill_distributed_barrier()
    model = trainer.model
    if not is_distill_distributed():
        model.to("cpu")
        torch.cuda.empty_cache()
    distill_distributed_barrier()
    return model


def _run_compressed_category_distill(
    *,
    model: nn.Module,
    category: str,
    mode: str,
    cat_args,
    vae_args,
    training_args,
    logger,
    lora_round_idx: int,
) -> AfterCategoryDistillResult:
    next_round = int(lora_round_idx) + 1
    targets = collect_compressed_category_targets(model, category)
    module_names = [name for name, _module in targets]
    if not module_names:
        logger.info(
            "After-category distill: mode=%s category=%s 没有当前类别 VAELinear，跳过。",
            str(mode),
            str(category),
        )
        return AfterCategoryDistillResult(model=model, next_lora_round_idx=next_round, trained_target_count=0)

    if mode in {"compressed_lora", "both"} and category_targets_have_low_rank(targets):
        logger.info(
            "After-category distill: mode=%s category=%s 已有 low_rank_a/b，自动跳过。",
            str(mode),
            str(category),
        )
        return AfterCategoryDistillResult(model=model, next_lora_round_idx=next_round, trained_target_count=0)

    cfg = _resolve_distill_stage_config(
        cat_args=cat_args,
        training_args=training_args,
        after_category=category,
        lora_round_idx=lora_round_idx,
    )
    if bool(cfg.use_dora) and mode in {"compressed_lora", "both"}:
        raise ValueError(f"--distill_after_category={mode} does not support --lora_use_dora=true.")
    if int(cfg.steps) <= 0:
        logger.info(
            "After-category distill: mode=%s category=%s steps=%d，跳过。",
            str(mode),
            str(category),
            int(cfg.steps),
        )
        return AfterCategoryDistillResult(model=model, next_lora_round_idx=next_round, trained_target_count=0)

    _ensure_lora_stack_available()
    _ensure_lora_tokenizer_ready(vae_args=vae_args, model=model)
    tokenizer = getattr(vae_args, "_cached_lora_tokenizer", None)
    max_seq_len = int(getattr(training_args, "distill_model_max_length", 2048))
    # Reuse the same lazy mix across categories; trainer seed still varies by round.
    dataset_cache = getattr(vae_args, "_cached_distill_datasets", None)
    if not isinstance(dataset_cache, dict):
        dataset_cache = {}
        setattr(vae_args, "_cached_distill_datasets", dataset_cache)
    dataset_cache_key = (str(cfg.dataset), int(max_seq_len), int(cfg.base_seed), id(tokenizer))
    cached_dataset = dataset_cache.get(dataset_cache_key)
    if cached_dataset is None:
        dataset_mix_spec, source_stats, train_ds, eval_ds, _eval_split = prepare_distill_datasets(
            cfg.dataset,
            seed=int(cfg.base_seed),
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        dataset_cache[dataset_cache_key] = (
            dataset_mix_spec,
            source_stats,
            train_ds,
            eval_ds,
            _eval_split,
        )
        logger.info(
            "After-category distill: prepared distill dataset cache key=%s",
            str(dataset_cache_key[:3]),
        )
    else:
        dataset_mix_spec, source_stats, train_ds, eval_ds, _eval_split = cached_dataset
        logger.info(
            "After-category distill: reused distill dataset cache key=%s",
            str(dataset_cache_key[:3]),
        )
    train_is_iterable = is_iterable_training_dataset(train_ds)
    train_len = dataset_length_or_none(train_ds)
    _log_distill_dataset(logger, dataset_mix_spec, source_stats, train_len, train_is_iterable)
    if train_len == 0:
        _logger_warning(logger, "After-category distill: 数据集为空，跳过。")
        return AfterCategoryDistillResult(model=model, next_lora_round_idx=next_round, trained_target_count=0)

    use_lora = mode in {"compressed_lora", "both"}
    use_decoder = mode in {"decoder", "both"}
    if use_lora:
        module_names = _wrap_targets_as_peft_proxies(model, targets)
        materialize_peft_proxy_decoded_linears(
            model,
            group_size=8,
            compute_device=cfg.device,
            logger=logger,
            log_prefix=f"After-category distill {category}: ",
        )
        injected = ensure_peft_vae_proxy_adapter(
            model,
            variant="plain",
            rank=cfg.rank,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            init_mode="peft_default",
            materialize_before_inject=False,
        )
        if int(injected) != int(len(module_names)):
            raise RuntimeError(
                f"After-category distill expected {len(module_names)} proxy adapters, injected {injected}."
            )

    previous_use_cache = _freeze_model_for_lora(model, device=cfg.device, logger=logger)
    try:
        _log_vae_linear_cache_status(
            model,
            logger,
            prefix=f"After-category distill {category}: before prewarm",
        )
        # decoder/both: current-category modules will clear cache under trainable_decode;
        # only prewarm other VAELinear that still use cache (e.g. completed prefix).
        # compressed_lora: proxies skip prewarm via _skip_global_cache_prewarm; warm the rest.
        if use_decoder:
            skip_names = set(module_names)
            prewarm_targets = []
            for name, module in model.named_modules():
                if not isinstance(module, VAELinear):
                    continue
                if str(name) in skip_names:
                    continue
                if any(str(name) == p or str(name).startswith(f"{p}.") for p in skip_names):
                    continue
                prewarm_targets.append(NamedVAELinearTarget(name=str(name), base_layer=module))
            if prewarm_targets:
                prewarm_stats = prime_named_vae_linear_cache(
                    prewarm_targets,
                    dtype=torch.bfloat16 if bool(getattr(training_args, "bf16", False)) else None,
                    clear_existing=False,
                    group_size=8,
                    compute_device=cfg.device,
                    logger=logger,
                )
            else:
                prewarm_stats = {"total": 0, "warmed": 0, "skipped": 0, "failed": 0}
            logger.info(
                "After-category distill %s: decoder/both prefix prewarm stats=%s (skipped current category)",
                str(category),
                str(prewarm_stats),
            )
        else:
            prewarm_stats = prime_model_vae_linear_cache(
                model,
                dtype=torch.bfloat16 if bool(getattr(training_args, "bf16", False)) else None,
                clear_existing=False,
                group_size=8,
                compute_device=cfg.device,
                logger=logger,
            )
            logger.info(
                "After-category distill %s: prewarmed VAELinear cache stats=%s",
                str(category),
                str(prewarm_stats),
            )
        _log_vae_linear_cache_status(
            model,
            logger,
            prefix=f"After-category distill {category}: after prewarm",
        )

        _set_proxy_decoder_adapter_mode(model, module_names, enabled=mode == "both")
        trainable = _enable_compressed_trainable_params(model, module_names, mode=mode)
        if not trainable:
            if use_lora:
                restored = _unwrap_peft_proxies_without_export(model, module_names)
                logger.info(
                    "After-category distill: 没有可训练参数，已拆掉 proxy=%d，跳过。",
                    int(restored),
                )
            if use_decoder:
                finalized = _finalize_decoder_trainables(model, module_names)
                logger.info(
                    "After-category distill: 没有可训练参数，已 finalize decoder modules=%d，跳过。",
                    int(finalized),
                )
            if not use_lora and not use_decoder:
                logger.info("After-category distill: 没有可训练参数，跳过。")
            return AfterCategoryDistillResult(model=model, next_lora_round_idx=next_round, trained_target_count=0)

        resolved_lora_loss = str(cfg.loss_type).strip().lower()
        use_custom_trainer = (
            resolved_lora_loss not in {"", "none", "sft"}
            or float(cfg.hidden_loss_weight) > 0.0
            or float(cfg.pre_mlp_hidden_loss_weight) > 0.0
        )
        _log_lora_stage_start(
            logger=logger,
            cfg=cfg,
            after_category=category,
            remaining_categories=[category],
            target_count=len(module_names),
            extra_trainable_names=[],
            use_custom_trainer=use_custom_trainer,
        )
        _ensure_lora_tokenizer_ready(vae_args=vae_args, model=model)
        sft_args = _build_sft_args(
            cat_args=cat_args,
            training_args=training_args,
            cfg=cfg,
            train_is_iterable=train_is_iterable,
            logger=logger,
        )
        hif4_act_controller = build_hif4_act_controller(cfg.use_distill_hif4_act)
        tokenizer = getattr(vae_args, "_cached_lora_tokenizer", None)
        trainer = _build_lora_trainer(
            model=model,
            train_ds=train_ds,
            eval_ds=eval_ds,
            sft_args=sft_args,
            training_args=training_args,
            logger=logger,
            lora_config=None,
            cfg=cfg,
            hif4_act_controller=hif4_act_controller,
            teacher_param_snapshots=[],
            tokenizer=tokenizer,
            train_is_iterable=train_is_iterable,
            use_lazy_tokenized_dataset=True,
        )
        if use_decoder:
            VAELinear.reset_fuse_stats()
        model = _train_without_merging_peft_adapters(
            trainer=trainer,
            hif4_act_controller=hif4_act_controller,
            logger=logger,
        )
        if use_decoder:
            fuse_stats = VAELinear.get_fuse_stats()
            logger.info(
                "After-category distill %s: fuse_stats hit=%d miss=%d miss_reasons=%s",
                str(category),
                int(fuse_stats["hit"]),
                int(fuse_stats["miss"]),
                str(fuse_stats["miss_reasons"]),
            )
            if int(fuse_stats["miss"]) > 0:
                logger.warning(
                    "After-category distill %s: fused decode missed %d times; reasons=%s",
                    str(category),
                    int(fuse_stats["miss"]),
                    str(fuse_stats["miss_reasons"]),
                )
            finalized = _finalize_decoder_trainables(model, module_names)
            logger.info("After-category distill: finalized trainable decoder modules=%d.", int(finalized))
        if use_lora:
            exported = export_peft_proxy_lora_to_low_rank(model, module_names=module_names)
            logger.info("After-category distill: exported compressed LoRA adapters to low_rank_a/b=%d.", int(exported))
            _log_vae_linear_cache_status(
                model,
                logger,
                prefix=f"After-category distill {category}: after export",
            )
        return AfterCategoryDistillResult(
            model=model,
            next_lora_round_idx=next_round,
            trained_target_count=len(module_names),
        )
    finally:
        _restore_model_use_cache(model, previous_use_cache, logger=logger)


def run_after_category_distill(
    *,
    model: nn.Module,
    category: str,
    cat_args,
    vae_args,
    training_args,
    logger,
    lora_round_idx: int,
    transpose_modules: Sequence[str],
    only_decoder_projections: bool,
    target_categories: Sequence[str],
) -> AfterCategoryDistillResult:
    mode = str(getattr(cat_args, "distill_after_category", "none")).strip().lower()
    if mode == "none":
        return AfterCategoryDistillResult(model=model, next_lora_round_idx=int(lora_round_idx), trained_target_count=0)

    if mode == "remaining_lora":
        current_remaining_linears = _collect_linears(
            model,
            transpose_modules=transpose_modules,
            only_decoder_projections=only_decoder_projections,
            target_categories=target_categories,
        )
        remaining_categories = list(dict.fromkeys(r.category for r in current_remaining_linears))
        model = lora_finetune_remaining_categories(
            model=model,
            remaining_categories=remaining_categories,
            target_names=[r.name for r in current_remaining_linears],
            cat_args=cat_args,
            vae_args=vae_args,
            training_args=training_args,
            logger=logger,
            lora_round_idx=lora_round_idx,
            after_category=category,
        )
        return AfterCategoryDistillResult(
            model=model,
            next_lora_round_idx=int(lora_round_idx) + 1,
            trained_target_count=len(current_remaining_linears),
        )

    if mode in _COMPRESSED_DISTILL_MODES:
        return _run_compressed_category_distill(
            model=model,
            category=category,
            mode=mode,
            cat_args=cat_args,
            vae_args=vae_args,
            training_args=training_args,
            logger=logger,
            lora_round_idx=int(lora_round_idx),
        )

    raise ValueError(
        "--distill_after_category must be one of: none, remaining_lora, compressed_lora, decoder, both."
    )
