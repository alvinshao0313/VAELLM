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
)
from litebsq.vae_linear import VAELinear
from train_utils.hif4_act import (
    build_hif4_act_controller,
    register_hif4_act_hooks,
    remove_hif4_act_hooks,
)
from train_utils.lora_data import prepare_lora_datasets
from train_utils.lora_utils import (
    _build_lora_trainer,
    _build_sft_args,
    _ensure_lora_stack_available,
    _ensure_lora_tokenizer_ready,
    _freeze_model_for_lora,
    _log_lora_stage_start,
    _resolve_lora_stage_config,
    _restore_model_use_cache,
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


def _log_lora_dataset(logger, dataset_mix_spec, source_stats, nsamples: int) -> None:
    logger.info(
        "After-category distill: 训练混合数据集=%s nsamples=%d eval_dataset=none",
        str(dataset_mix_spec),
        int(nsamples),
    )
    for source_info in source_stats:
        logger.info(
            "After-category distill: 混合数据源 alias=%s weight=%.6f target_rows=%d actual_rows=%d raw_rows=%d text_rows=%d hf=%s config=%s train_split=%s",
            str(source_info["alias"]),
            float(source_info["weight"]),
            int(source_info["target_rows"]),
            int(source_info["actual_rows"]),
            int(source_info["raw_rows"]),
            int(source_info["text_rows"]),
            str(source_info["path"]),
            "none" if source_info["config"] is None else str(source_info["config"]),
            str(source_info["train_split"]),
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
        base_layer.unpack_parallel_stage_decoder_()
        base_layer.disable_trainable_decode()
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
        if hasattr(trainer, "lora_hif4_act_controller"):
            trainer.lora_hif4_act_controller = hif4_act_controller
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

    model = trainer.model
    model.to("cpu")
    torch.cuda.empty_cache()
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

    cfg = _resolve_lora_stage_config(
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
    dataset_mix_spec, source_stats, train_ds, eval_ds, _eval_split = prepare_lora_datasets(
        cfg.dataset,
        nsamples=cfg.nsamples,
        seed=cfg.seed,
    )
    _log_lora_dataset(logger, dataset_mix_spec, source_stats, cfg.nsamples)
    if len(train_ds) == 0:
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
            init_mode="zero",
            materialize_before_inject=False,
        )
        if int(injected) != int(len(module_names)):
            raise RuntimeError(
                f"After-category distill expected {len(module_names)} proxy adapters, injected {injected}."
            )

    previous_use_cache = _freeze_model_for_lora(model, device=cfg.device, logger=logger)
    try:
        _set_proxy_decoder_adapter_mode(model, module_names, enabled=mode == "both")
        trainable = _enable_compressed_trainable_params(model, module_names, mode=mode)
        if not trainable:
            logger.info("After-category distill: 没有可训练参数，跳过。")
            return AfterCategoryDistillResult(model=model, next_lora_round_idx=next_round, trained_target_count=0)

        resolved_lora_loss = str(cfg.loss_type).strip().lower()
        use_custom_trainer = resolved_lora_loss not in {"", "none", "sft"} or float(cfg.hidden_loss_weight) > 0.0
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
        sft_args = _build_sft_args(cat_args=cat_args, training_args=training_args, cfg=cfg)
        hif4_act_controller = build_hif4_act_controller(cfg.use_lora_hif4_act)
        trainer = _build_lora_trainer(
            model=model,
            train_ds=train_ds,
            eval_ds=eval_ds,
            sft_args=sft_args,
            training_args=training_args,
            lora_config=None,
            cfg=cfg,
            hif4_act_controller=hif4_act_controller,
            teacher_param_snapshots=[],
        )
        model = _train_without_merging_peft_adapters(
            trainer=trainer,
            hif4_act_controller=hif4_act_controller,
            logger=logger,
        )
        if use_decoder:
            finalized = _finalize_decoder_trainables(model, module_names)
            logger.info("After-category distill: finalized trainable decoder modules=%d.", int(finalized))
        if use_lora:
            exported = export_peft_proxy_lora_to_low_rank(model, module_names=module_names)
            logger.info("After-category distill: exported compressed LoRA adapters to low_rank_a/b=%d.", int(exported))
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
    projection_suffixes: Sequence[str],
) -> AfterCategoryDistillResult:
    mode = str(getattr(cat_args, "distill_after_category", "none")).strip().lower()
    if mode == "none":
        return AfterCategoryDistillResult(model=model, next_lora_round_idx=int(lora_round_idx), trained_target_count=0)

    if mode == "remaining_lora":
        current_remaining_linears = _collect_linears(
            model,
            transpose_modules=transpose_modules,
            only_decoder_projections=only_decoder_projections,
            projection_suffixes=projection_suffixes,
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
