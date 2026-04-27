import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

try:
    from transformers import AutoTokenizer, TrainingArguments
except ImportError:
    AutoTokenizer = None
    TrainingArguments = None

from e2e_common.post_norm_head import ensure_post_norm_head_linear, resolve_post_norm_linear
from rotation.model_utils import get_model_type, get_pre_head_layernorm
from train_utils.cat_train_args import resolve_lora_runtime_config
from train_utils.hif4_act import (
    build_hif4_act_controller,
    register_hif4_act_hooks,
    remove_hif4_act_hooks,
)
from train_utils.lora_data import ensure_lora_dataset_stack_available, prepare_lora_datasets
from train_utils.lora_training import (
    CustomSFTTrainer,
    SFTTrainer,
    create_lora_adapters,
    ensure_lora_training_stack_available,
    merge_all_lora,
)


@dataclass(frozen=True)
class _ResolvedLoraStageConfig:
    device: str
    base_seed: int
    round_idx: int
    seed: int
    rank: int
    alpha: float
    dropout: float
    steps: int
    batch_size: int
    nsamples: int
    lr: float
    weight_decay: float
    log_every: int
    temperature: float
    loss_alpha: float
    loss_type: str
    dataset: str
    use_dora: bool
    use_lora_hif4_act: bool
    tune_final_norm: bool
    use_post_norm_head_linear: bool


@dataclass(frozen=True)
class _ExtraTrainableModule:
    name: str
    module: nn.Module


def _ensure_lora_stack_available() -> None:
    ensure_lora_training_stack_available()
    ensure_lora_dataset_stack_available()
    if AutoTokenizer is None or TrainingArguments is None:
        raise ImportError("未安装 transformers。请先安装：pip install transformers")


def _enum_to_value(value, default: str) -> str:
    raw = value if value is not None else default
    if hasattr(raw, "value"):
        raw = raw.value
    raw = str(raw).strip()
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw.lower()


def _resolve_lora_stage_config(
    *,
    cat_args,
    training_args,
    after_category: Optional[str],
    lora_round_idx: Optional[int],
) -> _ResolvedLoraStageConfig:
    round_idx = 0 if lora_round_idx is None else int(lora_round_idx)
    if round_idx < 0:
        raise ValueError(f"lora_round_idx must be >= 0, got {round_idx}")

    runtime_cfg = resolve_lora_runtime_config(cat_args, after_category)
    base_seed = int(getattr(cat_args, "seed", 0))
    return _ResolvedLoraStageConfig(
        device=str(getattr(cat_args, "train_device", "cuda")),
        base_seed=base_seed,
        round_idx=round_idx,
        seed=int(base_seed + round_idx),
        rank=int(runtime_cfg.rank),
        alpha=float(runtime_cfg.alpha),
        dropout=float(runtime_cfg.dropout),
        steps=int(runtime_cfg.steps),
        batch_size=int(runtime_cfg.batch_size),
        nsamples=int(runtime_cfg.nsamples),
        lr=float(runtime_cfg.lr),
        weight_decay=float(runtime_cfg.weight_decay),
        log_every=int(runtime_cfg.log_every),
        temperature=float(runtime_cfg.temperature),
        loss_alpha=float(runtime_cfg.loss_alpha),
        loss_type=str(runtime_cfg.loss_type),
        dataset=str(getattr(cat_args, "lora_dataset", "")).strip().lower(),
        use_dora=bool(runtime_cfg.use_dora),
        use_lora_hif4_act=bool(getattr(training_args, "lora_hif4_act", False)),
        tune_final_norm=bool(getattr(cat_args, "tune_final_norm", False)),
        use_post_norm_head_linear=bool(getattr(cat_args, "use_post_norm_head_linear", False)),
    )


def _freeze_model_for_lora(model: nn.Module, *, device: str, logger) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        logger.info("LoRA: 已启用输入梯度。")
    model.to(device)
    model.train()


def _find_module_name(model: nn.Module, target: nn.Module, fallback: str) -> str:
    for name, module in model.named_modules():
        if module is target:
            return str(name)
    return str(fallback)


def _collect_extra_trainable_modules(
    model: nn.Module,
    *,
    cfg: _ResolvedLoraStageConfig,
    logger,
) -> List[_ExtraTrainableModule]:
    modules: List[_ExtraTrainableModule] = []

    if bool(cfg.tune_final_norm):
        model_type = get_model_type(model)
        final_norm = get_pre_head_layernorm(model, model_type)
        final_norm_name = _find_module_name(model, final_norm, "model.norm")
        modules.append(_ExtraTrainableModule(name=final_norm_name, module=final_norm))

    if bool(cfg.use_post_norm_head_linear):
        attached = ensure_post_norm_head_linear(model)
        if attached:
            logger.info("LoRA: 已为 lm_head 挂载 identity 初始化的 post_norm_linear。")
        post_norm_linear = resolve_post_norm_linear(model)
        if post_norm_linear is None:
            raise ValueError("--use_post_norm_head_linear=true but model.lm_head is not LMHeadWithPostNormLinear.")
        post_norm_name = _find_module_name(model, post_norm_linear, "lm_head.post_norm_linear")
        modules.append(_ExtraTrainableModule(name=post_norm_name, module=post_norm_linear))

    return modules


def _snapshot_extra_trainable_params(
    modules: Sequence[_ExtraTrainableModule],
) -> List[Tuple[nn.Parameter, torch.Tensor]]:
    snapshots: List[Tuple[nn.Parameter, torch.Tensor]] = []
    seen = set()
    for item in modules:
        for _param_name, param in item.module.named_parameters(recurse=True):
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            snapshots.append((param, param.detach().clone()))
    return snapshots


def _enable_extra_trainable_params(modules: Sequence[_ExtraTrainableModule]) -> List[str]:
    enabled: List[str] = []
    seen = set()
    for item in modules:
        for param_name, param in item.module.named_parameters(recurse=True):
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            param.requires_grad = True
            enabled.append(str(item.name) if not param_name else f"{item.name}.{param_name}")
    return sorted(enabled)


def _log_lora_stage_start(
    *,
    logger,
    cfg: _ResolvedLoraStageConfig,
    after_category: Optional[str],
    remaining_categories: Sequence[str],
    target_count: int,
    extra_trainable_names: Sequence[str],
    use_custom_trainer: bool,
) -> None:
    if use_custom_trainer:
        logger.info(
            "LoRA: 使用 CustomSFTTrainer 微调，after_category=%s，loss_type=%s，use_dora=%s，目标类别=%s，目标模块=%d，额外参数=%s，rank=%d，alpha=%.2f，steps=%d，batch_size=%d，seed(base=%d,round=%d,effective=%d)",
            str(after_category),
            str(cfg.loss_type).strip().lower(),
            str(cfg.use_dora).lower(),
            ",".join(remaining_categories),
            int(target_count),
            ",".join(extra_trainable_names) if extra_trainable_names else "none",
            int(cfg.rank),
            float(cfg.alpha),
            int(cfg.steps),
            int(cfg.batch_size),
            int(cfg.base_seed),
            int(cfg.round_idx),
            int(cfg.seed),
        )
        logger.info(
            "LoRA: 蒸馏参数 loss_alpha=%.4f temperature=%.4f",
            float(cfg.loss_alpha),
            float(cfg.temperature),
        )
        return

    logger.info(
        "LoRA: 使用 SFTTrainer 微调，after_category=%s，use_dora=%s，目标类别=%s，目标模块=%d，额外参数=%s，rank=%d，alpha=%.2f，steps=%d，batch_size=%d，seed(base=%d,round=%d,effective=%d)",
        str(after_category),
        str(cfg.use_dora).lower(),
        ",".join(remaining_categories),
        int(target_count),
        ",".join(extra_trainable_names) if extra_trainable_names else "none",
        int(cfg.rank),
        float(cfg.alpha),
        int(cfg.steps),
        int(cfg.batch_size),
        int(cfg.base_seed),
        int(cfg.round_idx),
        int(cfg.seed),
    )


def _ensure_lora_tokenizer_ready(*, vae_args, model: nn.Module) -> None:
    tokenizer = getattr(vae_args, "_cached_lora_tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            vae_args.model_path,
            use_fast=True,
            token=getattr(vae_args, "access_token", None),
        )
        setattr(vae_args, "_cached_lora_tokenizer", tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def _build_sft_args(*, cat_args, training_args, cfg: _ResolvedLoraStageConfig):
    return TrainingArguments(
        output_dir=os.path.join(str(getattr(cat_args, "output_dir", ".result")), "lora_trainer_state"),
        per_device_train_batch_size=int(cfg.batch_size),
        gradient_accumulation_steps=int(getattr(training_args, "lora_gradient_accumulation_steps", 1)),
        optim=_enum_to_value(getattr(training_args, "lora_optim", "paged_adamw_8bit"), "paged_adamw_8bit"),
        logging_strategy="steps",
        logging_steps=max(1, int(cfg.log_every)),
        logging_first_step=True,
        learning_rate=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        fp16=bool(getattr(training_args, "fp16", False)),
        bf16=bool(getattr(training_args, "bf16", False)),
        max_grad_norm=float(getattr(training_args, "lora_max_grad_norm", 0.3)),
        max_steps=int(cfg.steps),
        warmup_ratio=float(getattr(training_args, "lora_warmup_ratio", 0.3)),
        group_by_length=bool(getattr(training_args, "lora_group_by_length", True)),
        lr_scheduler_type=_enum_to_value(getattr(training_args, "lora_lr_scheduler_type", "linear"), "linear"),
        report_to=[],
        disable_tqdm=False,
        save_strategy="no",
        seed=int(cfg.seed),
    )


def _build_lora_trainer(
    *,
    model: nn.Module,
    train_ds,
    eval_ds,
    sft_args,
    training_args,
    lora_config,
    cfg: _ResolvedLoraStageConfig,
    hif4_act_controller,
    teacher_param_snapshots,
):
    trainer_kwargs = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
        dataset_text_field="text",
        max_seq_length=int(getattr(training_args, "lora_model_max_length", 2048)),
    )
    if lora_config is not None:
        trainer_kwargs["peft_config"] = lora_config

    resolved_lora_loss = str(cfg.loss_type).strip().lower()
    if resolved_lora_loss not in {"", "none", "sft"}:
        return CustomSFTTrainer(
            **trainer_kwargs,
            loss_type=resolved_lora_loss,
            temperature=float(cfg.temperature),
            loss_alpha=float(cfg.loss_alpha),
            lora_hif4_act_controller=hif4_act_controller,
            teacher_param_snapshots=teacher_param_snapshots,
        )
    return SFTTrainer(**trainer_kwargs)


def _train_and_merge_lora_model(
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
            "LoRA: 已启用 HiFloat4 激活量化，student 前向量化类型=hifx4，hook 模块数=%d",
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

    model, merged_count = merge_all_lora(trainer.model)
    model.to("cpu")
    torch.cuda.empty_cache()
    logger.info("LoRA: 微调完成并融合，融合模块数量=%d", merged_count)
    return model


def lora_finetune_remaining_categories(
    model: nn.Module,
    remaining_categories: Sequence[str],
    *,
    target_names: Sequence[str],
    cat_args,
    vae_args,
    training_args,
    logger,
    lora_round_idx: Optional[int] = None,
    after_category: Optional[str] = None,
) -> nn.Module:
    cfg = _resolve_lora_stage_config(
        cat_args=cat_args,
        training_args=training_args,
        after_category=after_category,
        lora_round_idx=lora_round_idx,
    )
    has_extra_trainables = bool(cfg.tune_final_norm) or bool(cfg.use_post_norm_head_linear)
    if cfg.steps <= 0:
        return model
    if not remaining_categories and not has_extra_trainables:
        return model
    _ensure_lora_stack_available()

    if not target_names and not has_extra_trainables:
        logger.info("LoRA: 没有可微调的剩余 Linear，跳过。")
        return model

    _freeze_model_for_lora(model, device=cfg.device, logger=logger)
    extra_modules = _collect_extra_trainable_modules(model, cfg=cfg, logger=logger)
    teacher_param_snapshots = _snapshot_extra_trainable_params(extra_modules)
    model, lora_config, unique_target_names = create_lora_adapters(
        model,
        target_names=target_names,
        rank=cfg.rank,
        alpha=cfg.alpha,
        dropout=cfg.dropout,
        use_dora=cfg.use_dora,
    )
    extra_trainable_names = _enable_extra_trainable_params(extra_modules)
    if lora_config is None:
        logger.info("LoRA: 没有匹配到可插入 adapter 的 Linear，本轮仅训练额外解冻参数。")
        if not extra_trainable_names:
            logger.info("LoRA: 没有额外可训练参数，跳过。")
            return model

    resolved_lora_loss = str(cfg.loss_type).strip().lower()
    use_custom_trainer = resolved_lora_loss not in {"", "none", "sft"}
    _log_lora_stage_start(
        logger=logger,
        cfg=cfg,
        after_category=after_category,
        remaining_categories=remaining_categories,
        target_count=len(unique_target_names),
        extra_trainable_names=extra_trainable_names,
        use_custom_trainer=use_custom_trainer,
    )

    dataset_mix_spec, source_stats, train_ds, eval_ds, _eval_split = prepare_lora_datasets(
        cfg.dataset,
        nsamples=cfg.nsamples,
        seed=cfg.seed,
    )
    logger.info(
        "LoRA: 补偿训练混合数据集=%s nsamples=%d eval_dataset=none",
        str(dataset_mix_spec),
        int(cfg.nsamples),
    )
    for source_info in source_stats:
        logger.info(
            "LoRA: 混合数据源 alias=%s weight=%.6f target_rows=%d actual_rows=%d raw_rows=%d text_rows=%d hf=%s config=%s train_split=%s",
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
    if len(train_ds) == 0:
        logger.warning("LoRA: 数据集为空，跳过。")
        model, _merged_count = merge_all_lora(model)
        return model

    _ensure_lora_tokenizer_ready(vae_args=vae_args, model=model)
    sft_args = _build_sft_args(cat_args=cat_args, training_args=training_args, cfg=cfg)
    hif4_act_controller = build_hif4_act_controller(cfg.use_lora_hif4_act)
    trainer = _build_lora_trainer(
        model=model,
        train_ds=train_ds,
        eval_ds=eval_ds,
        sft_args=sft_args,
        training_args=training_args,
        lora_config=lora_config,
        cfg=cfg,
        hif4_act_controller=hif4_act_controller,
        teacher_param_snapshots=teacher_param_snapshots,
    )
    return _train_and_merge_lora_model(
        trainer=trainer,
        hif4_act_controller=hif4_act_controller,
        logger=logger,
    )
