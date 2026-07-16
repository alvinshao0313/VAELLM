import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

try:
    from transformers import AutoTokenizer, TrainerCallback, TrainingArguments
    from transformers.trainer_callback import ProgressCallback
except ImportError:
    AutoTokenizer = None
    ProgressCallback = None
    TrainerCallback = None
    TrainingArguments = None

from e2e_common.chat_template_utils import (
    infer_assistant_response_template,
    infer_user_instruction_template,
    render_messages,
)
from e2e_common.data import (
    VAELLM_EDGERAZOR_SFT_ALIASES,
    normalize_dataset_mix_spec,
)
from e2e_common.post_norm_head import ensure_post_norm_head_linear, resolve_post_norm_linear
from rotation.model_utils import get_model_type, get_pre_head_layernorm
from train_utils.cat_train_args import resolve_distill_runtime_config
from train_utils.hif4_act import (
    build_hif4_act_controller,
    register_hif4_act_hooks,
    remove_hif4_act_hooks,
)
from e2e_common.lazy_datasets import (
    build_edgerazor_data_collator,
    dataset_length_or_none,
    default_dataloader_num_workers,
    is_iterable_training_dataset,
)
from train_utils.lora_data import ensure_distill_dataset_stack_available, prepare_distill_datasets
from train_utils.lora_training import (
    CustomSFTTrainer,
    DataCollatorForCompletionOnlyLM,
    SFTTrainer,
    create_lora_adapters,
    ensure_lora_training_stack_available,
    merge_all_lora,
)


@dataclass(frozen=True)
class _ResolvedDistillStageConfig:
    device: str
    base_seed: int
    round_idx: int
    seed: int
    rank: int
    alpha: float
    dropout: float
    steps: int
    batch_size: int
    lr: float
    weight_decay: float
    log_every: int
    temperature: float
    loss_alpha: float
    loss_type: str
    hidden_loss_weight: float
    pre_mlp_hidden_loss_weight: float
    hidden_alignment_layer_weighting: str
    eakld_confidence_k: int
    dataset: str
    use_dora: bool
    use_distill_hif4_act: bool
    distill_tune_final_norm: bool
    distill_use_post_norm_head_linear: bool


@dataclass(frozen=True)
class _ExtraTrainableModule:
    name: str
    module: nn.Module


class _LoraTrainerLogCallback(TrainerCallback if TrainerCallback is not None else object):
    def __init__(self, *, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not bool(getattr(state, "is_world_process_zero", True)):
            return
        if not logs:
            return
        values = dict(logs)
        values.pop("total_flos", None)
        ordered_keys = (
            "loss",
            "train_loss",
            "eval_loss",
            "learning_rate",
            "grad_norm",
            "epoch",
        )
        parts = []
        for key in ordered_keys:
            if key in values:
                parts.append(f"{key}={values.pop(key)}")
        for key in sorted(values):
            parts.append(f"{key}={values[key]}")
        if not parts:
            return
        _log_lora_trainer_message_to_file_handlers(
            self.logger,
            "LoRA train: step=%s %s",
            str(getattr(state, "global_step", "unknown")),
            " ".join(parts),
        )


class _QuietProgressCallback(ProgressCallback if ProgressCallback is not None else object):
    def on_log(self, args, state, control, logs=None, **kwargs):
        return


def _log_lora_trainer_message_to_file_handlers(logger, message: str, *args) -> None:
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        fn="",
        lno=0,
        msg=message,
        args=args,
        exc_info=None,
    )
    for handler in list(getattr(logger, "handlers", [])):
        if not isinstance(handler, logging.FileHandler):
            continue
        if record.levelno < handler.level:
            continue
        handler.handle(record)


def _replace_progress_log_callback(trainer):
    if ProgressCallback is None:
        return trainer
    callback_handler = getattr(trainer, "callback_handler", None)
    callbacks = getattr(callback_handler, "callbacks", None)
    if not isinstance(callbacks, list):
        return trainer
    for idx, callback in enumerate(callbacks):
        if isinstance(callback, ProgressCallback) and not isinstance(callback, _QuietProgressCallback):
            callbacks[idx] = _QuietProgressCallback()
    return trainer


def _ensure_lora_stack_available() -> None:
    ensure_lora_training_stack_available()
    ensure_distill_dataset_stack_available()
    if AutoTokenizer is None or TrainingArguments is None:
        raise ImportError("未安装 transformers。请先安装：pip install transformers")


def distill_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_distill_distributed() -> bool:
    return distill_world_size() > 1


def resolve_distill_train_device(fallback: str) -> str:
    device = str(fallback).strip()
    if not is_distill_distributed():
        return device
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device.startswith("cuda") and torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return device


def is_distill_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()) == 0
    return int(os.environ.get("RANK", "0")) == 0


def distill_distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def ensure_distill_process_group_initialized() -> None:
    if not is_distill_distributed():
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is unavailable but WORLD_SIZE > 1.")
    backend = "nccl" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend=backend)


def distill_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "0"))


def get_distill_local_device(*, fallback: str = "cuda") -> str:
    return resolve_distill_train_device(str(fallback))


def unwrap_distill_model(model: nn.Module) -> nn.Module:
    current = model
    while hasattr(current, "module"):
        inner = getattr(current, "module")
        if inner is current:
            break
        current = inner
    return current


def split_tasks_for_distill_rank(
    task_names: Sequence[str],
    *,
    rank: int,
    world_size: int,
) -> List[str]:
    world = int(world_size)
    if world <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}.")
    current_rank = int(rank)
    if current_rank < 0 or current_rank >= world:
        raise ValueError(f"rank must be in [0, {world}), got {rank}.")
    return [str(name) for idx, name in enumerate(task_names) if idx % world == current_rank]


def _enum_to_value(value, default: str) -> str:
    raw = value if value is not None else default
    if hasattr(raw, "value"):
        raw = raw.value
    raw = str(raw).strip()
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw.lower()


def _resolve_distill_stage_config(
    *,
    cat_args,
    training_args,
    after_category: Optional[str],
    lora_round_idx: Optional[int],
) -> _ResolvedDistillStageConfig:
    round_idx = 0 if lora_round_idx is None else int(lora_round_idx)
    if round_idx < 0:
        raise ValueError(f"lora_round_idx must be >= 0, got {round_idx}")

    runtime_cfg = resolve_distill_runtime_config(cat_args, after_category)
    base_seed = int(getattr(cat_args, "seed", 0))
    return _ResolvedDistillStageConfig(
        device=resolve_distill_train_device(str(getattr(cat_args, "train_device", "cuda"))),
        base_seed=base_seed,
        round_idx=round_idx,
        seed=int(base_seed + round_idx),
        rank=int(runtime_cfg.rank),
        alpha=float(runtime_cfg.alpha),
        dropout=float(runtime_cfg.dropout),
        steps=int(runtime_cfg.steps),
        batch_size=int(runtime_cfg.batch_size),
        lr=float(runtime_cfg.lr),
        weight_decay=float(runtime_cfg.weight_decay),
        log_every=int(runtime_cfg.log_every),
        temperature=float(runtime_cfg.temperature),
        loss_alpha=float(runtime_cfg.loss_alpha),
        loss_type=str(runtime_cfg.loss_type),
        hidden_loss_weight=float(runtime_cfg.hidden_loss_weight),
        pre_mlp_hidden_loss_weight=float(runtime_cfg.pre_mlp_hidden_loss_weight),
        hidden_alignment_layer_weighting=str(runtime_cfg.hidden_alignment_layer_weighting),
        eakld_confidence_k=int(runtime_cfg.eakld_confidence_k),
        dataset=str(getattr(cat_args, "distill_dataset", "")).strip().lower(),
        use_dora=bool(runtime_cfg.use_dora),
        use_distill_hif4_act=bool(getattr(training_args, "distill_hif4_act", False)),
        distill_tune_final_norm=bool(getattr(cat_args, "distill_tune_final_norm", False)),
        distill_use_post_norm_head_linear=bool(getattr(cat_args, "distill_use_post_norm_head_linear", False)),
    )


def _freeze_model_for_lora(model: nn.Module, *, device: str, logger) -> Optional[bool]:
    previous_use_cache = None
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        previous_use_cache = bool(model.config.use_cache)
        model.config.use_cache = False
        logger.info("LoRA: 已关闭 model.config.use_cache。")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        logger.info("LoRA: 已启用输入梯度。")
    model.to(device)
    model.train()
    return previous_use_cache


def _restore_model_use_cache(model: nn.Module, previous_use_cache: Optional[bool], *, logger) -> None:
    if previous_use_cache is None:
        return
    if not hasattr(model, "config") or not hasattr(model.config, "use_cache"):
        return
    model.config.use_cache = bool(previous_use_cache)
    logger.info("LoRA: 已恢复 model.config.use_cache=%s。", str(bool(previous_use_cache)).lower())


def _find_module_name(model: nn.Module, target: nn.Module, fallback: str) -> str:
    for name, module in model.named_modules():
        if module is target:
            return str(name)
    return str(fallback)


def _collect_extra_trainable_modules(
    model: nn.Module,
    *,
    cfg: _ResolvedDistillStageConfig,
    logger,
) -> List[_ExtraTrainableModule]:
    modules: List[_ExtraTrainableModule] = []

    if bool(cfg.distill_tune_final_norm):
        model_type = get_model_type(model)
        final_norm = get_pre_head_layernorm(model, model_type)
        final_norm_name = _find_module_name(model, final_norm, "model.norm")
        modules.append(_ExtraTrainableModule(name=final_norm_name, module=final_norm))

    if bool(cfg.distill_use_post_norm_head_linear):
        attached = ensure_post_norm_head_linear(model)
        if attached:
            logger.info("LoRA: 已为 lm_head 挂载 identity 初始化的 post_norm_linear。")
        post_norm_linear = resolve_post_norm_linear(model)
        if post_norm_linear is None:
            raise ValueError("--distill_use_post_norm_head_linear=true but model.lm_head is not LMHeadWithPostNormLinear.")
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
    cfg: _ResolvedDistillStageConfig,
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
            "LoRA: 蒸馏参数 loss_alpha=%.4f temperature=%.4f hidden_loss_weight=%.6f pre_mlp_hidden_loss_weight=%.6f hidden_alignment_layer_weighting=%s",
            float(cfg.loss_alpha),
            float(cfg.temperature),
            float(cfg.hidden_loss_weight),
            float(cfg.pre_mlp_hidden_loss_weight),
            str(cfg.hidden_alignment_layer_weighting),
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


def _resolve_distill_dataloader_num_workers(training_args) -> int:
    raw = getattr(training_args, "distill_dataloader_num_workers", None)
    if raw is None:
        return int(default_dataloader_num_workers())
    workers = int(raw)
    if workers < 0:
        raise ValueError(f"distill_dataloader_num_workers must be >= 0, got {workers}.")
    return workers


def _build_sft_args(*, cat_args, training_args, cfg: _ResolvedDistillStageConfig, train_is_iterable: bool = False, logger=None):
    gradient_checkpointing_kwargs = None
    raw_gc_kwargs = getattr(training_args, "distill_gradient_checkpointing_kwargs", None)
    if raw_gc_kwargs is not None and str(raw_gc_kwargs).strip():
        gradient_checkpointing_kwargs = json.loads(str(raw_gc_kwargs))
        if not isinstance(gradient_checkpointing_kwargs, dict):
            raise ValueError("--distill_gradient_checkpointing_kwargs must be a JSON object.")

    requested_group_by_length = bool(getattr(training_args, "distill_group_by_length", True))
    training_kwargs = dict(
        output_dir=os.path.join(str(getattr(cat_args, "output_dir", ".result")), "lora_trainer_state"),
        per_device_train_batch_size=int(cfg.batch_size),
        gradient_accumulation_steps=int(getattr(training_args, "distill_gradient_accumulation_steps", 1)),
        gradient_checkpointing=bool(getattr(training_args, "distill_gradient_checkpointing", False)),
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        optim=_enum_to_value(getattr(training_args, "distill_optim", "paged_adamw_8bit"), "paged_adamw_8bit"),
        logging_strategy="steps",
        logging_steps=max(1, int(cfg.log_every)),
        logging_first_step=True,
        learning_rate=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        fp16=bool(getattr(training_args, "fp16", False)),
        bf16=bool(getattr(training_args, "bf16", False)),
        max_grad_norm=float(getattr(training_args, "distill_max_grad_norm", 0.3)),
        max_steps=int(cfg.steps),
        warmup_ratio=float(getattr(training_args, "distill_warmup_ratio", 0.3)),
        group_by_length=requested_group_by_length,
        lr_scheduler_type=_enum_to_value(getattr(training_args, "distill_lr_scheduler_type", "linear"), "linear"),
        report_to=[],
        disable_tqdm=not is_distill_main_process(),
        log_level="info" if is_distill_main_process() else "error",
        log_level_replica="error",
        save_strategy="no",
        seed=int(cfg.seed),
        data_seed=int(cfg.seed),
        full_determinism=bool(getattr(cat_args, "deterministic", False)),
        dataloader_num_workers=_resolve_distill_dataloader_num_workers(training_args),
        dataloader_pin_memory=True,
    )
    if train_is_iterable:
        training_kwargs["group_by_length"] = False
        if requested_group_by_length and logger is not None:
            logger.info(
                "LoRA: dataset is iterable，已忽略 --distill_group_by_length=true。"
            )
    if is_distill_distributed():
        # Only current-category params are trainable; frozen params are unused in the graph.
        training_kwargs["ddp_find_unused_parameters"] = True
        if logger is not None:
            logger.info("LoRA: DDP find_unused_parameters=True（仅当前类参数可训）。")
    return TrainingArguments(**training_kwargs)


def _distill_dataset_uses_edgerazor_messages(dataset_mix_spec: str) -> bool:
    if "=" not in str(dataset_mix_spec):
        return False
    sources, _, _ = normalize_dataset_mix_spec(str(dataset_mix_spec))
    return any(str(alias) in VAELLM_EDGERAZOR_SFT_ALIASES for alias in sources)


def _build_lora_trainer(
    *,
    model: nn.Module,
    train_ds,
    eval_ds,
    sft_args,
    training_args,
    logger,
    lora_config,
    cfg: _ResolvedDistillStageConfig,
    hif4_act_controller,
    teacher_param_snapshots,
    tokenizer=None,
    train_is_iterable: bool = False,
    use_lazy_tokenized_dataset: bool = False,
):
    max_seq_len = int(getattr(training_args, "distill_model_max_length", 2048))
    trainer_kwargs = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
        callbacks=[_LoraTrainerLogCallback(logger=logger)],
    )
    if use_lazy_tokenized_dataset:
        if tokenizer is None:
            raise ValueError("Lazy tokenized distill dataset requires tokenizer.")
        trainer_kwargs["processing_class"] = tokenizer
        trainer_kwargs["data_collator"] = build_edgerazor_data_collator(
            tokenizer,
            max_seq_len=max_seq_len,
        )
    elif tokenizer is not None and _distill_dataset_uses_edgerazor_messages(cfg.dataset):
        if DataCollatorForCompletionOnlyLM is None:
            raise ImportError("未安装 trl。EdgeRazor messages 蒸馏需要 DataCollatorForCompletionOnlyLM。")
        response_template = infer_assistant_response_template(tokenizer)
        instruction_template = infer_user_instruction_template(tokenizer)
        trainer_kwargs["processing_class"] = tokenizer
        trainer_kwargs["formatting_func"] = lambda example: render_messages(example["messages"], tokenizer)
        trainer_kwargs["data_collator"] = DataCollatorForCompletionOnlyLM(
            response_template,
            instruction_template=instruction_template,
            tokenizer=tokenizer,
            mlm=False,
        )
        trainer_kwargs["max_seq_length"] = max_seq_len
    else:
        trainer_kwargs["dataset_text_field"] = "text"
        trainer_kwargs["max_seq_length"] = max_seq_len
    del train_is_iterable
    if lora_config is not None:
        trainer_kwargs["peft_config"] = lora_config

    resolved_lora_loss = str(cfg.loss_type).strip().lower()
    hidden_loss_enabled = float(cfg.hidden_loss_weight) > 0.0
    pre_mlp_hidden_loss_enabled = float(cfg.pre_mlp_hidden_loss_weight) > 0.0
    if resolved_lora_loss not in {"", "none", "sft"} or hidden_loss_enabled or pre_mlp_hidden_loss_enabled:
        trainer_loss_type = "sft" if resolved_lora_loss in {"", "none"} else resolved_lora_loss
        trainer = CustomSFTTrainer(
            **trainer_kwargs,
            loss_type=trainer_loss_type,
            temperature=float(cfg.temperature),
            loss_alpha=float(cfg.loss_alpha),
            hidden_loss_weight=float(cfg.hidden_loss_weight),
            pre_mlp_hidden_loss_weight=float(cfg.pre_mlp_hidden_loss_weight),
            hidden_alignment_layer_weighting=str(cfg.hidden_alignment_layer_weighting),
            eakld_confidence_k=int(cfg.eakld_confidence_k),
            teacher_logits_cpu_staging=bool(
                getattr(training_args, "distill_teacher_logits_cpu_staging", False)
            ),
            distill_hif4_act_controller=hif4_act_controller,
            teacher_param_snapshots=teacher_param_snapshots,
        )
        return _replace_progress_log_callback(trainer)
    trainer = SFTTrainer(**trainer_kwargs)
    return _replace_progress_log_callback(trainer)


def _train_and_merge_lora_model(
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

    distill_distributed_barrier()
    model, merged_count = merge_all_lora(trainer.model)
    if is_distill_main_process():
        logger.info("LoRA: 微调完成并融合，融合模块数量=%d", merged_count)
    if is_distill_distributed():
        distill_distributed_barrier()
        return model
    model.to("cpu")
    torch.cuda.empty_cache()
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
    cfg = _resolve_distill_stage_config(
        cat_args=cat_args,
        training_args=training_args,
        after_category=after_category,
        lora_round_idx=lora_round_idx,
    )
    has_extra_trainables = bool(cfg.distill_tune_final_norm) or bool(cfg.distill_use_post_norm_head_linear)
    if cfg.steps <= 0:
        return model
    if not remaining_categories and not has_extra_trainables:
        return model
    _ensure_lora_stack_available()

    if not target_names and not has_extra_trainables:
        logger.info("LoRA: 没有可微调的剩余 Linear，跳过。")
        return model

    previous_use_cache = _freeze_model_for_lora(model, device=cfg.device, logger=logger)
    try:
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
        use_custom_trainer = (
            resolved_lora_loss not in {"", "none", "sft"}
            or float(cfg.hidden_loss_weight) > 0.0
            or float(cfg.pre_mlp_hidden_loss_weight) > 0.0
        )
        _log_lora_stage_start(
            logger=logger,
            cfg=cfg,
            after_category=after_category,
            remaining_categories=remaining_categories,
            target_count=len(unique_target_names),
            extra_trainable_names=extra_trainable_names,
            use_custom_trainer=use_custom_trainer,
        )

        _ensure_lora_tokenizer_ready(vae_args=vae_args, model=model)
        tokenizer = getattr(vae_args, "_cached_lora_tokenizer", None)
        dataset_mix_spec, source_stats, train_ds, eval_ds, _eval_split = prepare_distill_datasets(
            cfg.dataset,
            seed=cfg.seed,
            tokenizer=tokenizer,
            max_seq_len=int(getattr(training_args, "distill_model_max_length", 2048)),
        )
        train_is_iterable = is_iterable_training_dataset(train_ds)
        train_len = dataset_length_or_none(train_ds)
        logger.info(
            "LoRA: 补偿训练混合数据集=%s lazy_iterable=%s dataset_len=%s eval_dataset=none",
            str(dataset_mix_spec),
            str(train_is_iterable).lower(),
            "unknown" if train_len is None else str(train_len),
        )
        for source_info in source_stats:
            logger.info(
                "LoRA: 混合数据源 alias=%s weight=%.6f raw_rows=%d hf=%s config=%s train_split=%s lazy_iterable=%s",
                str(source_info["alias"]),
                float(source_info["weight"]),
                int(source_info["raw_rows"]),
                str(source_info["path"]),
                "none" if source_info["config"] is None else str(source_info["config"]),
                str(source_info["train_split"]),
                str(source_info.get("is_iterable", train_is_iterable)).lower(),
            )
        if train_len == 0:
            logger.warning("LoRA: 数据集为空，跳过。")
            model, _merged_count = merge_all_lora(model)
            return model

        sft_args = _build_sft_args(
            cat_args=cat_args,
            training_args=training_args,
            cfg=cfg,
            train_is_iterable=train_is_iterable,
            logger=logger,
        )
        hif4_act_controller = build_hif4_act_controller(cfg.use_distill_hif4_act)
        trainer = _build_lora_trainer(
            model=model,
            train_ds=train_ds,
            eval_ds=eval_ds,
            sft_args=sft_args,
            training_args=training_args,
            logger=logger,
            lora_config=lora_config,
            cfg=cfg,
            hif4_act_controller=hif4_act_controller,
            teacher_param_snapshots=teacher_param_snapshots,
            tokenizer=tokenizer,
            train_is_iterable=train_is_iterable,
            use_lazy_tokenized_dataset=True,
        )
        model = _train_and_merge_lora_model(
            trainer=trainer,
            hif4_act_controller=hif4_act_controller,
            logger=logger,
        )
        return model
    finally:
        _restore_model_use_cache(model, previous_use_cache, logger=logger)
