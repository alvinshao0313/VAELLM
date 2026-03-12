from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import math
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from transformers import AutoTokenizer, TrainingArguments
from train_utils.train_args import resolve_lora_schedule_for_category


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, loss_type: str = "r_kl_top_1000", temperature: float = 1.0, loss_alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = str(loss_type).strip().lower()
        self.temperature = float(temperature)
        self.loss_alpha = float(loss_alpha)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        args = self.args
        loss_type = self.loss_type
        teacher_inputs = dict(inputs)
        teacher_inputs.pop("labels", None)
        student_inputs = dict(inputs)
        if loss_type != "kd":
            student_inputs.pop("labels", None)
        full_inputs = dict(inputs)

        unwrapped_model = model
        if getattr(self, "accelerator", None) is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
        temporary_modules = [
            module
            for module in unwrapped_model.modules()
            if callable(getattr(module, "set_temporary", None))
        ]
        previous_temporary = [getattr(module, "temporary", None) for module in temporary_modules]
        peft_model_for_teacher = unwrapped_model if isinstance(unwrapped_model, PeftModel) else model

        def set_temporary(temporary: bool) -> None:
            for module in temporary_modules:
                module.set_temporary(temporary)

        def restore_temporary() -> None:
            for module, previous in zip(temporary_modules, previous_temporary):
                module.set_temporary(True if previous is None else bool(previous))

        def parse_k(prefix: str, default_k: int = 1000) -> int:
            if loss_type == prefix:
                return default_k
            suffix = loss_type[len(prefix):]
            if suffix.startswith("_"):
                suffix = suffix[1:]
            if not suffix:
                return default_k
            return max(1, int(suffix))

        @torch.no_grad()
        def get_ori_outputs():
            set_temporary(False)
            with peft_model_for_teacher.disable_adapter():
                outputs = model(**teacher_inputs, output_hidden_states=False)
            return outputs

        try:
            if loss_type in {"origin", "sft"}:
                try:
                    return super().compute_loss(
                        model,
                        full_inputs,
                        return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch,
                    )
                except TypeError:
                    # 兼容不支持 num_items_in_batch 的旧版 transformers/trl。
                    return super().compute_loss(
                        model,
                        full_inputs,
                        return_outputs=return_outputs,
                    )

            set_temporary(True)

            if loss_type == "rkl":
                ori_logits = get_ori_outputs().logits
                set_temporary(True)
                outputs = model(**student_inputs)
                logits = outputs.logits
                loss = F.kl_div(
                    F.log_softmax(ori_logits.flatten(0, -2), dim=-1),
                    F.softmax(logits, dim=-1).flatten(0, -2),
                    reduction="batchmean",
                )
                return (loss, outputs) if return_outputs else loss

            if loss_type == "kl":
                ori_logits = get_ori_outputs().logits
                set_temporary(True)
                outputs = model(**student_inputs)
                logits = outputs.logits
                loss = F.kl_div(
                    F.log_softmax(logits.flatten(0, -2), dim=-1),
                    F.softmax(ori_logits, dim=-1).flatten(0, -2),
                    reduction="batchmean",
                )
                return (loss, outputs) if return_outputs else loss

            if loss_type.startswith("r_kl_top"):
                k = parse_k("r_kl_top", default_k=1000)
                ori_logits = get_ori_outputs().logits
                set_temporary(True)
                outputs = model(**student_inputs)
                logits = outputs.logits
                k = min(k, int(logits.shape[-1]))
                top_logits, indices = logits.topk(k, dim=-1, sorted=False)
                top_ori_logits = ori_logits.gather(-1, indices)
                loss = F.kl_div(
                    F.log_softmax(top_ori_logits.flatten(0, -2), dim=-1),
                    F.softmax(top_logits.flatten(0, -2), dim=-1),
                    reduction="batchmean",
                )
                return (loss, outputs) if return_outputs else loss

            if loss_type.startswith("kl_top"):
                k = parse_k("kl_top", default_k=1000)
                ori_logits = get_ori_outputs().logits
                set_temporary(True)
                outputs = model(**student_inputs)
                logits = outputs.logits
                k = min(k, int(ori_logits.shape[-1]))
                top_ori_logits, indices = ori_logits.topk(k, dim=-1, sorted=False)
                if bool(getattr(args, "post_attn", False)):
                    ref = F.softmax(ori_logits, dim=-1).gather(-1, indices).flatten(0, -2)
                    can = F.log_softmax(logits, dim=-1).gather(-1, indices).flatten(0, -2)
                    loss = F.kl_div(can, ref, reduction="batchmean")
                else:
                    top_logits = logits.gather(-1, indices)
                    loss = F.kl_div(
                        F.log_softmax(top_logits, dim=-1).flatten(0, -2),
                        F.softmax(top_ori_logits, dim=-1).flatten(0, -2),
                        reduction="batchmean",
                    )
                return (loss, outputs) if return_outputs else loss

            if loss_type == "mse":
                ori_logits = get_ori_outputs().logits
                set_temporary(True)
                outputs = model(**student_inputs)
                logits = outputs.logits
                loss = F.mse_loss(logits, ori_logits)
                return (loss, outputs) if return_outputs else loss

            if loss_type == "kd":
                ori_logits = get_ori_outputs().logits
                set_temporary(True)
                outputs = model(**full_inputs)
                logits = outputs.logits
                T, alpha = self.temperature, self.loss_alpha
                ori_loss = outputs["loss"]
                logits = logits.view(-1, logits.size(-1))
                ori_logits = ori_logits.view(-1, ori_logits.size(-1))
                distill_loss = F.kl_div(
                    F.log_softmax(logits / T, dim=-1).flatten(0, -2),
                    F.softmax(ori_logits / T, dim=-1).flatten(0, -2),
                    reduction="batchmean",
                )
                loss = ori_loss * (1 - alpha) + distill_loss * (alpha * T * T)
                return (loss, outputs) if return_outputs else loss

            raise ValueError(
                f"Unsupported lora loss type: {loss_type}. "
                f"Supported: sft/origin, rkl, kl, r_kl_top[_K], kl_top[_K], mse, kd."
            )
        finally:
            restore_temporary()


def _ensure_lora_stack_available() -> None:
    if LoraConfig is None or TaskType is None or PeftModel is None:
        raise ImportError("未安装 peft。请先安装：pip install peft")
    if SFTTrainer is None or DataCollatorForCompletionOnlyLM is None:
        raise ImportError("未安装 trl。请先安装：pip install trl")
    if AutoTokenizer is None or TrainingArguments is None:
        raise ImportError("未安装 transformers。请先安装：pip install transformers")
    if load_dataset is None:
        raise ImportError("未安装 datasets。请先安装：pip install datasets")


def merge_all_lora(model: nn.Module) -> Tuple[nn.Module, int]:
    if PeftModel is None or not isinstance(model, PeftModel):
        return model, 0
    trainable_count = 0
    for name, _ in model.named_parameters():
        if "lora_" in name:
            trainable_count += 1
    merged_model = model.merge_and_unload()
    return merged_model, trainable_count


def _to_text(record: dict) -> Optional[str]:
    if "text" in record and record["text"] is not None:
        text = str(record["text"]).strip()
        if text:
            return text
    instruction = str(record.get("instruction", "")).strip()
    input_text = str(record.get("input", "")).strip()
    output = str(record.get("output", "")).strip()
    if not output and not instruction:
        return None
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def _enum_to_value(value, default: str) -> str:
    raw = value if value is not None else default
    if hasattr(raw, "value"):
        raw = raw.value
    raw = str(raw).strip()
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw.lower()


def _is_norm_param_name(name: str) -> bool:
    lower = str(name).lower()
    return "norm" in lower


def _parse_name_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _ensure_linear_bias_param(module: nn.Linear) -> bool:
    if module.bias is not None:
        return False
    weight = module.weight
    module.bias = nn.Parameter(
        torch.zeros(
            int(module.out_features),
            dtype=weight.dtype,
            device=weight.device,
        )
    )
    return True


def _resolve_lora_runtime_config(
    *,
    cat_args,
    after_category: Optional[str],
) -> Tuple[Dict[str, object], bool]:
    schedule = getattr(cat_args, "lora_schedule", None)
    merged = resolve_lora_schedule_for_category(schedule, after_category)
    return merged, bool(merged)


def lora_finetune_remaining_categories(
    model: nn.Module,
    remaining_categories: Sequence[str],
    *,
    collect_linears_fn: Callable,
    transpose_modules: Sequence[str],
    projection_suffixes: Sequence[str],
    only_decoder_projections: bool,
    cat_args,
    vae_args,
    training_args,
    logger,
    lora_round_idx: Optional[int] = None,
    after_category: Optional[str] = None,
) -> nn.Module:
    device = str(getattr(cat_args, "train_device", "cuda"))
    base_seed = int(getattr(cat_args, "seed", 0))
    if lora_round_idx is None:
        # 自动轮次：兼容旧调用方不传 round 的情况。
        auto_round = int(getattr(cat_args, "_lora_round_idx", 0))
        lora_round_idx = auto_round
        setattr(cat_args, "_lora_round_idx", auto_round + 1)
    round_idx = int(lora_round_idx)
    if round_idx < 0:
        raise ValueError(f"lora_round_idx must be >= 0, got {round_idx}")
    seed = int(base_seed + round_idx)
    rank = int(getattr(cat_args, "lora_rank", 8))
    alpha = float(getattr(cat_args, "lora_alpha", 16.0))
    dropout = float(getattr(cat_args, "lora_dropout", 0.0))
    steps = int(getattr(cat_args, "lora_steps", 0))
    batch_size = int(getattr(cat_args, "lora_batch_size", 2))
    nsamples = int(getattr(cat_args, "lora_nsamples", 128))
    lr = float(getattr(cat_args, "lora_lr", 1e-4))
    weight_decay = float(getattr(cat_args, "lora_weight_decay", 0.0))
    log_every = int(getattr(cat_args, "lora_log_every", 1))
    tune_norm = bool(getattr(cat_args, "lora_tune_norm", False))
    tune_lm_head = bool(getattr(cat_args, "lora_tune_lm_head", False))
    tune_bias = bool(getattr(cat_args, "lora_tune_bias", False))
    bias_categories = _parse_name_list(getattr(cat_args, "lora_bias_categories", []))
    lora_loss_type = str(getattr(cat_args, "lora_loss_type", "sft"))
    use_dora = bool(getattr(cat_args, "lora_use_dora", True))

    lora_overrides, has_overrides = _resolve_lora_runtime_config(
        cat_args=cat_args,
        after_category=after_category,
    )
    if has_overrides:
        rank = int(lora_overrides.get("rank", rank))
        alpha = float(lora_overrides.get("alpha", alpha))
        dropout = float(lora_overrides.get("dropout", dropout))
        steps = int(lora_overrides.get("steps", steps))
        batch_size = int(lora_overrides.get("batch_size", batch_size))
        nsamples = int(lora_overrides.get("nsamples", nsamples))
        lr = float(lora_overrides.get("lr", lr))
        weight_decay = float(lora_overrides.get("weight_decay", weight_decay))
        log_every = int(lora_overrides.get("log_every", log_every))
        tune_norm = bool(lora_overrides.get("tune_norm", tune_norm))
        tune_lm_head = bool(lora_overrides.get("tune_lm_head", tune_lm_head))
        tune_bias = bool(lora_overrides.get("tune_bias", tune_bias))
        if "bias_categories" in lora_overrides:
            bias_categories = _parse_name_list(lora_overrides.get("bias_categories", []))
        lora_loss_type = str(lora_overrides.get("loss_type", lora_loss_type))
        use_dora = bool(lora_overrides.get("use_dora", use_dora))
        logger.info(
            "LoRA: 应用按类别覆盖参数 after_category=%s, keys=%s",
            str(after_category),
            ",".join(sorted(str(k) for k in lora_overrides.keys())),
        )

    if steps <= 0 or not remaining_categories:
        return model
    _ensure_lora_stack_available()

    remaining_set = set(remaining_categories)
    current_linears = collect_linears_fn(
        model,
        transpose_modules,
        only_decoder_projections=only_decoder_projections,
        projection_suffixes=projection_suffixes,
    )
    target_names = [r.name for r in current_linears if r.category in remaining_set]
    lm_head_module = None
    if tune_lm_head:
        lm_head_module = getattr(model, "lm_head", None)
        if isinstance(lm_head_module, nn.Linear):
            target_names.append("lm_head")
        elif lm_head_module is None:
            logger.warning("LoRA: --lora_tune_lm_head 已开启，但模型不存在 lm_head，已忽略。")
        else:
            logger.warning(
                "LoRA: --lora_tune_lm_head 已开启，但 lm_head 类型为 %s（非 nn.Linear），已忽略。",
                type(lm_head_module).__name__,
            )
            lm_head_module = None
    if not target_names and not tune_norm:
        logger.info("LoRA: 没有可微调的剩余 Linear，跳过。")
        return model

    target_name_set = set(target_names)
    bias_category_set: Set[str] = set(bias_categories)
    tune_all_target_bias = len(bias_category_set) == 0
    bias_linear_by_name: Dict[str, nn.Linear] = {}
    if tune_bias:
        for ref in current_linears:
            if ref.name not in target_name_set:
                continue
            if not tune_all_target_bias and ref.category not in bias_category_set:
                continue
            bias_linear_by_name[ref.name] = ref.module
        if (
            isinstance(lm_head_module, nn.Linear)
            and "lm_head" in target_name_set
            and (tune_all_target_bias or "lm_head" in bias_category_set)
        ):
            bias_linear_by_name["lm_head"] = lm_head_module
        if bias_category_set:
            available_categories = {r.category for r in current_linears if r.name in target_name_set}
            if "lm_head" in target_name_set and isinstance(lm_head_module, nn.Linear):
                available_categories.add("lm_head")
            missing_bias_categories = sorted(bias_category_set - available_categories)
            if missing_bias_categories:
                logger.warning(
                    "LoRA: bias_categories 未匹配到可训练目标，已忽略: %s",
                    ",".join(missing_bias_categories),
                )
        if not bias_linear_by_name:
            logger.warning("LoRA: --lora_tune_bias 已开启，但未匹配到可训练 bias 的 Linear。")

    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        logger.info("LoRA: 已启用输入梯度。")
    model.to(device)
    model.train()

    created_bias_count = 0
    if tune_bias:
        for linear in bias_linear_by_name.values():
            if _ensure_linear_bias_param(linear):
                created_bias_count += 1

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(rank),
        lora_alpha=float(alpha),
        lora_dropout=float(dropout),
        target_modules=sorted(set(target_names)),
        inference_mode=False,
        bias="none",
        use_dora=bool(use_dora),
    )
    model = get_peft_model(model, lora_config)

    bias_trainable_count = 0
    if tune_bias:
        for linear in bias_linear_by_name.values():
            if linear.bias is None:
                continue
            linear.bias.requires_grad = True
            bias_trainable_count += 1
        logger.info(
            "LoRA: bias 训练已启用，目标模块=%d，新增 bias=%d，bias_categories=%s",
            len(bias_linear_by_name),
            created_bias_count,
            "all_target_modules" if tune_all_target_bias else ",".join(sorted(bias_category_set)),
        )

    norm_trainable_count = 0
    if tune_norm:
        for name, param in model.named_parameters():
            if _is_norm_param_name(name):
                param.requires_grad = True
                norm_trainable_count += 1
        logger.info("LoRA: 已额外解冻 norm 参数，数量=%d", norm_trainable_count)

    resolved_lora_loss = str(lora_loss_type).strip().lower() if lora_loss_type is not None else "sft"
    use_custom_trainer = resolved_lora_loss not in {"", "none", "sft"}
    if use_custom_trainer:
        logger.info(
            "LoRA: 使用 CustomSFTTrainer 微调，loss_type=%s，use_dora=%s，tune_norm=%s，tune_lm_head=%s，tune_bias=%s，目标类别=%s，目标模块=%d，bias模块=%d，rank=%d，alpha=%.2f，steps=%d，batch_size=%d，seed(base=%d,round=%d,effective=%d)",
            resolved_lora_loss,
            str(use_dora).lower(),
            str(tune_norm).lower(),
            str(tune_lm_head).lower(),
            str(tune_bias).lower(),
            ",".join(remaining_categories),
            len(set(target_names)),
            int(bias_trainable_count),
            int(rank),
            float(alpha),
            int(steps),
            int(batch_size),
            int(base_seed),
            int(round_idx),
            int(seed),
        )
    else:
        logger.info(
            "LoRA: 使用 SFTTrainer 微调，use_dora=%s，tune_norm=%s，tune_lm_head=%s，tune_bias=%s，目标类别=%s，目标模块=%d，bias模块=%d，rank=%d，alpha=%.2f，steps=%d，batch_size=%d，seed(base=%d,round=%d,effective=%d)",
            str(use_dora).lower(),
            str(tune_norm).lower(),
            str(tune_lm_head).lower(),
            str(tune_bias).lower(),
            ",".join(remaining_categories),
            len(set(target_names)),
            int(bias_trainable_count),
            int(rank),
            float(alpha),
            int(steps),
            int(batch_size),
            int(base_seed),
            int(round_idx),
            int(seed),
        )
    # dataset_dict = load_dataset("vicgalle/alpaca-gpt4")
    dataset_dict = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_ds = dataset_dict["train"]
    if int(nsamples) > 0:
        train_ds = train_ds.shuffle(seed=int(seed)).select(range(min(int(nsamples), len(train_ds))))
    train_ds = train_ds.map(lambda rec: {"text": _to_text(rec)})
    train_ds = train_ds.filter(lambda rec: rec["text"] is not None and len(rec["text"]) > 0)
    if len(train_ds) == 0:
        logger.warning("LoRA: 数据集为空，跳过。")
        return model

    if "validation" in dataset_dict:
        eval_ds = dataset_dict["validation"]
    elif "test" in dataset_dict:
        eval_ds = dataset_dict["test"]
    else:
        eval_ds = train_ds
    eval_ds = eval_ds.map(lambda rec: {"text": _to_text(rec)})
    eval_ds = eval_ds.filter(lambda rec: rec["text"] is not None and len(rec["text"]) > 0)

    tokenizer = AutoTokenizer.from_pretrained(
        vae_args.model_path,
        use_fast=True,
        token=getattr(vae_args, "access_token", None),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    sft_args = TrainingArguments(
        output_dir=getattr(training_args, "output_dir", ".result"),
        num_train_epochs=float(getattr(training_args, "num_train_epochs", 1.0)),
        per_device_train_batch_size=int(batch_size),
        gradient_accumulation_steps=int(getattr(training_args, "gradient_accumulation_steps", 1)),
        optim=_enum_to_value(getattr(training_args, "optim", "paged_adamw_8bit"), "paged_adamw_8bit"),
        logging_strategy="steps",
        logging_steps=max(1, int(log_every)),
        logging_first_step=True,
        learning_rate=float(lr),
        weight_decay=float(weight_decay),
        fp16=bool(getattr(training_args, "fp16", False)),
        bf16=bool(getattr(training_args, "bf16", False)),
        max_grad_norm=float(getattr(training_args, "max_grad_norm", 0.3)),
        max_steps=int(steps),
        warmup_ratio=float(getattr(training_args, "warmup_ratio", 0.3)),
        group_by_length=bool(getattr(training_args, "group_by_length", True)),
        lr_scheduler_type=_enum_to_value(getattr(training_args, "lr_scheduler_type", "linear"), "linear"),
        report_to=[],
        disable_tqdm=False,
        save_strategy="no",
        seed=int(seed),
    )

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=int(getattr(training_args, "model_max_length", 2048)),
    )
    if use_custom_trainer:
        trainer = CustomSFTTrainer(
            **trainer_kwargs,
            loss_type=resolved_lora_loss,
        )
    else:
        trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    model, merged_count = merge_all_lora(trainer.model)
    model.to("cpu")
    torch.cuda.empty_cache()
    logger.info("LoRA: 微调完成并融合，融合模块数量=%d", merged_count)
    return model
