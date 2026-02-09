import time
from typing import Callable, List, Sequence, Tuple

import torch

from torch import nn

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:  # pragma: no cover
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None


def _ensure_peft_available() -> None:
    if get_peft_model is None:
        raise ImportError(
            "未安装 peft。请先安装：pip install peft"
        )


def inject_lora_to_linear_names(
    model: nn.Module,
    linear_names: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> Tuple[nn.Module, List[str]]:
    _ensure_peft_available()
    target_names = sorted(set(linear_names))
    if not target_names:
        return model, []

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(rank),
        lora_alpha=float(alpha),
        lora_dropout=float(dropout),
        target_modules=target_names,
        inference_mode=False,
        bias="none",
    )
    peft_model = get_peft_model(model, peft_config)

    injected_names: List[str] = []
    for module_name, _ in peft_model.named_modules():
        if any(module_name.endswith(name) for name in target_names):
            injected_names.append(module_name)
    return peft_model, sorted(set(injected_names))


def collect_lora_parameters(model: nn.Module):
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)
    return params


def merge_all_lora(model: nn.Module) -> Tuple[nn.Module, int]:
    if PeftModel is None or not isinstance(model, PeftModel):
        return model, 0
    trainable_count = 0
    for name, _ in model.named_parameters():
        if "lora_" in name:
            trainable_count += 1
    merged_model = model.merge_and_unload()
    return merged_model, trainable_count


def lora_finetune_remaining_categories(
    model: nn.Module,
    remaining_categories: Sequence[str],
    *,
    collect_linears_fn: Callable,
    transpose_modules: Sequence[str],
    projection_suffixes: Sequence[str],
    only_decoder_projections: bool,
    vae_args,
    training_args,
    device: str,
    seed: int,
    rank: int,
    alpha: float,
    dropout: float,
    steps: int,
    batch_size: int,
    nsamples: int,
    lr: float,
    weight_decay: float,
    log_every: int,
    logger,
) -> nn.Module:
    if steps <= 0 or not remaining_categories:
        return model

    from train_utils.data_utils import get_wikitext2

    remaining_set = set(remaining_categories)
    current_linears = collect_linears_fn(
        model,
        transpose_modules,
        only_decoder_projections=only_decoder_projections,
        projection_suffixes=projection_suffixes,
    )
    target_names = [r.name for r in current_linears if r.category in remaining_set]
    if not target_names:
        logger.info("LoRA: 没有可微调的剩余 Linear，跳过。")
        return model

    for p in model.parameters():
        p.requires_grad = False

    model, injected_names = inject_lora_to_linear_names(
        model=model,
        linear_names=target_names,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    if not injected_names:
        logger.info("LoRA: 注入数量为 0，跳过。")
        return model

    lora_params = collect_lora_parameters(model)
    if not lora_params:
        logger.info("LoRA: 未收集到可训练参数，跳过。")
        return model

    logger.info(
        "LoRA: 开始微调，目标类别=%s，注入模块=%d，steps=%d，batch_size=%d",
        ",".join(remaining_categories),
        len(injected_names),
        steps,
        batch_size,
    )

    seq_len = getattr(model, "seqlen", int(getattr(training_args, "model_max_length", 2048)))
    train_samples = get_wikitext2(
        nsamples=max(nsamples, batch_size),
        seed=seed,
        seqlen=seq_len,
        model=vae_args.model_path,
    )
    if not train_samples:
        logger.warning("LoRA: 训练样本为空，跳过。")
        model, merged_count = merge_all_lora(model)
        logger.info("LoRA: 已融合模块数量=%d", merged_count)
        return model

    use_bf16 = bool(getattr(training_args, "bf16", False))
    use_fp16 = bool(getattr(training_args, "fp16", False))
    amp_enabled = use_bf16 or use_fp16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    model.train()
    model.to(device)

    start = time.time()
    total_samples = len(train_samples)
    for step_idx in range(int(steps)):
        batch_tensors = []
        for batch_idx in range(int(batch_size)):
            sample_idx = (step_idx * int(batch_size) + batch_idx) % total_samples
            input_ids = train_samples[sample_idx][0]
            batch_tensors.append(input_ids)
        batch = torch.cat(batch_tensors, dim=0).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(batch, labels=batch)
            loss = outputs.loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if log_every > 0 and (step_idx + 1) % int(log_every) == 0:
            speed = (time.time() - start) / int(log_every)
            logger.info(
                "LoRA step=%d/%d loss=%.6f speed=%.4fs/it",
                step_idx + 1,
                steps,
                float(loss.detach().float().item()),
                speed,
            )
            start = time.time()

    model, merged_count = merge_all_lora(model)
    model.to("cpu")
    torch.cuda.empty_cache()
    logger.info("LoRA: 微调完成并融合，融合模块数量=%d", merged_count)
    return model
