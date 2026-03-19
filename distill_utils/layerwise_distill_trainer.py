import argparse
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from distill_utils.layerwise_distill_hooks import LayerIOHook
from distill_utils.layerwise_distill_runtime import (
    build_layer_runtime_kwargs,
    build_shared_layer0_inputs,
    estimate_layer_cache_bytes,
    get_base_model,
    resolve_dtype,
)
from litebsq.vae_linear import VAELinear, clear_model_vae_linear_cache
from train_utils.train_args import create_optimizer


def freeze_student(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        if isinstance(module, VAELinear):
            module.cache_decoded_weight = True


def collect_layer_trainables(
    layer: nn.Module,
    *,
    train_bias: bool,
    train_o_proj_bias: bool,
    train_layernorm_weight: bool,
) -> List[Tuple[str, nn.Parameter]]:
    trainables: List[Tuple[str, nn.Parameter]] = []
    seen: Set[int] = set()

    def _append(name: str, p):
        if not isinstance(p, nn.Parameter):
            return
        pid = id(p)
        if pid in seen:
            return
        seen.add(pid)
        p.requires_grad = True
        trainables.append((name, p))

    for mod_name, module in layer.named_modules():
        if not isinstance(module, VAELinear):
            continue

        module.cache_decoded_weight = False
        module.clear_decoded_weight_cache()

        if hasattr(module, "decoder"):
            for pn, p in module.decoder.named_parameters():
                _append(f"{mod_name}.decoder.{pn}", p)
        elif hasattr(module, "decoders"):
            for i, dec in enumerate(module.decoders):
                for pn, p in dec.named_parameters():
                    _append(f"{mod_name}.decoders.{i}.{pn}", p)

        if train_bias:
            _append(f"{mod_name}.bias", module.bias)

    if train_o_proj_bias:
        for mod_name, module in layer.named_modules():
            if not str(mod_name).endswith("o_proj"):
                continue
            if not isinstance(module, (VAELinear, nn.Linear)):
                continue
            bias = getattr(module, "bias", None)
            if bias is None:
                out_features = getattr(module, "out_features", None)
                if out_features is None:
                    continue
                ref_weight = getattr(module, "original_weight", None)
                if ref_weight is None:
                    ref_weight = getattr(module, "weight", None)
                if isinstance(ref_weight, torch.Tensor):
                    device = ref_weight.device
                    dtype = ref_weight.dtype
                else:
                    device = torch.device("cpu")
                    dtype = torch.float32
                bias = nn.Parameter(torch.zeros(int(out_features), device=device, dtype=dtype))
                module.bias = bias
            _append(f"{mod_name}.bias", bias)

    if train_layernorm_weight:
        for norm_name in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_attention_layernorm",
            "post_feedforward_layernorm",
        ]:
            norm_module = getattr(layer, norm_name, None)
            if norm_module is None:
                continue
            _append(f"{norm_name}.weight", getattr(norm_module, "weight", None))

    return trainables


def snapshot_trainables(trainables: Sequence[Tuple[str, nn.Parameter]]) -> Dict[str, torch.Tensor]:
    snapshot: Dict[str, torch.Tensor] = {}
    for name, p in trainables:
        snapshot[name] = p.detach().float().clone()
    return snapshot


def anchor_loss(trainables: Sequence[Tuple[str, nn.Parameter]], snapshot: Dict[str, torch.Tensor]) -> torch.Tensor:
    terms = []
    for name, p in trainables:
        terms.append((p.float() - snapshot[name]).pow(2).mean())
    if not terms:
        raise RuntimeError("anchor loss requested with empty trainables.")
    return torch.stack(terms).mean()


def _resolve_attn_block_module(layer: nn.Module) -> nn.Module:
    for attr in ("self_attn", "attention", "attn"):
        mod = getattr(layer, attr, None)
        if isinstance(mod, nn.Module):
            return mod
    raise RuntimeError(f"Cannot resolve attention block from layer: {type(layer)}")


def _mean_seq_features(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise RuntimeError(f"Expected attention block output with ndim=3, got shape={tuple(x.shape)}")
    return x.float().mean(dim=1)


def _attention_weight_kl_loss(
    student_attn: torch.Tensor,
    teacher_attn: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    student_prob = student_attn.float().clamp_min(eps)
    teacher_prob = teacher_attn.float().clamp_min(eps)
    student_prob = student_prob / student_prob.sum(dim=-1, keepdim=True).clamp_min(eps)
    teacher_prob = teacher_prob / teacher_prob.sum(dim=-1, keepdim=True).clamp_min(eps)
    kl = F.kl_div(student_prob.log(), teacher_prob, reduction="none")
    return kl.sum(dim=-1).mean()


def _wandb_log(wandb_run, metrics: Dict[str, float]) -> None:
    if wandb_run is None or not metrics:
        return
    wandb_run.log(metrics)


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TiB"


def _move_layer_runtime_to_device(layer: nn.Module, base_model: nn.Module, device: str) -> None:
    layer.to(device)
    base_model.rotary_emb.to(device)


def _move_layer_runtime_to_cpu(layer: nn.Module, base_model: nn.Module) -> None:
    layer.to("cpu")
    base_model.rotary_emb.to("cpu")


def _validate_layer_cache_budget(model_t: nn.Module, calib_inputs: torch.Tensor, args) -> None:
    cache_dtype = resolve_dtype(args.cache_dtype)
    teacher_label_dtype = resolve_dtype(args.teacher_label_dtype)
    estimates = estimate_layer_cache_bytes(
        num_samples=int(calib_inputs.shape[0]),
        seqlen=int(calib_inputs.shape[1]),
        hidden_size=int(model_t.config.hidden_size),
        num_attention_heads=int(model_t.config.num_attention_heads),
        cache_dtype=cache_dtype,
        teacher_label_dtype=teacher_label_dtype,
        extra_teacher_out=float(getattr(args, "lambda_aug_loss", 0.0)) > 0.0,
    )
    limit = int(float(args.memory_safety_factor) * float(estimates["available"]))
    if int(estimates["total"]) <= int(limit):
        return

    details = ", ".join(
        f"{key}={_human_bytes(value)}"
        for key, value in estimates.items()
        if key not in {"total", "available"}
    )
    raise RuntimeError(
        "CPU memory budget exceeded for layer cache. "
        f"required={_human_bytes(estimates['total'])}, "
        f"available={_human_bytes(estimates['available'])}, "
        f"safety_limit={_human_bytes(limit)}, "
        f"details=[{details}]"
    )


def _forward_layer(
    *,
    model,
    layer: nn.Module,
    hidden_states: torch.Tensor,
    output_attentions: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    layer_param = next(layer.parameters(), None)
    if layer_param is not None and hidden_states.dtype != layer_param.dtype:
        hidden_states = hidden_states.to(dtype=layer_param.dtype)
    runtime_kwargs = build_layer_runtime_kwargs(
        model=model,
        hidden_states=hidden_states,
        output_attentions=output_attentions,
    )
    outputs = layer(
        hidden_states,
        attention_mask=runtime_kwargs["attention_mask"],
        position_ids=runtime_kwargs["position_ids"],
        past_key_value=None,
        output_attentions=bool(output_attentions),
        use_cache=False,
        cache_position=runtime_kwargs["cache_position"],
        position_embeddings=runtime_kwargs["position_embeddings"],
    )
    if not isinstance(outputs, (tuple, list)) or not outputs:
        raise RuntimeError(f"Unexpected layer output type: {type(outputs)}")
    hidden = outputs[0]
    if not isinstance(hidden, torch.Tensor):
        raise RuntimeError(f"Unexpected hidden output type: {type(hidden)}")
    attn = None
    if output_attentions:
        if len(outputs) < 2 or not isinstance(outputs[1], torch.Tensor):
            raise RuntimeError("Attention output is unavailable from direct layer forward.")
        attn = outputs[1]
    return hidden, attn


def _cache_teacher_layer_targets(
    *,
    model_t,
    layer_t: nn.Module,
    teacher_hidden_cpu: torch.Tensor,
    batch_size: int,
    distill_device: str,
    cache_dtype: torch.dtype,
    teacher_label_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_model_t = get_base_model(model_t)
    num_samples, seqlen, hidden_size = teacher_hidden_cpu.shape
    num_heads = int(model_t.config.num_attention_heads)
    teacher_out_cpu = torch.empty((num_samples, seqlen, hidden_size), dtype=cache_dtype, device="cpu")
    teacher_attn_cpu = torch.empty((num_samples, num_heads, seqlen, seqlen), dtype=teacher_label_dtype, device="cpu")
    teacher_attn_mean_cpu = torch.empty((num_samples, hidden_size), dtype=teacher_label_dtype, device="cpu")
    attn_hook = LayerIOHook(_resolve_attn_block_module(layer_t))

    _move_layer_runtime_to_device(layer_t, base_model_t, distill_device)
    try:
        with torch.inference_mode():
            for begin in range(0, int(num_samples), int(batch_size)):
                end = min(begin + int(batch_size), int(num_samples))
                teacher_hidden = teacher_hidden_cpu[begin:end].to(distill_device, non_blocking=True)
                attn_hook.clear()
                teacher_out, teacher_attn = _forward_layer(
                    model=model_t,
                    layer=layer_t,
                    hidden_states=teacher_hidden,
                    output_attentions=True,
                )
                teacher_attn_block = attn_hook.pop_output(detach=True)
                teacher_out_cpu[begin:end].copy_(teacher_out.to(device="cpu", dtype=cache_dtype))
                teacher_attn_cpu[begin:end].copy_(teacher_attn.to(device="cpu", dtype=teacher_label_dtype))
                teacher_attn_mean_cpu[begin:end].copy_(
                    _mean_seq_features(teacher_attn_block).to(device="cpu", dtype=teacher_label_dtype)
                )
                del teacher_hidden, teacher_out, teacher_attn, teacher_attn_block
    finally:
        attn_hook.remove()
        clear_model_vae_linear_cache(model_t)
        _move_layer_runtime_to_cpu(layer_t, base_model_t)
        _clear_cuda_cache()

    return teacher_out_cpu, teacher_attn_cpu, teacher_attn_mean_cpu


def _cache_teacher_layer_outputs_only(
    *,
    model_t,
    layer_t: nn.Module,
    hidden_cpu: torch.Tensor,
    batch_size: int,
    distill_device: str,
    cache_dtype: torch.dtype,
) -> torch.Tensor:
    base_model_t = get_base_model(model_t)
    teacher_out_cpu = torch.empty_like(hidden_cpu, dtype=cache_dtype, device="cpu")

    _move_layer_runtime_to_device(layer_t, base_model_t, distill_device)
    try:
        with torch.inference_mode():
            for begin in range(0, int(hidden_cpu.shape[0]), int(batch_size)):
                end = min(begin + int(batch_size), int(hidden_cpu.shape[0]))
                hidden = hidden_cpu[begin:end].to(distill_device, non_blocking=True)
                teacher_out, _ = _forward_layer(
                    model=model_t,
                    layer=layer_t,
                    hidden_states=hidden,
                    output_attentions=False,
                )
                teacher_out_cpu[begin:end].copy_(teacher_out.to(device="cpu", dtype=cache_dtype))
                del hidden, teacher_out
    finally:
        clear_model_vae_linear_cache(model_t)
        _move_layer_runtime_to_cpu(layer_t, base_model_t)
        _clear_cuda_cache()

    return teacher_out_cpu


def _train_student_layer(
    *,
    model_q,
    layer_q: nn.Module,
    teacher_hidden_cpu: torch.Tensor,
    teacher_out_cpu: torch.Tensor,
    teacher_aug_out_cpu: Optional[torch.Tensor],
    teacher_attn_cpu: torch.Tensor,
    teacher_attn_mean_cpu: torch.Tensor,
    student_hidden_cpu: torch.Tensor,
    layer_id: int,
    order: int,
    total_layers: int,
    args,
    log,
    wandb_run,
    global_step: int,
) -> int:
    base_model_q = get_base_model(model_q)
    _move_layer_runtime_to_device(layer_q, base_model_q, args.distill_device)

    freeze_student(model_q)
    clear_model_vae_linear_cache(model_q)
    trainables = collect_layer_trainables(
        layer_q,
        train_bias=bool(args.train_bias),
        train_o_proj_bias=bool(getattr(args, "train_o_proj_bias", False)),
        train_layernorm_weight=bool(args.train_layernorm_weight),
    )
    n_params = sum(int(p.numel()) for _name, p in trainables)
    total_steps = int(args.steps_per_layer)
    if total_steps < 1:
        raise ValueError("steps_per_layer must be >= 1.")

    log.info(
        "[L%d][%d/%d] start: trainable_tensors=%d trainable_params=%d steps=%d",
        layer_id,
        order,
        total_layers,
        len(trainables),
        n_params,
        total_steps,
    )
    _wandb_log(
        wandb_run,
        {
            "train/global_step": float(global_step),
            "layer/id": float(layer_id),
            "layer/order": float(order),
            "layer/started": 1.0,
            "layer/trainable_tensors": float(len(trainables)),
            "layer/trainable_params": float(n_params),
            "layer/total_steps": float(total_steps),
        },
    )

    if not trainables:
        log.warning("[L%d] no trainable parameters found, skip optimizer step.", layer_id)
        return global_step

    snapshot = snapshot_trainables(trainables)
    optimizer = create_optimizer([p for _name, p in trainables], argparse.Namespace(
        optimizer=args.optimizer,
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        weight_decay=float(args.weight_decay),
    ), float(args.lr))
    attn_hook = LayerIOHook(_resolve_attn_block_module(layer_q))
    num_samples = int(student_hidden_cpu.shape[0])
    num_batches = math.ceil(num_samples / int(args.batch_size))
    sample_order = torch.arange(num_samples, dtype=torch.long)

    try:
        for step in range(total_steps):
            batch_pos = step % num_batches
            if batch_pos == 0:
                sample_order = torch.randperm(num_samples) if bool(
                    args.shuffle) else torch.arange(num_samples, dtype=torch.long)

            begin = batch_pos * int(args.batch_size)
            end = min(begin + int(args.batch_size), num_samples)
            batch_idx = sample_order[begin:end]

            q_in = student_hidden_cpu.index_select(0, batch_idx).to(args.distill_device, non_blocking=True)
            t_in = teacher_hidden_cpu.index_select(0, batch_idx).to(args.distill_device, non_blocking=True)
            t_out = teacher_out_cpu.index_select(0, batch_idx).to(args.distill_device, non_blocking=True)
            t_aug_out = None
            if teacher_aug_out_cpu is not None:
                t_aug_out = teacher_aug_out_cpu.index_select(0, batch_idx).to(args.distill_device, non_blocking=True)
            t_attn = teacher_attn_cpu.index_select(0, batch_idx).to(args.distill_device, non_blocking=True)
            t_attn_mean = teacher_attn_mean_cpu.index_select(0, batch_idx).to(args.distill_device, non_blocking=True)

            attn_hook.clear()
            q_out, q_attn = _forward_layer(
                model=model_q,
                layer=layer_q,
                hidden_states=q_in,
                output_attentions=True,
            )
            q_attn_block = attn_hook.pop_output(detach=False)

            t_in = t_in.to(device=q_in.device, dtype=q_in.dtype)
            t_out = t_out.to(device=q_out.device, dtype=q_out.dtype)
            if t_aug_out is not None:
                t_aug_out = t_aug_out.to(device=q_out.device, dtype=q_out.dtype)
            t_attn = t_attn.to(device=q_attn.device, dtype=q_attn.dtype)
            t_attn_mean = t_attn_mean.to(device=q_attn_block.device, dtype=q_attn_block.dtype)

            loss_blk = F.mse_loss((q_out - q_in).float(), (t_out - t_in).float())
            loss_res = F.mse_loss(q_out.float(), t_out.float())
            if t_aug_out is not None:
                loss_aug = F.mse_loss(q_out.float(), t_aug_out.float())
            else:
                loss_aug = q_out.new_zeros(())
            loss_anchor = anchor_loss(trainables, snapshot)
            loss_norm = F.mse_loss(
                q_out.float().mean(dim=(0, 1)),
                t_out.float().mean(dim=(0, 1)),
            )
            loss_attn_map = _attention_weight_kl_loss(q_attn, t_attn)
            loss_attn_block_mean = F.mse_loss(
                _mean_seq_features(q_attn_block),
                t_attn_mean.float(),
            )
            loss = (
                float(args.lambda_blk) * loss_blk
                + float(args.lambda_res) * loss_res
                + float(getattr(args, "lambda_aug_loss", 0.0)) * loss_aug
                + float(args.lambda_anchor) * loss_anchor
                + float(args.lambda_norm) * loss_norm
                + float(args.lambda_attn_map) * loss_attn_map
                + float(args.lambda_attn_block_mean) * loss_attn_block_mean
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1

            _wandb_log(
                wandb_run,
                {
                    "train/global_step": float(global_step),
                    "train/layer_step": float(step + 1),
                    "train/layer_id": float(layer_id),
                    "train/layer_order": float(order),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "loss/total": float(loss.detach().item()),
                    "loss/blk": float(loss_blk.detach().item()),
                    "loss/res": float(loss_res.detach().item()),
                    "loss/aug": float(loss_aug.detach().item()),
                    "loss/anchor": float(loss_anchor.detach().item()),
                    "loss/norm": float(loss_norm.detach().item()),
                    "loss/attn_map": float(loss_attn_map.detach().item()),
                    "loss/attn_block_mean": float(loss_attn_block_mean.detach().item()),
                },
            )

            if int(args.log_every) > 0 and ((step + 1) % int(args.log_every) == 0 or step == 0 or (step + 1) == total_steps):
                log.info(
                    "[L%d] step %d/%d loss=%.3e blk=%.3e res=%.3e aug=%.3e anchor=%.3e norm=%.3e attn_map=%.3e attn_blk_mean=%.3e",
                    layer_id,
                    step + 1,
                    total_steps,
                    float(loss.detach().item()),
                    float(loss_blk.detach().item()),
                    float(loss_res.detach().item()),
                    float(loss_aug.detach().item()),
                    float(loss_anchor.detach().item()),
                    float(loss_norm.detach().item()),
                    float(loss_attn_map.detach().item()),
                    float(loss_attn_block_mean.detach().item()),
                )

            del q_in, t_in, t_out, t_attn, t_attn_mean, q_out, q_attn, q_attn_block
            del t_aug_out
            del loss, loss_blk, loss_res, loss_aug, loss_anchor, loss_norm, loss_attn_map, loss_attn_block_mean
    finally:
        attn_hook.remove()
        del optimizer
        del snapshot

    return global_step


def _rollout_student_layer(
    *,
    model_q,
    layer_q: nn.Module,
    student_hidden_cpu: torch.Tensor,
    batch_size: int,
    distill_device: str,
    cache_dtype: torch.dtype,
) -> torch.Tensor:
    student_next_hidden_cpu = torch.empty_like(student_hidden_cpu, dtype=cache_dtype, device="cpu")
    for begin in range(0, int(student_hidden_cpu.shape[0]), int(batch_size)):
        end = min(begin + int(batch_size), int(student_hidden_cpu.shape[0]))
        student_hidden = student_hidden_cpu[begin:end].to(distill_device, non_blocking=True)
        with torch.inference_mode():
            student_out, _ = _forward_layer(
                model=model_q,
                layer=layer_q,
                hidden_states=student_hidden,
                output_attentions=False,
            )
        student_next_hidden_cpu[begin:end].copy_(student_out.to(device="cpu", dtype=cache_dtype))
        del student_hidden, student_out
    return student_next_hidden_cpu


def _cleanup_student_layer_runtime(model_q, layer_q: nn.Module) -> None:
    base_model_q = get_base_model(model_q)
    clear_model_vae_linear_cache(model_q)
    _move_layer_runtime_to_cpu(layer_q, base_model_q)
    _clear_cuda_cache()


def _eval_student_ppl_after_layer(model_q: nn.Module, args, log, layer_id: int) -> Optional[Dict[str, float]]:
    from train_utils.eval_utils import calculate_ppl

    ppl_args = argparse.Namespace(
        model_path=str(getattr(args, "model_path", getattr(args, "teacher_model_path", ""))),
        seqlen=int(getattr(args, "ppl_seqlen", 2048)),
        limit=int(getattr(args, "ppl_limit", -1)),
    )
    if not ppl_args.model_path:
        log.warning("[L%d] skipped PPL eval: missing model_path.", layer_id)
        return None

    log.info(
        "[L%d] start student PPL eval (seqlen=%d, limit=%d)...",
        layer_id,
        int(ppl_args.seqlen),
        int(ppl_args.limit),
    )
    model_q.to(args.distill_device)
    try:
        with torch.no_grad():
            ppl_result = calculate_ppl(model_q, ppl_args)
    finally:
        model_q.to("cpu")
        _clear_cuda_cache()

    log.info(
        "[L%d] student PPL=%.4f (nsamples=%d, seqlen=%d)",
        layer_id,
        float(ppl_result.get("wiki_ppl", float("nan"))),
        int(ppl_result.get("nsamples", 0)),
        int(ppl_result.get("seqlen", int(ppl_args.seqlen))),
    )
    return {
        "wiki_ppl": float(ppl_result.get("wiki_ppl", float("nan"))),
        "nsamples": float(ppl_result.get("nsamples", 0)),
        "seqlen": float(ppl_result.get("seqlen", int(ppl_args.seqlen))),
    }


def distill_layers(
    *,
    model_q: nn.Module,
    model_t: nn.Module,
    layers_q: Sequence[nn.Module],
    layers_t: Sequence[nn.Module],
    layer_indices: Sequence[int],
    calib_inputs: torch.Tensor,
    args,
    log,
    wandb_run=None,
) -> None:
    cache_dtype = resolve_dtype(args.cache_dtype)
    teacher_label_dtype = resolve_dtype(args.teacher_label_dtype)
    teacher_hidden_cpu = build_shared_layer0_inputs(
        model=model_t,
        input_ids=calib_inputs,
        cache_dtype=cache_dtype,
    )
    student_hidden_cpu = teacher_hidden_cpu

    log.info(
        "Sequential layer-wise distill: device=%s cache_dtype=%s teacher_label_dtype=%s seqlen=%d",
        args.distill_device,
        str(cache_dtype),
        str(teacher_label_dtype),
        int(calib_inputs.shape[1]),
    )

    global_step = 0
    for order, layer_id in enumerate(layer_indices, start=1):
        _validate_layer_cache_budget(model_t, calib_inputs, args)
        layer_q = layers_q[layer_id]
        layer_t = layers_t[layer_id]

        teacher_out_cpu, teacher_attn_cpu, teacher_attn_mean_cpu = _cache_teacher_layer_targets(
            model_t=model_t,
            layer_t=layer_t,
            teacher_hidden_cpu=teacher_hidden_cpu,
            batch_size=int(args.batch_size),
            distill_device=args.distill_device,
            cache_dtype=cache_dtype,
            teacher_label_dtype=teacher_label_dtype,
        )
        teacher_aug_out_cpu = None
        if float(getattr(args, "lambda_aug_loss", 0.0)) > 0.0:
            teacher_aug_out_cpu = _cache_teacher_layer_outputs_only(
                model_t=model_t,
                layer_t=layer_t,
                hidden_cpu=student_hidden_cpu,
                batch_size=int(args.batch_size),
                distill_device=args.distill_device,
                cache_dtype=cache_dtype,
            )

        try:
            global_step = _train_student_layer(
                model_q=model_q,
                layer_q=layer_q,
                teacher_hidden_cpu=teacher_hidden_cpu,
                teacher_out_cpu=teacher_out_cpu,
                teacher_aug_out_cpu=teacher_aug_out_cpu,
                teacher_attn_cpu=teacher_attn_cpu,
                teacher_attn_mean_cpu=teacher_attn_mean_cpu,
                student_hidden_cpu=student_hidden_cpu,
                layer_id=layer_id,
                order=order,
                total_layers=len(layer_indices),
                args=args,
                log=log,
                wandb_run=wandb_run,
                global_step=global_step,
            )

            student_next_hidden_cpu = _rollout_student_layer(
                model_q=model_q,
                layer_q=layer_q,
                student_hidden_cpu=student_hidden_cpu,
                batch_size=int(args.batch_size),
                distill_device=args.distill_device,
                cache_dtype=cache_dtype,
            )
        finally:
            _cleanup_student_layer_runtime(model_q, layer_q)

        old_teacher_hidden_cpu = teacher_hidden_cpu
        old_student_hidden_cpu = student_hidden_cpu
        teacher_hidden_cpu = teacher_out_cpu
        student_hidden_cpu = student_next_hidden_cpu
        del old_teacher_hidden_cpu, old_student_hidden_cpu, teacher_attn_cpu, teacher_attn_mean_cpu, teacher_aug_out_cpu

        ppl_result = None
        if not bool(getattr(args, "skip_ppl_eval", False)):
            ppl_result = _eval_student_ppl_after_layer(model_q, args, log, layer_id)

        layer_metrics = {
            "train/global_step": float(global_step),
            "layer/id": float(layer_id),
            "layer/order": float(order),
            "layer/completed": 1.0,
        }
        if ppl_result is not None:
            layer_metrics["eval/wiki_ppl"] = float(ppl_result["wiki_ppl"])
            layer_metrics["eval/nsamples"] = float(ppl_result["nsamples"])
            layer_metrics["eval/seqlen"] = float(ppl_result["seqlen"])
        _wandb_log(wandb_run, layer_metrics)
        _clear_cuda_cache()
        log.info("[L%d] completed.", layer_id)
