import argparse
import math
from typing import Dict, List, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from litebsq.vae_linear import VAELinear, clear_model_vae_linear_cache
from train_utils.train_args import create_optimizer

from distill_utils.layerwise_distill_hooks import LayerIOHook


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
    snap: Dict[str, torch.Tensor] = {}
    for name, p in trainables:
        snap[name] = p.detach().float().clone()
    return snap


def anchor_loss(trainables: Sequence[Tuple[str, nn.Parameter]], snapshot: Dict[str, torch.Tensor]) -> torch.Tensor:
    terms = []
    for name, p in trainables:
        ref = snapshot[name]
        terms.append((p.float() - ref).pow(2).mean())
    if not terms:
        raise RuntimeError("anchor loss requested with empty trainables.")
    return torch.stack(terms).mean()


def _forward_model(model: nn.Module, input_ids: torch.Tensor) -> None:
    model(input_ids=input_ids, use_cache=False)


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
) -> None:
    opt_args = argparse.Namespace(
        optimizer=args.optimizer,
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        weight_decay=float(args.weight_decay),
    )

    num_samples = int(calib_inputs.shape[0])
    num_batches = math.ceil(num_samples / int(args.batch_size))

    for order, layer_id in enumerate(layer_indices, start=1):
        clear_model_vae_linear_cache(model_q)
        freeze_student(model_q)

        layer_q = layers_q[layer_id]
        layer_t = layers_t[layer_id]
        trainables = collect_layer_trainables(
            layer_q,
            train_bias=bool(args.train_bias),
            train_layernorm_weight=bool(args.train_layernorm_weight),
        )
        if not trainables:
            log.warning("[L%d] no trainable parameters found, skip.", layer_id)
            continue

        snapshot = snapshot_trainables(trainables)
        optimizer = create_optimizer([p for _name, p in trainables], opt_args, float(args.lr))

        hook_t = LayerIOHook(layer_t)
        hook_q = LayerIOHook(layer_q)

        total_steps = int(args.steps_per_layer) if int(args.steps_per_layer) > 0 else int(args.epochs_per_layer) * num_batches
        if total_steps < 1:
            hook_t.remove()
            hook_q.remove()
            raise ValueError("No training steps scheduled. Increase steps_per_layer or epochs_per_layer.")

        n_params = sum(int(p.numel()) for _n, p in trainables)
        log.info(
            "[L%d][%d/%d] start: trainable_tensors=%d trainable_params=%d steps=%d",
            layer_id,
            order,
            len(layer_indices),
            len(trainables),
            n_params,
            total_steps,
        )

        sample_order = torch.arange(num_samples, dtype=torch.long)
        for step in range(total_steps):
            batch_pos = step % num_batches
            if batch_pos == 0:
                if bool(args.shuffle):
                    perm = torch.randperm(num_samples)
                    sample_order = sample_order.index_select(0, perm)
                else:
                    sample_order = torch.arange(num_samples, dtype=torch.long)

            begin = batch_pos * int(args.batch_size)
            end = min(begin + int(args.batch_size), num_samples)
            batch_idx = sample_order[begin:end]
            input_ids_cpu = calib_inputs.index_select(0, batch_idx)

            teacher_input = input_ids_cpu.to(args.teacher_device, non_blocking=True)
            student_input = input_ids_cpu.to(args.student_device, non_blocking=True)

            hook_t.clear()
            hook_q.clear()
            with torch.no_grad():
                _forward_model(model_t, teacher_input)
                t_in, t_out = hook_t.pop(detach=True)

            _forward_model(model_q, student_input)
            q_in, q_out = hook_q.pop(detach=False)

            t_in = t_in.to(device=q_in.device, dtype=q_in.dtype, non_blocking=True)
            t_out = t_out.to(device=q_out.device, dtype=q_out.dtype, non_blocking=True)

            loss_blk = F.mse_loss((q_out - q_in).float(), (t_out - t_in).float())
            loss_res = F.mse_loss(q_out.float(), t_out.float())
            loss_anchor = anchor_loss(trainables, snapshot)

            loss = (
                float(args.lambda_blk) * loss_blk
                + float(args.lambda_res) * loss_res
                + float(args.lambda_anchor) * loss_anchor
            )

            loss_norm = None
            if bool(args.use_norm_loss):
                loss_norm = F.mse_loss(
                    q_out.float().mean(dim=(0, 1)),
                    t_out.float().mean(dim=(0, 1)),
                )
                loss = loss + float(args.lambda_norm) * loss_norm

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if int(args.log_every) > 0 and ((step + 1) % int(args.log_every) == 0 or step == 0 or (step + 1) == total_steps):
                if loss_norm is None:
                    log.info(
                        "[L%d] step %d/%d loss=%.6f blk=%.6f res=%.6f anchor=%.6f",
                        layer_id,
                        step + 1,
                        total_steps,
                        float(loss.detach().item()),
                        float(loss_blk.detach().item()),
                        float(loss_res.detach().item()),
                        float(loss_anchor.detach().item()),
                    )
                else:
                    log.info(
                        "[L%d] step %d/%d loss=%.6f blk=%.6f res=%.6f norm=%.6f anchor=%.6f",
                        layer_id,
                        step + 1,
                        total_steps,
                        float(loss.detach().item()),
                        float(loss_blk.detach().item()),
                        float(loss_res.detach().item()),
                        float(loss_norm.detach().item()),
                        float(loss_anchor.detach().item()),
                    )

        hook_t.remove()
        hook_q.remove()
        clear_model_vae_linear_cache(model_q)
        log.info("[L%d] completed.", layer_id)
