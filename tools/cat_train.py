import argparse
import json
import os
import sys
import time
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train_utils.train_args import (
    process_cat_train_args,
    create_optimizer,
    resolve_skip_layer_matches,
)
from train_utils.model_checkpoint_io import (
    _build_run_output_dir,
    _safe_path_token,
    save_model_checkpoint,
)
from train_utils.utils import get_logger, set_seed


log = get_logger("linear_by_category")


def _split_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def _resolve_category_order(order: str, discovered: Sequence[str]) -> List[str]:
    if order.strip().lower() == "auto":
        return sorted(set(discovered))
    requested = _split_csv(order)
    if "others" in requested:
        known = [c for c in requested if c != "others"]
        rest = sorted([c for c in set(discovered) if c not in set(known)])
        return known + rest
    return requested


@dataclass(frozen=True)
class LinearRef:
    name: str
    module: nn.Linear
    category: str
    transpose: bool


def _is_decoder_layer_projection(name: str, projection_suffixes: Sequence[str]) -> bool:
    # Llama/Mistral/Qwen 结构示例: "model.layers.{i}.<...>.<proj>"
    # OPT 结构示例: "model.decoder.layers.{i}.<...>.<proj>"
    in_decoder_layers = (
        ".model.layers." in name
        or name.startswith("model.layers.")
        or ".model.decoder.layers." in name
        or name.startswith("model.decoder.layers.")
    )
    if not in_decoder_layers:
        return False
    return any(name.endswith(f".{sfx}") or name.endswith(sfx) for sfx in projection_suffixes)


def _collect_linears(
    model: nn.Module,
    transpose_modules: Sequence[str],
    *,
    only_decoder_projections: bool,
    projection_suffixes: Sequence[str],
) -> List[LinearRef]:
    transpose_set = set(transpose_modules)
    suffix_set = set(projection_suffixes)
    out: List[LinearRef] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        category = name.split(".")[-1]
        if only_decoder_projections:
            if category not in suffix_set:
                continue
            if not _is_decoder_layer_projection(name, projection_suffixes):
                continue
        out.append(
            LinearRef(
                name=name,
                module=module,
                category=category,
                transpose=(category in transpose_set),
            )
        )
    return out


_LAYER_IDX_PATTERNS = [
    re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\."),
    re.compile(r"(?:^|\.)(?:model\.)?decoder\.layers\.(\d+)\."),
]


def _extract_layer_idx(name: str) -> Optional[int]:
    for pat in _LAYER_IDX_PATTERNS:
        m = pat.search(name)
        if m:
            return int(m.group(1))
    return None


def _clone_namespace(ns, **overrides):
    data = dict(vars(ns))
    data.update(overrides)
    return argparse.Namespace(**data)


def _format_namespace(ns: argparse.Namespace) -> str:
    return json.dumps(vars(ns), ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _fuse_q_scale_linear(linear: nn.Linear, q_scale: float) -> None:
    with torch.no_grad():
        weight = linear.weight.data
        bias_delta = -q_scale * weight.sum(dim=1)
        weight.mul_(q_scale * 2)
        if linear.bias is not None:
            linear.bias.data.add_(bias_delta)
        else:
            linear.bias = nn.Parameter(bias_delta)


def _fuse_q_scale_into_decoder(decoder: nn.Module) -> None:
    # 对齐 litebsq.llm_vae.Decoder._fuse_q_scale 的行为（单模型子 decoder）。
    in_dim = int(getattr(decoder, "in_dim"))
    q_scale = 1.0 / (in_dim ** 0.5)
    decoder_type = str(getattr(decoder, "decoder_type"))
    if decoder_type == "linear":
        _fuse_q_scale_linear(decoder.linear, q_scale)
    elif decoder_type == "symmetric":
        _fuse_q_scale_linear(decoder.linear_in, q_scale)


def _fuse_norm_into_decoder(decoder: nn.Module, mean: float, std: float) -> None:
    decoder_type = str(getattr(decoder, "decoder_type"))
    if decoder_type == "linear":
        last = decoder.linear
    elif decoder_type == "symmetric":
        last = decoder.linear_out
    else:
        raise ValueError(f"Unsupported decoder_type={decoder_type} for norm fusion")

    if not isinstance(last, nn.Linear):
        raise TypeError(f"Expected nn.Linear as last layer, got {type(last)}")

    with torch.no_grad():
        last.weight.mul_(std)
        if last.bias is None:
            last.bias = nn.Parameter(torch.zeros(last.out_features, device=last.weight.device, dtype=last.weight.dtype))
        last.bias.mul_(std).add_(mean)


def _eval_ppl_after_category(model: nn.Module, vae_args, ppl_limit: int, category: str, eval_device: str = "cuda") -> None:
    from train_utils.eval_utils import calculate_ppl

    log.info("开始类别 %s 的 PPL 评估...", category)
    model.eval()
    model.to(eval_device)
    with torch.no_grad():
        setattr(vae_args, "limit", int(ppl_limit))
        ppl_result = calculate_ppl(model, vae_args)
    model.to("cpu")
    torch.cuda.empty_cache()
    log.info("类别 %s 训练后 PPL: %.2f", category, float(ppl_result.get("wiki_ppl", float("nan"))))


def _split_linear_into_parts(weight: torch.Tensor, transpose: bool, num_parts: int) -> torch.Tensor:
    """
    将单个 linear 权重切分为 num_parts 个子块，返回形状 [num_parts, -1]。
    切分规则与 BSQLinear 一致：先按 transpose 选择方向，再沿第 0 维均分。
    """
    w = weight.detach().float()
    if transpose:
        w = w.t()
    if w.shape[0] % num_parts != 0:
        raise ValueError(
            f"weight dim0={w.shape[0]} not divisible by num_parts={num_parts} (transpose={transpose})"
        )
    return w.reshape(num_parts, w.shape[0] // num_parts, w.shape[1]).reshape(num_parts, -1)


def _train_group_vae_and_replace(
    *,
    model: nn.Module,
    group_refs: Sequence[LinearRef],
    group_tag: str,
    vae_args,
    training_args,
    train_device: str,
    convert_device: str,
    do_convert: bool,
    steps: int,
    batch_size: int,
    log_every: int,
    eval_every: int,
    eval_blocks: int,
    output_dir: str,
    intra_parallel: int,
    skip_layer_keys: Optional[Set[Tuple[int, str]]] = None,
) -> None:
    from litebsq.llm_vae import MultiLayerVAE
    from litebsq.vae_linear import VAELinear
    from litebsq.bsq_linear import set_module_by_name

    # 根据训练参数选择输入精度。
    if bool(getattr(training_args, "bf16", False)):
        train_dtype = torch.bfloat16
    elif bool(getattr(training_args, "fp16", False)):
        train_dtype = torch.float16
    else:
        train_dtype = torch.float32

    if intra_parallel < 1:
        raise ValueError(f"intra_parallel must be >= 1, got {intra_parallel}")
    num_linear = len(group_refs)
    num_models = num_linear * intra_parallel
    group_vae_args = _clone_namespace(vae_args, parallel_layers=num_models)
    vae = MultiLayerVAE(group_vae_args).to(train_device)

    # 1) 组内权重堆叠与预处理：
    #    每个 linear 先切成 intra_parallel 份，再拼成 [num_linear * intra_parallel, N]。
    split_list = []
    for r in group_refs:
        split_parts = _split_linear_into_parts(r.module.weight, r.transpose, intra_parallel).cpu()
        split_list.append(split_parts)
    per_linear_flat = torch.stack(split_list, dim=0)  # [num_linear, intra_parallel, N]
    stacked_flat = per_linear_flat.reshape(num_models, -1)  # [num_models, N]

    d_mean = stacked_flat.mean(dim=1, keepdim=True)
    d_std = stacked_flat.std(dim=1, keepdim=True)
    if bool(getattr(group_vae_args, "normalize_weight", False)):
        stacked_flat = (stacked_flat - d_mean) / (d_std + 1e-6)

    codebook_dim = int(getattr(group_vae_args, "codebook_dim"))
    numel = stacked_flat.shape[1]
    if numel % codebook_dim != 0:
        raise ValueError(f"[{group_tag}] flatten_len={numel} not divisible by codebook_dim={codebook_dim}")

    stacked_data = stacked_flat.view(num_models, -1, codebook_dim).permute(1, 0, 2).contiguous()
    # stacked_data 形状: [N_blocks, P, codebook_dim]
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(stacked_data),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    eval_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(stacked_data),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 2) 训练当前分组对应的 VAE。
    optimizer = create_optimizer(vae.parameters(), group_vae_args, group_vae_args.lr)
    lr_scheduler = None
    lr_scheduler_name = str(getattr(group_vae_args, "lr_scheduler", "none"))
    if lr_scheduler_name != "none":
        import transformers

        lr_scheduler = transformers.get_scheduler(
            lr_scheduler_name,
            optimizer,
            num_warmup_steps=int(getattr(group_vae_args, "lr_warmup_steps", 0)),
            num_training_steps=int(steps),
        )

    start = time.time()
    train_iter = iter(train_loader)
    for step in range(int(steps)):
        try:
            (x_batch,) = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            (x_batch,) = next(train_iter)

        x = x_batch.to(device=train_device, dtype=train_dtype, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        _, loss_dict = vae(x, is_train=True)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if log_every > 0 and (step + 1) % int(log_every) == 0:
            speed = (time.time() - start) / int(log_every)
            recon = loss_dict.get("train/recon_loss")
            commit = loss_dict.get("train/commitment_loss")
            log.info(
                "[%s] step=%d/%d loss=%.6f recon=%.6f commit=%.6f speed=%.4fs/it",
                group_tag,
                step + 1,
                steps,
                float(loss.detach().float().item()),
                float(recon.detach().float().item()) if isinstance(recon, torch.Tensor) else float("nan"),
                float(commit.detach().float().item()) if isinstance(commit, torch.Tensor) else float("nan"),
                speed,
            )
            start = time.time()

        if eval_every > 0 and (step + 1) % int(eval_every) == 0:
            vae.eval()
            with torch.no_grad():
                mse_acc = []
                top_k_mse_acc = []
                total = 0
                for (x_eval_batch,) in eval_loader:
                    if total >= int(eval_blocks):
                        break
                    x_eval_batch = x_eval_batch[: max(0, int(eval_blocks) - total)]
                    total += x_eval_batch.shape[0]
                    x_eval = x_eval_batch.to(device=train_device, dtype=train_dtype, non_blocking=True)
                    x_recon, _ = vae(x_eval, is_train=False)
                    x_eval_f = x_eval.float()
                    x_recon_f = x_recon.float()
                    mse_acc.append(torch.nn.functional.mse_loss(x_recon_f, x_eval_f))

                    # 对每个并行模型（P 维）独立选 top-k：
                    # x_eval/x_recon: [B, P, C] -> [P, B*C]
                    flat_eval = x_eval_f.permute(1, 0, 2).reshape(x_eval_f.shape[1], -1)
                    flat_recon = x_recon_f.permute(1, 0, 2).reshape(x_recon_f.shape[1], -1)
                    k = min(100, flat_eval.shape[1])
                    _, topk_idx = torch.topk(flat_eval.abs(), k=k, dim=1)
                    top_eval = torch.gather(flat_eval, dim=1, index=topk_idx)
                    top_recon = torch.gather(flat_recon, dim=1, index=topk_idx)
                    top_k_mse_acc.append(torch.nn.functional.mse_loss(top_recon, top_eval))
                mse = torch.stack(mse_acc).mean() if mse_acc else torch.tensor(0.0)
                top_k_mse = torch.stack(top_k_mse_acc).mean() if top_k_mse_acc else torch.tensor(0.0)
            log.info(
                "[%s] eval@step=%d mse=%.6e top_k_mse(k=100)=%.6e",
                group_tag,
                step + 1,
                float(mse.detach().cpu().item()),
                float(top_k_mse.detach().cpu().item()),
            )
            vae.train()

    # # 保存分组 VAE，便于复现实验和离线分析。
    # group_dir = os.path.join(output_dir, "vae_by_category", group_tag.replace("/", "_"))
    # os.makedirs(group_dir, exist_ok=True)
    # torch.save(vae.state_dict(), os.path.join(group_dir, "vae_state.pt"))

    if not do_convert:
        del vae, stacked_data, per_linear_flat, stacked_flat
        torch.cuda.empty_cache()
        return

    # 3) 对所有块做量化得到 bit 索引，再替换为 VAELinear。
    vae.eval()
    bit_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for (x_in_batch,) in eval_loader:
            x_in = x_in_batch.to(device=train_device, dtype=train_dtype, non_blocking=True)
            _, bit_idx = vae(x_in, is_train=False)  # [B, P, latent_dim]，布尔索引
            bit_chunks.append(bit_idx.detach().to("cpu"))
    full_bits = torch.cat(bit_chunks, dim=0)  # [N_blocks, P, latent_dim]

    decoders: List[nn.Module] = []
    for i in range(num_models):
        dec = vae.model.decoder.get_sub_decoder(i)
        _fuse_q_scale_into_decoder(dec)
        if bool(getattr(group_vae_args, "normalize_weight", False)):
            _fuse_norm_into_decoder(dec, mean=float(d_mean[i].item()), std=float(d_std[i].item()))
        decoders.append(dec)

    for i, r in enumerate(group_refs):
        old = r.module
        layer_idx = _extract_layer_idx(r.name)
        skip_this = bool(
            skip_layer_keys
            and layer_idx is not None
            and (int(layer_idx), str(r.category)) in skip_layer_keys
        )
        start_idx = i * intra_parallel
        end_idx = start_idx + intra_parallel
        part_bits = []
        part_decoders = []
        for model_idx in range(start_idx, end_idx):
            part_bits.append(full_bits[:, model_idx, :].unsqueeze(1))  # [N_blocks, 1, latent_dim]
            part_decoders.append(decoders[model_idx])
        new_linear = VAELinear(
            in_features=old.in_features,
            out_features=old.out_features,
            bias=old.bias,
            original_weight=old.weight,
            vq_weight=part_bits if intra_parallel > 1 else part_bits[0],
            decoder=part_decoders if intra_parallel > 1 else part_decoders[0],
            codebook_dim=codebook_dim,
            transpose=r.transpose,
            parallel_parts=intra_parallel,
            always_use_original=skip_this,
            protect_original_weight=skip_this,
        ).to(convert_device)
        # 替换时预热：后续 LoRA / PPL 前向可直接复用缓存权重，避免重复重构。
        try:
            new_linear.prime_decoded_weight_cache(dtype=train_dtype)
        except Exception as e:
            log.warning("[%s] cache warmup failed for %s: %s", group_tag, r.name, e)
        # 替换后将模块放回 CPU，降低显存占用。
        new_linear.to("cpu")
        set_module_by_name(model, r.name, new_linear)

    del vae, stacked_data, per_linear_flat, stacked_flat, full_bits, decoders
    torch.cuda.empty_cache()


def main(argv: Optional[Sequence[str]] = None) -> None:
    global log
    cat_args, hf_args, training_args, vae_args = process_cat_train_args(argv)
    set_seed(cat_args.seed)

    os.makedirs(cat_args.output_dir, exist_ok=True)
    run_output_dir = _build_run_output_dir(cat_args.output_dir, vae_args.model_path)
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "linear_by_category.log")
    log = get_logger("linear_by_category")
    cat_args.output_dir = run_output_dir

    log.info("Run output directory: %s", run_output_dir)
    log.info(
        "Args:\nscript=%s\nvae=%s\ntraining=%s",
        _format_namespace(cat_args),
        _format_namespace(vae_args),
        _format_namespace(training_args),
    )

    log.info("Loading model: %s", vae_args.model_path)
    from rotation.model_utils import get_model

    model = get_model(vae_args.model_path, hf_args.access_token)

    transpose_modules = _split_csv(cat_args.transpose_modules)
    projection_suffixes = _split_csv(cat_args.projection_suffixes)
    only_decoder_projections = bool(cat_args.only_decoder_projections) and not bool(cat_args.include_all_linears)
    all_linears = _collect_linears(
        model,
        transpose_modules,
        only_decoder_projections=only_decoder_projections,
        projection_suffixes=projection_suffixes,
    )
    discovered_categories = [r.category for r in all_linears]
    category_order = _resolve_category_order(cat_args.category_order, discovered_categories)

    refs_by_cat: Dict[str, List[LinearRef]] = {}
    for r in all_linears:
        refs_by_cat.setdefault(r.category, []).append(r)
    discovered_skip_keys = []
    for r in all_linears:
        li = _extract_layer_idx(r.name)
        if li is not None:
            discovered_skip_keys.append((li, r.category))
    skip_layer_keys, matched, missing = resolve_skip_layer_matches(
        getattr(cat_args, "skip_layers", ""),
        discovered_skip_keys,
    )
    if skip_layer_keys:
        if matched:
            log.info(
                "skip_layers 生效: %s",
                ",".join(f"{li}.{cat}" for li, cat in matched),
            )
        if missing:
            log.warning(
                "skip_layers 未匹配到任何 Linear: %s",
                ",".join(f"{li}.{cat}" for li, cat in missing),
            )

    steps_per_group = int(cat_args.steps_per_group) if cat_args.steps_per_group is not None else int(
        cat_args.steps_per_category)
    linear_group_size = int(cat_args.linear_group_size)
    intra_parallel = int(cat_args.intra_parallel)
    if linear_group_size < 1:
        raise ValueError(f"linear_group_size must be >= 1, got {linear_group_size}")
    if intra_parallel < 1:
        raise ValueError(f"intra_parallel must be >= 1, got {intra_parallel}")
    if int(getattr(vae_args, "parallel_layers", 1)) != 1:
        log.warning("检测到 --parallel_layers=%d，但当前脚本不再使用该参数；请使用 --intra_parallel。", int(vae_args.parallel_layers))
    log.info(
        "并行配置: linear_group_size=%d, intra_parallel=%d, total_num_models=%d",
        linear_group_size,
        intra_parallel,
        linear_group_size * intra_parallel,
    )

    active_categories = [c for c in category_order if c in refs_by_cat]
    for cat_idx, cat in enumerate(active_categories):
        if cat not in refs_by_cat:
            continue

        refs = refs_by_cat[cat]
        if not refs:
            continue

        log.info("=== Category: %s (%d linears) ===", cat, len(refs))

        refs_sorted = []
        missing = 0
        for r in refs:
            li = _extract_layer_idx(r.name)
            if li is None:
                missing += 1
                continue
            refs_sorted.append((li, r))
        if missing:
            log.warning("[%s] %d modules missing layer_idx, skipped.", cat, missing)
        refs_sorted.sort(key=lambda x: x[0])
        ordered_refs = [r for _, r in refs_sorted]

        for start in range(0, len(ordered_refs), linear_group_size):
            group_refs = ordered_refs[start:start + linear_group_size]
            if len(group_refs) < linear_group_size and not cat_args.allow_tail_group:
                log.info("[%s] tail group size=%d skipped (set --allow_tail_group to include).", cat, len(group_refs))
                break
            layer_indices = [idx for idx, _ in refs_sorted[start:start + linear_group_size]]
            group_tag = f"{cat}.L{layer_indices[0]}-{layer_indices[-1]}"
            log.info(
                "---- Group: %s (linears=%d, intra_parallel=%d, num_models=%d) ----",
                group_tag,
                len(group_refs),
                intra_parallel,
                len(group_refs) * intra_parallel,
            )
            _train_group_vae_and_replace(
                model=model,
                group_refs=group_refs,
                group_tag=group_tag,
                vae_args=vae_args,
                training_args=training_args,
                train_device=cat_args.train_device,
                convert_device=cat_args.convert_device,
                do_convert=bool(cat_args.convert),
                steps=steps_per_group,
                batch_size=cat_args.batch_size,
                log_every=cat_args.log_every,
                eval_every=cat_args.eval_every,
                eval_blocks=cat_args.eval_blocks,
                output_dir=cat_args.output_dir,
                intra_parallel=intra_parallel,
                skip_layer_keys=skip_layer_keys,
            )

        if cat_args.lora_after_category:
            from train_utils.lora_utils import lora_finetune_remaining_categories
            log.info("LoRA 微调前评估...")
            _eval_ppl_after_category(
                model=model,
                vae_args=vae_args,
                ppl_limit=cat_args.ppl_limit,
                category=cat,
                eval_device=cat_args.train_device,
            )

            remaining_categories = active_categories[cat_idx + 1:]
            model = lora_finetune_remaining_categories(
                model=model,
                remaining_categories=remaining_categories,
                collect_linears_fn=_collect_linears,
                transpose_modules=transpose_modules,
                projection_suffixes=projection_suffixes,
                only_decoder_projections=only_decoder_projections,
                cat_args=cat_args,
                vae_args=vae_args,
                training_args=training_args,
                logger=log,
            )

        _eval_ppl_after_category(
            model=model,
            vae_args=vae_args,
            ppl_limit=cat_args.ppl_limit,
            category=cat,
            eval_device=cat_args.train_device,
        )
        cat_dir_name = _safe_path_token(cat)
        cat_model_dir = os.path.join(run_output_dir, cat_dir_name)
        save_paths = save_model_checkpoint(
            model,
            cat_model_dir,
            base_model_path=vae_args.model_path,
            tokenizer=None,
            save_config=True,
            extra_meta={
                "stage": "after_category",
                "category": cat,
                "category_index": int(cat_idx),
                "lora_after_category": bool(cat_args.lora_after_category),
            },
        )
        log.info("Saved category checkpoint (%s): %s", cat, save_paths["output_dir"])

    if cat_args.save_model:
        if not cat_args.convert:
            raise ValueError("--save_model requires --convert")
        from transformers import AutoTokenizer
        from litebsq.vae_linear import clear_model_vae_linear_cache

        model_out = os.path.join(run_output_dir, "final_model")
        tok = AutoTokenizer.from_pretrained(vae_args.model_path, use_fast=True, token=hf_args.access_token)
        cleared = clear_model_vae_linear_cache(model)
        log.info("Final save: cleared decoded cache for %d VAELinear modules.", cleared)
        save_paths = save_model_checkpoint(
            model,
            model_out,
            base_model_path=vae_args.model_path,
            tokenizer=tok,
            save_config=True,
            extra_meta={"stage": "final"},
            unload_vae_original_weights=bool(cat_args.unload_vae_original_weights_on_final_save),
        )
        log.info("Saved final model to %s", save_paths["output_dir"])

    log.info("Done.")


if __name__ == "__main__":
    main()
