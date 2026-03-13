import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train_utils.train_args import (
    process_cat_train_args,
    create_optimizer,
    resolve_codebook_int_for_category,
    resolve_intra_parallel_for_category,
    resolve_skip_layer_matches,
)
from train_utils.cat_data_prep import (
    LinearPrepRef,
    gather_wa_mse_act_max_batch,
    load_activation_weight_dict,
    prepare_group_weight_data,
    resolve_intra_parallel,
)
from train_utils.activation_utils import (
    ActivationCalibrationCache,
    collect_act_max_for_linears,
)
from train_utils.model_checkpoint_io import (
    _build_run_output_dir,
    _safe_path_token,
    save_model_checkpoint,
)
from train_utils.utils import (
    LinearRef,
    clone_namespace as _clone_namespace,
    collect_linears as _collect_linears,
    extract_layer_idx as _extract_layer_idx,
    format_intra_parallel_desc as _format_intra_parallel_desc,
    format_namespace as _format_namespace,
    get_logger,
    resolve_category_order as _resolve_category_order,
    set_seed,
    split_csv as _split_csv,
)


log = get_logger("linear_by_category")


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
    elif decoder_type in {"symmetric", "asymmetric"}:
        _fuse_q_scale_linear(decoder.linear_in, q_scale)


def _fuse_norm_into_decoder(decoder: nn.Module, mean: float, std: float) -> None:
    decoder_type = str(getattr(decoder, "decoder_type"))
    if decoder_type == "linear":
        last = decoder.linear
    elif decoder_type in {"symmetric", "asymmetric"}:
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
    intra_parallel,
    intra_part_sort_mode: str,
    skip_layer_keys: Optional[Set[Tuple[int, str]]] = None,
    activation_weight_by_linear: Optional[Dict[str, torch.Tensor]] = None,
    wa_mse_runtime: Optional[Dict[str, object]] = None,
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

    use_wa_mse_loss = str(getattr(vae_args, "recon_loss_type", "")).lower() == "wa_mse"
    row_parts, col_parts = resolve_intra_parallel(intra_parallel)
    parts_per_linear = int(row_parts) * int(col_parts)
    effective_activation_weight = activation_weight_by_linear
    if use_wa_mse_loss and wa_mse_runtime is not None:
        if bool(wa_mse_runtime.get("dynamic", False)):
            calib_device = str(wa_mse_runtime.get("device") or train_device)
            linear_items = [(r.name, r.module) for r in group_refs]
            dynamic_act_max, new_cache = collect_act_max_for_linears(
                model=model,
                linear_items=linear_items,
                model_path=str(wa_mse_runtime["model_path"]),
                access_token=wa_mse_runtime.get("access_token"),
                dataset=str(wa_mse_runtime.get("dataset", "wikitext2")),
                nsamples=int(wa_mse_runtime.get("nsamples", 512)),
                seqlen=int(wa_mse_runtime.get("seqlen", 512)),
                seed=int(wa_mse_runtime.get("seed", 0)),
                device=calib_device,
                cache=wa_mse_runtime.get("cache"),  # type: ignore[arg-type]
                log_every=int(wa_mse_runtime.get("log_every", 0)),
                logger=log,
            )
            wa_mse_runtime["cache"] = new_cache
            effective_activation_weight = dynamic_act_max
            log.info(
                "[%s] refreshed act_max from current model (linears=%d, dataset=%s, nsamples=%d, seqlen=%d).",
                group_tag,
                len(dynamic_act_max),
                str(wa_mse_runtime.get("dataset", "wikitext2")),
                int(wa_mse_runtime.get("nsamples", 512)),
                int(wa_mse_runtime.get("seqlen", 512)),
            )
        elif effective_activation_weight is None:
            static_dict = wa_mse_runtime.get("static_dict")
            if isinstance(static_dict, dict):
                effective_activation_weight = static_dict

    prep_refs = [
        LinearPrepRef(
            name=r.name,
            weight=r.module.weight,
            in_features=int(r.module.in_features),
            out_features=int(r.module.out_features),
            transpose=bool(r.transpose),
        )
        for r in group_refs
    ]
    prep_result = prepare_group_weight_data(
        group_refs=prep_refs,
        intra_parallel=(row_parts, col_parts),
        codebook_dim=int(getattr(vae_args, "codebook_dim")),
        batch_size=int(batch_size),
        normalize_weight=bool(getattr(vae_args, "normalize_weight", False)),
        recon_loss_type=str(getattr(vae_args, "recon_loss_type", "")),
        activation_weight_by_linear=effective_activation_weight,
        train_device=train_device,
        intra_part_sort_mode=str(intra_part_sort_mode),
    )
    num_models = int(prep_result.num_models)
    group_vae_args = _clone_namespace(vae_args, parallel_layers=num_models)
    vae = MultiLayerVAE(group_vae_args).to(train_device)
    codebook_dim = int(prep_result.codebook_dim)
    d_mean = prep_result.d_mean
    d_std = prep_result.d_std
    stacked_data = prep_result.stacked_data
    train_loader = prep_result.train_loader
    eval_loader = prep_result.eval_loader
    use_wa_mse = bool(prep_result.use_wa_mse)
    part_metas = prep_result.part_metas
    split_metas = prep_result.split_metas
    if len(split_metas) != len(group_refs):
        raise RuntimeError(
            f"[{group_tag}] split metadata mismatch: len(split_metas)={len(split_metas)} "
            f"vs len(group_refs)={len(group_refs)}"
        )
    if use_wa_mse:
        log.info("[%s] wa_mse enabled with online act_max gather.", group_tag)

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
            x_batch, block_idx_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_batch, block_idx_batch = next(train_iter)

        x = x_batch.to(device=train_device, dtype=train_dtype, non_blocking=True)
        act_max_batch = None
        if use_wa_mse:
            act_max_batch = gather_wa_mse_act_max_batch(
                block_idx_batch=block_idx_batch,
                part_metas=part_metas,
                codebook_dim=codebook_dim,
                train_device=train_device,
                target_dtype=train_dtype,
            )
        optimizer.zero_grad(set_to_none=True)
        _, loss_dict = vae(x, is_train=True, act_max=act_max_batch)
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
                for x_eval_batch, _eval_idx_batch in eval_loader:
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
        del vae, stacked_data
        torch.cuda.empty_cache()
        return

    # 3) 对所有块做量化得到 bit 索引，再替换为 VAELinear。
    vae.eval()
    bit_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for x_in_batch, _eval_idx_batch in eval_loader:
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
        split_meta = split_metas[i]
        if str(split_meta.linear_name) != str(r.name):
            raise RuntimeError(
                f"[{group_tag}] split metadata order mismatch at idx={i}: "
                f"meta={split_meta.linear_name}, ref={r.name}"
            )
        if int(split_meta.parallel_rows) * int(split_meta.parallel_cols) != int(parts_per_linear):
            raise RuntimeError(
                f"[{group_tag}] split parts mismatch at idx={i}: "
                f"meta={split_meta.parallel_rows}x{split_meta.parallel_cols}, expected={parts_per_linear}"
            )
        layer_idx = _extract_layer_idx(r.name)
        skip_this = bool(
            skip_layer_keys
            and layer_idx is not None
            and (int(layer_idx), str(r.category)) in skip_layer_keys
        )
        start_idx = i * parts_per_linear
        end_idx = start_idx + parts_per_linear
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
            vq_weight=part_bits if parts_per_linear > 1 else part_bits[0],
            decoder=part_decoders if parts_per_linear > 1 else part_decoders[0],
            codebook_dim=codebook_dim,
            transpose=r.transpose,
            parallel_parts=parts_per_linear,
            parallel_rows=row_parts,
            parallel_cols=col_parts,
            restore_row_indices=split_meta.restore_row_indices,
            restore_col_indices=split_meta.restore_col_indices,
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

    del vae, stacked_data, full_bits, decoders
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
    intra_part_sort_mode = str(getattr(cat_args, "intra_part_sort_mode", "row_l2")).strip().lower()
    intra_parallel_raw = getattr(cat_args, "intra_parallel", 1)
    if isinstance(intra_parallel_raw, dict):
        log.info(
            "intra_parallel category overrides enabled: keys=%s",
            ",".join(sorted(str(k) for k in intra_parallel_raw.keys())),
        )
    activation_weight_by_linear: Optional[Dict[str, torch.Tensor]] = None
    wa_mse_runtime: Optional[Dict[str, object]] = None
    act_path = getattr(cat_args, "activation_weight_path", None)
    if act_path:
        activation_weight_by_linear = load_activation_weight_dict(str(act_path))
        log.info(
            "Loaded static activation abs-max dict: %s (entries=%d)",
            act_path,
            len(activation_weight_by_linear),
        )

    if str(getattr(vae_args, "recon_loss_type", "")).lower() == "wa_mse":
        wa_mse_runtime = {
            "dynamic": str(getattr(cat_args, "wa_mse_act_mode", "dynamic")).strip().lower() == "dynamic",
            "cache": None,  # type: Optional[ActivationCalibrationCache]
            "dataset": str(getattr(cat_args, "wa_mse_calib_dataset", "wikitext2")),
            "nsamples": int(getattr(cat_args, "wa_mse_calib_nsamples", 512)),
            "seqlen": int(getattr(cat_args, "wa_mse_calib_seqlen", 512)),
            "seed": int(getattr(cat_args, "wa_mse_calib_seed", 0)),
            "device": str(getattr(cat_args, "wa_mse_calib_device", "")).strip() or str(cat_args.train_device),
            "log_every": int(getattr(cat_args, "wa_mse_calib_log_every", 0)),
            "model_path": str(vae_args.model_path),
            "access_token": hf_args.access_token,
            "static_dict": activation_weight_by_linear,
        }
        if not bool(wa_mse_runtime["dynamic"]) and activation_weight_by_linear is None:
            raise ValueError(
                "wa_mse requires either --wa_mse_act_mode dynamic or --activation_weight_path in static mode."
            )
        if bool(wa_mse_runtime["dynamic"]):
            log.info(
                "wa_mse dynamic act_max enabled: dataset=%s nsamples=%d seqlen=%d seed=%d device=%s",
                str(wa_mse_runtime["dataset"]),
                int(wa_mse_runtime["nsamples"]),
                int(wa_mse_runtime["seqlen"]),
                int(wa_mse_runtime["seed"]),
                str(wa_mse_runtime["device"]),
            )
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
    if linear_group_size < 1:
        raise ValueError(f"linear_group_size must be >= 1, got {linear_group_size}")

    active_categories = [c for c in category_order if c in refs_by_cat]
    category_intra_parallel: Dict[str, Tuple[int, int]] = {}
    category_codebook: Dict[str, Tuple[int, int]] = {}
    for cat in active_categories:
        category_intra_parallel[cat] = resolve_intra_parallel_for_category(intra_parallel_raw, cat)
        category_codebook[cat] = (
            resolve_codebook_int_for_category(
                getattr(vae_args, "codebook_bits"),
                cat,
                arg_name="codebook_bits",
            ),
            resolve_codebook_int_for_category(
                getattr(vae_args, "codebook_dim"),
                cat,
                arg_name="codebook_dim",
            ),
        )

    sort_needs_act = (
        intra_part_sort_mode == "act_row_l2"
        and any((int(rp) * int(cp)) > 1 for rp, cp in category_intra_parallel.values())
    )
    if sort_needs_act and activation_weight_by_linear is None and wa_mse_runtime is None:
        raise ValueError(
            "intra_part_sort_mode=act_row_l2 requires activation vectors. "
            "Please provide --activation_weight_path."
        )

    unique_parallel = sorted(set(category_intra_parallel.values()))
    if int(getattr(vae_args, "parallel_layers", 1)) != 1:
        log.warning("检测到 --parallel_layers=%d，但当前脚本不再使用该参数；请使用 --intra_parallel。", int(vae_args.parallel_layers))
    if unique_parallel:
        if len(unique_parallel) == 1:
            intra_row_parts, intra_col_parts = unique_parallel[0]
            intra_parts_per_linear = int(intra_row_parts) * int(intra_col_parts)
            intra_parallel_desc = _format_intra_parallel_desc(intra_row_parts, intra_col_parts)
            log.info(
                "并行配置: linear_group_size=%d, intra_parallel=%s (rows=%d, cols=%d), intra_part_sort_mode=%s, total_num_models=%d",
                linear_group_size,
                intra_parallel_desc,
                intra_row_parts,
                intra_col_parts,
                intra_part_sort_mode,
                linear_group_size * intra_parts_per_linear,
            )
        else:
            per_cat_desc = ",".join(
                f"{cat}:{_format_intra_parallel_desc(*category_intra_parallel[cat])}"
                for cat in active_categories
            )
            models_per_group_values = sorted(
                linear_group_size * int(rp) * int(cp)
                for rp, cp in unique_parallel
            )
            log.info(
                "并行配置: linear_group_size=%d, intra_parallel=per_category{%s}, intra_part_sort_mode=%s, total_num_models_per_group=[%d,%d]",
                linear_group_size,
                per_cat_desc,
                intra_part_sort_mode,
                models_per_group_values[0],
                models_per_group_values[-1],
            )
    unique_codebook = sorted(set(category_codebook.values()))
    if unique_codebook:
        if len(unique_codebook) == 1:
            cb_bits, cb_dim = unique_codebook[0]
            log.info("codebook 配置: bits=%d, dim=%d", cb_bits, cb_dim)
        else:
            per_cat_cb_desc = ",".join(
                f"{cat}:[bits={category_codebook[cat][0]},dim={category_codebook[cat][1]}]"
                for cat in active_categories
            )
            log.info("codebook 配置: per_category{%s}", per_cat_cb_desc)
    lora_round_idx = 0
    lora_schedule = getattr(cat_args, "lora_schedule", None)
    if isinstance(lora_schedule, dict) and lora_schedule:
        log.info(
            "LoRA category schedule enabled. keys=%s",
            ",".join(sorted(str(k) for k in lora_schedule.keys())),
        )
    for cat_idx, cat in enumerate(active_categories):
        if cat not in refs_by_cat:
            continue

        refs = refs_by_cat[cat]
        if not refs:
            continue

        cat_row_parts, cat_col_parts = category_intra_parallel[cat]
        cat_codebook_bits, cat_codebook_dim = category_codebook[cat]
        cat_vae_args = _clone_namespace(
            vae_args,
            codebook_bits=int(cat_codebook_bits),
            codebook_dim=int(cat_codebook_dim),
        )
        cat_parts_per_linear = int(cat_row_parts) * int(cat_col_parts)
        cat_intra_parallel_desc = _format_intra_parallel_desc(cat_row_parts, cat_col_parts)
        log.info(
            "=== Category: %s (%d linears, intra_parallel=%s rows=%d cols=%d, codebook_bits=%d, codebook_dim=%d) ===",
            cat,
            len(refs),
            cat_intra_parallel_desc,
            cat_row_parts,
            cat_col_parts,
            int(cat_codebook_bits),
            int(cat_codebook_dim),
        )

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
                "---- Group: %s (linears=%d, intra_parallel=%s, num_models=%d) ----",
                group_tag,
                len(group_refs),
                cat_intra_parallel_desc,
                len(group_refs) * cat_parts_per_linear,
            )
            _train_group_vae_and_replace(
                model=model,
                group_refs=group_refs,
                group_tag=group_tag,
                vae_args=cat_vae_args,
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
                intra_parallel=(cat_row_parts, cat_col_parts),
                intra_part_sort_mode=intra_part_sort_mode,
                skip_layer_keys=skip_layer_keys,
                activation_weight_by_linear=activation_weight_by_linear,
                wa_mse_runtime=wa_mse_runtime,
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
                lora_round_idx=lora_round_idx,
                after_category=cat,
            )
            lora_round_idx += 1

        _eval_ppl_after_category(
            model=model,
            vae_args=vae_args,
            ppl_limit=cat_args.ppl_limit,
            category=cat,
            eval_device=cat_args.train_device,
        )
        # cat_dir_name = _safe_path_token(cat)
        # cat_model_dir = os.path.join(run_output_dir, cat_dir_name)
        # save_paths = save_model_checkpoint(
        #     model,
        #     cat_model_dir,
        #     base_model_path=vae_args.model_path,
        #     tokenizer=None,
        #     save_config=True,
        #     extra_meta={
        #         "stage": "after_category",
        #         "category": cat,
        #         "category_index": int(cat_idx),
        #         "lora_after_category": bool(cat_args.lora_after_category),
        #     },
        # )
        # log.info("Saved category checkpoint (%s): %s", cat, save_paths["output_dir"])

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
