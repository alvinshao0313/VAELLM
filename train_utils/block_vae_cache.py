import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from train_utils import cat_train_pipeline as cat_train_impl
from e2e_common.checkpoint_io import load_e2e_model_checkpoint, save_e2e_model_checkpoint
from train_utils.block_distill import validate_qwen3_model
from train_utils.block_vae_lora_args import BlockVaeLoraArgs, format_skip_layers
from train_utils.cat_train_runtime import load_model_for_cat_train
from train_utils.utils import LinearRef, configure_deterministic_mode, get_logger, set_seed


BLOCK_VAE_CACHE_PAYLOAD_VERSION = 1
BLOCK_VAE_CATEGORY_PRETRAIN_STAGE = "block_vae_category_pretrained"


def _to_jsonable(value):
    if hasattr(value, "to_jsonable") and callable(getattr(value, "to_jsonable")):
        return value.to_jsonable()
    if is_dataclass(value):
        return {key: _to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, argparse.Namespace):
        return {key: _to_jsonable(val) for key, val in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _set_cat_train_logger(logger) -> None:
    cat_train_impl.log = logger


def stable_json_hash(value: object) -> str:
    payload = json.dumps(_to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _module_shape_key(ref: LinearRef, runtime_cfg) -> Tuple[object, ...]:
    weight = ref.module.weight
    effective = weight.t() if bool(ref.transpose) else weight
    row_parts, col_parts = tuple(runtime_cfg.intra_parallel)
    return (
        int(effective.numel()),
        int(row_parts) * int(col_parts),
        int(runtime_cfg.residual_stages),
        int(runtime_cfg.steps),
        tuple(runtime_cfg.intra_parallel),
        str(runtime_cfg.intra_part_sort_mode),
        int(runtime_cfg.codebook_bits),
        int(runtime_cfg.codebook_dim),
        str(runtime_cfg.recon_loss_type),
        int(runtime_cfg.base_ch),
        int(runtime_cfg.num_res_blocks),
        None if runtime_cfg.decoder_base_ch is None else int(runtime_cfg.decoder_base_ch),
        None if runtime_cfg.decoder_num_res_blocks is None else int(runtime_cfg.decoder_num_res_blocks),
        str(runtime_cfg.norm_type),
        str(runtime_cfg.decoder_type),
    )


def collect_block_linear_refs(
    model: nn.Module,
    *,
    layer_idx: int,
    transpose_modules: Sequence[str],
) -> Dict[str, LinearRef]:
    transpose_set = set(str(item) for item in transpose_modules)
    layer = model.model.layers[int(layer_idx)]
    refs = {
        "q_proj": layer.self_attn.q_proj,
        "k_proj": layer.self_attn.k_proj,
        "v_proj": layer.self_attn.v_proj,
        "o_proj": layer.self_attn.o_proj,
        "gate_proj": layer.mlp.gate_proj,
        "up_proj": layer.mlp.up_proj,
        "down_proj": layer.mlp.down_proj,
    }
    out: Dict[str, LinearRef] = {}
    for category, module in refs.items():
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"Layer {layer_idx}.{category} must be nn.Linear before VAE conversion, got {type(module)}."
            )
        name = f"model.layers.{int(layer_idx)}."
        if category in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            name += f"self_attn.{category}"
        else:
            name += f"mlp.{category}"
        out[category] = LinearRef(
            name=name,
            module=module,
            category=category,
            transpose=category in transpose_set,
        )
    return out


def _get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    current = model
    for token in str(module_name).split("."):
        if not hasattr(current, token):
            raise ValueError(f"Failed to resolve module {module_name!r}: missing {token!r}.")
        current = getattr(current, token)
    if not isinstance(current, nn.Module):
        raise TypeError(f"Resolved object at {module_name!r} is not an nn.Module: {type(current)}")
    return current


def _block_projection_module_name(layer_idx: int, category: str) -> str:
    parent = "self_attn" if category in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
    return f"model.layers.{int(layer_idx)}.{parent}.{category}"


def _collect_category_group_refs(
    model: nn.Module,
    *,
    module_names: Sequence[str],
    category: str,
    transpose_modules: Sequence[str],
) -> List[LinearRef]:
    transpose_set = set(str(item) for item in transpose_modules)
    refs: List[LinearRef] = []
    for name in module_names:
        module = _get_module_by_name(model, str(name))
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{name}: expected nn.Linear before category VAE pretrain, got {type(module)}.")
        refs.append(
            LinearRef(
                name=str(name),
                module=module,
                category=str(category),
                transpose=str(category) in transpose_set,
            )
        )
    return refs


def planned_block_groups(
    refs_by_category: Dict[str, LinearRef],
    resolved_cfgs,
) -> List[List[LinearRef]]:
    preferred_groups = (
        ("q_proj", "o_proj"),
        ("k_proj", "v_proj"),
        ("gate_proj", "up_proj", "down_proj"),
    )
    groups: List[List[LinearRef]] = []
    for categories in preferred_groups:
        by_key: Dict[Tuple[object, ...], List[LinearRef]] = {}
        for category in categories:
            if category not in refs_by_category:
                continue
            ref = refs_by_category[category]
            key = _module_shape_key(ref, resolved_cfgs[category])
            by_key.setdefault(key, []).append(ref)
        groups.extend(by_key.values())
    return groups


def _runtime_cfg_manifest(runtime_cfg) -> Dict[str, Any]:
    return {
        "category": str(runtime_cfg.category),
        "residual_stages": int(runtime_cfg.residual_stages),
        "steps": int(runtime_cfg.steps),
        "intra_parallel": [int(runtime_cfg.intra_parallel[0]), int(runtime_cfg.intra_parallel[1])],
        "intra_part_sort_mode": str(runtime_cfg.intra_part_sort_mode),
        "codebook_bits": int(runtime_cfg.codebook_bits),
        "codebook_dim": int(runtime_cfg.codebook_dim),
        "recon_loss_type": str(runtime_cfg.recon_loss_type),
        "base_ch": int(runtime_cfg.base_ch),
        "num_res_blocks": int(runtime_cfg.num_res_blocks),
        "decoder_base_ch": None if runtime_cfg.decoder_base_ch is None else int(runtime_cfg.decoder_base_ch),
        "decoder_num_res_blocks": None
        if runtime_cfg.decoder_num_res_blocks is None
        else int(runtime_cfg.decoder_num_res_blocks),
        "norm_type": str(runtime_cfg.norm_type),
        "decoder_type": str(runtime_cfg.decoder_type),
    }


def build_category_pretrain_manifest(
    *,
    args: BlockVaeLoraArgs,
    selected_layers: Sequence[int],
    skip_layer_keys: Sequence[Tuple[int, str]],
    transpose_modules: Sequence[str],
    resolved_cfgs,
) -> Dict[str, Any]:
    return {
        "format": "block_vae_category_pretrain",
        "payload_version": int(BLOCK_VAE_CACHE_PAYLOAD_VERSION),
        "model_path": str(args.model_path),
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "block_layers": sorted(int(v) for v in selected_layers),
        "skip_layers": format_skip_layers(sorted(skip_layer_keys)),
        "transpose_modules": list(str(v) for v in transpose_modules),
        "block_vae_categories": [str(category) for category in args.block_vae_categories],
        "linear_group_size": int(args.block_vae_linear_group_size),
        "allow_tail_group": bool(args.block_vae_allow_tail_group),
        "runtime_cfgs": {
            str(category): _runtime_cfg_manifest(resolved_cfgs[category])
            for category in args.block_vae_categories
        },
    }


def build_category_pretrain_tasks(
    *,
    model: nn.Module,
    args: BlockVaeLoraArgs,
    selected_layers: Sequence[int],
    skip_layer_keys: Sequence[Tuple[int, str]],
    transpose_modules: Sequence[str],
    resolved_cfgs,
) -> Tuple[List[Dict[str, Any]], str]:
    selected = sorted(int(v) for v in selected_layers)
    skip_set = set(skip_layer_keys)
    pretrain_manifest = build_category_pretrain_manifest(
        args=args,
        selected_layers=selected,
        skip_layer_keys=skip_layer_keys,
        transpose_modules=transpose_modules,
        resolved_cfgs=resolved_cfgs,
    )
    pretrain_hash = stable_json_hash(pretrain_manifest)
    tasks: List[Dict[str, Any]] = []
    group_size = int(args.block_vae_linear_group_size)
    for category_idx, category in enumerate(args.block_vae_categories):
        category_pairs: List[Tuple[int, str]] = []
        for layer_idx in selected:
            if (int(layer_idx), str(category)) in skip_set:
                continue
            category_pairs.append((int(layer_idx), _block_projection_module_name(int(layer_idx), str(category))))
        if not category_pairs:
            continue
        for group_idx, start in enumerate(range(0, len(category_pairs), group_size)):
            group_pairs = category_pairs[start:start + group_size]
            if len(group_pairs) < group_size and not bool(args.block_vae_allow_tail_group):
                raise ValueError(
                    f"Category {category}: tail group size={len(group_pairs)} is smaller than "
                    f"--block_vae_linear_group_size={group_size}. "
                    "Set --block_vae_allow_tail_group true or choose a compatible group size."
                )
            module_names = [name for _layer_idx, name in group_pairs]
            group_refs = _collect_category_group_refs(
                model,
                module_names=module_names,
                category=str(category),
                transpose_modules=transpose_modules,
            )
            layer_indices = [layer_idx for layer_idx, _name in group_pairs]
            tasks.append(
                {
                    "task_kind": "category_group",
                    "category": str(category),
                    "group_idx": int(group_idx),
                    "layer_indices": layer_indices,
                    "module_names": module_names,
                    "group_tag": f"{category}.L{int(layer_indices[0])}-{int(layer_indices[-1])}",
                    "shuffle_seed": int(args.seed) + int(category_idx) * 100000 + int(group_idx) * 1000,
                }
            )
    return tasks, pretrain_hash


def _train_category_tasks_on_loaded_model(
    *,
    model: nn.Module,
    tasks: Sequence[Dict[str, Any]],
    device: str,
    local_cat_args,
    training_args,
    vae_args,
    transpose_modules: Sequence[str],
    resolved_cfgs,
    logger,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    payloads: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for task in tasks:
        set_seed(int(task["shuffle_seed"]))
        group_refs = _collect_category_group_refs(
            model,
            module_names=[str(v) for v in task["module_names"]],
            category=str(task["category"]),
            transpose_modules=transpose_modules,
        )
        payload = cat_train_impl.train_group_vae_payload(
            model=model,
            group_refs=group_refs,
            group_tag=str(task["group_tag"]),
            runtime_cfg=resolved_cfgs[str(task["category"])],
            vae_args=vae_args,
            training_args=training_args,
            train_device=str(device),
            convert_device=str(device),
            do_convert=True,
            batch_size=local_cat_args.batch_size,
            log_every=local_cat_args.log_every,
            eval_every=local_cat_args.eval_every,
            eval_blocks=local_cat_args.eval_blocks,
            gpu_resident_data=bool(getattr(local_cat_args, "gpu_resident_data", False)),
            skip_layer_keys=set(),
            activation_runtime=None,
            outlier_protect_mode="none",
            outlier_rank_metric="sparse_residual_abs",
            outlier_residual_min_abs=0.0,
            outlier_protect_axis="input",
            outlier_residual_codec="coo_fp16",
            outlier_residual_index_bits=8,
            outlier_residual_value_bits=8,
            outlier_residual_block_shape=(256, 256),
            deterministic=bool(local_cat_args.deterministic),
            shuffle_seed=int(task["shuffle_seed"]),
        )
        if payload is None:
            raise RuntimeError(f"{task['group_tag']}: missing VAE payload.")
        logger.info(
            "[category %s] trained VAE group=%d layers=%s",
            str(task["category"]),
            int(task["group_idx"]),
            ",".join(str(layer_idx) for layer_idx in task["layer_indices"]),
        )
        payloads.append((task, payload))
    return payloads


def _run_category_pretrain_worker(
    *,
    worker_idx: int,
    device: str,
    tasks: Sequence[Dict[str, Any]],
    args: BlockVaeLoraArgs,
    hf_args,
    training_args,
    vae_args,
    cat_args,
    transpose_modules: Sequence[str],
    resolved_cfgs,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    logger = get_logger(f"block_vae_category_pretrain_worker_{int(worker_idx)}")
    _set_cat_train_logger(logger)
    local_cat_args = argparse.Namespace(**vars(cat_args))
    local_cat_args.train_device = str(device)
    local_cat_args.convert_device = str(device)
    configure_deterministic_mode(bool(args.deterministic))
    model = load_model_for_cat_train(cat_args=local_cat_args, hf_args=hf_args, vae_args=vae_args)
    validate_qwen3_model(model)
    payloads = _train_category_tasks_on_loaded_model(
        model=model,
        tasks=tasks,
        device=str(device),
        local_cat_args=local_cat_args,
        training_args=training_args,
        vae_args=vae_args,
        transpose_modules=transpose_modules,
        resolved_cfgs=resolved_cfgs,
        logger=logger,
    )
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payloads


def _run_category_tasks(
    *,
    tasks: Sequence[Dict[str, Any]],
    args: BlockVaeLoraArgs,
    hf_args,
    training_args,
    vae_args,
    cat_args,
    transpose_modules: Sequence[str],
    resolved_cfgs,
    logger,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if not tasks:
        return []
    devices = _parse_block_vae_pretrain_devices(args)
    worker_count = max(1, int(args.block_vae_pretrain_workers or len(devices)))
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(worker_count)]
    for idx, task in enumerate(tasks):
        buckets[idx % worker_count].append(task)
    worker_kwargs = [
        {
            "worker_idx": idx,
            "device": devices[idx % len(devices)],
            "tasks": bucket,
            "args": args,
            "hf_args": hf_args,
            "training_args": training_args,
            "vae_args": vae_args,
            "cat_args": cat_args,
            "transpose_modules": transpose_modules,
            "resolved_cfgs": resolved_cfgs,
        }
        for idx, bucket in enumerate(buckets)
        if bucket
    ]
    logger.info(
        "Category VAE pretrain workers start: category=%s missing_groups=%d workers=%d devices=%s",
        str(tasks[0]["category"]) if tasks else "",
        len(tasks),
        len(worker_kwargs),
        ",".join(devices),
    )
    if len(worker_kwargs) == 1:
        return _run_category_pretrain_worker(**worker_kwargs[0])
    else:
        payloads: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        with ProcessPoolExecutor(max_workers=len(worker_kwargs)) as executor:
            futures = [executor.submit(_run_category_pretrain_worker, **kwargs) for kwargs in worker_kwargs]
            for future in futures:
                payloads.extend(future.result())
        return sorted(payloads, key=lambda item: int(item[0]["group_idx"]))


def _apply_category_payloads(
    *,
    model: nn.Module,
    payloads: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    category: str,
    transpose_modules: Sequence[str],
    convert_device: str,
    logger,
) -> None:
    for task, payload in sorted(payloads, key=lambda item: int(item[0]["group_idx"])):
        group_refs = _collect_category_group_refs(
            model,
            module_names=[str(item) for item in task["module_names"]],
            category=str(category),
            transpose_modules=transpose_modules,
        )
        cat_train_impl.apply_group_vae_payload(
            model=model,
            group_refs=group_refs,
            group_tag=str(task["group_tag"]),
            payload=payload,
            convert_device=str(convert_device),
            skip_layer_keys=set(),
        )
        logger.info(
            "[category %s] applied VAE group=%d layers=%s",
            str(category),
            int(task["group_idx"]),
            ",".join(str(layer_idx) for layer_idx in task["layer_indices"]),
        )


def _checkpoint_extra_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    extra = meta.get("extra_meta", {})
    if not isinstance(extra, dict):
        raise TypeError("checkpoint_meta.extra_meta must be a dict.")
    return extra


def _category_pretrain_extra_meta(
    *,
    args: BlockVaeLoraArgs,
    selected_layers: Sequence[int],
    skip_layer_keys: Sequence[Tuple[int, str]],
    transpose_modules: Sequence[str],
    pretrain_hash: str,
) -> Dict[str, Any]:
    return {
        "stage": BLOCK_VAE_CATEGORY_PRETRAIN_STAGE,
        "block_distill": _to_jsonable(args),
        "block_distill_train_mode": str(args.block_distill_train_mode),
        "block_vae_pipeline_mode": str(args.block_vae_pipeline_mode),
        "block_vae_pretrain_manifest_hash": str(pretrain_hash),
        "block_vae_categories": [str(category) for category in args.block_vae_categories],
        "block_vae_linear_group_size": int(args.block_vae_linear_group_size),
        "block_vae_allow_tail_group": bool(args.block_vae_allow_tail_group),
        "selected_block_layers": sorted(int(layer_idx) for layer_idx in selected_layers),
        "skip_layers": format_skip_layers(sorted(skip_layer_keys)),
        "transpose_modules": [str(item) for item in transpose_modules],
    }


def validate_block_vae_category_pretrained_meta(
    meta: Dict[str, Any],
    *,
    args: BlockVaeLoraArgs,
    selected_layers: Sequence[int],
    skip_layer_keys: Sequence[Tuple[int, str]],
    transpose_modules: Sequence[str],
    resolved_cfgs,
) -> str:
    extra = _checkpoint_extra_meta(meta)
    stage = str(extra.get("stage", "")).strip()
    if stage != BLOCK_VAE_CATEGORY_PRETRAIN_STAGE:
        raise ValueError(
            "Block VAE pretrained checkpoint stage mismatch: "
            f"checkpoint={stage!r} expected={BLOCK_VAE_CATEGORY_PRETRAIN_STAGE!r}."
        )
    return str(extra.get("block_vae_pretrain_manifest_hash", ""))


def compute_block_vae_category_pretrain_hash(
    *,
    args: BlockVaeLoraArgs,
    selected_layers: Sequence[int],
    skip_layer_keys: Sequence[Tuple[int, str]],
    transpose_modules: Sequence[str],
    resolved_cfgs,
) -> str:
    expected_manifest = build_category_pretrain_manifest(
        args=args,
        selected_layers=selected_layers,
        skip_layer_keys=skip_layer_keys,
        transpose_modules=transpose_modules,
        resolved_cfgs=resolved_cfgs,
    )
    return stable_json_hash(expected_manifest)


def load_block_vae_category_pretrained_model(
    checkpoint_dir: str,
    *,
    access_token: Optional[str],
    proxy_group_size: int,
    proxy_compute_device: object,
    logger=None,
):
    model, meta, load_result = load_e2e_model_checkpoint(
        str(checkpoint_dir),
        access_token=access_token,
        map_location="cpu",
        strict=True,
        materialize_proxy_decoded_linears=True,
        proxy_group_size=int(proxy_group_size),
        proxy_compute_device=proxy_compute_device,
        proxy_logger=logger,
    )
    extra = _checkpoint_extra_meta(meta)
    stage = str(extra.get("stage", "")).strip()
    if stage != BLOCK_VAE_CATEGORY_PRETRAIN_STAGE:
        raise ValueError(
            "Checkpoint is not a block VAE category-pretrained checkpoint: "
            f"{checkpoint_dir}. Expected stage={BLOCK_VAE_CATEGORY_PRETRAIN_STAGE!r}, got {stage!r}."
        )
    return model, meta, load_result


def run_block_vae_category_pretrain(
    *,
    model: nn.Module,
    tasks: Sequence[Dict[str, Any]],
    pretrain_hash: str,
    output_dir: str,
    args: BlockVaeLoraArgs,
    hf_args,
    training_args,
    vae_args,
    cat_args,
    transpose_modules: Sequence[str],
    resolved_cfgs,
    selected_layers: Sequence[int],
    skip_layer_keys: Sequence[Tuple[int, str]],
    logger,
) -> Dict[str, str]:
    if not str(output_dir).strip():
        raise ValueError("Block VAE category pretrain output_dir must be non-empty.")
    devices = _parse_block_vae_pretrain_devices(args)
    worker_count = max(1, int(args.block_vae_pretrain_workers or len(devices)))
    use_loaded_model_fast_path = int(worker_count) == 1
    if use_loaded_model_fast_path:
        logger.info(
            "Category VAE pretrain uses loaded main model fast path: workers=1 device=%s",
            str(devices[0]),
        )

    for category in args.block_vae_categories:
        category_tasks = [
            task
            for task in tasks
            if str(task["category"]) == str(category)
        ]
        if not category_tasks:
            continue
        category_tasks = sorted(category_tasks, key=lambda item: int(item["group_idx"]))
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if use_loaded_model_fast_path:
            worker_logger = get_logger("block_vae_category_pretrain_worker_0")
            _set_cat_train_logger(worker_logger)
            local_cat_args = argparse.Namespace(**vars(cat_args))
            local_cat_args.train_device = str(devices[0])
            local_cat_args.convert_device = str(devices[0])
            configure_deterministic_mode(bool(args.deterministic))
            logger.info(
                "Category VAE pretrain in-process: category=%s groups=%d device=%s",
                str(category),
                len(category_tasks),
                str(devices[0]),
            )
            payloads = _train_category_tasks_on_loaded_model(
                model=model,
                tasks=category_tasks,
                device=str(devices[0]),
                local_cat_args=local_cat_args,
                training_args=training_args,
                vae_args=vae_args,
                transpose_modules=transpose_modules,
                resolved_cfgs=resolved_cfgs,
                logger=worker_logger,
            )
            _set_cat_train_logger(logger)
        else:
            payloads = _run_category_tasks(
                tasks=category_tasks,
                args=args,
                hf_args=hf_args,
                training_args=training_args,
                vae_args=vae_args,
                cat_args=cat_args,
                transpose_modules=transpose_modules,
                resolved_cfgs=resolved_cfgs,
                logger=logger,
            )
        _apply_category_payloads(
            model=model,
            payloads=payloads,
            category=str(category),
            transpose_modules=transpose_modules,
            convert_device=str(cat_args.convert_device),
            logger=logger,
        )
        logger.info("[category %s] VAE pretrain and replacement done.", str(category))

    save_paths = save_e2e_model_checkpoint(
        model,
        str(output_dir),
        base_model_path=str(args.model_path),
        tokenizer=None,
        save_config=True,
        extra_meta=_category_pretrain_extra_meta(
            args=args,
            selected_layers=selected_layers,
            skip_layer_keys=skip_layer_keys,
            transpose_modules=transpose_modules,
            pretrain_hash=str(pretrain_hash),
        ),
        unload_vae_original_weights=False,
        compact_unload_vae_original_weights=False,
    )
    logger.info("Saved block VAE category-pretrained checkpoint: %s", save_paths["output_dir"])
    return save_paths


def _parse_block_vae_pretrain_devices(args: BlockVaeLoraArgs) -> List[str]:
    raw = str(args.block_vae_pretrain_devices or "").strip()
    devices = [item.strip() for item in raw.split(",") if item.strip()] if raw else [str(args.train_device)]
    if not devices:
        raise ValueError("--block_vae_pretrain_devices resolved to an empty device list.")
    return devices
