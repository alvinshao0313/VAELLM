import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer, default_data_collator

from dense_e2e_fintuning.args import needs_teacher
from dense_e2e_fintuning.checkpoint_bridge import (
    get_decode_device_diagnostics,
    load_compressed_student_checkpoint,
    resolve_base_model_path,
)
from dense_e2e_fintuning.runtime import _build_datasets_with_main_process_first, _eval_final_ppl
from e2e_common.data import build_tokenizer
from e2e_common.post_norm_head import ensure_post_norm_head_linear
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from litebsq.vae_linear_prewarm import NamedVAELinearDecodeTarget, decode_named_vae_linear_weights
from rotation.model_utils import get_layers, get_model
from train_utils.model_checkpoint_io import (
    STATE_DICT_FILENAME,
    _build_distributed_run_output_dir,
    save_model_checkpoint,
)
from train_utils.utils import get_logger
from vae_e2e_fintuning.device_map import apply_boundary_device_map, apply_layer_device_map, resolve_layer_device_map
from vae_e2e_fintuning.offload import (
    SavedTensorOffloadContext,
    unwrap_streaming_offload_layers,
    validate_streaming_layer_devices,
    wrap_model_layers_for_streaming_offload,
)
from vae_e2e_fintuning.trainables import (
    VAEDecoderTrainableSelection,
    collect_selected_vae_linears,
    resolve_target_layer_ids,
    select_vae_decoder_trainables,
    unpack_parallel_stage_decoders,
    validate_selected_low_rank_payloads,
)
from vae_e2e_fintuning.trainer import VAEDecoderE2ETrainer


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    return str(value)


def _namespace_to_dict(ns) -> Dict[str, Any]:
    if ns is None:
        return {}
    if is_dataclass(ns):
        return {str(k): _jsonable(v) for k, v in asdict(ns).items()}
    if hasattr(ns, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(ns).items()}
    return {"value": _jsonable(ns)}


def _selection_to_meta(selection) -> Dict[str, object]:
    data = asdict(selection)
    data["target_module_count"] = len(selection.target_modules)
    data["decoder_layer_count"] = len(selection.decoder_layer_ids)
    return data


def _iter_named_vae_linears(model: nn.Module) -> Iterable[Tuple[str, VAELinear]]:
    for name, module in model.named_modules():
        if isinstance(module, VAELinear):
            yield str(name), module


def _set_vae_decoder_checkpoint(model: nn.Module, enabled: bool) -> Tuple[int, int]:
    changed = 0
    total = 0
    for _name, module in _iter_named_vae_linears(model):
        for stage_idx in range(int(module.residual_stages)):
            for part_idx in range(int(module.parallel_parts)):
                decoder = module.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)
                total += 1
                if bool(getattr(decoder, "use_checkpoint", False)) != bool(enabled):
                    decoder.use_checkpoint = bool(enabled)
                    changed += 1
    return changed, total


def _resolve_reference_dtype(module: nn.Module) -> torch.dtype:
    for param in module.parameters():
        if param.is_floating_point():
            return param.dtype
    for buffer in module.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return torch.float32


def _copy_low_rank_payloads(
    selected_modules: Sequence[Tuple[str, VAELinear]],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, module in selected_modules:
        low_rank_a = getattr(module, "low_rank_a", None)
        low_rank_b = getattr(module, "low_rank_b", None)
        if low_rank_a is None or low_rank_b is None:
            raise ValueError(f"{name}: selected VAELinear has no complete low_rank_a/low_rank_b payload.")
        payloads[str(name)] = (
            low_rank_a.detach().to(device="cpu").contiguous(),
            low_rank_b.detach().to(device="cpu").contiguous(),
        )
    return payloads


@torch.no_grad()
def _materialize_selected_vae_linears_without_low_rank(
    model: nn.Module,
    selected_modules: Sequence[Tuple[str, VAELinear]],
    *,
    group_size: int,
    compute_device: str,
    log,
) -> int:
    decode_targets = [
        NamedVAELinearDecodeTarget(
            name=name,
            base_layer=module,
            target_dtype=_resolve_reference_dtype(module),
            include_low_rank=False,
        )
        for name, module in selected_modules
    ]
    if not decode_targets:
        raise ValueError("No selected VAELinear modules to materialize.")
    log.info(
        "Start low-rank dense init: selected=%d group_size=%d compute_device=%s include_low_rank=false",
        len(decode_targets),
        int(group_size),
        str(compute_device),
    )
    decoded_results = decode_named_vae_linear_weights(
        decode_targets,
        group_size=int(group_size),
        compute_device=compute_device,
        logger=log,
        respect_cache_policy=False,
    )
    decoded_by_name = {item.name: item for item in decoded_results}
    if len(decoded_by_name) != len(selected_modules):
        raise RuntimeError(
            f"Low-rank dense init decode count mismatch: decoded={len(decoded_by_name)} expected={len(selected_modules)}."
        )

    converted = 0
    for name, old_module in selected_modules:
        decoded = decoded_by_name[str(name)]
        dense_linear = nn.Linear(
            int(old_module.in_features),
            int(old_module.out_features),
            bias=old_module.bias is not None,
            device=decoded.decoded_weight.device,
            dtype=decoded.decoded_weight.dtype,
        )
        dense_linear.weight.copy_(decoded.decoded_weight)
        if dense_linear.bias is not None and old_module.bias is not None:
            dense_linear.bias.copy_(
                old_module.bias.detach().to(device=dense_linear.bias.device, dtype=dense_linear.bias.dtype)
            )
        dense_linear.train(old_module.training)
        dense_linear.to("cpu")
        set_module_by_name(model, str(name), dense_linear)
        converted += 1
    log.info("Finished low-rank dense init: converted=%d", converted)
    return int(converted)


def _strip_peft_module_prefix(name: str) -> str:
    text = str(name)
    for prefix in ("base_model.model.", "model."):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _iter_lora_target_modules(model: nn.Module):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            yield _strip_peft_module_prefix(str(name)), module


def _initialize_lora_from_low_rank(
    peft_model: nn.Module,
    low_rank_payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
) -> int:
    initialized = 0
    for base_name, module in _iter_lora_target_modules(peft_model):
        if base_name not in low_rank_payloads:
            continue
        low_rank_a, low_rank_b = low_rank_payloads[base_name]
        lora_a = module.lora_A["default"].weight
        lora_b = module.lora_B["default"].weight
        if tuple(lora_a.shape) != tuple(low_rank_b.shape):
            raise RuntimeError(f"{base_name}: lora_A shape {tuple(lora_a.shape)} != low_rank_b {tuple(low_rank_b.shape)}.")
        if tuple(lora_b.shape) != tuple(low_rank_a.shape):
            raise RuntimeError(f"{base_name}: lora_B shape {tuple(lora_b.shape)} != low_rank_a {tuple(low_rank_a.shape)}.")
        lora_a.data.copy_(low_rank_b.to(device=lora_a.device, dtype=lora_a.dtype))
        lora_b.data.copy_(low_rank_a.to(device=lora_b.device, dtype=lora_b.dtype))
        initialized += 1
    if initialized != len(low_rank_payloads):
        raise RuntimeError(
            f"LoRA init target count mismatch: initialized={initialized} expected={len(low_rank_payloads)}."
        )
    return int(initialized)


def _build_low_rank_peft_model(
    model: nn.Module,
    *,
    low_rank_payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    rank: int,
    decoder_layer_ids: Sequence[int],
    target_module_suffixes: Sequence[str],
    parallel_stage_decode: bool,
    log,
) -> Tuple[nn.Module, VAEDecoderTrainableSelection]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover
        raise ImportError("low_rank 训练需要 peft。请先安装：pip install peft") from exc

    target_modules = sorted(low_rank_payloads.keys())
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(rank),
        target_modules=target_modules,
        lora_alpha=float(rank),
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=True,
    )
    peft_model = get_peft_model(model, peft_config)
    initialized = _initialize_lora_from_low_rank(peft_model, low_rank_payloads)
    trainable_names = sorted(name for name, param in peft_model.named_parameters() if bool(param.requires_grad))
    trainable_count = int(
        sum(int(param.numel()) for _name, param in peft_model.named_parameters() if bool(param.requires_grad))
    )
    if trainable_count < 1:
        raise RuntimeError("No trainable low-rank LoRA parameters found.")
    log.info("Initialized low-rank LoRA targets from checkpoint payloads: initialized=%d rank=%d", initialized, int(rank))
    selection = VAEDecoderTrainableSelection(
        decoder_layer_ids=[int(idx) for idx in decoder_layer_ids],
        target_modules=target_modules,
        target_module_suffixes=list(target_module_suffixes),
        bias_modules=[],
        final_norm_modules=[],
        post_norm_head_modules=[],
        low_rank_modules=target_modules,
        trainable_parameter_names=trainable_names,
        trainable_parameter_count=trainable_count,
        parallel_stage_decode=bool(parallel_stage_decode),
        train_mode="low_rank",
    )
    return peft_model, selection


def _extract_low_rank_payloads_from_lora(
    peft_model: nn.Module,
    target_modules: Sequence[str],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    target_set = {str(name) for name in target_modules}
    payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for base_name, module in _iter_lora_target_modules(peft_model):
        if base_name not in target_set:
            continue
        lora_a = module.lora_A["default"].weight.detach()
        lora_b = module.lora_B["default"].weight.detach()
        scaling = float(module.scaling["default"])
        low_rank_b = lora_a.to(device="cpu").contiguous()
        low_rank_a = (lora_b.to(device="cpu", dtype=torch.float32) * scaling).to(dtype=lora_b.dtype).contiguous()
        payloads[base_name] = (low_rank_a, low_rank_b)
    if len(payloads) != len(target_set):
        missing = sorted(target_set - set(payloads.keys()))
        raise RuntimeError(f"Missing trained LoRA payloads for low-rank export: {missing}")
    return payloads


def _write_low_rank_payloads_to_compressed_model(
    model: nn.Module,
    payloads: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
) -> int:
    modules = dict(_iter_named_vae_linears(model))
    written = 0
    for name, (low_rank_a, low_rank_b) in payloads.items():
        module = modules.get(str(name))
        if module is None:
            raise RuntimeError(f"Cannot export low-rank payload; VAELinear not found: {name}")
        if getattr(module, "low_rank_a", None) is None or getattr(module, "low_rank_b", None) is None:
            raise RuntimeError(f"Cannot export low-rank payload; checkpoint module has no low_rank_a/b: {name}")
        if tuple(module.low_rank_a.shape) != tuple(low_rank_a.shape):
            raise RuntimeError(f"{name}: low_rank_a shape mismatch: {tuple(module.low_rank_a.shape)} != {tuple(low_rank_a.shape)}.")
        if tuple(module.low_rank_b.shape) != tuple(low_rank_b.shape):
            raise RuntimeError(f"{name}: low_rank_b shape mismatch: {tuple(module.low_rank_b.shape)} != {tuple(low_rank_b.shape)}.")
        module.low_rank_a.data.copy_(low_rank_a.to(device=module.low_rank_a.device, dtype=module.low_rank_a.dtype))
        module.low_rank_b.data.copy_(low_rank_b.to(device=module.low_rank_b.device, dtype=module.low_rank_b.dtype))
        module.clear_decoded_weight_cache()
        written += 1
    return int(written)


def _peft_base_model(model: nn.Module) -> nn.Module:
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        return get_base_model()
    return model


def _load_teacher(*, args, hf_args, base_model_path: str, log):
    if not needs_teacher(args.loss_type):
        return None, "disabled"
    teacher_path = str(args.teacher_model_path or base_model_path)
    log.info("Loading teacher model from %s", teacher_path)
    teacher_model = get_model(teacher_path, hf_args.access_token)
    teacher_model.eval()
    if hasattr(teacher_model, "config"):
        teacher_model.config.use_cache = False
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model, "external_teacher"


def _resolve_eval_device(requested_device: str) -> str:
    device = str(requested_device or "").strip()
    if not device:
        raise ValueError("--eval_device cannot be empty.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested eval_device={device}, but CUDA is not available.")
    return device


def _disable_trainable_decode_for_eval(model: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        disable_fn = getattr(module, "disable_trainable_decode", None)
        if callable(disable_fn):
            disable_fn()
            count += 1
    return count


@torch.no_grad()
def _prewarm_sparse_residual_cache(model: torch.nn.Module, *, dtype: Optional[torch.dtype], log) -> Dict[str, int]:
    from e2e_common.proxy_trainables import iter_named_vae_module_refs

    total = 0
    warmed = 0
    skipped = 0
    failed = 0
    for ref in iter_named_vae_module_refs(model):
        total += 1
        module = ref.base_layer
        has_sparse_fn = getattr(module, "has_sparse_residual", None)
        has_sparse = bool(has_sparse_fn()) if callable(has_sparse_fn) else False
        if not has_sparse:
            skipped += 1
            continue
        prime_fn = getattr(module, "prime_sparse_residual_cache", None)
        if not callable(prime_fn):
            failed += 1
            raise RuntimeError(f"{ref.name}: VAELinear has no prime_sparse_residual_cache method.")
        try:
            did_warm = bool(prime_fn(dtype=dtype))
        except Exception as exc:
            failed += 1
            raise RuntimeError(f"Sparse residual prewarm failed for '{ref.name}': {exc}") from exc
        if did_warm:
            warmed += 1
        else:
            skipped += 1

    log.info(
        "[sparse_residual_prewarm] done: total=%d warmed=%d skipped=%d failed=%d dtype=%s",
        int(total),
        int(warmed),
        int(skipped),
        int(failed),
        str(dtype),
    )
    return {
        "total": int(total),
        "warmed": int(warmed),
        "skipped": int(skipped),
        "failed": int(failed),
    }


def _resolve_train_sparse_residual_dtype(training_args) -> Optional[torch.dtype]:
    if bool(getattr(training_args, "bf16", False)):
        return torch.bfloat16
    if bool(getattr(training_args, "fp16", False)):
        return torch.float16
    return None


def _clear_vae_linear_cache(model: torch.nn.Module, log, *, reason: str) -> int:
    from litebsq.vae_linear import clear_model_vae_linear_cache

    cleared = clear_model_vae_linear_cache(model)
    log.info("%s: cleared decoded cache for %d VAELinear modules.", reason, int(cleared))
    return int(cleared)


def _prewarm_final_eval_model(*, model: torch.nn.Module, args, log) -> Dict[str, int]:
    device = _resolve_eval_device(str(args.eval_device))
    group_size = int(getattr(args, "eval_prewarm_group_size", 8))
    log.info("[eval_prewarm] Moving final model to %s ...", device)
    model.to(device)

    from e2e_common.proxy_trainables import iter_named_vae_module_refs
    from litebsq.vae_linear import NamedVAELinearTarget, prime_named_vae_linear_cache

    named_targets = [
        NamedVAELinearTarget(name=ref.name, base_layer=ref.base_layer)
        for ref in iter_named_vae_module_refs(model)
    ]
    stats = prime_named_vae_linear_cache(
        named_targets,
        clear_existing=True,
        group_size=group_size,
        compute_device=device,
        logger=log,
    )
    if int(stats.get("failed", 0)) > 0:
        raise RuntimeError(f"Final eval prewarm failed: {stats}")
    log.info(
        "[eval_prewarm] done: total=%d warmed=%d skipped=%d failed=%d group_size=%d device=%s",
        int(stats.get("total", 0)),
        int(stats.get("warmed", 0)),
        int(stats.get("skipped", 0)),
        int(stats.get("failed", 0)),
        group_size,
        device,
    )
    return stats


def _run_final_lm_eval(*, model, tokenizer, args, base_model_path: str, output_dir: str, log):
    tasks = None if args.eval_tasks is None else str(args.eval_tasks).strip()
    if not tasks:
        return None

    device = _resolve_eval_device(str(args.eval_device))
    eval_log_dir = os.path.join(output_dir, "lm_eval")
    os.makedirs(eval_log_dir, exist_ok=True)
    log.info("[lm_eval] Moving final saved model to %s ...", device)
    model.to(device)

    from train_utils.eval_utils import run_lm_eval
    from train_utils.hif4_act import applied_hif4_act

    lm_args = argparse.Namespace(
        tasks=tasks,
        num_fewshot=int(args.eval_num_fewshot),
        batch_size=str(args.eval_lm_batch_size),
        lm_limit=args.eval_lm_limit,
        model_path=str(base_model_path),
        eval_log_dir=eval_log_dir,
        eval_run_ts="final",
    )
    log.info(
        "[lm_eval] tasks=%s fewshot=%d batch_size=%s limit=%s",
        tasks,
        int(args.eval_num_fewshot),
        str(args.eval_lm_batch_size),
        str(args.eval_lm_limit),
    )
    with applied_hif4_act(
        model,
        enabled=bool(args.eval_hif4_act),
        logger=log,
        log_prefix="[lm_eval] ",
    ):
        result = run_lm_eval(model, tokenizer, lm_args)

    table = str(result.get("summary_table", "")).strip()
    if table:
        log.info("[lm_eval] Summary table:\n%s", table)
    return {
        "result": result,
        "json_path": os.path.join(eval_log_dir, "lm_eval_results_final.json"),
        "summary_path": os.path.join(eval_log_dir, "lm_eval_summary_final.md"),
    }


def _build_run_meta(
    *,
    args,
    hf_args,
    training_args,
    data_info: Dict[str, Any],
    trainable_info: Dict[str, Any],
    teacher_source: str,
    resolved_student_checkpoint_dir: str,
    base_model_path: str,
    layer_device_map: Dict[str, str],
    offload_mode: str,
    global_step: int,
) -> Dict[str, Any]:
    return {
        "format": "vae_decoder_e2e_run_meta",
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "vae_e2e_fintuning",
        "teacher_source": str(teacher_source),
        "student_checkpoint_dir": str(resolved_student_checkpoint_dir),
        "base_model_path": str(base_model_path),
        "source_checkpoint_state_dict_file": STATE_DICT_FILENAME,
        "layer_device_map": dict(layer_device_map),
        "offload_mode": str(offload_mode),
        "global_step": int(global_step),
        "vae_e2e_args": _namespace_to_dict(args),
        "hf_args": _namespace_to_dict(hf_args),
        "training_args": _namespace_to_dict(training_args),
        "dataset": _jsonable(data_info),
        "trainables": _jsonable(trainable_info),
    }


def run(args, hf_args, training_args):
    run_output_dir = _build_distributed_run_output_dir(
        args.run_root_dir,
        os.path.basename(args.student_checkpoint_dir),
    )
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "vae_e2e_fintuning.log")
    log = get_logger("vae_e2e_fintuning")
    resume_from_checkpoint = None if args.resume_from_checkpoint is None else str(args.resume_from_checkpoint).strip()

    log.info("Run output directory: %s", run_output_dir)
    log.info("Resolved student checkpoint directory: %s", args.student_checkpoint_dir)
    log.info("Input VAE decoder e2e args:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2))

    model, meta, resolved_student_checkpoint_dir = load_compressed_student_checkpoint(
        args.student_checkpoint_dir,
        access_token=hf_args.access_token,
        logger=log,
    )
    base_model_path = resolve_base_model_path(meta, args.teacher_model_path)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if bool(args.use_post_norm_head_linear):
        ensure_post_norm_head_linear(model)

    decoder_ckpt_changed, decoder_ckpt_total = _set_vae_decoder_checkpoint(
        model,
        enabled=bool(args.vae_decoder_checkpoint),
    )
    log.info(
        "Applied VAE decoder checkpoint config: enabled=%s changed=%d total=%d",
        str(bool(args.vae_decoder_checkpoint)).lower(),
        int(decoder_ckpt_changed),
        int(decoder_ckpt_total),
    )

    layers = list(get_layers(model))
    decoder_layer_ids = resolve_target_layer_ids(args.decoder_layer_ids, len(layers))
    train_mode = str(getattr(args, "vae_train_mode", "decoder")).strip().lower()
    low_rank_payloads_for_export: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
    if train_mode == "low_rank":
        selected_modules, target_module_suffixes = collect_selected_vae_linears(
            model,
            decoder_layer_ids=decoder_layer_ids,
            target_module_names=args.target_module_names,
        )
        if not selected_modules:
            raise ValueError("No eligible VAELinear modules found for requested decoder_layers / target_modules.")
        low_rank_rank = validate_selected_low_rank_payloads(
            selected_modules,
            require_uniform_rank=True,
        )
        low_rank_payloads = _copy_low_rank_payloads(selected_modules)
        requested_decode_device = str(getattr(args, "decode_device", "auto"))
        decode_device_diag = get_decode_device_diagnostics(requested_decode_device)
        resolved_decode_device = str(decode_device_diag["resolved_device"])
        log.info(
            "Low-rank dense init decode config: requested_device=%s resolved_device=%s group_size=%d",
            requested_decode_device,
            resolved_decode_device,
            int(args.decode_group_size),
        )
        log.info(
            "Low-rank decode device diagnostics: LOCAL_RANK=%s CUDA_VISIBLE_DEVICES=%s visible_cuda_count=%d",
            str(decode_device_diag["local_rank"]),
            str(decode_device_diag["cuda_visible_devices"]),
            int(decode_device_diag["visible_cuda_count"]),
        )
        _materialize_selected_vae_linears_without_low_rank(
            model,
            selected_modules,
            group_size=int(args.decode_group_size),
            compute_device=resolved_decode_device,
            log=log,
        )
        model, selection = _build_low_rank_peft_model(
            model,
            low_rank_payloads=low_rank_payloads,
            rank=int(low_rank_rank),
            decoder_layer_ids=decoder_layer_ids,
            target_module_suffixes=target_module_suffixes,
            parallel_stage_decode=bool(args.parallel_stage_decode),
            log=log,
        )
    else:
        selection = select_vae_decoder_trainables(
            model,
            decoder_layer_ids=decoder_layer_ids,
            target_module_names=args.target_module_names,
            parallel_stage_decode=bool(args.parallel_stage_decode),
            tune_final_norm=bool(args.tune_final_norm),
            use_post_norm_head_linear=bool(args.use_post_norm_head_linear),
            vae_tune_bias=bool(args.vae_tune_bias),
            train_mode=train_mode,
        )
    log.info(
        "Selected VAE trainables: mode=%s layers=%s targets=%d suffixes=%s bias_modules=%d low_rank_modules=%d final_norm=%s post_norm_head=%s trainable_tensors=%d trainable_params=%d parallel_stage_decode=%s",
        selection.train_mode,
        selection.decoder_layer_ids,
        len(selection.target_modules),
        selection.target_module_suffixes,
        len(selection.bias_modules),
        len(selection.low_rank_modules),
        selection.final_norm_modules,
        selection.post_norm_head_modules,
        len(selection.trainable_parameter_names),
        int(selection.trainable_parameter_count),
        str(selection.parallel_stage_decode).lower(),
    )

    resolved_layer_device_map = resolve_layer_device_map(args.layer_device_map, len(layers))
    device_map_model = _peft_base_model(model) if train_mode == "low_rank" else model
    offload_mode = str(args.offload_mode).strip().lower()
    streaming_manager = None
    saved_tensor_offload = None
    if offload_mode == "streaming":
        validate_streaming_layer_devices(resolved_layer_device_map)
        if bool(getattr(training_args, "gradient_checkpointing", False)):
            log.info(
                "offload_mode=streaming manages layer checkpointing itself; overriding HF gradient_checkpointing=false."
            )
            training_args.gradient_checkpointing = False
        hook_handles, boundary_map = apply_boundary_device_map(device_map_model, layer_device_map=resolved_layer_device_map)
        streaming_manager, streaming_map = wrap_model_layers_for_streaming_offload(
            device_map_model,
            layer_devices=resolved_layer_device_map,
            prefetch_distance=int(args.offload_prefetch_distance),
            checkpoint_layers=bool(args.offload_checkpoint),
        )
        hf_device_map = {**boundary_map, **streaming_map}
    else:
        hook_handles, hf_device_map = apply_layer_device_map(device_map_model, layer_device_map=resolved_layer_device_map)
    if train_mode == "low_rank":
        setattr(model, "hf_device_map", hf_device_map)
        setattr(model, "is_parallelizable", True)
        setattr(model, "model_parallel", True)

    if offload_mode in {"saved_tensors", "streaming"}:
        saved_tensor_offload = SavedTensorOffloadContext(
            enabled=True,
            min_tensor_bytes=int(args.offload_min_tensor_bytes),
            pin_memory=bool(args.offload_pin_memory),
        )

    log.info(
        "Applied layer device/offload config: offload_mode=%s offload_checkpoint=%s map=%s saved_tensor_min_bytes=%d pin_memory=%s",
        offload_mode,
        str(bool(args.offload_checkpoint)).lower(),
        json.dumps(hf_device_map, ensure_ascii=False, sort_keys=True),
        int(args.offload_min_tensor_bytes),
        str(bool(args.offload_pin_memory)).lower(),
    )
    sparse_residual_prewarm = _prewarm_sparse_residual_cache(
        model,
        dtype=_resolve_train_sparse_residual_dtype(training_args),
        log=log,
    )

    tokenizer = build_tokenizer(str(base_model_path), access_token=hf_args.access_token)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset, eval_dataset, data_info = _build_datasets_with_main_process_first(
        args,
        training_args,
        tokenizer,
        log,
    )
    if len(train_dataset) < 1:
        raise ValueError("Packed training dataset is empty. Increase input text volume or lower --model_max_length.")
    if eval_dataset is not None and len(eval_dataset) < 1:
        eval_dataset = None
    log.info(
        "Prepared datasets: train=%d eval=%s block_size=%d",
        len(train_dataset),
        "none" if eval_dataset is None else str(len(eval_dataset)),
        int(data_info["block_size"]),
    )

    teacher_model, teacher_source = _load_teacher(
        args=args,
        hf_args=hf_args,
        base_model_path=str(base_model_path),
        log=log,
    )
    log.info("Teacher source: %s", teacher_source)

    training_args.output_dir = os.path.join(run_output_dir, "trainer_state")
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False
    if bool(getattr(training_args, "save_safetensors", False)):
        log.info("Disabling safetensors for Trainer checkpoints; VAE state_dict may contain shared storage.")
        training_args.save_safetensors = False

    trainer = VAEDecoderE2ETrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        loss_type=args.loss_type,
        teacher_model=teacher_model,
        distill_temperature=args.distill_temperature,
        distill_alpha=args.distill_alpha,
        post_attn=bool(args.post_attn),
        saved_tensor_offload=saved_tensor_offload,
        streaming_offload_manager=streaming_manager,
    )
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)
    except Exception:
        if streaming_manager is not None:
            streaming_manager.offload_all(synchronize=True)
        for handle in hook_handles:
            handle.remove()
        raise

    final_model = trainer.accelerator.unwrap_model(trainer.model) if getattr(trainer, "accelerator", None) else trainer.model
    final_model.eval()
    if teacher_model is not None:
        teacher_model.to("cpu")

    for handle in hook_handles:
        handle.remove()
    if streaming_manager is not None:
        streaming_manager.offload_all(synchronize=True)
        unwrap_target = _peft_base_model(final_model) if train_mode == "low_rank" else final_model
        unwrapped_streaming = unwrap_streaming_offload_layers(unwrap_target)
        log.info("Unwrapped %d streaming offload layers before final save.", unwrapped_streaming)
    if train_mode == "low_rank":
        low_rank_payloads_for_export = _extract_low_rank_payloads_from_lora(
            final_model,
            selection.target_modules,
        )
        export_model, _export_meta, _export_resolved_dir = load_compressed_student_checkpoint(
            args.student_checkpoint_dir,
            access_token=hf_args.access_token,
            logger=log,
        )
        written = _write_low_rank_payloads_to_compressed_model(export_model, low_rank_payloads_for_export)
        log.info("Exported trained LoRA payloads back to compressed low-rank branches: written=%d", written)
        final_model = export_model
        if hasattr(final_model, "config"):
            final_model.config.use_cache = False
            if getattr(final_model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
                final_model.config.pad_token_id = tokenizer.pad_token_id
    else:
        unpacked = unpack_parallel_stage_decoders(final_model)
        log.info("Unpacked %d parallel stage decoder modules before final save.", unpacked)
        disabled_decode = _disable_trainable_decode_for_eval(final_model)
        log.info("Disabled trainable decode mode on %d VAELinear modules before final save/eval.", disabled_decode)
    _clear_vae_linear_cache(final_model, log, reason="Final save/eval prep")
    final_model.to("cpu")

    model_out = None
    run_meta_path = None
    ppl_eval = None
    lm_eval = None
    eval_prewarm = None
    if bool(getattr(training_args, "should_save", True)):
        model_out = os.path.join(run_output_dir, "final_model")
        tok = None
        if bool(args.save_tokenizer):
            tok = AutoTokenizer.from_pretrained(str(base_model_path), use_fast=True, token=hf_args.access_token)
        save_paths = save_model_checkpoint(
            final_model,
            model_out,
            base_model_path=str(base_model_path),
            tokenizer=tok,
            save_config=True,
            extra_meta={
                "stage": "vae_e2e_fintuning",
                "vae_train_mode": str(train_mode),
                "tune_final_norm": bool(args.tune_final_norm),
                "use_post_norm_head_linear": bool(args.use_post_norm_head_linear),
                "vae_tune_bias": bool(args.vae_tune_bias),
                "sparse_residual_prewarm": sparse_residual_prewarm,
            },
            unload_vae_original_weights=False,
        )
        run_meta = _build_run_meta(
            args=args,
            hf_args=hf_args,
            training_args=training_args,
            data_info=data_info,
            trainable_info=_selection_to_meta(selection),
            teacher_source=teacher_source,
            resolved_student_checkpoint_dir=str(resolved_student_checkpoint_dir),
            base_model_path=str(base_model_path),
            layer_device_map=hf_device_map,
            offload_mode=offload_mode,
            global_step=int(getattr(trainer.state, "global_step", 0)),
        )
        run_meta_path = os.path.join(run_output_dir, "run_meta.json")
        with open(run_meta_path, "w", encoding="utf-8") as handle:
            json.dump(run_meta, handle, ensure_ascii=False, indent=2)
        log.info("Saved final compressed model to %s", save_paths["output_dir"])
        eval_prewarm = _prewarm_final_eval_model(
            model=final_model,
            args=args,
            log=log,
        )
        ppl_eval = _eval_final_ppl(
            model=final_model,
            args=args,
            model_path=str(base_model_path),
            output_dir=run_output_dir,
            log=log,
        )
        lm_eval = _run_final_lm_eval(
            model=final_model,
            tokenizer=tokenizer,
            args=args,
            base_model_path=str(base_model_path),
            output_dir=run_output_dir,
            log=log,
        )
    else:
        log.info("Skipping final model save on this rank because should_save=false.")

    return {
        "run_output_dir": run_output_dir,
        "saved_model_dir": model_out,
        "teacher_source": teacher_source,
        "final_ppl": None if ppl_eval is None else ppl_eval["result"],
        "final_ppl_path": None if ppl_eval is None else ppl_eval["path"],
        "final_lm_eval_path": None if lm_eval is None else lm_eval["json_path"],
        "final_lm_eval_summary_path": None if lm_eval is None else lm_eval["summary_path"],
        "final_eval_prewarm": eval_prewarm,
        "sparse_residual_prewarm": sparse_residual_prewarm,
        "run_meta_path": run_meta_path,
    }
