import gc
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from peft import PeftModel
from peft.tuners.adalora.config import AdaLoraConfig
from peft.tuners.adalora.layer import SVDLinear as DenseAdaLoraLinear
from peft.tuners.lora.layer import Linear as DenseLoraLinear
from peft.utils.other import ModulesToSaveWrapper
from torch import nn

from dense_e2e_fintuning.trainables import inject_dense_peft_adapters, resolve_target_layer_ids
from e2e_common.checkpoint_io import load_e2e_model_checkpoint, save_e2e_model_checkpoint
from e2e_common.peft_proxy import (
    PeftVAELinearProxy,
    _ADALORA_RANKALLOCATOR_ATTR,
    ensure_peft_vae_proxy_adapter,
    is_peft_adalora_linear,
    is_peft_lora_linear,
)
from e2e_common.post_norm_head import ensure_post_norm_head_linear
from e2e_common.proxy_trainables import select_e2e_trainables_peft_proxy
from litebsq.misc import set_module_by_name
from litebsq.vae_linear import VAELinear
from litebsq.vae_linear_prewarm import NamedVAELinearDecodeTarget, decode_named_vae_linear_weights
from rotation.model_utils import get_layers
from train_utils.model_checkpoint_io import META_FILENAME, resolve_checkpoint_dir


_DEFAULT_ADAPTER_NAME = "default"
_E2E_FINETUNE_MODE = "vae_lora"
_DENSE_MODEL_PREFIX = "base_model.model."


def _resolve_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    module = model
    if not module_name:
        return module
    for token in str(module_name).split("."):
        if not hasattr(module, token):
            raise ValueError(f"Failed to resolve module '{module_name}': missing '{token}'.")
        module = getattr(module, token)
    if not isinstance(module, nn.Module):
        raise TypeError(f"Resolved object at '{module_name}' is not an nn.Module: {type(module)}")
    return module


def _resolve_reference_dtype(module: nn.Module) -> torch.dtype:
    for param in module.parameters():
        if param.is_floating_point():
            return param.dtype
    for buffer in module.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return torch.float32


def _resolve_model_peft_config(model: nn.Module):
    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, dict):
        if _DEFAULT_ADAPTER_NAME in peft_config:
            return peft_config[_DEFAULT_ADAPTER_NAME]
        if len(peft_config) == 1:
            return next(iter(peft_config.values()))
    return peft_config


def _resolve_default_adapter_name(module) -> str:
    for attr_name in ("lora_A", "lora_E", "ranknum", "modules_to_save"):
        mapping = getattr(module, attr_name, None)
        if mapping is None:
            continue
        if _DEFAULT_ADAPTER_NAME in mapping:
            return _DEFAULT_ADAPTER_NAME
        if len(mapping) == 1:
            return next(iter(mapping.keys()))
    raise ValueError(f"Failed to resolve adapter name from module {type(module)}.")


def _iter_named_vae_linears(model: nn.Module) -> Iterable[Tuple[str, VAELinear]]:
    for name, module in model.named_modules():
        if isinstance(module, VAELinear):
            yield str(name), module


def load_checkpoint_meta(student_checkpoint_dir: str) -> Tuple[str, Dict[str, object]]:
    resolved_dir = resolve_checkpoint_dir(student_checkpoint_dir)
    meta_path = os.path.join(resolved_dir, META_FILENAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing checkpoint meta: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise TypeError(f"Checkpoint meta must be a dict, got {type(meta)}")
    return resolved_dir, meta


def checkpoint_has_adapters(meta: Dict[str, object]) -> bool:
    adapter_count = meta.get("adapter_module_count")
    if adapter_count is not None and int(adapter_count) > 0:
        return True
    adapter_modules = meta.get("adapter_modules", [])
    return isinstance(adapter_modules, list) and len(adapter_modules) > 0


def reject_checkpoint_with_adapters(meta: Dict[str, object]) -> None:
    if checkpoint_has_adapters(meta):
        raise ValueError("dense_e2e_fintuning 首版只接受不带 adapter 的压缩 checkpoint。")


def resolve_base_model_path(meta: Dict[str, object], teacher_model_path: Optional[str] = None) -> str:
    explicit_path = None if teacher_model_path is None else str(teacher_model_path).strip()
    if explicit_path:
        return explicit_path
    base_model_path = meta.get("base_model_path")
    if base_model_path:
        return str(base_model_path)
    raise ValueError("Cannot resolve base model path from checkpoint meta or --teacher_model_path.")


def resolve_decode_device(requested: Optional[str]) -> str:
    normalized = "auto" if requested is None else str(requested).strip().lower()
    if normalized == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("decode_device=cuda requested, but CUDA is unavailable.")
        return "cuda:0"
    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"decode_device={normalized} requested, but CUDA is unavailable.")
        try:
            device_idx = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid decode device '{requested}'.") from exc
        if device_idx < 0 or device_idx >= torch.cuda.device_count():
            raise ValueError(
                f"decode_device={normalized} is out of range for visible CUDA device count={torch.cuda.device_count()}."
            )
        return f"cuda:{device_idx}"
    raise ValueError(f"Invalid decode device '{requested}'.")


def load_compressed_student_checkpoint(
    student_checkpoint_dir: str,
    *,
    access_token: Optional[str] = None,
    base_model_path: Optional[str] = None,
    logger=None,
) -> Tuple[nn.Module, Dict[str, object], str]:
    resolved_dir, meta = load_checkpoint_meta(student_checkpoint_dir)
    reject_checkpoint_with_adapters(meta)
    model, loaded_meta, _load_result = load_e2e_model_checkpoint(
        resolved_dir,
        access_token=access_token,
        base_model_path=base_model_path,
        map_location="cpu",
        strict=True,
        materialize_proxy_decoded_linears=False,
        proxy_logger=logger,
    )
    return model, loaded_meta, resolved_dir


@torch.no_grad()
def materialize_vae_linears_to_dense(
    model: nn.Module,
    *,
    group_size: int = 8,
    compute_device: Optional[object] = "cpu",
    logger=None,
) -> int:
    vae_refs = list(_iter_named_vae_linears(model))
    if not vae_refs:
        return 0

    decode_targets = [
        NamedVAELinearDecodeTarget(
            name=name,
            base_layer=module,
            target_dtype=_resolve_reference_dtype(module),
        )
        for name, module in vae_refs
    ]
    if logger is not None:
        logger.info(
            "Start dense rebuild from VAELinear: total=%d group_size=%d compute_device=%s",
            len(decode_targets),
            int(group_size),
            str(compute_device),
        )
    decoded_results = decode_named_vae_linear_weights(
        decode_targets,
        group_size=int(group_size),
        compute_device=compute_device,
        logger=logger,
        respect_cache_policy=False,
    )
    decoded_by_name = {item.name: item for item in decoded_results}
    if len(decoded_by_name) != len(vae_refs):
        raise RuntimeError(
            f"Dense rebuild decode count mismatch: decoded={len(decoded_by_name)} expected={len(vae_refs)}."
        )

    converted = 0
    for name, old_module in vae_refs:
        decoded = decoded_by_name[name]
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
        set_module_by_name(model, name, dense_linear)
        converted += 1

    gc.collect()
    if logger is not None:
        logger.info("Finished dense rebuild from VAELinear: converted=%d", converted)
    return converted


def build_dense_model_from_checkpoint(
    student_checkpoint_dir: str,
    *,
    access_token: Optional[str] = None,
    base_model_path: Optional[str] = None,
    logger=None,
    decode_group_size: int = 8,
    decode_device: str = "auto",
) -> Tuple[nn.Module, Dict[str, object], str]:
    resolved_decode_device = resolve_decode_device(decode_device)
    if logger is not None:
        logger.info(
            "Dense rebuild decode config: requested_device=%s resolved_device=%s group_size=%d",
            str(decode_device),
            resolved_decode_device,
            int(decode_group_size),
        )
    model, meta, resolved_dir = load_compressed_student_checkpoint(
        student_checkpoint_dir,
        access_token=access_token,
        base_model_path=base_model_path,
        logger=logger,
    )
    converted = materialize_vae_linears_to_dense(
        model,
        group_size=int(decode_group_size),
        compute_device=resolved_decode_device,
        logger=logger,
    )
    if logger is not None:
        logger.info("Dense student is ready: source=%s converted_modules=%d", resolved_dir, converted)
    return model, meta, resolved_dir


def _unwrap_dense_peft_base_model(model: nn.Module) -> nn.Module:
    if isinstance(model, PeftModel):
        return model.get_base_model()
    return model


def _resolve_dense_source_module(model: nn.Module, module_name: str) -> nn.Module:
    return _resolve_module_by_name(_unwrap_dense_peft_base_model(model), module_name)


def _resolve_proxy_module(model: nn.Module, module_name: str) -> PeftVAELinearProxy:
    module = _resolve_module_by_name(model, module_name)
    if not isinstance(module, PeftVAELinearProxy):
        raise TypeError(f"Expected PeftVAELinearProxy at '{module_name}', got {type(module)}")
    return module


def _copy_bias_if_present(source_bias: Optional[torch.Tensor], target_bias: Optional[torch.Tensor]) -> None:
    if target_bias is None:
        if source_bias is not None:
            raise ValueError("Target bias is missing while source bias exists.")
        return
    if source_bias is None:
        target_bias.zero_()
        return
    target_bias.copy_(source_bias.detach().to(device=target_bias.device, dtype=target_bias.dtype))


@torch.no_grad()
def _copy_lora_linear_state(source_module: DenseLoraLinear, target_module) -> None:
    source_adapter = _resolve_default_adapter_name(source_module)
    target_adapter = _resolve_default_adapter_name(target_module)
    target_module.lora_A[target_adapter].weight.copy_(
        source_module.lora_A[source_adapter].weight.detach().to(
            device=target_module.lora_A[target_adapter].weight.device,
            dtype=target_module.lora_A[target_adapter].weight.dtype,
        )
    )
    target_module.lora_B[target_adapter].weight.copy_(
        source_module.lora_B[source_adapter].weight.detach().to(
            device=target_module.lora_B[target_adapter].weight.device,
            dtype=target_module.lora_B[target_adapter].weight.dtype,
        )
    )
    if bool(source_module.use_dora.get(source_adapter, False)):
        target_module.lora_magnitude_vector[target_adapter].copy_(
            source_module.lora_magnitude_vector[source_adapter].detach().to(
                device=target_module.lora_magnitude_vector[target_adapter].device,
                dtype=target_module.lora_magnitude_vector[target_adapter].dtype,
            )
        )
    _copy_bias_if_present(source_module.base_layer.bias, target_module.get_base_layer().bias)


@torch.no_grad()
def _copy_adalora_linear_state(source_module: DenseAdaLoraLinear, target_module) -> None:
    source_adapter = _resolve_default_adapter_name(source_module)
    target_adapter = _resolve_default_adapter_name(target_module)
    target_module.lora_A[target_adapter].copy_(
        source_module.lora_A[source_adapter].detach().to(
            device=target_module.lora_A[target_adapter].device,
            dtype=target_module.lora_A[target_adapter].dtype,
        )
    )
    target_module.lora_B[target_adapter].copy_(
        source_module.lora_B[source_adapter].detach().to(
            device=target_module.lora_B[target_adapter].device,
            dtype=target_module.lora_B[target_adapter].dtype,
        )
    )
    target_module.lora_E[target_adapter].copy_(
        source_module.lora_E[source_adapter].detach().to(
            device=target_module.lora_E[target_adapter].device,
            dtype=target_module.lora_E[target_adapter].dtype,
        )
    )
    target_module.ranknum[target_adapter].copy_(
        source_module.ranknum[source_adapter].detach().to(
            device=target_module.ranknum[target_adapter].device,
            dtype=target_module.ranknum[target_adapter].dtype,
        )
    )
    _copy_bias_if_present(source_module.base_layer.bias, target_module.get_base_layer().bias)


def _resolve_modules_to_save_source(module: nn.Module) -> nn.Module:
    if not isinstance(module, ModulesToSaveWrapper):
        return module
    adapter_name = _resolve_default_adapter_name(module)
    return module.modules_to_save[adapter_name]


@torch.no_grad()
def _copy_modules_to_save_state(
    dense_model: nn.Module,
    compact_model: nn.Module,
    module_names: Sequence[str],
) -> int:
    copied = 0
    for module_name in module_names:
        source_module = _resolve_modules_to_save_source(_resolve_dense_source_module(dense_model, module_name))
        target_module = _resolve_module_by_name(compact_model, module_name)
        target_module.load_state_dict(source_module.state_dict(), strict=True)
        copied += 1
    return copied


def _resolve_export_modules_to_save(selection) -> List[str]:
    modules = []
    modules.extend(getattr(selection, "modules_to_save", []) or [])
    modules.extend(getattr(selection, "final_norm_modules", []) or [])
    modules.extend(getattr(selection, "post_norm_head_modules", []) or [])
    return sorted(set(str(name) for name in modules if str(name)))


def _map_dense_runtime_name_to_proxy(name: str) -> str:
    mapped = str(name)
    if mapped.startswith(_DENSE_MODEL_PREFIX):
        mapped = mapped[len(_DENSE_MODEL_PREFIX):]
    markers = (".lora_A.", ".lora_B.", ".lora_E.", ".ranknum.", ".base_layer.")
    for marker in markers:
        if marker in mapped:
            prefix, suffix = mapped.split(marker, 1)
            return f"{prefix}.per_decoded_linear{marker}{suffix}"
    return mapped


def _transfer_adalora_runtime(dense_model: nn.Module, compact_model: nn.Module) -> None:
    dense_config = _resolve_model_peft_config(dense_model)
    if not isinstance(dense_config, AdaLoraConfig):
        return

    dense_base_model = getattr(dense_model, "base_model", None)
    source_allocator = getattr(dense_base_model, "rankallocator", None)
    target_allocator = getattr(compact_model, _ADALORA_RANKALLOCATOR_ATTR, None)
    if source_allocator is None or target_allocator is None:
        return

    named_params = dict(compact_model.named_parameters())
    for group_name in ("ipt", "exp_avg_ipt", "exp_avg_unc"):
        source_group = getattr(source_allocator, group_name, {})
        restored_group = {}
        for dense_name, tensor in source_group.items():
            mapped_name = _map_dense_runtime_name_to_proxy(str(dense_name))
            if mapped_name not in named_params:
                raise ValueError(f"Missing mapped AdaLoRA runtime parameter '{mapped_name}' in compact model.")
            ref_param = named_params[mapped_name]
            restored_group[mapped_name] = tensor.detach().to(device=ref_param.device, dtype=ref_param.dtype)
        setattr(target_allocator, group_name, restored_group)

    rank_pattern = getattr(dense_config, "rank_pattern", None)
    target_config = _resolve_model_peft_config(compact_model)
    if isinstance(rank_pattern, dict) and isinstance(target_config, AdaLoraConfig):
        target_config.rank_pattern = {
            _map_dense_runtime_name_to_proxy(str(name)): list(value)
            for name, value in rank_pattern.items()
        }


def _resolve_lora_bias_mode(args) -> str:
    return "lora_only" if bool(getattr(args, "lora_tune_bias", False)) else "none"


def prepare_compact_model_for_export(
    student_checkpoint_dir: str,
    *,
    access_token: Optional[str],
    args,
    training_args,
    decode_group_size: int = 8,
    decode_device: str = "auto",
    logger=None,
) -> Tuple[nn.Module, Dict[str, object], object]:
    resolved_decode_device = resolve_decode_device(decode_device)
    if logger is not None:
        logger.info(
            "Compact export decode config: requested_device=%s resolved_device=%s group_size=%d",
            str(decode_device),
            resolved_decode_device,
            int(decode_group_size),
        )
    compact_model, meta, _resolved_dir = load_compressed_student_checkpoint(
        student_checkpoint_dir,
        access_token=access_token,
        logger=logger,
    )
    if bool(getattr(args, "use_post_norm_head_linear", False)):
        ensure_post_norm_head_linear(compact_model)

    layers = list(get_layers(compact_model))
    decoder_layer_ids = resolve_target_layer_ids(getattr(args, "decoder_layer_ids", None), len(layers))
    selection = select_e2e_trainables_peft_proxy(
        compact_model,
        decoder_layer_ids=decoder_layer_ids,
        target_module_names=getattr(args, "target_module_names", None),
        tune_final_norm=bool(getattr(args, "tune_final_norm", False)),
        use_post_norm_head_linear=bool(getattr(args, "use_post_norm_head_linear", False)),
    )
    if selection.peft_proxy_modules:
        ensure_peft_vae_proxy_adapter(
            compact_model,
            variant=str(getattr(args, "lora_variant")).strip().lower(),
            rank=int(getattr(args, "lora_rank")),
            alpha=float(getattr(args, "lora_alpha")),
            dropout=float(getattr(args, "lora_dropout")),
            init_mode="zero",
            total_step=int(training_args.max_steps) if str(getattr(args, "lora_variant")).strip().lower() == "adalora" else None,
            adalora_target_r=int(getattr(args, "adalora_target_r")),
            adalora_init_r=int(getattr(args, "adalora_init_r")),
            adalora_tinit=int(getattr(args, "adalora_tinit")),
            adalora_tfinal=int(getattr(args, "adalora_tfinal")),
            adalora_delta_t=int(getattr(args, "adalora_delta_t")),
            adalora_beta1=float(getattr(args, "adalora_beta1")),
            adalora_beta2=float(getattr(args, "adalora_beta2")),
            adalora_orth_reg_weight=float(getattr(args, "adalora_orth_reg_weight")),
            bias_mode=_resolve_lora_bias_mode(args),
            materialize_before_inject=True,
            materialize_group_size=int(decode_group_size),
            materialize_compute_device=resolved_decode_device,
            materialize_logger=logger,
        )
    setattr(compact_model, "_e2e_vae_lora_tune_bias", bool(getattr(args, "lora_tune_bias", False)))
    return compact_model, meta, selection


@torch.no_grad()
def export_dense_peft_to_compact_checkpoint(
    dense_model: nn.Module,
    *,
    student_checkpoint_dir: str,
    access_token: Optional[str],
    output_dir: str,
    args,
    training_args,
    base_model_path: str,
    tokenizer=None,
    save_tokenizer: bool = False,
    extra_meta: Optional[Dict[str, object]] = None,
    decode_group_size: int = 8,
    decode_device: str = "auto",
    logger=None,
) -> Dict[str, str]:
    compact_model, _meta, selection = prepare_compact_model_for_export(
        student_checkpoint_dir,
        access_token=access_token,
        args=args,
        training_args=training_args,
        decode_group_size=int(decode_group_size),
        decode_device=decode_device,
        logger=logger,
    )

    for module_name in selection.target_modules:
        source_module = _resolve_dense_source_module(dense_model, module_name)
        target_proxy = _resolve_proxy_module(compact_model, module_name)
        target_module = target_proxy.per_decoded_linear
        if isinstance(source_module, DenseAdaLoraLinear):
            if not is_peft_adalora_linear(target_module):
                raise TypeError(f"Expected AdaLoRA proxy under '{module_name}', got {type(target_module)}")
            _copy_adalora_linear_state(source_module, target_module)
            continue
        if isinstance(source_module, DenseLoraLinear):
            if not is_peft_lora_linear(target_module):
                raise TypeError(f"Expected LoRA proxy under '{module_name}', got {type(target_module)}")
            _copy_lora_linear_state(source_module, target_module)
            continue
        raise TypeError(f"Expected PEFT linear module at '{module_name}', got {type(source_module)}")

    _copy_modules_to_save_state(
        dense_model,
        compact_model,
        _resolve_export_modules_to_save(selection),
    )
    _transfer_adalora_runtime(dense_model, compact_model)

    setattr(compact_model, "_e2e_finetune_mode", _E2E_FINETUNE_MODE)
    setattr(compact_model, "_e2e_vae_lora_tune_bias", bool(getattr(args, "lora_tune_bias", False)))
    compact_model.eval()
    return save_e2e_model_checkpoint(
        compact_model,
        output_dir,
        base_model_path=str(base_model_path),
        tokenizer=tokenizer if bool(save_tokenizer) else None,
        save_config=True,
        extra_meta=extra_meta,
        compact_unload_vae_original_weights=True,
    )


def rebuild_dense_peft_model_for_export(
    student_checkpoint_dir: str,
    *,
    access_token: Optional[str],
    args,
    training_args,
    state_dict: Dict[str, torch.Tensor],
    decode_group_size: int = 8,
    decode_device: str = "auto",
    logger=None,
) -> Tuple[nn.Module, Dict[str, object], object]:
    dense_model, meta, _resolved_dir = build_dense_model_from_checkpoint(
        student_checkpoint_dir,
        access_token=access_token,
        logger=logger,
        decode_group_size=int(decode_group_size),
        decode_device=decode_device,
    )
    if hasattr(dense_model, "config"):
        dense_model.config.use_cache = False
    if hasattr(dense_model, "enable_input_require_grads"):
        dense_model.enable_input_require_grads()
    if bool(getattr(args, "use_post_norm_head_linear", False)):
        ensure_post_norm_head_linear(dense_model)

    layers = list(get_layers(dense_model))
    decoder_layer_ids = resolve_target_layer_ids(getattr(args, "decoder_layer_ids", None), len(layers))
    peft_model, selection = inject_dense_peft_adapters(
        dense_model,
        args=args,
        decoder_layer_ids=decoder_layer_ids,
        total_step=int(training_args.max_steps),
    )
    load_result = peft_model.load_state_dict(state_dict, strict=True)
    if getattr(load_result, "missing_keys", None) or getattr(load_result, "unexpected_keys", None):
        raise RuntimeError(
            f"Failed to rebuild dense export model from state_dict: "
            f"missing={getattr(load_result, 'missing_keys', [])} "
            f"unexpected={getattr(load_result, 'unexpected_keys', [])}"
        )
    return peft_model, meta, selection


@torch.no_grad()
def compare_dense_and_compressed_logits(
    student_checkpoint_dir: str,
    *,
    access_token: Optional[str] = None,
    seq_len: int = 4,
    seed: int = 0,
    decode_group_size: int = 8,
    decode_device: str = "cpu",
    logger=None,
) -> Dict[str, float]:
    compressed_model, _meta, _resolved_dir = load_compressed_student_checkpoint(
        student_checkpoint_dir,
        access_token=access_token,
        logger=logger,
    )
    compressed_model.eval()
    if hasattr(compressed_model, "config"):
        compressed_model.config.use_cache = False
    torch.manual_seed(int(seed))
    vocab_size = int(getattr(compressed_model.config, "vocab_size"))
    input_ids = torch.randint(0, vocab_size, (1, int(seq_len)), device=torch.device("cpu"))
    with torch.no_grad():
        reference_logits = compressed_model(input_ids=input_ids).logits.detach().to(dtype=torch.float32)
    converted = materialize_vae_linears_to_dense(
        compressed_model,
        group_size=int(decode_group_size),
        compute_device=resolve_decode_device(decode_device),
        logger=logger,
    )
    compressed_model.eval()
    with torch.no_grad():
        dense_logits = compressed_model(input_ids=input_ids).logits.detach().to(dtype=torch.float32)
    diff = (dense_logits - reference_logits).abs()
    return {
        "converted_modules": float(converted),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }
