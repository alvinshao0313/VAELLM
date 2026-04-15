import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from litebsq.autoencoder import Decoder, pack_decoders

if TYPE_CHECKING:
    from litebsq.vae_linear import VAELinear


_PREWARM_CATEGORY_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_LAYER_IDX_PATTERNS = (
    re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\."),
    re.compile(r"(?:^|\.)(?:model\.)?decoder\.layers\.(\d+)\."),
)


@dataclass(frozen=True)
class NamedVAELinearTarget:
    name: str
    base_layer: "VAELinear"


@dataclass(frozen=True)
class NamedVAELinearDecodeTarget:
    name: str
    base_layer: "VAELinear"
    target_dtype: Optional[torch.dtype] = None


@dataclass(frozen=True)
class DecodedVAELinearWeight:
    name: str
    base_layer: "VAELinear"
    decoded_weight: torch.Tensor
    target_dtype: torch.dtype
    compute_device: torch.device


@dataclass(frozen=True)
class _DecoderPackSignature:
    decoder_type: str
    in_dim: int
    out_dim: int
    hidden_dim: int
    num_res_blocks: int
    norm_type: str
    use_checkpoint: bool
    param_dtype: torch.dtype


@dataclass(frozen=True)
class _VAELinearPrewarmSignature:
    category: str
    device: str
    target_dtype: torch.dtype
    parallel_rows: int
    parallel_cols: int
    transpose: bool
    compressed_in_features: int
    compressed_out_features: int
    residual_stages: int
    stage_codebook_dims: Tuple[int, ...]
    stage_vq_shapes: Tuple[Tuple[Tuple[int, int], ...], ...]
    stage_decoder_signatures: Tuple[Tuple[_DecoderPackSignature, ...], ...]


def _extract_layer_idx(name: str) -> Optional[int]:
    for pattern in _LAYER_IDX_PATTERNS:
        match = pattern.search(str(name))
        if match:
            return int(match.group(1))
    return None


def _category_sort_key(category: str) -> Tuple[int, str]:
    value = str(category)
    try:
        return (_PREWARM_CATEGORY_ORDER.index(value), value)
    except ValueError:
        return (len(_PREWARM_CATEGORY_ORDER), value)


def _resolve_decoder_param_dtype(decoder: nn.Module) -> torch.dtype:
    for param in decoder.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def _resolve_decoder_param_device(decoder: nn.Module) -> torch.device:
    for param in decoder.parameters():
        return param.device
    for buffer in decoder.buffers():
        return buffer.device
    return torch.device("cpu")


def _normalize_named_vae_targets(named_targets: Sequence[Any]) -> List[NamedVAELinearTarget]:
    from litebsq.vae_linear import VAELinear

    out: List[NamedVAELinearTarget] = []
    for idx, item in enumerate(named_targets):
        if isinstance(item, NamedVAELinearTarget):
            target = item
        elif isinstance(item, tuple) and len(item) == 2:
            target = NamedVAELinearTarget(name=str(item[0]), base_layer=item[1])
        else:
            name = getattr(item, "name", None)
            base_layer = getattr(item, "base_layer", None)
            if name is None or base_layer is None:
                raise TypeError(
                    f"Named VAE target at idx={idx} must provide 'name' and 'base_layer', got {type(item)}."
                )
            target = NamedVAELinearTarget(name=str(name), base_layer=base_layer)
        if not isinstance(target.base_layer, VAELinear):
            raise TypeError(
                f"Named VAE target '{target.name}' must reference VAELinear, got {type(target.base_layer)}."
            )
        out.append(target)
    return out


def _normalize_named_vae_decode_targets(named_targets: Sequence[Any]) -> List[NamedVAELinearDecodeTarget]:
    from litebsq.vae_linear import VAELinear

    out: List[NamedVAELinearDecodeTarget] = []
    for idx, item in enumerate(named_targets):
        if isinstance(item, NamedVAELinearDecodeTarget):
            target = item
        elif isinstance(item, NamedVAELinearTarget):
            target = NamedVAELinearDecodeTarget(
                name=str(item.name),
                base_layer=item.base_layer,
                target_dtype=None,
            )
        elif isinstance(item, tuple) and len(item) in {2, 3}:
            target = NamedVAELinearDecodeTarget(
                name=str(item[0]),
                base_layer=item[1],
                target_dtype=None if len(item) == 2 else item[2],
            )
        else:
            name = getattr(item, "name", None)
            base_layer = getattr(item, "base_layer", None)
            target_dtype = getattr(item, "target_dtype", None)
            if name is None or base_layer is None:
                raise TypeError(
                    f"Named VAE decode target at idx={idx} must provide 'name' and 'base_layer', got {type(item)}."
                )
            target = NamedVAELinearDecodeTarget(
                name=str(name),
                base_layer=base_layer,
                target_dtype=target_dtype,
            )
        if not isinstance(target.base_layer, VAELinear):
            raise TypeError(
                f"Named VAE decode target '{target.name}' must reference VAELinear, got {type(target.base_layer)}."
            )
        out.append(target)
    return out


def _resolve_cache_dtype_for_layer(base_layer: "VAELinear", dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is not None:
        return dtype
    for param in base_layer.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def _resolve_base_layer_device(base_layer: "VAELinear") -> torch.device:
    for param in base_layer.parameters():
        return param.device
    for buffer in base_layer.buffers():
        return buffer.device
    return torch.device("cpu")


def _resolve_decoder_pack_signature(decoder: nn.Module) -> _DecoderPackSignature:
    if not isinstance(decoder, Decoder):
        raise TypeError(f"Grouped prewarm only supports Decoder stage payloads, got {type(decoder)}.")
    if int(decoder.num_models) != 1:
        raise ValueError(
            f"Grouped prewarm expects single-model Decoder payload, got num_models={decoder.num_models}."
        )
    return _DecoderPackSignature(
        decoder_type=str(decoder.decoder_type),
        in_dim=int(decoder.in_dim),
        out_dim=int(decoder.out_dim),
        hidden_dim=int(decoder.hidden_dim),
        num_res_blocks=int(decoder.num_res_blocks),
        norm_type=str(decoder.norm_type),
        use_checkpoint=bool(decoder.use_checkpoint),
        param_dtype=_resolve_decoder_param_dtype(decoder),
    )


def _build_vae_linear_prewarm_signature(
    *,
    name: str,
    base_layer: "VAELinear",
    target_dtype: torch.dtype,
) -> _VAELinearPrewarmSignature:
    stage_vq_shapes: List[Tuple[Tuple[int, int], ...]] = []
    stage_decoder_signatures: List[Tuple[_DecoderPackSignature, ...]] = []
    for stage_idx in range(int(base_layer.residual_stages)):
        one_stage_vq_shapes = []
        one_stage_decoder_sigs = []
        for part_idx in range(int(base_layer.parallel_parts)):
            vq_weight = base_layer.get_stage_part_vq_weight(stage_idx=stage_idx, part_idx=part_idx)
            if vq_weight.ndim != 3 or int(vq_weight.shape[1]) != 1:
                raise ValueError(
                    f"Grouped prewarm expects vq_weight shape [N_blocks, 1, latent_dim], "
                    f"got {tuple(vq_weight.shape)} for '{name}' stage={stage_idx} part={part_idx}."
                )
            one_stage_vq_shapes.append((int(vq_weight.shape[0]), int(vq_weight.shape[-1])))
            one_stage_decoder_sigs.append(
                _resolve_decoder_pack_signature(base_layer.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx))
            )
        stage_vq_shapes.append(tuple(one_stage_vq_shapes))
        stage_decoder_signatures.append(tuple(one_stage_decoder_sigs))
    return _VAELinearPrewarmSignature(
        category=str(name).rsplit(".", 1)[-1],
        device=str(_resolve_base_layer_device(base_layer)),
        target_dtype=target_dtype,
        parallel_rows=int(base_layer.parallel_rows),
        parallel_cols=int(base_layer.parallel_cols),
        transpose=bool(base_layer.transpose),
        compressed_in_features=int(base_layer.compressed_in_features),
        compressed_out_features=int(base_layer.compressed_out_features),
        residual_stages=int(base_layer.residual_stages),
        stage_codebook_dims=tuple(int(dim) for dim in base_layer.stage_codebook_dims),
        stage_vq_shapes=tuple(stage_vq_shapes),
        stage_decoder_signatures=tuple(stage_decoder_signatures),
    )


def _named_target_sort_key(target: NamedVAELinearTarget) -> Tuple[int, str]:
    layer_idx = _extract_layer_idx(target.name)
    if layer_idx is None:
        return (10**9, str(target.name))
    return (int(layer_idx), str(target.name))


@torch.no_grad()
def _decode_named_vae_linear_weights_chunk(
    chunk_targets: Sequence[NamedVAELinearDecodeTarget],
    *,
    compute_device: Optional[torch.device],
) -> Tuple[List[DecodedVAELinearWeight], int]:
    if not chunk_targets:
        raise ValueError("_decode_named_vae_linear_weights_chunk expects a non-empty chunk.")

    first = chunk_targets[0].base_layer
    parts_per_linear = int(first.parallel_parts)
    residual_stages = int(first.residual_stages)
    chunk_num_models = int(len(chunk_targets)) * parts_per_linear
    resolved_compute_device: Optional[torch.device] = None
    target_dtypes = [
        _resolve_cache_dtype_for_layer(target.base_layer, target.target_dtype)
        for target in chunk_targets
    ]
    accumulated_compressed_weights: List[Optional[torch.Tensor]] = [None for _ in chunk_targets]

    for stage_idx in range(residual_stages):
        stage_decoders: List[Decoder] = []
        stage_vq_weights: List[torch.Tensor] = []
        for target in chunk_targets:
            base_layer = target.base_layer
            for part_idx in range(parts_per_linear):
                stage_decoders.append(base_layer.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx))
                stage_vq_weights.append(base_layer.get_stage_part_vq_weight(stage_idx=stage_idx, part_idx=part_idx))

        grouped_decoder = pack_decoders(stage_decoders)
        grouped_vq = torch.cat(stage_vq_weights, dim=1).contiguous()
        decode_dtype = _resolve_decoder_param_dtype(grouped_decoder)
        decode_device = _resolve_decoder_param_device(grouped_decoder) if compute_device is None else torch.device(compute_device)
        grouped_decoder = grouped_decoder.to(device=decode_device, dtype=decode_dtype)
        stage_out = grouped_decoder(grouped_vq.to(device=decode_device, dtype=decode_dtype, non_blocking=True))
        stage_flat = stage_out.permute(1, 0, 2).contiguous().view(chunk_num_models, -1)
        for linear_idx, target in enumerate(chunk_targets):
            base_layer = target.base_layer
            start = linear_idx * parts_per_linear
            end = start + parts_per_linear
            part_flats = stage_flat[start:end]
            stage_compressed_weight = base_layer._decode_compressed_weight_from_part_flats(
                part_flats,
                dtype=target_dtypes[linear_idx],
                stage_idx=stage_idx,
            )
            previous_weight = accumulated_compressed_weights[linear_idx]
            if previous_weight is None:
                accumulated_compressed_weights[linear_idx] = stage_compressed_weight
            else:
                if tuple(previous_weight.shape) != tuple(stage_compressed_weight.shape):
                    raise ValueError(
                        f"Grouped prewarm compressed weight shape mismatch for '{target.name}': "
                        f"prev={tuple(previous_weight.shape)} vs stage={tuple(stage_compressed_weight.shape)}."
                    )
                accumulated_compressed_weights[linear_idx] = previous_weight + stage_compressed_weight
        del grouped_decoder, grouped_vq, stage_out, stage_flat
        resolved_compute_device = decode_device

    if resolved_compute_device is None:
        resolved_compute_device = torch.device("cpu")

    decoded_results: List[DecodedVAELinearWeight] = []
    for linear_idx, target in enumerate(chunk_targets):
        base_layer = target.base_layer
        compressed_weight = accumulated_compressed_weights[linear_idx]
        if compressed_weight is None:
            raise RuntimeError(f"Grouped prewarm produced no compressed weight for '{target.name}'.")
        target_dtype = target_dtypes[linear_idx]
        decoded_weight = base_layer._finalize_decoded_weight_from_compressed(
            compressed_weight,
            dtype=target_dtype,
        ).detach()
        decoded_results.append(
            DecodedVAELinearWeight(
                name=str(target.name),
                base_layer=base_layer,
                decoded_weight=decoded_weight,
                target_dtype=target_dtype,
                compute_device=resolved_compute_device,
            )
        )
    return decoded_results, chunk_num_models


def resolve_grouped_decode_compute_device(
    requested_device: Optional[Any],
    *,
    logger: Optional[logging.Logger] = None,
    log_prefix: str = "",
) -> torch.device:
    if requested_device is None:
        return torch.device("cpu")
    try:
        device = torch.device(str(requested_device))
    except Exception as exc:
        raise ValueError(f"{log_prefix}Invalid grouped decode device: {requested_device!r}.") from exc
    if device.type != "cuda":
        if device.type != "cpu":
            raise ValueError(f"{log_prefix}Grouped decode only supports cuda/cpu, got {device!s}.")
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(f"{log_prefix}Requested grouped decode device {device!s}, but CUDA is unavailable.")
    if device.index is not None and (device.index < 0 or device.index >= torch.cuda.device_count()):
        raise ValueError(
            f"{log_prefix}Requested grouped decode device {device!s} is out of range for "
            f"{torch.cuda.device_count()} visible CUDA devices."
        )
    return device


def _should_skip_decode_target(
    target: NamedVAELinearDecodeTarget,
    *,
    respect_cache_policy: bool,
) -> bool:
    if not respect_cache_policy:
        return False
    base_layer = target.base_layer
    use_original = bool(getattr(base_layer, "always_use_original", False)) or not bool(
        getattr(base_layer, "temporary", True)
    )
    if bool(getattr(base_layer, "_skip_global_cache_prewarm", False)):
        return True
    if not bool(getattr(base_layer, "cache_decoded_weight", True)):
        return True
    if use_original:
        return True
    return False


@torch.no_grad()
def decode_named_vae_linear_weights(
    named_targets: Sequence[Any],
    dtype: Optional[torch.dtype] = None,
    group_size: int = 8,
    compute_device: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    respect_cache_policy: bool = True,
) -> List[DecodedVAELinearWeight]:
    normalized_targets = _normalize_named_vae_decode_targets(named_targets)
    group_size = int(group_size)
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}.")

    grouped: Dict[str, Dict[_VAELinearPrewarmSignature, List[NamedVAELinearDecodeTarget]]] = {}
    requested_compute_device = None
    if compute_device is not None:
        requested_compute_device = resolve_grouped_decode_compute_device(
            compute_device,
            logger=logger,
        )
    for target in normalized_targets:
        if _should_skip_decode_target(target, respect_cache_policy=respect_cache_policy):
            continue
        target_dtype = _resolve_cache_dtype_for_layer(target.base_layer, target.target_dtype or dtype)
        resolved_target = NamedVAELinearDecodeTarget(
            name=str(target.name),
            base_layer=target.base_layer,
            target_dtype=target_dtype,
        )
        signature = _build_vae_linear_prewarm_signature(
            name=resolved_target.name,
            base_layer=resolved_target.base_layer,
            target_dtype=target_dtype,
        )
        grouped.setdefault(signature.category, {}).setdefault(signature, []).append(resolved_target)

    decoded_results: List[DecodedVAELinearWeight] = []
    ordered_categories = sorted(grouped.keys(), key=_category_sort_key)
    for category in ordered_categories:
        category_chunk_index = 0
        signature_groups = []
        for signature, targets in grouped[category].items():
            ordered_targets = sorted(
                targets,
                key=lambda item: _named_target_sort_key(NamedVAELinearTarget(name=item.name, base_layer=item.base_layer)),
            )
            signature_groups.append((signature, ordered_targets))
        signature_groups.sort(key=lambda item: _named_target_sort_key(NamedVAELinearTarget(name=item[1][0].name, base_layer=item[1][0].base_layer)))

        for _signature, targets in signature_groups:
            for start in range(0, len(targets), group_size):
                chunk = targets[start:start + group_size]
                category_chunk_index += 1
                chunk_start_time = time.time()
                try:
                    chunk_results, chunk_num_models = _decode_named_vae_linear_weights_chunk(
                        chunk,
                        compute_device=requested_compute_device,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Grouped VAELinear decode failed for category={category}, chunk_index={category_chunk_index}, "
                        f"chunk_linears={len(chunk)}, parts_per_linear={int(chunk[0].base_layer.parallel_parts)}: {exc}"
                    ) from exc
                decoded_results.extend(chunk_results)
                if logger is not None:
                    logger.info(
                        "VAELinear decode chunk: category=%s chunk_index=%d chunk_linears=%d parts_per_linear=%d "
                        "chunk_num_models=%d decoded=%d duration_sec=%.2f compute_device=%s",
                        category,
                        category_chunk_index,
                        len(chunk),
                        int(chunk[0].base_layer.parallel_parts),
                        int(chunk_num_models),
                        len(chunk_results),
                        float(time.time() - chunk_start_time),
                        str(chunk_results[0].compute_device if chunk_results else requested_compute_device),
                    )
    return decoded_results


def clear_model_vae_linear_cache(model: nn.Module) -> int:
    from litebsq.vae_linear import VAELinear

    cleared = 0
    for module in model.modules():
        if isinstance(module, VAELinear):
            module.clear_decoded_weight_cache()
            cleared += 1
    return cleared


@torch.no_grad()
def _prime_named_vae_linear_cache_individually(
    normalized_targets: Sequence[NamedVAELinearTarget],
    *,
    dtype: Optional[torch.dtype],
    clear_existing: bool,
) -> Dict[str, int]:
    total = 0
    warmed = 0
    skipped = 0
    failed = 0

    for target in normalized_targets:
        base_layer = target.base_layer
        total += 1
        if clear_existing:
            base_layer.clear_decoded_weight_cache()
        use_original = bool(getattr(base_layer, "always_use_original", False)) or not bool(
            getattr(base_layer, "temporary", True)
        )
        if bool(getattr(base_layer, "_skip_global_cache_prewarm", False)):
            skipped += 1
            continue
        if not bool(getattr(base_layer, "cache_decoded_weight", True)):
            skipped += 1
            continue
        if use_original:
            skipped += 1
            continue
        target_dtype = _resolve_cache_dtype_for_layer(base_layer, dtype=dtype)
        try:
            did_warm = bool(base_layer.prime_decoded_weight_cache(dtype=target_dtype))
        except Exception as exc:
            failed += 1
            raise RuntimeError(f"VAELinear prewarm failed for '{target.name}': {exc}") from exc
        if did_warm:
            warmed += 1
        else:
            skipped += 1

    return {
        "total": int(total),
        "warmed": int(warmed),
        "skipped": int(skipped),
        "failed": int(failed),
    }


@torch.no_grad()
def prime_named_vae_linear_cache(
    named_targets: Sequence[Any],
    dtype: Optional[torch.dtype] = None,
    clear_existing: bool = False,
    group_size: int = 8,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, int]:
    normalized_targets = _normalize_named_vae_targets(named_targets)
    group_size = int(group_size)
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}.")
    if group_size == 1:
        return _prime_named_vae_linear_cache_individually(
            normalized_targets,
            dtype=dtype,
            clear_existing=clear_existing,
        )
    total = len(normalized_targets)
    if clear_existing:
        for target in normalized_targets:
            target.base_layer.clear_decoded_weight_cache()
    try:
        decoded_results = decode_named_vae_linear_weights(
            normalized_targets,
            dtype=dtype,
            group_size=group_size,
            compute_device=None,
            logger=logger,
            respect_cache_policy=True,
        )
    except Exception as exc:
        raise RuntimeError(f"VAELinear prewarm failed: {exc}") from exc
    for result in decoded_results:
        result.base_layer._cached_weight = result.decoded_weight
    warmed = len(decoded_results)
    return {
        "total": int(total),
        "warmed": int(warmed),
        "skipped": int(total - warmed),
        "failed": 0,
    }


@torch.no_grad()
def prime_model_vae_linear_cache(
    model: nn.Module,
    dtype: Optional[torch.dtype] = None,
    clear_existing: bool = False,
    group_size: int = 1,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, int]:
    from litebsq.vae_linear import VAELinear

    named_targets = [
        NamedVAELinearTarget(name=str(name), base_layer=module)
        for name, module in model.named_modules()
        if isinstance(module, VAELinear)
    ]
    return prime_named_vae_linear_cache(
        named_targets,
        dtype=dtype,
        clear_existing=clear_existing,
        group_size=group_size,
        logger=logger,
    )


__all__ = [
    "DecodedVAELinearWeight",
    "NamedVAELinearTarget",
    "NamedVAELinearDecodeTarget",
    "clear_model_vae_linear_cache",
    "decode_named_vae_linear_weights",
    "prime_model_vae_linear_cache",
    "prime_named_vae_linear_cache",
    "resolve_grouped_decode_compute_device",
]
