"""Central decoder execution planning for CAT/E2E/mid-eval.

Deletes the old pattern of forcing ``parallel_stage_decode=True`` /
``decode_group_size=8`` / ``decode_device=auto`` as hidden constants.
Call sites must consume plans from this module instead of guessing bools.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import torch
from torch import nn

from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear


DECODER_EXECUTION_MODES = ("trainable_decoder", "sparse_bit", "decoder_sparse_bit")
DEFAULT_DECODE_GROUP_SIZE = 8
DECODE_GROUP_SIZE_FALLBACK_SEQUENCE = (8, 4, 2, 1)

T = TypeVar("T")


class DecodeCapacityError(RuntimeError):
    """Explicit capacity failure that is safe to retry with a smaller group size."""


@dataclass(frozen=True)
class DecoderExecutionPlan:
    mode: str
    use_packed: bool
    reason: str
    decoder_count: int
    pack_compatible: bool
    incompatibility_reason: Optional[str] = None


@dataclass(frozen=True)
class ResolvedDecodeGroupSize:
    group_size: int
    fallback_reason: str
    attempted: Tuple[int, ...]
    num_targets: int


def resolve_module_execution_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def bucket_modules_by_execution_device(
    named_modules: Sequence[Tuple[str, nn.Module]],
) -> Dict[torch.device, List[Tuple[str, nn.Module]]]:
    buckets: Dict[torch.device, List[Tuple[str, nn.Module]]] = {}
    for name, module in named_modules:
        device = resolve_module_execution_device(module)
        buckets.setdefault(device, []).append((str(name), module))
    return buckets


def _decoder_pack_config_key(decoder: nn.Module) -> Tuple[object, ...]:
    if not isinstance(decoder, Decoder):
        raise TypeError(f"Packed decode expects Decoder instances, got {type(decoder)}.")
    return (
        int(decoder.in_dim),
        int(decoder.out_dim),
        int(decoder.hidden_dim),
        int(decoder.num_res_blocks),
        str(decoder.norm_type),
        str(decoder.activation_type),
        str(decoder.decoder_type),
        bool(decoder.use_checkpoint),
        int(decoder.num_models),
        bool(getattr(decoder, "_q_scale_fused", False)),
    )


def inspect_parallel_stage_pack_compatibility(module: VAELinear) -> Tuple[bool, Optional[str]]:
    """Return (compatible, reason) without mutating the module."""
    if getattr(module, "_parallel_stage_decoder", None) is not None:
        return True, None

    residual_stages = int(getattr(module, "residual_stages", 1))
    parallel_parts = int(getattr(module, "parallel_parts", 1))
    decoder_count = residual_stages * parallel_parts
    if decoder_count <= 1:
        return True, None

    stage_codebook_dims = [int(v) for v in getattr(module, "stage_codebook_dims", [])]
    if len(stage_codebook_dims) != residual_stages:
        return (
            False,
            f"stage_codebook_dims length {len(stage_codebook_dims)} != residual_stages={residual_stages}",
        )
    if len(set(stage_codebook_dims)) != 1:
        return (
            False,
            f"stage codebook dims are not identical: {stage_codebook_dims}",
        )

    decoders: List[nn.Module] = []
    for stage_idx in range(residual_stages):
        for part_idx in range(parallel_parts):
            decoders.append(module.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx))

    try:
        first_key = _decoder_pack_config_key(decoders[0])
    except TypeError as exc:
        return False, str(exc)

    training = bool(decoders[0].training)
    first_device = None
    first_dtype = None
    for param in decoders[0].parameters():
        if param.is_floating_point():
            first_device = param.device
            first_dtype = param.dtype
            break

    for idx, decoder in enumerate(decoders[1:], start=1):
        try:
            key = _decoder_pack_config_key(decoder)
        except TypeError as exc:
            return False, str(exc)
        if key != first_key:
            return False, f"decoder layout/config mismatch at pack index={idx}"
        if bool(decoder.training) != training:
            return False, "decoder training-mode mismatch across stages/parts"
        for param in decoder.parameters():
            if not param.is_floating_point():
                continue
            if first_device is None:
                first_device = param.device
                first_dtype = param.dtype
            elif param.device != first_device or param.dtype != first_dtype:
                return (
                    False,
                    f"decoder dtype/device mismatch at pack index={idx}: "
                    f"device={param.device}, dtype={param.dtype} vs device={first_device}, dtype={first_dtype}",
                )
            break
    return True, None


def resolve_decoder_execution_plan(
    module: VAELinear,
    *,
    mode: str,
) -> DecoderExecutionPlan:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in DECODER_EXECUTION_MODES:
        raise ValueError(
            f"decoder execution mode must be one of {DECODER_EXECUTION_MODES}, got {mode!r}."
        )
    if not isinstance(module, VAELinear):
        raise TypeError(f"resolve_decoder_execution_plan expects VAELinear, got {type(module)}.")

    residual_stages = int(getattr(module, "residual_stages", 1))
    parallel_parts = int(getattr(module, "parallel_parts", 1))
    decoder_count = residual_stages * parallel_parts

    if getattr(module, "_parallel_stage_decoder", None) is not None:
        return DecoderExecutionPlan(
            mode=normalized_mode,
            use_packed=True,
            reason="already_packed",
            decoder_count=decoder_count,
            pack_compatible=True,
            incompatibility_reason=None,
        )

    if decoder_count <= 1:
        return DecoderExecutionPlan(
            mode=normalized_mode,
            use_packed=False,
            reason="single_decoder_serial",
            decoder_count=decoder_count,
            pack_compatible=True,
            incompatibility_reason=None,
        )

    compatible, incompat_reason = inspect_parallel_stage_pack_compatibility(module)
    if compatible:
        return DecoderExecutionPlan(
            mode=normalized_mode,
            use_packed=True,
            reason="compatible_multi_stage_pack",
            decoder_count=decoder_count,
            pack_compatible=True,
            incompatibility_reason=None,
        )

    if normalized_mode in {"sparse_bit", "decoder_sparse_bit"}:
        raise RuntimeError(
            "Sparse Bit requires packed decode for multi-stage/part VAELinear, "
            f"but pack is incompatible: {incompat_reason}"
        )

    return DecoderExecutionPlan(
        mode=normalized_mode,
        use_packed=False,
        reason=f"fallback_serial:{incompat_reason}",
        decoder_count=decoder_count,
        pack_compatible=False,
        incompatibility_reason=str(incompat_reason),
    )


def apply_decoder_execution_plan(
    module: VAELinear,
    plan: DecoderExecutionPlan,
) -> DecoderExecutionPlan:
    """Apply a resolved plan; never re-guess pack from a naked bool."""
    if plan.mode == "sparse_bit":
        module.enable_sparse_bit_decode_graph(parallel_stage_decode=bool(plan.use_packed))
    elif plan.mode == "decoder_sparse_bit":
        module.enable_trainable_sparse_bit_decode_graph(
            parallel_stage_decode=bool(plan.use_packed)
        )
    else:
        module.enable_trainable_decode(parallel_stage_decode=bool(plan.use_packed))
    return plan


def enable_vae_linear_by_execution_plan(
    module: VAELinear,
    *,
    mode: str,
) -> DecoderExecutionPlan:
    plan = resolve_decoder_execution_plan(module, mode=mode)
    return apply_decoder_execution_plan(module, plan)


def iter_decode_group_size_candidates(
    num_targets: int,
    *,
    initial_group_size: Optional[int] = None,
) -> Tuple[int, ...]:
    n = max(0, int(num_targets))
    if n < 1:
        return (1,)
    if initial_group_size is None:
        start = min(DEFAULT_DECODE_GROUP_SIZE, n)
    else:
        requested = int(initial_group_size)
        if requested < 1:
            raise ValueError(f"initial_group_size must be >= 1, got {initial_group_size}.")
        start = min(requested, n)
    candidates: List[int] = [int(start)]
    for size in DECODE_GROUP_SIZE_FALLBACK_SEQUENCE:
        size_i = int(size)
        if size_i < start and size_i not in candidates:
            candidates.append(size_i)
    return tuple(candidates)


def _exception_chain(exc: BaseException) -> Iterable[BaseException]:
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__ or current.__context__


def _message_indicates_cuda_allocator_oom(message: str) -> bool:
    msg = str(message).lower()
    if "cudnn error: cudnn_status_alloc_failed" in msg:
        return True
    if "cuda out of memory" in msg or "cuda error: out of memory" in msg:
        return True
    # Require both CUDA context and OOM wording; reject bare "OutOfMemoryError" business failures.
    return "cuda" in msg and "out of memory" in msg


def is_retryable_decode_capacity_error(exc: BaseException) -> bool:
    for item in _exception_chain(exc):
        if isinstance(item, DecodeCapacityError):
            return True
        if isinstance(item, torch.cuda.OutOfMemoryError):
            return True
        msg = str(item)
        # Legacy/aliased OOM class names are accepted only with CUDA/allocator semantics.
        if type(item).__name__ == "OutOfMemoryError" and _message_indicates_cuda_allocator_oom(msg):
            return True
        if _message_indicates_cuda_allocator_oom(msg):
            return True
    return False


def run_with_decode_group_size_fallback(
    fn: Callable[[int], T],
    *,
    num_targets: int,
    initial_group_size: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    log_prefix: str = "",
) -> Tuple[T, ResolvedDecodeGroupSize]:
    """Run ``fn(group_size)`` with the fixed ``8 -> 4 -> 2 -> 1`` capacity fallback.

    Only CUDA/allocator OOM or explicit ``DecodeCapacityError`` may shrink the group.
    Shape/metadata/codebook/contract errors propagate immediately.
    """
    candidates = iter_decode_group_size_candidates(
        num_targets,
        initial_group_size=initial_group_size,
    )
    attempted: List[int] = []
    last_exc: Optional[BaseException] = None
    for group_size in candidates:
        attempted.append(int(group_size))
        try:
            result = fn(int(group_size))
        except Exception as exc:
            if not is_retryable_decode_capacity_error(exc):
                raise
            last_exc = exc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if logger is not None:
                logger.warning(
                    "%sdecode group_size=%d hit capacity/OOM (%s); trying smaller group if available.",
                    log_prefix,
                    int(group_size),
                    type(exc).__name__,
                )
            continue
        reason = "default" if len(attempted) == 1 else "capacity_oom_fallback"
        resolved = ResolvedDecodeGroupSize(
            group_size=int(group_size),
            fallback_reason=reason,
            attempted=tuple(attempted),
            num_targets=int(num_targets),
        )
        if logger is not None and reason != "default":
            logger.info(
                "%sresolved decode group_size=%d after fallback attempts=%s",
                log_prefix,
                int(group_size),
                list(attempted),
            )
        return result, resolved

    assert last_exc is not None
    raise RuntimeError(
        f"{log_prefix}Grouped decode exhausted group_size candidates {list(candidates)} "
        f"due to capacity/OOM errors."
    ) from last_exc


def decode_named_vae_linear_weights_with_group_fallback(
    named_targets: Sequence[Any],
    *,
    dtype: Optional[torch.dtype] = None,
    compute_device: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    respect_cache_policy: bool = True,
) -> Tuple[List[Any], ResolvedDecodeGroupSize]:
    """Grouped decode using actual target devices by default and auto group-size fallback.

    ``compute_device`` must be an actual execution device or ``None`` (per-target device).
    Do not pass legacy ``decode_device=auto`` strings.
    """
    from litebsq.vae_linear_prewarm import decode_named_vae_linear_weights

    if isinstance(compute_device, str) and str(compute_device).strip().lower() == "auto":
        raise ValueError(
            "decode_device='auto' is not a valid runtime truth source; "
            "pass the actual execution device or None for per-target devices."
        )

    targets = tuple(named_targets)
    num_targets = len(targets)

    def _run(group_size: int):
        return decode_named_vae_linear_weights(
            targets,
            dtype=dtype,
            group_size=int(group_size),
            compute_device=compute_device,
            logger=logger,
            respect_cache_policy=respect_cache_policy,
        )

    return run_with_decode_group_size_fallback(
        _run,
        num_targets=num_targets,
        logger=logger,
        log_prefix="[decode_group_fallback] ",
    )


def prime_named_vae_linear_cache_with_group_fallback(
    named_targets: Sequence[Any],
    *,
    dtype: Optional[torch.dtype] = None,
    clear_existing: bool = False,
    compute_device: Optional[Any] = None,
    initial_group_size: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, int], ResolvedDecodeGroupSize]:
    from litebsq.vae_linear_prewarm import prime_named_vae_linear_cache

    if isinstance(compute_device, str) and str(compute_device).strip().lower() == "auto":
        raise ValueError(
            "decode_device='auto' is not a valid runtime truth source; "
            "pass the actual execution device or None for per-target devices."
        )

    targets = tuple(named_targets)
    num_targets = len(targets)

    def _run(group_size: int):
        return prime_named_vae_linear_cache(
            targets,
            dtype=dtype,
            clear_existing=bool(clear_existing),
            group_size=int(group_size),
            compute_device=compute_device,
            logger=logger,
        )

    return run_with_decode_group_size_fallback(
        _run,
        num_targets=num_targets,
        initial_group_size=initial_group_size,
        logger=logger,
        log_prefix="[prewarm_group_fallback] ",
    )
