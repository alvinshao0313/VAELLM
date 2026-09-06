from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from e2e_common.proxy_trainables import iter_named_vae_module_refs
from litebsq.vae_linear import NamedVAELinearTarget, VAELinear
from litebsq.vae_linear_prewarm import decode_named_vae_linear_weights
from train_utils.v6_model_loader import load_v6_model_checkpoint

_DOWN_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj$")
_FORMAL_NUM_LAYERS = 36


@dataclass(frozen=True)
class DownLayerRef:
    layer_idx: int
    name: str
    module: VAELinear


def discover_down_layers(model) -> list[DownLayerRef]:
    expected_layers = int(model.config.num_hidden_layers)
    if expected_layers != _FORMAL_NUM_LAYERS:
        raise ValueError(
            f"Formal sensitivity run requires num_hidden_layers={_FORMAL_NUM_LAYERS}, got {expected_layers}."
        )

    refs: list[DownLayerRef] = []
    for name, module in model.named_modules():
        match = _DOWN_RE.match(name)
        if match is None:
            continue
        if not isinstance(module, VAELinear):
            raise TypeError(f"Expected VAELinear at {name}, got {type(module).__name__}.")
        layer_idx = int(match.group(1))
        refs.append(DownLayerRef(layer_idx=layer_idx, name=name, module=module))

    refs.sort(key=lambda ref: ref.layer_idx)
    layer_indexes = [ref.layer_idx for ref in refs]
    expected_indexes = list(range(_FORMAL_NUM_LAYERS))
    if layer_indexes != expected_indexes:
        raise ValueError(
            f"Expected contiguous down layer indexes {expected_indexes}, got {layer_indexes}."
        )

    for ref in refs:
        if ref.module.original_weight is None:
            raise ValueError(f"Down layer {ref.name} is missing original_weight.")
        if bool(getattr(ref.module, "always_use_original", False)):
            raise ValueError(f"Down layer {ref.name} has always_use_original=True.")

    return refs


def reset_all_vae_to_compressed(model) -> None:
    for module in model.modules():
        if isinstance(module, VAELinear):
            if bool(getattr(module, "always_use_original", False)):
                raise ValueError("Formal sensitivity run requires no always_use_original VAELinear.")
            module.set_temporary(True)


def set_down_restore_set(
    down_layers: list[DownLayerRef],
    restore_layers: set[int],
) -> None:
    valid = {ref.layer_idx for ref in down_layers}
    unknown = sorted(set(restore_layers) - valid)
    if unknown:
        raise ValueError(f"Unknown down layer indices: {unknown}")

    for ref in down_layers:
        ref.module.set_temporary(ref.layer_idx not in restore_layers)


def assert_down_restore_set(
    down_layers: list[DownLayerRef],
    restore_layers: set[int],
) -> None:
    for ref in down_layers:
        expected_temporary = ref.layer_idx not in restore_layers
        actual_temporary = bool(getattr(ref.module, "temporary", True))
        if actual_temporary != expected_temporary:
            raise ValueError(
                f"Down layer {ref.name} temporary={actual_temporary}, expected {expected_temporary}."
            )
        if not expected_temporary and ref.module.original_weight is None:
            raise ValueError(
                f"Down layer {ref.name} expected original path but original_weight is None."
            )


def unload_non_down_original_weights(
    model,
    down_names: set[str],
) -> dict:
    total_vae = 0
    down_original_kept = 0
    non_down_original_unloaded = 0
    non_down_already_unloaded = 0
    non_down_protected_original_kept = 0

    for name, module in model.named_modules():
        if not isinstance(module, VAELinear):
            continue
        total_vae += 1
        if name in down_names:
            if module.original_weight is not None:
                down_original_kept += 1
            continue

        unloaded = module.unload_original_linear()
        if unloaded:
            if module.original_weight is not None:
                raise RuntimeError(
                    f"VAELinear {name} reported unload success but original_weight is still present."
                )
            non_down_original_unloaded += 1
            continue

        if module.original_weight is None:
            non_down_already_unloaded += 1
            continue

        if bool(getattr(module, "protect_original_weight", False)):
            non_down_protected_original_kept += 1
            continue

        raise RuntimeError(
            f"VAELinear {name} retained original_weight without protect_original_weight."
        )

    return {
        "total_vae": total_vae,
        "down_original_kept": down_original_kept,
        "non_down_original_unloaded": non_down_original_unloaded,
        "non_down_already_unloaded": non_down_already_unloaded,
        "non_down_protected_original_kept": non_down_protected_original_kept,
    }


def stage_cached_weight_to_cpu(module: VAELinear, decoded_weight: torch.Tensor) -> None:
    """Assign a decoded weight into module cache and move it to CPU."""
    module._cached_weight = decoded_weight.detach().to("cpu")


def hoist_cached_weights_to_device(model, device) -> int:
    """Move every non-None VAELinear._cached_weight onto ``device``. Returns count moved."""
    target = torch.device(device)
    moved = 0
    for module in model.modules():
        if not isinstance(module, VAELinear):
            continue
        cached = module._cached_weight
        if cached is None:
            continue
        if cached.device != target:
            module._cached_weight = cached.to(target, non_blocking=False)
        moved += 1
    return moved


def assert_cached_weights_on_device(modules: list[VAELinear], device) -> None:
    target = torch.device(device)
    for module in modules:
        cached = module._cached_weight
        if cached is None:
            raise RuntimeError("Missing prewarmed decoded-weight cache after staged prewarm.")
        if cached.device != target:
            raise RuntimeError(
                f"Prewarmed cache device mismatch: expected {target}, got {cached.device}."
            )


def _move_original_weight_to_device(module: VAELinear, device) -> None:
    weight = module.original_weight
    if weight is None:
        raise RuntimeError("Cannot move original_weight: it is None.")
    target = torch.device(device)
    if weight.device == target:
        return
    module.register_parameter(
        "original_weight",
        nn.Parameter(weight.detach().to(target, non_blocking=False), requires_grad=False),
    )


def pin_down_original_weights_to_cpu(down_layers: list[DownLayerRef]) -> int:
    """Keep every down original_weight on CPU. Returns how many were moved."""
    moved = 0
    cpu = torch.device("cpu")
    for ref in down_layers:
        weight = ref.module.original_weight
        if weight is None:
            raise RuntimeError(f"Down layer {ref.name} is missing original_weight.")
        if weight.device != cpu:
            _move_original_weight_to_device(ref.module, cpu)
            moved += 1
    return moved


def hoist_down_original_weights(
    down_layers: list[DownLayerRef],
    restore_layers: set[int],
    device,
) -> int:
    """Move only restore-set down originals to ``device``; leave others on CPU."""
    valid = {ref.layer_idx for ref in down_layers}
    unknown = sorted(set(restore_layers) - valid)
    if unknown:
        raise ValueError(f"Unknown down layer indices: {unknown}")

    target = torch.device(device)
    cpu = torch.device("cpu")
    moved = 0
    for ref in down_layers:
        if ref.layer_idx in restore_layers:
            if ref.module.original_weight is None:
                raise RuntimeError(f"Down layer {ref.name} is missing original_weight.")
            if ref.module.original_weight.device != target:
                _move_original_weight_to_device(ref.module, target)
                moved += 1
        else:
            if ref.module.original_weight is None:
                raise RuntimeError(f"Down layer {ref.name} is missing original_weight.")
            if ref.module.original_weight.device != cpu:
                _move_original_weight_to_device(ref.module, cpu)
    return moved


def assert_down_original_devices(
    down_layers: list[DownLayerRef],
    restore_layers: set[int],
    device,
) -> None:
    target = torch.device(device)
    cpu = torch.device("cpu")
    for ref in down_layers:
        weight = ref.module.original_weight
        if weight is None:
            raise RuntimeError(f"Down layer {ref.name} is missing original_weight.")
        expected = target if ref.layer_idx in restore_layers else cpu
        if weight.device != expected:
            raise RuntimeError(
                f"Down layer {ref.name} original_weight on {weight.device}, expected {expected}."
            )


def _move_vae_linear_compute_state_to_cpu(module: VAELinear) -> None:
    """Return a module (and decode scratch state) to CPU after a decode batch."""
    clear_rt = getattr(module, "_clear_parallel_stage_decode_runtime_cache", None)
    if callable(clear_rt):
        clear_rt()
    clear_prot = getattr(module, "_clear_protected_residual_parallel_runtime_cache", None)
    if callable(clear_prot):
        clear_prot()

    module.to("cpu")
    packed = getattr(module, "_parallel_stage_decoder", None)
    if packed is not None:
        packed.to(device=torch.device("cpu"))

    # Buffers / attrs that may remain on CUDA if not registered as child modules.
    for attr in (
        "_parallel_stage_grouped_vq_weight",
        "_parallel_stage_grouped_vq_runtime",
        "_parallel_stage_model_indices_runtime",
        "_protected_residual_parallel_grouped_vq_weight",
        "_protected_residual_parallel_grouped_vq_runtime",
    ):
        val = getattr(module, attr, None)
        if not torch.is_tensor(val):
            continue
        if val.device.type != "cuda":
            continue
        if attr.endswith("_weight"):
            setattr(module, attr, val.detach().to("cpu"))
        else:
            setattr(module, attr, None)


def _release_all_vae_compute_state_to_cpu(model) -> None:
    for module in model.modules():
        if isinstance(module, VAELinear):
            _move_vae_linear_compute_state_to_cpu(module)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prewarm_compressed_weights(
    model,
    device,
    group_size: int,
    down_layers: list[DownLayerRef],
) -> dict[str, int]:
    """Decode in batches on GPU while keeping the model body on CPU, then prepare for eval.

    Each batch: decode on ``device`` → stage ``_cached_weight`` to CPU → move batch modules
    back to CPU. After all batches: hoist caches to ``device``, move model body to ``device``,
    then pin all down ``original_weight`` back to CPU for lazy restore.
    """
    reset_all_vae_to_compressed(model)
    model.eval()

    group_size = int(group_size)
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}.")

    named_targets = [
        NamedVAELinearTarget(name=ref.name, base_layer=ref.base_layer)
        for ref in iter_named_vae_module_refs(model)
    ]
    total = len(named_targets)
    warmed_modules: list[VAELinear] = []
    warmed_names: set[str] = set()

    for start in range(0, total, group_size):
        batch = named_targets[start : start + group_size]
        results = decode_named_vae_linear_weights(
            batch,
            group_size=group_size,
            compute_device=device,
            respect_cache_policy=True,
        )
        batch_modules: list[VAELinear] = []
        for result in results:
            stage_cached_weight_to_cpu(result.base_layer, result.decoded_weight)
            batch_modules.append(result.base_layer)
            if result.name not in warmed_names:
                warmed_names.add(result.name)
                warmed_modules.append(result.base_layer)
        del results
        for module in batch_modules:
            _move_vae_linear_compute_state_to_cpu(module)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _release_all_vae_compute_state_to_cpu(model)

    # Detach CPU caches before model.to so .to() does not try to move ~all decoded
    # weights in the same pass as the model body + down originals.
    stashed_caches: dict[int, torch.Tensor] = {}
    for module in warmed_modules:
        cached = module._cached_weight
        if cached is None:
            raise RuntimeError("Missing staged CPU cache before model body move.")
        stashed_caches[id(module)] = cached.detach().to("cpu")
        module._cached_weight = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.to(device)
    pin_down_original_weights_to_cpu(down_layers)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for module in warmed_modules:
        cached = stashed_caches.get(id(module))
        if cached is None:
            raise RuntimeError("Missing stashed decoded-weight cache after model body move.")
        module._cached_weight = cached.to(device, non_blocking=False)
    del stashed_caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    assert_cached_weights_on_device(warmed_modules, device)
    assert_down_original_devices(down_layers, set(), device)

    warmed = len(warmed_modules)
    skipped = total - warmed
    failed = 0
    stats = {
        "total": int(total),
        "warmed": int(warmed),
        "skipped": int(skipped),
        "failed": int(failed),
    }
    if failed != 0:
        raise RuntimeError(f"VAELinear prewarm failed: {stats}")
    return stats


@torch.no_grad()
def compute_down_weight_metrics(down_layers: list[DownLayerRef]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for ref in down_layers:
        if ref.module.original_weight is None:
            raise RuntimeError(f"Missing original_weight for {ref.name}")
        if ref.module._cached_weight is None:
            raise RuntimeError(f"Missing prewarmed decoded-weight cache for {ref.name}")
        # Compare on CPU so NMSE never hoists all 36 originals onto GPU.
        w_orig = ref.module.original_weight.detach().float().cpu()
        w_comp = ref.module._cached_weight.detach().float().cpu()
        if w_orig.shape != w_comp.shape:
            raise RuntimeError(
                f"Shape mismatch for {ref.name}: original={tuple(w_orig.shape)} cached={tuple(w_comp.shape)}"
            )

        err = w_comp - w_orig
        sse = err.pow(2).sum(dtype=torch.float64)
        orig_ss = w_orig.pow(2).sum(dtype=torch.float64)
        if float(orig_ss.item()) <= 0.0:
            raise RuntimeError(f"original_weight sum of squares must be > 0 for {ref.name}")

        numel = int(w_orig.numel())
        mse = float((sse / numel).item())
        nmse = float((sse / orig_ss).item())
        metrics.append(
            {
                "layer_idx": ref.layer_idx,
                "name": ref.name,
                "numel": numel,
                "mse": mse,
                "nmse": nmse,
                "relative_fro_error": math.sqrt(nmse),
                "original_rms": math.sqrt(float((orig_ss / numel).item())),
                "error_rms": math.sqrt(float((sse / numel).item())),
            }
        )
    return metrics


def load_worker_model(checkpoint_dir: str, device, prewarm_group_size: int) -> dict[str, Any]:
    model, meta, _load_result = load_v6_model_checkpoint(
        checkpoint_dir,
        map_location="cpu",
        strict=True,
    )
    model.eval()
    reset_all_vae_to_compressed(model)
    down_layers = discover_down_layers(model)
    down_names = {ref.name for ref in down_layers}
    unload_non_down_original_weights(model, down_names)
    prewarm_stats = prewarm_compressed_weights(
        model,
        device,
        prewarm_group_size,
        down_layers,
    )
    assert_down_restore_set(down_layers, set())
    assert_down_original_devices(down_layers, set(), device)
    return {
        "model": model,
        "meta": meta,
        "down_layers": down_layers,
        "prewarm_stats": prewarm_stats,
    }
