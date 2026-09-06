from __future__ import annotations

import argparse
import re
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union

from torch import nn

from litebsq.vae_linear import VAELinear
from train_utils.utils import LinearRef, extract_layer_idx, is_decoder_layer_projection


TargetLayers = Union[str, Tuple[int, ...]]
TargetModules = Union[str, Tuple[str, ...]]
SkipLayerKey = Tuple[int, str]

_SKIP_LAYER_PATTERN = re.compile(r"^(\d+)\.([A-Za-z0-9_]+)$")
_MODULE_ALIAS_REJECT = {
    "q": "q_proj",
    "query": "q_proj",
    "k": "k_proj",
    "key": "k_proj",
    "v": "v_proj",
    "value": "v_proj",
    "o": "o_proj",
    "out": "o_proj",
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}


def parse_target_layers(value: object) -> TargetLayers:
    raw = "" if value is None else str(value).strip()
    if raw == "":
        raise argparse.ArgumentTypeError(
            "--target_layers cannot be empty. Use 'all' or an explicit index list such as 0-7,12."
        )
    if raw == "*":
        raise argparse.ArgumentTypeError("--target_layers does not accept '*'. Use 'all'.")
    lowered = raw.lower()
    if lowered == "all":
        return "all"
    if lowered == "*":
        raise argparse.ArgumentTypeError("--target_layers does not accept '*'. Use 'all'.")

    out = []
    seen = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            raise argparse.ArgumentTypeError(
                f"Invalid --target_layers token '{item}'. Empty entries are not allowed."
            )
        if "-" in token:
            parts = [p.strip() for p in token.split("-", 1)]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise argparse.ArgumentTypeError(
                    f"Invalid --target_layers token '{token}'. Expected <idx> or <begin>-<end>."
                )
            try:
                begin = int(parts[0])
                end = int(parts[1])
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid --target_layers token '{token}'. Expected integer indices."
                ) from exc
            if begin < 0 or end < 0:
                raise argparse.ArgumentTypeError(
                    f"Invalid --target_layers range '{token}'. Negative indices are not allowed."
                )
            if end < begin:
                raise argparse.ArgumentTypeError(
                    f"Invalid --target_layers range '{token}'. Reverse ranges are not allowed."
                )
            indices = range(begin, end + 1)
        else:
            try:
                idx = int(token)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid --target_layers token '{token}'. Expected a non-negative integer."
                ) from exc
            if idx < 0:
                raise argparse.ArgumentTypeError(
                    f"Invalid --target_layers token '{token}'. Negative indices are not allowed."
                )
            indices = (idx,)

        for idx in indices:
            if idx in seen:
                raise argparse.ArgumentTypeError(
                    f"Duplicate --target_layers index {idx}."
                )
            seen.add(idx)
            out.append(idx)

    if not out:
        raise argparse.ArgumentTypeError("--target_layers cannot be empty.")
    return tuple(sorted(out))


def resolve_target_layers(parsed: TargetLayers, *, num_layers: int) -> Tuple[int, ...]:
    n = int(num_layers)
    if n < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
    if parsed == "all":
        return tuple(range(n))
    layers = tuple(int(idx) for idx in parsed)
    overflow = [idx for idx in layers if idx >= n]
    if overflow:
        raise ValueError(
            f"--target_layers exceeds model layer range [0, {n - 1}]: {overflow}."
        )
    return layers


def parse_target_modules(value: object) -> TargetModules:
    raw = "" if value is None else str(value).strip()
    if raw == "":
        raise argparse.ArgumentTypeError(
            "--target_modules cannot be empty. Use 'all' or explicit suffixes such as q_proj,k_proj."
        )
    if raw == "*":
        raise argparse.ArgumentTypeError("--target_modules does not accept '*'. Use 'all'.")
    lowered = raw.lower()
    if lowered == "all":
        return "all"

    out = []
    seen = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            raise argparse.ArgumentTypeError(
                f"Invalid --target_modules token '{item}'. Empty entries are not allowed."
            )
        if token == "*":
            raise argparse.ArgumentTypeError("--target_modules does not accept '*'. Use 'all'.")
        alias_target = _MODULE_ALIAS_REJECT.get(token.lower())
        if alias_target is not None:
            raise argparse.ArgumentTypeError(
                f"Invalid --target_modules token '{token}'. Module aliases are not accepted; "
                f"use the exact suffix '{alias_target}'."
            )
        if token in seen:
            raise argparse.ArgumentTypeError(f"Duplicate --target_modules suffix '{token}'.")
        seen.add(token)
        out.append(token)
    if not out:
        raise argparse.ArgumentTypeError("--target_modules cannot be empty.")
    return tuple(out)


def parse_compression_categories(value: object) -> Tuple[str, ...]:
    raw = "" if value is None else str(value).strip()
    if not raw:
        raise argparse.ArgumentTypeError("--compression_categories must not be empty.")
    categories = []
    seen = set()
    duplicates = []
    reserved = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in {"auto", "others", "all", "*"}:
            reserved.append(token)
            continue
        if token in seen:
            if token not in duplicates:
                duplicates.append(token)
            continue
        seen.add(token)
        categories.append(token)
    if reserved:
        raise argparse.ArgumentTypeError(
            "--compression_categories only accepts explicit categories; "
            f"unsupported values: {','.join(reserved)}"
        )
    if duplicates:
        raise argparse.ArgumentTypeError(
            "--compression_categories contains duplicate categories: " + ",".join(duplicates)
        )
    if not categories:
        raise argparse.ArgumentTypeError("--compression_categories must not be empty.")
    return tuple(categories)


def parse_skip_layers(value: object) -> FrozenSet[SkipLayerKey]:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return frozenset()
    out = []
    seen = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        match = _SKIP_LAYER_PATTERN.match(token)
        if not match:
            raise argparse.ArgumentTypeError(
                f"Invalid --skip_layers entry '{token}'. Expected format: <layer_idx>.<category>, "
                "for example 0.down_proj or 30.q_proj."
            )
        key = (int(match.group(1)), str(match.group(2)))
        if key in seen:
            raise argparse.ArgumentTypeError(f"Duplicate --skip_layers entry '{token}'.")
        seen.add(key)
        out.append(key)
    return frozenset(out)


def validate_skip_layers_scope(
    skip_layers: Iterable[SkipLayerKey],
    *,
    target_layers: TargetLayers,
    compression_categories: Sequence[str],
) -> None:
    skip_set = {(int(layer_idx), str(category)) for layer_idx, category in skip_layers}
    if not skip_set:
        return
    category_set = {str(category) for category in compression_categories}
    unknown_categories = sorted(
        {category for _layer_idx, category in skip_set if category not in category_set}
    )
    if unknown_categories:
        raise ValueError(
            "--skip_layers categories must belong to --compression_categories. "
            f"Unknown: {unknown_categories}."
        )
    if target_layers == "all":
        return
    allowed_layers = {int(idx) for idx in target_layers}
    unknown_layers = sorted({layer_idx for layer_idx, _category in skip_set if layer_idx not in allowed_layers})
    if unknown_layers:
        raise ValueError(
            "--skip_layers layer indices must belong to --target_layers. "
            f"Outside target_layers: {unknown_layers}."
        )


def is_legal_compressed_vaelinear(module: nn.Module) -> bool:
    return isinstance(module, VAELinear) and not bool(getattr(module, "always_use_original", False))


def _module_suffix(name: str) -> str:
    return str(name).rsplit(".", 1)[-1]


def collect_e2e_compressed_targets(
    model: nn.Module,
    *,
    target_layers: TargetLayers,
    target_modules: TargetModules,
    num_layers: int,
) -> List[Tuple[str, VAELinear]]:
    layer_ids = set(resolve_target_layers(target_layers, num_layers=num_layers))
    requested_suffixes: Optional[Tuple[str, ...]]
    if target_modules == "all" or target_modules is None:
        requested_suffixes = None
    else:
        requested_suffixes = tuple(str(suffix) for suffix in target_modules)

    selected: List[Tuple[str, VAELinear]] = []
    hits = {suffix: 0 for suffix in requested_suffixes or ()}
    for name, module in model.named_modules():
        if not is_legal_compressed_vaelinear(module):
            continue
        layer_idx = extract_layer_idx(name)
        if layer_idx is None or int(layer_idx) not in layer_ids:
            continue
        suffix = _module_suffix(name)
        if requested_suffixes is not None and suffix not in hits:
            continue
        selected.append((str(name), module))
        if requested_suffixes is not None:
            hits[suffix] = hits.get(suffix, 0) + 1

    if requested_suffixes is not None:
        missing = [suffix for suffix in requested_suffixes if hits.get(suffix, 0) < 1]
        if missing:
            raise ValueError(
                "Each --target_modules suffix must hit at least one legal compressed VAELinear "
                f"in the selected target_layers. Missing: {missing}."
            )
    return selected


def discover_cat_projection_name_inventory(
    model: nn.Module,
    *,
    compression_categories: Sequence[str],
) -> Dict[SkipLayerKey, str]:
    """Discover the canonical CAT transformer projection inventory once.

    Dict insertion order follows ``model.named_modules()`` and is therefore the
    stable source order for remaining-target selection. A <layer, category> key
    must resolve to exactly one logical projection name.
    """
    categories = tuple(str(category) for category in compression_categories)
    category_set = set(categories)
    inventory: Dict[SkipLayerKey, str] = {}
    for name, _module in model.named_modules():
        category = _module_suffix(name)
        if category not in category_set:
            continue
        layer_idx = extract_layer_idx(name)
        if layer_idx is None:
            continue
        if not is_decoder_layer_projection(name, categories):
            continue
        key = (int(layer_idx), str(category))
        existing = inventory.get(key)
        if existing is not None and existing != str(name):
            raise ValueError(
                "CAT projection inventory key is ambiguous: "
                f"{key} -> {existing!r} and {str(name)!r}."
            )
        inventory[key] = str(name)
    return inventory


def discover_cat_projection_inventory(
    model: nn.Module,
    *,
    compression_categories: Sequence[str],
) -> FrozenSet[SkipLayerKey]:
    return frozenset(
        discover_cat_projection_name_inventory(
            model,
            compression_categories=compression_categories,
        ).keys()
    )


def validate_skip_layers_against_inventory(
    skip_layers: Iterable[SkipLayerKey],
    *,
    target_layers: TargetLayers,
    compression_categories: Sequence[str],
    inventory: Iterable[SkipLayerKey],
) -> None:
    skip_set = frozenset((int(layer_idx), str(category)) for layer_idx, category in skip_layers)
    validate_skip_layers_scope(
        skip_set,
        target_layers=target_layers,
        compression_categories=compression_categories,
    )
    if not skip_set:
        return
    inventory_set = {(int(layer_idx), str(category)) for layer_idx, category in inventory}
    missing = sorted(skip_set - inventory_set)
    if missing:
        raise ValueError(
            "skip_layers contains <layer,category> pairs that are not present in the CAT "
            "transformer projection inventory: "
            + ",".join(f"{layer_idx}.{category}" for layer_idx, category in missing)
        )


def select_remaining_dense_refs(
    refs: Sequence[LinearRef],
    *,
    remaining_categories: Sequence[str],
    skip_layers: Iterable[SkipLayerKey],
    target_layers: TargetLayers,
) -> List[LinearRef]:
    remaining_set = {str(category) for category in remaining_categories}
    skip_set = {(int(layer_idx), str(category)) for layer_idx, category in skip_layers}
    selected: List[LinearRef] = []
    for ref in refs:
        if str(ref.category) not in remaining_set:
            continue
        if not isinstance(ref.module, nn.Linear) or isinstance(ref.module, VAELinear):
            continue
        layer_idx = extract_layer_idx(ref.name)
        if layer_idx is None:
            continue
        if target_layers != "all" and int(layer_idx) not in {int(idx) for idx in target_layers}:
            continue
        if (int(layer_idx), str(ref.category)) in skip_set:
            continue
        selected.append(ref)
    return selected
