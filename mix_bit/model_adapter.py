from __future__ import annotations

import logging
import re
from typing import Any, Callable, Protocol

from torch import nn
from transformers import AutoTokenizer

from mix_bit.model_inventory import TargetLinearSpec
from mix_bit.schema import CategorySpec, ModelProfile

logger = logging.getLogger(__name__)

_ADAPTER_REGISTRY: dict[str, Callable[[], "ModelAdapter"]] = {}


def normalize_tokenizer_for_mix_bit(tokenizer: Any, *, source_label: str) -> Any:
    """Force right padding and normalize a missing pad token to eos exactly once.

    Order is fixed: set ``padding_side="right"``; reset
    ``mix_bit_pad_token_normalized_from_eos`` to False; if ``pad_token_id`` is
    None, require ``eos_token_id`` and set ``pad_token_id=eos_token_id`` with
    the normalization flag set to True. Returns the same tokenizer object.
    """
    tokenizer.padding_side = "right"
    tokenizer.mix_bit_pad_token_normalized_from_eos = False
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                f"Tokenizer for {source_label!r} has neither pad_token_id nor eos_token_id"
            )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.mix_bit_pad_token_normalized_from_eos = True
        logger.info(
            "Normalized missing pad_token_id to eos_token_id=%s for %s",
            tokenizer.eos_token_id,
            source_label,
        )
    return tokenizer


class ModelAdapter(Protocol):
    name: str

    def load_model(
        self,
        profile: ModelProfile,
        *,
        access_token: str | None = None,
    ) -> nn.Module: ...

    def load_tokenizer(
        self,
        profile: ModelProfile,
        *,
        access_token: str | None = None,
    ): ...

    def discover_target_linears(
        self,
        model: nn.Module,
        profile: ModelProfile,
    ) -> tuple[TargetLinearSpec, ...]: ...


def register_model_adapter(name: str, factory: Callable[[], ModelAdapter]) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("Adapter name must be non-empty")
    _ADAPTER_REGISTRY[key] = factory


def get_model_adapter(name: str) -> ModelAdapter:
    key = str(name).strip()
    try:
        factory = _ADAPTER_REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(_ADAPTER_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown model adapter {key!r}. Known adapters: {known}") from exc
    return factory()


def _suffix_matches(module_name: str, suffix: str) -> bool:
    return module_name == suffix or module_name.endswith("." + suffix)


def _match_category(module_name: str, categories: tuple[CategorySpec, ...]) -> CategorySpec | None:
    matches = [cat for cat in categories if _suffix_matches(module_name, cat.module_suffix)]
    if not matches:
        return None
    if len(matches) > 1:
        names = ", ".join(cat.name for cat in matches)
        raise ValueError(
            f"Ambiguous suffix match for module {module_name!r}: categories [{names}]"
        )
    return matches[0]


def _extract_block_index(module_name: str, patterns: tuple[str, ...]) -> int:
    for pattern in patterns:
        match = re.search(pattern, module_name)
        if match is None:
            continue
        if match.lastindex is None:
            continue
        return int(match.group(1))
    raise ValueError(f"Missing block index for module {module_name!r}")


def _iter_named_modules_allow_shared(module: nn.Module, prefix: str = ""):
    """Walk ``_modules`` so shared child objects remain visible under every name."""
    for name, child in module._modules.items():
        if child is None:
            continue
        full_name = f"{prefix}.{name}" if prefix else name
        yield full_name, child
        yield from _iter_named_modules_allow_shared(child, full_name)


class GenericDecoderAdapter:
    name = "generic_decoder"

    def load_model(
        self,
        profile: ModelProfile,
        *,
        access_token: str | None = None,
    ) -> nn.Module:
        from rotation.model_utils import get_model

        return get_model(profile.model_path, access_token)

    def load_tokenizer(
        self,
        profile: ModelProfile,
        *,
        access_token: str | None = None,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            profile.model_path,
            use_fast=True,
            token=access_token,
            trust_remote_code=False,
        )
        return normalize_tokenizer_for_mix_bit(
            tokenizer, source_label=profile.model_path
        )

    def discover_target_linears(
        self,
        model: nn.Module,
        profile: ModelProfile,
    ) -> tuple[TargetLinearSpec, ...]:
        if not profile.categories:
            raise ValueError("Model profile categories must be non-empty")

        category_order = {cat.name: idx for idx, cat in enumerate(profile.categories)}
        seen_names: set[str] = set()
        seen_ids: dict[int, str] = {}
        matched_by_category: dict[str, int] = {cat.name: 0 for cat in profile.categories}
        targets: list[TargetLinearSpec] = []

        for module_name, module in _iter_named_modules_allow_shared(model):
            category = _match_category(module_name, profile.categories)
            if category is None:
                continue
            if type(module) is not nn.Linear:
                raise ValueError(
                    f"Target module {module_name!r} is not a plain nn.Linear "
                    f"(got {type(module).__name__})"
                )
            if module_name in seen_names:
                raise ValueError(f"Duplicated module name {module_name!r}")
            module_id = id(module)
            if module_id in seen_ids:
                raise ValueError(
                    f"Shared target module object appears under {seen_ids[module_id]!r} "
                    f"and {module_name!r}"
                )
            block_index = _extract_block_index(module_name, profile.layer_index_patterns)
            in_features = int(module.in_features)
            out_features = int(module.out_features)
            has_bias = module.bias is not None
            targets.append(
                TargetLinearSpec(
                    module_name=module_name,
                    category=category.name,
                    module_suffix=category.module_suffix,
                    block_index=block_index,
                    in_features=in_features,
                    out_features=out_features,
                    has_bias=has_bias,
                    param_count=in_features * out_features,
                    transpose=category.transpose,
                )
            )
            seen_names.add(module_name)
            seen_ids[module_id] = module_name
            matched_by_category[category.name] += 1

        empty = [name for name, count in matched_by_category.items() if count == 0]
        if empty:
            raise ValueError(f"Empty categories with no matched modules: {empty}")

        targets.sort(
            key=lambda item: (
                item.block_index,
                category_order[item.category],
                item.module_name,
            )
        )
        return tuple(targets)


def _build_generic_decoder() -> ModelAdapter:
    return GenericDecoderAdapter()


register_model_adapter("generic_decoder", _build_generic_decoder)
