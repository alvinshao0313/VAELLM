"""Shared model-level trainable enablement for CAT/E2E (no Sparse Bit imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from e2e_common.full_lora import (
    _logical_adapter_target_name,
    build_full_compressed_peft_model,
    finalize_model_level_lora,
    iter_named_peft_lora_layers,
)
from e2e_common.post_norm_head import (
    LMHeadWithPostNormLinear,
    ensure_post_norm_head_linear,
    fuse_post_norm_head_linear,
    resolve_post_norm_linear,
)
from litebsq.vae_linear import VAELinear
from rotation.model_utils import get_model_type, get_pre_head_layernorm
from train_utils.config.configs import AuxTrainableConfig, LM_HEAD_TRAIN_MODES, NORM_TRAIN_MODES


def _collect_norm_type_registry() -> Tuple[type, ...]:
    types: List[type] = [nn.LayerNorm, nn.RMSNorm]
    try:
        import transformers

        for path in (
            ("models", "llama", "modeling_llama", "LlamaRMSNorm"),
            ("models", "mistral", "modeling_mistral", "MistralRMSNorm"),
            ("models", "qwen2", "modeling_qwen2", "Qwen2RMSNorm"),
            ("models", "qwen3", "modeling_qwen3", "Qwen3RMSNorm"),
            ("models", "opt", "modeling_opt", "OPTRMSNorm"),
        ):
            mod = transformers
            try:
                for part in path[:-1]:
                    mod = getattr(mod, part)
                cls = getattr(mod, path[-1], None)
                if isinstance(cls, type):
                    types.append(cls)
            except Exception:
                continue
    except Exception:
        pass
    # Deduplicate while preserving order.
    seen = set()
    unique: List[type] = []
    for cls in types:
        if cls in seen:
            continue
        seen.add(cls)
        unique.append(cls)
    return tuple(unique)


NORM_TYPE_REGISTRY: Tuple[type, ...] = _collect_norm_type_registry()


def is_backbone_norm_module(module: nn.Module) -> bool:
    return isinstance(module, NORM_TYPE_REGISTRY)


def _base_model(model: nn.Module) -> nn.Module:
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        base = getter()
        if isinstance(base, nn.Module):
            return base
    return model


def _find_module_name(model: nn.Module, target: nn.Module, fallback: str) -> str:
    for name, module in model.named_modules():
        if module is target:
            return str(name)
    return str(fallback)


def freeze_all_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def _module_is_under_vae(module_name: str, model: nn.Module) -> bool:
    """Exclude norms owned by VAELinear / decoder subgraphs."""
    modules = dict(model.named_modules())
    parts = str(module_name).split(".")
    for idx in range(len(parts)):
        prefix = ".".join(parts[: idx + 1])
        child = modules.get(prefix)
        if isinstance(child, VAELinear):
            return True
        if isinstance(child, nn.Module) and child.__class__.__name__ in {
            "Decoder",
            "MultiStageDecoder",
            "FusedMultiStageDecoder",
        }:
            return True
    return False


def iter_named_backbone_norm_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    root = _base_model(model)
    for name, module in root.named_modules():
        if not name:
            continue
        if not is_backbone_norm_module(module):
            continue
        if _module_is_under_vae(name, root):
            continue
        yield str(name), module


def enable_norm_trainables(
    model: nn.Module,
    *,
    norm_train_mode: str,
) -> Dict[str, nn.Parameter]:
    mode = str(norm_train_mode or "none").strip().lower()
    if mode not in NORM_TRAIN_MODES:
        raise ValueError(f"norm_train_mode must be one of {NORM_TRAIN_MODES}, got {norm_train_mode!r}.")
    if mode == "none":
        return {}

    root = _base_model(model)
    selected: Dict[str, nn.Parameter] = {}
    if mode == "final":
        model_type = get_model_type(root)
        final_norm = get_pre_head_layernorm(root, model_type)
        final_name = _find_module_name(root, final_norm, "model.norm")
        for rel_name, param in final_norm.named_parameters(recurse=True):
            key = f"norm::{final_name}" if not rel_name else f"norm::{final_name}.{rel_name}"
            param.requires_grad_(True)
            selected[key] = param
        return selected

    # mode == "all": type-registry based, excluding VAE internals.
    for name, module in iter_named_backbone_norm_modules(root):
        for rel_name, param in module.named_parameters(recurse=True):
            key = f"norm::{name}" if not rel_name else f"norm::{name}.{rel_name}"
            if key in selected:
                raise RuntimeError(f"duplicate norm trainable key: {key}")
            param.requires_grad_(True)
            selected[key] = param
    return selected


def _resolve_input_embedding(model: nn.Module) -> Optional[nn.Embedding]:
    """Resolve the true input embedding for lm_head tied detection.

    Prefer ``get_input_embeddings()`` when the model exposes that API. Only scan
    ``embed_tokens`` / ``wte`` when no standard API exists.
    """
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        embed = getter()
        if embed is None:
            return None
        if not isinstance(embed, nn.Embedding):
            raise TypeError(
                f"get_input_embeddings() must return nn.Embedding or None, got {type(embed)}."
            )
        return embed

    for candidate_name in ("embed_tokens", "wte"):
        for name, module in model.named_modules():
            if name.endswith(candidate_name) and isinstance(module, nn.Embedding):
                return module
    return None


def _untie_lm_head_if_needed(model: nn.Module) -> nn.Linear:
    root = _base_model(model)
    lm_head = getattr(root, "lm_head", None)
    if not isinstance(lm_head, nn.Linear):
        raise TypeError(f"lm_head full mode expects nn.Linear, got {type(lm_head)}.")
    embed = _resolve_input_embedding(root)
    if embed is not None and lm_head.weight.data_ptr() == embed.weight.data_ptr():
        # Clone/detach to keep step-0 numerics while untying.
        cloned = nn.Linear(
            lm_head.in_features,
            lm_head.out_features,
            bias=lm_head.bias is not None,
            device=lm_head.weight.device,
            dtype=lm_head.weight.dtype,
        )
        with torch.no_grad():
            cloned.weight.copy_(lm_head.weight)
            if lm_head.bias is not None and cloned.bias is not None:
                cloned.bias.copy_(lm_head.bias)
        root.lm_head = cloned
        lm_head = cloned
        config = getattr(root, "config", None)
        if config is not None and hasattr(config, "tie_word_embeddings"):
            config.tie_word_embeddings = False
    return lm_head


def setup_lm_head_trainables(
    model: nn.Module,
    *,
    lm_head_train_mode: str,
) -> Dict[str, nn.Parameter]:
    mode = str(lm_head_train_mode or "none").strip().lower()
    if mode not in LM_HEAD_TRAIN_MODES:
        raise ValueError(f"lm_head_train_mode must be one of {LM_HEAD_TRAIN_MODES}, got {lm_head_train_mode!r}.")
    if mode == "none":
        return {}

    root = _base_model(model)
    selected: Dict[str, nn.Parameter] = {}
    if mode == "linear":
        ensure_post_norm_head_linear(root)
        post = resolve_post_norm_linear(root)
        if post is None:
            raise RuntimeError("lm_head linear mode failed to create post-norm linear.")
        for rel_name, param in post.named_parameters(recurse=True):
            key = f"lm_head_linear::{rel_name}" if rel_name else "lm_head_linear"
            param.requires_grad_(True)
            selected[key] = param
        return selected

    if mode == "full":
        lm_head = _untie_lm_head_if_needed(root)
        for rel_name, param in lm_head.named_parameters(recurse=False):
            key = f"lm_head_full::{rel_name}" if rel_name else "lm_head_full"
            param.requires_grad_(True)
            selected[key] = param
        return selected

    # mode == "lora": adapter is created by enable_model_level_lora_targets.
    return selected


@dataclass
class ModelLevelTrainableSelection:
    """Labeled trainable parameter inventory for optimizer grouping."""

    decoder_parameters: Dict[str, nn.Parameter] = field(default_factory=dict)
    lora_parameters: Dict[str, nn.Parameter] = field(default_factory=dict)
    norm_parameters: Dict[str, nn.Parameter] = field(default_factory=dict)
    lm_head_parameters: Dict[str, nn.Parameter] = field(default_factory=dict)
    compressed_lora_targets: List[str] = field(default_factory=list)
    include_lm_head_lora: bool = False
    peft_model: Optional[nn.Module] = None


def _dedupe_params_by_id(
    params: Dict[str, nn.Parameter],
    *,
    inventory_name: str,
) -> Dict[str, nn.Parameter]:
    """Keep one canonical entry per Parameter object/id within an inventory."""
    selected: Dict[str, nn.Parameter] = {}
    seen_ids: Dict[int, str] = {}
    for key, param in params.items():
        if not isinstance(param, nn.Parameter):
            raise TypeError(f"{inventory_name} entry {key!r} is not nn.Parameter: {type(param)}.")
        pid = id(param)
        if pid in seen_ids:
            continue
        if key in selected and selected[key] is not param:
            raise RuntimeError(f"duplicate key with different Parameter in {inventory_name}: {key}")
        selected[key] = param
        seen_ids[pid] = key
    return selected


def assert_disjoint_component_inventories(
    *,
    decoder_parameters: Dict[str, nn.Parameter],
    lora_parameters: Dict[str, nn.Parameter],
    norm_parameters: Dict[str, nn.Parameter],
    lm_head_parameters: Dict[str, nn.Parameter],
) -> None:
    """Hard error if the same Parameter id appears in more than one component inventory."""
    inventories = {
        "decoder_parameters": decoder_parameters,
        "lora_parameters": lora_parameters,
        "norm_parameters": norm_parameters,
        "lm_head_parameters": lm_head_parameters,
    }
    owner: Dict[int, Tuple[str, str]] = {}
    for inv_name, params in inventories.items():
        for key, param in params.items():
            pid = id(param)
            if pid in owner:
                other_inv, other_key = owner[pid]
                raise RuntimeError(
                    "Parameter id conflict across component inventories: "
                    f"{inv_name}[{key!r}] vs {other_inv}[{other_key!r}]."
                )
            owner[pid] = (inv_name, key)


def enable_decoder_trainables(
    model: nn.Module,
    *,
    selected_modules: Sequence[Tuple[str, VAELinear]],
    execution_mode: str = "trainable_decoder",
) -> Dict[str, nn.Parameter]:
    """Enable the exact decoder graph under selected VAELinear modules.

    Uses the shared Task-7 execution-plan resolver, supports residual stages and
    intra-parallel parts, and deduplicates shared decoder parameters by object id.
    """
    _ = model  # selection is driven by explicit module refs
    from train_utils.decoder_execution import enable_vae_linear_by_execution_plan

    selected: Dict[str, nn.Parameter] = {}
    seen_ids: Dict[int, str] = {}
    for module_name, vae in selected_modules:
        plan = enable_vae_linear_by_execution_plan(vae, mode=str(execution_mode))
        packed = getattr(vae, "_parallel_stage_decoder", None)
        decoder_refs: List[Tuple[str, nn.Module]] = []
        if bool(plan.use_packed) and isinstance(packed, nn.Module):
            decoder_refs.append(("packed", packed))
        else:
            local_seen: set[int] = set()
            for stage_idx in range(int(getattr(vae, "residual_stages", 1))):
                for part_idx in range(int(getattr(vae, "parallel_parts", 1))):
                    decoder = vae.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)
                    decoder_id = id(decoder)
                    if decoder_id in local_seen:
                        continue
                    local_seen.add(decoder_id)
                    decoder_refs.append((f"stage{stage_idx}.part{part_idx}", decoder))

        if not decoder_refs:
            raise RuntimeError(f"{module_name}: selected VAELinear has no decoder modules.")
        for decoder_name, decoder in decoder_refs:
            decoder.requires_grad_(True)
            for rel_name, param in decoder.named_parameters(recurse=True):
                key = f"decoder::{module_name}.{decoder_name}.{rel_name}"
                pid = id(param)
                if pid in seen_ids:
                    continue
                param.requires_grad_(True)
                selected[key] = param
                seen_ids[pid] = key
    return selected


def enable_model_level_lora_targets(
    model: nn.Module,
    *,
    compressed_modules: Sequence[Tuple[str, VAELinear]] = (),
    dense_target_modules: Sequence[str] = (),
    include_lm_head_lora: bool = False,
    rank: int,
    alpha: float,
    dropout: float,
    rank_explicit: bool = False,
    initial_low_rank_payloads: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> nn.Module:
    """Create at most one PEFT adapter for compressed and/or dense/lm_head targets."""
    if not compressed_modules and not dense_target_modules and not include_lm_head_lora:
        return model
    if compressed_modules or include_lm_head_lora or dense_target_modules:
        # Dense-only / lm_head-only still go through the shared builder when compressed
        # list is empty: build_full_compressed_peft_model supports that via target_modules.
        return build_full_compressed_peft_model(
            model,
            selected_modules=compressed_modules,
            initial_low_rank_payloads=initial_low_rank_payloads,
            rank=int(rank),
            alpha=float(alpha),
            dropout=float(dropout),
            rank_explicit=bool(rank_explicit),
            include_lm_head=bool(include_lm_head_lora),
            dense_target_modules=dense_target_modules,
        )
    return model


def apply_aux_trainables(
    model: nn.Module,
    aux: AuxTrainableConfig,
) -> Tuple[Dict[str, nn.Parameter], Dict[str, nn.Parameter]]:
    aux.validate()
    norm_params = enable_norm_trainables(model, norm_train_mode=aux.norm_train_mode)
    lm_head_params = setup_lm_head_trainables(model, lm_head_train_mode=aux.lm_head_train_mode)
    return norm_params, lm_head_params


def classify_peft_lora_parameters(
    model: nn.Module,
) -> Tuple[Dict[str, nn.Parameter], Dict[str, nn.Parameter]]:
    """Split one PEFT adapter's trainable LoRA params by logical target.

    - logical target ``lm_head`` -> lm_head LoRA inventory
    - all other logical targets (compressed carriers / remaining dense) -> backbone LoRA inventory

    Classification uses exact adapter layer inventory + ``_logical_adapter_target_name``,
    never a blanket ``\"lora_\" in name`` dump of the whole adapter.
    """
    backbone: Dict[str, nn.Parameter] = {}
    lm_head: Dict[str, nn.Parameter] = {}
    backbone_ids: Dict[int, str] = {}
    lm_head_ids: Dict[int, str] = {}

    for peft_name, lora_layer in iter_named_peft_lora_layers(model):
        logical = _logical_adapter_target_name(peft_name)
        is_lm_head = logical == "lm_head"
        dest = lm_head if is_lm_head else backbone
        seen_ids = lm_head_ids if is_lm_head else backbone_ids
        prefix = "lm_head_lora" if is_lm_head else "lora"
        for rel_name, param in lora_layer.named_parameters(recurse=True):
            if not bool(param.requires_grad):
                continue
            key = f"{prefix}::{logical}.{rel_name}" if not is_lm_head else f"{prefix}::{rel_name}"
            pid = id(param)
            if pid in seen_ids:
                continue
            if key in dest and dest[key] is not param:
                raise RuntimeError(f"duplicate LoRA key with different Parameter: {key}")
            dest[key] = param
            seen_ids[pid] = key
    return backbone, lm_head


def collect_lora_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    """Collect backbone/remaining LoRA parameters only (excludes lm_head logical target)."""
    backbone, _lm_head = classify_peft_lora_parameters(model)
    return backbone


def collect_lm_head_lora_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    """Collect lm_head logical-target LoRA parameters from the shared PEFT adapter."""
    _backbone, lm_head = classify_peft_lora_parameters(model)
    return lm_head


def build_model_level_trainable_selection(
    model: nn.Module,
    *,
    aux: AuxTrainableConfig,
    compressed_modules: Sequence[Tuple[str, VAELinear]] = (),
    dense_target_modules: Sequence[str] = (),
    decoder_modules: Optional[Sequence[Tuple[str, VAELinear]]] = None,
    rank: int,
    alpha: float,
    dropout: float,
    rank_explicit: bool = False,
    initial_low_rank_payloads: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    train_decoder: bool = False,
    train_lora: bool = True,
    decoder_execution_mode: str = "trainable_decoder",
    freeze: bool = True,
) -> ModelLevelTrainableSelection:
    """Freeze/enable trainables and return labeled inventories for Task 7 optimizer grouping.

    Inventories are filled by component ownership (not by parameter-name heuristics):
    - decoder_parameters
    - lora_parameters (compressed + remaining dense LoRA)
    - norm_parameters
    - lm_head_parameters (linear / full / lm_head LoRA)
    """
    aux.validate()
    if freeze:
        freeze_all_parameters(model)

    include_lm_head_lora = str(aux.lm_head_train_mode).strip().lower() == "lora"
    lora_compressed_modules = tuple(compressed_modules) if bool(train_lora) else ()
    lora_dense_targets = tuple(dense_target_modules) if bool(train_lora) else ()
    decoder_selected_modules = (
        tuple(compressed_modules)
        if decoder_modules is None
        else tuple(decoder_modules)
    )
    peft_model = enable_model_level_lora_targets(
        model,
        compressed_modules=lora_compressed_modules,
        dense_target_modules=lora_dense_targets,
        include_lm_head_lora=include_lm_head_lora,
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        rank_explicit=bool(rank_explicit),
        initial_low_rank_payloads=initial_low_rank_payloads if bool(train_lora) else None,
    )

    # PEFT freezes base parameters while injecting the adapter. Enable decoder
    # trainables only after adapter construction so decoder_lora modes keep the
    # intended requires_grad inventory.
    decoder_parameters: Dict[str, nn.Parameter] = {}
    if train_decoder and decoder_selected_modules:
        decoder_parameters = enable_decoder_trainables(
            peft_model,
            selected_modules=decoder_selected_modules,
            execution_mode=str(decoder_execution_mode),
        )

    norm_parameters, lm_head_non_lora = apply_aux_trainables(peft_model, aux)
    lora_parameters, lm_head_lora = classify_peft_lora_parameters(peft_model)

    lm_head_parameters: Dict[str, nn.Parameter] = {}
    lm_head_parameters.update(lm_head_non_lora)
    for key, param in lm_head_lora.items():
        if key in lm_head_parameters and lm_head_parameters[key] is not param:
            raise RuntimeError(f"lm_head inventory key conflict: {key}")
        lm_head_parameters[key] = param

    decoder_parameters = _dedupe_params_by_id(decoder_parameters, inventory_name="decoder_parameters")
    lora_parameters = _dedupe_params_by_id(lora_parameters, inventory_name="lora_parameters")
    norm_parameters = _dedupe_params_by_id(norm_parameters, inventory_name="norm_parameters")
    lm_head_parameters = _dedupe_params_by_id(lm_head_parameters, inventory_name="lm_head_parameters")
    assert_disjoint_component_inventories(
        decoder_parameters=decoder_parameters,
        lora_parameters=lora_parameters,
        norm_parameters=norm_parameters,
        lm_head_parameters=lm_head_parameters,
    )

    return ModelLevelTrainableSelection(
        decoder_parameters=decoder_parameters,
        lora_parameters=lora_parameters,
        norm_parameters=norm_parameters,
        lm_head_parameters=lm_head_parameters,
        compressed_lora_targets=[str(name) for name, _ in lora_compressed_modules],
        include_lm_head_lora=bool(include_lm_head_lora),
        peft_model=peft_model,
    )


def finalize_lm_head_linear_if_needed(model: nn.Module, *, lm_head_train_mode: str) -> bool:
    mode = str(lm_head_train_mode or "none").strip().lower()
    if mode != "linear":
        return False
    return bool(fuse_post_norm_head_linear(_base_model(model)))


# Re-export finalize helper for callers that already hold a PEFT model.
__all__ = [
    "NORM_TYPE_REGISTRY",
    "ModelLevelTrainableSelection",
    "apply_aux_trainables",
    "assert_disjoint_component_inventories",
    "build_model_level_trainable_selection",
    "classify_peft_lora_parameters",
    "collect_lm_head_lora_parameters",
    "collect_lora_parameters",
    "enable_decoder_trainables",
    "enable_model_level_lora_targets",
    "enable_norm_trainables",
    "finalize_lm_head_linear_if_needed",
    "finalize_model_level_lora",
    "freeze_all_parameters",
    "is_backbone_norm_module",
    "iter_named_backbone_norm_modules",
    "setup_lm_head_trainables",
]
