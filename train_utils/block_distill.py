from contextlib import contextmanager
from dataclasses import dataclass
import re
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from peft import AdaLoraConfig, LoraConfig
from peft.mapping import inject_adapter_in_model
from torch import nn

from e2e_common.peft_proxy import (
    PeftVAELinearProxy,
    ensure_peft_vae_linear_proxy,
    initialize_peft_linear_from_residual_svd,
    iter_named_peft_vae_proxies,
    is_peft_adalora_linear,
    is_peft_lora_linear,
    is_peft_proxy_adapter_linear,
    materialize_peft_proxy_decoded_linears,
    update_peft_vae_proxy_adalora,
)
from e2e_common.temporary_mode import set_model_temporary
from litebsq.vae_linear import VAELinear
from litebsq.vae_linear_prewarm import (
    NamedVAELinearDecodeTarget,
    NamedVAELinearTarget,
    decode_named_vae_linear_weights,
    prime_named_vae_linear_cache,
)
from train_utils.hif4_act import applied_hif4_act


QWEN3_BLOCK_CATEGORIES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _normalize_target_categories(target_categories: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if target_categories is None:
        return tuple(QWEN3_BLOCK_CATEGORIES)
    seen = set()
    allowed = set(QWEN3_BLOCK_CATEGORIES)
    for item in target_categories:
        category = str(item)
        if category not in allowed:
            raise ValueError(
                f"Invalid block distill target category {category!r}. "
                f"Allowed values: {','.join(QWEN3_BLOCK_CATEGORIES)}."
            )
        if category in seen:
            raise ValueError(f"Duplicate block distill target category {category!r}.")
        seen.add(category)
    return tuple(category for category in QWEN3_BLOCK_CATEGORIES if category in seen)


@dataclass(frozen=True)
class BlockDistillConfig:
    steps: int
    seqlen: int
    rank: int
    lr: float
    lora_variant: str
    lora_alpha: float
    lora_dropout: float
    lora_bias: str
    lora_hif4_act: bool
    adalora_init_rank: int
    adalora_tinit: int
    adalora_tfinal: int
    adalora_delta_t: int
    adalora_beta1: float
    adalora_beta2: float
    adalora_orth_reg_weight: float
    alpha: float
    beta: float
    attn_query_chunk_size: int
    log_every: int
    device: str
    train_mode: str = "lora"
    decode_group_size: int = 8
    eps: float = 1e-6


def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    current = model
    for token in str(module_name).split("."):
        if not hasattr(current, token):
            raise ValueError(f"Failed to resolve module {module_name!r}: missing {token!r}.")
        current = getattr(current, token)
    if not isinstance(current, nn.Module):
        raise TypeError(f"Resolved object at {module_name!r} is not an nn.Module: {type(current)}")
    return current


def iter_named_block_vae_linears(
    model: nn.Module,
    layer_idx: int,
    *,
    target_categories: Optional[Sequence[str]] = None,
) -> Iterator[Tuple[str, VAELinear]]:
    prefix = f"model.layers.{int(layer_idx)}."
    categories = set(_normalize_target_categories(target_categories))
    for name, module in model.named_modules():
        if name.startswith(prefix) and isinstance(module, VAELinear):
            category = name.rsplit(".", 1)[-1]
            if category in categories:
                yield name, module


def iter_named_block_peft_proxies(
    model: nn.Module,
    layer_idx: int,
    *,
    target_categories: Optional[Sequence[str]] = None,
) -> Iterator[Tuple[str, PeftVAELinearProxy]]:
    prefix = f"model.layers.{int(layer_idx)}."
    categories = set(_normalize_target_categories(target_categories))
    skip_prefixes: List[str] = []
    for name, module in model.named_modules():
        if any(name == prefix_item or name.startswith(f"{prefix_item}.") for prefix_item in skip_prefixes):
            continue
        if not name.startswith(prefix) or not isinstance(module, PeftVAELinearProxy):
            continue
        category = name.rsplit(".", 1)[-1]
        if category not in categories:
            continue
        skip_prefixes.append(f"{name}.base_layer")
        skip_prefixes.append(f"{name}.per_decoded_linear")
        yield name, module


def validate_qwen3_model(model: nn.Module) -> None:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type != "qwen3":
        raise ValueError(f"block_vae_lora_train only supports Qwen3 in v1, got model_type={model_type!r}.")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) < 1:
        raise ValueError("Qwen3 model must expose model.layers.")


def validate_block_categories(model: nn.Module, layer_idx: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    prefix = f"model.layers.{int(layer_idx)}."
    for name, module in model.named_modules():
        if not name.startswith(prefix):
            continue
        category = name.rsplit(".", 1)[-1]
        if category in QWEN3_BLOCK_CATEGORIES and isinstance(module, (nn.Linear, VAELinear, PeftVAELinearProxy)):
            out[category] = name
    missing = [category for category in QWEN3_BLOCK_CATEGORIES if category not in out]
    if missing:
        raise ValueError(f"Layer {layer_idx} is missing target projections: {missing}")
    return out


def _iter_plain_named_vae_linears(model: nn.Module) -> Iterator[Tuple[str, VAELinear]]:
    skip_prefixes: List[str] = []
    for name, module in model.named_modules():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in skip_prefixes):
            continue
        if isinstance(module, PeftVAELinearProxy):
            skip_prefixes.append(f"{name}.base_layer")
            skip_prefixes.append(f"{name}.per_decoded_linear")
            continue
        if isinstance(module, VAELinear):
            yield str(name), module


@contextmanager
def prepare_block_eval_decoded_weights(
    *,
    model: nn.Module,
    eval_device: str,
    group_size: int,
    train_mode: str = "lora",
    logger=None,
) -> Iterator[Dict[str, int]]:
    if int(group_size) < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}.")
    mode = _normalize_block_distill_train_mode(str(train_mode))
    device = str(eval_device)
    model.to(device)

    proxy_refs = list(iter_named_peft_vae_proxies(model))
    plain_targets = [
        NamedVAELinearTarget(name=name, base_layer=module)
        for name, module in _iter_plain_named_vae_linears(model)
    ]
    merged_adapters: List[Tuple[str, nn.Module]] = []
    proxy_decode_flags: List[Tuple[PeftVAELinearProxy, bool]] = []
    plain_stats = {"total": 0, "warmed": 0, "skipped": 0, "failed": 0}
    proxy_stats = {"total": 0, "warmed": 0, "skipped": 0, "failed": 0}
    proxy_decode_skipped = False

    try:
        if plain_targets:
            plain_stats = prime_named_vae_linear_cache(
                plain_targets,
                clear_existing=True,
                group_size=int(group_size),
                compute_device=device,
                logger=logger,
            )
        if proxy_refs:
            if mode == "lora":
                unmaterialized = [
                    name
                    for name, proxy in proxy_refs
                    if not bool(getattr(proxy, "_dense_base_materialized", False))
                ]
                if unmaterialized:
                    raise RuntimeError(
                        "Block eval lora mode expects pre-materialized PEFT proxy dense bases, "
                        f"but these proxies are missing decoded weights: {unmaterialized}"
                    )
                proxy_stats = {
                    "total": int(len(proxy_refs)),
                    "warmed": 0,
                    "skipped": int(len(proxy_refs)),
                    "failed": 0,
                }
                proxy_decode_skipped = True
            else:
                proxy_stats = materialize_peft_proxy_decoded_linears(
                    model,
                    group_size=int(group_size),
                    compute_device=device,
                    logger=logger,
                    log_prefix="[block-eval-predecode] ",
                )
            for name, proxy in proxy_refs:
                proxy_decode_flags.append((proxy, bool(getattr(proxy, "_train_decoder_with_adapter", False))))
                proxy._train_decoder_with_adapter = False
                peft_linear = proxy.per_decoded_linear
                if not is_peft_proxy_adapter_linear(peft_linear):
                    continue
                if bool(getattr(peft_linear, "merged", False)):
                    raise RuntimeError(f"{name}: PEFT adapter is already merged before block eval predecode.")
                peft_linear.merge(safe_merge=True)
                if not bool(getattr(peft_linear, "merged", False)):
                    raise RuntimeError(f"{name}: PEFT adapter merge did not mark the module as merged.")
                merged_adapters.append((name, peft_linear))

        stats = {
            "plain_total": int(plain_stats.get("total", 0)),
            "plain_warmed": int(plain_stats.get("warmed", 0)),
            "proxy_total": int(proxy_stats.get("total", 0)),
            "proxy_warmed": int(proxy_stats.get("warmed", 0)),
            "merged_adapters": int(len(merged_adapters)),
            "group_size": int(group_size),
            "proxy_decode_skipped": int(bool(proxy_decode_skipped)),
        }
        if logger is not None:
            logger.info(
                "Block eval predecode ready: plain_total=%d plain_warmed=%d proxy_total=%d proxy_warmed=%d merged_adapters=%d proxy_decode_skipped=%s group_size=%d eval_device=%s train_mode=%s",
                int(stats["plain_total"]),
                int(stats["plain_warmed"]),
                int(stats["proxy_total"]),
                int(stats["proxy_warmed"]),
                int(stats["merged_adapters"]),
                str(bool(stats["proxy_decode_skipped"])).lower(),
                int(stats["group_size"]),
                device,
                mode,
            )
        yield stats
    finally:
        for _name, peft_linear in reversed(merged_adapters):
            if bool(getattr(peft_linear, "merged", False)):
                peft_linear.unmerge()
        for proxy, previous_flag in proxy_decode_flags:
            proxy._train_decoder_with_adapter = bool(previous_flag)


def relative_mse(student: torch.Tensor, teacher: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    student_f = student.float()
    teacher_f = teacher.float()
    return (student_f - teacher_f).pow(2).mean() / (teacher_f.pow(2).mean() + float(eps))


def _build_causal_inputs(model: nn.Module, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    qwen_model = model.model
    batch, seqlen = int(hidden_states.shape[0]), int(hidden_states.shape[1])
    cache_position = torch.arange(seqlen, device=hidden_states.device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0).expand(batch, -1)
    causal_mask = qwen_model._update_causal_mask(
        None,
        hidden_states,
        cache_position,
        None,
        True,
    )
    position_embeddings = qwen_model.rotary_emb(hidden_states, position_ids)
    return causal_mask, position_ids, position_embeddings


def _materialized_causal_mask(hidden_states: torch.Tensor) -> torch.Tensor:
    batch, seqlen = int(hidden_states.shape[0]), int(hidden_states.shape[1])
    dtype = hidden_states.dtype
    device = hidden_states.device
    min_dtype = torch.finfo(dtype).min
    mask = torch.full((seqlen, seqlen), min_dtype, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seqlen, seqlen).expand(batch, 1, seqlen, seqlen)


def run_qwen3_block(
    model: nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    *,
    output_attentions: bool = False,
) -> torch.Tensor:
    layer = model.model.layers[int(layer_idx)]
    causal_mask, position_ids, position_embeddings = _build_causal_inputs(model, hidden_states)
    outputs = layer(
        hidden_states,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=bool(output_attentions),
        use_cache=False,
        cache_position=torch.arange(int(hidden_states.shape[1]), device=hidden_states.device, dtype=torch.long),
        position_embeddings=position_embeddings,
    )
    return outputs[0]


@contextmanager
def capture_linear_io(module_by_name: Mapping[str, nn.Module]):
    captured: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs, output):
            if not inputs:
                raise RuntimeError(f"{name}: linear hook received no inputs.")
            captured[name] = (inputs[0].detach(), output.detach())
        return hook

    for name, module in module_by_name.items():
        handles.append(module.register_forward_hook(make_hook(str(name))))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if int(n_rep) == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, int(n_rep), slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * int(n_rep), slen, head_dim)


def _qk_states_for_attention(model: nn.Module, layer_idx: int, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = model.model.layers[int(layer_idx)]
    attn = layer.self_attn
    causal_mask, _position_ids, position_embeddings = _build_causal_inputs(model, hidden_states)
    if causal_mask is None:
        causal_mask = _materialized_causal_mask(hidden_states)
    attn_input = layer.input_layernorm(hidden_states)
    input_shape = attn_input.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    query_states = attn.q_norm(attn.q_proj(attn_input).view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(attn_input).view(hidden_shape)).transpose(1, 2)
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = _repeat_kv(key_states, int(attn.num_key_value_groups))
    return query_states, key_states, causal_mask


def attention_map_kl_loss(
    model: nn.Module,
    layer_idx: int,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    *,
    query_chunk_size: int,
    eps: float = 1e-6,
    hif4_controller=None,
) -> torch.Tensor:
    if int(query_chunk_size) < 1:
        raise ValueError(f"query_chunk_size must be >= 1, got {query_chunk_size}.")
    previous_hif4_enabled = None if hif4_controller is None else bool(getattr(hif4_controller, "enabled", False))
    set_model_temporary(model, False)
    try:
        if hif4_controller is not None:
            hif4_controller.enabled = False
        with torch.no_grad():
            teacher_q, teacher_k, teacher_mask = _qk_states_for_attention(model, layer_idx, teacher_hidden)
        set_model_temporary(model, True)
        if hif4_controller is not None:
            hif4_controller.enabled = True
        student_q, student_k, student_mask = _qk_states_for_attention(model, layer_idx, student_hidden)
    finally:
        if hif4_controller is not None:
            hif4_controller.enabled = bool(previous_hif4_enabled)
    scaling = float(model.model.layers[int(layer_idx)].self_attn.scaling)
    seqlen = int(student_q.shape[-2])
    kl_sum = student_q.new_zeros(())
    valid_count = student_q.new_zeros(())
    for start in range(0, seqlen, int(query_chunk_size)):
        end = min(start + int(query_chunk_size), seqlen)
        teacher_logits = torch.matmul(
            teacher_q[:, :, start:end, :],
            teacher_k.transpose(2, 3),
        ) * scaling
        teacher_logits = teacher_logits + teacher_mask[:, :, start:end, : teacher_k.shape[-2]]
        teacher_attn = F.softmax(teacher_logits, dim=-1, dtype=torch.float32).to(dtype=student_q.dtype)
        student_logits = torch.matmul(
            student_q[:, :, start:end, :],
            student_k.transpose(2, 3),
        ) * scaling
        student_logits = student_logits + student_mask[:, :, start:end, : student_k.shape[-2]]
        student_attn = F.softmax(student_logits, dim=-1, dtype=torch.float32).to(dtype=student_q.dtype)
        teacher_attn_f = teacher_attn.float()
        student_attn_f = student_attn.float()
        valid = teacher_attn_f > 0
        teacher_prob = teacher_attn_f.clamp_min(float(eps))
        student_prob = student_attn_f.clamp_min(float(eps))
        kl = teacher_prob * (teacher_prob.log() - student_prob.log())
        kl_per_query = kl.masked_fill(~valid, 0.0).sum(dim=-1)
        valid_query = valid.any(dim=-1)
        kl_sum = kl_sum + kl_per_query.masked_select(valid_query).sum()
        valid_count = valid_count + valid_query.sum().to(device=kl_sum.device, dtype=kl_sum.dtype)
    return kl_sum / valid_count.clamp_min(1.0)


def _resolve_proxy_base_linear(module_name: str, proxy: PeftVAELinearProxy) -> nn.Linear:
    decoded_linear = proxy.per_decoded_linear
    if is_peft_proxy_adapter_linear(decoded_linear):
        decoded_linear = decoded_linear.get_base_layer()
    if not isinstance(decoded_linear, nn.Linear):
        raise TypeError(f"Expected nn.Linear under '{module_name}.per_decoded_linear', got {type(decoded_linear)}")
    return decoded_linear


def _block_projection_module_name(layer_idx: int, category: str) -> str:
    parent = "self_attn" if category in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
    return f"model.layers.{int(layer_idx)}.{parent}.{category}"


def _block_peft_target_regex(layer_idx: int, target_categories: Optional[Sequence[str]] = None) -> str:
    categories = _normalize_target_categories(target_categories)
    if not categories:
        raise ValueError("block distill PEFT target categories cannot be empty.")
    names = [
        f"{_block_projection_module_name(int(layer_idx), category)}.per_decoded_linear"
        for category in categories
    ]
    return "(" + "|".join(re.escape(name) for name in names) + ")"


def _default_adapter_name(module: nn.Module) -> str:
    if hasattr(module, "lora_A") and "default" in getattr(module, "lora_A"):
        return "default"
    if hasattr(module, "lora_E") and "default" in getattr(module, "lora_E"):
        return "default"
    raise ValueError(f"PEFT module does not expose a default adapter: {type(module)}")


def _ensure_block_proxy_dense_bias(proxy_refs: Sequence[Tuple[str, PeftVAELinearProxy]]) -> int:
    created = 0
    for name, proxy in proxy_refs:
        decoded_linear = _resolve_proxy_base_linear(name, proxy)
        if decoded_linear.bias is not None:
            continue
        decoded_linear.bias = nn.Parameter(
            torch.zeros(
                (int(decoded_linear.out_features),),
                dtype=decoded_linear.weight.dtype,
                device=decoded_linear.weight.device,
            ),
            requires_grad=False,
        )
        created += 1
    return int(created)


@torch.no_grad()
def materialize_block_peft_proxy_decoded_linears(
    model: nn.Module,
    layer_idx: int,
    *,
    compute_device: str,
    group_size: int,
    target_categories: Optional[Sequence[str]] = None,
    logger=None,
) -> int:
    categories = _normalize_target_categories(target_categories)
    proxy_refs = list(iter_named_block_peft_proxies(model, int(layer_idx), target_categories=categories))
    if len(proxy_refs) != len(categories):
        raise ValueError(
            f"Layer {layer_idx}: expected {len(categories)} PEFT VAELinear proxies, got {len(proxy_refs)}."
        )
    targets = [
        NamedVAELinearDecodeTarget(
            name=name,
            base_layer=proxy.base_layer,
            target_dtype=_resolve_proxy_base_linear(name, proxy).weight.dtype,
            include_low_rank=False,
        )
        for name, proxy in proxy_refs
    ]
    decoded = decode_named_vae_linear_weights(
        targets,
        group_size=int(group_size),
        compute_device=torch.device(compute_device),
        logger=logger,
        respect_cache_policy=False,
    )
    decoded_by_name = {item.name: item.decoded_weight for item in decoded}
    materialized = 0
    for name, proxy in proxy_refs:
        if name not in decoded_by_name:
            raise RuntimeError(f"Missing grouped decode result for proxy '{name}'.")
        decoded_linear = _resolve_proxy_base_linear(name, proxy)
        decoded_linear.weight.copy_(
            decoded_by_name[name].to(device=decoded_linear.weight.device, dtype=decoded_linear.weight.dtype)
        )
        base_bias = proxy.base_layer.bias
        if decoded_linear.bias is None:
            if base_bias is not None:
                raise ValueError(f"Decoded linear under '{name}' is missing bias while base VAELinear has bias.")
        else:
            if base_bias is None:
                decoded_linear.bias.zero_()
            else:
                decoded_linear.bias.copy_(base_bias.detach().to(device=decoded_linear.bias.device, dtype=decoded_linear.bias.dtype))
        proxy.base_layer.clear_decoded_weight_cache()
        proxy._dense_base_materialized = True
        materialized += 1
    return int(materialized)


def wrap_block_vae_linears_as_peft_proxies(
    model: nn.Module,
    layer_idx: int,
    *,
    target_categories: Optional[Sequence[str]] = None,
) -> List[str]:
    categories = _normalize_target_categories(target_categories)
    named_modules = list(iter_named_block_vae_linears(model, int(layer_idx), target_categories=categories))
    if len(named_modules) != len(categories):
        raise ValueError(
            f"Layer {layer_idx}: expected {len(categories)} VAELinear modules, got {len(named_modules)}."
        )
    wrapped: List[str] = []
    for name, module in named_modules:
        ensure_peft_vae_linear_proxy(model, name, module)
        wrapped.append(name)
    return wrapped


def inject_block_peft_lora_adapters(
    model: nn.Module,
    layer_idx: int,
    *,
    config: BlockDistillConfig,
    target_categories: Optional[Sequence[str]] = None,
) -> List[str]:
    categories = _normalize_target_categories(target_categories)
    proxy_refs = list(iter_named_block_peft_proxies(model, int(layer_idx), target_categories=categories))
    if len(proxy_refs) != len(categories):
        raise ValueError(
            f"Layer {layer_idx}: expected {len(categories)} PEFT VAELinear proxies, got {len(proxy_refs)}."
        )
    target_regex = _block_peft_target_regex(int(layer_idx), categories)
    variant = str(config.lora_variant).strip().lower()
    if str(config.lora_bias) == "lora_only":
        _ensure_block_proxy_dense_bias(proxy_refs)
    if variant == "adalora":
        peft_config = AdaLoraConfig(
            task_type=None,
            r=int(config.adalora_init_rank),
            init_r=int(config.adalora_init_rank),
            target_r=int(config.rank),
            tinit=int(config.adalora_tinit),
            tfinal=int(config.adalora_tfinal),
            deltaT=int(config.adalora_delta_t),
            beta1=float(config.adalora_beta1),
            beta2=float(config.adalora_beta2),
            orth_reg_weight=float(config.adalora_orth_reg_weight),
            total_step=int(config.steps),
            lora_alpha=float(config.lora_alpha),
            lora_dropout=float(config.lora_dropout),
            target_modules=target_regex,
            bias=str(config.lora_bias),
            inference_mode=False,
            init_lora_weights=True,
        )
    else:
        peft_config = LoraConfig(
            task_type=None,
            r=int(config.rank),
            lora_alpha=float(config.lora_alpha),
            lora_dropout=float(config.lora_dropout),
            target_modules=target_regex,
            bias=str(config.lora_bias),
            inference_mode=False,
            use_rslora=variant == "rslora",
            use_dora=variant == "dora",
            init_lora_weights=True,
        )
    inject_adapter_in_model(peft_config, model, adapter_name="default")

    injected_names: List[str] = []
    with torch.no_grad():
        for name, proxy in proxy_refs:
            peft_linear = proxy.per_decoded_linear
            if variant == "adalora":
                if not is_peft_adalora_linear(peft_linear):
                    raise RuntimeError(f"Failed to inject AdaLoRA into '{name}.per_decoded_linear'.")
                adapter_name = _default_adapter_name(peft_linear)
                peft_linear.lora_E[adapter_name].zero_()
                if torch.count_nonzero(peft_linear.get_delta_weight(adapter_name)).item() != 0:
                    raise RuntimeError(f"Failed to zero-initialize AdaLoRA delta at '{name}'.")
            else:
                if not is_peft_lora_linear(peft_linear):
                    raise RuntimeError(f"Failed to inject LoRA into '{name}.per_decoded_linear'.")
                decoded_weight = peft_linear.get_base_layer().weight.detach()
                original_weight = proxy.base_layer.original_weight
                if original_weight is None:
                    raise RuntimeError(f"{name}: original_weight is required for residual SVD LoRA init.")
                residual = original_weight.detach().to(device=decoded_weight.device, dtype=torch.float32) - decoded_weight.to(dtype=torch.float32)
                initialize_peft_linear_from_residual_svd(peft_linear, residual, module_name=name)
            injected_names.append(name)
    return injected_names


def _normalize_block_distill_train_mode(train_mode: str) -> str:
    mode = str(train_mode or "lora").strip().lower()
    if mode not in {"lora", "decoder", "both"}:
        raise ValueError("block_distill_train_mode must be one of: lora | decoder | both.")
    return mode


def _resolve_block_base_layer(module_name: str, module: nn.Module) -> VAELinear:
    if isinstance(module, PeftVAELinearProxy):
        return module.base_layer
    if isinstance(module, VAELinear):
        return module
    raise TypeError(f"{module_name}: expected VAELinear or PeftVAELinearProxy, got {type(module)}")


def _enable_only_decoder_params(module: VAELinear) -> List[nn.Parameter]:
    trainable: List[nn.Parameter] = []
    packed_decoder = getattr(module, "_parallel_stage_decoder", None)
    if packed_decoder is not None:
        for param in packed_decoder.parameters():
            param.requires_grad = True
            trainable.append(param)
        return trainable

    for stage_idx in range(int(module.residual_stages)):
        for part_idx in range(int(module.parallel_parts)):
            decoder = module.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)
            for param in decoder.parameters():
                param.requires_grad = True
                trainable.append(param)
    return trainable


def _collect_block_modules_for_decoder(
    model: nn.Module,
    layer_idx: int,
    *,
    target_categories: Optional[Sequence[str]] = None,
) -> List[str]:
    categories = _normalize_target_categories(target_categories)
    proxy_refs = list(iter_named_block_peft_proxies(model, int(layer_idx), target_categories=categories))
    if proxy_refs:
        if len(proxy_refs) != len(categories):
            raise ValueError(
                f"Layer {layer_idx}: expected {len(categories)} PEFT VAELinear proxies, got {len(proxy_refs)}."
            )
        return [name for name, _proxy in proxy_refs]

    vae_refs = list(iter_named_block_vae_linears(model, int(layer_idx), target_categories=categories))
    if len(vae_refs) != len(categories):
        raise ValueError(
            f"Layer {layer_idx}: expected {len(categories)} VAELinear modules, got {len(vae_refs)}."
        )
    return [name for name, _module in vae_refs]


def freeze_except_block_distill_trainables(
    model: nn.Module,
    module_names: Iterable[str],
    *,
    train_mode: str,
    lora_bias: str,
) -> List[nn.Parameter]:
    mode = _normalize_block_distill_train_mode(train_mode)
    for param in model.parameters():
        param.requires_grad = False

    trainable: List[nn.Parameter] = []
    for name in module_names:
        module = get_module_by_name(model, str(name))
        if mode in {"decoder", "both"}:
            base_layer = _resolve_block_base_layer(str(name), module)
            base_layer.enable_trainable_decode(parallel_stage_decode=True)
            trainable.extend(_enable_only_decoder_params(base_layer))
        if mode in {"lora", "both"}:
            if not isinstance(module, PeftVAELinearProxy):
                raise TypeError(f"{name}: expected PeftVAELinearProxy for LoRA distill, got {type(module)}")
            peft_linear = module.per_decoded_linear
            if not is_peft_proxy_adapter_linear(peft_linear):
                raise TypeError(f"{name}: expected PEFT adapter linear, got {type(peft_linear)}")
            for param_name, param in peft_linear.named_parameters():
                if param_name == "base_layer.weight":
                    continue
                if param_name == "base_layer.bias" and str(lora_bias) != "lora_only":
                    continue
                param.requires_grad = True
                trainable.append(param)

    if not trainable:
        raise RuntimeError("No trainable parameters were selected for block distill.")
    return trainable


def _set_block_proxy_decoder_adapter_mode(model: nn.Module, module_names: Iterable[str], enabled: bool) -> None:
    for name in module_names:
        module = get_module_by_name(model, str(name))
        if isinstance(module, PeftVAELinearProxy):
            module._train_decoder_with_adapter = bool(enabled)


def _finalize_block_decoder_trainables(model: nn.Module, module_names: Iterable[str]) -> int:
    finalized = 0
    for name in module_names:
        module = get_module_by_name(model, str(name))
        base_layer = _resolve_block_base_layer(str(name), module)
        base_layer.unpack_parallel_stage_decoder_()
        base_layer.disable_trainable_decode()
        finalized += 1
    return int(finalized)


def train_block_lora_distill(
    *,
    model: nn.Module,
    layer_idx: int,
    teacher_hiddens_cpu: Sequence[torch.Tensor],
    student_hiddens_cpu: Sequence[torch.Tensor],
    config: BlockDistillConfig,
    target_categories: Optional[Sequence[str]] = None,
    logger=None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    categories = _normalize_target_categories(target_categories)
    if not categories:
        raise ValueError("block distill target categories cannot be empty.")
    device = torch.device(config.device)
    layer = model.model.layers[int(layer_idx)].to(device)
    layer.eval()
    train_mode = _normalize_block_distill_train_mode(str(config.train_mode))
    use_lora = train_mode in {"lora", "both"}
    use_decoder = train_mode in {"decoder", "both"}
    if use_lora:
        module_names = wrap_block_vae_linears_as_peft_proxies(
            model,
            int(layer_idx),
            target_categories=categories,
        )
        materialized = materialize_block_peft_proxy_decoded_linears(
            model,
            int(layer_idx),
            compute_device=str(device),
            group_size=int(config.decode_group_size),
            target_categories=categories,
            logger=logger,
        )
        if logger is not None:
            logger.info("[block %d] materialized decoded PEFT proxy bases: %d", int(layer_idx), int(materialized))
        injected_names = inject_block_peft_lora_adapters(
            model,
            int(layer_idx),
            config=config,
            target_categories=categories,
        )
        if sorted(injected_names) != sorted(module_names):
            raise RuntimeError(f"Layer {layer_idx}: injected PEFT target names do not match wrapped names.")
    else:
        module_names = _collect_block_modules_for_decoder(
            model,
            int(layer_idx),
            target_categories=categories,
        )
    _set_block_proxy_decoder_adapter_mode(model, module_names, enabled=train_mode == "both")
    trainable = freeze_except_block_distill_trainables(
        model,
        module_names,
        train_mode=train_mode,
        lora_bias=str(config.lora_bias),
    )
    optimizer = torch.optim.AdamW(trainable, lr=float(config.lr), weight_decay=0.0)
    module_by_name = {name: get_module_by_name(model, name) for name in module_names}
    num_samples = len(student_hiddens_cpu)
    if num_samples < 1:
        raise ValueError("block distill requires at least one calibration hidden sample.")
    attn_weight = float(config.alpha)
    linear_weight = float(config.beta)
    hidden_weight = 1.0 - attn_weight - linear_weight

    for step in range(int(config.steps)):
        sample_idx = int(step) % int(num_samples)
        teacher_in = teacher_hiddens_cpu[sample_idx].to(device=device, non_blocking=True)
        student_in = student_hiddens_cpu[sample_idx].to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        set_model_temporary(model, False)
        needs_teacher_block = linear_weight > 0.0 or hidden_weight > 0.0
        if linear_weight > 0.0:
            with torch.no_grad(), capture_linear_io(module_by_name) as teacher_io:
                teacher_out = run_qwen3_block(model, int(layer_idx), teacher_in, output_attentions=False).detach()
            if len(teacher_io) != len(module_names):
                missing = sorted(set(module_names) - set(teacher_io.keys()))
                raise RuntimeError(f"Layer {layer_idx}: missing teacher linear captures: {missing}")
        elif needs_teacher_block:
            teacher_io = {}
            with torch.no_grad():
                teacher_out = run_qwen3_block(model, int(layer_idx), teacher_in, output_attentions=False).detach()
        else:
            teacher_io = {}
            teacher_out = None

        hif4_logger = logger if (logger is not None and step == 0 and bool(config.lora_hif4_act)) else None
        with applied_hif4_act(
            model,
            enabled=bool(config.lora_hif4_act),
            logger=hif4_logger,
            log_prefix=f"[block {int(layer_idx)} {train_mode}] ",
        ) as hif4_ctx:
            hif4_controller = hif4_ctx.get("controller")
            set_model_temporary(model, True)
            if linear_weight > 0.0:
                linear_losses = []
                for name in module_names:
                    local_in, local_teacher_out = teacher_io[name]
                    student_local = module_by_name[name](local_in.to(device=device, non_blocking=True))
                    linear_losses.append(
                        relative_mse(
                            student_local,
                            local_teacher_out.to(device=device, non_blocking=True),
                            eps=config.eps,
                        )
                    )
                linear_loss = torch.stack(linear_losses).mean()
            else:
                linear_loss = student_in.new_zeros(())

            if hidden_weight > 0.0:
                if teacher_out is None:
                    raise RuntimeError("hidden loss requires teacher block output.")
                student_out = run_qwen3_block(model, int(layer_idx), student_in, output_attentions=False)
                hidden_loss = relative_mse(student_out, teacher_out, eps=config.eps)
            else:
                student_out = None
                hidden_loss = student_in.new_zeros(())
            if attn_weight > 0.0:
                attn_loss = attention_map_kl_loss(
                    model,
                    int(layer_idx),
                    student_in,
                    teacher_in,
                    query_chunk_size=int(config.attn_query_chunk_size),
                    eps=config.eps,
                    hif4_controller=hif4_controller,
                )
            else:
                attn_loss = student_in.new_zeros(())
        loss = attn_weight * attn_loss + linear_weight * linear_loss + hidden_weight * hidden_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if use_lora and str(config.lora_variant).strip().lower() == "adalora":
            update_peft_vae_proxy_adalora(model, global_step=int(step + 1))

        if logger is not None and (step + 1 == 1 or (step + 1) % int(config.log_every) == 0 or step + 1 == int(config.steps)):
            logger.info(
                "[block %d] distill step=%d/%d loss=%.6e attn_kl=%.6e linear=%.6e hidden=%.6e",
                int(layer_idx),
                int(step + 1),
                int(config.steps),
                float(loss.detach().cpu()),
                float(attn_loss.detach().cpu()),
                float(linear_loss.detach().cpu()),
                float(hidden_loss.detach().cpu()),
            )

        del teacher_in, student_in, teacher_out, student_out, loss, linear_loss, attn_loss, hidden_loss

    if logger is not None:
        if use_lora:
            logger.info("[block %d] kept PEFT proxy adapters for final checkpoint: %d", int(layer_idx), int(len(module_names)))
        if use_decoder:
            finalized = _finalize_block_decoder_trainables(model, module_names)
            logger.info("[block %d] finalized trainable decoder modules: %d", int(layer_idx), int(finalized))
    elif use_decoder:
        _finalize_block_decoder_trainables(model, module_names)

    next_teacher: List[torch.Tensor] = []
    next_student: List[torch.Tensor] = []
    layer.eval()
    with torch.no_grad():
        with applied_hif4_act(model, enabled=bool(config.lora_hif4_act), require_targets=False) as hif4_ctx:
            hif4_controller = hif4_ctx.get("controller")
            for teacher_cpu, student_cpu in zip(teacher_hiddens_cpu, student_hiddens_cpu):
                teacher_in = teacher_cpu.to(device=device, non_blocking=True)
                student_in = student_cpu.to(device=device, non_blocking=True)
                if hif4_controller is not None:
                    hif4_controller.enabled = False
                set_model_temporary(model, False)
                teacher_next = run_qwen3_block(model, int(layer_idx), teacher_in, output_attentions=False)
                if hif4_controller is not None:
                    hif4_controller.enabled = True
                set_model_temporary(model, True)
                student_next = run_qwen3_block(model, int(layer_idx), student_in, output_attentions=False)
                next_teacher.append(teacher_next.detach().to(device="cpu", dtype=torch.bfloat16).contiguous())
                next_student.append(student_next.detach().to(device="cpu", dtype=torch.bfloat16).contiguous())
    layer.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return next_teacher, next_student


@torch.no_grad()
def build_initial_hidden_states(
    model: nn.Module,
    input_id_blocks: Sequence[torch.Tensor],
    *,
    device: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    embed = model.model.embed_tokens.to(device)
    hiddens: List[torch.Tensor] = []
    for block in input_id_blocks:
        input_ids = block.to(device=device, dtype=torch.long, non_blocking=True)
        hidden = embed(input_ids).detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        hiddens.append(hidden)
    embed.to("cpu")
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return list(hiddens), [item.clone() for item in hiddens]


def validate_final_block_checkpoint(
    model: nn.Module,
    *,
    expected_rank: int,
    expected_init_rank: int,
    expected_count: int,
    lora_variant: str,
    train_mode: str = "lora",
) -> None:
    mode = _normalize_block_distill_train_mode(train_mode)
    adapter_count = 0
    for _name, module in model.named_modules():
        if not isinstance(module, PeftVAELinearProxy):
            continue
        category = _name.rsplit(".", 1)[-1]
        if category not in QWEN3_BLOCK_CATEGORIES:
            continue
        peft_linear = module.per_decoded_linear
        has_adapter = is_peft_adalora_linear(peft_linear) or is_peft_lora_linear(peft_linear)
        if mode == "decoder":
            if has_adapter:
                raise RuntimeError(f"{_name}: decoder-only block distill must not leave a PEFT adapter.")
            continue
        adapter_count += 1
        variant = str(lora_variant).strip().lower()
        if variant == "adalora":
            if not is_peft_adalora_linear(peft_linear):
                raise RuntimeError(f"{_name}: expected AdaLoRA PEFT proxy adapter, got {type(peft_linear)}.")
            adapter_name = _default_adapter_name(peft_linear)
            if int(peft_linear.r[adapter_name]) != int(expected_init_rank):
                raise RuntimeError(f"{_name}: AdaLoRA init rank mismatch, expected {expected_init_rank}.")
        else:
            if not is_peft_lora_linear(peft_linear):
                raise RuntimeError(f"{_name}: expected LoRA-family PEFT proxy adapter, got {type(peft_linear)}.")
            adapter_name = _default_adapter_name(peft_linear)
            if int(peft_linear.r[adapter_name]) != int(expected_rank):
                raise RuntimeError(f"{_name}: LoRA rank mismatch, expected {expected_rank}.")
    if mode == "decoder":
        return
    if int(adapter_count) != int(expected_count):
        raise RuntimeError(f"Final PEFT adapter count mismatch: got {adapter_count}, expected {expected_count}.")
