from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from torch import nn

from e2e_common.post_norm_head import resolve_post_norm_linear
from litebsq.vae_linear import VAELinear
from rotation.model_utils import get_model_type, get_pre_head_layernorm
from train_utils.utils import extract_layer_idx


@dataclass
class VAEDecoderTrainableSelection:
    decoder_layer_ids: List[int]
    target_modules: List[str]
    target_module_suffixes: List[str]
    bias_modules: List[str]
    final_norm_modules: List[str]
    post_norm_head_modules: List[str]
    low_rank_modules: List[str]
    trainable_parameter_names: List[str]
    trainable_parameter_count: int
    parallel_stage_decode: bool
    train_mode: str


def resolve_target_layer_ids(requested: Optional[Sequence[int]], num_layers: int) -> List[int]:
    if requested is None:
        return list(range(int(num_layers)))
    resolved = sorted(set(int(idx) for idx in requested))
    for idx in resolved:
        if idx < 0 or idx >= int(num_layers):
            raise ValueError(f"Invalid decoder layer id {idx}; valid range is [0, {int(num_layers) - 1}].")
    return resolved


def _iter_named_vae_linears(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, VAELinear):
            yield str(name), module


def collect_selected_vae_linears(
    model: nn.Module,
    *,
    decoder_layer_ids: Sequence[int],
    target_module_names: Optional[Sequence[str]],
) -> Tuple[List[Tuple[str, VAELinear]], List[str]]:
    selected_layers: Set[int] = {int(idx) for idx in decoder_layer_ids}
    selected_suffixes: Optional[Set[str]] = None
    if target_module_names is not None:
        selected_suffixes = {
            str(name).strip().lower()
            for name in target_module_names
            if str(name).strip()
        }

    target_modules: List[Tuple[str, VAELinear]] = []
    target_suffixes: Set[str] = set()
    for name, module in _iter_named_vae_linears(model):
        layer_idx = extract_layer_idx(name)
        if layer_idx is None or int(layer_idx) not in selected_layers:
            continue
        suffix = str(name).rsplit(".", 1)[-1].lower()
        if selected_suffixes is not None and suffix not in selected_suffixes:
            continue
        target_modules.append((name, module))
        target_suffixes.add(suffix)
    return target_modules, sorted(target_suffixes)


def _find_module_name(model: nn.Module, target: nn.Module, fallback: str) -> str:
    for name, module in model.named_modules():
        if module is target:
            return str(name)
    return str(fallback)


def _freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for _name, module in _iter_named_vae_linears(model):
        module.cache_decoded_weight = True
        module.trainable_decode = False
        module.parallel_stage_decode = False
        module.clear_decoded_weight_cache()


def _enable_only_decoder_params(module: VAELinear) -> None:
    packed_decoder = getattr(module, "_parallel_stage_decoder", None)
    if packed_decoder is not None:
        for param in packed_decoder.parameters():
            param.requires_grad = True
        return

    for stage_idx in range(int(module.residual_stages)):
        for part_idx in range(int(module.parallel_parts)):
            decoder = module.get_stage_part_decoder(stage_idx=stage_idx, part_idx=part_idx)
            for param in decoder.parameters():
                param.requires_grad = True


def _enable_module_params(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def _low_rank_rank(module: VAELinear, *, name: str) -> int:
    low_rank_a = getattr(module, "low_rank_a", None)
    low_rank_b = getattr(module, "low_rank_b", None)
    if low_rank_a is None or low_rank_b is None:
        raise ValueError(f"{name}: selected VAELinear has no complete low_rank_a/low_rank_b payload.")
    if int(low_rank_a.ndim) != 2 or int(low_rank_b.ndim) != 2:
        raise ValueError(f"{name}: low_rank_a/low_rank_b must be 2D tensors.")
    if int(low_rank_a.shape[1]) != int(low_rank_b.shape[0]):
        raise ValueError(
            f"{name}: low rank inner dim mismatch: {int(low_rank_a.shape[1])} != {int(low_rank_b.shape[0])}."
        )
    if int(low_rank_a.numel()) < 1 or int(low_rank_b.numel()) < 1:
        raise ValueError(f"{name}: low_rank_a/low_rank_b cannot be empty.")
    return int(low_rank_a.shape[1])


def validate_selected_low_rank_payloads(
    selected_modules: Sequence[Tuple[str, VAELinear]],
    *,
    require_uniform_rank: bool,
) -> int:
    ranks = []
    for name, module in selected_modules:
        ranks.append(_low_rank_rank(module, name=name))
    unique_ranks = sorted(set(ranks))
    if not unique_ranks:
        raise ValueError("No selected VAELinear modules found for low-rank training.")
    if bool(require_uniform_rank) and len(unique_ranks) != 1:
        raise ValueError(f"--finetune_mode lora requires uniform low-rank rank, got {unique_ranks}.")
    return int(unique_ranks[0])


def _enable_low_rank_params(module: VAELinear, *, name: str) -> None:
    _low_rank_rank(module, name=name)
    module.low_rank_a.requires_grad = True
    module.low_rank_b.requires_grad = True


def select_vae_decoder_trainables(
    model: nn.Module,
    *,
    decoder_layer_ids: Sequence[int],
    target_module_names: Optional[Sequence[str]],
    parallel_stage_decode: bool,
    tune_final_norm: bool = False,
    use_post_norm_head_linear: bool = False,
    vae_tune_bias: bool = False,
    train_mode: str = "decoder",
) -> VAEDecoderTrainableSelection:
    _freeze_all(model)
    train_mode = str(train_mode or "decoder").strip().lower()
    if train_mode not in {"decoder", "low_rank", "both"}:
        raise ValueError(f"Invalid train_mode={train_mode!r}.")

    selected_modules, target_module_suffixes = collect_selected_vae_linears(
        model,
        decoder_layer_ids=decoder_layer_ids,
        target_module_names=target_module_names,
    )
    target_modules: List[str] = []
    bias_modules: List[str] = []
    low_rank_modules: List[str] = []
    for name, module in selected_modules:
        if train_mode in {"decoder", "both"}:
            module.enable_trainable_decode(parallel_stage_decode=bool(parallel_stage_decode))
            _enable_only_decoder_params(module)
        if bool(vae_tune_bias) and module.bias is not None:
            module.bias.requires_grad = True
            bias_modules.append(name)
        if train_mode in {"low_rank", "both"}:
            if train_mode == "low_rank":
                module.trainable_decode = True
                module.cache_decoded_weight = False
                module.clear_decoded_weight_cache()
            _enable_low_rank_params(module, name=name)
            low_rank_modules.append(name)
        target_modules.append(name)

    if not target_modules:
        raise ValueError("No eligible VAELinear modules found for requested decoder_layers / target_modules.")

    final_norm_modules: List[str] = []
    if bool(tune_final_norm):
        model_type = get_model_type(model)
        final_norm = get_pre_head_layernorm(model, model_type)
        _enable_module_params(final_norm)
        final_norm_modules.append(_find_module_name(model, final_norm, "model.norm"))

    post_norm_head_modules: List[str] = []
    if bool(use_post_norm_head_linear):
        post_norm_linear = resolve_post_norm_linear(model)
        if post_norm_linear is None:
            raise ValueError(
                "--use_post_norm_head_linear=true but model.lm_head is not LMHeadWithPostNormLinear."
            )
        _enable_module_params(post_norm_linear)
        post_norm_head_modules.append(_find_module_name(model, post_norm_linear, "lm_head.post_norm_linear"))

    trainable_names = sorted(name for name, param in model.named_parameters() if bool(param.requires_grad))
    trainable_count = int(sum(int(param.numel()) for _name, param in model.named_parameters() if bool(param.requires_grad)))
    if trainable_count < 1:
        raise RuntimeError("No trainable decoder parameters found.")

    return VAEDecoderTrainableSelection(
        decoder_layer_ids=[int(idx) for idx in decoder_layer_ids],
        target_modules=sorted(set(target_modules)),
        target_module_suffixes=list(target_module_suffixes),
        bias_modules=sorted(set(bias_modules)),
        final_norm_modules=sorted(set(final_norm_modules)),
        post_norm_head_modules=sorted(set(post_norm_head_modules)),
        low_rank_modules=sorted(set(low_rank_modules)),
        trainable_parameter_names=trainable_names,
        trainable_parameter_count=trainable_count,
        parallel_stage_decode=bool(parallel_stage_decode),
        train_mode=train_mode,
    )


def unpack_parallel_stage_decoders(model: nn.Module) -> int:
    count = 0
    for _name, module in _iter_named_vae_linears(model):
        if module.unpack_parallel_stage_decoder_():
            count += 1
    return count
