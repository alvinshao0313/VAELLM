from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from torch import nn

from e2e_fintuning.post_norm_head import resolve_post_norm_linear
from rotation.model_utils import get_model_type, get_pre_head_layernorm
from train_utils.utils import extract_layer_idx

try:
    from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise ImportError("未安装 peft。请先安装：pip install peft") from exc


@dataclass
class RawTrainableSelection:
    decoder_layer_ids: List[int]
    target_modules: List[str]
    target_module_suffixes: List[str]
    modules_to_save: List[str]
    final_norm_modules: List[str]
    post_norm_head_modules: List[str]
    trainable_parameter_names: List[str]
    trainable_parameter_count: int


def resolve_target_layer_ids(requested: Optional[Sequence[int]], num_layers: int) -> List[int]:
    if requested is None:
        return list(range(int(num_layers)))

    resolved = sorted(set(int(idx) for idx in requested))
    for idx in resolved:
        if idx < 0 or idx >= int(num_layers):
            raise ValueError(f"Invalid decoder layer id {idx}; valid range is [0, {int(num_layers) - 1}].")
    return resolved


def _find_module_name(model: nn.Module, target: nn.Module, fallback: str) -> str:
    for name, module in model.named_modules():
        if module is target:
            return str(name)
    return str(fallback)


def _resolve_lora_bias_mode(tune_bias: bool) -> str:
    return "lora_only" if bool(tune_bias) else "none"


def _resolve_variant_flags(variant: str) -> tuple[bool, bool]:
    norm = str(variant).strip().lower()
    return norm == "rslora", norm == "dora"


def _collect_selected_linear_modules(
    model: nn.Module,
    *,
    decoder_layer_ids: Sequence[int],
    target_module_names: Optional[Sequence[str]],
) -> tuple[List[str], List[str]]:
    selected_layers: Set[int] = {int(idx) for idx in decoder_layer_ids}
    selected_suffixes: Optional[Set[str]] = None
    if target_module_names is not None:
        selected_suffixes = {
            str(name).strip().lower()
            for name in target_module_names
            if str(name).strip()
        }

    target_full_names: List[str] = []
    target_suffixes: Set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        layer_idx = extract_layer_idx(name)
        if layer_idx is None or int(layer_idx) not in selected_layers:
            continue
        suffix = str(name).rsplit(".", 1)[-1].lower()
        if selected_suffixes is not None and suffix not in selected_suffixes:
            continue
        target_full_names.append(str(name))
        target_suffixes.add(suffix)

    dedup_target_names = sorted(set(target_full_names))
    if not dedup_target_names:
        raise ValueError("No eligible nn.Linear target modules found for requested decoder layers / target_modules.")
    return dedup_target_names, sorted(target_suffixes)


def _collect_modules_to_save(
    model: nn.Module,
    *,
    tune_final_norm: bool,
    use_post_norm_head_linear: bool,
) -> tuple[List[str], List[str], List[str]]:
    modules_to_save: List[str] = []
    final_norm_modules: List[str] = []
    post_norm_head_modules: List[str] = []

    if bool(tune_final_norm):
        model_type = get_model_type(model)
        final_norm = get_pre_head_layernorm(model, model_type)
        final_norm_name = _find_module_name(model, final_norm, "model.norm")
        final_norm_modules.append(final_norm_name)
        modules_to_save.append(final_norm_name)

    if bool(use_post_norm_head_linear):
        post_norm_linear = resolve_post_norm_linear(model)
        if post_norm_linear is None:
            raise ValueError(
                "--use_post_norm_head_linear=true but model.lm_head is not LMHeadWithPostNormLinear."
            )
        post_norm_name = _find_module_name(model, post_norm_linear, "lm_head.post_norm_linear")
        post_norm_head_modules.append(post_norm_name)
        modules_to_save.append(post_norm_name)

    return (
        sorted(set(modules_to_save)),
        sorted(set(final_norm_modules)),
        sorted(set(post_norm_head_modules)),
    )


def _build_peft_config(
    *,
    args,
    target_modules: Sequence[str],
    modules_to_save: Sequence[str],
    total_step: int,
):
    init_mode = str(getattr(args, "lora_init_mode", "zero")).strip().lower()
    init_lora_weights = "gaussian" if init_mode == "gaussian" else True
    bias_mode = _resolve_lora_bias_mode(bool(getattr(args, "lora_tune_bias", False)))
    variant = str(getattr(args, "lora_variant", "plain")).strip().lower()

    common = dict(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(getattr(args, "lora_rank")),
        target_modules=list(target_modules),
        lora_alpha=float(getattr(args, "lora_alpha")),
        lora_dropout=float(getattr(args, "lora_dropout")),
        bias=str(bias_mode),
        init_lora_weights=init_lora_weights,
        modules_to_save=None if len(modules_to_save) == 0 else list(modules_to_save),
    )

    if variant == "adalora":
        return AdaLoraConfig(
            **common,
            target_r=int(getattr(args, "adalora_target_r")),
            init_r=int(getattr(args, "adalora_init_r")),
            tinit=int(getattr(args, "adalora_tinit")),
            tfinal=int(getattr(args, "adalora_tfinal")),
            deltaT=int(getattr(args, "adalora_delta_t")),
            beta1=float(getattr(args, "adalora_beta1")),
            beta2=float(getattr(args, "adalora_beta2")),
            orth_reg_weight=float(getattr(args, "adalora_orth_reg_weight")),
            total_step=int(total_step),
        )

    use_rslora, use_dora = _resolve_variant_flags(variant)
    return LoraConfig(
        **common,
        use_rslora=bool(use_rslora),
        use_dora=bool(use_dora),
    )


def inject_raw_peft_adapters(
    model: nn.Module,
    *,
    args,
    decoder_layer_ids: Sequence[int],
    total_step: int,
):
    target_modules, target_module_suffixes = _collect_selected_linear_modules(
        model,
        decoder_layer_ids=decoder_layer_ids,
        target_module_names=getattr(args, "target_module_names", None),
    )
    modules_to_save, final_norm_modules, post_norm_head_modules = _collect_modules_to_save(
        model,
        tune_final_norm=bool(getattr(args, "tune_final_norm", False)),
        use_post_norm_head_linear=bool(getattr(args, "use_post_norm_head_linear", False)),
    )

    peft_config = _build_peft_config(
        args=args,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        total_step=int(total_step),
    )
    peft_model = get_peft_model(model, peft_config)

    trainable_parameter_names = sorted(
        name for name, param in peft_model.named_parameters() if bool(param.requires_grad)
    )
    trainable_parameter_count = int(
        sum(int(param.numel()) for _name, param in peft_model.named_parameters() if bool(param.requires_grad))
    )
    if trainable_parameter_count < 1:
        raise RuntimeError("No trainable parameters found after PEFT adapter injection.")

    selection = RawTrainableSelection(
        decoder_layer_ids=[int(idx) for idx in decoder_layer_ids],
        target_modules=list(target_modules),
        target_module_suffixes=list(target_module_suffixes),
        modules_to_save=list(modules_to_save),
        final_norm_modules=list(final_norm_modules),
        post_norm_head_modules=list(post_norm_head_modules),
        trainable_parameter_names=trainable_parameter_names,
        trainable_parameter_count=int(trainable_parameter_count),
    )
    return peft_model, selection
