from contextlib import contextmanager, nullcontext
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from train_utils.distill_losses import (
    build_distill_token_mask,
    compute_dual_kl_loss,
    compute_dual_kl_topk_loss,
    compute_dual_rkl_loss,
    compute_dual_rkl_topk_loss,
    compute_eakld,
    compute_eakld_topk,
    compute_entropy_aware_kl_loss,
    compute_forward_kl_loss,
    compute_kl_topk,
    compute_reverse_kl_loss,
    compute_rkl_topk,
    is_eakld_top_loss,
    parse_eakld_top_k,
)
from train_utils.hif4_act import Hif4ActController

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None

try:
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
except ImportError:
    DataCollatorForCompletionOnlyLM = None
    SFTTrainer = None


_DISTILL_HIDDEN_LAYER_WEIGHTING_STATIC_CHOICES = ("uniform", "linear_depth")
_DISTILL_HIDDEN_LAYER_WEIGHTING_CHOICES = (
    "uniform",
    "linear_depth",
    "adaptive",
    "adaptive_top_<K>",
)
_DEFAULT_ADAPTIVE_TOPK = 3


def parse_distill_hidden_alignment_layer_weighting(raw: str) -> str:
    mode = str(raw).strip().lower()
    if mode in _DISTILL_HIDDEN_LAYER_WEIGHTING_STATIC_CHOICES:
        return mode
    if mode == "adaptive":
        return mode
    if mode.startswith("adaptive_top"):
        suffix = mode[len("adaptive_top") :]
        if suffix.startswith("_"):
            suffix = suffix[1:]
        if not suffix.isdigit() or int(suffix) < 1:
            raise ValueError(
                f"Invalid --distill_hidden_alignment_layer_weighting: {raw!r}. "
                "adaptive_top suffix must be a positive integer, e.g. adaptive_top_3."
            )
        return mode
    raise ValueError(
        f"Invalid --distill_hidden_alignment_layer_weighting: {raw!r}. "
        "Supported: uniform, linear_depth, adaptive, adaptive_top_<K>."
    )


def is_adaptive_hidden_alignment_layer_weighting(layer_weighting: str) -> bool:
    mode = str(layer_weighting).strip().lower()
    return mode == "adaptive" or mode.startswith("adaptive_top")


def parse_adaptive_hidden_alignment_topk(layer_weighting: str, default_k: int = _DEFAULT_ADAPTIVE_TOPK) -> int:
    mode = str(layer_weighting).strip().lower()
    if not is_adaptive_hidden_alignment_layer_weighting(mode):
        raise ValueError(
            f"parse_adaptive_hidden_alignment_topk expects adaptive layer weighting, got {layer_weighting!r}."
        )
    if mode == "adaptive":
        return max(1, int(default_k))
    suffix = mode[len("adaptive_top") :]
    if suffix.startswith("_"):
        suffix = suffix[1:]
    if not suffix:
        raise ValueError(
            f"Invalid adaptive layer weighting: {layer_weighting!r}. "
            "Use adaptive or adaptive_top_<K> with a positive integer K."
        )
    return max(1, int(suffix))


def build_distill_hidden_layer_weights(
    *,
    num_layers: int,
    layer_weighting: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_layers = int(num_layers)
    if num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {num_layers}.")
    mode = str(layer_weighting).strip().lower()
    if is_adaptive_hidden_alignment_layer_weighting(mode):
        raise ValueError(
            "adaptive layer weighting must not use build_distill_hidden_layer_weights; "
            "use _aggregate_hidden_alignment_layer_losses instead."
        )
    if mode == "uniform":
        return torch.ones(num_layers, device=device, dtype=dtype)
    if mode == "linear_depth":
        denom = max(num_layers - 1, 1)
        raw = 1.0 + torch.arange(num_layers, device=device, dtype=dtype) / float(denom)
        return raw / raw.mean()
    raise ValueError(
        f"Unsupported distill hidden layer weighting: {layer_weighting}. "
        f"Supported: {', '.join(_DISTILL_HIDDEN_LAYER_WEIGHTING_STATIC_CHOICES)}."
    )


def _masked_mean_cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    a = a.float().detach()
    b = b.float().detach()
    cos = F.cosine_similarity(a, b, dim=-1)
    if attention_mask is None:
        return cos.mean()
    mask = attention_mask.to(device=cos.device, dtype=cos.dtype)
    while mask.ndim < cos.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(cos)
    count = mask.sum().clamp_min(1.0)
    return (cos * mask).sum() / count


def _select_adaptive_hidden_layer_indices(
    teacher_sequence: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    topk: int,
    *,
    reference_hidden: Optional[torch.Tensor] = None,
) -> List[int]:
    num_layers = len(teacher_sequence)
    if num_layers <= 0:
        raise ValueError("teacher_sequence must be non-empty for adaptive layer selection.")
    topk = min(max(1, int(topk)), num_layers)

    scores: List[Tuple[int, float]] = []
    for layer_idx in range(num_layers):
        hidden = teacher_sequence[layer_idx]
        if layer_idx == 0:
            if reference_hidden is None:
                raise ValueError("reference_hidden is required for adaptive selection at layer 0.")
            previous = reference_hidden
        else:
            previous = teacher_sequence[layer_idx - 1]
        cosine = _masked_mean_cosine_similarity(hidden, previous, attention_mask)
        scores.append((layer_idx, float(cosine.item())))

    selected = sorted(scores, key=lambda item: item[1])[:topk]
    return [layer_idx for layer_idx, _ in selected]


def _aggregate_hidden_alignment_layer_losses(
    layer_losses: List[torch.Tensor],
    layer_weighting: str,
    *,
    teacher_sequence_for_selection: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    reference_hidden: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    stacked = torch.stack(layer_losses)
    if is_adaptive_hidden_alignment_layer_weighting(layer_weighting):
        topk = parse_adaptive_hidden_alignment_topk(layer_weighting)
        selected = _select_adaptive_hidden_layer_indices(
            teacher_sequence_for_selection,
            attention_mask,
            topk,
            reference_hidden=reference_hidden,
        )
        return stacked[selected].mean()

    weights = build_distill_hidden_layer_weights(
        num_layers=len(layer_losses),
        layer_weighting=layer_weighting,
        device=stacked.device,
        dtype=stacked.dtype,
    )
    return (stacked * weights).mean()


def _masked_mean_square(value: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    value = value.float()
    square = value.pow(2)
    if attention_mask is None:
        return square.mean()
    mask = attention_mask.to(device=value.device, dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    count = mask.sum().clamp_min(1.0)
    return (square * mask).sum() / count


def compute_distill_hidden_alignment_loss(
    *,
    teacher_hidden_states: Sequence[torch.Tensor],
    student_hidden_states: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    layer_weighting: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    if teacher_hidden_states is None or student_hidden_states is None:
        raise ValueError("Hidden states are required when LoRA hidden alignment loss is enabled.")
    if len(teacher_hidden_states) != len(student_hidden_states):
        raise ValueError(
            "Teacher/student hidden state counts differ: "
            f"{len(teacher_hidden_states)} vs {len(student_hidden_states)}."
        )
    if len(teacher_hidden_states) <= 1:
        raise ValueError("Hidden states must include embedding output plus at least one transformer block output.")

    layer_losses: List[torch.Tensor] = []
    for layer_idx, (teacher_hidden, student_hidden) in enumerate(
        zip(teacher_hidden_states[1:], student_hidden_states[1:])
    ):
        if tuple(teacher_hidden.shape) != tuple(student_hidden.shape):
            raise ValueError(
                f"Teacher/student hidden shape mismatch at block layer {layer_idx}: "
                f"{tuple(teacher_hidden.shape)} vs {tuple(student_hidden.shape)}."
            )
        teacher_hidden = teacher_hidden.detach()
        diff = student_hidden.float() - teacher_hidden.float()
        numerator = _masked_mean_square(diff, attention_mask)
        denominator = _masked_mean_square(teacher_hidden, attention_mask)
        layer_losses.append(numerator / (denominator + float(eps)))

    teacher_block_hiddens = [hidden.detach() for hidden in teacher_hidden_states[1:]]
    return _aggregate_hidden_alignment_layer_losses(
        layer_losses,
        layer_weighting,
        teacher_sequence_for_selection=teacher_block_hiddens,
        attention_mask=attention_mask,
        reference_hidden=teacher_hidden_states[0].detach(),
    )


def compute_distill_pre_mlp_hidden_alignment_loss(
    *,
    teacher_pre_mlp_hiddens: Sequence[torch.Tensor],
    student_pre_mlp_hiddens: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    layer_weighting: str,
    teacher_reference_hidden: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if teacher_pre_mlp_hiddens is None or student_pre_mlp_hiddens is None:
        raise ValueError("Pre-MLP hidden states are required when pre-MLP hidden alignment loss is enabled.")
    if len(teacher_pre_mlp_hiddens) != len(student_pre_mlp_hiddens):
        raise ValueError(
            "Teacher/student pre-MLP hidden state counts differ: "
            f"{len(teacher_pre_mlp_hiddens)} vs {len(student_pre_mlp_hiddens)}."
        )
    if len(teacher_pre_mlp_hiddens) == 0:
        raise ValueError("Pre-MLP hidden states must include at least one transformer block.")
    if is_adaptive_hidden_alignment_layer_weighting(layer_weighting) and teacher_reference_hidden is None:
        raise ValueError(
            "teacher_reference_hidden is required for adaptive pre-MLP hidden alignment layer weighting."
        )

    layer_losses: List[torch.Tensor] = []
    for layer_idx, (teacher_hidden, student_hidden) in enumerate(
        zip(teacher_pre_mlp_hiddens, student_pre_mlp_hiddens)
    ):
        if tuple(teacher_hidden.shape) != tuple(student_hidden.shape):
            raise ValueError(
                f"Teacher/student pre-MLP hidden shape mismatch at block layer {layer_idx}: "
                f"{tuple(teacher_hidden.shape)} vs {tuple(student_hidden.shape)}."
            )
        teacher_hidden = teacher_hidden.detach()
        diff = student_hidden.float() - teacher_hidden.float()
        numerator = _masked_mean_square(diff, attention_mask)
        denominator = _masked_mean_square(teacher_hidden, attention_mask)
        layer_losses.append(numerator / (denominator + float(eps)))

    teacher_sequence = [hidden.detach() for hidden in teacher_pre_mlp_hiddens]
    reference_hidden = (
        teacher_reference_hidden.detach()
        if teacher_reference_hidden is not None
        else None
    )
    return _aggregate_hidden_alignment_layer_losses(
        layer_losses,
        layer_weighting,
        teacher_sequence_for_selection=teacher_sequence,
        attention_mask=attention_mask,
        reference_hidden=reference_hidden,
    )


@contextmanager
def capture_pre_mlp_hiddens(model: nn.Module):
    qwen_model = getattr(model, "model", None)
    layers = getattr(qwen_model, "layers", None)
    if layers is None:
        raise ValueError("pre-MLP hidden alignment requires model.model.layers.")

    captured: List[torch.Tensor] = []
    handles = []

    for layer_idx, layer in enumerate(layers):
        module = getattr(layer, "post_attention_layernorm", None)
        if not isinstance(module, nn.Module):
            raise ValueError(
                "pre-MLP hidden alignment requires every model.model.layers[*] "
                f"to expose post_attention_layernorm; missing at layer {layer_idx}."
            )

        def hook(_module, inputs, _layer_idx=layer_idx):
            if not inputs:
                raise RuntimeError(f"post_attention_layernorm pre-hook at layer {_layer_idx} received no inputs.")
            captured.append(inputs[0])

        handles.append(module.register_forward_pre_hook(hook))

    if not handles:
        raise ValueError("pre-MLP hidden alignment requires at least one model.model.layers entry.")

    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def ensure_lora_training_stack_available() -> None:
    if LoraConfig is None or TaskType is None or get_peft_model is None:
        raise ImportError("未安装 peft。请先安装：pip install peft")
    if SFTTrainer is None or DataCollatorForCompletionOnlyLM is None:
        raise ImportError("未安装 trl。请先安装：pip install trl")


def create_lora_adapters(
    model: nn.Module,
    *,
    target_names: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
    use_dora: bool,
):
    unique_target_names = sorted(set(str(name) for name in target_names if str(name).strip()))
    if not unique_target_names:
        return model, None, unique_target_names

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(rank),
        lora_alpha=float(alpha),
        lora_dropout=float(dropout),
        target_modules=unique_target_names,
        inference_mode=False,
        bias="none",
        use_dora=bool(use_dora),
    )
    return get_peft_model(model, lora_config), lora_config, unique_target_names


def merge_all_lora(model: nn.Module) -> Tuple[nn.Module, int]:
    if PeftModel is None or not isinstance(model, PeftModel):
        return model, 0
    trainable_count = 0
    for name, _ in model.named_parameters():
        if "lora_" in name:
            trainable_count += 1
    merged_model = model.merge_and_unload()
    return merged_model, trainable_count


if SFTTrainer is None:
    class CustomSFTTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("未安装 trl。请先安装：pip install trl")
else:
    class CustomSFTTrainer(SFTTrainer):
        def __init__(
            self,
            *args,
            loss_type: str = "r_kl_top_1000",
            temperature: float = 1.0,
            loss_alpha: float = 0.5,
            hidden_loss_weight: float = 0.0,
            pre_mlp_hidden_loss_weight: float = 0.0,
            hidden_alignment_layer_weighting: str = "uniform",
            eakld_confidence_k: int = 16,
            teacher_logits_cpu_staging: bool = False,
            distill_hif4_act_controller: Optional[Hif4ActController] = None,
            teacher_param_snapshots: Optional[Sequence[Tuple[nn.Parameter, torch.Tensor]]] = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.loss_type = str(loss_type).strip().lower()
            self.temperature = float(temperature)
            self.loss_alpha = float(loss_alpha)
            self.hidden_loss_weight = float(hidden_loss_weight)
            if self.hidden_loss_weight < 0.0:
                raise ValueError(f"hidden_loss_weight must be >= 0, got {self.hidden_loss_weight}.")
            self.pre_mlp_hidden_loss_weight = float(pre_mlp_hidden_loss_weight)
            if self.pre_mlp_hidden_loss_weight < 0.0:
                raise ValueError(
                    f"pre_mlp_hidden_loss_weight must be >= 0, got {self.pre_mlp_hidden_loss_weight}."
                )
            self.hidden_alignment_layer_weighting = parse_distill_hidden_alignment_layer_weighting(
                hidden_alignment_layer_weighting
            )
            self.eakld_confidence_k = int(eakld_confidence_k)
            if self.eakld_confidence_k < 2:
                raise ValueError(f"eakld_confidence_k must be >= 2, got {self.eakld_confidence_k}.")
            self.teacher_logits_cpu_staging = bool(teacher_logits_cpu_staging)
            self.distill_hif4_act_controller = distill_hif4_act_controller
            self.teacher_param_snapshots = list(teacher_param_snapshots or [])

        def _teacher_logits_staging_dtype(self) -> torch.dtype:
            if bool(getattr(self.args, "bf16", False)):
                return torch.bfloat16
            if bool(getattr(self.args, "fp16", False)):
                return torch.float16
            return torch.float32

        def _stage_teacher_logits(self, logits: torch.Tensor) -> torch.Tensor:
            if not bool(getattr(self, "teacher_logits_cpu_staging", False)):
                return logits
            return logits.detach().to(
                device=torch.device("cpu"),
                dtype=self._teacher_logits_staging_dtype(),
            )

        def _teacher_logits_for_loss(
            self,
            staged_logits: torch.Tensor,
            student_logits: torch.Tensor,
        ) -> torch.Tensor:
            if staged_logits.device.type == "cpu":
                return staged_logits.to(device=student_logits.device, non_blocking=True)
            return staged_logits

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            args = self.args
            loss_type = self.loss_type
            hidden_loss_enabled = float(self.hidden_loss_weight) > 0.0
            pre_mlp_hidden_loss_enabled = float(self.pre_mlp_hidden_loss_weight) > 0.0
            teacher_inputs = dict(inputs)
            teacher_inputs.pop("labels", None)
            student_inputs = dict(inputs)
            uses_ce_loss = (
                loss_type == "kd"
                or loss_type == "dual_kd"
                or loss_type == "eakld_kd"
                or loss_type.startswith("kd_top")
                or loss_type.startswith("dual_kd_top")
            )
            if not uses_ce_loss:
                student_inputs.pop("labels", None)
            full_inputs = dict(inputs)

            unwrapped_model = model
            if getattr(self, "accelerator", None) is not None:
                unwrapped_model = self.accelerator.unwrap_model(model)
            temporary_modules = [
                module
                for module in unwrapped_model.modules()
                if callable(getattr(module, "set_temporary", None))
            ]
            previous_temporary = [getattr(module, "temporary", None) for module in temporary_modules]
            hif4_act_controller = self.distill_hif4_act_controller
            previous_hif4_enabled = bool(getattr(hif4_act_controller, "enabled", False))
            peft_model_for_teacher = unwrapped_model if isinstance(unwrapped_model, PeftModel) else model
            teacher_pre_mlp_hiddens = None
            student_pre_mlp_hiddens = None

            def set_temporary(temporary: bool) -> None:
                for module in temporary_modules:
                    module.set_temporary(temporary)

            def restore_temporary() -> None:
                for module, previous in zip(temporary_modules, previous_temporary):
                    module.set_temporary(True if previous is None else bool(previous))

            def set_hif4_act_enabled(enabled: bool) -> None:
                if hif4_act_controller is not None:
                    hif4_act_controller.enabled = bool(enabled)

            def prepare_student_path() -> None:
                set_temporary(True)
                set_hif4_act_enabled(previous_hif4_enabled)

            def parse_k(prefix: str, default_k: int = 1000) -> int:
                if loss_type == prefix:
                    return default_k
                suffix = loss_type[len(prefix):]
                if suffix.startswith("_"):
                    suffix = suffix[1:]
                if not suffix:
                    return default_k
                return max(1, int(suffix))

            def use_post_attn() -> bool:
                return bool(getattr(args, "distill_post_attn", False))

            @contextmanager
            def teacher_param_context():
                if not self.teacher_param_snapshots:
                    yield
                    return

                current_values: List[torch.Tensor] = []
                try:
                    for param, snapshot in self.teacher_param_snapshots:
                        current_values.append(param.detach().clone())
                        param.data.copy_(snapshot.to(device=param.device, dtype=param.dtype))
                    yield
                finally:
                    for (param, _snapshot), current in zip(self.teacher_param_snapshots, current_values):
                        param.data.copy_(current.to(device=param.device, dtype=param.dtype))

            @torch.no_grad()
            def get_ori_outputs():
                nonlocal teacher_pre_mlp_hiddens
                set_temporary(False)
                set_hif4_act_enabled(False)
                adapter_context = (
                    peft_model_for_teacher.disable_adapter()
                    if hasattr(peft_model_for_teacher, "disable_adapter")
                    else nullcontext()
                )
                pre_mlp_context = capture_pre_mlp_hiddens(unwrapped_model) if pre_mlp_hidden_loss_enabled else nullcontext()
                with adapter_context, teacher_param_context(), pre_mlp_context as captured_pre_mlp:
                    outputs = model(**teacher_inputs, output_hidden_states=hidden_loss_enabled)
                if pre_mlp_hidden_loss_enabled:
                    teacher_pre_mlp_hiddens = tuple(captured_pre_mlp)
                return outputs

            def student_forward(model_inputs):
                nonlocal student_pre_mlp_hiddens
                pre_mlp_context = capture_pre_mlp_hiddens(unwrapped_model) if pre_mlp_hidden_loss_enabled else nullcontext()
                with pre_mlp_context as captured_pre_mlp:
                    if hidden_loss_enabled:
                        outputs = model(**model_inputs, output_hidden_states=True)
                    else:
                        outputs = model(**model_inputs)
                if pre_mlp_hidden_loss_enabled:
                    student_pre_mlp_hiddens = tuple(captured_pre_mlp)
                return outputs

            def add_hidden_alignment_loss(loss, teacher_outputs, student_outputs):
                if hidden_loss_enabled:
                    hidden_loss = compute_distill_hidden_alignment_loss(
                        teacher_hidden_states=teacher_outputs.hidden_states,
                        student_hidden_states=student_outputs.hidden_states,
                        attention_mask=full_inputs.get("attention_mask"),
                        layer_weighting=self.hidden_alignment_layer_weighting,
                    )
                    loss = loss + float(self.hidden_loss_weight) * hidden_loss
                if pre_mlp_hidden_loss_enabled:
                    if teacher_pre_mlp_hiddens is None or student_pre_mlp_hiddens is None:
                        raise RuntimeError("pre-MLP hidden alignment requires teacher and student captured hiddens.")
                    pre_mlp_hidden_loss = compute_distill_pre_mlp_hidden_alignment_loss(
                        teacher_pre_mlp_hiddens=teacher_pre_mlp_hiddens,
                        student_pre_mlp_hiddens=student_pre_mlp_hiddens,
                        attention_mask=full_inputs.get("attention_mask"),
                        layer_weighting=self.hidden_alignment_layer_weighting,
                        teacher_reference_hidden=teacher_outputs.hidden_states[0],
                    )
                    loss = loss + float(self.pre_mlp_hidden_loss_weight) * pre_mlp_hidden_loss
                return loss

            try:
                if loss_type in {"origin", "sft"}:
                    if hidden_loss_enabled or pre_mlp_hidden_loss_enabled:
                        teacher_outputs = get_ori_outputs()
                        prepare_student_path()
                        outputs = student_forward(full_inputs)
                        loss = add_hidden_alignment_loss(outputs["loss"], teacher_outputs, outputs)
                        return (loss, outputs) if return_outputs else loss
                    try:
                        return super().compute_loss(
                            model,
                            full_inputs,
                            return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch,
                        )
                    except TypeError:
                        return super().compute_loss(
                            model,
                            full_inputs,
                            return_outputs=return_outputs,
                        )

                prepare_student_path()

                if loss_type == "rkl":
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_reverse_kl_loss(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                        temperature=float(self.temperature),
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "dual_rkl":
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_dual_rkl_loss(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "kl":
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_forward_kl_loss(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                        temperature=float(self.temperature),
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("r_kl_top"):
                    k = parse_k("r_kl_top", default_k=1000)
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_rkl_topk(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                        k=k,
                        temperature=float(self.temperature),
                        post_attn=use_post_attn(),
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("dual_r_kl_top"):
                    k = parse_k("dual_r_kl_top", default_k=1000)
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_dual_rkl_topk_loss(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                        k=k,
                        post_attn=use_post_attn(),
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("kl_top"):
                    k = parse_k("kl_top", default_k=1000)
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_kl_topk(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                        k=k,
                        temperature=float(self.temperature),
                        post_attn=use_post_attn(),
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("kd_top"):
                    k = parse_k("kd_top", default_k=1000)
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(full_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    T, alpha = self.temperature, self.loss_alpha
                    ori_loss = outputs["loss"]
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    distill_loss = compute_kl_topk(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                        k=k,
                        temperature=float(T),
                        post_attn=use_post_attn(),
                    )
                    # T² is already applied inside compute_kl_topk.
                    loss = ori_loss * (1 - alpha) + distill_loss * alpha
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "mse":
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    loss = F.mse_loss(logits, teacher_logits)
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "kd":
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(full_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    T, alpha = self.temperature, self.loss_alpha
                    ori_loss = outputs["loss"]
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    distill_loss = compute_forward_kl_loss(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                        temperature=float(T),
                    )
                    # T² is already applied inside compute_forward_kl_loss.
                    loss = ori_loss * (1 - alpha) + distill_loss * alpha
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "dual_kl":
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_dual_kl_loss(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("dual_kl_top"):
                    k = parse_k("dual_kl_top", default_k=1000)
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_dual_kl_topk_loss(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                        k=k,
                        post_attn=use_post_attn(),
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("dual_kd_top"):
                    k = parse_k("dual_kd_top", default_k=1000)
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(full_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    ori_loss = outputs["loss"]
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    distill_loss = compute_dual_kl_topk_loss(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                        k=k,
                        post_attn=use_post_attn(),
                    )
                    alpha = self.loss_alpha
                    loss = ori_loss * (1 - alpha) + distill_loss * alpha
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "dual_kd":
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(full_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    ori_loss = outputs["loss"]
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    distill_loss = compute_dual_kl_loss(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                    )
                    alpha = self.loss_alpha
                    loss = ori_loss * (1 - alpha) + distill_loss * alpha
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if is_eakld_top_loss(loss_type):
                    k = parse_eakld_top_k(loss_type, default_k=1000)
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_eakld_topk(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                        k=k,
                        temperature=float(self.temperature),
                        confidence_k=int(self.eakld_confidence_k),
                        post_attn=use_post_attn(),
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "eakld":
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_eakld(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                        temperature=float(self.temperature),
                        confidence_k=int(self.eakld_confidence_k),
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "eakld_kd":
                    teacher_outputs = get_ori_outputs()
                    ori_logits = self._stage_teacher_logits(teacher_outputs.logits)
                    prepare_student_path()
                    outputs = student_forward(full_inputs)
                    logits = outputs.logits
                    teacher_logits = self._teacher_logits_for_loss(ori_logits, logits)
                    T, alpha = self.temperature, self.loss_alpha
                    ori_loss = outputs["loss"]
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    distill_loss = compute_entropy_aware_kl_loss(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        mask=token_mask,
                        temperature=float(T),
                        confidence_k=int(self.eakld_confidence_k),
                    )
                    # T² is already applied inside compute_eakld.
                    loss = ori_loss * (1 - alpha) + distill_loss * alpha
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                raise ValueError(
                    f"Unsupported lora loss type: {loss_type}. "
                    f"Supported: sft/origin, rkl, dual_rkl, kl, r_kl_top[_K], dual_r_kl_top[_K], "
                    f"kl_top[_K], kd_top[_K], eakld, eakld_kd, eakld_top[_K]/eakld_topk[_K], "
                    f"dual_kl, dual_kd, dual_kl_top[_K], dual_kd_top[_K], mse, kd."
                )
            finally:
                restore_temporary()
                set_hif4_act_enabled(previous_hif4_enabled)
