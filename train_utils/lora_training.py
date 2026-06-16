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


_DISTILL_HIDDEN_LAYER_WEIGHTING_CHOICES = ("uniform", "linear_depth")


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
    if mode == "uniform":
        return torch.ones(num_layers, device=device, dtype=dtype)
    if mode == "linear_depth":
        denom = max(num_layers - 1, 1)
        raw = 1.0 + torch.arange(num_layers, device=device, dtype=dtype) / float(denom)
        return raw / raw.mean()
    raise ValueError(
        f"Unsupported distill hidden layer weighting: {layer_weighting}. "
        f"Supported: {', '.join(_DISTILL_HIDDEN_LAYER_WEIGHTING_CHOICES)}."
    )


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

    stacked = torch.stack(layer_losses)
    weights = build_distill_hidden_layer_weights(
        num_layers=len(layer_losses),
        layer_weighting=layer_weighting,
        device=stacked.device,
        dtype=stacked.dtype,
    )
    return (stacked * weights).mean()



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
            hidden_layer_weighting: str = "uniform",
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
            self.hidden_layer_weighting = str(hidden_layer_weighting).strip().lower()
            if self.hidden_layer_weighting not in _DISTILL_HIDDEN_LAYER_WEIGHTING_CHOICES:
                raise ValueError(
                    f"Unsupported hidden_layer_weighting: {hidden_layer_weighting}. "
                    f"Supported: {', '.join(_DISTILL_HIDDEN_LAYER_WEIGHTING_CHOICES)}."
                )
            self.distill_hif4_act_controller = distill_hif4_act_controller
            self.teacher_param_snapshots = list(teacher_param_snapshots or [])

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            args = self.args
            loss_type = self.loss_type
            hidden_loss_enabled = float(self.hidden_loss_weight) > 0.0
            teacher_inputs = dict(inputs)
            teacher_inputs.pop("labels", None)
            student_inputs = dict(inputs)
            uses_ce_loss = (
                loss_type == "kd"
                or loss_type == "dual_kd"
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
                set_temporary(False)
                set_hif4_act_enabled(False)
                adapter_context = (
                    peft_model_for_teacher.disable_adapter()
                    if hasattr(peft_model_for_teacher, "disable_adapter")
                    else nullcontext()
                )
                with adapter_context, teacher_param_context():
                    outputs = model(**teacher_inputs, output_hidden_states=hidden_loss_enabled)
                return outputs

            def student_forward(model_inputs):
                if hidden_loss_enabled:
                    return model(**model_inputs, output_hidden_states=True)
                return model(**model_inputs)

            def add_hidden_alignment_loss(loss, teacher_outputs, student_outputs):
                if not hidden_loss_enabled:
                    return loss
                hidden_loss = compute_distill_hidden_alignment_loss(
                    teacher_hidden_states=teacher_outputs.hidden_states,
                    student_hidden_states=student_outputs.hidden_states,
                    attention_mask=full_inputs.get("attention_mask"),
                    layer_weighting=self.hidden_layer_weighting,
                )
                return loss + float(self.hidden_loss_weight) * hidden_loss

            try:
                if loss_type in {"origin", "sft"}:
                    if hidden_loss_enabled:
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
                    ori_logits = teacher_outputs.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    loss = F.kl_div(
                        F.log_softmax(ori_logits.flatten(0, -2), dim=-1),
                        F.softmax(logits, dim=-1).flatten(0, -2),
                        reduction="batchmean",
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "dual_rkl":
                    teacher_outputs = get_ori_outputs()
                    ori_logits = teacher_outputs.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_dual_rkl_loss(
                        student_logits=logits,
                        teacher_logits=ori_logits,
                        mask=token_mask,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "kl":
                    teacher_outputs = get_ori_outputs()
                    ori_logits = teacher_outputs.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    loss = F.kl_div(
                        F.log_softmax(logits.flatten(0, -2), dim=-1),
                        F.softmax(ori_logits, dim=-1).flatten(0, -2),
                        reduction="batchmean",
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("r_kl_top"):
                    k = parse_k("r_kl_top", default_k=1000)
                    teacher_outputs = get_ori_outputs()
                    ori_logits = teacher_outputs.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    k = min(k, int(logits.shape[-1]))
                    top_logits, indices = logits.topk(k, dim=-1, sorted=False)
                    top_ori_logits = ori_logits.gather(-1, indices)
                    loss = F.kl_div(
                        F.log_softmax(top_ori_logits.flatten(0, -2), dim=-1),
                        F.softmax(top_logits.flatten(0, -2), dim=-1),
                        reduction="batchmean",
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("dual_r_kl_top"):
                    k = parse_k("dual_r_kl_top", default_k=1000)
                    teacher_outputs = get_ori_outputs()
                    ori_logits = teacher_outputs.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_dual_rkl_topk_loss(
                        student_logits=logits,
                        teacher_logits=ori_logits,
                        mask=token_mask,
                        k=k,
                        post_attn=use_post_attn(),
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("kl_top"):
                    k = parse_k("kl_top", default_k=1000)
                    teacher_outputs = get_ori_outputs()
                    ori_logits = teacher_outputs.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    k = min(k, int(ori_logits.shape[-1]))
                    top_ori_logits, indices = ori_logits.topk(k, dim=-1, sorted=False)
                    if use_post_attn():
                        ref = F.softmax(ori_logits, dim=-1).gather(-1, indices).flatten(0, -2)
                        can = F.log_softmax(logits, dim=-1).gather(-1, indices).flatten(0, -2)
                        loss = F.kl_div(can, ref, reduction="batchmean")
                    else:
                        top_logits = logits.gather(-1, indices)
                        loss = F.kl_div(
                            F.log_softmax(top_logits, dim=-1).flatten(0, -2),
                            F.softmax(top_ori_logits, dim=-1).flatten(0, -2),
                            reduction="batchmean",
                        )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("kd_top"):
                    k = parse_k("kd_top", default_k=1000)
                    teacher_outputs = get_ori_outputs()
                    ori_logits = teacher_outputs.logits
                    prepare_student_path()
                    outputs = student_forward(full_inputs)
                    logits = outputs.logits
                    T, alpha = self.temperature, self.loss_alpha
                    ori_loss = outputs["loss"]
                    k = min(k, int(ori_logits.shape[-1]))
                    top_ori_logits, indices = ori_logits.topk(k, dim=-1, sorted=False)
                    if use_post_attn():
                        ref = F.softmax(ori_logits / T, dim=-1).gather(-1, indices).flatten(0, -2)
                        can = F.log_softmax(logits / T, dim=-1).gather(-1, indices).flatten(0, -2)
                        distill_loss = F.kl_div(can, ref, reduction="batchmean")
                    else:
                        top_logits = logits.gather(-1, indices)
                        distill_loss = F.kl_div(
                            F.log_softmax(top_logits / T, dim=-1).flatten(0, -2),
                            F.softmax(top_ori_logits / T, dim=-1).flatten(0, -2),
                            reduction="batchmean",
                        )
                    loss = ori_loss * (1 - alpha) + distill_loss * (alpha * T * T)
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "mse":
                    teacher_outputs = get_ori_outputs()
                    ori_logits = teacher_outputs.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    loss = F.mse_loss(logits, ori_logits)
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "kd":
                    teacher_outputs = get_ori_outputs()
                    ori_logits = teacher_outputs.logits
                    prepare_student_path()
                    outputs = student_forward(full_inputs)
                    logits = outputs.logits
                    T, alpha = self.temperature, self.loss_alpha
                    ori_loss = outputs["loss"]
                    logits = logits.view(-1, logits.size(-1))
                    ori_logits = ori_logits.view(-1, ori_logits.size(-1))
                    distill_loss = F.kl_div(
                        F.log_softmax(logits / T, dim=-1).flatten(0, -2),
                        F.softmax(ori_logits / T, dim=-1).flatten(0, -2),
                        reduction="batchmean",
                    )
                    loss = ori_loss * (1 - alpha) + distill_loss * (alpha * T * T)
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type == "dual_kl":
                    teacher_outputs = get_ori_outputs()
                    ori_logits = teacher_outputs.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_dual_kl_loss(
                        student_logits=logits,
                        teacher_logits=ori_logits,
                        mask=token_mask,
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("dual_kl_top"):
                    k = parse_k("dual_kl_top", default_k=1000)
                    teacher_outputs = get_ori_outputs()
                    ori_logits = teacher_outputs.logits
                    prepare_student_path()
                    outputs = student_forward(student_inputs)
                    logits = outputs.logits
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    loss = compute_dual_kl_topk_loss(
                        student_logits=logits,
                        teacher_logits=ori_logits,
                        mask=token_mask,
                        k=k,
                        post_attn=use_post_attn(),
                    )
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                if loss_type.startswith("dual_kd_top"):
                    k = parse_k("dual_kd_top", default_k=1000)
                    teacher_outputs = get_ori_outputs()
                    ori_logits = teacher_outputs.logits
                    prepare_student_path()
                    outputs = student_forward(full_inputs)
                    logits = outputs.logits
                    ori_loss = outputs["loss"]
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    distill_loss = compute_dual_kl_topk_loss(
                        student_logits=logits,
                        teacher_logits=ori_logits,
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
                    ori_logits = teacher_outputs.logits
                    prepare_student_path()
                    outputs = student_forward(full_inputs)
                    logits = outputs.logits
                    ori_loss = outputs["loss"]
                    token_mask = build_distill_token_mask(
                        labels=full_inputs.get("labels"),
                        attention_mask=full_inputs.get("attention_mask"),
                        reference_logits=logits,
                    )
                    distill_loss = compute_dual_kl_loss(
                        student_logits=logits,
                        teacher_logits=ori_logits,
                        mask=token_mask,
                    )
                    alpha = self.loss_alpha
                    loss = ori_loss * (1 - alpha) + distill_loss * alpha
                    loss = add_hidden_alignment_loss(loss, teacher_outputs, outputs)
                    return (loss, outputs) if return_outputs else loss

                raise ValueError(
                    f"Unsupported lora loss type: {loss_type}. "
                    f"Supported: sft/origin, rkl, dual_rkl, kl, r_kl_top[_K], dual_r_kl_top[_K], "
                    f"kl_top[_K], kd_top[_K], dual_kl, dual_kd, dual_kl_top[_K], dual_kd_top[_K], mse, kd."
                )
            finally:
                restore_temporary()
                set_hif4_act_enabled(previous_hif4_enabled)
