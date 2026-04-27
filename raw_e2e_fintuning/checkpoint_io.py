import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from raw_e2e_fintuning.smooth import collect_lora_smooth_fusion_specs, fuse_lora_smooth_into_linear_weights


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    return str(value)


def _namespace_to_dict(ns) -> Dict[str, Any]:
    if ns is None:
        return {}
    if is_dataclass(ns):
        return {str(k): _jsonable(v) for k, v in asdict(ns).items()}
    if hasattr(ns, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(ns).items()}
    return {"value": _jsonable(ns)}


def build_run_meta(
    *,
    raw_args,
    hf_args,
    training_args,
    data_info: Dict[str, Any],
    trainable_info: Dict[str, Any],
    teacher_source: str,
    global_step: int,
) -> Dict[str, Any]:
    return {
        "format": "e2e_raw_run_meta",
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "e2e_raw_fintuning",
        "teacher_source": str(teacher_source),
        "global_step": int(global_step),
        "raw_args": _namespace_to_dict(raw_args),
        "hf_args": _namespace_to_dict(hf_args),
        "training_args": _namespace_to_dict(training_args),
        "dataset": _jsonable(data_info),
        "trainables": _jsonable(trainable_info),
    }


def save_final_artifacts(
    *,
    model,
    run_output_dir: str,
    tokenizer,
    raw_args,
    hf_args,
    training_args,
    data_info: Dict[str, Any],
    trainable_info: Dict[str, Any],
    teacher_source: str,
    global_step: int,
    should_save: bool,
    state_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    adapter_dir = os.path.join(run_output_dir, "final_adapter")
    tokenizer_dir = os.path.join(run_output_dir, "tokenizer")
    merged_dir = os.path.join(run_output_dir, "final_merged_model")
    run_meta_path = os.path.join(run_output_dir, "run_meta.json")
    trainable_info = dict(trainable_info)

    if not bool(should_save):
        return {
            "adapter_dir": None,
            "tokenizer_dir": None,
            "merged_dir": None,
            "run_meta_path": None,
        }

    os.makedirs(run_output_dir, exist_ok=True)
    use_lora_smooth = bool(getattr(raw_args, "lora_smooth", False))
    if use_lora_smooth and not bool(getattr(raw_args, "raw_merge_and_save", False)):
        raise ValueError("lora_smooth=true requires raw_merge_and_save=true.")
    smooth_fusion_specs = collect_lora_smooth_fusion_specs(model) if use_lora_smooth else []
    if use_lora_smooth and not smooth_fusion_specs:
        raise RuntimeError("lora_smooth=true 但保存前未找到任何 LoRA smooth 参数。")

    save_kwargs = {"safe_serialization": True}
    if state_dict is not None:
        save_kwargs["state_dict"] = state_dict
    model.save_pretrained(adapter_dir, **save_kwargs)
    if bool(getattr(raw_args, "save_tokenizer", False)) and tokenizer is not None:
        tokenizer.save_pretrained(tokenizer_dir)

    actual_merged_dir = None
    if bool(getattr(raw_args, "raw_merge_and_save", False)):
        merge_fn = getattr(model, "merge_and_unload", None)
        if not callable(merge_fn):
            raise ValueError("raw_merge_and_save=true 但当前模型不支持 merge_and_unload。")
        merged_model = merge_fn()
        fused_smooth_count = fuse_lora_smooth_into_linear_weights(merged_model, smooth_fusion_specs)
        trainable_info["lora_smooth_fused_module_count"] = int(fused_smooth_count)
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        if bool(getattr(raw_args, "save_tokenizer", False)) and tokenizer is not None:
            tokenizer.save_pretrained(merged_dir)
        actual_merged_dir = merged_dir
    else:
        trainable_info["lora_smooth_fused_module_count"] = 0

    run_meta = build_run_meta(
        raw_args=raw_args,
        hf_args=hf_args,
        training_args=training_args,
        data_info=data_info,
        trainable_info=trainable_info,
        teacher_source=teacher_source,
        global_step=int(global_step),
    )
    with open(run_meta_path, "w", encoding="utf-8") as handle:
        json.dump(run_meta, handle, ensure_ascii=False, indent=2)

    return {
        "adapter_dir": adapter_dir,
        "tokenizer_dir": tokenizer_dir if os.path.isdir(tokenizer_dir) else None,
        "merged_dir": actual_merged_dir,
        "run_meta_path": run_meta_path,
    }
