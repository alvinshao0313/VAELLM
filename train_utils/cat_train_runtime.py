import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass, is_dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from train_utils.cat_category_runtime import ResolvedCategoryRuntimeConfig
from train_utils.checkpoint_v6 import (
    META_FILENAME,
    load_v6_cat_runtime_state,
    load_v6_full_checkpoint_into_model,
    load_v6_meta,
    load_v6_training_step_meta,
    resolve_training_step_round_base_ref,
    resolve_v6_checkpoint_dir,
    validate_v6_meta,
)
from train_utils.distributed_guard import distributed_guarded_main
from train_utils.utils import get_logger


def _safe_path_token(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "unknown_model"
    value = value.replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._/-]+", "_", value)
    value = value.replace("/", "__")
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "unknown_model"


def build_cat_run_output_dir(root_output_dir: str, model_path: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model_tag = _safe_path_token(model_path)
    base_run_dir = os.path.join(root_output_dir, f"{model_tag}_{ts}")
    run_dir = base_run_dir
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = f"{base_run_dir}_{suffix}"
        suffix += 1
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def build_distributed_cat_run_output_dir(root_output_dir: str, model_path: str) -> str:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            raise RuntimeError(
                "Distributed CAT run output dir creation requires torch.distributed to be initialized."
            )
        return build_cat_run_output_dir(root_output_dir, model_path)

    if int(torch.distributed.get_world_size()) <= 1:
        return build_cat_run_output_dir(root_output_dir, model_path)

    run_dir = distributed_guarded_main(
        lambda: build_cat_run_output_dir(root_output_dir, model_path),
        barrier=True,
    )
    if not isinstance(run_dir, str) or not run_dir:
        raise RuntimeError(f"Invalid distributed CAT run output dir broadcast payload: {run_dir!r}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


@dataclass(frozen=True)
class CatResumeSource:
    source_kind: str
    requested_checkpoint_dir: str
    model_checkpoint_dir: str
    model_checkpoint_kind: str
    model_checkpoint_meta: Dict[str, object]
    training_step_meta: Optional[Dict[str, object]] = None
    active_category: Optional[str] = None


@dataclass(frozen=True)
class CatResumeDistillProgress:
    completed_categories: Tuple[str, ...]
    distill_stage_history: Tuple[Dict[str, object], ...]
    lora_round_idx: int
    active_category: Optional[str] = None
    training_step_checkpoint: Optional[str] = None


def _read_v6_meta_unresolved(path: str) -> Tuple[str, Dict[str, object]]:
    checkpoint_dir = os.path.abspath(str(path))
    if os.path.isfile(checkpoint_dir):
        if os.path.basename(checkpoint_dir) != META_FILENAME:
            raise FileNotFoundError(f"Expected {META_FILENAME} or checkpoint directory, got {checkpoint_dir}")
        checkpoint_dir = os.path.dirname(checkpoint_dir)
    meta_path = os.path.join(checkpoint_dir, META_FILENAME)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing {META_FILENAME} under {checkpoint_dir}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        raw_meta = json.load(handle)
    if not isinstance(raw_meta, dict):
        raise TypeError(f"{META_FILENAME} must contain a JSON object, got {type(raw_meta)}.")
    return checkpoint_dir, validate_v6_meta(raw_meta)


def resolve_cat_resume_source(resume_from_checkpoint: str) -> CatResumeSource:
    requested_dir, raw_meta = _read_v6_meta_unresolved(str(resume_from_checkpoint))
    kind = str(raw_meta.get("checkpoint_kind"))
    if kind == "category_boundary":
        resolved = resolve_v6_checkpoint_dir(requested_dir)
        meta = validate_v6_meta(load_v6_meta(resolved), expected_kind="category_boundary")
        return CatResumeSource(
            source_kind="category_boundary",
            requested_checkpoint_dir=requested_dir,
            model_checkpoint_dir=resolved,
            model_checkpoint_kind="category_boundary",
            model_checkpoint_meta=meta,
        )
    if kind == "training_step":
        step_meta = load_v6_training_step_meta(requested_dir)
        round_base_dir, round_base_meta = resolve_training_step_round_base_ref(requested_dir, step_meta)
        round_base_meta = validate_v6_meta(round_base_meta, expected_kind="round_base")
        base_extra = round_base_meta.get("extra_meta")
        step_extra = step_meta.get("extra_meta")
        active_base = base_extra.get("active_category") if isinstance(base_extra, dict) else None
        active_step = step_extra.get("active_category") if isinstance(step_extra, dict) else None
        if not isinstance(active_base, str) or not active_base.strip():
            raise ValueError("CAT training-step round_base metadata requires non-empty active_category.")
        if active_step != active_base:
            raise ValueError(
                "CAT training-step active_category mismatch between step and round_base: "
                f"step={active_step!r}, round_base={active_base!r}."
            )
        return CatResumeSource(
            source_kind="training_step",
            requested_checkpoint_dir=requested_dir,
            model_checkpoint_dir=round_base_dir,
            model_checkpoint_kind="round_base",
            model_checkpoint_meta=round_base_meta,
            training_step_meta=step_meta,
            active_category=str(active_base),
        )
    if kind in {"round_base", "final_model"}:
        raise ValueError(
            f"Online CAT --resume_from_checkpoint does not accept checkpoint_kind={kind!r} directly. "
            "Use a category_boundary checkpoint or a training_step checkpoint that references its round_base."
        )
    raise ValueError(f"Unsupported CAT v6 resume checkpoint_kind={kind!r}.")


def _load_checkpoint_meta_payload(checkpoint_dir: str) -> Dict[str, object]:
    resolved = resolve_v6_checkpoint_dir(checkpoint_dir)
    return validate_v6_meta(
        load_v6_meta(resolved),
        expected_kind="category_boundary",
    )


def _validate_completed_categories(raw) -> Tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"completed_categories must be a list/tuple, got {type(raw)}.")
    completed = []
    seen = set()
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("completed_categories entries must be non-empty strings.")
        category = str(item)
        if category in seen:
            raise ValueError(f"completed_categories contains duplicate category: {category}")
        seen.add(category)
        completed.append(category)
    return tuple(completed)


def _validate_distill_stage_history(source: Dict[str, object]) -> Tuple[Dict[str, object], ...]:
    raw_history = source.get("distill_stage_history")
    if raw_history is not None:
        if not isinstance(raw_history, (list, tuple)):
            raise TypeError(f"distill_stage_history must be a list/tuple, got {type(raw_history)}.")
        history = []
        for item in raw_history:
            if not isinstance(item, dict):
                raise TypeError(f"distill_stage_history entries must be dicts, got {type(item)}.")
            history.append(dict(item))
        return tuple(history)

    raw_stage = source.get("distill_stage")
    if raw_stage is None:
        return ()
    if not isinstance(raw_stage, dict):
        raise TypeError(f"distill_stage must be a dict when present, got {type(raw_stage)}.")
    return (dict(raw_stage),)


def _resolve_lora_round_idx(history: Tuple[Dict[str, object], ...]) -> int:
    trained = 0
    for idx, item in enumerate(history):
        if "did_train" not in item:
            raise ValueError(
                f"distill_stage_history[{idx}] is missing did_train; "
                "cannot reconstruct exact CAT recovery round seed."
            )
        did_train = item["did_train"]
        if not isinstance(did_train, bool):
            raise TypeError(
                f"distill_stage_history[{idx}].did_train must be bool, got {type(did_train)}."
            )
        trained += int(did_train)
    return int(trained)


def load_cat_resume_distill_progress(
    resume_from_checkpoint: Optional[str],
) -> CatResumeDistillProgress:
    if resume_from_checkpoint is None or not str(resume_from_checkpoint).strip():
        return CatResumeDistillProgress(
            completed_categories=(),
            distill_stage_history=(),
            lora_round_idx=0,
        )

    source = resolve_cat_resume_source(str(resume_from_checkpoint))
    progress_meta = (
        source.training_step_meta
        if source.source_kind == "training_step"
        else source.model_checkpoint_meta
    )
    if not isinstance(progress_meta, dict):
        raise TypeError("Resolved CAT resume progress metadata must be a dict.")
    completed = _validate_completed_categories(progress_meta.get("completed_categories"))
    base_completed = _validate_completed_categories(
        source.model_checkpoint_meta.get("completed_categories")
    )
    if completed != base_completed:
        raise ValueError(
            "CAT training-step completed_categories mismatch with round_base: "
            f"step={completed}, round_base={base_completed}."
        )
    extra_meta = progress_meta.get("extra_meta")
    history_source = extra_meta if isinstance(extra_meta, dict) else {}
    history = _validate_distill_stage_history(history_source)
    return CatResumeDistillProgress(
        completed_categories=completed,
        distill_stage_history=history,
        lora_round_idx=_resolve_lora_round_idx(history),
        active_category=source.active_category,
        training_step_checkpoint=(
            source.requested_checkpoint_dir if source.source_kind == "training_step" else None
        ),
    )


def normalize_cat_runtime_vae_original_state(model: nn.Module) -> int:
    from litebsq.vae_linear import VAELinear

    stripped = 0
    legacy_skip_names = []
    for name, module in model.named_modules():
        if not isinstance(module, VAELinear):
            continue
        if bool(getattr(module, "always_use_original", False)):
            legacy_skip_names.append(str(name))
            continue
        if bool(getattr(module, "protect_original_weight", False)):
            continue
        if getattr(module, "original_weight", None) is not None:
            module.register_parameter("original_weight", None)
            stripped += 1
        module.always_use_original = False
        module.protect_original_weight = False
        module.temporary = True
        module.clear_decoded_weight_cache()
    if legacy_skip_names:
        raise ValueError(
            "Legacy skip-as-VAELinear checkpoint is not supported by the new CAT skip semantics: "
            + ", ".join(legacy_skip_names)
        )
    return stripped


def _to_jsonable(value):
    if hasattr(value, "to_jsonable") and callable(getattr(value, "to_jsonable")):
        return value.to_jsonable()
    if hasattr(value, "value") and not isinstance(value, (str, bytes, bytearray)):
        return _to_jsonable(value.value)
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, argparse.Namespace):
        return {
            k: _to_jsonable(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_") and not callable(v)
        }
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_to_jsonable(v) for v in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _resolve_rot_block_size(codebook_dim_value) -> int:
    if hasattr(codebook_dim_value, "has_default"):
        if not bool(getattr(codebook_dim_value, "has_default", False)):
            raise ValueError("--rot_llm requires --codebook_dim to provide a default value.")
        return int(getattr(codebook_dim_value, "default"))
    return int(codebook_dim_value)


def save_normalized_cat_train_snapshot(
    *,
    run_output_dir: str,
    cat_args,
    vae_args,
    training_args,
    resolved_category_cfgs: Dict[str, ResolvedCategoryRuntimeConfig],
) -> str:
    snapshot_path = os.path.join(run_output_dir, "normalized_cat_runtime_args.json")
    payload = {
        "cat_args": _to_jsonable(cat_args),
        "vae_args": _to_jsonable(vae_args),
        "training_args": _to_jsonable(training_args),
        "resolved_category_runtime": {
            category: _to_jsonable(cfg)
            for category, cfg in resolved_category_cfgs.items()
        },
    }
    with open(snapshot_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return snapshot_path


def load_model_for_cat_train(*, cat_args, hf_args, vae_args) -> nn.Module:
    log = get_logger("linear_by_category")
    if getattr(cat_args, "resume_from_checkpoint", None):
        if bool(getattr(cat_args, "rot_llm", False)):
            raise ValueError(
                "--resume_from_checkpoint cannot be combined with --rot_llm because "
                "the checkpoint already contains model weights to resume from."
            )

        source = resolve_cat_resume_source(str(cat_args.resume_from_checkpoint))
        checkpoint_dir = source.model_checkpoint_dir
        meta = source.model_checkpoint_meta
        base_model_path = meta.get("base_model_path") or getattr(vae_args, "model_path", None)
        if not base_model_path:
            raise ValueError(
                f"Cannot determine base model path for resumed v6 checkpoint: {checkpoint_dir}. "
                "Save the checkpoint with base_model_path metadata or pass --model_path."
            )

        log.info(
            "Resuming CAT from v6 source: source_kind=%s requested=%s model_checkpoint=%s",
            source.source_kind,
            source.requested_checkpoint_dir,
            checkpoint_dir,
        )
        log.info("Resume base model path: %s", str(base_model_path))
        from rotation.model_utils import get_model

        model = get_model(str(base_model_path), hf_args.access_token)
        model, load_meta, load_result = load_v6_full_checkpoint_into_model(
            model,
            checkpoint_dir,
            expected_kind=source.model_checkpoint_kind,
            map_location="cpu",
            strict=True,
        )
        vae_args.model_path = str(load_meta.get("base_model_path") or base_model_path)
        runtime_state_payload = load_v6_cat_runtime_state(checkpoint_dir, required=False)
        setattr(cat_args, "_v6_resume_source", source)
        setattr(cat_args, "_v6_cat_runtime_state_payload", runtime_state_payload)
        log.info(
            "v6 checkpoint loaded. missing_keys=%d unexpected_keys=%d converted_module_count=%s checkpoint_id=%s",
            len(getattr(load_result, "missing_keys", [])),
            len(getattr(load_result, "unexpected_keys", [])),
            str(load_meta.get("converted_module_count")),
            str(load_meta.get("checkpoint_id")),
        )
        stripped = normalize_cat_runtime_vae_original_state(model)
        if stripped:
            log.info(
                "Normalized resumed CAT checkpoint: stripped inactive original_weight from %d VAELinear modules.",
                stripped,
            )
        return model

    log.info("Loading model: %s", vae_args.model_path)
    from rotation.model_utils import get_model

    model = get_model(vae_args.model_path, hf_args.access_token)
    if bool(getattr(cat_args, "rot_llm", False)):
        from rotation.model_rotation import prepare_model

        rot_block_size = _resolve_rot_block_size(getattr(vae_args, "codebook_dim", 32))
        log.info("Applying offline LLM rotation fusion before VAE compression.")
        log.info("Rotation block size resolved from --codebook_dim default: %d", rot_block_size)
        model = prepare_model(model, rot_block_size=rot_block_size)
    return model
