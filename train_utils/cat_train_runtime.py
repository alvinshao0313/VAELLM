import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Dict

from torch import nn

from train_utils.cat_train_args import ResolvedCategoryRuntimeConfig
from train_utils.model_checkpoint_io import (
    META_FILENAME,
    load_model_checkpoint,
    resolve_checkpoint_dir,
)
from train_utils.utils import get_logger


# def init_sort_prep_worker() -> None:
#     # 排序代码，已关闭：旧排序预处理 worker 初始化保留如下，不再执行。
#     # os.environ["OMP_NUM_THREADS"] = "1"
#     # os.environ["MKL_NUM_THREADS"] = "1"
#     # os.environ["OPENBLAS_NUM_THREADS"] = "1"
#     # try:
#     #     torch.set_num_threads(1)
#     # except Exception:
#     #     pass
#     return None
#
#
# def resolve_sort_prep_workers(requested_workers: int, *, linear_group_size: int) -> int:
#     # 排序代码，已关闭：旧排序 worker 解析保留如下，实际固定为串行 none 路径。
#     # requested = int(requested_workers)
#     # if requested < 0:
#     #     raise ValueError(f"sort_prep_workers must be >= 0, got {requested}.")
#     # max_tasks = max(1, int(linear_group_size))
#     # if requested == 0:
#     #     cpu_count = os.cpu_count() or 1
#     #     return max(1, min(int(cpu_count), max_tasks))
#     # return max(1, min(requested, max_tasks))
#     return 1


def _to_jsonable(value):
    if hasattr(value, "to_jsonable") and callable(getattr(value, "to_jsonable")):
        return value.to_jsonable()
    if hasattr(value, "value") and not isinstance(value, (str, bytes, bytearray)):
        return _to_jsonable(value.value)
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, argparse.Namespace):
        return {k: _to_jsonable(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
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
    snapshot_path = os.path.join(run_output_dir, "normalized_cat_train_args.json")
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
                "--resume_from_checkpoint cannot be combined with --rot_llm because the checkpoint already contains model weights to resume from.")

        checkpoint_dir = resolve_checkpoint_dir(str(cat_args.resume_from_checkpoint))
        meta_path = os.path.join(checkpoint_dir, META_FILENAME)
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        base_model_path = meta.get("base_model_path")
        if base_model_path is None:
            base_model_path = getattr(vae_args, "model_path", None)
        if not base_model_path:
            raise ValueError(
                f"Cannot determine base model path for resumed checkpoint: {checkpoint_dir}. "
                "Please save checkpoints with base_model_path metadata or pass --model_path."
            )

        log.info("Resuming from checkpoint: %s", checkpoint_dir)
        log.info("Resume base model path: %s", str(base_model_path))
        model, load_meta, load_result = load_model_checkpoint(
            checkpoint_dir,
            access_token=hf_args.access_token,
            base_model_path=str(base_model_path),
            map_location="cpu",
            strict=True,
        )
        vae_args.model_path = str(load_meta.get("base_model_path") or base_model_path)
        log.info(
            "Checkpoint loaded. missing_keys=%d unexpected_keys=%d converted_module_count=%s",
            len(getattr(load_result, "missing_keys", [])),
            len(getattr(load_result, "unexpected_keys", [])),
            str(load_meta.get("converted_module_count")),
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
