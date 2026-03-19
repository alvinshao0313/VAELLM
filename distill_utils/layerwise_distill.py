import json
import os
import sys
from typing import Any, Dict, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from distill_utils.layerwise_distill_args import build_parser, parse_layer_indices
from distill_utils.layerwise_distill_runtime import (
    collect_calib_inputs,
    resolve_checkpoint_dir,
    resolve_distill_device,
)
from distill_utils.layerwise_distill_trainer import distill_layers
from litebsq.vae_linear import clear_model_vae_linear_cache
from rotation.model_utils import get_layers, get_model
from train_utils.model_checkpoint_io import (
    _build_run_output_dir,
    load_model_checkpoint,
    save_model_checkpoint,
)
from train_utils.utils import get_logger, set_seed


log = get_logger("layerwise_distill")


def _split_csv(value: Optional[str]) -> Sequence[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _init_wandb_run(
    *,
    args,
    run_output_dir: str,
    student_ckpt_dir: str,
    teacher_model_path: str,
    layer_indices: Sequence[int],
    meta: Dict[str, Any],
):
    project = str(getattr(args, "wandb_project", "") or "").strip()
    if not project:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging requested via --wandb_project, but the `wandb` package is not installed in the current environment."
        ) from exc

    config = dict(vars(args))
    config.update(
        {
            "run_output_dir": run_output_dir,
            "resolved_student_checkpoint_dir": student_ckpt_dir,
            "resolved_teacher_model_path": str(teacher_model_path),
            "resolved_layer_indices": list(layer_indices),
            "resolved_target_layer_count": len(layer_indices),
            "converted_module_count": meta.get("converted_module_count"),
        }
    )
    run_name = str(getattr(args, "wandb_name", "") or "").strip() or os.path.basename(run_output_dir)
    wandb_run = wandb.init(
        project=project,
        entity=str(getattr(args, "wandb_entity", "") or "").strip() or None,
        name=run_name,
        group=str(getattr(args, "wandb_group", "") or "").strip() or None,
        tags=list(_split_csv(getattr(args, "wandb_tags", None))) or None,
        mode=str(getattr(args, "wandb_mode", "online") or "online"),
        dir=run_output_dir,
        config=config,
    )
    if wandb_run is None:
        return None

    wandb_run.define_metric("train/global_step")
    for pattern in (
        "loss/*",
        "eval/*",
        "layer/*",
        "system/*",
        "train/layer_step",
        "train/layer_id",
        "train/layer_order",
        "train/lr",
    ):
        wandb_run.define_metric(pattern, step_metric="train/global_step")

    wandb_run.summary["run_output_dir"] = run_output_dir
    wandb_run.summary["student_checkpoint_dir"] = student_ckpt_dir
    wandb_run.summary["teacher_model_path"] = str(teacher_model_path)
    wandb_run.summary["target_layer_count"] = len(layer_indices)
    wandb_run.log(
        {
            "train/global_step": 0,
            "layer/target_count": len(layer_indices),
            "system/converted_module_count": float(meta.get("converted_module_count", 0) or 0),
        }
    )
    return wandb_run


def main(argv: Optional[Sequence[str]] = None) -> None:
    global log

    args = build_parser().parse_args(argv)
    set_seed(int(args.seed))

    student_ckpt_dir = resolve_checkpoint_dir(args.student_checkpoint_dir)

    run_output_dir = _build_run_output_dir(args.output_dir, os.path.basename(student_ckpt_dir))
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "layerwise_distill.log")
    log = get_logger("layerwise_distill")
    log.info("Run output directory: %s", run_output_dir)
    log.info("Input args:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2))
    log.info("Resolved student checkpoint directory: %s", student_ckpt_dir)

    log.info("Loading student (quantized) checkpoint...")
    model_q, meta, load_result = load_model_checkpoint(
        student_ckpt_dir,
        access_token=args.access_token,
        base_model_path=args.teacher_model_path,
        map_location=args.map_location,
        strict=bool(args.strict),
    )
    log.info(
        "Student loaded. missing_keys=%d unexpected_keys=%d converted_module_count=%s",
        len(getattr(load_result, "missing_keys", [])),
        len(getattr(load_result, "unexpected_keys", [])),
        str(meta.get("converted_module_count")),
    )

    teacher_model_path = args.teacher_model_path or meta.get("base_model_path")
    if not teacher_model_path:
        raise ValueError("teacher_model_path is missing. Pass --teacher_model_path or include base_model_path in checkpoint meta.")
    args.model_path = str(teacher_model_path)

    log.info("Loading teacher model from: %s", teacher_model_path)
    model_t = get_model(teacher_model_path, args.access_token)

    distill_device = resolve_distill_device(args.distill_device)
    if distill_device != args.distill_device:
        log.warning("distill_device fallback: %s -> %s", args.distill_device, distill_device)
    args.distill_device = distill_device

    model_q.to("cpu")
    model_t.to("cpu")
    model_q.eval()
    model_t.eval()
    if hasattr(model_q, "config"):
        model_q.config.use_cache = False
    if hasattr(model_t, "config"):
        model_t.config.use_cache = False

    calib_inputs = collect_calib_inputs(
        model_path=str(teacher_model_path),
        nsamples=int(args.nsamples),
        seed=int(args.seed),
        seqlen=int(args.seqlen),
        access_token=args.access_token,
    )
    if int(calib_inputs.shape[0]) < 1:
        raise RuntimeError("No calibration inputs available.")

    layers_q = list(get_layers(model_q))
    layers_t = list(get_layers(model_t))
    if len(layers_q) != len(layers_t):
        raise RuntimeError(f"Layer count mismatch: student={len(layers_q)} teacher={len(layers_t)}")

    layer_indices = parse_layer_indices(args.layer_indices)
    if layer_indices is None:
        layer_indices = list(range(len(layers_q)))
    if args.max_layers is not None:
        layer_indices = layer_indices[: int(args.max_layers)]
    for layer_id in layer_indices:
        if layer_id < 0 or layer_id >= len(layers_q):
            raise ValueError(f"Invalid layer_id={layer_id}, valid range=[0, {len(layers_q) - 1}]")

    log.info(
        "Distillation target layers: total=%d, first=%s, last=%s",
        len(layer_indices),
        str(layer_indices[0] if layer_indices else None),
        str(layer_indices[-1] if layer_indices else None),
    )

    wandb_run = _init_wandb_run(
        args=args,
        run_output_dir=run_output_dir,
        student_ckpt_dir=student_ckpt_dir,
        teacher_model_path=str(teacher_model_path),
        layer_indices=layer_indices,
        meta=meta,
    )

    try:
        distill_layers(
            model_q=model_q,
            model_t=model_t,
            layers_q=layers_q,
            layers_t=layers_t,
            layer_indices=layer_indices,
            calib_inputs=calib_inputs,
            args=args,
            log=log,
            wandb_run=wandb_run,
        )

        if bool(args.save_model):
            log.info("Saving distilled checkpoint...")
            model_out = os.path.join(run_output_dir, "final_model")
            tokenizer = None
            if bool(args.save_tokenizer):
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    teacher_model_path,
                    use_fast=True,
                    token=args.access_token,
                )

            cleared = clear_model_vae_linear_cache(model_q)
            log.info("Cleared decoded cache for %d VAELinear modules before save.", cleared)
            save_paths = save_model_checkpoint(
                model_q,
                model_out,
                base_model_path=str(teacher_model_path),
                tokenizer=tokenizer,
                save_config=True,
                extra_meta={
                    "stage": "layerwise_distill",
                    "source_checkpoint_dir": student_ckpt_dir,
                },
                unload_vae_original_weights=bool(args.unload_vae_original_weights_on_save),
            )
            log.info("Saved distilled model to %s", save_paths["output_dir"])
            if wandb_run is not None:
                wandb_run.summary["saved_model_dir"] = save_paths["output_dir"]

        if wandb_run is not None:
            wandb_run.summary["status"] = "finished"
        log.info("Layer-wise distillation finished.")
    except BaseException:
        if wandb_run is not None:
            wandb_run.summary["status"] = "failed"
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
