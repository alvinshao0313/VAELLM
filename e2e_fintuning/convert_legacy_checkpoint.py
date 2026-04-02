import argparse
import glob
import json
import os
import shutil
from typing import Dict, List, Optional

from transformers.modeling_utils import load_sharded_checkpoint

from e2e_fintuning.args import parse_decoder_layers, parse_target_modules
from e2e_fintuning.checkpoint_io import load_e2e_model_checkpoint, save_e2e_model_checkpoint
from e2e_fintuning.trainables import resolve_target_layer_ids, select_e2e_trainables
from rotation.common import separate_embeddings_and_lm_head
from rotation.model_utils import get_layers
from train_utils.train_args import _parse_bool_like


_E2E_FINETUNE_MODE = "vae_lora"
_COPY_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "training_args.bin",
    "scaler.pt",
)


def _embedding_and_lm_head_are_tied(model) -> bool:
    embedding = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if embedding is None or lm_head is None:
        return False
    if not hasattr(embedding, "weight") or not hasattr(lm_head, "weight"):
        return False
    return embedding.weight.data_ptr() == lm_head.weight.data_ptr()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a legacy HF e2e trainer checkpoint into the new compact resumable checkpoint format."
    )
    parser.add_argument("--legacy_checkpoint_dir", type=str, required=True)
    parser.add_argument("--student_checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_checkpoint_dir", type=str, default=None)
    parser.add_argument("--access_token", type=str, default=None)
    parser.add_argument("--decoder_layers", type=str, default="all")
    parser.add_argument("--target_modules", type=str, default="all")
    parser.add_argument("--vae_lora_rank", type=int, default=8)
    parser.add_argument("--vae_lora_alpha", type=float, default=16.0)
    parser.add_argument("--vae_lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_embedding",
        type=lambda v: _parse_bool_like(v, arg_name="--lora_embedding"),
        default=False,
    )
    parser.add_argument(
        "--lora_lm_head",
        type=lambda v: _parse_bool_like(v, arg_name="--lora_lm_head"),
        default=False,
    )
    parser.add_argument("--loss_type", type=str, default="sft")
    parser.add_argument(
        "--post_attn",
        type=lambda v: _parse_bool_like(v, arg_name="--post_attn"),
        default=False,
    )
    parser.add_argument(
        "--lora_hif4_act",
        type=lambda v: _parse_bool_like(v, arg_name="--lora_hif4_act"),
        default=False,
    )
    return parser


def _validate_args(args) -> None:
    args.legacy_checkpoint_dir = os.path.abspath(str(args.legacy_checkpoint_dir))
    args.student_checkpoint_dir = os.path.abspath(str(args.student_checkpoint_dir))
    if args.output_checkpoint_dir is None:
        args.output_checkpoint_dir = f"{args.legacy_checkpoint_dir}_compact"
    args.output_checkpoint_dir = os.path.abspath(str(args.output_checkpoint_dir))

    if not os.path.isdir(args.legacy_checkpoint_dir):
        raise FileNotFoundError(f"legacy checkpoint dir does not exist: {args.legacy_checkpoint_dir}")
    if not os.path.isdir(args.student_checkpoint_dir):
        raise FileNotFoundError(f"student checkpoint dir does not exist: {args.student_checkpoint_dir}")
    if os.path.exists(args.output_checkpoint_dir):
        raise FileExistsError(f"output checkpoint dir already exists: {args.output_checkpoint_dir}")

    trainer_state_path = os.path.join(args.legacy_checkpoint_dir, "trainer_state.json")
    index_path = os.path.join(args.legacy_checkpoint_dir, "model.safetensors.index.json")
    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(f"legacy checkpoint is missing trainer_state.json: {trainer_state_path}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"legacy checkpoint is missing model.safetensors.index.json: {index_path}")

    if int(args.vae_lora_rank) < 1:
        raise ValueError("--vae_lora_rank must be >= 1.")
    if float(args.vae_lora_alpha) <= 0.0:
        raise ValueError("--vae_lora_alpha must be > 0.")
    if float(args.vae_lora_dropout) < 0.0 or float(args.vae_lora_dropout) >= 1.0:
        raise ValueError("--vae_lora_dropout must satisfy 0 <= dropout < 1.")

    args.decoder_layer_ids = parse_decoder_layers(args.decoder_layers)
    args.target_module_names = parse_target_modules(args.target_modules)


def _copy_resume_state_files(src_dir: str, dst_dir: str) -> List[str]:
    copied: List[str] = []
    for filename in _COPY_FILES:
        src_path = os.path.join(src_dir, filename)
        if not os.path.exists(src_path):
            continue
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy2(src_path, dst_path)
        copied.append(dst_path)

    for src_path in sorted(glob.glob(os.path.join(src_dir, "rng_state*.pth"))):
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        copied.append(dst_path)
    return copied


def _build_extra_meta(args, selection) -> Dict[str, object]:
    return {
        "stage": "e2e_fintuning",
        "source_checkpoint_dir": args.student_checkpoint_dir,
        "legacy_hf_checkpoint_dir": args.legacy_checkpoint_dir,
        "conversion_source_format": "legacy_hf_trainer_sharded_checkpoint",
        "teacher_source": "legacy_conversion_unknown",
        "target_decoder_layers": list(selection.decoder_layer_ids),
        "target_module_names": None if args.target_module_names is None else list(args.target_module_names),
        "loss_type": str(args.loss_type),
        "post_attn": bool(args.post_attn),
        "lora_embedding": bool(args.lora_embedding),
        "lora_lm_head": bool(args.lora_lm_head),
        "lora_hif4_act": bool(args.lora_hif4_act),
        "finetune_mode": _E2E_FINETUNE_MODE,
        "vae_lora_rank": int(args.vae_lora_rank),
        "vae_lora_alpha": float(args.vae_lora_alpha),
        "vae_lora_dropout": float(args.vae_lora_dropout),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)

    print(json.dumps({"legacy_checkpoint_dir": args.legacy_checkpoint_dir, "output_checkpoint_dir": args.output_checkpoint_dir}, ensure_ascii=False))

    model, meta, load_result = load_e2e_model_checkpoint(
        args.student_checkpoint_dir,
        access_token=args.access_token,
        map_location="cpu",
        strict=True,
    )
    print(
        json.dumps(
            {
                "student_checkpoint_dir": args.student_checkpoint_dir,
                "student_missing_keys": len(getattr(load_result, "missing_keys", [])),
                "student_unexpected_keys": len(getattr(load_result, "unexpected_keys", [])),
            },
            ensure_ascii=False,
        )
    )

    if (bool(args.lora_embedding) or bool(args.lora_lm_head)) and _embedding_and_lm_head_are_tied(model):
        separate_embeddings_and_lm_head(model)

    decoder_layer_ids = resolve_target_layer_ids(args.decoder_layer_ids, len(list(get_layers(model))))
    selection = select_e2e_trainables(
        model,
        decoder_layer_ids=decoder_layer_ids,
        target_module_names=args.target_module_names,
        vae_lora_rank=int(args.vae_lora_rank),
        vae_lora_alpha=float(args.vae_lora_alpha),
        vae_lora_dropout=float(args.vae_lora_dropout),
        lora_embedding=bool(args.lora_embedding),
        lora_lm_head=bool(args.lora_lm_head),
    )
    if not selection.trainable_params:
        raise RuntimeError("No trainable parameters found for requested decoder layers.")

    setattr(model, "_e2e_finetune_mode", _E2E_FINETUNE_MODE)
    load_result = load_sharded_checkpoint(model, args.legacy_checkpoint_dir, strict=True, prefer_safe=True)
    print(
        json.dumps(
            {
                "legacy_missing_keys": len(getattr(load_result, "missing_keys", [])),
                "legacy_unexpected_keys": len(getattr(load_result, "unexpected_keys", [])),
                "adapter_modules": len(selection.adapter_modules),
            },
            ensure_ascii=False,
        )
    )

    os.makedirs(args.output_checkpoint_dir, exist_ok=False)
    copied_files = _copy_resume_state_files(args.legacy_checkpoint_dir, args.output_checkpoint_dir)
    save_paths = save_e2e_model_checkpoint(
        model,
        args.output_checkpoint_dir,
        base_model_path=meta.get("base_model_path"),
        save_config=False,
        extra_meta=_build_extra_meta(args, selection),
        compact_unload_vae_original_weights=True,
    )

    print(
        json.dumps(
            {
                "converted_checkpoint_dir": save_paths["output_dir"],
                "copied_resume_files": [os.path.basename(path) for path in copied_files],
                "base_model_path": meta.get("base_model_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
