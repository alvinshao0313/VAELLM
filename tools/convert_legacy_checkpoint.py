import argparse
import glob
import json
import os
import shutil
from typing import Dict, List, Optional

from transformers.modeling_utils import load_sharded_checkpoint

from e2e_common.checkpoint_io import load_e2e_model_checkpoint, save_e2e_model_checkpoint
from e2e_common.peft_proxy import ensure_peft_vae_proxy_adapter
from e2e_common.proxy_trainables import resolve_target_layer_ids, select_e2e_trainables_peft_proxy
from rotation.model_utils import get_layers
from train_utils.train_args import _parse_bool_like


_E2E_FINETUNE_MODE = "vae_lora"
_VALID_VAE_LORA_VARIANTS = {"plain", "rslora", "dora", "adalora"}
_VALID_VAE_LORA_INIT_MODES = {"zero", "gaussian", "residual_svd"}
_TARGET_MODULE_ALIASES = {
    "q": "q_proj",
    "query": "q_proj",
    "k": "k_proj",
    "key": "k_proj",
    "v": "v_proj",
    "value": "v_proj",
    "o": "o_proj",
    "out": "o_proj",
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}
_COPY_FILES = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "training_args.bin",
    "scaler.pt",
)


def parse_decoder_layers(value: Optional[str]) -> Optional[List[int]]:
    raw = str(value or "").strip().lower()
    if raw in {"", "all", "*"}:
        return None
    out = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            begin_text, end_text = [part.strip() for part in token.split("-", 1)]
            begin = int(begin_text)
            end = int(end_text)
            if begin < 0 or end < 0 or end < begin:
                raise argparse.ArgumentTypeError(
                    f"Invalid --decoder_layers range '{token}'. Expected non-negative begin <= end."
                )
            out.update(range(begin, end + 1))
            continue
        idx = int(token)
        if idx < 0:
            raise argparse.ArgumentTypeError(
                f"Invalid --decoder_layers token '{token}'. Expected non-negative layer index."
            )
        out.add(idx)
    if not out:
        raise argparse.ArgumentTypeError("--decoder_layers cannot be empty.")
    return sorted(out)


def parse_target_modules(value: Optional[str]) -> Optional[List[str]]:
    raw = str(value or "").strip().lower()
    if raw in {"", "all", "*"}:
        return None
    out = []
    seen = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        normalized = _TARGET_MODULE_ALIASES.get(token, token)
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    if not out:
        raise argparse.ArgumentTypeError("--target_modules cannot be empty.")
    return out


def parse_vae_lora_variant(value: Optional[str]) -> str:
    norm = str(value or "").strip().lower()
    if not norm:
        norm = "plain"
    if norm not in _VALID_VAE_LORA_VARIANTS:
        raise argparse.ArgumentTypeError(
            f"Invalid --vae_lora_variant '{value}'. Expected one of: {sorted(_VALID_VAE_LORA_VARIANTS)}."
        )
    return norm


def parse_vae_lora_init_mode(value: Optional[str]) -> str:
    norm = str(value or "").strip().lower()
    if not norm:
        norm = "zero"
    if norm not in _VALID_VAE_LORA_INIT_MODES:
        raise argparse.ArgumentTypeError(
            f"Invalid --vae_lora_init_mode '{value}'. Expected one of: {sorted(_VALID_VAE_LORA_INIT_MODES)}."
        )
    return norm


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
    parser.add_argument("--vae_lora_variant", type=parse_vae_lora_variant, default="plain")
    parser.add_argument("--vae_lora_rank", type=int, default=8)
    parser.add_argument("--vae_lora_alpha", type=float, default=16.0)
    parser.add_argument("--vae_lora_dropout", type=float, default=0.0)
    parser.add_argument("--vae_lora_init_mode", type=parse_vae_lora_init_mode, default="zero")
    parser.add_argument("--vae_adalora_target_r", type=int, default=8)
    parser.add_argument("--vae_adalora_init_r", type=int, default=12)
    parser.add_argument("--vae_adalora_tinit", type=int, default=0)
    parser.add_argument("--vae_adalora_tfinal", type=int, default=0)
    parser.add_argument("--vae_adalora_delta_t", type=int, default=1)
    parser.add_argument("--vae_adalora_beta1", type=float, default=0.85)
    parser.add_argument("--vae_adalora_beta2", type=float, default=0.85)
    parser.add_argument("--vae_adalora_orth_reg_weight", type=float, default=0.5)
    parser.add_argument("--loss_type", type=str, default="sft")
    parser.add_argument(
        "--lora_hif4_act",
        type=lambda v: _parse_bool_like(v, arg_name="--lora_hif4_act"),
        default=False,
    )
    return parser


def _load_legacy_trainer_state(legacy_checkpoint_dir: str) -> Dict[str, object]:
    trainer_state_path = os.path.join(legacy_checkpoint_dir, "trainer_state.json")
    with open(trainer_state_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
    if str(args.vae_lora_variant) == "adalora":
        if str(args.vae_lora_init_mode) == "residual_svd":
            raise ValueError("AdaLoRA does not support residual_svd init.")
        if int(args.vae_adalora_target_r) < 1:
            raise ValueError("--vae_adalora_target_r must be >= 1.")
        if int(args.vae_adalora_init_r) < int(args.vae_adalora_target_r):
            raise ValueError("--vae_adalora_init_r must be >= --vae_adalora_target_r.")
        if int(args.vae_adalora_delta_t) < 1:
            raise ValueError("--vae_adalora_delta_t must be >= 1.")
        if not (0.0 < float(args.vae_adalora_beta1) < 1.0):
            raise ValueError("--vae_adalora_beta1 must satisfy 0 < beta1 < 1.")
        if not (0.0 < float(args.vae_adalora_beta2) < 1.0):
            raise ValueError("--vae_adalora_beta2 must satisfy 0 < beta2 < 1.")
    args.decoder_layer_ids = parse_decoder_layers(args.decoder_layers)
    args.target_module_names = parse_target_modules(args.target_modules)
    args.legacy_trainer_state = _load_legacy_trainer_state(args.legacy_checkpoint_dir)


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
    use_rslora = str(args.vae_lora_variant) == "rslora"
    use_dora = str(args.vae_lora_variant) == "dora"
    extra_meta: Dict[str, object] = {
        "stage": "dense_e2e_fintuning",
        "source_checkpoint_dir": args.student_checkpoint_dir,
        "legacy_hf_checkpoint_dir": args.legacy_checkpoint_dir,
        "conversion_source_format": "legacy_hf_trainer_sharded_checkpoint",
        "teacher_source": "legacy_conversion_unknown",
        "target_decoder_layers": list(selection.decoder_layer_ids),
        "target_module_names": None if args.target_module_names is None else list(args.target_module_names),
        "loss_type": str(args.loss_type),
        "lora_hif4_act": bool(args.lora_hif4_act),
        "finetune_mode": _E2E_FINETUNE_MODE,
        "vae_lora_variant": str(args.vae_lora_variant),
        "vae_lora_rank": int(args.vae_lora_rank),
        "vae_lora_alpha": float(args.vae_lora_alpha),
        "vae_lora_dropout": float(args.vae_lora_dropout),
        "vae_lora_init_mode": str(args.vae_lora_init_mode),
        "vae_lora_use_rslora": bool(use_rslora),
        "vae_lora_use_dora": bool(use_dora),
    }
    if str(args.vae_lora_variant) == "adalora":
        max_steps = int(args.legacy_trainer_state.get("max_steps", -1))
        if max_steps <= 0:
            raise ValueError("Legacy trainer_state.json does not contain a valid max_steps for AdaLoRA conversion.")
        extra_meta.update(
            {
                "vae_adalora_target_r": int(args.vae_adalora_target_r),
                "vae_adalora_init_r": int(args.vae_adalora_init_r),
                "vae_adalora_tinit": int(args.vae_adalora_tinit),
                "vae_adalora_tfinal": int(args.vae_adalora_tfinal),
                "vae_adalora_delta_t": int(args.vae_adalora_delta_t),
                "vae_adalora_beta1": float(args.vae_adalora_beta1),
                "vae_adalora_beta2": float(args.vae_adalora_beta2),
                "vae_adalora_orth_reg_weight": float(args.vae_adalora_orth_reg_weight),
                "vae_adalora_total_step": max_steps,
            }
        )
    return extra_meta


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

    decoder_layer_ids = resolve_target_layer_ids(args.decoder_layer_ids, len(list(get_layers(model))))
    selection = select_e2e_trainables_peft_proxy(
        model,
        decoder_layer_ids=decoder_layer_ids,
        target_module_names=args.target_module_names,
    )
    if not selection.peft_proxy_modules:
        raise RuntimeError("No PEFT proxy modules found for requested decoder layers.")

    ensure_peft_vae_proxy_adapter(
        model,
        variant=str(args.vae_lora_variant),
        rank=int(args.vae_lora_rank),
        alpha=float(args.vae_lora_alpha),
        dropout=float(args.vae_lora_dropout),
        init_mode=str(args.vae_lora_init_mode),
        total_step=int(args.legacy_trainer_state.get("max_steps", -1)) if str(args.vae_lora_variant) == "adalora" else None,
        adalora_target_r=int(args.vae_adalora_target_r),
        adalora_init_r=int(args.vae_adalora_init_r),
        adalora_tinit=int(args.vae_adalora_tinit),
        adalora_tfinal=int(args.vae_adalora_tfinal),
        adalora_delta_t=int(args.vae_adalora_delta_t),
        adalora_beta1=float(args.vae_adalora_beta1),
        adalora_beta2=float(args.vae_adalora_beta2),
        adalora_orth_reg_weight=float(args.vae_adalora_orth_reg_weight),
        materialize_before_inject=False,
    )

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
