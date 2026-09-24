"""Train the isolated A/B workflow and publish a native v6 final_model."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_utils.checkpoint_v6 import save_v6_full_checkpoint
from .artifacts import dump_json, load_linear, tensor_digest
from .calibration import CalibrationStream, build_bundle
from .compression import train_linear
from .config import parser, stage_steps, validate


def select_linears(model: nn.Module, args) -> list[str]:
    if args.target_linears:
        names = list(args.target_linears)
    else:
        categories = [c.strip() for c in args.compression_categories.split(",") if c.strip()]
        names = [name for name, module in model.named_modules()
                 if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in categories]
        missing = set(categories) - {name.rsplit(".", 1)[-1] for name in names}
        if missing:
            raise ValueError(f"Categories not present in this model: {sorted(missing)}")
    if not names or len(names) != len(set(names)):
        raise ValueError("Target list must be nonempty and have no duplicates.")
    for name in names:
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Linear) or layer.in_features % 32:
            raise ValueError(f"Expected a full Linear with input width divisible by 32: {name}")
    return names


def run(model, tok, args) -> dict:
    validate(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    model.eval().requires_grad_(False)
    names = select_linears(model, args)
    bundle = build_bundle(args, tok)
    manifest = {
        "algorithm": "independent_linear_output", "status": "TRAINING", "config": vars(args),
        "modules": names, "steps_per_stage": stage_steps(args.steps, args.residual_stages),
        "transpose": False, "rotation": "none", "channel_protection": "none",
        "teacher": "Original model; previously compressed matrices are NOT used for calibration.",
        "mask": "All attention_mask-valid positions, including prompt; no next-token shift.",
        "data_sources": bundle.source_stats, "dataset_mix_resolved": bundle.dataset_mix_spec,
        "main_loss": "mean((X @ (decoded_physical_residual - target_residual).T)**2) for output mode; normalized weight MSE for control.",
        "auxiliary": "One unchanged production BSQ call per full Linear/stage/update; no per-chunk entropy averaging.",
        "records": [],
    }
    dump_json(out / "manifest.json", manifest)
    try:
        for index, name in enumerate(names):
            manifest["current_module"] = name
            dump_json(out / "manifest.json", manifest)
            stream = CalibrationStream(model, tok, bundle, args)
            linear = model.get_submodule(name)
            source_hash = tensor_digest(linear.weight)
            destination = out / "linears" / name
            print(f"LINEAR {index + 1}/{len(names)} {name} shape={tuple(linear.weight.shape)}", flush=True)
            record = train_linear(linear, name=name, args=args, next_inputs=stream.next_inputs, output=destination)
            if tensor_digest(linear.weight) != source_hash:
                raise RuntimeError("Compression modified the original calibration teacher.")
            record["calibration"] = stream.metadata()
            dump_json(destination / "record.json", record)
            manifest["records"].append(record)
            dump_json(out / "manifest.json", manifest)
            del stream
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
        model.cpu()
        for name in names:
            packed = load_linear(name, model.get_submodule(name), out / "linears" / name / "packed")
            model.set_submodule(name, packed)
        result = save_v6_full_checkpoint(
            model, str(out / "final_model"), checkpoint_kind="final_model", compressed_targets=names,
            compression_categories=sorted({n.rsplit(".", 1)[-1] for n in names}),
            base_model_path=args.model_path, tokenizer=tok,
            extra_meta={"algorithm": "independent_linear_output", "objective": args.objective,
                        "experiment_config": vars(args), "transpose": False},
        )
        manifest["status"] = "COMPLETE"
        manifest["final_model"] = result["output_dir"]
        manifest.pop("current_module", None)
        dump_json(out / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = str(exc)
        dump_json(out / "manifest.json", manifest)
        raise


def main():
    args = parser().parse_args()
    validate(args)
    if Path(args.output_dir).exists():
        raise FileExistsError(args.output_dir)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This isolated workflow is single-device; use python, not torchrun.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = Path(args.model_path)
    if not root.is_dir():
        root = Path(snapshot_download(repo_id=args.model_path, local_files_only=True))
    args.model_path = str(root.resolve())
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": tok.eos_token})
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, low_cpu_mem_usage=True,
    ).to(args.device)
    result = run(model, tok, args)
    print(f"COMPLETE {result['final_model']}", flush=True)


if __name__ == "__main__":
    main()
