"""Run a bounded, real Qwen3 downstream A/B experiment without changing CAT defaults."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

from experiments.activation_loss_ab.evaluation import TASKS, dump_json, evaluate, load_lm, paired_comparison
from experiments.activation_loss_ab.training import MODES, train_weight


def prepare(args):
    from experiments.activation_loss_ab.calibration import collect
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    names = [f"model.layers.{i}.mlp.up_proj" for i in args.layers]
    lm, snapshot = load_lm(args.model, args.device, args.eval_batch_size)
    weights = {name: lm.model.get_submodule(name).weight.detach().cpu().clone() for name in names}
    manifest = {
        "model": args.model, "snapshot": snapshot, "modules": names,
        "layers_zero_based": args.layers, "steps_per_stage": args.steps,
        "vae_batch_size": args.vae_batch_size, "eval_batch_size": args.eval_batch_size,
        "limit_per_task": args.limit, "tasks": list(TASKS), "seeds": args.seeds,
        "calibration_seed": 20260922, "evaluation_seed": 31,
        "modes": list(MODES), "objective_scale": "Raw activation second moments; quadratic divided by 32 to match native AMSE.",
        "weight_normalization": "Production residual-stage mean/std, identical protocol for all objectives.",
        "inference": "Production packed BSQ payload decoded to weights, then BF16 dense weight materialization for task evaluation.",
        "scope": "Only listed full up_proj matrices are compressed to 2 code bits/weight. Other model weights remain original BF16.",
        "rotation": "none", "transpose": False, "channel_protection": "none", "recovery_training": "none",
        "total_model_parameters": sum(p.numel() for p in lm.model.parameters()),
        "compressed_parameters": sum(w.numel() for w in weights.values()),
        "source_weights": {
            name: {"shape": list(w.shape), "sha256": hashlib.sha256(w.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            for name, w in weights.items()
        },
        "python": sys.executable, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "status": "PREPARING",
    }
    dump_json(out / "manifest.json", manifest)
    torch.save(weights, out / "original_weights.pt")
    grams, calib_meta = collect(lm, names, nsamples=args.calib_samples, seqlen=args.calib_length, seed=manifest["calibration_seed"])
    torch.save(grams, out / "activation_grams.pt")
    dump_json(out / "calibration.json", calib_meta)
    base = out / "bf16"
    base.mkdir()
    with (base / "evaluation.log").open("w") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        accuracy = evaluate(lm, limit=args.limit, batch_size=args.eval_batch_size, seed=31, output=base)
    manifest["status"] = "PREPARED"
    dump_json(out / "manifest.json", manifest)
    print(json.dumps({"status": "PREPARED", "baseline_macro_accuracy": accuracy["macro_accuracy"], "tasks": accuracy["tasks"]}), flush=True)


def variant(args):
    root = Path(args.output_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "PREPARED" or args.seed not in manifest["seeds"]:
        raise ValueError("Run is not prepared or seed was not prespecified.")
    out = root / f"{args.mode}_seed{args.seed}"
    out.mkdir(exist_ok=False)
    weights = torch.load(root / "original_weights.pt", map_location="cpu", weights_only=True)
    grams = torch.load(root / "activation_grams.pt", map_location="cpu", weights_only=True)
    decoded, records = {}, []
    with (out / "training.log").open("w") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        for name in manifest["modules"]:
            reconstructed, record = train_weight(
                weights[name], module_name=name, mode=args.mode, grams=grams[name], seed=args.seed,
                steps=manifest["steps_per_stage"], batch_size=manifest["vae_batch_size"], device=args.device,
                save_path=out / (name + ".pt"),
            )
            decoded[name] = reconstructed.to(torch.bfloat16)
            records.append(record)
            dump_json(out / "training_records.json", records)
    print(f"TRAINING_COMPLETE mode={args.mode} seed={args.seed}", flush=True)
    lm, snapshot = load_lm(manifest["model"], args.device, manifest["eval_batch_size"])
    if snapshot != manifest["snapshot"]:
        raise RuntimeError("Model snapshot changed after baseline evaluation.")
    with torch.no_grad():
        for name in manifest["modules"]:
            layer = lm.model.get_submodule(name)
            if not torch.equal(layer.weight.detach().cpu(), weights[name]):
                raise RuntimeError(f"Base weights changed: {name}")
            layer.weight.copy_(decoded[name].to(layer.weight.device))
    del decoded, weights, grams
    torch.cuda.empty_cache()
    with (out / "evaluation.log").open("w") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        accuracy = evaluate(
            lm, limit=manifest["limit_per_task"], batch_size=manifest["eval_batch_size"],
            seed=manifest["evaluation_seed"], output=out,
        )
    baseline = json.loads((root / "bf16" / "accuracy.json").read_text())
    comparison = paired_comparison(baseline, accuracy)
    dump_json(out / "versus_bf16.json", comparison)
    print(json.dumps({"mode": args.mode, "seed": args.seed, "macro_accuracy": accuracy["macro_accuracy"], "tasks": accuracy["tasks"]}), flush=True)


def summarize(args):
    root = Path(args.output_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    baseline = json.loads((root / "bf16" / "accuracy.json").read_text())
    rows, comparisons = [], {}
    table = ["| Objective | Seed | PIQA | ARC-E | ARC-C | Mean |", "|---|---:|---:|---:|---:|---:|"]
    def row_text(mode, seed, accuracy):
        values = [accuracy["tasks"][t]["accuracy"] * 100 for t in TASKS]
        return f"| {mode} | {seed} | " + " | ".join(f"{v:.2f}" for v in values + [accuracy["macro_accuracy"] * 100]) + " |"
    table.append(row_text("BF16", "-", baseline))
    for seed in manifest["seeds"]:
        by_mode = {}
        reference_hashes = None
        for mode in MODES:
            folder = root / f"{mode}_seed{seed}"
            accuracy = json.loads((folder / "accuracy.json").read_text())
            training = json.loads((folder / "training_records.json").read_text())
            hashes = [[s["initial_state_sha256"] for s in item["stages"]] for item in training]
            if reference_hashes is None:
                reference_hashes = hashes
            elif hashes != reference_hashes:
                raise RuntimeError("Initial VAE weights differ between objectives.")
            by_mode[mode] = accuracy
            rows.append({"mode": mode, "seed": seed, "macro_accuracy": accuracy["macro_accuracy"], "tasks": accuracy["tasks"]})
            table.append(row_text(mode, seed, accuracy))
        comparisons[str(seed)] = {
            "block_output_vs_mse": paired_comparison(by_mode["mse"], by_mode["block_output"]),
            "block_output_vs_amse": paired_comparison(by_mode["amse"], by_mode["block_output"]),
            "amse_vs_mse": paired_comparison(by_mode["mse"], by_mode["amse"]),
        }
    means = {mode: sum(r["macro_accuracy"] for r in rows if r["mode"] == mode) / len(manifest["seeds"]) for mode in MODES}
    result = {"manifest": manifest, "rows": rows, "mean_accuracy_by_mode": means, "paired_comparisons": comparisons}
    dump_json(root / "summary.json", result)
    text = "# VAE activation-loss downstream A/B\n\n" + "\n".join(table) + "\n\n"
    text += "Only the prespecified full up_proj matrices are compressed. This is not an all-layer 2-bit result.\n\n"
    text += "Task score: acc_norm, 0-shot, no chat template, same fixed evaluation items. No evaluation data were used to train or select checkpoints.\n\n"
    text += "Training settings were fixed before seeing task accuracy; final step is evaluated, not the best task-scoring checkpoint.\n\n"
    text += "Question bootstrap intervals do not measure training-seed uncertainty. Two training seeds are only an exploratory screen.\n\n"
    text += "Paired comparisons and initialization parity are in summary.json.\n"
    (root / "REPORT.md").write_text(text)
    print(text, flush=True)
    print(json.dumps({"means": means, "comparisons": comparisons}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "variant", "summarize"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 17, 35])
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 47])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--vae-batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--calib-samples", type=int, default=32)
    parser.add_argument("--calib-length", type=int, default=512)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action == "variant" and (args.mode is None or args.seed is None):
        parser.error("variant requires --mode and --seed")
    if any(getattr(args, name) < 1 for name in ("steps", "vae_batch_size", "eval_batch_size", "limit", "calib_samples", "calib_length")):
        parser.error("Sample counts, lengths, steps and batch sizes must be positive.")
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"Set {key}=1 before invoking this offline experiment.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    {"prepare": prepare, "variant": variant, "summarize": summarize}[args.action](args)


if __name__ == "__main__":
    main()
