"""Evaluate saved W2 Linears with explicitly documented remaining dense layers.

This produces a PARTIAL compression result, never a full-W2 result. No source
artifacts or running trainers are modified. GPU memory usage is bounded.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download

from .artifacts import dump_json, load_linear, tensor_digest
from train_utils.checkpoint_v6 import save_v6_full_checkpoint

CATEGORIES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
FAST_TASKS = "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa"


def collect_saved(source):
    saved = {}
    for lo, hi in ((0, 9), (9, 18), (18, 27), (27, 36)):
        worker = source / f"worker_{lo}_{hi}"
        for path in sorted((worker / "linears").glob("*/record.json")):
            rec = json.loads(path.read_text())
            name = rec["module"]
            if name in saved or name != path.parent.name:
                raise ValueError(f"Duplicate/misnamed source artifact: {path}")
            if (rec["steps"], rec["residual_stages"], rec["code_payload_bpw"], rec["objective"]) != (
                    5000, 1, 2.0, "linear_output_mse"):
                raise ValueError(f"Unexpected source settings: {path}")
            packed = path.parent / "packed"
            if not (packed / "checkpoint_meta.json").is_file() or not (packed / "pytorch_model.bin").is_file():
                raise FileNotFoundError(f"Incomplete source checkpoint: {packed}")
            saved[name] = (packed, rec)
    if len(saved) != 192:
        raise ValueError(f"Expected exactly 192 saved targets, found {len(saved)}")
    return saved


def build_partial(source, out):
    saved = collect_saved(source)
    base = Path(snapshot_download(repo_id="Qwen/Qwen3-8B", local_files_only=True)).resolve()
    model = AutoModelForCausalLM.from_pretrained(str(base), torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True, local_files_only=True).cpu()
    targets = {n for n,m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] in CATEGORIES}
    if len(targets) != 252 or not set(saved) < targets:
        raise RuntimeError("Source modules do not match the expected partial model")
    pending = sorted(targets - set(saved))
    for i, (name, (packed, rec)) in enumerate(sorted(saved.items()), 1):
        original = model.get_submodule(name)
        if tensor_digest(original.weight.detach().cpu().float()) != rec["source_weight_sha256"]:
            raise RuntimeError(f"Base-model weight mismatch: {name}")
        if list(original.weight.shape) != rec["shape"]:
            raise RuntimeError(f"Source shape mismatch: {name}")
        layer = load_linear(name, original, packed).cpu()
        parent, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attr, layer)
        if i % 16 == 0:
            print(f"ASSEMBLE {i}/192", flush=True)
    for name in pending:
        layer = model.get_submodule(name)
        if not isinstance(layer, torch.nn.Linear) or layer.weight.dtype != torch.bfloat16:
            raise RuntimeError(f"Expected untouched BF16 Linear: {name}")
    scope = dict(evaluation_scope="PARTIAL_192_OF_252_NOT_FULL_W2",
                 compressed_count=192, dense_target_count=60, target_count=252,
                 dense_target_dtype="bfloat16", compressed_targets=sorted(saved),
                 pending_dense_targets=pending, source_root=str(source),
                 source_checkpoints={n:str(p) for n,(p,_) in saved.items()},
                 note="lm_head/embedding/norm remain original; no LoRA or E2E recovery.")
    ckpt = out / "partial_model"
    save_v6_full_checkpoint(model, str(ckpt), checkpoint_kind="category_boundary",
        compressed_targets=sorted(saved), pending_dense_targets=pending,
        compression_categories=list(CATEGORIES), target_modules=list(CATEGORIES),
        target_layers=list(range(36)), base_model_path=str(base), save_config=True,
        extra_meta={**scope, "algorithm":"independent_linear_output", "objective":"linear_output_mse"})
    dump_json(out / "scope.json", scope)
    del model
    gc.collect()
    return ckpt, scope


def save_results(log_dir, path):
    logs = sorted(log_dir.glob("cat_eval_*.log"))
    if not logs:
        raise RuntimeError("No evaluation log was produced")
    text = logs[-1].read_text(errors="replace")
    if "All evaluations completed." not in text:
        raise RuntimeError("Evaluation did not finish")
    summary, _ = json.JSONDecoder().raw_decode(text.rsplit("Evaluation summary:\n", 1)[1].lstrip())
    summary["evaluation_scope"] = "PARTIAL_192_OF_252_NOT_FULL_W2"
    dump_json(path, summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--gpu_memory_fraction", type=float, default=0.35)
    p.add_argument("--resume", action="store_true", help="Reuse the saved partial model and completed task groups")
    args = p.parse_args()
    if not 0 < args.gpu_memory_fraction < 1:
        p.error("GPU memory fraction must be between 0 and 1")
    torch.set_num_threads(4)
    out = args.output_dir.resolve()
    if args.resume:
        if not (out / "scope.json").is_file() or not (out / "partial_model" / "pytorch_model.bin").is_file():
            p.error("Resume requires a saved partial model and scope.json")
    else:
        out.mkdir(parents=True, exist_ok=False)
    state = {"status":"BUILDING_PARTIAL", "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "evaluation_scope":"PARTIAL_192_OF_252_NOT_FULL_W2", "gpu_memory_fraction":args.gpu_memory_fraction,
             "batch_size":1, "num_fewshot":0, "sample_limit":None,
             "prewarm_dtype":"bfloat16", "decoder_parameters":"unchanged", "resumed":args.resume}
    dump_json(out / "status.json", state)
    try:
        if args.resume:
            checkpoint = out / "partial_model"
            scope = json.loads((out / "scope.json").read_text())
            if (scope["evaluation_scope"] != "PARTIAL_192_OF_252_NOT_FULL_W2"
                    or scope["compressed_count"] != 192 or scope["dense_target_count"] != 60
                    or Path(scope["source_root"]).resolve() != args.source.resolve()):
                raise RuntimeError("Resume scope does not match requested partial model")
        else:
            checkpoint, scope = build_partial(args.source.resolve(), out)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
        from tools.cat_eval import main as evaluate
        import litebsq.vae_linear as vae_linear
        original_prime = vae_linear.prime_named_vae_linear_cache

        def prime_for_bf16_forward(named_targets, **kwargs):
            # Match the native forward's activation dtype without converting decoder parameters.
            kwargs["dtype"] = torch.bfloat16
            return original_prime(named_targets, **kwargs)

        results = {}
        for group, tasks in (("seven_tasks", FAST_TASKS), ("mmlu", "mmlu")):
            state["status"] = "EVALUATING_" + group.upper()
            dump_json(out / "status.json", state)
            log_dir = out / group
            result_path = out / f"{group}_results.json"
            if args.resume and result_path.is_file():
                results[group] = json.loads(result_path.read_text())
            else:
                with patch.object(vae_linear, "prime_named_vae_linear_cache", prime_for_bf16_forward):
                    evaluate(["--checkpoint_dir", str(checkpoint), "--eval_lm_eval", "--tasks", tasks,
                              "--num_fewshot", "0", "--lm_batch_size", "1", "--eval_device", "cuda",
                              "--prewarm_group_size", "1", "--eval_hif4_act", "false", "--eval_log_dir", str(log_dir)])
                results[group] = save_results(log_dir, result_path)
            lm = results[group]["evals"]["lm_eval"]
            if lm["limit"] is not None or set(lm["tasks"]) != set(tasks.split(",")):
                raise RuntimeError("Unexpected task coverage or evaluation subset")
            if any(not isinstance(lm["task_metrics"].get(t), (float, int))
                   or not math.isfinite(lm["task_metrics"][t]) for t in tasks.split(",")):
                raise RuntimeError("Missing or nonfinite task metric")
            state[group] = "COMPLETE"
            dump_json(out / "status.json", state)
            gc.collect()
            torch.cuda.empty_cache()
        metrics = {}
        for result in results.values():
            metrics.update(result["evals"]["lm_eval"]["task_metrics"])
        dump_json(out / "partial_results.json", {"scope":scope, "task_metrics":metrics,
                "num_fewshot":0, "batch_size":1, "limit":None})
        state["status"] = "COMPLETE"
    except Exception as exc:
        state.update(status="FAILED", error=repr(exc))
        raise
    finally:
        dump_json(out / "status.json", state)


if __name__ == "__main__":
    main()
