"""Run native A/B checkpoints through VAELLM's unchanged LM-eval task utility."""
import argparse
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from experiments.liftquant_recovery.recovery_runtime import clear_caches, prime_packed_cache
from train_utils.eval_utils import run_lm_eval
from train_utils.v6_model_loader import load_v6_model_checkpoint

TASKS = "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu"


@contextmanager
def streamed_inference(model, device):
    """Inference-only residency hooks; outputs stay on GPU, state stays native."""
    handles = []
    model.config.use_cache = False
    roots = [model.model.embed_tokens, model.model.rotary_emb, model.model.norm, model.lm_head]
    for module in roots:
        module.to(device)
    def before(block, _args):
        block.to(device)
        prime_packed_cache(block)
    def after(block, _args, output):
        clear_caches(block)
        block.cpu()
        return output
    try:
        for block in model.model.layers:
            handles.append(block.register_forward_pre_hook(before))
            handles.append(block.register_forward_hook(after))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for block in model.model.layers:
            clear_caches(block)
            block.cpu()
        for module in roots:
            module.cpu()
        torch.cuda.empty_cache()


@contextmanager
def resident_inference(model, device):
    """Keep the same native packed-BF16 weights resident for every eval request."""
    model.config.use_cache = False
    try:
        model.to(device)
        for block in model.model.layers:
            prime_packed_cache(block)
        yield
    finally:
        for block in model.model.layers:
            clear_caches(block)
        model.cpu()
        torch.cuda.empty_cache()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default=TASKS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu-memory-gib", type=float, default=12)
    parser.add_argument("--residency", choices=("streamed", "resident"), default="streamed",
                        help="resident keeps the same packed-BF16 decode cached on GPU for all requests")
    parser.add_argument("--only", choices=("A", "B"), help="evaluate one label; both input metadata still define the comparison scope")
    return parser.parse_args(argv)


def validate_result(result, tasks, limit):
    """Check requested task/group metrics, not MMLU's individual subject names."""
    expected = [name.strip() for name in tasks.split(",") if name.strip()]
    if (result.get("tasks") != expected or result.get("num_fewshot") != 0
            or result.get("batch_size") != 1 or result.get("limit") != limit):
        raise ValueError("LM-eval result protocol differs from the requested zero-shot batch-1 comparison.")
    metrics = result.get("task_metrics", {})
    for name in expected:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Missing/nonfinite requested task or group metric: {name}")


def main():
    args = arguments()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(args.gpu_memory_gib * 2**30 / total, 1), device)
    torch.set_num_threads(4)
    metas = [json.loads((Path(p) / "checkpoint_meta.json").read_text()) for p in (args.a, args.b)]
    for key in ("base_model_path", "compressed_targets", "pending_dense_targets"):
        if metas[0][key] != metas[1][key]:
            raise ValueError(f"A/B scope mismatch: {key}")
    labels = [(label, source) for label, source in (("A", args.a), ("B", args.b))
              if args.only is None or args.only == label]
    config = dict(tasks=args.tasks, num_fewshot=0, batch_size="1", lm_limit=args.limit,
                  model_path=metas[0]["base_model_path"], eval_log_dir=None,
                  eval_run_ts=None, mmlu_debug_samples=0, mmlu_debug_log_dir=str(output),
                  mmlu_debug_run_ts=None)
    (output / "config.json").write_text(json.dumps(dict(
        config, residency=args.residency, labels=[label for label, _ in labels],
        checkpoints=dict(labels), decode="native packed-u8 BF16 cache",
        seed_policy="unchanged run_lm_eval / lm_eval.simple_evaluate defaults",
    ), indent=2))
    inference = resident_inference if args.residency == "resident" else streamed_inference
    for label, source in labels:
        model, _, load = load_v6_model_checkpoint(source, map_location="cpu", strict=True)
        if load.missing_keys or load.unexpected_keys:
            raise ValueError("Strict evaluation load failed.")
        model.eval().requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False)
        with inference(model, device), torch.no_grad():
            result = run_lm_eval(model, tokenizer, SimpleNamespace(**config))
        validate_result(result, args.tasks, args.limit)
        (output / f"{label}.json").write_text(json.dumps(result, indent=2, default=str))
        print(f"{label} evaluation complete", flush=True)
        del model
        gc.collect()
    (output / "summary.json").write_text(json.dumps(dict(
        status="PASS", tasks=args.tasks, sample_limit=args.limit,
        labels=[label for label, _ in labels], residency=args.residency,
        num_fewshot=0, batch_size=1, decode="native packed-u8 BF16 cache",
        metrics="unchanged run_lm_eval task metrics; MMLU uses the existing aggregate group",
        historical_baseline_reused=False,
        scope="identical existing train_utils.eval_utils.run_lm_eval path and strict v6 loading",
    ), indent=2))


if __name__ == "__main__":
    main()
