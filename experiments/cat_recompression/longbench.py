"""Use the existing LongBench runner with this experiment's native rank0/2.6 budget."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01"
sys.path.insert(0, str(SNAPSHOT))

import torch
from experiments.longbench_v1 import runner
from experiments.longbench_v1.common import check_hashes
from experiments.longbench_v1.protocol import model_policy
from train_utils.checkpoint_bpw import audit_state_dict
from train_utils.v6_model_loader import load_v6_model_checkpoint
from tools.cat_eval import _prepare_model_for_eval


def load_model(config, logger):
    record = config["model"]
    declaration = record["recompression"]
    check_hashes(declaration["source_sha256"])
    if (record["kind"] != "compressed" or declaration["max_linear_bpw"] != 2.6
            or declaration["max_lora_rank"] != 0 or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1):
        raise ValueError("Use one assigned GPU and the actual native rank0/2.6 endpoint.")
    torch.cuda.reset_peak_memory_stats()
    model, meta, loaded = load_v6_model_checkpoint(record["checkpoint"], base_model_path=record["hf_path"],
                                                  map_location="cpu", strict=True)
    if (loaded.missing_keys or loaded.unexpected_keys or meta["checkpoint_id"] != record["checkpoint_id"]
            or meta.get("lora_config") is not None or meta["pending_dense_targets"]
            or meta["converted_module_count"] != 252):
        raise ValueError("The actual complete checkpoint must load fresh and strict without adapters.")
    budget = audit_state_dict(meta, model.state_dict(), max_bpw=2.6, max_lora_rank=0)
    expected = declaration["bit_budget"]
    keys = ("components", "user_linear_bits", "packed_linear_bits", "outside_linear_bits", "original_linear_weights")
    if (not budget["budget_passed"] or budget["contract_violations"]
            or budget["components"].get("residual_lora", {}).get("storage_bits", 0)
            or any(budget[key] != expected[key] for key in keys)):
        raise ValueError("The fresh actual projection-storage/rank budget changed.")
    model.eval().requires_grad_(False)
    warmup = _prepare_model_for_eval(model, config["device"], 8, logger, {})
    policy = model_policy(record["hf_path"])
    if (model.config.model_type != policy["model_type"] or model.config.rope_scaling != policy["rope_scaling"]
            or model.config.rope_theta != policy["rope_theta"]
            or model.config._attn_implementation != policy["attn_implementation"]
            or model.get_input_embeddings().weight.dtype != torch.bfloat16):
        raise ValueError("Keep the original BF16/FA2/RoPE model policy.")
    return model, dict(strict_missing=[], strict_unexpected=[], checkpoint_id=meta["checkpoint_id"],
        persistent_tensors=len(model.state_dict()), bit_budget=budget, warmup=warmup, model_policy=policy,
        rope_buffers={name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                      for name, value in model.named_buffers() if "inv_freq" in name},
        load_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        load_peak_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if json.loads(args.config.read_text())["mode"] != "full":
        parser.error("This entry evaluates all21 official tasks.")
    runner.load_model = load_model
    result = runner.run(args.config, args.output)
    print(json.dumps({key: result[key] for key in ("status", "new_predictions", "already_present", "duration_seconds")}))


if __name__ == "__main__":
    main()
