"""CPU-only contract preparation for native full-matrix bit/decoder fitting."""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

from experiments.full_weight_bits_probe.core import BIT_CONFIG, TOTAL_STEPS
from experiments.joint_reconstruction_probe.core import require
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.rotation_factor_probe.run import load_archive
from sparse_bit_tuning.config import resolve_round_steps, resolve_stable_steps
from sparse_bit_tuning.module import BankSpec, layout_bank_specs


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prior-preparation", type=Path, required=True)
    parser.add_argument("--prior-review", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    require(not (output / "preparation.json").exists() and not (output / "mechanism").exists()
            and not (output / "formal").exists(), "Never overwrite a previous preparation or run.")
    prior = json.loads(args.prior_preparation.read_text())
    review = json.loads(args.prior_review.read_text())
    require(prior["status"] == "cpu_ready" and review["status"] == "verified"
            and review["preparation_sha256"] == sha256(args.prior_preparation), "Verified native reconstruction source required.")
    check_hashes(prior["source_sha256"])
    check_hashes(prior["input_sha256"])
    require(resolve_round_steps(BIT_CONFIG.round_steps, total_optimizer_steps=500, active_ratio=1) == 512
            and resolve_stable_steps(512) == 10, "Native round/early-stop schedule changed.")
    shapes, inputs = {}, {str(args.prior_preparation.resolve()): sha256(args.prior_preparation),
        str(args.prior_review.resolve()): sha256(args.prior_review)}
    for model in ("qwen", "llama"):
        archive, manifest = load_archive(args.data_dir / model)
        bundle = archive["targets"]["model.layers.0.mlp.gate_proj"]
        weight, indices = bundle["reference_weight"], bundle["protected_input_indices"]
        rows, cols = weight.shape
        require(weight.dtype == torch.bfloat16 and len(indices) == 64 and len(indices.unique()) == 64
                and rows % 32 == (cols - 64) % 32 == 0, "Actual gate/protection geometry changed.")
        blocks, bits = rows * (cols - 64) // 32, rows * (cols - 64)
        specs = [BankSpec(canonical_key=f"layer|stage={s}|part=0", module_path="layer", stage_idx=s,
            part_idx=0, logical_shape=(blocks, 1, 32), n_bits=bits, n_active=bits,
            device=torch.device("cpu")) for s in range(2)]
        laid, chunks = layout_bank_specs(specs, score_dtype=torch.float32)
        require(len(chunks) == 1 and chunks[0].numel == 2 * bits
                and laid[0].score_start == 0 and laid[0].score_end == laid[1].score_start
                and laid[1].score_end == 2 * bits, "Native grouped full-score layout differs.")
        shapes[model] = {"original_shape": [rows, cols], "rotated_shape": [rows, cols - 64],
            "blocks": blocks, "packed_bank_shape": [blocks, 1, 4], "active_bits_total": 2 * bits,
            "score_chunks": len(chunks), "fp32_score_grad_adam_m_v_bytes": 2 * bits * 16,
            "persistent_two_banks_bytes": 2 * bits // 8,
            "source_model_type": archive["model_type"], "capture_sha256": manifest["inputs_sha256"],
            "protected_count": 64, "transpose": False}
        for filename in ("data_manifest.json", "inputs.pt"):
            p = (args.data_dir / model / filename).resolve(); inputs[str(p)] = sha256(p)
        del archive
    sources = {str(ROOT / "experiments/longbench_v1/common.py"): sha256(ROOT / "experiments/longbench_v1/common.py")}
    for folder in ("experiments/full_weight_bits_probe", "experiments/joint_reconstruction_probe",
                   "experiments/rotation_factor_probe", "train_utils", "litebsq", "rotation", "sparse_bit_tuning"):
        sources.update({str(p): sha256(p) for p in sorted((ROOT / folder).rglob("*.py"))})
    for path in Path(__file__).resolve().parent.glob("*.py"):
        ast.parse(path.read_text())
    require(not torch.cuda.is_initialized(), "CPU preparation must not initialize CUDA.")
    record = {"status": "cpu_ready", "python": sys.executable, "torch": torch.__version__,
        "source_sha256": sources, "input_sha256": inputs, "output_root": str(output),
        "data_dir": str(args.data_dir.resolve()), "full_geometry": shapes, "bit_config": asdict(BIT_CONFIG),
        "steps": TOTAL_STEPS, "short_stop_steps": 2, "schedule_steps": 500, "wall_budget_seconds": 3600,
        "existing_evidence": "Previous strict native rebuild and four real BF16 packed decoder updates/outputs; this test adds full-matrix SparseBit binding.",
        "new_required_evidence": "Two real updates for both models and both modes, native FP32 score grads/moments, preflip loss/decoder gradients, committed payload, resource/time.",
        "cuda_initialized": False,
        "mechanism_command": [sys.executable, "-m", "experiments.full_weight_bits_probe.run", "--preparation",
                              str(output / "preparation.json"), "--mode", "mechanism"],
        "formal_command": [sys.executable, "-m", "experiments.full_weight_bits_probe.run", "--preparation",
                           str(output / "preparation.json"), "--mode", "formal"]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preparation.json", record)
    print(json.dumps({"status": record["status"], "preparation_sha256": sha256(output / "preparation.json"),
                      "source_count": len(sources), "full_geometry": shapes}))


if __name__ == "__main__":
    main()
