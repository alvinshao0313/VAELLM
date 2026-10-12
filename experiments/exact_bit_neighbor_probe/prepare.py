"""CPU contracts using the real two-model native2k payloads and native sampler."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

import torch

from experiments.exact_bit_neighbor_probe.core import CHUNK, flip_grouped, score_ordinals
from experiments.joint_reconstruction_probe.core import rebuild, require
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.rotation_factor_probe.run import load_archive
from sparse_bit_tuning.sampler import AffineSamplerState


ROOT = Path(__file__).resolve().parents[2]
NAME = "model.layers.0.mlp.gate_proj"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    previous, output = args.prior_root.resolve(), args.output_root.resolve()
    require(not (output / "preparation.json").exists() and not (output / "run").exists(), "Never replace a preparation or run.")
    prior = json.loads((previous / "preparation.json").read_text())
    mechanism = json.loads((previous / "mechanism/report.json").read_text())
    review = json.loads((previous / "review.json").read_text())
    require(review["status"] == "verified" and mechanism["status"] == "verified"
            and review["preparation_sha256"] == sha256(previous / "preparation.json")
            and review["mechanism_sha256"] == sha256(previous / "mechanism/report.json"), "Verified native source required.")
    check_hashes(prior["source_sha256"])
    check_hashes(prior["input_sha256"])
    inputs, models = {}, {}
    for relative in ("preparation.json", "review.json", "mechanism/report.json"):
        path = previous / relative
        inputs[str(path)] = sha256(path)
    for model in ("qwen", "llama"):
        archive, manifest = load_archive(Path(prior["data_dir"]) / model)
        path = previous / "mechanism" / f"{model}_initial_payload.pt"
        require(sha256(path) == mechanism["cases"][model]["initial_payload_file_sha256"], "Fresh initial payload changed.")
        diagnostic = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        layer = rebuild(diagnostic, "gate_proj")
        target = diagnostic["rotated_weight"]
        reference = archive["targets"][NAME]["reference_weight"]
        require(target.shape == (reference.shape[0], reference.shape[1] - 64), "Real gate geometry changed.")
        grouped = torch.stack([layer.get_stage_part_vq_storage(s)[:, 0] for s in range(2)], dim=1)
        sample = grouped[torch.tensor([0, 1, len(grouped) - 1])]
        for candidate in range(64):
            flipped = flip_grouped(sample, candidate)
            diff = (sample ^ flipped).reshape(3, 8)
            stage, bit = divmod(candidate, 32)
            require(bool((torch.count_nonzero(diff, dim=1) == 1).all())
                    and bool((diff[:, stage * 4 + bit // 8] == 1 << (bit % 8)).all()),
                    "Candidate packed LSB position or one-bit contract failed.")
        # Check the vectorized native inverse against the authoritative scalar method,
        # both exhaustively at a tiny bank and at actual full-bank boundary coordinates.
        samplers = []
        for stage in range(2):
            for n in (128, len(grouped) * 32):
                state = AffineSamplerState.create(canonical_key=f"layer|stage={stage}|part=0",
                    training_seed=31, n_bits=n, n_active=n)
                ordinals = list(range(n)) if n == 128 else [0, 1, 7, 8, 31, 32, n // 2, n - 2, n - 1]
                logical = torch.tensor([state.logical_index(q) for q in ordinals], dtype=torch.int64)
                require(torch.equal(score_ordinals(state, logical), torch.tensor(ordinals)),
                        "Inverse mapping differs from native AffineSamplerState.logical_index.")
                if n > 128:
                    samplers.append(state.to_metadata())
        models[model] = {"model_type": archive["model_type"], "checkpoint_id": archive["checkpoint_id"],
            "target_shape": list(target.shape), "blocks": len(grouped), "two_banks_bits": len(grouped) * 64,
            "initial_payload_file": str(path), "initial_payload_sha256": sha256(path),
            "native_samplers": samplers, "baseline": mechanism["cases"][model]["baseline"],
            "real_cpu_strict_rebuild": True, "all_64_packed_single_flips_verified": True,
            "native_full_bank_inverse_verified": True}
        inputs[str(path)] = sha256(path)
        for name in ("data_manifest.json", "inputs.pt"):
            p = Path(prior["data_dir"]) / model / name
            inputs[str(p)] = sha256(p)
        del layer, archive, diagnostic, grouped
    sources = dict(prior["source_sha256"])
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        ast.parse(path.read_text())
        sources[str(path)] = sha256(path)
    require(not torch.cuda.is_initialized(), "CPU preparation must not initialize CUDA.")
    record = {"status": "cpu_ready", "source_sha256": sources, "input_sha256": inputs,
        "models": models, "data_dir": prior["data_dir"], "output_root": str(output),
        "python": sys.executable, "torch": torch.__version__, "cuda_initialized": False,
        "passes": 1, "chunk_blocks": CHUNK, "candidates_per_block": 64,
        "cost_estimate": {"seconds": [120, 360], "peak_allocated_gib": 10,
            "basis": "Existing full-bank gradient took <0.2s/update; 64 forward-only candidates with bounded chunk launch overhead, plus endpoint measurements. Estimate only."},
        "required_runtime_evidence": ["Frozen decoder and native real fullbank gradient with all mapped score signs verified.",
            "Every selected block strictly improves its true same-chunk SSE and mixed selected-code decode equals scored output.",
            "Independent fullpacked/fused/original-coordinate and 32+32 document metrics, without imposing cross-kernel bitwise equality.",
            "No non-code mutation, exactly one bit per selected block, unchanged persistent bit count and compact selection replay."],
        "command": [sys.executable, "-m", "experiments.exact_bit_neighbor_probe.run", "--preparation", str(output / "preparation.json")]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preparation.json", record)
    print(json.dumps({"status": "cpu_ready", "sha256": sha256(output / "preparation.json"),
        "source_count": len(sources), "input_count": len(inputs), "blocks": {m: row["blocks"] for m, row in models.items()}}))


if __name__ == "__main__":
    main()
