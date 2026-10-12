"""CPU contracts and immutable inputs for the four real-matrix reconstruction pairs."""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import sys

import torch

from experiments.joint_reconstruction_probe.core import (BATCH_SIZE, STEPS, decode, index_batches,
    rebuild, require, target_blocks)
from experiments.joint_reconstruction_probe.run import source_pins
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.rotation_factor_probe.run import load_archive
from litebsq.autoencoder import Decoder
from litebsq.bitpack import pack_bool_tensor_to_uint8, unpack_uint8_tensor_to_bool


def cpu_contracts():
    """Real classes with small tensors for layout only; no claim of CUDA equivalence."""
    results = {}
    torch.manual_seed(31)
    for category in ("gate_proj", "down_proj"):
        latent = 48 if category == "down_proj" else 32
        target = torch.arange(96 * 128, dtype=torch.float32).view(96, 128) / 10000
        codes, states = [], []
        for _stage in range(2):
            decoder = Decoder(in_dim=latent, out_dim=32, hidden_dim=128, num_res_blocks=0,
                norm_type="layer", activation_type="swish", decoder_type="symmetric",
                use_checkpoint=True).get_sub_decoder(0).to(torch.bfloat16)
            decoder._fuse_q_scale()
            states.append({name: value.detach().clone() for name, value in decoder.state_dict().items()})
            bits = torch.randint(2, (target.numel() // 32, 1, latent), dtype=torch.bool)
            codes.append(pack_bool_tensor_to_uint8(bits, logical_shape=bits.shape))
        payload = {"rotated_weight": target, "stage_codes": codes, "stage_decoders": states}
        layer = rebuild(payload, category)
        with torch.no_grad():
            grouped = torch.cat([unpack_uint8_tensor_to_bool(value,
                logical_shape=(target.numel() // 32, 1, latent)) for value in codes], dim=1).to(torch.bfloat16)
            blocks = layer._parallel_stage_decoder(grouped).sum(dim=1)
            expected = blocks.reshape((128, 96) if layer.transpose else (96, 128))
            if layer.transpose:
                expected = expected.T.contiguous()
            native = decode(layer, deployment=True)
            require(torch.equal(native, expected), "CPU native stage sum or transpose contract failed.")
            require(torch.equal(target_blocks(target, layer.transpose).reshape(
                (128, 96) if layer.transpose else (96, 128)), target.T if layer.transpose else target),
                "The original target block order differs from native layout.")
            layer._parallel_stage_decoder.float()
            require(torch.equal(decode(layer, deployment=False), native), "FP32 masters changed BF16 CPU decode.")
        layer.enable_trainable_decode(parallel_stage_decode=True)
        require(all(value.requires_grad and value.dtype == torch.float32
                    for value in layer._parallel_stage_decoder.parameters()), "Unregistered trainable decoder.")
        try:
            rebuild({**payload, "stage_codes": [codes[0][:-1], codes[1]]}, category)
        except ValueError:
            rejected = True
        else:
            rejected = False
        require(rejected, "Mismatched actual packed bank was not rejected.")
        results[category] = {"native_stage_sum_and_transpose_exact": True,
            "payload_strict_roundtrip_exact": True, "fp32_master_bf16_compute_exact_cpu": True,
            "invalid_bank_shape_rejected": True,
            "decoder_elements": sum(value.numel() for value in layer._parallel_stage_decoder.parameters())}
    batches = list(index_batches(23, steps=8, batch_size=7, seed=31))
    for epoch in (0, 1):
        values = torch.cat([indices for observed, indices in batches if observed == epoch])
        require(torch.equal(values.sort().values, torch.arange(23)), "Epoch drops or repeats weight blocks.")
    repeat = list(index_batches(23, steps=8, batch_size=7, seed=31))
    require(all(a == b and torch.equal(x, y) for (a, x), (b, y) in zip(batches, repeat, strict=True)),
            "Shuffled complete epochs are not deterministic.")
    results["sampler"] = {"two_complete_epochs": True, "partial_batch_retained": True,
                           "deterministic": True, "cpu_only": True}
    require(not torch.cuda.is_initialized(), "CPU preparation initialized CUDA.")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--native-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    args = parser.parse_args()
    require(not args.preparation.exists() and not args.output_dir.exists(), "Do not overwrite an existing preparation/run.")
    torch.set_num_threads(4)
    contracts = cpu_contracts()
    verification = json.loads(args.native_verification.read_text())
    require(verification["status"] == "verified", "Existing native block32 verification is required.")
    check_hashes(verification["source_sha256"])
    input_pins = {str(args.native_verification.resolve()): sha256(args.native_verification)}
    geometry = {}
    for model in ("qwen", "llama"):
        archive, manifest = load_archive(args.data_dir / model)
        require(archive["model_type"] == {"qwen": "qwen3", "llama": "llama"}[model], "Model type changed.")
        for category in ("gate_proj", "down_proj"):
            name = "model.layers.0.mlp." + category
            bundle = archive["targets"][name]
            weight, indices = bundle["reference_weight"], bundle["protected_input_indices"]
            rows, cols = weight.shape
            count = 256 if category == "down_proj" else 64
            require(weight.dtype == torch.bfloat16 and indices.dtype == torch.int64
                    and len(indices) == count and len(indices.unique()) == count
                    and int(indices.min()) >= 0 and int(indices.max()) < cols
                    and count % 32 == 0 and rows % 32 == (cols - count) % 32 == 0,
                    "Actual input-protection geometry violates the fixed block32 contract.")
            blocks = rows * (cols - count) // 32
            latent = 48 if category == "down_proj" else 32
            geometry[f"{model}/{category}"] = {"weight_shape": [rows, cols],
                "unprotected_shape": [rows, cols - count], "protected_count": count,
                "stage_packed_shape": [blocks, 1, latent // 8], "stage_logical_shape": [blocks, 1, latent],
                "transpose": category == "down_proj", "batches_per_epoch": math.ceil(blocks / BATCH_SIZE),
                "joint_steps": STEPS, "capture_sha256": manifest["inputs_sha256"],
                "protection_source_checkpoint_id": archive["checkpoint_id"]}
        for filename in ("data_manifest.json", "inputs.pt"):
            path = (args.data_dir / model / filename).resolve()
            input_pins[str(path)] = sha256(path)
        del archive
    pins = source_pins()
    for path in Path(__file__).resolve().parent.glob("*.py"):
        ast.parse(path.read_text())
    record = {"status": "cpu_ready", "python": sys.executable, "torch": torch.__version__,
        "data_dir": str(args.data_dir.resolve()), "output_dir": str(args.output_dir.resolve()),
        "source_sha256": pins, "input_sha256": input_pins, "cpu_contracts": contracts,
        "actual_capture_geometry": geometry, "cuda_initialized": torch.cuda.is_initialized(),
        "command": [sys.executable, "-m", "experiments.joint_reconstruction_probe.run",
                    "--preparation", str(args.preparation.resolve())]}
    args.preparation.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.preparation, record)
    print(json.dumps({"status": record["status"], "preparation": str(args.preparation.resolve()),
                      "sha256": sha256(args.preparation), "source_count": len(pins), "geometry": geometry}))


if __name__ == "__main__":
    main()
