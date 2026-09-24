"""Native packed VAELinear/v6 artifacts, not pickled full weight substitutes."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from train_utils.cat_train_pipeline import apply_group_vae_payload
from train_utils.checkpoint_v6 import save_v6_full_checkpoint, load_v6_full_checkpoint_into_model
from train_utils.utils import LinearRef


def dump_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor_digest(value).encode())
    return digest.hexdigest()


def linear_host(name: str, linear: nn.Linear) -> nn.Module:
    """Build a state-only host with native indexed decoder containers.

    The v6 loader inspects model.layers even for a standalone Linear. A short
    test name therefore gets one parameter-free decoder slot; the serialized
    parameter names and values remain exactly those of the requested Linear.
    """
    host = nn.Module()
    parent = host
    pieces = name.split(".")
    for index, part in enumerate(pieces[:-1]):
        child = nn.ModuleList() if pieces[index + 1].isdecimal() else nn.Module()
        if isinstance(parent, nn.ModuleList):
            slot = int(part)
            parent.extend(nn.Module() for _ in range(slot + 1 - len(parent)))
            parent[slot] = child
        else:
            parent.add_module(part, child)
        parent = child
    parent.add_module(pieces[-1], copy.deepcopy(linear).cpu())
    if not hasattr(host, "model"):
        host.add_module("model", nn.Module())
    if not hasattr(host.model, "layers"):
        host.model.add_module("layers", nn.ModuleList([nn.Module()]))
    return host


def load_linear(name: str, original: nn.Linear, checkpoint: Path):
    host = linear_host(name, original)
    host, _, _ = load_v6_full_checkpoint_into_model(host, str(checkpoint), expected_kind="final_model")
    return host.get_submodule(name)


def export_linear(name: str, original: nn.Linear, payload: dict, expected_weight: torch.Tensor,
                  checkpoint: Path, args, *, training_weight: torch.Tensor | None = None) -> dict:
    """Validate packed export against an unfused, fixed-bit FP32 reference.

    ``training_weight`` is the training decoder's output (possibly BF16). Its
    difference from deployment is reported separately, never used as an export
    parity tolerance.
    """
    host = linear_host(name, original)
    source = host.get_submodule(name)
    refs = [LinearRef(name, source, name.rsplit(".", 1)[-1], False)]
    apply_group_vae_payload(model=host, group_refs=refs, group_tag=name,
                            payload=payload, convert_device=args.device)
    # Validate mathematical packing/fusion in IEEE FP32 on CPU. The native
    # CUDA decoder can use TF32 internally despite requesting float32 tensors.
    compressed = host.get_submodule(name).cpu()
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
        packed_fp32 = compressed._decode_weight(dtype=torch.float32).detach().cpu()
    if not torch.isfinite(packed_fp32).all() or not torch.isfinite(expected_weight).all():
        raise RuntimeError(f"Nonfinite packed decode: {name}")
    rel = float((packed_fp32 - expected_weight).norm() / expected_weight.norm().clamp_min(1e-12))
    tolerance = 5e-6
    if rel > tolerance:
        raise RuntimeError(f"FP32 unfused-to-packed relative L2 {rel:.6g} exceeds {tolerance} for {name}")
    compressed.to(args.device)
    with torch.no_grad(), torch.autocast(device_type=torch.device(args.device).type, enabled=False):
        decoded = compressed._decode_weight(dtype=torch.float32).detach().cpu()
    if not torch.isfinite(decoded).all():
        raise RuntimeError(f"Nonfinite deployment decode: {name}")
    deployment_delta = float((decoded - packed_fp32).norm() / packed_fp32.norm().clamp_min(1e-12))
    precision_delta = None
    if training_weight is not None:
        precision_delta = float((decoded - training_weight).norm() / training_weight.norm().clamp_min(1e-12))
    packed_bytes = sum(compressed.get_stage_part_vq_storage(stage_idx=s, part_idx=0).numel()
                       for s in range(args.residual_stages))
    state_bytes = sum(t.numel() * t.element_size() for t in compressed.state_dict().values())
    save_v6_full_checkpoint(
        host.cpu(), str(checkpoint), checkpoint_kind="final_model", compressed_targets=[name],
        base_model_path=args.model_path, save_config=False,
        extra_meta={"algorithm": "independent_linear_output", "objective": args.objective,
                    "transpose": False, "standalone_linear_only": True,
                    "export_reference": "fixed_bits_unfused_fp32",
                    "export_validation_device": "cpu",
                    "export_validation_decoder_dtype": str(packed_fp32.dtype),
                    "deployment_decode_device": str(args.device),
                    "deployment_decode_tensor_dtype": str(decoded.dtype),
                    "deployment_to_fp32_relative_l2": deployment_delta,
                    "training_to_packed_relative_l2": precision_delta},
    )
    restored = load_linear(name, original, checkpoint).to(args.device)
    with torch.no_grad(), torch.autocast(device_type=torch.device(args.device).type, enabled=False):
        roundtrip = restored._decode_weight(dtype=torch.float32).detach().cpu()
    torch.testing.assert_close(roundtrip, decoded, rtol=1e-6, atol=1e-7)
    result = dict(
        fp32_unfused_to_packed_relative_l2=rel, fp32_export_tolerance=tolerance,
        training_to_packed_relative_l2=precision_delta,
        export_reference="fixed_bits_unfused_fp32",
        export_validation_device="cpu",
        export_validation_decoder_dtype=str(packed_fp32.dtype),
        deployment_decode_device=str(args.device),
        deployment_decode_tensor_dtype=str(decoded.dtype),
        deployment_to_fp32_relative_l2=deployment_delta,
        deployment_decoder_parameter_dtypes=sorted({str(p.dtype) for p in compressed.parameters()}),
        deployment_decoder_compute_dtype_override=str(getattr(compressed, "_decoder_compute_dtype", None)),
        packed_roundtrip_max_abs=float((roundtrip - decoded).abs().max()),
        code_payload_bpw=packed_bytes * 8 / original.weight.numel(),
        total_linear_state_bpw=state_bytes * 8 / original.weight.numel(),
    )
    return result
