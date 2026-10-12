"""Restore specified projections directly from official BF16 safetensors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from litebsq.vae_linear import VAELinear


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_inventory(model, excluded):
    def keep(name):
        return not any(name.startswith(target + ".") for target in excluded)
    return {"state": {name: tensor_digest(value) for name, value in model.state_dict().items() if keep(name)},
            "buffers": {name: tensor_digest(value) for name, value in model.named_buffers() if keep(name)},
            "parameter_identity": {name: [id(value), value.data_ptr(), value._version]
                                   for name, value in model.named_parameters() if keep(name)}}


def restore_targets(model, hf_path, targets):
    targets = tuple(targets)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Specify unique, nonempty projection names.")
    before = frozen_inventory(model, targets)
    hf_path = Path(hf_path)
    index = json.loads((hf_path / "model.safetensors.index.json").read_text())["weight_map"]
    replacements = {}
    for name in targets:
        old = model.get_submodule(name)
        if not isinstance(old, VAELinear):
            raise ValueError(f"Expected an original compressed projection: {name}")
        tensors = {}
        for suffix in ("weight", "bias"):
            key = name + "." + suffix
            if key not in index:
                if suffix == "weight" or old.bias is not None:
                    raise ValueError(f"Missing original HF parameter: {key}")
                continue
            with safe_open(str(hf_path / index[key]), framework="pt", device="cpu") as handle:
                tensors[suffix] = handle.get_tensor(key)
        weight = tensors["weight"]
        if weight.dtype != torch.bfloat16 or list(weight.shape) != [old.out_features, old.in_features]:
            raise ValueError(f"HF weight dtype/geometry mismatch: {name}")
        if ("bias" in tensors) != (old.bias is not None):
            raise ValueError("Projection bias topology differs from the HF origin.")
        device = model.get_input_embeddings().weight.device
        dense = torch.nn.Linear(old.in_features, old.out_features, bias="bias" in tensors,
                                device="meta", dtype=torch.bfloat16)
        for key, value in tensors.items():
            if value.dtype != torch.bfloat16:
                raise ValueError("All restored parameters must be original BF16 tensors.")
            setattr(dense, key, torch.nn.Parameter(value.to(device), requires_grad=False))
            if not torch.equal(getattr(dense, key).detach().cpu(), value):
                raise ValueError("HF parameter copy is not exact.")
        dense.eval()
        parent, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, dense)
        replacements[name] = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                    "sha256": tensor_digest(value), "hf_file": index[name + "." + key]}
                              for key, value in tensors.items()}
        del old, tensors, weight
    after = frozen_inventory(model, targets)
    if after != before:
        raise ValueError("Restoring target projections changed a non-target tensor or identity.")
    return {"selected_targets": list(targets), "selected_count": len(targets), "restored_hf_tensors": replacements,
            "frozen_state_count": len(before["state"]), "frozen_buffer_count": len(before["buffers"]),
            "frozen_inventory": before, "non_target_state_and_identity_exact": True,
            "mixed_precision_diagnostic_only": True, "model_exported": False}
