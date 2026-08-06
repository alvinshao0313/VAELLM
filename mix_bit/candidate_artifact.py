from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn

from litebsq.bitpack import BITPACK_U8_STORAGE_FORMAT, validate_bitpack_u8_spec
from litebsq.vae_linear import VAELinear
from mix_bit.candidate_contract import (
    candidate_mode_from_payload,
    validate_module_spec_mode_contract,
)
from mix_bit.candidate_pool import load_trial_spec
from mix_bit.module_swap import build_candidate_module
from train_utils.model_checkpoint_io import (
    _collect_vae_linear_specs,
    temporarily_pack_parallel_stage_decoders_for_checkpoint,
)

CANDIDATE_FORMAT = "vaellm_candidate_modules_v1"
MODULE_STATE_FILENAME = "module_state.pt"
CANDIDATE_META_FILENAME = "candidate_meta.json"
COMPLETED_FILENAME = "completed.json"

def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(int(x) for x in tensor.shape),
        "nbytes": int(tensor.numel() * tensor.element_size()),
    }


def _iter_named_vae_linears(model: nn.Module) -> list[tuple[str, VAELinear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, VAELinear)]


def _reject_forbidden_keys(state: Mapping[str, Any]) -> None:
    for key in state:
        leaf = key.split(".")[-1]
        if leaf == "original_weight" or "cached" in leaf:
            raise ValueError(f"Candidate artifact rejects forbidden state key: {key}")
        if leaf.startswith("protected_input_") or leaf.startswith("protected_output_"):
            raise ValueError(f"Candidate artifact rejects protected-channel key: {key}")
        if leaf.startswith("protected_residual_") or "protected_residual_" in key:
            # decoder nested names won't include this prefix at leaf for normal decode path
            if "vq_weight" in leaf or leaf.startswith("protected_residual"):
                raise ValueError(f"Candidate artifact rejects protected-residual key: {key}")
        if leaf.startswith("sparse_residual_") or leaf in {"low_rank_a", "low_rank_b"}:
            raise ValueError(f"Candidate artifact rejects residual/LoRA key: {key}")
        top = key.split(".")[0]
        if top in {"embed_tokens", "lm_head", "norm", "dense_backbone"}:
            raise ValueError(f"Candidate artifact rejects backbone key: {key}")


def _validate_vq_buffers(module: VAELinear, local_state: Mapping[str, torch.Tensor]) -> None:
    for key, tensor in local_state.items():
        leaf = key.split(".")[-1]
        if not leaf.startswith("vq_weight"):
            continue
        if tensor.dtype != torch.uint8:
            raise ValueError(f"{key}: VQ payload must be torch.uint8, got {tensor.dtype}")
        # Prefer module storage specs when available.
        specs = []
        residual_stages = int(getattr(module, "residual_stages", 1))
        parallel_parts = int(getattr(module, "parallel_parts", 1))
        for stage_idx in range(residual_stages):
            for part_idx in range(parallel_parts):
                specs.append(module.get_stage_part_vq_spec(stage_idx=stage_idx, part_idx=part_idx))
        matched = False
        for spec in specs:
            normalized = validate_bitpack_u8_spec(spec)
            if (
                normalized["storage_format"] == BITPACK_U8_STORAGE_FORMAT
                and tuple(int(x) for x in normalized["shape"]) == tuple(int(x) for x in tensor.shape)
            ):
                matched = True
                break
        if not matched:
            expected_shapes = [
                tuple(int(x) for x in validate_bitpack_u8_spec(spec)["shape"]) for spec in specs
            ]
            raise ValueError(
                f"{key}: uint8 VQ payload shape {tuple(int(x) for x in tensor.shape)} "
                f"does not match any bitpack_u8 storage spec shapes {expected_shapes}"
            )


def _forward_equivalence_check(source: VAELinear, rebuilt: VAELinear) -> None:
    device = next(source.parameters(), torch.empty(0)).device
    if device.type == "meta":
        device = torch.device("cpu")
    # Prefer buffer device if parameters are empty-ish.
    for _, buf in source.named_buffers():
        if isinstance(buf, torch.Tensor) and buf.device.type != "meta":
            device = buf.device
            break
    dtype = torch.float32
    for param in source.parameters():
        if torch.is_floating_point(param):
            dtype = param.dtype
            break
    x = torch.randn(2, int(source.in_features), device=device, dtype=dtype)
    source_mod = source.to(device=device)
    rebuilt_mod = rebuilt.to(device=device)
    with torch.no_grad():
        y0 = source_mod(x)
        y1 = rebuilt_mod(x)
    torch.testing.assert_close(y1, y0, rtol=1e-4, atol=1e-4)


def save_candidate_artifact_from_model(
    *,
    model: nn.Module,
    trial_spec_path: str,
    output_dir: str,
    source_run_dir: str,
) -> dict[str, str]:
    trial_spec = load_trial_spec(trial_spec_path)
    expected_names = [str(x) for x in trial_spec["expected_module_names"]]
    expected_set = set(expected_names)
    if len(expected_names) != len(expected_set):
        raise ValueError("trial_spec.expected_module_names contains duplicates")

    mode = candidate_mode_from_payload(
        trial_spec["mode"], label="trial_spec.mode"
    )

    named = dict(_iter_named_vae_linears(model))
    missing = sorted(expected_set - set(named))
    if missing:
        raise ValueError(f"Missing converted VAELinear modules: {missing}")
    # Reject unexpected additional converted modules in the trial category suffix.
    target_suffix = str(trial_spec["target_module_suffix"])
    unexpected = sorted(
        name
        for name in named
        if name not in expected_set and (name == target_suffix or name.endswith("." + target_suffix))
    )
    if unexpected:
        raise ValueError(
            f"Unexpected additional converted modules in category suffix {target_suffix!r}: {unexpected}"
        )
    if set(named) & expected_set != expected_set:
        raise ValueError("Expected module-name set does not match converted target modules exactly")

    for name in expected_names:
        module = named[name]
        if not isinstance(module, VAELinear):
            raise TypeError(f"{name}: expected VAELinear, got {type(module)}")
        unloaded = module.unload_original_linear()
        if module.original_weight is not None:
            raise ValueError(
                f"{name}: original_weight must be None after unload_original_linear "
                f"(unload_returned={unloaded}, protect={module.protect_original_weight})"
            )
        module.clear_decoded_weight_cache()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Entering export: drop any stale completed marker first so a failed re-export never
    # leaves a seemingly-complete artifact behind. Old state/meta are kept so partial
    # progress is recoverable; only completed.json is removed before validation.
    completed_path = out_dir / COMPLETED_FILENAME
    if completed_path.is_file():
        completed_path.unlink()

    with temporarily_pack_parallel_stage_decoders_for_checkpoint(model):
        all_specs = _collect_vae_linear_specs(model)
        selected_specs = [spec for spec in all_specs if str(spec["name"]) in expected_set]
        if len(selected_specs) != len(expected_set):
            found = {str(s["name"]) for s in selected_specs}
            raise ValueError(
                f"Converted-module specs mismatch: expected={sorted(expected_set)} found={sorted(found)}"
            )
        for spec in selected_specs:
            if bool(spec.get("has_original_weight", False)):
                raise ValueError(
                    f"{spec['name']}: candidate export rejects has_original_weight=true"
                )
            validate_module_spec_mode_contract(
                spec,
                mode,
                label=f"export/{trial_spec.get('category_name', '?')}/{mode.name}/{spec['name']}",
            )

        artifact_state: dict[str, torch.Tensor] = {}
        for name in expected_names:
            module = named[name]
            local = module.state_dict()
            if any("cached" in key for key in local):
                raise ValueError(f"{name}: decoded caches must not appear in state_dict")
            if "original_weight" in local:
                raise ValueError(f"{name}: original_weight leaked into module state_dict")
            _validate_vq_buffers(module, local)
            for key, value in local.items():
                artifact_state[f"{name}.{key}"] = value.detach().cpu()

        # Full-model state_dict must never be used; also reject foreign prefixes.
        for key in artifact_state:
            if not any(key.startswith(f"{name}.") for name in expected_names):
                raise ValueError(f"Artifact key escapes target prefixes: {key}")
        banned_prefixes = ("embed_tokens", "lm_head", "norm", "dense_backbone")
        for key in artifact_state:
            top = key.split(".")[0]
            if top in banned_prefixes:
                raise ValueError(f"Artifact contains banned backbone key: {key}")
        _reject_forbidden_keys(artifact_state)

        state_path = out_dir / MODULE_STATE_FILENAME
        tmp_state = out_dir / (MODULE_STATE_FILENAME + ".tmp")
        torch.save(artifact_state, tmp_state)
        os.replace(tmp_state, state_path)

        payload_summaries = {key: _tensor_summary(tensor) for key, tensor in artifact_state.items()}
        state_sha = _sha256_file(state_path)
        canonical_mode = {
            "name": mode.name,
            "nominal_bit": mode.nominal_bit,
            "codebook_bits": mode.codebook_bits,
            "codebook_dim": mode.codebook_dim,
            "residual_stages": mode.residual_stages,
        }
        meta = {
            "format": CANDIDATE_FORMAT,
            "module_specs": selected_specs,
            "expected_module_names": expected_names,
            "payload_summaries": payload_summaries,
            "run_config_sha256": trial_spec.get("run_config_sha256"),
            "candidate_space_sha256": trial_spec.get("candidate_space_sha256"),
            "training_recipe_sha256": trial_spec.get("training_recipe_sha256"),
            "model_profile_sha256": trial_spec.get("model_profile_sha256"),
            "model_inventory_fingerprint": trial_spec.get("model_inventory_fingerprint"),
            "mode": canonical_mode,
            "category_name": trial_spec.get("category_name"),
            "source_run_dir": str(source_run_dir),
            "trial_spec_path": str(Path(trial_spec_path).resolve()),
            "module_state_file": MODULE_STATE_FILENAME,
            "module_state_sha256": state_sha,
        }
        meta_path = out_dir / CANDIDATE_META_FILENAME
        _write_json_atomic(meta_path, meta)
        meta_sha = _sha256_file(meta_path)

        # Round-trip verify every module before completed.json.
        for spec in selected_specs:
            name = str(spec["name"])
            source_module = named[name]
            prefix = f"{name}."
            local_prefixed = {k: v for k, v in artifact_state.items() if k.startswith(prefix)}
            rebuilt = build_candidate_module(
                SimpleNamespace(
                    module_name=name,
                    module_spec=spec,
                    in_features=int(spec["in_features"]),
                    out_features=int(spec["out_features"]),
                    has_bias=bool(spec.get("has_bias", False)),
                ),
                local_prefixed,
                device="cpu",
            )
            if not isinstance(rebuilt, VAELinear):
                raise TypeError(f"Expected VAELinear after rebuild, got {type(rebuilt)}")
            rebuilt_state = rebuilt.state_dict()
            source_local = {k[len(prefix) :]: v for k, v in local_prefixed.items()}
            if set(rebuilt_state) != set(source_local):
                raise ValueError(
                    f"{name}: rebuilt state keys mismatch "
                    f"missing={sorted(set(source_local) - set(rebuilt_state))} "
                    f"extra={sorted(set(rebuilt_state) - set(source_local))}"
                )
            for key, value in source_local.items():
                torch.testing.assert_close(rebuilt_state[key].cpu(), value.cpu())
            _forward_equivalence_check(source_module, rebuilt)

        completed = {
            "format": CANDIDATE_FORMAT,
            "module_state_sha256": state_sha,
            "candidate_meta_sha256": meta_sha,
            "module_count": len(expected_names),
        }
        _write_json_atomic(out_dir / COMPLETED_FILENAME, completed)

    return {
        "output_dir": str(out_dir),
        "module_state": str(state_path),
        "candidate_meta": str(meta_path),
        "completed": str(out_dir / COMPLETED_FILENAME),
    }
