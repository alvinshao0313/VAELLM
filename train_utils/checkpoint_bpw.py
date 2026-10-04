"""Account for stable v6 Linear payloads without constructing the language model."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Mapping

import torch

from litebsq.bitpack import validate_bitpack_u8_spec


_SHARED = "_vaellm_shared_protected_residual_decoders."
_DECODER = re.compile(r"^(?:decoder(?:s)?(?:_s\d+)?|_parallel_stage_decoder|protected_residual_decoder(?:_s\d+)?|_protected_residual_parallel_decoder)\.")
_PROJECTION = re.compile(r"^(.*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))\.")


def _summary(components, denominator):
    logical = sum(item["budget_bits"] for item in components.values())
    physical = sum(item["storage_bits"] for item in components.values())
    return {
        "original_linear_weights": denominator,
        "user_linear_bits": logical,
        "user_linear_bpw": logical / denominator,
        "packed_linear_bits": physical,
        "packed_linear_bpw": physical / denominator,
        "code_padding_bits": physical - logical,
        "components": dict(sorted(components.items())),
    }


def _component(relative):
    if _DECODER.match(relative):
        return "decoder"
    if relative in {"low_rank_a", "low_rank_b"}:
        return "lora"
    if relative.startswith("weight_rotation."):
        return "rotation"
    if relative.startswith(("protected_input_", "protected_output_")):
        if relative.endswith("indices"):
            return "protected_indices"
        if relative.endswith("scales"):
            return "protected_scales"
        return "protected_weights"
    if relative.startswith("sparse_residual_"):
        return "sparse_residual"
    if relative == "protected_residual_indices":
        return "protected_indices"
    if relative == "bias":
        return "linear_bias"
    if relative in {"weight", "original_weight"}:
        return "dense_weights"
    return "other_target"


def _code_specs(spec):
    parts = int(spec["parallel_parts"])
    stages = int(spec["residual_stages"])
    dims = spec.get("stage_codebook_dims") or [int(spec["codebook_dim"])] * stages
    if len(dims) != stages or parts < 1 or stages < 1:
        raise ValueError(f"Invalid stage/part layout for {spec['name']}.")
    stage_specs = spec["stage_vq_weights"] if stages > 1 else [spec["vq_weights"]]
    compressed = int(spec.get("compressed_in_features", spec["in_features"])) * int(
        spec.get("compressed_out_features", spec["out_features"]))
    for stage in range(stages):
        raw = stage_specs[stage]
        entries = [raw] if isinstance(raw, dict) else raw
        if len(entries) != parts:
            raise ValueError(f"Wrong VQ part count: {spec['name']}, stage {stage}.")
        covered = 0
        for part, entry in enumerate(entries):
            normalized = validate_bitpack_u8_spec(dict(entry))
            covered += math.prod(normalized["logical_shape"][:-1]) * int(dims[stage])
            suffix = "" if stage == 0 else f"_s{stage}"
            suffix += f"_{part}" if parts > 1 else ""
            yield f"{spec['name']}.vq_weight{suffix}", normalized
        if covered != compressed:
            raise ValueError(f"VQ blocks do not cover the compressed matrix: {spec['name']}, stage {stage}.")
    residual = spec.get("protected_residual_stage_vq_weights") or []
    for stage, entry in enumerate(residual):
        suffix = "" if stage == 0 else f"_s{stage}"
        yield f"{spec['name']}.protected_residual_vq_weight{suffix}", validate_bitpack_u8_spec(dict(entry))


def audit_state_dict(meta: Mapping, state: Mapping[str, torch.Tensor], *, max_bpw=2.5, max_lora_rank=16):
    """Count unique serialized storages; exact aliases are counted once.

    Multiple views of one storage are counted at the full storage byte size,
    matching torch.save. Code views must be disjoint or exact aliases; aliases
    crossing a budget component/scope are rejected instead of guessed.
    """
    if meta.get("format") != "vaellm_model_checkpoint_v6" or meta.get("schema_version") != 6:
        raise ValueError("Expected vaellm_model_checkpoint_v6, schema_version=6.")
    if meta.get("checkpoint_kind") not in {"category_boundary", "final_model"}:
        raise ValueError("Audit requires a stable category_boundary or final_model checkpoint.")
    if meta.get("lora_config") is not None:
        raise ValueError("Stable checkpoints must have lora_config=null.")
    if not math.isfinite(max_bpw) or max_bpw <= 0 or max_lora_rank < 0:
        raise ValueError("Budget must be positive and finite; maximum rank must be non-negative (0 forbids LoRA).")
    specs = {item["name"]: item for item in meta.get("converted_modules", [])}
    if len(specs) != len(meta.get("converted_modules", [])):
        raise ValueError("Duplicate converted module names.")
    targets = set()
    for field in ("compressed_targets", "pending_dense_targets", "skip_targets"):
        values = set(meta.get(field) or [])
        if targets & values:
            raise ValueError(f"Overlapping target inventories at {field}.")
        targets |= values
    if not targets or set(specs) != set(meta.get("compressed_targets") or []):
        raise ValueError("Converted modules must exactly match the non-empty compressed target inventory.")
    discovered = {match.group(1) for key in state if (match := _PROJECTION.match(key))}
    if discovered - targets:
        raise ValueError(f"Projection state omitted from checkpoint target inventory: {sorted(discovered - targets)}.")
    rows = {}
    code_specs = {}
    violations = []
    for name in sorted(targets):
        if name in specs:
            spec = specs[name]
            rows[name] = int(spec["in_features"]) * int(spec["out_features"])
            for field, tensor_spec in spec.items():
                if isinstance(tensor_spec, dict) and "shape" in tensor_spec and "dtype" in tensor_spec:
                    key = f"{name}.{field}"
                    tensor = state.get(key)
                    if tensor is None or list(tensor.shape) != tensor_spec["shape"] or str(tensor.dtype).removeprefix("torch.") != tensor_spec["dtype"]:
                        raise ValueError(f"Missing or inconsistent metadata tensor: {key}.")
            decoder_keys = [key for key in state if key.startswith(name + ".") and _DECODER.match(key[len(name) + 1:])]
            if not decoder_keys:
                raise ValueError(f"Missing base decoder tensors: {name}.")
            if spec.get("has_original_weight") and f"{name}.original_weight" not in state:
                raise ValueError(f"Missing original_weight: {name}.")
            if spec.get("has_bias") and f"{name}.bias" not in state:
                raise ValueError(f"Missing Linear bias: {name}.")
            if spec.get("weight_rotation") is not None:
                for axis, field in (("input", "compressed_in_features"), ("output", "compressed_out_features")):
                    key = f"{name}.weight_rotation.{axis}_signs"
                    tensor = state.get(key)
                    if tensor is None or tensor.dtype != torch.int8 or tensor.numel() != int(spec[field]):
                        raise ValueError(f"Missing or inconsistent rotation signs: {key}.")
            for key, code in _code_specs(spec):
                tensor = state.get(key)
                if tensor is None or tensor.dtype != torch.uint8 or list(tensor.shape) != code["shape"]:
                    raise ValueError(f"Missing or inconsistent packed VQ tensor: {key}.")
                code_specs[key] = code
            dims = spec.get("stage_codebook_dims") or [spec["codebook_dim"]]
            for axis in ("input", "output"):
                key = f"{name}.protected_{axis}_indices"
                count = state[key].numel() if key in state else 0
                if any(count % int(dim) for dim in dims):
                    violations.append(f"{key}: {count} channels are not multiples of every codebook_dim.")
            a, b = state.get(f"{name}.low_rank_a"), state.get(f"{name}.low_rank_b")
            if (a is None) != (b is None):
                raise ValueError(f"Incomplete low-rank payload: {name}.")
            if a is not None:
                if a.ndim != 2 or b.ndim != 2 or a.shape != (int(spec["out_features"]), b.shape[0]) or b.shape[1] != int(spec["in_features"]):
                    raise ValueError(f"Invalid low-rank shapes: {name}.")
                if int(a.shape[1]) > max_lora_rank:
                    violations.append(f"{name}: LoRA rank {a.shape[1]} exceeds {max_lora_rank}.")
        else:
            tensor = state.get(f"{name}.weight")
            if tensor is None or tensor.ndim != 2:
                raise ValueError(f"Missing dense Linear weight: {name}.")
            rows[name] = tensor.numel()
    denominator = sum(rows.values())
    if denominator <= 0:
        raise ValueError("No positive Linear weight denominator.")
    shared_owners = defaultdict(set)
    for name, spec in specs.items():
        for ref in spec.get("protected_residual_shared_decoder_refs") or []:
            shared_owners[ref].add(name)
    unknown = []
    groups = {}
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.device.type != "cpu":
            raise ValueError(f"Expected a CPU strided tensor: {key}.")
        owner = next((name for name in targets if key.startswith(name + ".")), None)
        owners = {owner} if owner else set()
        component = _component(key[len(owner) + 1:]) if owner else "other_model"
        if key in code_specs:
            component = "code"
        elif key.startswith(_SHARED):
            ref = key[len(_SHARED):].split(".", 1)[0]
            owners = shared_owners.get(ref, set())
            if not owners:
                raise ValueError(f"Unreferenced shared decoder storage: {key}.")
            component = "decoder"
        elif "._residual_lora_" in key:
            component = "residual_lora"
            owners = set(targets)
            if key.endswith(".lora_A.weight") and tensor.shape[0] > max_lora_rank:
                violations.append(f"{key}: residual LoRA rank exceeds {max_lora_rank}.")
        elif not owner:
            if ".embed_tokens." in key or key.startswith("lm_head."):
                component = "embedding_or_head"
            elif "norm" in key.lower():
                component = "model_norm"
        if component == "other_target":
            unknown.append(key)
        storage = tensor.untyped_storage()
        identity = (storage.data_ptr(), storage.nbytes()) if storage.nbytes() else ("empty", key)
        scope = bool(owners)
        item = groups.setdefault(identity, {"component": component, "scope": scope, "owners": set(), "keys": [], "storage_bits": storage.nbytes() * 8, "views": {}, "dtypes": set()})
        if item["component"] != component or item["scope"] != scope:
            raise ValueError(f"Storage alias crosses accounting components/scopes: {item['keys']} and {key}.")
        item["owners"].update(owners)
        item["keys"].append(key)
        item["dtypes"].add(str(tensor.dtype).removeprefix("torch."))
        if component == "code":
            if not tensor.is_contiguous():
                raise ValueError(f"Packed code storage must be contiguous: {key}.")
            span = (tensor.storage_offset(), tensor.storage_offset() + tensor.numel())
            logical = math.prod(code_specs[key]["logical_shape"])
            for old_span, old_logical in item["views"].items():
                if span == old_span and logical != old_logical:
                    raise ValueError(f"Code alias has conflicting logical shape: {key}.")
                if span != old_span and max(span[0], old_span[0]) < min(span[1], old_span[1]):
                    raise ValueError(f"Overlapping non-identical code views: {key}.")
            item["views"][span] = logical
    components = defaultdict(lambda: {"budget_bits": 0, "storage_bits": 0, "unique_storages": 0, "dtypes": set()})
    outside = defaultdict(int)
    per_target = {name: {"original_linear_weights": count, "budget_bits": 0.0, "storage_bits": 0.0} for name, count in rows.items()}
    aliases = []
    for item in groups.values():
        bits = item["storage_bits"]
        if len(item["keys"]) > 1:
            aliases.append(item["keys"])
        if not item["scope"]:
            outside[item["component"]] += bits
            continue
        logical = sum(item["views"].values()) if item["component"] == "code" else bits
        total = components[item["component"]]
        total["budget_bits"] += logical
        total["storage_bits"] += bits
        total["unique_storages"] += 1
        total["dtypes"].update(item["dtypes"])
        owner_weight = sum(rows[name] for name in item["owners"])
        for name in item["owners"]:
            share = rows[name] / owner_weight
            per_target[name]["budget_bits"] += logical * share
            per_target[name]["storage_bits"] += bits * share
    for value in components.values():
        value["dtypes"] = sorted(value["dtypes"])
    result = _summary(components, denominator)
    if result["user_linear_bpw"] > max_bpw:
        violations.append(f"User Linear bpw {result['user_linear_bpw']:.9f} exceeds {max_bpw}.")
    result.update({
        "measurement": "checkpoint_tensor_storage", "target_count": len(targets),
        "compressed_target_count": len(specs), "checkpoint_id": meta.get("checkpoint_id"),
        "full_model_tensor_bytes": sum(item["storage_bits"] for item in groups.values()) // 8,
        "outside_linear_bits": dict(sorted(outside.items())),
        "per_target": per_target, "storage_aliases": aliases,
        "unclassified_target_keys": unknown, "contract_violations": violations,
        "budget_passed": not violations,
        "scope": "Linear inventory plus shared decoders and residual LoRA; embeddings/head/model norms are separate. Decoder norms are included. Shared storage is apportioned by original Linear weight count.",
    })
    return result


def audit_checkpoint(path, *, max_bpw=2.5, max_lora_rank=16):
    directory = Path(path).resolve()
    with (directory / "checkpoint_meta.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("state_dict_file", "pytorch_model.bin") != "pytorch_model.bin":
        raise ValueError("Stable v6 checkpoints must use pytorch_model.bin.")
    state_path = directory / "pytorch_model.bin"
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    result = audit_state_dict(meta, state, max_bpw=max_bpw, max_lora_rank=max_lora_rank)
    result["checkpoint_dir"] = str(directory)
    result["state_dict_file_bytes"] = state_path.stat().st_size
    return result


def estimate_qwen3_cat_bpw(model_config: Mapping, category_configs: Mapping[str, Mapping], *, lora_rank=0, parameter_dtype=torch.bfloat16):
    """Preflight estimate from model config and resolved per-category CAT settings.

    Uses the actual Decoder architecture on the meta device. This is explicitly
    an estimate, not a checkpoint measurement; all seven projections are required.
    """
    from litebsq.autoencoder import Decoder
    from litebsq.vae_args import resolve_autoencoder_arch_spec

    if model_config.get("model_type") != "qwen3":
        raise ValueError("This preflight estimator currently accepts Qwen3 configs only.")
    hidden = int(model_config["hidden_size"])
    head_dim = int(model_config.get("head_dim", hidden // int(model_config["num_attention_heads"])))
    query = head_dim * int(model_config["num_attention_heads"])
    kv = head_dim * int(model_config["num_key_value_heads"])
    intermediate = int(model_config["intermediate_size"])
    layers = int(model_config["num_hidden_layers"])
    shapes = {"q_proj": (query, hidden), "k_proj": (kv, hidden), "v_proj": (kv, hidden), "o_proj": (hidden, query), "gate_proj": (intermediate, hidden), "up_proj": (intermediate, hidden), "down_proj": (hidden, intermediate)}
    if set(category_configs) != set(shapes) or not 0 <= lora_rank <= 16:
        raise ValueError("Supply all seven categories and a LoRA rank in [0, 16].")
    components = defaultdict(lambda: {"budget_bits": 0, "storage_bits": 0})
    resolved = {}
    denominator = layers * sum(out * inp for out, inp in shapes.values())
    dtype_bits = torch.empty((), dtype=parameter_dtype).element_size() * 8
    for category, (out, inp) in shapes.items():
        cfg = dict(category_configs[category])
        arch = resolve_autoencoder_arch_spec(SimpleNamespace(**cfg))
        stages = int(cfg.get("residual_stages", 1))
        parts = math.prod(cfg.get("intra_parallel", (1, 1)))
        count = int(cfg.get("channel_protect_count", 0)) if cfg.get("channel_protect_mode", "none") == "channel" else 0
        axis = cfg.get("channel_axis", "input")
        if count < 0 or count % arch.codebook_dim or axis not in {"input", "output"}:
            raise ValueError(f"Invalid codebook-aligned channel protection: {category}.")
        c_out, c_in = (out, inp - count) if axis == "input" else (out - count, inp)
        if min(c_out, c_in, stages, parts) < 1 or (c_out * c_in) % (parts * arch.codebook_dim):
            raise ValueError(f"Invalid compressed layout: {category}.")
        with torch.device("meta"):
            decoder = Decoder(in_dim=arch.codebook_bits, out_dim=arch.codebook_dim, hidden_dim=arch.decoder_hidden_dim, num_res_blocks=arch.decoder_num_res_blocks, norm_type=arch.norm_type, activation_type=arch.activation_type, decoder_type=arch.decoder_type).to(dtype=parameter_dtype)
        decoder_bits = sum(t.numel() * t.element_size() * 8 for t in decoder.state_dict().values()) * stages * parts
        blocks = c_out * c_in // arch.codebook_dim
        values = {"code": (blocks * arch.codebook_bits * stages, blocks * math.ceil(arch.codebook_bits / 8) * 8 * stages), "decoder": (decoder_bits, decoder_bits), "lora": (lora_rank * (out + inp) * dtype_bits,) * 2}
        if count:
            if cfg.get("channel_quant") not in {"int8", "fp8_e4m3", "fp8_e5m2"}:
                raise ValueError("Preflight protection requires an explicit supported 8-bit format.")
            values["protected_weights"] = (count * (out if axis == "input" else inp) * 8,) * 2
            values["protected_scales"] = (count * 16,) * 2
            values["protected_indices"] = (count * 64,) * 2
        if cfg.get("weight_rotation", "none") == "two_sided":
            values["rotation"] = ((c_out + c_in) * 8,) * 2
        if model_config.get("attention_bias", False) and category in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            values["linear_bias"] = (out * dtype_bits,) * 2
        for component, (logical, physical) in values.items():
            components[component]["budget_bits"] += logical * layers
            components[component]["storage_bits"] += physical * layers
        resolved[category] = {"decoder_hidden_dim": arch.decoder_hidden_dim, "decoder_num_res_blocks": arch.decoder_num_res_blocks, "decoder_bits_per_linear": decoder_bits, "protected_channels": count}
    result = _summary(components, denominator)
    result.update({"measurement": "architecture_estimate_not_checkpoint", "target_count": len(shapes) * layers, "resolved_categories": resolved})
    return result
