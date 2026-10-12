"""Native full-gate reload/cache checks and real H188/H220 down equivalence.

The two complete gate short runs provide model/save coverage. The down cases
are deliberately single-projection CAT calls, never exported as full models.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import logging
import os
from pathlib import Path
import sys
import time

import torch
from transformers import AutoTokenizer

from experiments.full_rht_gate_pair.common import ORIGINAL, ROOT, check, read
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _assert_state_equal, _require
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.rotation_factor_probe.factors import load_factors, validate_factor
from experiments.rotation_factor_probe.run import load_archive
from experiments.rotation_factor_probe.train import host_for_weight, run_matrix
from experiments.rotation_factor_probe.verify_native import compare_tensor, native_recipe
from litebsq.protected_channel_quant import decode_protected_channel_weight
from litebsq.vae_linear import VAELinear, clear_model_vae_linear_cache
from rotation.weight_preprocess import build_weight_rotation, rotation_spec
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_pipeline import apply_group_vae_payload, train_group_vae_payload
from train_utils.utils import LinearRef, configure_deterministic_mode, set_seed


SEED, STEPS, BATCH = 31, 2, 8192
DOWN = "model.layers.0.mlp.down_proj"


def source_pins():
    paths = {p for directory in ("train_utils", "litebsq", "rotation", "e2e_common",
                                "compressed_e2e_fintuning",
                                "experiments/full_rht_gate_pair")
             for p in (ROOT / directory).rglob("*.py")}
    paths.update(ROOT / name for name in (
        "compressed_e2e_fintuning/runtime_v6_pipeline.py", "tools/cat_eval.py",
        "experiments/longbench_v1/common.py", "experiments/qwen3_w25/assemble_categories.py",
        "experiments/rotation_factor_probe/capture.py", "experiments/rotation_factor_probe/run.py",
        "experiments/rotation_factor_probe/train.py", "experiments/rotation_factor_probe/transform.py",
        "experiments/rotation_factor_probe/factors.py", "experiments/rotation_factor_probe/verify_native.py"))
    return {str(p): sha256(p) for p in sorted(paths)}


def loaded_repository_files():
    """Only existing source files count; torch dynamic namespaces have pseudo files."""
    result = {}
    for name, module in tuple(sys.modules.items()):
        value = getattr(module, "__file__", None)
        if not isinstance(value, (str, Path)):
            continue
        path = Path(value).resolve()
        if path.is_file() and path.is_relative_to(ORIGINAL):
            _require(path.is_relative_to(ROOT), "Imported active public source: " + str(path))
            result[name] = str(path)
    return result


def rope_record(model):
    values = {n: v for n, v in model.named_buffers() if "inv_freq" in n}
    _require(values and all(v.dtype == torch.float32 for v in values.values()), "Expected native FP32 RoPE.")
    return {n: tensor_record(v) for n, v in values.items()}


def weight_pair(before, after):
    """Existing BF16 two-ULP parity scale, with a relative-energy guard."""
    _require(before.shape == after.shape and before.dtype == after.dtype == torch.bfloat16,
             "Original-coordinate BF16 geometry changed.")
    epsilon = 2 * torch.finfo(torch.bfloat16).eps
    a, b = before.reshape(-1), after.reshape(-1)
    error_sq = reference_sq = maximum = 0.0
    for start in range(0, a.numel(), 1 << 20):
        x, y = a[start:start + (1 << 20)].float(), b[start:start + (1 << 20)].float()
        _require(bool(torch.isfinite(x).all() and torch.isfinite(y).all()), "Nonfinite decoded weight.")
        torch.testing.assert_close(x, y, atol=epsilon, rtol=epsilon)
        delta = y - x
        maximum = max(maximum, float(delta.abs().max()))
        error_sq += float(delta.double().square().sum())
        reference_sq += float(x.double().square().sum())
    relative = (error_sq / max(reference_sq, 1e-24)) ** .5
    _require(relative <= epsilon, "Decoded weight discrepancy exceeds the existing BF16 parity scale.")
    return {"exact": bool(torch.equal(before, after)), "max_abs": maximum, "relative_l2": relative,
            "atol": epsilon, "rtol": epsilon, "relative_l2_limit": epsilon,
            "tolerance_source": "tests/test_fused_multistage_decoder_parity.py two BF16 epsilon scale"}


def gate_case(path, model_type, device):
    from compressed_e2e_fintuning.runtime_v6_pipeline import (
        _assert_finalization_probe_close, _build_finalization_probe_inputs, _run_finalization_probe,
    )
    from tools.cat_eval import _prepare_model_for_eval
    from train_utils.checkpoint_bpw import audit_state_dict
    from train_utils.v6_model_loader import load_v6_model_checkpoint

    record = read(path); directory = path.parent; checkpoint = directory / "final_model"
    _require(record["status"] == "completed" and record["exit_code"] == 0
             and record["model_type"] == model_type and record["steps_per_stage"] == STEPS
             and record["block_size"] == 0 and not record["full_evaluation"],
             "Require the actual completed full-category, full-RHT two-step short run.")
    check(record)
    for name in ("train", "assembly", "dispatcher", "outer"):
        _require((directory / (name + ".exit")).read_text().strip() == "0", "Short stage did not exit zero.")
    audit = read(directory / "audit.json"); assembly = read(checkpoint / "assembly_report.json")
    _require(audit["status"] == assembly["status"] == "verified"
             and audit["checkpoint_id"] == assembly["checkpoint_id"]
             and assembly["serialized_tensors_compared"] == assembly["reloaded_tensors_compared"]
             == audit["source_union_save_fresh_tensors"]
             and sha256(checkpoint / "assembly_report.json") == audit["assembly_report_sha256"],
             "Existing native source-union/save/fresh audit is incomplete.")
    model, meta, loaded = load_v6_model_checkpoint(str(checkpoint), map_location="cpu", strict=True,
                                                expected_kind="final_model")
    _require(not loaded.missing_keys and not loaded.unexpected_keys
             and meta["checkpoint_id"] == audit["checkpoint_id"], "Independent strict native load failed.")
    arch = record["architecture"]
    expected_backend = "flash_attention_2" if model_type == "qwen3" else "sdpa"
    _require(model.config.model_type == model_type and model.config._attn_implementation == expected_backend
             and all(p.dtype == torch.bfloat16 for p in model.parameters() if p.is_floating_point()),
             "Model family, attention backend, or actual BF16 parameters changed.")
    rope = rope_record(model)
    gates = {n: m for n, m in model.named_modules() if isinstance(m, VAELinear) and n.endswith(".gate_proj")}
    _require(len(gates) == arch["num_hidden_layers"] == audit["gate_count"], "Missing complete gate category.")
    for name, module in gates.items():
        _require(module.weight_rotation.block_size == 0 and module.weight_rotation.input_factor == 252
                 and not module.transpose and module.residual_stages == 2
                 and module.compressed_in_features == 4032 and not module.trainable_decode,
                 "Saved full-RHT gate geometry differs: " + name)
    budget = audit_state_dict(meta, model.state_dict(), max_bpw=2.5, max_lora_rank=0)
    for key in ("original_linear_weights", "user_linear_bits", "packed_linear_bits", "components",
                "outside_linear_bits", "full_model_tensor_bytes"):
        _require(budget[key] == audit["bit_budget"][key], "Actual runtime storage changed: " + key)
    _require(budget["budget_passed"] and not budget["contract_violations"], "Actual storage exceeds the contract.")
    tokenizer = AutoTokenizer.from_pretrained(arch["base_model_path"], use_fast=True, local_files_only=True)
    _require(tokenizer.is_fast and type(tokenizer).__name__ == arch["tokenizer_class"], "Tokenizer changed.")
    model.eval().requires_grad_(False); model.config.use_cache = False; model.to(device)
    probe = _build_finalization_probe_inputs(tokenizer)
    clear_model_vae_linear_cache(model); VAELinear.reset_fuse_stats()
    cold, dtype = _run_finalization_probe(model, probe)
    cold_stats = VAELinear.get_fuse_stats()
    cold_weights = {}
    for name, module in gates.items():
        value = module._cached_weight
        _require(value is not None and value.dtype == torch.bfloat16, "Cold model forward missed a gate cache.")
        cold_weights[name] = value.detach().cpu()
    del value
    clear_model_vae_linear_cache(model); VAELinear.reset_fuse_stats()
    warmup = _prepare_model_for_eval(model, str(device), 8, logging.getLogger(__name__), {})
    _require(warmup["warmed"] == arch["projection_count"] and warmup["failed"] == 0
             and warmup["prewarm_dtype"] == "torch.bfloat16" and warmup["prewarm_group_size"] == 8,
             "Original group8 prewarm did not cover the complete assembled model.")
    warm_stats = VAELinear.get_fuse_stats()
    weights = {name: weight_pair(cold_weights[name], module._cached_weight.detach().cpu())
               for name, module in gates.items()}
    pointers = {name: module._cached_weight.data_ptr() for name, module in model.named_modules()
                if isinstance(module, VAELinear)}
    warmed, warm_dtype = _run_finalization_probe(model, probe)
    _require(warm_dtype == dtype == torch.bfloat16 and VAELinear.get_fuse_stats() == warm_stats
             and pointers == {name: module._cached_weight.data_ptr() for name, module in model.named_modules()
                              if isinstance(module, VAELinear)}, "Warm model forward regenerated decoded weights.")
    parity = _assert_finalization_probe_close(cold, warmed, output_dtype=dtype,
                                             label=model_type + " cold versus native group8")
    _require(rope_record(model) == rope, "Native model forward altered FP32 RoPE.")
    clear_model_vae_linear_cache(model); model.cpu()
    saved = torch.load(checkpoint / meta["state_dict_file"], map_location="cpu", weights_only=True, mmap=True)
    count = _assert_state_equal(model.state_dict(), saved, "Persistent state after cold/group8/forward")
    _require(count == audit["source_union_save_fresh_tensors"], "Incomplete persistent-state preservation.")
    result = {"status": "verified", "checkpoint_id": meta["checkpoint_id"], "checkpoint": str(checkpoint),
        "model_type": model_type, "attention_backend": expected_backend, "rope": rope,
        "gate_count": len(gates), "gate_input_factor": 252, "gate_cold_group8": weights,
        "warmup": warmup, "cold_decode_stats": cold_stats, "warm_decode_stats": warm_stats,
        "warm_cache_reused": True, "probe_input_tokens": probe["input_ids"].tolist(), "probe_parity": parity,
        "persistent_state_after_runtime_exact": count, "bit_budget": budget,
        "assembly_save_fresh_audit_reused": audit["assembly_report_sha256"],
        "checkpoint_weight_stat": file_stat(checkpoint / meta["state_dict_file"]),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "scope": "Complete fresh gate short training plus existing native assembly/save/fresh proof; added independent load, cold/group8 and four-token model forwards."}
    del model, gates, module, saved, cold_weights
    gc.collect(); torch.cuda.empty_cache()
    return result


def down_case(directory, model_type, factors, device):
    archive, manifest = load_archive(directory)
    _require(archive["model_type"] == model_type, "Wrong real-model capture.")
    bundle = archive["targets"][DOWN]; weight = bundle["reference_weight"]; indices = bundle["protected_input_indices"]
    expected_width, expected_factor = (12288, 188) if model_type == "qwen3" else (14336, 220)
    _require(weight.dtype == torch.bfloat16 and tuple(weight.shape) == (4096, expected_width)
             and indices.dtype == torch.int64 and indices.numel() == 256, "Wrong actual L0 down geometry.")
    argv, _, _, _ = native_recipe(archive["model_path"], "down_proj", 256)
    argv[argv.index("--weight_rotation_block_size") + 1] = "0"
    cat, _, training, vae = parse_cat_runtime_args(argv); training.seed = SEED
    runtime = resolve_category_runtime_configs(cat, vae, ["down_proj"])["down_proj"]
    _require(runtime.steps == STEPS and runtime.codebook_bits == 48 and runtime.codebook_dim == 32
             and runtime.residual_stages == 2 and runtime.channel_protect_count == 256
             and vae.weight_rotation_block_size == 0 and training.bf16, "Native down recipe differs.")
    configure_deterministic_mode(True); set_seed(SEED)
    host, linear = host_for_weight(weight, DOWN)
    refs = [LinearRef(DOWN, linear, "down_proj", True)]
    torch.manual_seed(SEED)
    payload = train_group_vae_payload(model=host, group_refs=refs, group_tag=model_type + "/down/full/native",
        runtime_cfg=runtime, vae_args=vae, training_args=training, train_device=str(device),
        convert_device=str(device), do_convert=True, batch_size=BATCH, gpu_resident_data=True,
        log_every=1, eval_every=0, eval_blocks=256, channel_protect_mode="channel",
        channel_plan={DOWN: indices}, channel_rank_metric="channel_weight_abs", channel_axis="input",
        channel_quant="int8", deterministic=True, shuffle_seed=SEED)
    apply_group_vae_payload(model=host, group_refs=refs, group_tag=DOWN, payload=payload, convert_device=str(device))
    layer = host.get_submodule(DOWN).to(device).eval()
    _require(layer.transpose and layer.parallel_stage_decode and layer.residual_stages == 2
             and layer.weight_rotation.block_size == 0 and layer.weight_rotation.input_factor == expected_factor,
             "Native down did not execute the newly supported full factor.")
    with torch.no_grad():
        codes = [layer.get_stage_part_vq_storage(s, 0).detach().cpu().clone() for s in range(2)]
        decoders = [{k: v.detach().cpu().clone() for k, v in layer.get_stage_part_decoder(s, 0).state_dict().items()}
                    for s in range(2)]
        native = layer._decode_weight(dtype=torch.bfloat16).cpu()
        protection = {k: getattr(layer, k).detach().cpu().clone() for k in
                      ("protected_input_indices", "protected_input_qvalues", "protected_input_scales")}
        keep = torch.ones(weight.shape[1], dtype=torch.bool); keep[indices] = False
        transform = build_weight_rotation(weight.shape[0], int(keep.sum()), rotation_spec("two_sided", 0, SEED, DOWN)).to(device)
        rotated_reference = transform(weight[:, keep].to(device=device, dtype=torch.float32)).cpu()
    del host, linear, layer, payload, transform, refs
    gc.collect(); torch.cuda.empty_cache()
    diagnostic = {}
    external, external_report = run_matrix(bundle, name=DOWN, model_path=archive["model_path"], full=True,
        factors=factors, steps=STEPS, seed=SEED, device=device, diagnostic_out=diagnostic)
    checks = {}
    for stage in range(2):
        checks[f"stage{stage}_codes"] = compare_tensor(codes[stage], diagnostic["stage_codes"][stage])
        other = diagnostic["stage_decoders"][stage]
        _require(set(decoders[stage]) == set(other), "Stage decoder keys changed.")
        for key, value in decoders[stage].items():
            _require(not value.is_floating_point() or value.dtype == torch.bfloat16, "Finalized decoder is not BF16.")
            checks[f"stage{stage}_decoder/{key}"] = compare_tensor(value, other[key])
    for key, value in protection.items():
        checks[key] = compare_tensor(value, bundle[key])
    expected_protected = decode_protected_channel_weight(bundle["protected_input_qvalues"], bundle["protected_input_scales"],
        quant_format="int8", dtype=torch.bfloat16, device=torch.device("cpu")).T.contiguous()
    checks["native_protected_columns"] = compare_tensor(native[:, indices], expected_protected)
    checks["external_protected_columns"] = compare_tensor(external[:, indices], expected_protected)
    _require(all(x["exact"] for x in checks.values()), "Packed codes, finalized decoder or protected bytes differ.")
    torch.testing.assert_close(rotated_reference, diagnostic["rotated_weight"], rtol=2e-6, atol=2e-6)
    result = {"status": "verified", "model_type": model_type, "target": DOWN,
        "input_factor": expected_factor, "weight_shape": list(weight.shape), "protected_count": 256,
        "transpose": True, "steps_per_stage": STEPS, "batch_size": BATCH, "seed": SEED,
        "native_recipe": argv, "native_runtime": asdict(runtime), "exact_persistent_and_protection": checks,
        "original_coordinate_bf16": weight_pair(native, external),
        "fp32_rotation_reference": {"comparison": compare_tensor(rotated_reference, diagnostic["rotated_weight"]),
            "scope": "Recomputed native transform versus external preprocessing, not an observed CAT training intermediate."},
        "external_report": external_report, "capture_sha256": manifest["inputs_sha256"],
        "checkpoint_indices_source": archive["checkpoint_id"],
        "scope": "Real original-HF single down matrix and fixed captured256 indices; independent two-stage CAT updates. No complete-model export or quality claim."}
    del native, external, archive, diagnostic, codes, decoders, rotated_reference
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-short", type=Path, required=True)
    parser.add_argument("--llama-short", type=Path, required=True)
    parser.add_argument("--qwen-inputs", type=Path, required=True)
    parser.add_argument("--llama-inputs", type=Path, required=True)
    parser.add_argument("--factor-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and os.environ.get("PYTHONPATH") == str(ROOT),
             "Use the independent physical snapshot and its PYTHONPATH.")
    _require(not output.exists() and not output.is_relative_to(ROOT), "Use a fresh validation output outside source.")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expose one root-assigned CUDA device.")
    configure_deterministic_mode(True); set_seed(SEED); torch.set_num_threads(4)
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    factors = load_factors(args.factor_source)
    pins = source_pins(); inputs = {str(args.factor_source.resolve()): sha256(args.factor_source)}
    for path in (args.qwen_short.resolve(), args.llama_short.resolve()):
        for p in (path, path.parent / "audit.json", path.parent / "final_model/assembly_report.json",
                  path.parent / "final_model/checkpoint_meta.json", path.parent / "train.log"):
            inputs[str(p)] = sha256(p)
    for directory in (args.qwen_inputs.resolve(), args.llama_inputs.resolve()):
        for name in ("data_manifest.json", "inputs.pt"):
            inputs[str(directory / name)] = sha256(directory / name)
    record = {"status": "running", "source_root": str(ROOT), "source_sha256": pins,
        "input_sha256": inputs, "pid": os.getpid(), "argv": sys.argv, "started_unix": time.time(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gate_models": {}, "down_projections": {},
        "factor_records": {str(k): validate_factor(v) for k, v in factors.items()},
        "snapshot_cpu_import_inventory_note": "The immutable earlier CPU review listed 83 module paths; 81 existed as files. torch.ops/_ops.py and torch.classes/_classes.py were dynamic pseudo __file__ values, not repository imports. Explicit native import checks were unaffected; this report counts existing files only.",
        "scope": "Native H252 complete gate short models plus independent H188/H220 real down projections; no full8/LB or accuracy acceptance."}
    output.parent.mkdir(parents=True, exist_ok=True); write_json(output, record)
    try:
        for kind, path in (("qwen3", args.qwen_short), ("llama", args.llama_short)):
            torch.cuda.reset_peak_memory_stats(device)
            record["gate_models"][kind] = gate_case(path.resolve(), kind, device); write_json(output, record)
        for kind, path in (("qwen3", args.qwen_inputs), ("llama", args.llama_inputs)):
            torch.cuda.reset_peak_memory_stats(device)
            record["down_projections"][kind] = down_case(path.resolve(), kind, factors, device)
            record["down_projections"][kind]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            write_json(output, record)
        check_hashes(pins); check_hashes(inputs)
        for proof in record["gate_models"].values():
            check_stat(proof["checkpoint_weight_stat"])
        record.update(status="verified", finished_unix=time.time(), quality_evaluation="not_run",
                      loaded_repository_files=loaded_repository_files())
        write_json(output, record)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_unix=time.time())
        write_json(output, record)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
