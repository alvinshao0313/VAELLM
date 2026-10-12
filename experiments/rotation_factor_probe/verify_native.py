"""Compare real single-matrix native CAT with the isolated block32 wrapper."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch

from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import FORMAT, TARGETS, tensor_record
from experiments.rotation_factor_probe.train import host_for_weight, recipe, run_matrix
from litebsq.protected_channel_quant import decode_protected_channel_weight
from rotation.weight_preprocess import build_weight_rotation, rotation_spec
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_pipeline import apply_group_vae_payload, train_group_vae_payload
from train_utils.utils import LinearRef, configure_deterministic_mode, set_seed


ROOT = Path(__file__).resolve().parents[2]
SEED, STEPS, BATCH_SIZE = 31, 2, 8192


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_pins():
    paths = {Path(__file__).resolve(), ROOT / "experiments/longbench_v1/common.py"}
    paths.update(Path(__file__).with_name(name) for name in (
        "capture.py", "train.py", "transform.py", "factors.py"))
    for directory in ("train_utils", "litebsq", "rotation"):
        paths.update((ROOT / directory).rglob("*.py"))
    return {str(path): sha256(path) for path in sorted(paths)}


def compare_tensor(left, right):
    record = {"native": tensor_record(left), "external": tensor_record(right)}
    same_layout = left.shape == right.shape and left.dtype == right.dtype
    record["exact"] = bool(same_layout and torch.equal(left, right))
    if same_layout and not record["exact"]:
        # Chunk error statistics so a mismatch does not allocate another full FP64 matrix.
        a, b = left.reshape(-1), right.reshape(-1)
        count, max_abs, square_sum, reference_square_sum = 0, 0.0, 0.0, 0.0
        for start in range(0, a.numel(), 1 << 20):
            x, y = a[start:start + (1 << 20)].double(), b[start:start + (1 << 20)].double()
            delta = x - y
            count += int(torch.count_nonzero(delta))
            max_abs = max(max_abs, float(delta.abs().max()))
            square_sum += float(delta.square().sum())
            reference_square_sum += float(x.square().sum())
        record.update(different_elements=count, max_abs=max_abs, error_square_sum=square_sum,
                      reference_square_sum=reference_square_sum)
    return record


def load_inputs(input_dir):
    manifest_path, archive_path = input_dir / "data_manifest.json", input_dir / "inputs.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["format"] == FORMAT and manifest["status"] == "captured"
            and manifest["model_type"] == "qwen3", "This verification requires completed real Qwen captures.")
    require(Path(manifest["archive"]["path"]).resolve() == archive_path,
            "Capture archive path differs from the selected input directory.")
    archive_stat = file_stat(archive_path)
    check_stat({key: manifest["archive"][key] for key in archive_stat})
    require(sha256(archive_path) == manifest["archive"]["sha256"] == manifest["inputs_sha256"],
            "Captured archive bytes changed.")
    archive = torch.load(archive_path, map_location="cpu", weights_only=True, mmap=True)
    require(archive["format"] == FORMAT and archive["model_type"] == "qwen3"
            and archive["checkpoint_id"] == manifest["checkpoint_id"]
            and archive["model_path"] == manifest["model_path"]
            and set(archive["targets"]) == set(TARGETS), "Capture identity or target set differs.")
    for name in TARGETS:
        bundle = archive["targets"][name]
        for key in ("reference_weight", "protected_input_indices", "protected_input_qvalues",
                    "protected_input_scales"):
            require(tensor_record(bundle[key]) == manifest["tensors"][name][key],
                    f"Captured tensor identity differs: {name}/{key}.")
        expected_shape = (12288, 4096) if name.endswith("gate_proj") else (4096, 12288)
        require(tuple(bundle["reference_weight"].shape) == expected_shape
                and bundle["reference_weight"].dtype == torch.bfloat16,
                f"Unexpected Qwen L0 reference geometry: {name}.")
    return archive, {str(manifest_path): sha256(manifest_path),
                     str(archive_path): manifest["inputs_sha256"]}, archive_stat


def native_recipe(model_path, category, protected_count):
    argv, _, _, _ = recipe(model_path, category, steps=STEPS, seed=SEED)
    argv[argv.index("--channel_protect_mode") + 1] = "channel"
    argv[argv.index("--weight_rotation") + 1] = "two_sided"
    argv.extend(["--channel_protect_count", f"default={protected_count}", "--channel_axis", "input",
                 "--channel_quant", "int8", "--weight_rotation_block_size", "32"])
    cat_args, _, training_args, vae_args = parse_cat_runtime_args(argv)
    training_args.seed = SEED
    runtime = resolve_category_runtime_configs(cat_args, vae_args, [category])[category]
    return argv, runtime, vae_args, training_args


def native_matrix(bundle, *, name, model_path, device):
    """Original BF16 host, native fixed protection and native block32 preprocessing."""
    configure_deterministic_mode(True)
    set_seed(SEED)
    weight, indices = bundle["reference_weight"], bundle["protected_input_indices"]
    category = name.rsplit(".", 1)[-1]
    argv, runtime, vae_args, training_args = native_recipe(model_path, category, indices.numel())
    host, linear = host_for_weight(weight, name)
    require(linear.weight.dtype == torch.bfloat16, "Native host must retain original BF16 weights.")
    refs = [LinearRef(name, linear, category, category == "down_proj")]
    # Host allocation consumes RNG; it must not alter the matched VAE initialization.
    torch.manual_seed(SEED)
    payload = train_group_vae_payload(
        model=host, group_refs=refs, group_tag=f"{name}/native_block32/seed{SEED}",
        runtime_cfg=runtime, vae_args=vae_args, training_args=training_args,
        train_device=str(device), convert_device=str(device), do_convert=True,
        batch_size=BATCH_SIZE, gpu_resident_data=True, log_every=100, eval_every=0, eval_blocks=256,
        channel_protect_mode="channel", channel_plan={name: indices},
        channel_rank_metric="channel_weight_abs", channel_axis="input", channel_quant="int8",
        deterministic=True, shuffle_seed=SEED,
    )
    apply_group_vae_payload(model=host, group_refs=refs, group_tag=name,
                            payload=payload, convert_device=str(device))
    layer = host.get_submodule(name).to(device).eval()
    require(layer.parallel_stage_decode and layer.residual_stages == 2,
            "Verification requires the native finalized two-stage packed decoder.")
    with torch.no_grad():
        codes = [layer.get_stage_part_vq_storage(s, 0).detach().cpu().clone() for s in range(2)]
        decoders = [{key: value.detach().cpu().clone() for key, value in
                     layer.get_stage_part_decoder(s, 0).state_dict().items()} for s in range(2)]
        require(all(value.dtype == torch.bfloat16 for state in decoders for value in state.values()
                    if value.is_floating_point()), "Native finalized decoder dtype changed.")
        protection = {key: getattr(layer, key).detach().cpu().clone() for key in (
            "protected_input_indices", "protected_input_qvalues", "protected_input_scales")}
        reconstructed = layer._decode_weight(dtype=torch.bfloat16).cpu()
        # This is a separate evaluation of the native transform, not an observed
        # training intermediate; final code/decoder comparisons verify the training path.
        keep = torch.ones(weight.shape[1], dtype=torch.bool)
        keep[indices] = False
        transform = build_weight_rotation(weight.shape[0], int(keep.sum()),
                                           rotation_spec("two_sided", 32, SEED, name)).to(device)
        reference_rotated = transform(weight[:, keep].to(device=device, dtype=torch.float32)).cpu()
    del host, linear, layer, payload, transform
    gc.collect()
    torch.cuda.empty_cache()
    return reconstructed, {"stage_codes": codes, "stage_decoders": decoders,
                           "protection": protection, "reference_rotated": reference_rotated,
                           "cli_recipe": argv, "runtime": asdict(runtime)}


def compare_case(bundle, *, name, model_path, device):
    started = time.perf_counter()
    native, evidence = native_matrix(bundle, name=name, model_path=model_path, device=device)
    external_evidence = {}
    external, external_report = run_matrix(
        bundle, name=name, model_path=model_path, full=False, factors={}, steps=STEPS,
        seed=SEED, device=device, diagnostic_out=external_evidence)
    require(set(external_evidence) == {"stage_codes", "stage_decoders", "rotated_weight"},
            "External diagnostic contract changed.")
    require(len(external_evidence["stage_codes"]) == len(external_evidence["stage_decoders"]) == 2,
            "External stage count changed.")
    checks = {"deployment_bf16_weight": compare_tensor(native, external),
              "fp32_rotation_against_native_recomputation": compare_tensor(
                  evidence["reference_rotated"], external_evidence["rotated_weight"])}
    for stage in range(2):
        checks[f"stage{stage}_packed_codes"] = compare_tensor(
            evidence["stage_codes"][stage], external_evidence["stage_codes"][stage])
        left, right = evidence["stage_decoders"][stage], external_evidence["stage_decoders"][stage]
        require(set(left) == set(right), "Native and external decoder state keys differ.")
        for key in sorted(left):
            checks[f"stage{stage}_decoder/{key}"] = compare_tensor(left[key], right[key])
    indices = bundle["protected_input_indices"]
    for key, value in evidence["protection"].items():
        checks[key] = compare_tensor(value, bundle[key])
    expected_protected = decode_protected_channel_weight(
        bundle["protected_input_qvalues"], bundle["protected_input_scales"],
        quant_format="int8", dtype=torch.bfloat16, device=torch.device("cpu")).T.contiguous()
    checks["native_protected_columns"] = compare_tensor(native[:, indices], expected_protected)
    checks["external_protected_columns"] = compare_tensor(external[:, indices], expected_protected)
    result = {"name": name, "exact": all(item["exact"] for item in checks.values()),
              "checks": checks, "native_cli_recipe": evidence["cli_recipe"],
              "native_runtime": evidence["runtime"], "external_report": external_report,
              "elapsed_seconds": time.perf_counter() - started}
    del native, external, evidence, external_evidence, expected_protected
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs, output = args.inputs_dir.resolve(), args.output_dir.resolve()
    archive, input_pins, archive_stat = load_inputs(inputs)
    require(not output.exists() and not output.is_relative_to(inputs)
            and not output.is_relative_to(Path(archive["model_path"]).resolve()),
            "Verification output must be a new directory outside captured inputs and original HF weights.")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly one externally assigned visible CUDA GPU is required.")
    configure_deterministic_mode(True)
    set_seed(SEED)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "verification.json"
    record = {"format": "rotation_factor_probe_native_equivalence_v1", "status": "running",
              "scope": "Real Qwen L0 single-matrix block32 wrapper verification; not quality or full CAT equivalence.",
              "started_at_unix": time.time(), "argv": sys.argv, "python": sys.executable, "pid": os.getpid(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "seed": SEED, "shuffle_seed": SEED, "steps_per_stage": STEPS, "batch_size": BATCH_SIZE,
              "targets": list(TARGETS), "input_files_sha256": input_pins, "archive_stat": archive_stat,
              "source_sha256": source_pins(), "model_path": archive["model_path"],
              "protection_source_checkpoint_id": archive["checkpoint_id"], "cases": {},
              "numerical_policy": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                  "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                  "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}}
    write_json(path, record)
    try:
        for name in TARGETS:
            record["cases"][name] = compare_case(archive["targets"][name], name=name,
                                                 model_path=archive["model_path"], device=torch.device("cuda:0"))
            write_json(path, record)
        check_hashes(record["source_sha256"])
        check_hashes(input_pins)
        check_stat(archive_stat)
        exact = all(case["exact"] for case in record["cases"].values())
        record.update(status="verified" if exact else "mismatch", finished_at_unix=time.time(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(path, record)
        require(exact, "Native/external differences are recorded; inspect before using this wrapper for the screen.")
    except BaseException as exc:
        record.update(status="failed" if record["status"] != "mismatch" else "mismatch",
                      error=type(exc).__name__ + ": " + str(exc), finished_at_unix=time.time())
        write_json(path, record)
        raise


if __name__ == "__main__":
    main()
