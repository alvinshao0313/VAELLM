"""Strict pending-geometry fork; native exact-resume validation stays unchanged."""
from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from e2e_common.full_lora import iter_named_peft_lora_layers
from litebsq.vae_linear import VAELinear
from rotation.model_utils import temporarily_disable_torch_weight_init
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_runtime_state_v6 import build_cat_cross_category_runtime_identity, validate_cat_runtime_identity
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload, load_cat_resume_distill_progress
from train_utils.checkpoint_v6 import (load_v6_cat_runtime_state, load_v6_full_checkpoint_into_model,
                                     load_v6_meta, save_v6_full_checkpoint)


CATEGORIES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PROFILES = {"C": (32, 32, 48), "G": (34, 32, 48), "D": (32, 32, 50)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stat(path):
    path = Path(path).resolve()
    item = path.stat()
    return dict(path=str(path), bytes=item.st_size, inode=item.st_ino, device=item.st_dev,
                mtime_ns=item.st_mtime_ns, allocated_bytes=item.st_blocks * 512)


def check_stable_stat(expected):
    actual = stat(expected["path"])
    # Allocation accounting may settle after the writer closes; it is not file content.
    require(all(actual[key] == value for key, value in expected.items() if key != "allocated_bytes"),
            "Protected source stat changed: " + expected["path"])
    return actual


def state_digest(state):
    digest = hashlib.sha256()
    total = 0
    for name, value in sorted(state.items()):
        require(value.device.type == "cpu", "State comparison must remain on CPU.")
        value = value.detach().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)], separators=(",", ":")).encode())
        digest.update(memoryview(raw))
        total += value.numel() * value.element_size()
    return dict(sha256=digest.hexdigest(), tensors=len(state), tensor_bytes=total)


def equal_nested(actual, expected, path="runtime"):
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype
                and actual.shape == expected.shape and torch.equal(actual, expected), "Tensor changed: " + path)
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), "Mapping changed: " + path)
        for key in expected:
            equal_nested(actual[key], expected[key], path + "." + key)
    elif isinstance(expected, (list, tuple)):
        require(type(actual) is type(expected) and len(actual) == len(expected), "Sequence changed: " + path)
        for idx, (left, right) in enumerate(zip(actual, expected, strict=True)):
            equal_nested(left, right, path + "." + str(idx))
    else:
        require(actual == expected, "Value changed: " + path)


def validate_future_change(saved, current, completed):
    """Only pending MLP K/S may differ; compare every other identity field exactly."""
    expected = deepcopy(saved)
    require(list(saved["compression_categories"]) == list(CATEGORIES)
            and list(current["compression_categories"]) == list(CATEGORIES), "Keep all seven original positions.")
    changes = {}
    for category in CATEGORIES:
        before = saved["resolved_category_runtime"][category]
        after = current["resolved_category_runtime"][category]
        if category in CATEGORIES[4:] and category not in completed:
            for key in ("codebook_bits", "residual_stages"):
                if before[key] != after[key]:
                    changes[category + "." + key] = {"source": before[key], "fork": after[key]}
                    expected["resolved_category_runtime"][category][key] = after[key]
    require(expected == current, "Fork changed a completed category or a non-K/S runtime setting.")
    return changes


def make_recipe(template, profile, native_output, runtime):
    argv = list(template["training_argv"][2:])
    gate, up, down = PROFILES[profile]
    replacements = {
        "--output_dir": str(native_output),
        "--codebook_bits": f"default=32,cat:q_proj=38,cat:k_proj=38,cat:o_proj=48,cat:gate_proj={gate},cat:up_proj={up},cat:down_proj={down}",
        "--residual_stages": "default=2",
    }
    for key, value in replacements.items():
        require(argv.count(key) == 1, "Recipe must have exactly one " + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    cfgs = resolve_category_runtime_configs(cat, vae, CATEGORIES)
    training.seed = int(cat.seed)
    require(cat.target_layers == "all" and not cat.skip_layers and cat.seed == 31
            and cat.resolve_after_category_config(CATEGORIES[0]).data.seed == 31,
            "Keep the original all-layer seed31 recipe.")
    identity = build_cat_cross_category_runtime_identity(
        cat_args=cat, vae_args=vae, resolved_category_cfgs=cfgs, compression_categories=CATEGORIES,
        target_layers=runtime["runtime_identity"]["target_layers"], skip_layers=(),
        transpose_modules=cat.transpose_modules.split(","))
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae, training_args=training,
                                                    resolved_category_cfgs=cfgs)
    return argv, identity, normalized


def check_model(model, meta):
    require(not list(iter_named_peft_lora_layers(model)), "Source still contains live PEFT adapters.")
    compressed, dense = meta["compressed_targets"], meta["pending_dense_targets"]
    require(all(isinstance(model.get_submodule(name), VAELinear) for name in compressed), "Missing compressed prefix.")
    require(all(isinstance(model.get_submodule(name), nn.Linear) for name in dense), "Future MLP must retain actual dense weights.")
    require(all(not model.get_submodule(name).trainable_decode for name in compressed), "Source decoder is not finalized.")
    require(not any("lora_" in name for name in model.state_dict()), "Stable source has persistent LoRA state.")


def load_cpu_boundary(path):
    # Native convenience loading always requests FA2, which HF rejects without
    # visible CUDA. This container runs no forward; native v6 owns strict state loading.
    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    require(config.model_type == "qwen3" and config.torch_dtype == torch.bfloat16,
            "This CPU state fork is bound to the original BF16 Qwen3 architecture.")
    with temporarily_disable_torch_weight_init():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16,
                                               attn_implementation="eager")
    model.seqlen = 2048
    return load_v6_full_checkpoint_into_model(model, str(path), map_location="cpu", strict=True,
                                              expected_kind="category_boundary")


def verify_profile(report_path, profile):
    report = read(report_path)
    row = report["profiles"][profile]
    model, meta, result = load_cpu_boundary(row["boundary"])
    require(not result.missing_keys and not result.unexpected_keys, "Fresh strict load failed.")
    check_model(model, meta)
    digest = state_digest(model.state_dict())
    require(digest == report["source_state"], "Fresh strict state differs from the actual source.")
    require(meta["checkpoint_id"] == row["checkpoint_id"], "Fresh checkpoint identity changed.")
    require(not torch.cuda.is_initialized(), "CPU fresh verification initialized CUDA.")
    print(json.dumps(dict(fresh_strict_profile=profile, state=digest, cuda_initialized=False)), flush=True)


def verify_saved_preparation(report_path, output_path):
    """Finish the saved D fork whose terminal allocation-only stat check failed."""
    report_path, output_path = report_path.resolve(), output_path.resolve()
    require(not output_path.exists(), "Refuse to overwrite verification evidence.")
    old = read(report_path)
    require(old["status"] == "failed" and old["error"] == "ValueError: Protected source stat changed."
            and set(old["profiles"]) == {"D"} and old["fresh_strict_representative"] == "D",
            "Only the recorded fully saved/strict-verified D allocation-check failure is eligible.")
    started = time.time()
    result = deepcopy(old)
    result.pop("error"); result.pop("traceback")
    result.update(status="verifying_saved_preparation", exit_code=None,
                  previous_attempt_report=dict(path=str(report_path), sha256=sha(report_path)),
                  verification_started_unix=started,
                  verification_implementation_sha256={str(p): sha(p) for p in Path(__file__).parent.glob("*.py")})
    try:
        current_source_stats = [check_stable_stat(item) for item in old["source_stats"]]
        source = Path(old["source"])
        require(sha(source / "checkpoint_meta.json") == old["source_metadata_sha256"]
                and sha(source / "cat_runtime_state.pt") == old["source_runtime_sha256"], "Protected source changed.")
        meta = load_v6_meta(str(source))
        source_state = torch.load(source / meta["state_dict_file"], map_location="cpu", weights_only=True, mmap=True)
        require(state_digest(source_state) == old["source_state"], "Actual source state changed.")
        del source_state
        row = old["profiles"]["D"]
        boundary = Path(row["boundary"])
        saved_meta = load_v6_meta(str(boundary))
        require(saved_meta["checkpoint_id"] == row["checkpoint_id"]
                and sha(boundary / "checkpoint_meta.json") == row["metadata_sha256"]
                and sha(boundary / "cat_runtime_state.pt") == row["runtime_sha256"], "Saved D identity changed.")
        check_stable_stat(row["weight_stat"])
        saved_state = torch.load(boundary / saved_meta["state_dict_file"], map_location="cpu", weights_only=True, mmap=True)
        require(state_digest(saved_state) == old["source_state"], "Saved D state changed after fresh strict verification.")
        del saved_state
        for path, digest in old["snapshot_source_sha256"].items():
            require(sha(path) == digest, "Frozen source changed: " + path)
        require(not torch.cuda.is_initialized(), "CPU verification initialized CUDA.")
        result.update(status="cpu_verified", exit_code=0, cuda_initialized=False,
                      source_stats_after_verification=current_source_stats,
                      reused_independent_strict_verification=True,
                      saved_weight_content_reverified=True,
                      weight_bytes_written=0,
                      verification_note="Existing independent fresh strict D load is reused only after complete source/D state digests and stable file/metadata/runtime bindings match; allocated_bytes is observational.")
    except BaseException as error:
        result.update(status="failed", exit_code=1, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        result.update(verification_finished_unix=time.time(), verification_elapsed_seconds=time.time() - started)
        write(output_path, result)
    return result


def prepare(args, *, snapshot, topic):
    source, output_root, report_path = args.source.resolve(), args.output_root.resolve(), args.report.resolve()
    previous_attempts = []
    if report_path.exists():
        previous = read(report_path)
        require(args.retry_failed_preparation and previous["status"] == "failed" and not previous["profiles"],
                "An existing report requires an explicit retry of a failed preparation with no saved profiles.")
        previous_attempts = previous.pop("previous_attempts", []) + [previous]
    else:
        require(not args.retry_failed_preparation, "No failed preparation to retry.")
    require(len(set(args.profiles)) == len(args.profiles), "Duplicate profile.")
    started = time.time()
    report = dict(schema="cat_pending_geometry_fork_v1", status="preparing", source=str(source),
                  snapshot=str(snapshot), profiles={}, command=[sys.executable, *sys.argv], started_unix=started,
                  previous_attempts=previous_attempts)
    write(report_path, report)
    try:
        template_path = topic / "formal_k38_seq500/launch.json"
        template = read(template_path)
        require(Path(template["cwd"]).resolve() == snapshot and template["status"] == "completed",
                "Use the actual completed native S2 source snapshot.")
        pins = {path: value for path, value in template["source_sha256"].items()
                if Path(path).is_relative_to(snapshot) and Path(path).suffix == ".py"}
        for path, digest in pins.items():
            require(sha(path) == digest, "Frozen source changed: " + path)
        for module in (sys.modules[load_v6_full_checkpoint_into_model.__module__], sys.modules[parse_cat_runtime_args.__module__]):
            require(Path(module.__file__).resolve().is_relative_to(snapshot), "Mixed public/snapshot imports.")
        meta = load_v6_meta(str(source))
        require(meta["checkpoint_kind"] == "category_boundary" and meta["completed_categories"] in
                (list(CATEGORIES[:4]), list(CATEGORIES[:6])), "Fork only a real after_o or after_up boundary.")
        completed = meta["completed_categories"]
        require(meta["compression_categories"] == list(CATEGORIES) and meta["lora_config"] is None
                and meta["after_category_mode"] == "remaining_lora_prefix_decoder", "Source CAT topology differs.")
        require(all(meta["finalized_status"].get(key) is True for key in
                    ("lora_finalized", "decoder_finalized", "aux_finalized", "stable_category_boundary")),
                "Source is not a stable finalized boundary.")
        require(meta["converted_module_count"] == len(meta["compressed_targets"]) == 36 * len(completed)
                and len(meta["pending_dense_targets"]) == 36 * (7 - len(completed)), "Source inventory differs.")
        require(meta["extra_meta"]["category"] == completed[-1], "Boundary category differs from completion.")
        runtime = load_v6_cat_runtime_state(str(source), map_location="cpu")
        require(set(runtime) == {"format", "version", "runtime_identity", "activation_runtime", "global_adaptive_plan"},
                "Unexpected source runtime fields; inspect before forking.")
        progress = load_cat_resume_distill_progress(str(source))
        require(tuple(completed) == progress.completed_categories and progress.lora_round_idx == len(completed),
                "Recovery history does not preserve original next-round position.")
        source_stats = [stat(source / name) for name in (meta["state_dict_file"], "cat_runtime_state.pt", "checkpoint_meta.json")]
        plans = {}
        for profile in args.profiles:
            boundary = output_root / profile / ("initial_after_o" if len(completed) == 4 else "initial_after_up")
            require(not boundary.exists(), "Refuse to overwrite " + str(boundary))
            argv, identity, normalized = make_recipe(template, profile, output_root / profile / "native", runtime)
            changes = validate_future_change(runtime["runtime_identity"], identity, completed)
            plans[profile] = (boundary, argv, identity, normalized, changes)
        # Negative checks exercise the same validator used for every real fork.
        trial = deepcopy(plans[args.profiles[0]][2])
        trial["resolved_category_runtime"][completed[0]]["codebook_bits"] += 2
        try:
            validate_future_change(runtime["runtime_identity"], trial, completed)
        except ValueError:
            pass
        else:
            raise AssertionError("Completed-geometry guard failed.")
        try:
            validate_cat_runtime_identity(runtime["runtime_identity"], plans[args.profiles[0]][2])
        except ValueError:
            strict_rejected_change = True
        else:
            strict_rejected_change = False
        report.update(source_checkpoint_id=meta["checkpoint_id"], source_stats=source_stats,
                      source_metadata_sha256=sha(source / "checkpoint_meta.json"),
                      source_runtime_sha256=sha(source / "cat_runtime_state.pt"),
                      snapshot_source_sha256=pins, recipe_launch=str(template_path), recipe_launch_sha256=sha(template_path),
                      fork_code_sha256={str(p): sha(p) for p in Path(__file__).parent.glob("*.py")},
                      original_exact_resume_rejects_changed_identity=strict_rejected_change,
                      completed_geometry_negative_check="passed", completed_categories=completed,
                      next_lora_round_idx=progress.lora_round_idx, seed=31,
                      cpu_model_container="Same HF BF16 Qwen3 config, eager construction without any forward; original N strict v6 loader owns full state restoration. GPU training retains native FA2.",
                      rng_boundary="Source category boundary has no global RNG payload. Native entry resets seed31; category positions and recovery history remain unchanged. VAE shuffle=31+category_idx*100000+group_offset; recovery stage_seed=data_seed+round_idx.",
                      deferred_profiles={"D": "Prepare only from the future real C after_up boundary; share its gate/up work."}
                      if "D" not in args.profiles and len(completed) == 4 else {})
        write(report_path, report)
        print("Strictly loading actual finalized source on CPU", flush=True)
        model, loaded_meta, result = load_cpu_boundary(source)
        require(loaded_meta["checkpoint_id"] == meta["checkpoint_id"] and not result.missing_keys
                and not result.unexpected_keys, "Source strict loading failed.")
        check_model(model, meta)
        expected = torch.load(source / meta["state_dict_file"], map_location="cpu", weights_only=True, mmap=True)
        report["source_state"] = state_digest(expected)
        require(state_digest(model.state_dict()) == report["source_state"], "Source load changed persistent values.")
        del expected
        tokenizer = AutoTokenizer.from_pretrained(meta["base_model_path"], use_fast=True, local_files_only=True)
        for profile, (boundary, argv, identity, normalized, changes) in plans.items():
            fork_runtime = dict(runtime, runtime_identity=identity)
            extra = deepcopy(meta["extra_meta"])
            extra["experiment_fork"] = dict(source_dir=str(source), source_checkpoint_id=meta["checkpoint_id"],
                source_metadata_sha256=report["source_metadata_sha256"], profile=profile,
                purpose="New pending-MLP geometry experiment; not exact continuation of the old future geometry.",
                allowed_changes=changes, inherited_state=report["source_state"], inherited_history=True,
                inherited_global_rng=False, seed=31, preparation_report=str(report_path))
            print("Saving independent profile " + profile + " to " + str(boundary), flush=True)
            saved = save_v6_full_checkpoint(model, str(boundary), checkpoint_kind="category_boundary",
                compressed_targets=meta["compressed_targets"], pending_dense_targets=meta["pending_dense_targets"],
                skip_targets=meta["skip_targets"], legacy_original_only_sources=meta["legacy_original_only_sources"],
                train_mode=meta["train_mode"], after_category_mode=meta["after_category_mode"],
                norm_train_mode=meta["norm_train_mode"], lm_head_train_mode=meta["lm_head_train_mode"], lora_config=None,
                resolved_learning_rates=meta["resolved_learning_rates"], completed_categories=completed,
                compression_categories=CATEGORIES, target_layers=meta["target_layers"], target_modules=meta["target_modules"],
                finalized_status=meta["finalized_status"], base_model_path=meta["base_model_path"], tokenizer=tokenizer,
                cat_runtime_state=fork_runtime, extra_meta=extra)
            fresh_meta = load_v6_meta(str(boundary))
            require(read(boundary / "config.json") == read(source / "config.json"),
                    "CPU construction changed the saved model configuration.")
            require(fresh_meta["checkpoint_id"] != meta["checkpoint_id"], "Fork reused source UUID.")
            for key in ("compressed_targets", "pending_dense_targets", "completed_categories", "compression_categories",
                        "converted_modules", "finalized_status", "lora_config"):
                require(fresh_meta[key] == meta[key], "Fork changed actual model topology: " + key)
            payload = torch.load(boundary / fresh_meta["state_dict_file"], map_location="cpu", weights_only=True, mmap=True)
            require(state_digest(payload) == report["source_state"], "Saved fork state changed.")
            del payload
            equal_nested(load_v6_cat_runtime_state(str(boundary), map_location="cpu"), fork_runtime)
            require(stat(boundary / fresh_meta["state_dict_file"])["inode"] != source_stats[0]["inode"],
                    "Fork weight must have its own inode.")
            validate_cat_runtime_identity(identity, fork_runtime["runtime_identity"])
            actual_progress = load_cat_resume_distill_progress(str(boundary))
            require(actual_progress == progress, "Fork changed recovery history/position.")
            report["profiles"][profile] = dict(boundary=str(boundary), checkpoint_id=saved["checkpoint_id"],
                allowed_changes=changes, normalized=normalized, runtime_identity=identity,
                training_argv=[sys.executable, "tools/cat_train.py", *argv, "--resume_from_checkpoint", str(boundary)],
                training_cwd=str(snapshot), state=report["source_state"],
                weight_stat=stat(boundary / fresh_meta["state_dict_file"]),
                metadata_sha256=sha(boundary / "checkpoint_meta.json"), runtime_sha256=sha(boundary / "cat_runtime_state.pt"))
            write(report_path, report)
        require(state_digest(model.state_dict()) == report["source_state"], "Save mutated the shared actual model.")
        del model
        gc.collect()
        representative = args.profiles[0]
        argv = [sys.executable, "-B", str(Path(__file__).with_name("fork_boundary.py")), "--report", str(report_path),
                "--verify-profile", representative]
        subprocess.run(argv, check=True, stdin=subprocess.DEVNULL)
        report["fresh_strict_representative"] = representative
        report["source_stats_after_verification"] = [check_stable_stat(item) for item in source_stats]
        require(sha(source / "checkpoint_meta.json") == report["source_metadata_sha256"]
                and sha(source / "cat_runtime_state.pt") == report["source_runtime_sha256"], "Protected source changed.")
        for path, digest in pins.items():
            require(sha(path) == digest, "Frozen execution source changed during preparation: " + path)
        require(not torch.cuda.is_initialized(), "CPU fork initialized CUDA.")
        report.update(status="cpu_verified", exit_code=0, cuda_initialized=False,
                      weight_bytes_written=sum(row["weight_stat"]["bytes"] for row in report["profiles"].values()))
    except BaseException as error:
        report.update(status="failed", exit_code=1, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      child_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024)
        write(report_path, report)
