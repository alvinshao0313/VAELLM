from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mix_bit.model_inventory import (
    ModelInventory,
    TargetLinearSpec,
    inventory_from_targets,
    validate_inventory_for_run,
)
from mix_bit.schema import (
    CandidateMode,
    ResolvedRunConfig,
    default_repo_root,
    sha256_file,
)

_OVERRIDE_RECIPE_KEYS = frozenset(
    {
        "vae_steps",
        "base_ch",
        "num_res_blocks",
        "decoder_base_ch",
        "decoder_num_res_blocks",
        "norm_type",
        "activation_type",
        "decoder_type",
        "recon_loss_type",
        "channel_protect_count",
    }
)
_STORE_TRUE_RECIPE_KEYS = frozenset({"normalize_weight", "new_quant"})
_BOOL_STRING_RECIPE_KEYS = frozenset(
    {
        "deterministic",
        "vae_decoder_checkpoint",
        "skip_ppl_eval",
        "rot_llm",
        "fp16",
        "bf16",
    }
)
_ALWAYS_FIXED_ARGS: tuple[tuple[str, str | None], ...] = (
    ("--convert", None),
    ("--save_candidate_artifact", "true"),
    ("--train_device", "cuda"),
    ("--convert_device", "cuda"),
    ("--skip_layers", ""),
    ("--after_category_mode", "none"),
    ("--skip_ppl_eval", "true"),
    ("--eval_tasks", ""),
    ("--rot_llm", "false"),
    ("--channel_protect_mode", "channel"),
    ("--channel_protect_count", "default=0"),
    ("--channel_min_per_layer", "0"),
    ("--bf16", "true"),
    ("--fp16", "false"),
)


@dataclass(frozen=True)
class TrialSpec:
    model_id: str
    run_id: str
    category_name: str
    target_module_suffix: str
    transpose_module_suffixes: tuple[str, ...]
    expected_module_names: tuple[str, ...]
    resolved_linear_group_size: int
    model_inventory_fingerprint: str
    run_config_sha256: str
    candidate_space_sha256: str
    training_recipe_sha256: str
    model_profile_sha256: str
    mode: CandidateMode
    trial_root: str
    cat_train_output_parent: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _append_manifest(manifest_path: Path, record: Mapping[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def _format_recipe_value(key: str, value: Any) -> str:
    if key in _OVERRIDE_RECIPE_KEYS:
        text = str(value)
        if "=" not in text:
            return f"default={text}"
        return text
    if key in _BOOL_STRING_RECIPE_KEYS:
        return _bool_text(value)
    if isinstance(value, bool):
        return _bool_text(value)
    return str(value)


def _resolve_group_size(
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    category_name: str,
) -> int:
    group = resolved.config.model_profile.candidate_training.linear_group_size
    category_count = sum(1 for t in inventory.targets if t.category == category_name)
    if category_count < 1:
        raise ValueError(f"Category {category_name!r} has no inventory targets")
    if group == "all":
        return int(category_count)
    size = int(group)
    allow_tail = bool(resolved.config.model_profile.candidate_training.allow_tail_group)
    if not allow_tail and category_count % size != 0:
        raise ValueError(
            f"Category {category_name!r} count {category_count} is not divisible by "
            f"linear_group_size={size} and allow_tail_group=false"
        )
    return size


def candidate_pool_root(resolved: ResolvedRunConfig, output_root: str | None = None) -> Path:
    if output_root is not None:
        return Path(output_root).resolve()
    return Path(resolved.canonical_run_root) / "candidate_pool"


def generate_candidate_trials(
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    *,
    output_root: str | None = None,
) -> list[TrialSpec]:
    validate_inventory_for_run(inventory, resolved)
    profile = resolved.config.model_profile
    modes = resolved.config.candidate_space.modes
    transpose_suffixes = tuple(cat.module_suffix for cat in profile.categories if cat.transpose)
    pool_root = candidate_pool_root(resolved, output_root=output_root)
    trials: list[TrialSpec] = []
    for category in profile.categories:
        expected = tuple(
            t.module_name for t in inventory.targets if t.category == category.name
        )
        if not expected:
            raise ValueError(f"No inventory modules for category {category.name!r}")
        group_size = _resolve_group_size(resolved, inventory, category.name)
        for mode in modes:
            trial_root = pool_root / category.name / mode.name
            trials.append(
                TrialSpec(
                    model_id=profile.model_id,
                    run_id=resolved.config.run_id,
                    category_name=category.name,
                    target_module_suffix=category.module_suffix,
                    transpose_module_suffixes=transpose_suffixes,
                    expected_module_names=expected,
                    resolved_linear_group_size=group_size,
                    model_inventory_fingerprint=inventory.fingerprint_sha256,
                    run_config_sha256=resolved.run_config_sha256,
                    candidate_space_sha256=resolved.candidate_space_sha256,
                    training_recipe_sha256=resolved.training_recipe_sha256,
                    model_profile_sha256=resolved.model_profile_sha256,
                    mode=mode,
                    trial_root=str(trial_root),
                    cat_train_output_parent=str(trial_root / "runs"),
                )
            )
    return trials


def trial_id(trial: TrialSpec) -> str:
    return f"{trial.category_name}/{trial.mode.name}"


def _resolved_python_executable() -> str:
    python_executable = str(Path(sys.executable).resolve())
    if not Path(python_executable).is_file():
        raise FileNotFoundError(
            f"Current Python executable does not exist: {python_executable}"
        )
    return python_executable


def build_trial_command(trial: TrialSpec, resolved: ResolvedRunConfig, gpu_id: str) -> list[str]:
    repo_root = default_repo_root()
    script = str(repo_root / "mix_bit" / "scripts" / "train_candidate_single.sh")
    python_executable = _resolved_python_executable()
    cmd: list[str] = [script, str(gpu_id), python_executable]

    recipe_values = dict(resolved.config.training_recipe.values)
    # Forced candidate-pool values win over recipe.
    recipe_values["channel_protect_mode"] = "channel"
    recipe_values["channel_protect_count"] = 0
    recipe_values["channel_min_per_layer"] = 0
    recipe_values["after_category_mode"] = "none"
    recipe_values["skip_ppl_eval"] = True
    recipe_values["eval_tasks"] = ""
    recipe_values["rot_llm"] = False
    recipe_values["bf16"] = True
    recipe_values["fp16"] = False

    for key, value in recipe_values.items():
        flag = f"--{key}"
        if key in _STORE_TRUE_RECIPE_KEYS:
            if bool(value):
                cmd.append(flag)
            continue
        cmd.extend([flag, _format_recipe_value(key, value)])

    # Ensure always-fixed flags are present with exact required values.
    def _set_or_replace(flag: str, value: str | None) -> None:
        if flag in cmd:
            idx = cmd.index(flag)
            if value is None:
                return
            if idx + 1 < len(cmd) and not cmd[idx + 1].startswith("--"):
                cmd[idx + 1] = value
            else:
                cmd.insert(idx + 1, value)
            return
        if value is None:
            cmd.append(flag)
        else:
            cmd.extend([flag, value])

    for flag, value in _ALWAYS_FIXED_ARGS:
        _set_or_replace(flag, value)

    if "--save_model" in cmd:
        raise ValueError("Candidate trial command must not include --save_model")

    profile = resolved.config.model_profile
    cmd.extend(
        [
            "--model_path",
            profile.model_path,
            "--compression_categories",
            trial.target_module_suffix,
            "--transpose_modules",
            ",".join(trial.transpose_module_suffixes),
            "--linear_group_size",
            str(trial.resolved_linear_group_size),
            "--allow_tail_group",
            _bool_text(profile.candidate_training.allow_tail_group),
            "--output_dir",
            trial.cat_train_output_parent,
            "--candidate_artifact_spec",
            str(Path(trial.trial_root) / "trial_spec.json"),
            "--candidate_artifact_output_dir",
            str(Path(trial.trial_root) / "artifact"),
            "--codebook_bits",
            f"default={trial.mode.codebook_bits}",
            "--codebook_dim",
            f"default={trial.mode.codebook_dim}",
            "--residual_stages",
            f"default={trial.mode.residual_stages}",
        ]
    )
    return cmd


def trial_spec_to_dict(trial: TrialSpec, command: Sequence[str]) -> dict[str, Any]:
    return {
        "cat_train_output_parent": trial.cat_train_output_parent,
        "category_name": trial.category_name,
        "command_args": list(command),
        "python_executable": command[2],
        "expected_module_names": list(trial.expected_module_names),
        "mode": asdict(trial.mode),
        "model_id": trial.model_id,
        "model_inventory_fingerprint": trial.model_inventory_fingerprint,
        "resolved_linear_group_size": trial.resolved_linear_group_size,
        "run_config_sha256": trial.run_config_sha256,
        "candidate_space_sha256": trial.candidate_space_sha256,
        "run_id": trial.run_id,
        "target_module_suffix": trial.target_module_suffix,
        "training_recipe_sha256": trial.training_recipe_sha256,
        "model_profile_sha256": trial.model_profile_sha256,
        "transpose_module_suffixes": list(trial.transpose_module_suffixes),
        "trial_root": trial.trial_root,
    }


def write_trial_spec(trial: TrialSpec, command: Sequence[str]) -> str:
    path = Path(trial.trial_root) / "trial_spec.json"
    _write_json_atomic(path, trial_spec_to_dict(trial, command))
    return str(path)


def load_trial_spec(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"trial_spec must be a JSON object: {path}")
    return raw


def is_trial_complete(trial: TrialSpec) -> bool:
    """Return True only if validate_trial_completion succeeds.

    Replaces the previous hash-only check; the full contract (mode, hashes, module
    specs, expected names) must hold before a trial is considered resumable.
    """
    try:
        validate_trial_completion(trial)
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def validate_trial_completion(trial: TrialSpec) -> None:
    """Raise ValueError unless the existing artifact exactly belongs to this trial.

    Verifies the three artifact files exist, their hashes cross-agree, the meta and
    completed formats match, and every TrialSpec field (hashes, category, mode,
    expected module names with order and uniqueness) matches the artifact. Each
    module spec must also pass the Task 1 mode contract.
    """
    from mix_bit.candidate_contract import (
        validate_mode_payload,
        validate_module_spec_mode_contract,
    )
    from mix_bit.candidate_artifact import (
        CANDIDATE_META_FILENAME,
        COMPLETED_FILENAME,
        MODULE_STATE_FILENAME,
    )

    artifact = Path(trial.trial_root) / "artifact"
    completed_path = artifact / COMPLETED_FILENAME
    meta_path = artifact / CANDIDATE_META_FILENAME
    state_path = artifact / MODULE_STATE_FILENAME
    for path in (completed_path, meta_path, state_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing artifact file for {trial_id(trial)}: {path}")

    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(completed, dict):
        raise ValueError(f"{trial_id(trial)}: completed.json must be a JSON object")
    if not isinstance(meta, dict):
        raise ValueError(f"{trial_id(trial)}: candidate_meta.json must be a JSON object")

    expected_format = "vaellm_candidate_modules_v1"
    if str(completed.get("format")) != expected_format:
        raise ValueError(
            f"{trial_id(trial)}: completed.json format {completed.get('format')!r} "
            f"!= expected {expected_format!r}"
        )
    if str(meta.get("format")) != expected_format:
        raise ValueError(
            f"{trial_id(trial)}: candidate_meta.json format {meta.get('format')!r} "
            f"!= expected {expected_format!r}"
        )

    state_sha = sha256_file(state_path)
    meta_sha = sha256_file(meta_path)
    if str(completed.get("module_state_sha256")) != state_sha:
        raise ValueError(
            f"{trial_id(trial)}: completed module_state_sha256 mismatch: "
            f"completed={completed.get('module_state_sha256')!r} file={state_sha!r}"
        )
    if str(completed.get("candidate_meta_sha256")) != meta_sha:
        raise ValueError(
            f"{trial_id(trial)}: completed candidate_meta_sha256 mismatch: "
            f"completed={completed.get('candidate_meta_sha256')!r} file={meta_sha!r}"
        )
    if str(meta.get("module_state_sha256")) != state_sha:
        raise ValueError(
            f"{trial_id(trial)}: candidate_meta module_state_sha256 mismatch: "
            f"meta={meta.get('module_state_sha256')!r} file={state_sha!r}"
        )

    label = trial_id(trial)
    hash_fields = (
        "run_config_sha256",
        "candidate_space_sha256",
        "training_recipe_sha256",
        "model_profile_sha256",
        "model_inventory_fingerprint",
    )
    for key in hash_fields:
        expected = getattr(trial, key)
        found = meta.get(key)
        if found is None:
            raise ValueError(f"{label}: candidate_meta missing required field {key!r}")
        if str(found) != str(expected):
            raise ValueError(
                f"{label}: {key} mismatch: meta={found!r} trial={expected!r}"
            )

    if str(meta.get("category_name")) != trial.category_name:
        raise ValueError(
            f"{label}: candidate_meta category_name {meta.get('category_name')!r} "
            f"!= trial {trial.category_name!r}"
        )

    meta_mode = meta.get("mode")
    if not isinstance(meta_mode, dict):
        raise ValueError(f"{label}: candidate_meta mode must be an object, got {meta_mode!r}")
    validate_mode_payload(meta_mode, trial.mode, label=f"{label}/candidate_meta.mode")

    expected_names = list(trial.expected_module_names)
    if len(expected_names) != len(set(expected_names)):
        raise ValueError(f"{label}: trial expected_module_names contains duplicates: {expected_names}")
    meta_expected = meta.get("expected_module_names")
    if not isinstance(meta_expected, list):
        raise ValueError(f"{label}: candidate_meta.expected_module_names must be a list")
    meta_expected_names = [str(x) for x in meta_expected]
    if meta_expected_names != expected_names:
        raise ValueError(
            f"{label}: expected_module_names mismatch (including order): "
            f"meta={meta_expected_names} trial={expected_names}"
        )

    module_specs = meta.get("module_specs")
    if not isinstance(module_specs, list):
        raise ValueError(f"{label}: candidate_meta.module_specs must be a list")
    spec_by_name: dict[str, dict[str, Any]] = {}
    for raw_spec in module_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError(f"{label}: module_spec entries must be objects")
        name = str(raw_spec.get("name", ""))
        if not name:
            raise ValueError(f"{label}: module_spec missing name")
        if name in spec_by_name:
            raise ValueError(f"{label}: duplicate module_spec for {name!r}")
        spec_by_name[name] = dict(raw_spec)
    if set(spec_by_name) != set(expected_names):
        missing = sorted(set(expected_names) - set(spec_by_name))
        extra = sorted(set(spec_by_name) - set(expected_names))
        raise ValueError(
            f"{label}: module_spec name set mismatch: missing={missing} extra={extra}"
        )

    for name in expected_names:
        validate_module_spec_mode_contract(
            spec_by_name[name],
            trial.mode,
            label=f"{label}/{name}",
        )

    if int(completed.get("module_count", -1)) != len(expected_names):
        raise ValueError(
            f"{label}: completed module_count {completed.get('module_count')!r} "
            f"!= expected {len(expected_names)}"
        )


def resolve_new_cat_train_run_dir(runs_dir: str | Path, *, before_snapshot: set[str]) -> Path:
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        raise ValueError(f"cat_train runs parent does not exist: {runs_path}")
    after = {p.name for p in runs_path.iterdir() if p.is_dir()}
    new_names = sorted(after - set(before_snapshot))
    if len(new_names) != 1:
        raise ValueError(
            f"Expected exactly one new cat_train run dir under {runs_path}, "
            f"got {len(new_names)}: {new_names}"
        )
    return runs_path / new_names[0]


def preflight_loader_inventory(
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    *,
    access_token: str | None = None,
) -> ModelInventory:
    """Load via production get_model + cat_train discover helpers; require inventory equality."""
    from rotation.model_utils import get_model
    from train_utils.utils import collect_linears, extract_layer_idx

    validate_inventory_for_run(inventory, resolved)
    profile = resolved.config.model_profile
    suffix_to_category = {cat.module_suffix: cat for cat in profile.categories}
    concrete_suffixes = [cat.module_suffix for cat in profile.categories]
    transpose_suffixes = [cat.module_suffix for cat in profile.categories if cat.transpose]

    model = get_model(profile.model_path, access_token)
    try:
        refs = collect_linears(
            model,
            transpose_suffixes,
            only_decoder_projections=bool(profile.only_decoder_projections),
            categories=concrete_suffixes,
        )
        unsupported: list[str] = []
        targets: list[TargetLinearSpec] = []
        seen: set[str] = set()
        for ref in refs:
            block_index = extract_layer_idx(ref.name)
            if block_index is None:
                unsupported.append(ref.name)
                continue
            category = suffix_to_category.get(ref.category)
            if category is None:
                raise ValueError(
                    f"collect_linears returned unknown concrete category {ref.category!r} "
                    f"for module {ref.name!r}"
                )
            if ref.name in seen:
                raise ValueError(f"Duplicated discovered module name {ref.name!r}")
            seen.add(ref.name)
            in_features = int(ref.module.in_features)
            out_features = int(ref.module.out_features)
            has_bias = ref.module.bias is not None
            targets.append(
                TargetLinearSpec(
                    module_name=ref.name,
                    category=category.name,
                    module_suffix=category.module_suffix,
                    block_index=int(block_index),
                    in_features=in_features,
                    out_features=out_features,
                    has_bias=has_bias,
                    param_count=in_features * out_features,
                    transpose=bool(category.transpose),
                )
            )
        if unsupported:
            preview = "\n".join(unsupported)
            raise ValueError(
                "Production cat_train extract_layer_idx does not support these module paths:\n"
                f"{preview}"
            )
        category_order = {cat.name: idx for idx, cat in enumerate(profile.categories)}
        targets.sort(
            key=lambda item: (
                item.block_index,
                category_order[item.category],
                item.module_name,
            )
        )
        discovered = inventory_from_targets(
            profile=profile,
            model=model,
            targets=tuple(targets),
            model_profile_sha256=resolved.model_profile_sha256,
        )
        _assert_inventories_equal(inventory, discovered)
        return discovered
    finally:
        del model


def _assert_inventories_equal(expected: ModelInventory, actual: ModelInventory) -> None:
    if expected.fingerprint_sha256 != actual.fingerprint_sha256:
        raise ValueError(
            "Loader preflight inventory fingerprint mismatch: "
            f"persisted={expected.fingerprint_sha256} discovered={actual.fingerprint_sha256}"
        )
    exp_names = {t.module_name for t in expected.targets}
    act_names = {t.module_name for t in actual.targets}
    if exp_names != act_names:
        missing = sorted(exp_names - act_names)
        extra = sorted(act_names - exp_names)
        raise ValueError(
            f"Loader preflight module-name set mismatch. missing={missing} extra={extra}"
        )
    exp_by_name = {t.module_name: t for t in expected.targets}
    for target in actual.targets:
        other = exp_by_name[target.module_name]
        if (
            target.category != other.category
            or target.module_suffix != other.module_suffix
            or target.block_index != other.block_index
            or target.in_features != other.in_features
            or target.out_features != other.out_features
            or target.has_bias != other.has_bias
            or target.transpose != other.transpose
            or target.param_count != other.param_count
        ):
            raise ValueError(
                f"Loader preflight target mismatch for {target.module_name}: "
                f"persisted={other} discovered={target}"
            )


def _snapshot_run_children(runs_dir: Path) -> set[str]:
    if not runs_dir.is_dir():
        return set()
    return {p.name for p in runs_dir.iterdir() if p.is_dir()}


def _run_one_trial(
    trial: TrialSpec,
    resolved: ResolvedRunConfig,
    gpu_id: str,
    *,
    manifest_path: Path,
    manifest_lock: threading.Lock,
) -> int:
    command = build_trial_command(trial, resolved, gpu_id)
    write_trial_spec(trial, command)
    runs_dir = Path(trial.cat_train_output_parent)
    runs_dir.mkdir(parents=True, exist_ok=True)
    before = _snapshot_run_children(runs_dir)
    log_path = Path(trial.trial_root) / "trial.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    with manifest_lock:
        _append_manifest(
            manifest_path,
            {
                "event": "start",
                "trial_id": trial_id(trial),
                "category": trial.category_name,
                "mode": trial.mode.name,
                "gpu": gpu_id,
                "command": shlex.join(command),
                "start_time": started,
            },
        )
    with open(log_path, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"command: {shlex.join(command)}\n")
        log_handle.flush()
        proc = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(default_repo_root()),
        )
        exit_code = int(proc.wait())
    ended = _utc_now()
    artifact_dir = str(Path(trial.trial_root) / "artifact")
    run_dir: Optional[str] = None
    if exit_code == 0:
        resolved_run = resolve_new_cat_train_run_dir(runs_dir, before_snapshot=before)
        run_dir = str(resolved_run)
        try:
            validate_trial_completion(trial)
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            exit_code = 1
            with open(log_path, "a", encoding="utf-8") as log_handle:
                log_handle.write(
                    f"error: trial exited 0 but compact artifact completion failed: "
                    f"{type(exc).__name__}: {exc}\n"
                )
        # Production path must not leave a full-model state file as the completion contract.
        full_state = Path(run_dir) / "final_model" / "pytorch_model.bin"
        if full_state.is_file():
            exit_code = 1
            with open(log_path, "a", encoding="utf-8") as log_handle:
                log_handle.write(f"error: unexpected full-model state at {full_state}\n")
    with manifest_lock:
        _append_manifest(
            manifest_path,
            {
                "event": "end",
                "trial_id": trial_id(trial),
                "category": trial.category_name,
                "mode": trial.mode.name,
                "gpu": gpu_id,
                "command": shlex.join(command),
                "start_time": started,
                "end_time": ended,
                "exit_code": exit_code,
                "resolved_cat_train_run_dir": run_dir,
                "candidate_artifact_dir": artifact_dir,
            },
        )
    return exit_code


def run_candidate_pool(
    *,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    inventory_path: str,
    gpus: Sequence[str],
    dry_run: bool,
    output_root: str | None = None,
    access_token: str | None = None,
) -> int:
    if not gpus:
        raise ValueError("--gpus must be a non-empty comma-separated GPU id list")
    validate_inventory_for_run(inventory, resolved)
    preflight_loader_inventory(resolved, inventory, access_token=access_token)

    pool_root = candidate_pool_root(resolved, output_root=output_root)
    trials = generate_candidate_trials(resolved, inventory, output_root=output_root)

    pool_root.mkdir(parents=True, exist_ok=True)
    manifest_path = pool_root / "run_manifest.jsonl"
    meta = {
        "inventory_path": str(Path(inventory_path).resolve()),
        "output_root": str(pool_root),
        "output_root_overridden": output_root is not None,
        "run_config_path": resolved.run_config_path,
        "run_config_sha256": resolved.run_config_sha256,
        "gpus": list(gpus),
        "dry_run": bool(dry_run),
        "trial_count": len(trials),
        "created_at_utc": _utc_now(),
        "python_executable": _resolved_python_executable(),
    }
    _write_json_atomic(pool_root / "scheduler_meta.json", meta)

    pending = [t for t in trials if not is_trial_complete(t)]
    print(f"candidate_pool_root={pool_root}")
    print(f"total_trials={len(trials)}")
    print(f"pending_trials={len(pending)}")
    print(f"completed_trials={len(trials) - len(pending)}")

    if dry_run:
        seen_commands: set[str] = set()
        for trial in trials:
            command = build_trial_command(trial, resolved, gpus[0])
            rendered = shlex.join(command)
            if rendered in seen_commands:
                raise ValueError(f"Duplicate dry-run command for {trial_id(trial)}")
            seen_commands.add(rendered)
            if "--save_model" in command:
                raise ValueError("dry-run command unexpectedly contains --save_model")
            if "--save_candidate_artifact" not in command:
                raise ValueError("dry-run command missing --save_candidate_artifact")
            print(rendered)
        print(f"dry_run_unique_commands={len(seen_commands)}")
        return 0

    if not pending:
        return 0

    manifest_lock = threading.Lock()
    failures = 0
    gpu_list = [str(g).strip() for g in gpus if str(g).strip()]
    with ThreadPoolExecutor(max_workers=len(gpu_list)) as executor:
        futures: dict[Future[int], str] = {}
        pending_iter = iter(pending)

        def _submit(gpu: str, trial: TrialSpec) -> None:
            fut = executor.submit(
                _run_one_trial,
                trial,
                resolved,
                gpu,
                manifest_path=manifest_path,
                manifest_lock=manifest_lock,
            )
            futures[fut] = gpu

        for gpu in gpu_list:
            try:
                trial = next(pending_iter)
            except StopIteration:
                break
            _submit(gpu, trial)

        while futures:
            done = [fut for fut in list(futures) if fut.done()]
            if not done:
                time.sleep(0.05)
                continue
            for fut in done:
                gpu = futures.pop(fut)
                code = int(fut.result())
                if code != 0:
                    failures += 1
                try:
                    nxt = next(pending_iter)
                except StopIteration:
                    continue
                _submit(gpu, nxt)

    return 1 if failures else 0
