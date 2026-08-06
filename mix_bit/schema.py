from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

JsonScalar = str | int | float | bool | None

ALLOWED_TRAINING_RECIPE_KEYS: frozenset[str] = frozenset(
    {
        "recipe_id",
        "seed",
        "deterministic",
        "steps_per_category",
        "batch_size",
        "base_ch",
        "num_res_blocks",
        "decoder_base_ch",
        "decoder_num_res_blocks",
        "norm_type",
        "activation_type",
        "decoder_type",
        "recon_loss_type",
        "quantizer_type",
        "gamma0",
        "gamma",
        "zeta",
        "inv_temperature",
        "lr",
        "beta1",
        "beta2",
        "weight_decay",
        "optimizer",
        "lr_scheduler",
        "lr_warmup_steps",
        "l1_weight",
        "lfq_weight",
        "commitment_loss_weight",
        "entropy_loss_weight",
        "normalize_weight",
        "vae_decoder_checkpoint",
        "new_quant",
        "log_every",
        "eval_every",
        "eval_blocks",
        "outlier_protect_mode",
        "outlier_protect_count",
        "outlier_protect_min_per_layer",
        "distill_after_category",
        "eval_ppl",
        "eval_tasks",
        "rot_llm",
        "fp16",
        "bf16",
    }
)


@dataclass(frozen=True)
class CandidateMode:
    name: str
    nominal_bit: float
    codebook_bits: int
    codebook_dim: int
    residual_stages: int


@dataclass(frozen=True)
class CandidateSpaceConfig:
    candidate_space_id: str
    baseline_mode: str
    target_average_bit: float
    modes: tuple[CandidateMode, ...]


@dataclass(frozen=True)
class CategorySpec:
    name: str
    module_suffix: str
    transpose: bool


@dataclass(frozen=True)
class CandidateTrainingSpec:
    linear_group_size: int | Literal["all"]
    allow_tail_group: bool


@dataclass(frozen=True)
class TrainingRecipeConfig:
    recipe_id: str
    values: Mapping[str, JsonScalar]


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    model_path: str
    adapter: str
    only_decoder_projections: bool
    candidate_training: CandidateTrainingSpec
    layer_index_patterns: tuple[str, ...]
    categories: tuple[CategorySpec, ...]
    regression_expectations: dict[str, int]


@dataclass(frozen=True)
class CalibrationConfig:
    source_jsonl: str
    input_format: Literal["auto", "messages", "text", "prompt_response"]
    max_samples: int
    max_length: int
    seed: int
    label_mode: Literal["all_nonpad"]


@dataclass(frozen=True)
class MixBitRunConfig:
    run_id: str
    model_profile: ModelProfile
    candidate_space: CandidateSpaceConfig
    training_recipe: TrainingRecipeConfig
    calibration: CalibrationConfig


@dataclass(frozen=True)
class ResolvedRunConfig:
    config: MixBitRunConfig
    run_config_path: str
    run_config_sha256: str
    model_profile_path: str
    model_profile_sha256: str
    candidate_space_path: str
    candidate_space_sha256: str
    training_recipe_path: str
    training_recipe_sha256: str
    canonical_model_root: str
    canonical_run_root: str


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def load_model_profile(path: str) -> ModelProfile:
    raw = _read_json(path)
    categories_raw = raw.get("categories")
    if not isinstance(categories_raw, list) or not categories_raw:
        raise ValueError(f"Model profile {path} must define a non-empty categories list")

    names: list[str] = []
    suffixes: list[str] = []
    categories: list[CategorySpec] = []
    for item in categories_raw:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid category entry in {path}")
        name = str(item["name"])
        suffix = str(item["module_suffix"])
        transpose = bool(item["transpose"])
        names.append(name)
        suffixes.append(suffix)
        categories.append(CategorySpec(name=name, module_suffix=suffix, transpose=transpose))

    if len(names) != len(set(names)):
        raise ValueError(f"Model profile {path} has duplicate logical categories")
    if len(suffixes) != len(set(suffixes)):
        raise ValueError(f"Model profile {path} has ambiguous module_suffix mapping")

    training_raw = raw.get("candidate_training")
    if not isinstance(training_raw, dict):
        raise ValueError(f"Model profile {path} missing candidate_training")
    group_size = training_raw["linear_group_size"]
    if group_size != "all" and not isinstance(group_size, int):
        raise ValueError(
            f"Model profile {path} candidate_training.linear_group_size must be int or 'all'"
        )

    patterns = raw.get("layer_index_patterns")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError(f"Model profile {path} must define layer_index_patterns")

    expectations_raw = raw.get("regression_expectations", {})
    if expectations_raw is None:
        expectations_raw = {}
    if not isinstance(expectations_raw, dict):
        raise ValueError(f"Model profile {path} regression_expectations must be an object")
    expectations = {str(k): int(v) for k, v in expectations_raw.items()}

    return ModelProfile(
        model_id=str(raw["model_id"]),
        model_path=str(raw["model_path"]),
        adapter=str(raw["adapter"]),
        only_decoder_projections=bool(raw["only_decoder_projections"]),
        candidate_training=CandidateTrainingSpec(
            linear_group_size=group_size,
            allow_tail_group=bool(training_raw["allow_tail_group"]),
        ),
        layer_index_patterns=tuple(str(p) for p in patterns),
        categories=tuple(categories),
        regression_expectations=expectations,
    )


def load_training_recipe(path: str) -> TrainingRecipeConfig:
    raw = _read_json(path)
    unknown = sorted(set(raw) - ALLOWED_TRAINING_RECIPE_KEYS)
    if unknown:
        raise ValueError(f"Training recipe {path} has unknown keys: {unknown}")
    if "recipe_id" not in raw:
        raise ValueError(f"Training recipe {path} missing recipe_id")

    values: dict[str, JsonScalar] = {}
    for key, value in raw.items():
        if key == "recipe_id":
            continue
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise ValueError(f"Training recipe {path} key {key!r} is not a JSON scalar")
        values[key] = value

    if bool(values.get("fp16")) and bool(values.get("bf16")):
        raise ValueError(f"Training recipe {path} enables both fp16 and bf16")

    return TrainingRecipeConfig(recipe_id=str(raw["recipe_id"]), values=values)


def _resolve_relative(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def resolved_run_config_to_dict(resolved: ResolvedRunConfig) -> dict[str, Any]:
    config = resolved.config
    return {
        "calibration": asdict(config.calibration),
        "candidate_space": {
            "baseline_mode": config.candidate_space.baseline_mode,
            "candidate_space_id": config.candidate_space.candidate_space_id,
            "modes": [asdict(mode) for mode in config.candidate_space.modes],
            "target_average_bit": config.candidate_space.target_average_bit,
        },
        "candidate_space_path": resolved.candidate_space_path,
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "canonical_model_root": resolved.canonical_model_root,
        "canonical_run_root": resolved.canonical_run_root,
        "model_profile": {
            "adapter": config.model_profile.adapter,
            "candidate_training": asdict(config.model_profile.candidate_training),
            "categories": [asdict(cat) for cat in config.model_profile.categories],
            "layer_index_patterns": list(config.model_profile.layer_index_patterns),
            "model_id": config.model_profile.model_id,
            "model_path": config.model_profile.model_path,
            "only_decoder_projections": config.model_profile.only_decoder_projections,
            "regression_expectations": dict(config.model_profile.regression_expectations),
        },
        "model_profile_path": resolved.model_profile_path,
        "model_profile_sha256": resolved.model_profile_sha256,
        "run_config_path": resolved.run_config_path,
        "run_config_sha256": resolved.run_config_sha256,
        "run_id": config.run_id,
        "training_recipe": {
            "recipe_id": config.training_recipe.recipe_id,
            "values": dict(config.training_recipe.values),
        },
        "training_recipe_path": resolved.training_recipe_path,
        "training_recipe_sha256": resolved.training_recipe_sha256,
    }


def write_resolved_run_config(resolved: ResolvedRunConfig, path: str | Path | None = None) -> str:
    out_path = Path(path) if path is not None else Path(resolved.canonical_run_root) / "resolved_run_config.json"
    _write_json_atomic(out_path, resolved_run_config_to_dict(resolved))
    return str(out_path)


def load_resolved_run_config_file(path: str | Path) -> dict[str, Any]:
    return _read_json(path)


def validate_resolved_run_config_resume(resolved: ResolvedRunConfig) -> None:
    existing_path = Path(resolved.canonical_run_root) / "resolved_run_config.json"
    if not existing_path.is_file():
        return
    existing = load_resolved_run_config_file(existing_path)
    checks = {
        "run_config_sha256": resolved.run_config_sha256,
        "model_profile_sha256": resolved.model_profile_sha256,
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "training_recipe_sha256": resolved.training_recipe_sha256,
    }
    for key, expected in checks.items():
        found = existing.get(key)
        if found != expected:
            raise ValueError(
                f"Resolved run config hash mismatch for {key}: "
                f"existing={found!r} current={expected!r}"
            )


def resolve_run_config(
    run_config_path: str,
    *,
    repo_root: str | None = None,
    result_root: str | None = None,
    write: bool = True,
) -> ResolvedRunConfig:
    from mix_bit.candidate_space import load_candidate_space

    run_path = Path(run_config_path).resolve()
    raw = _read_json(run_path)
    base_dir = run_path.parent
    root = Path(repo_root).resolve() if repo_root is not None else default_repo_root()

    model_profile_path = _resolve_relative(str(raw["model_profile"]), base_dir)
    candidate_space_path = _resolve_relative(str(raw["candidate_space"]), base_dir)
    training_recipe_path = _resolve_relative(str(raw["training_recipe"]), base_dir)

    calibration_raw = raw["calibration"]
    if not isinstance(calibration_raw, dict):
        raise ValueError(f"Run config {run_path} calibration must be an object")
    # Calibration JSONL resolves relative to the repository root (not the run-config dir).
    # Absolute paths are kept as-is. Profile/space/recipe refs stay run-config-relative.
    source_jsonl = str(
        _resolve_relative(str(calibration_raw["source_jsonl"]), root)
    )

    input_format = str(calibration_raw["input_format"])
    if input_format not in {"auto", "messages", "text", "prompt_response"}:
        raise ValueError(f"Unsupported calibration.input_format: {input_format!r}")
    label_mode = str(calibration_raw["label_mode"])
    if label_mode != "all_nonpad":
        raise ValueError(f"Unsupported calibration.label_mode: {label_mode!r}")

    model_profile = load_model_profile(str(model_profile_path))
    candidate_space = load_candidate_space(str(candidate_space_path))
    training_recipe = load_training_recipe(str(training_recipe_path))

    run_id = str(raw["run_id"])
    results = Path(result_root).resolve() if result_root is not None else (root / ".result")
    canonical_model_root = results / "mix_bit" / model_profile.model_id
    canonical_run_root = canonical_model_root / "runs" / run_id

    resolved = ResolvedRunConfig(
        config=MixBitRunConfig(
            run_id=run_id,
            model_profile=model_profile,
            candidate_space=candidate_space,
            training_recipe=training_recipe,
            calibration=CalibrationConfig(
                source_jsonl=source_jsonl,
                input_format=input_format,  # type: ignore[arg-type]
                max_samples=int(calibration_raw["max_samples"]),
                max_length=int(calibration_raw["max_length"]),
                seed=int(calibration_raw["seed"]),
                label_mode=label_mode,  # type: ignore[arg-type]
            ),
        ),
        run_config_path=str(run_path),
        run_config_sha256=sha256_file(run_path),
        model_profile_path=str(model_profile_path),
        model_profile_sha256=sha256_file(model_profile_path),
        candidate_space_path=str(candidate_space_path),
        candidate_space_sha256=sha256_file(candidate_space_path),
        training_recipe_path=str(training_recipe_path),
        training_recipe_sha256=sha256_file(training_recipe_path),
        canonical_model_root=str(canonical_model_root),
        canonical_run_root=str(canonical_run_root),
    )

    if write:
        existing = Path(resolved.canonical_run_root) / "resolved_run_config.json"
        if existing.is_file():
            validate_resolved_run_config_resume(resolved)
        else:
            write_resolved_run_config(resolved)
    return resolved
