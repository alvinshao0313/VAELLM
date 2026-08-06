from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest
from torch import nn

from mix_bit.candidate_pool import (
    build_trial_command,
    generate_candidate_trials,
    is_trial_complete,
    preflight_loader_inventory,
    resolve_new_cat_train_run_dir,
    run_candidate_pool,
    trial_spec_to_dict,
    validate_trial_completion,
    write_trial_spec,
)
from mix_bit.model_inventory import (
    inventory_from_targets,
    write_model_inventory,
)
from mix_bit.schema import (
    CalibrationConfig,
    CandidateMode,
    CandidateSpaceConfig,
    CandidateTrainingSpec,
    CategorySpec,
    MixBitRunConfig,
    ModelProfile,
    ResolvedRunConfig,
    TrainingRecipeConfig,
    resolve_run_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json"
INVENTORY_PATH = REPO_ROOT / ".result/mix_bit/qwen3_8b/model_inventory.json"
SHELL_SCRIPT = REPO_ROOT / "mix_bit/scripts/train_candidate_single.sh"


def _make_temp_executable(tmp_path: Path, content: str = "#!/bin/sh\nexit 0\n") -> Path:
    path = tmp_path / "fake_python"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path.resolve()


class _Block(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(hidden, hidden, bias=False),
                "k_proj": nn.Linear(hidden, hidden, bias=False),
                "v_proj": nn.Linear(hidden, hidden, bias=False),
                "o_proj": nn.Linear(hidden, hidden, bias=False),
            }
        )
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(hidden, hidden * 2, bias=False),
                "up_proj": nn.Linear(hidden, hidden * 2, bias=False),
                "down_proj": nn.Linear(hidden * 2, hidden, bias=False),
            }
        )


class _ToyModel(nn.Module):
    def __init__(self, n_layers: int = 2, hidden: int = 8):
        super().__init__()
        self.model = nn.ModuleDict({"layers": nn.ModuleList([_Block(hidden) for _ in range(n_layers)])})
        self.config = type("Cfg", (), {"model_type": "toy", "_name_or_path": "toy"})()


def _toy_profile(categories: list[CategorySpec] | None = None, *, group_size="all", allow_tail=True) -> ModelProfile:
    if categories is None:
        categories = [
            CategorySpec("q_proj", "q_proj", True),
            CategorySpec("k_proj", "k_proj", False),
            CategorySpec("v_proj", "v_proj", True),
        ]
    return ModelProfile(
        model_id="toy",
        model_path="toy-model",
        adapter="generic_decoder",
        only_decoder_projections=True,
        candidate_training=CandidateTrainingSpec(linear_group_size=group_size, allow_tail_group=allow_tail),
        layer_index_patterns=(r"(?:^|\.)model\.layers\.(\d+)\.",),
        categories=tuple(categories),
        regression_expectations={},
    )


def _toy_modes(n: int = 2) -> tuple[CandidateMode, ...]:
    return tuple(
        CandidateMode(
            name=f"b{16 * (i + 1)}d4s2",
            nominal_bit=float(2 * (16 * (i + 1)) // 4),
            codebook_bits=16 * (i + 1),
            codebook_dim=4,
            residual_stages=2,
        )
        for i in range(n)
    )


def _make_resolved(
    tmp_path: Path,
    *,
    profile: ModelProfile | None = None,
    modes: tuple[CandidateMode, ...] | None = None,
) -> ResolvedRunConfig:
    profile = profile or _toy_profile()
    modes = modes or _toy_modes()
    recipe = TrainingRecipeConfig(
        recipe_id="toy_recipe",
        values={
            "seed": 31,
            "deterministic": True,
            "steps_per_category": 10,
            "batch_size": 8,
            "base_ch": 8,
            "num_res_blocks": 0,
            "decoder_base_ch": 8,
            "decoder_num_res_blocks": 0,
            "norm_type": "layer",
            "activation_type": "swish",
            "decoder_type": "linear",
            "recon_loss_type": "mse",
            "quantizer_type": "BSQ",
            "gamma0": 1.0,
            "gamma": 1.0,
            "zeta": 1.0,
            "inv_temperature": 100.0,
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.95,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "lr_scheduler": "linear",
            "lr_warmup_steps": 0,
            "l1_weight": 1.0,
            "lfq_weight": 1.0,
            "commitment_loss_weight": 0.25,
            "entropy_loss_weight": 0.01,
            "normalize_weight": True,
            "vae_decoder_checkpoint": True,
            "new_quant": True,
            "log_every": 1,
            "eval_every": 0,
            "eval_blocks": 8,
            "outlier_protect_mode": "channel",
            "outlier_protect_count": 0,
            "outlier_protect_min_per_layer": 0,
            "distill_after_category": "none",
            "eval_ppl": False,
            "eval_tasks": "",
            "rot_llm": False,
            "fp16": False,
            "bf16": True,
        },
    )
    config = MixBitRunConfig(
        run_id="toy_run",
        model_profile=profile,
        candidate_space=CandidateSpaceConfig(
            candidate_space_id="toy_space",
            baseline_mode=modes[0].name,
            target_average_bit=2.0,
            modes=modes,
        ),
        training_recipe=recipe,
        calibration=CalibrationConfig(
            source_jsonl=str(tmp_path / "calib.jsonl"),
            input_format="text",
            max_samples=1,
            max_length=8,
            seed=0,
            label_mode="all_nonpad",
        ),
    )
    root = tmp_path / "result" / "mix_bit" / profile.model_id
    return ResolvedRunConfig(
        config=config,
        run_config_path=str(tmp_path / "run.json"),
        run_config_sha256="r" * 64,
        model_profile_path=str(tmp_path / "profile.json"),
        model_profile_sha256="p" * 64,
        candidate_space_path=str(tmp_path / "space.json"),
        candidate_space_sha256="c" * 64,
        training_recipe_path=str(tmp_path / "recipe.json"),
        training_recipe_sha256="t" * 64,
        canonical_model_root=str(root),
        canonical_run_root=str(root / "runs" / config.run_id),
    )


def _inventory_for(profile: ModelProfile, model: nn.Module) -> ModelInventory:
    from mix_bit.model_adapter import get_model_adapter

    adapter = get_model_adapter("generic_decoder")
    targets = adapter.discover_target_linears(model, profile)
    return inventory_from_targets(
        profile=profile,
        model=model,
        targets=targets,
        model_profile_sha256="p" * 64,
    )


def test_trial_command_passes_resolved_sys_executable_as_second_argument(tmp_path: Path):
    fake_python = _make_temp_executable(tmp_path)
    resolved = _make_resolved(tmp_path)
    inventory = _inventory_for(resolved.config.model_profile, _ToyModel())
    trial = generate_candidate_trials(resolved, inventory)[0]
    with mock.patch("mix_bit.candidate_pool.sys.executable", str(fake_python)):
        cmd = build_trial_command(trial, resolved, gpu_id="4")
    assert cmd[0].endswith("train_candidate_single.sh")
    assert cmd[1] == "4"
    assert cmd[2] == str(fake_python)
    assert cmd[3].startswith("--")


def test_scheduler_meta_records_python_executable(tmp_path: Path):
    fake_python = _make_temp_executable(tmp_path)
    profile = _toy_profile()
    resolved = _make_resolved(tmp_path, profile=profile)
    inventory = _inventory_for(profile, _ToyModel())
    inv_path = tmp_path / "inventory.json"
    write_model_inventory(inventory, inv_path)
    with mock.patch("mix_bit.candidate_pool.sys.executable", str(fake_python)), mock.patch(
        "mix_bit.candidate_pool.preflight_loader_inventory", return_value=inventory
    ):
        run_candidate_pool(
            resolved=resolved,
            inventory=inventory,
            inventory_path=str(inv_path),
            gpus=["4"],
            dry_run=True,
            output_root=str(tmp_path / "pool"),
        )
    meta = json.loads((tmp_path / "pool" / "scheduler_meta.json").read_text(encoding="utf-8"))
    assert meta["python_executable"] == str(fake_python)


def test_trial_spec_records_python_executable(tmp_path: Path):
    fake_python = _make_temp_executable(tmp_path)
    resolved = _make_resolved(tmp_path)
    inventory = _inventory_for(resolved.config.model_profile, _ToyModel())
    trial = generate_candidate_trials(resolved, inventory)[0]
    with mock.patch("mix_bit.candidate_pool.sys.executable", str(fake_python)):
        cmd = build_trial_command(trial, resolved, gpu_id="0")
        payload = trial_spec_to_dict(trial, cmd)
    assert payload["python_executable"] == str(fake_python)
    spec_path = write_trial_spec(trial, cmd)
    written = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    assert written["python_executable"] == str(fake_python)


def test_candidate_shell_requires_python_argument():
    result = subprocess.run(
        ["bash", str(SHELL_SCRIPT), "4"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    assert "PYTHON_EXECUTABLE" in result.stderr


def test_candidate_shell_rejects_non_executable_python(tmp_path: Path):
    not_exec = tmp_path / "not_exec.py"
    not_exec.write_text("#!/bin/sh\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(SHELL_SCRIPT), "4", str(not_exec), "--sentinel"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    assert "not executable" in result.stderr.lower()


def test_candidate_shell_executes_explicit_interpreter(tmp_path: Path):
    argv_file = tmp_path / "captured_argv.txt"
    stub = tmp_path / "stub_python"
    stub.write_text(
        f"""#!/bin/sh
printf '%s\\n' "$@" > {argv_file}
""",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SHELL_SCRIPT), "4", str(stub.resolve()), "--flag", "value"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    lines = argv_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "tools/cat_train.py"
    assert lines[1:] == ["--flag", "value"]


def test_trial_spec_records_all_resolved_hashes(tmp_path: Path):
    resolved = _make_resolved(tmp_path)
    inventory = _inventory_for(resolved.config.model_profile, _ToyModel())
    trial = generate_candidate_trials(resolved, inventory)[0]
    cmd = build_trial_command(trial, resolved, gpu_id="0")
    payload = trial_spec_to_dict(trial, cmd)
    assert payload["run_config_sha256"] == resolved.run_config_sha256
    assert payload["candidate_space_sha256"] == resolved.candidate_space_sha256
    assert payload["training_recipe_sha256"] == resolved.training_recipe_sha256
    assert payload["model_profile_sha256"] == resolved.model_profile_sha256
    assert payload["model_inventory_fingerprint"] == inventory.fingerprint_sha256

    spec_path = write_trial_spec(trial, cmd)
    written = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    for key in (
        "run_config_sha256",
        "candidate_space_sha256",
        "training_recipe_sha256",
        "model_profile_sha256",
        "model_inventory_fingerprint",
    ):
        assert written[key] == payload[key]


def test_trial_count_is_profile_categories_times_candidate_modes(tmp_path: Path):
    profile = _toy_profile()
    modes = _toy_modes(3)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    trials = generate_candidate_trials(resolved, inventory)
    assert len(trials) == len(profile.categories) * len(modes)


def test_qwen3_profile_generates_35_regression_trials():
    assert INVENTORY_PATH.is_file(), f"missing inventory fixture: {INVENTORY_PATH}"
    resolved = resolve_run_config(str(RUN_CONFIG), write=False)
    from mix_bit.model_inventory import load_model_inventory

    inventory = load_model_inventory(str(INVENTORY_PATH))
    trials = generate_candidate_trials(resolved, inventory)
    assert len(trials) == 35
    assert len({(t.category_name, t.mode.name) for t in trials}) == 35
    assert all(t.mode.residual_stages == 2 for t in trials)


def test_scheduler_supports_different_category_counts(tmp_path: Path):
    cats_a = [CategorySpec("q_proj", "q_proj", True), CategorySpec("k_proj", "k_proj", False)]
    cats_b = cats_a + [CategorySpec("v_proj", "v_proj", True)]
    modes = _toy_modes(2)
    resolved_a = _make_resolved(tmp_path, profile=_toy_profile(cats_a), modes=modes)
    resolved_b = _make_resolved(tmp_path / "b", profile=_toy_profile(cats_b), modes=modes)
    inv_a = _inventory_for(resolved_a.config.model_profile, _ToyModel())
    inv_b = _inventory_for(resolved_b.config.model_profile, _ToyModel())
    assert len(generate_candidate_trials(resolved_a, inv_a)) == 4
    assert len(generate_candidate_trials(resolved_b, inv_b)) == 6


def test_loader_preflight_inventory_matches_adapter_inventory(tmp_path: Path):
    profile = _toy_profile()
    model = _ToyModel(n_layers=2)
    resolved = _make_resolved(tmp_path, profile=profile)
    inventory = _inventory_for(profile, model)
    with mock.patch("rotation.model_utils.get_model", return_value=model):
        matched = preflight_loader_inventory(resolved, inventory)
    assert matched.fingerprint_sha256 == inventory.fingerprint_sha256


def test_loader_preflight_rejects_unsupported_block_path_before_jobs(tmp_path: Path):
    profile = _toy_profile()
    resolved = _make_resolved(tmp_path, profile=profile)
    model = _ToyModel()
    inventory = _inventory_for(profile, model)

    bad = nn.Module()
    bad.weird = nn.ModuleDict({"blocks": nn.ModuleList([_Block()])})
    # modules named weird.blocks.0.self_attn.q_proj — extract_layer_idx returns None
    bad.weird.blocks[0] = _Block()
    # Build a model whose linear names won't match extract_layer_idx patterns used by cat_train
    class _Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.ModuleDict({"layer": nn.ModuleList([_Block()])})
            self.config = type("Cfg", (), {"model_type": "toy"})()

    bad_model = _Bad()
    # Force collect_linears to find modules by renaming to end with q_proj under unsupported path
    # Use monkeypatch on collect_linears return with unsupported names instead.
    from train_utils.utils import LinearRef

    fake_refs = [
        LinearRef(
            name="encoder.layer.0.self_attn.q_proj",
            module=bad_model.encoder["layer"][0].self_attn["q_proj"],
            category="q_proj",
            transpose=True,
        )
    ]
    with mock.patch("rotation.model_utils.get_model", return_value=bad_model), mock.patch(
        "train_utils.utils.collect_linears", return_value=fake_refs
    ):
        with pytest.raises(ValueError, match="unsupported|block|extract_layer_idx|Unsupported"):
            preflight_loader_inventory(resolved, inventory)


def test_loader_preflight_rejects_missing_category_before_jobs(tmp_path: Path):
    profile = _toy_profile()
    resolved = _make_resolved(tmp_path, profile=profile)
    model = _ToyModel()
    inventory = _inventory_for(profile, model)

    # Drop one category from discovered linears
    from train_utils.utils import collect_linears as real_collect

    def _partial_collect(model, transpose_modules, *, only_decoder_projections, target_categories):
        refs = real_collect(
            model,
            transpose_modules,
            only_decoder_projections=only_decoder_projections,
            target_categories=target_categories,
        )
        return [r for r in refs if r.category != "v_proj"]

    with mock.patch("rotation.model_utils.get_model", return_value=model), mock.patch(
        "train_utils.utils.collect_linears", side_effect=_partial_collect
    ):
        with pytest.raises(ValueError, match="fingerprint|mismatch|category|Missing"):
            preflight_loader_inventory(resolved, inventory)


def test_trial_output_path_is_category_then_mode(tmp_path: Path):
    resolved = _make_resolved(tmp_path)
    inventory = _inventory_for(resolved.config.model_profile, _ToyModel())
    trials = generate_candidate_trials(resolved, inventory)
    for trial in trials:
        root = Path(trial.trial_root)
        assert root.parent.name == trial.mode.name or root.name == trial.mode.name
        assert trial.category_name in root.parts
        assert trial.mode.name in root.parts
        # category then mode
        parts = root.parts
        cat_idx = parts.index(trial.category_name)
        mode_idx = parts.index(trial.mode.name)
        assert cat_idx < mode_idx


def test_command_passes_exact_codebook_fields(tmp_path: Path):
    resolved = _make_resolved(tmp_path)
    inventory = _inventory_for(resolved.config.model_profile, _ToyModel())
    trial = generate_candidate_trials(resolved, inventory)[0]
    cmd = build_trial_command(trial, resolved, gpu_id="4")
    joined = " ".join(cmd)
    assert f"--codebook_bits default={trial.mode.codebook_bits}" in joined
    assert f"--codebook_dim default={trial.mode.codebook_dim}" in joined
    assert f"--residual_stages default={trial.mode.residual_stages}" in joined


def test_command_disables_category_distillation(tmp_path: Path):
    resolved = _make_resolved(tmp_path)
    inventory = _inventory_for(resolved.config.model_profile, _ToyModel())
    trial = generate_candidate_trials(resolved, inventory)[0]
    cmd = build_trial_command(trial, resolved, gpu_id="4")
    assert "--distill_after_category" in cmd
    assert cmd[cmd.index("--distill_after_category") + 1] == "none"


def test_command_disables_inline_ppl_and_task_evaluation(tmp_path: Path):
    resolved = _make_resolved(tmp_path)
    inventory = _inventory_for(resolved.config.model_profile, _ToyModel())
    trial = generate_candidate_trials(resolved, inventory)[0]
    cmd = build_trial_command(trial, resolved, gpu_id="4")
    assert cmd[cmd.index("--eval_ppl") + 1] == "false"
    assert cmd[cmd.index("--eval_tasks") + 1] == ""


def test_command_disables_rotation(tmp_path: Path):
    resolved = _make_resolved(tmp_path)
    inventory = _inventory_for(resolved.config.model_profile, _ToyModel())
    trial = generate_candidate_trials(resolved, inventory)[0]
    cmd = build_trial_command(trial, resolved, gpu_id="4")
    assert "--rot_llm" in cmd
    assert cmd[cmd.index("--rot_llm") + 1] == "false"


def test_command_uses_candidate_only_save_and_omits_save_model(tmp_path: Path):
    resolved = _make_resolved(tmp_path)
    inventory = _inventory_for(resolved.config.model_profile, _ToyModel())
    trial = generate_candidate_trials(resolved, inventory)[0]
    cmd = build_trial_command(trial, resolved, gpu_id="4")
    assert "--save_candidate_artifact" in cmd
    assert "--save_model" not in cmd
    assert "--candidate_artifact_spec" in cmd
    assert "--candidate_artifact_output_dir" in cmd


def test_candidate_only_save_rejects_save_model_combination():
    from train_utils.cat_train_args import process_cat_train_args

    with pytest.raises(ValueError, match="mutually exclusive|save_candidate_artifact|save_model"):
        process_cat_train_args(
            [
                "--convert",
                "--save_candidate_artifact",
                "--save_model",
                "--candidate_artifact_spec",
                "/tmp/spec.json",
                "--candidate_artifact_output_dir",
                "/tmp/artifact",
                "--target_categories",
                "q_proj",
                "--model_path",
                "toy",
            ]
        )


def test_trial_resolves_exactly_one_new_cat_train_run_dir(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    before = {p.name for p in runs.iterdir()}
    (runs / "orphan_old").mkdir()
    before_with_orphan = {p.name for p in runs.iterdir()}
    new_dir = runs / "model_20260101_000000"
    new_dir.mkdir()
    resolved = resolve_new_cat_train_run_dir(runs, before_snapshot=before_with_orphan - {new_dir.name})
    # snapshot before launch should be before_with_orphan without new; simulate properly:
    snapshot = before_with_orphan - {new_dir.name}
    assert resolve_new_cat_train_run_dir(runs, before_snapshot=snapshot) == new_dir
    with pytest.raises(ValueError, match="exactly one|0|multiple"):
        resolve_new_cat_train_run_dir(runs, before_snapshot=set())


def test_completed_trial_requires_compact_artifact_and_hashes(tmp_path: Path):
    resolved = _make_resolved(tmp_path)
    inventory = _inventory_for(resolved.config.model_profile, _ToyModel())
    trial = generate_candidate_trials(resolved, inventory)[0]
    Path(trial.trial_root).mkdir(parents=True)
    artifact = Path(trial.trial_root) / "artifact"
    artifact.mkdir()
    assert is_trial_complete(trial) is False
    state_path = artifact / "module_state.pt"
    meta_path = artifact / "candidate_meta.json"
    state_path.write_bytes(b"x")
    meta_path.write_text("{}\n", encoding="utf-8")
    assert is_trial_complete(trial) is False
    # completed.json with mismatched hashes still incomplete
    (artifact / "completed.json").write_text(
        json.dumps({"module_state_sha256": "0" * 64, "candidate_meta_sha256": "1" * 64}),
        encoding="utf-8",
    )
    assert is_trial_complete(trial) is False
    # matching hashes alone no longer mark the trial complete: the full contract
    # (mode, hashes, module specs, expected names) must also hold. With an empty
    # meta the trial stays incomplete under the strict resume validator.
    state_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    meta_hash = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    (artifact / "completed.json").write_text(
        json.dumps(
            {
                "module_state_sha256": state_hash,
                "candidate_meta_sha256": meta_hash,
            }
        ),
        encoding="utf-8",
    )
    assert is_trial_complete(trial) is False


def test_orphan_timestamp_run_is_not_silently_selected(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    orphan = runs / "Qwen_Qwen3-8B_20260101_000000"
    orphan.mkdir()
    with pytest.raises(ValueError, match="exactly one|orphan|0 new|no new"):
        resolve_new_cat_train_run_dir(runs, before_snapshot={orphan.name})


def test_dry_run_does_not_create_fake_completion(tmp_path: Path):
    profile = _toy_profile()
    resolved = _make_resolved(tmp_path, profile=profile)
    inventory = _inventory_for(profile, _ToyModel())
    inv_path = tmp_path / "inventory.json"
    write_model_inventory(inventory, inv_path)

    with mock.patch("mix_bit.candidate_pool.preflight_loader_inventory", return_value=inventory), mock.patch(
        "subprocess.Popen"
    ) as popen:
        code = run_candidate_pool(
            resolved=resolved,
            inventory=inventory,
            inventory_path=str(inv_path),
            gpus=["4", "5"],
            dry_run=True,
            output_root=str(tmp_path / "pool"),
        )
    assert code == 0
    popen.assert_not_called()
    pool = tmp_path / "pool"
    completed = list(pool.rglob("completed.json"))
    assert completed == []


# ---------------------------------------------------------------------------
# Resume contract: validate_trial_completion / is_trial_complete
# ---------------------------------------------------------------------------

from mix_bit.candidate_artifact import (  # noqa: E402
    CANDIDATE_META_FILENAME,
    COMPLETED_FILENAME,
    MODULE_STATE_FILENAME,
)
from mix_bit.schema import sha256_file  # noqa: E402


def _packed_vq_bits(bits: int) -> dict[str, object]:
    return {
        "storage_format": "bitpack_u8",
        "dtype": "uint8",
        "logical_dtype": "bool",
        "pack_bits": 8,
        "logical_shape": [8, 1, bits],
        "shape": [8, 1, (bits + 7) // 8],
    }


def _decoder_bits(bits: int, dim: int) -> dict[str, object]:
    return {
        "in_dim": bits,
        "out_dim": dim,
        "hidden_dim": 8,
        "num_res_blocks": 0,
        "norm_type": "layer",
        "activation_type": "swish",
        "decoder_type": "linear",
        "use_checkpoint": False,
        "param_dtype": "float32",
    }


def _contract_module_spec(name: str, mode: CandidateMode) -> dict[str, object]:
    bits = int(mode.codebook_bits)
    dim = int(mode.codebook_dim)
    stages = int(mode.residual_stages)
    return {
        "name": name,
        "in_features": 8,
        "out_features": 8,
        "has_bias": False,
        "transpose": True,
        "has_original_weight": False,
        "codebook_dim": dim,
        "residual_stages": stages,
        "stage_codebook_dims": [dim] * stages,
        "parallel_parts": 1,
        "stage_vq_weights": [_packed_vq_bits(bits) for _ in range(stages)],
        "stage_decoders": [_decoder_bits(bits, dim) for _ in range(stages)],
        "vq_weights": [_packed_vq_bits(bits)],
        "decoders": [_decoder_bits(bits, dim)],
        "protected_input_indices": None,
        "protected_input_weight": None,
        "protected_output_indices": None,
        "protected_output_weight": None,
        "low_rank_a": None,
        "low_rank_b": None,
        "protected_residual_indices": None,
        "protected_residual_stages": 0,
        "sparse_residual_indices": None,
        "sparse_residual_values": None,
    }


def _write_full_artifact(
    trial: "TrialSpec",
    *,
    mutate_meta: dict[str, object] | None = None,
    mutate_completed: dict[str, object] | None = None,
    module_specs: list[dict[str, object]] | None = None,
    expected_names: list[str] | None = None,
) -> None:
    from mix_bit.candidate_pool import TrialSpec  # noqa: F401

    artifact = Path(trial.trial_root) / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    names = expected_names if expected_names is not None else list(trial.expected_module_names)
    specs = module_specs if module_specs is not None else [
        _contract_module_spec(name, trial.mode) for name in names
    ]
    state_bytes = ("STATE:" + "|".join(names)).encode("utf-8")
    state_path = artifact / MODULE_STATE_FILENAME
    state_path.write_bytes(state_bytes)
    state_sha = sha256_file(state_path)
    payload_summaries = {
        f"{name}.vq_weight": {"dtype": "uint8", "shape": [1], "nbytes": 1} for name in names
    }
    meta = {
        "format": "vaellm_candidate_modules_v1",
        "module_specs": specs,
        "expected_module_names": names,
        "payload_summaries": payload_summaries,
        "run_config_sha256": trial.run_config_sha256,
        "candidate_space_sha256": trial.candidate_space_sha256,
        "training_recipe_sha256": trial.training_recipe_sha256,
        "model_profile_sha256": trial.model_profile_sha256,
        "model_inventory_fingerprint": trial.model_inventory_fingerprint,
        "mode": {
            "name": trial.mode.name,
            "nominal_bit": trial.mode.nominal_bit,
            "codebook_bits": trial.mode.codebook_bits,
            "codebook_dim": trial.mode.codebook_dim,
            "residual_stages": trial.mode.residual_stages,
        },
        "category_name": trial.category_name,
        "source_run_dir": str(Path(trial.trial_root) / "runs" / "fake"),
        "module_state_file": MODULE_STATE_FILENAME,
        "module_state_sha256": state_sha,
    }
    if mutate_meta:
        meta.update(mutate_meta)
    meta_path = artifact / CANDIDATE_META_FILENAME
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    meta_sha = sha256_file(meta_path)
    completed = {
        "format": "vaellm_candidate_modules_v1",
        "module_state_sha256": state_sha,
        "candidate_meta_sha256": meta_sha,
        "module_count": len(names),
    }
    if mutate_completed:
        completed.update(mutate_completed)
    (artifact / COMPLETED_FILENAME).write_text(
        json.dumps(completed, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _first_trial(tmp_path: Path) -> tuple["TrialSpec", "ResolvedRunConfig", "ModelInventory"]:
    profile = _toy_profile()
    resolved = _make_resolved(tmp_path, profile=profile)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    trial = generate_candidate_trials(resolved, inventory)[0]
    Path(trial.trial_root).mkdir(parents=True, exist_ok=True)
    return trial, resolved, inventory


def test_resume_accepts_exact_valid_artifact(tmp_path: Path):
    trial, _resolved, _inv = _first_trial(tmp_path)
    _write_full_artifact(trial)
    assert is_trial_complete(trial) is True
    validate_trial_completion(trial)  # must not raise


def test_resume_retrains_when_mode_metadata_differs(tmp_path: Path):
    trial, _resolved, _inv = _first_trial(tmp_path)
    bad_mode = {
        "name": trial.mode.name,
        "nominal_bit": float(trial.mode.nominal_bit) + 1.0,
        "codebook_bits": trial.mode.codebook_bits,
        "codebook_dim": trial.mode.codebook_dim,
        "residual_stages": trial.mode.residual_stages,
    }
    _write_full_artifact(trial, mutate_meta={"mode": bad_mode})
    assert is_trial_complete(trial) is False
    with pytest.raises(ValueError, match="nominal_bit"):
        validate_trial_completion(trial)


def test_resume_retrains_when_inventory_fingerprint_differs(tmp_path: Path):
    trial, _resolved, _inv = _first_trial(tmp_path)
    _write_full_artifact(
        trial,
        mutate_meta={"model_inventory_fingerprint": "0" * 64},
    )
    assert is_trial_complete(trial) is False
    with pytest.raises(ValueError, match="model_inventory_fingerprint"):
        validate_trial_completion(trial)


def test_resume_retrains_when_expected_module_order_differs(tmp_path: Path):
    trial, _resolved, _inv = _first_trial(tmp_path)
    names = list(trial.expected_module_names)
    if len(names) < 2:
        pytest.skip("need >=2 expected modules to reorder")
    reordered = [names[1], names[0]] + list(names[2:])
    _write_full_artifact(trial, expected_names=reordered)
    assert is_trial_complete(trial) is False
    with pytest.raises(ValueError, match="expected_module_names|order"):
        validate_trial_completion(trial)


def test_resume_retrains_when_module_spec_mode_contract_fails(tmp_path: Path):
    trial, _resolved, _inv = _first_trial(tmp_path)
    names = list(trial.expected_module_names)
    # Build specs that violate the mode contract: residual_stages=1 while mode claims 2.
    bad_specs = []
    for name in names:
        spec = _contract_module_spec(name, trial.mode)
        spec["residual_stages"] = 1
        spec["stage_codebook_dims"] = [int(trial.mode.codebook_dim)]
        spec["stage_vq_weights"] = None
        spec["stage_decoders"] = None
        bad_specs.append(spec)
    _write_full_artifact(trial, module_specs=bad_specs)
    assert is_trial_complete(trial) is False
    with pytest.raises(ValueError, match="residual_stages"):
        validate_trial_completion(trial)
