from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mix_bit.candidate_pool import candidate_pool_root
from mix_bit.checkpoint_pool import (
    build_candidate_pool_index,
    build_candidate_pool_index_from_manifest,
)
from mix_bit.model_inventory import ModelInventory
from mix_bit.schema import ResolvedRunConfig, sha256_file

# Reuse the toy fixture helpers from test_checkpoint_pool to build a real,
# contract-consistent compact candidate pool under a custom root.
from mix_bit.tests.test_checkpoint_pool import (
    _ToyModel,
    _inventory_for,
    _make_resolved,
    _populate_valid_pool,
    _toy_modes,
    _toy_profile,
)


def _build_custom_root_world(tmp_path: Path) -> tuple[ResolvedRunConfig, ModelInventory, Path]:
    profile = _toy_profile()
    modes = _toy_modes(2)
    resolved = _make_resolved(tmp_path, profile=profile, modes=modes)
    inventory = _inventory_for(profile, _ToyModel(n_layers=2))
    custom_root = tmp_path / "custom_pool"
    assert custom_root != candidate_pool_root(resolved)
    _populate_valid_pool(resolved, inventory, output_root=custom_root)
    return resolved, inventory, custom_root


def _manifest_path_for(custom_root: Path) -> Path:
    return custom_root / "candidate_manifest.json"


# ---------------------------------------------------------------------------
# Step 2: helper loads custom root
# ---------------------------------------------------------------------------


def test_index_from_manifest_uses_manifest_parent(tmp_path: Path):
    resolved, inventory, custom_root = _build_custom_root_world(tmp_path)
    manifest_path = _manifest_path_for(custom_root)

    index_canonical = build_candidate_pool_index(
        resolved, inventory, output_root=str(custom_root), write_manifest=True
    )
    assert Path(index_canonical.manifest_path).resolve() == manifest_path.resolve()
    assert manifest_path.is_file()

    supplied_sha = sha256_file(manifest_path)
    before_bytes = manifest_path.read_bytes()

    index = build_candidate_pool_index_from_manifest(resolved, inventory, manifest_path)
    assert Path(index.manifest_path).resolve() == manifest_path.resolve()
    assert manifest_path.read_bytes() == before_bytes
    assert sha256_file(manifest_path) == supplied_sha
    assert index.model_id == inventory.model_id
    assert index.run_id == resolved.config.run_id


def test_index_from_manifest_rejects_wrong_filename(tmp_path: Path):
    resolved, inventory, custom_root = _build_custom_root_world(tmp_path)
    build_candidate_pool_index(
        resolved, inventory, output_root=str(custom_root), write_manifest=True
    )
    wrong = custom_root / "not_candidate_manifest.json"
    wrong.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_manifest.json"):
        build_candidate_pool_index_from_manifest(resolved, inventory, wrong)


def test_index_from_manifest_rejects_missing_file(tmp_path: Path):
    resolved, inventory, custom_root = _build_custom_root_world(tmp_path)
    missing = custom_root / "candidate_manifest.json"
    assert not missing.is_file()
    with pytest.raises(FileNotFoundError):
        build_candidate_pool_index_from_manifest(resolved, inventory, missing)


def test_index_from_manifest_does_not_modify_manifest_bytes(tmp_path: Path):
    resolved, inventory, custom_root = _build_custom_root_world(tmp_path)
    manifest_path = _manifest_path_for(custom_root)
    build_candidate_pool_index(
        resolved, inventory, output_root=str(custom_root), write_manifest=True
    )
    before_bytes = manifest_path.read_bytes()
    before_sha = sha256_file(manifest_path)
    build_candidate_pool_index_from_manifest(resolved, inventory, manifest_path)
    assert manifest_path.read_bytes() == before_bytes
    assert sha256_file(manifest_path) == before_sha


def test_index_from_manifest_rejects_stale_payload_without_rewrite(tmp_path: Path):
    resolved, inventory, custom_root = _build_custom_root_world(tmp_path)
    manifest_path = _manifest_path_for(custom_root)
    build_candidate_pool_index(
        resolved, inventory, output_root=str(custom_root), write_manifest=True
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Stale: corrupt one artifact sha while keeping valid JSON and kind.
    payload["artifacts"][0]["candidate_meta_sha256"] = "0" * 64
    stale_bytes = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    manifest_path.write_bytes(stale_bytes)

    with pytest.raises(ValueError, match="manifest payload|payload|stale|mismatch"):
        build_candidate_pool_index_from_manifest(resolved, inventory, manifest_path)
    # File must not be rewritten by the helper.
    assert manifest_path.read_bytes() == stale_bytes


def test_build_index_write_manifest_false_creates_no_manifest(tmp_path: Path):
    resolved, inventory, custom_root = _build_custom_root_world(tmp_path)
    manifest_path = _manifest_path_for(custom_root)
    # Remove any pre-existing manifest.
    if manifest_path.is_file():
        manifest_path.unlink()
    index = build_candidate_pool_index(
        resolved, inventory, output_root=str(custom_root), write_manifest=False
    )
    assert not manifest_path.is_file()
    # manifest_path field still points to the would-be location.
    assert Path(index.manifest_path).resolve() == manifest_path.resolve()


# ---------------------------------------------------------------------------
# Step 3: every CLI uses the authoritative helper
# ---------------------------------------------------------------------------


def _patch_pool_manifest_cli(
    monkeypatch: pytest.MonkeyPatch,
    cli_module,
    *,
    helper_name: str,
    post_helper_name: str,
    post_helper_return,
) -> tuple[dict, dict]:
    """Patch a CLI module so main(argv) runs without real files.

    Returns (from_manifest_calls, canonical_calls). Asserts canonical is unused.
    """
    supplied = []

    def _fake_from_manifest(resolved, inventory, manifest_path):
        supplied.append(str(manifest_path))
        return SimpleNamespace(manifest_path=str(manifest_path))

    canonical = []

    def _fake_canonical(*args, **kwargs):
        canonical.append(True)
        raise AssertionError("canonical build_candidate_pool_index must not be called")

    monkeypatch.setattr(cli_module, "build_candidate_pool_index_from_manifest", _fake_from_manifest)
    # CLIs that always use the supplied manifest may not import the canonical
    # builder at all; patch it with raising=False so the assertion "canonical
    # is not called" still holds (it simply cannot be invoked).
    monkeypatch.setattr(cli_module, "build_candidate_pool_index", _fake_canonical, raising=False)
    monkeypatch.setattr(cli_module, "resolve_run_config", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(cli_module, "load_model_inventory", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(cli_module, post_helper_name, lambda *a, **k: post_helper_return)
    return {"supplied": supplied, "canonical": canonical}


def test_prepare_baseline_cli_uses_supplied_pool_manifest(tmp_path, monkeypatch):
    import mix_bit.cli.prepare_uniform_baseline as cli

    supplied_manifest = str(tmp_path / "custom_pool" / "candidate_manifest.json")
    tracker = _patch_pool_manifest_cli(
        monkeypatch,
        cli,
        helper_name="build_candidate_pool_index_from_manifest",
        post_helper_name="prepare_uniform_baseline_overlay",
        post_helper_return={
            "mode": "b16d4s2",
            "assignment_count": 1,
            "baseline_dir": "x",
            "baseline_overlay": str(tmp_path / "overlay.json"),
            "assembly_audit": {},
        },
    )
    monkeypatch.setattr(cli, "sha256_file", lambda p: "deadbeef")
    rc = cli.main(
        [
            "--run_config", "run.json",
            "--inventory", "inv.json",
            "--pool_manifest", supplied_manifest,
            "--skip_audit",
        ]
    )
    assert rc == 0
    assert tracker["supplied"] == [supplied_manifest]
    assert tracker["canonical"] == []


def test_cost_cli_uses_supplied_pool_manifest(tmp_path, monkeypatch):
    import mix_bit.cli.compute_cost_table as cli

    supplied_manifest = str(tmp_path / "custom_pool" / "candidate_manifest.json")
    tracker = _patch_pool_manifest_cli(
        monkeypatch,
        cli,
        helper_name="build_candidate_pool_index_from_manifest",
        post_helper_name="compute_cost_table",
        post_helper_return=SimpleNamespace(
            dry_run=True,
            finalized=False,
            source_job_count=0,
            non_baseline_module_evaluation_count=0,
            complete_row_count=0,
            pending_job_count=0,
            cost_run_root=str(tmp_path),
            meta_path=None,
        ),
    )
    monkeypatch.setattr(cli, "parse_gpu_list", lambda s: ["0"])
    monkeypatch.setattr(cli, "sha256_file", lambda p: "deadbeef")
    Path(tmp_path / "overlay.json").write_text("{}", encoding="utf-8")
    rc = cli.main(
        [
            "--run_config", "run.json",
            "--inventory", "inv.json",
            "--pool_manifest", supplied_manifest,
            "--baseline_overlay", str(tmp_path / "overlay.json"),
            "--dataset", str(tmp_path / "ds.pt"),
            "--dataset_manifest", str(tmp_path / "ds_manifest.json"),
            "--kl_mode", "exact_full_vocab",
            "--gpus", "0",
            "--dry_run",
        ]
    )
    assert rc == 0
    assert tracker["supplied"] == [supplied_manifest]
    assert tracker["canonical"] == []


def test_solve_cli_uses_supplied_pool_manifest(tmp_path, monkeypatch):
    import mix_bit.cli.solve_allocation as cli

    supplied_manifest = str(tmp_path / "custom_pool" / "candidate_manifest.json")
    tracker = _patch_pool_manifest_cli(
        monkeypatch,
        cli,
        helper_name="build_candidate_pool_index_from_manifest",
        post_helper_name="solve_mixed_bit_allocation",
        post_helper_return=SimpleNamespace(
            is_globally_optimal=True,
            allow_suboptimal=False,
            objective_delta_kl=0.0,
            achieved_average_bit=2.0,
            used_bit_units=0,
            budget_bit_units=0,
        ),
    )
    monkeypatch.setattr(cli, "validate_inventory_for_run", lambda *a, **k: None)
    monkeypatch.setattr(cli, "load_cost_table_for_solve", lambda *a, **k: [])
    monkeypatch.setattr(cli, "write_allocation_outputs", lambda *a, **k: {"json": "x"})
    monkeypatch.setattr(cli, "sha256_file", lambda p: "deadbeef")
    monkeypatch.setattr(cli, "derive_allocation_dir", lambda p: tmp_path)
    # Richer fakes: solve reads hash fields from resolved/inventory.
    monkeypatch.setattr(
        cli,
        "resolve_run_config",
        lambda *a, **k: SimpleNamespace(
            run_config_sha256="r" * 64,
            candidate_space_sha256="c" * 64,
            config=SimpleNamespace(
                run_id="toy_run",
                candidate_space=SimpleNamespace(target_average_bit=2.0),
            ),
        ),
    )
    monkeypatch.setattr(
        cli,
        "load_model_inventory",
        lambda *a, **k: SimpleNamespace(
            fingerprint_sha256="f" * 64,
            model_id="toy",
        ),
    )
    # Provide a real-ish meta file content via open() override.
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"kl_mode": "exact_full_vocab", "metric_name": "m", "teacher_topk": None}), encoding="utf-8")
    Path(tmp_path / "cost.jsonl").write_text("", encoding="utf-8")
    rc = cli.main(
        [
            "--run_config", "run.json",
            "--inventory", "inv.json",
            "--cost_table", str(tmp_path / "cost.jsonl"),
            "--cost_table_meta", str(meta_path),
            "--pool_manifest", supplied_manifest,
        ]
    )
    assert rc == 0
    assert tracker["supplied"] == [supplied_manifest]
    assert tracker["canonical"] == []


def test_assemble_cli_uses_supplied_pool_manifest(tmp_path, monkeypatch):
    import mix_bit.cli.assemble_mixed_model as cli

    supplied_manifest = str(tmp_path / "custom_pool" / "candidate_manifest.json")
    tracker = _patch_pool_manifest_cli(
        monkeypatch,
        cli,
        helper_name="build_candidate_pool_index_from_manifest",
        post_helper_name="assemble_optimal_mixed_checkpoint",
        post_helper_return={
            "output_dir": str(tmp_path),
            "allocation_sha256": "deadbeef",
            "assignment_count": 1,
            "skipped_identical": False,
            "state_dict": "x",
            "meta": {},
        },
    )
    monkeypatch.setattr(cli, "validate_inventory_for_run", lambda *a, **k: None)
    monkeypatch.setattr(cli, "derive_mixed_model_dir", lambda p: str(tmp_path))
    monkeypatch.setattr(cli, "sha256_file", lambda p: "deadbeef")
    Path(tmp_path / "alloc.json").write_text("{}", encoding="utf-8")
    rc = cli.main(
        [
            "--run_config", "run.json",
            "--inventory", "inv.json",
            "--pool_manifest", supplied_manifest,
            "--allocation", str(tmp_path / "alloc.json"),
        ]
    )
    assert rc == 0
    assert tracker["supplied"] == [supplied_manifest]
    assert tracker["canonical"] == []


def test_validate_cli_uses_supplied_pool_manifest(tmp_path, monkeypatch):
    import mix_bit.cli.validate_mixed_model as cli

    supplied_manifest = str(tmp_path / "custom_pool" / "candidate_manifest.json")
    tracker = _patch_pool_manifest_cli(
        monkeypatch,
        cli,
        helper_name="build_candidate_pool_index_from_manifest",
        post_helper_name="validate_mixed_model",
        post_helper_return={
            "passed": True,
            "validation_json": "x",
            "validation_md": "y",
            "allocation_sha256": "deadbeef",
            "budget": {"used_bit_units": 0, "budget_bit_units": 0},
            "kl": {
                "predicted_mixed_model_kl": 0.0,
                "actual_mixed_model_kl": 0.0,
                "absolute_gap": 0.0,
                "relative_gap": 0.0,
            },
        },
    )
    monkeypatch.setattr(cli, "validate_inventory_for_run", lambda *a, **k: None)
    monkeypatch.setattr(cli, "sha256_file", lambda p: "deadbeef")
    rc = cli.main(
        [
            "--run_config", "run.json",
            "--inventory", "inv.json",
            "--pool_manifest", supplied_manifest,
            "--cost_table", str(tmp_path / "cost.jsonl"),
            "--cost_table_meta", str(tmp_path / "meta.json"),
            "--allocation", str(tmp_path / "alloc.json"),
            "--baseline_overlay", str(tmp_path / "overlay.json"),
            "--mixed_model_dir", str(tmp_path),
            "--dataset", str(tmp_path / "ds.pt"),
            "--dataset_manifest", str(tmp_path / "ds_manifest.json"),
            "--skip_downstream_eval",
        ]
    )
    assert rc == 0
    assert tracker["supplied"] == [supplied_manifest]
    assert tracker["canonical"] == []
