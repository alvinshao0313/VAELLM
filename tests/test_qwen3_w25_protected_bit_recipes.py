from copy import deepcopy

import pytest

from experiments.qwen3_w25.local_recovery import TARGETS
from experiments.qwen3_w25.protected_bit_recovery import (
    FOCUSED_TARGETS, MLP_RECIPES, load_focused_recipe_selection,
    projection_recipe, recovery_configuration,
)


def _configuration_identity(tmp_path, recipe):
    # Configuration-shape fixtures only; these are never loaded as reports or
    # presented as optimization evidence. Actual summary acceptance reads both
    # completed reports through load_focused_summary.
    return {
        "summary_path": str(tmp_path / "summary.json"), "summary_sha256": "1" * 64,
        "focused_reports": {target: {"path": str(tmp_path / target / "pilot.json"), "sha256": "2" * 64}
                            for target in FOCUSED_TARGETS},
        "selected_attention_recipe": {"lr_divisor": 128, "steps": 128, "round_steps": 128},
        "selected_mlp_recipe": dict(zip(("lr_divisor", "steps", "round_steps"), recipe)),
    }


@pytest.mark.parametrize("recipe", MLP_RECIPES)
def test_one_recipe_per_family_covers_all_252_projections(tmp_path, recipe):
    identity = _configuration_identity(tmp_path, recipe)
    cfg = recovery_configuration(lr_divisor=128, num_shards=4, focused_selection=identity)
    assert cfg["recipe_mode"] == "attention_mlp"
    counts = {"attention": 0, "mlp": 0}
    for target in TARGETS:
        group, actual = projection_recipe(cfg, target)
        counts[group] += 1
        expected = identity["selected_attention_recipe" if group == "attention" else "selected_mlp_recipe"]
        assert actual == expected
        actual["steps"] = -1
        assert cfg["recipes"][group] == expected
    assert counts == {"attention": 144, "mlp": 108}
    assert set(cfg["focused_selection"]["focused_reports"]) == FOCUSED_TARGETS
    assert not cfg["repeated_diagnostic_used_for_selection"]


def test_focused_recipe_requires_both_source_reports_and_predeclared_budget(tmp_path):
    original = _configuration_identity(tmp_path, (256, 128, 256))
    for key in ("summary_sha256", "focused_reports", "selected_attention_recipe"):
        broken = deepcopy(original)
        broken.pop(key)
        with pytest.raises(ValueError, match="report identities"):
            recovery_configuration(lr_divisor=128, num_shards=4, focused_selection=broken)
    broken = deepcopy(original)
    broken["focused_reports"].pop(next(iter(FOCUSED_TARGETS)))
    with pytest.raises(ValueError, match="report identities"):
        recovery_configuration(lr_divisor=128, num_shards=4, focused_selection=broken)
    for recipe in ((256, 128, 128), (128, 256, 256), (256, 192, 256)):
        with pytest.raises(ValueError, match="three unified MLP recipes"):
            recovery_configuration(lr_divisor=128, num_shards=4,
                                   focused_selection=_configuration_identity(tmp_path, recipe))
    with pytest.raises(ValueError, match="attention s128"):
        recovery_configuration(lr_divisor=64, num_shards=4, focused_selection=original)
    broken = deepcopy(original)
    broken["selected_attention_recipe"]["steps"] = 256
    with pytest.raises(ValueError, match="attention s128"):
        recovery_configuration(lr_divisor=128, num_shards=4, focused_selection=broken)


def test_omitting_focused_summary_preserves_uniform_dispatch_without_reading_a_file():
    assert load_focused_recipe_selection(None, pilot_json="not_read", source_dir="not_read") is None
    for divisor in (64, 128):
        cfg = recovery_configuration(lr_divisor=divisor, num_shards=3)
        assert cfg["recipe_mode"] == "uniform" and cfg["focused_selection"] is None
        assert all(projection_recipe(cfg, target)[1] == {"lr_divisor": divisor, "steps": 128, "round_steps": 128}
                   for target in TARGETS)
    with pytest.raises(ValueError, match="252 native"):
        projection_recipe(cfg, "model.layers.36.mlp.down_proj")
