"""CPU-only checks for the new process handoff and command override path."""

import subprocess
import sys

import pytest

from experiments.e2e_0920_search.search_artifacts import atomic_json
from experiments.e2e_0920_search.search_workflow import (
    checkpoint_available, predecessor_start, preview, replace_flags,
    wait_for_predecessor,
)
from experiments.e2e_0920_search import run_search as pilot


def test_stage_command_keeps_final_schedule_and_removes_pause_and_old_overrides():
    command = pilot.command_for("candidate", "1e-4")
    command += ["--learning_rate", "3e-4"]
    result = replace_flags(command, {
        "learning_rate": 3e-5, "steps": 5000, "save_steps": 500,
        "save_total_limit": 12, "stop_after_step": None,
        "resume_from_checkpoint": "/run/trainer_state/checkpoint-2500",
    })
    assert result.count("--learning_rate") == 1
    assert result[result.index("--learning_rate") + 1] == "3e-05"
    assert result[result.index("--steps") + 1] == "5000"
    assert result[result.index("--save_steps") + 1] == "500"
    assert result[result.index("--save_total_limit") + 1] == "12"
    assert "--stop_after_step" not in result
    assert result[result.index("--resume_from_checkpoint") + 1].endswith("checkpoint-2500")
    assert result[result.index("--lora_rank") + 1] == "8"
    assert "--eval_limit" not in result


def test_process_handoff_waits_for_real_exit_and_requires_success_manifest(tmp_path):
    manifest = tmp_path / "pilot.json"
    atomic_json(manifest, {"status": "screen_complete", "trials": [
        {"status": "paused", "exit_code": 0} for _ in range(3)
    ]})
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.1)"]) as child:
        start = predecessor_start(child.pid)
        assert start is not None
        with pytest.raises(RuntimeError, match="reused"):
            wait_for_predecessor(child.pid, "wrong_start", manifest)
        assert wait_for_predecessor(child.pid, start, manifest)["status"] == "screen_complete"
        assert child.poll() == 0
    atomic_json(manifest, {"status": "failed"})
    with pytest.raises(RuntimeError, match="did not complete"):
        wait_for_predecessor(child.pid, start, manifest)


def test_empty_checkpoint_directory_is_not_a_deliverable(tmp_path):
    point = {"checkpoint": str(tmp_path)}
    assert not checkpoint_available(point)
    for name in ("training_model_state.pt", "checkpoint_meta.json", "trainer_state.json"):
        (tmp_path / name).touch()
    assert checkpoint_available(point)


def test_authorized_plan_has_fixed_budget_and_only_requested_axes():
    plan = preview("loss_weights", 7)
    assert plan["rank"] == 8
    assert plan["base_additional_steps"] == 57000
    assert plan["challenger_schedule"] == [2500, 5000]
    assert [item["key"] for item in plan["coordinates"]] == [
        "lm_head_lr", "norm_lr", "lora_dropout", "hidden_loss_weight",
        "pre_mlp_hidden_loss_weight", "alpha",
    ]
