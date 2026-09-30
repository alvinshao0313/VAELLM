from __future__ import annotations

import csv

import pytest

from experiments.e2e_0920_search.search_artifacts import (
    CHECKPOINT_ID, TASK_KEYS, atomic_json, collect_candidate, delete_checkpoints,
    read_json, read_metrics, write_leaderboard,
)


def _metrics(path, score=0.7):
    value = {"tasks": list(TASK_KEYS), "task_metrics": dict.fromkeys(TASK_KEYS, score),
             "task_metric_keys": dict(TASK_KEYS)}
    atomic_json(path, value)
    return value


def _run(run, step):
    cfg = {"train_mode": "lora", "lora": {"rank": 8}, "opt": {"steps": 5000},
           "aux": {"residual_lora_mode": "none", "lm_head_train_mode": "linear"},
           "runtime": {"parallel_mode": "dp", "evaluation": {
               "eval_limit": None, "eval_num_fewshot": 0, "eval_hif4_act": False,
               "eval_after_save": True, "eval_tasks": ",".join(TASK_KEYS)}}}
    atomic_json(run / "normalized_e2e_runtime_args.json",
                {"canonical_config": cfg, "training_args": {"max_steps": 5000}})
    checkpoint = run / "trainer_state" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    atomic_json(checkpoint / "trainer_state.json", {"global_step": step, "max_steps": 5000})
    atomic_json(checkpoint / "checkpoint_meta.json", {
        "checkpoint_kind": "training_step", "round_base_checkpoint_id": CHECKPOINT_ID})
    (checkpoint / "training_model_state.pt").write_bytes(b"fixture")
    if step < 5000:
        for name in ("optimizer.pt", "scheduler.pt",
                     *(f"rng_state_{rank}.pth" for rank in range(4))):
            (checkpoint / name).write_bytes(b"fixture")
        (run / "compressed_e2e_fintuning.log").write_text(
            f"E2E status=paused global_step={step} max_steps=5000; finalization skipped.\n")
        _metrics(run / "lm_eval" / f"lm_eval_results_step_{step}.json")
    else:
        atomic_json(run / "run_meta.json", {"global_step": 5000, "round_base_checkpoint_id": CHECKPOINT_ID,
                                           "final_checkpoint_id": "final-id"})
        atomic_json(run / "final_model" / "checkpoint_meta.json", {
            "checkpoint_kind": "final_model", "checkpoint_id": "final-id", "train_mode": "lora",
            "lm_head_train_mode": "linear", "state_dict_file": "pytorch_model.bin"})
        (run / "final_model" / "pytorch_model.bin").write_bytes(b"fixture")
        _metrics(run / "lm_eval" / "lm_eval_results_final.json", .69)
    return cfg


def test_metrics_reject_changed_task_protocol_and_invalid_scores(tmp_path):
    path = tmp_path / "metrics.json"
    value = _metrics(path)
    assert read_metrics(path, 1000)["mean_percent"] == pytest.approx(70)
    value["task_metric_keys"]["arc_easy"] = "acc,none"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="metric keys"):
        read_metrics(path, 1000)
    value = _metrics(path)
    value["task_metrics"]["boolq"] = 1.1
    atomic_json(path, value)
    with pytest.raises(ValueError, match="finite"):
        read_metrics(path, 1000)


def test_collect_requires_paused_state_and_marks_rotated_checkpoint_unavailable(tmp_path):
    run = tmp_path / "run"
    _run(run, 1000)
    _metrics(run / "lm_eval" / "lm_eval_results_step_500.json", .71)
    result = collect_candidate(run, 1000)
    assert result["status"] == "paused"
    assert [row["available"] for row in result["evaluations"]] == [False, True]
    (run / "compressed_e2e_fintuning.log").write_text("Still training\n")
    with pytest.raises(ValueError, match="pause evidence"):
        collect_candidate(run, 1000)


def test_completed_run_uses_real_export_metric_when_training_endpoint_absent(tmp_path):
    run = tmp_path / "run"
    _run(run, 5000)
    _metrics(run / "lm_eval" / "lm_eval_results_step_4500.json", .70)
    result = collect_candidate(run, 5000)
    assert result["status"] == "completed"
    assert result["evaluations"][-1]["kind"] == "final_export"
    assert result["evaluations"][-1]["mean_percent"] == pytest.approx(69)
    _metrics(run / "lm_eval" / "lm_eval_results_step_5000.json", .705)
    result = collect_candidate(run, 5000)
    assert len(result["evaluations"]) == 2
    assert result["evaluations"][-1]["kind"] == "training"
    assert result["final_evaluation"]["mean_percent"] == pytest.approx(69)
    meta = read_json(run / "run_meta.json")
    meta["final_checkpoint_id"] = "wrong-id"
    atomic_json(run / "run_meta.json", meta)
    with pytest.raises(ValueError, match="metadata"):
        collect_candidate(run, 5000)


def test_completed_run_requires_a_reexportable_last_training_checkpoint(tmp_path):
    run = tmp_path / "run"
    _run(run, 5000)
    checkpoint = run / "trainer_state" / "checkpoint-5000"
    (checkpoint / "training_model_state.pt").unlink()
    with pytest.raises(FileNotFoundError, match="training_model_state"):
        collect_candidate(run, 5000)
    (checkpoint / "training_model_state.pt").write_bytes(b"fixture")
    atomic_json(checkpoint / "trainer_state.json", {"global_step": 4500, "max_steps": 5000})
    with pytest.raises(ValueError, match="Checkpoint state"):
        collect_candidate(run, 5000)


def test_cleanup_is_exact_keeps_named_steps_and_records_logical_bytes(tmp_path):
    root, run = tmp_path / "search", tmp_path / "search" / "run"
    _run(run, 1000)
    retired = run / "trainer_state" / "checkpoint-500"
    retired.mkdir()
    (retired / "weights").write_bytes(b"12345")
    unknown = run / "trainer_state" / "checkpoint-not-numeric"
    unknown.mkdir()
    records = delete_checkpoints(run, {1000}, (root,))
    assert len(records) == 1 and records[0]["bytes"] == 5
    assert not retired.exists()
    assert unknown.is_dir()
    assert (run / "trainer_state" / "checkpoint-1000").is_dir()
    assert (run / "lm_eval" / "lm_eval_results_step_1000.json").is_file()


def test_cleanup_preflights_symlinks_and_rejects_outside_roots(tmp_path):
    root, run = tmp_path / "search", tmp_path / "search" / "run"
    _run(run, 1000)
    external = tmp_path / "external"
    external.mkdir()
    (external / "protected").write_bytes(b"keep")
    (run / "trainer_state" / "checkpoint-500").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        delete_checkpoints(run, set(), (root,))
    assert (run / "trainer_state" / "checkpoint-1000").is_dir()
    assert (external / "protected").read_bytes() == b"keep"
    with pytest.raises(ValueError, match="outside"):
        delete_checkpoints(external, set(), (root,))


def test_csv_marks_the_kind_of_best_score_and_final_export_separately(tmp_path):
    run = tmp_path / "run"
    _run(run, 5000)
    candidate = {"name": "trial", "params": {"learning_rate": 1e-4}, **collect_candidate(run, 5000)}
    path = tmp_path / "leaderboard.csv"
    write_leaderboard(path, [candidate])
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["best_kind"] == "final_export"
    assert float(row["final_export_mean_percent"]) == pytest.approx(69)
    _metrics(run / "lm_eval" / "lm_eval_results_step_500.json", .71)
    candidate.update(collect_candidate(run, 5000))
    write_leaderboard(path, [candidate])
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["best_mean_percent"]) == pytest.approx(71)
    assert row["best_step"] == "500"
    assert float(row["best_available_mean_percent"]) == pytest.approx(69)
    assert row["best_available_step"] == "5000"
