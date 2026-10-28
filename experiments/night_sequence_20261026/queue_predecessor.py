"""Read-only completeness checks for the five already running deoscillation arms."""
from __future__ import annotations

from pathlib import Path

from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS
from experiments.night_sequence_20261026.queue_wait import read_json, require


ROOT = Path(__file__).resolve().parents[2]
DEOSC = ROOT / "result/qwen3_8b_mmlu65_20261004/qwen_deoscillation_20261026/formal"
PREDECESSORS = (
    {"gpu": 0, "arm": "baseline", "pid": 3867026, "start_ticks": 232870862},
    {"gpu": 1, "arm": "risk", "pid": 3867047, "start_ticks": 232870885},
    {"gpu": 2, "arm": "matched", "pid": 3867061, "start_ticks": 232870908},
    {"gpu": 3, "arm": "half_lr", "pid": 3867072, "start_ticks": 232870933},
    {"gpu": 4, "arm": "freeze_codes", "pid": 3867082, "start_ticks": 232870953},
)


def validate_predecessor(binding: dict) -> dict:
    arm = binding["arm"]
    directory = DEOSC / "arms" / arm
    chain_path = directory / "chain.json"
    chain = read_json(chain_path)
    stages = ("train", "export_" + arm, "full9_" + arm)
    require(chain.get("pid") == binding["pid"] and chain.get("arm") == arm
            and chain.get("status") == "complete" and set(chain.get("stages", {})) == set(stages)
            and all(chain["stages"][name].get("exit_code") == 0 for name in stages),
            f"Bound predecessor {arm} did not complete all fixed stages successfully.")
    training = read_json(directory / "training/report.json")
    require(training.get("status") == "training_complete" and training.get("completed_updates") == 256,
            f"Predecessor {arm} has no complete actual256 trajectory.")
    endpoint = directory / "training/step_0256"
    audit = read_json(endpoint / "native_audit.json")
    checkpoint_id = audit["checkpoint_id"]
    output = endpoint / "evaluation_full9"
    evaluation = read_json(output / "evaluation.json")
    require(audit.get("status") == "verified" and evaluation.get("checkpoint_id") == checkpoint_id
            and evaluation.get("arm") == arm and evaluation.get("global_step") == 256
            and evaluation.get("status") == "complete_full8_full21_nine"
            and set(evaluation.get("stages", {})) == {"mmlu", "seven", "longbench"}
            and all(str(row.get("exit_code")) == "0" for row in evaluation["stages"].values()),
            f"Predecessor {arm} has no successful complete evaluation.")
    full8 = read_json(output / "full8_summary.json")
    require(full8.get("checkpoint_id") == checkpoint_id and full8.get("sample_count") == 24742
            and full8.get("leaf_tasks") == 64 and full8.get("sample_counts") == {"mmlu": 14042, "seven": 10700},
            f"Predecessor {arm} full8 sample coverage is incomplete.")
    leaves = 0
    for name, expected in (("mmlu", 14042), ("seven", 10700)):
        lm = read_json(output / (name + ".json"))["evals"]["lm_eval"]
        artifacts = lm["sample_artifacts"]
        rows = artifacts["tasks"]
        require(artifacts.get("total_count") == expected and set(rows) == set(lm["n_samples"])
                and sum(row["count"] for row in rows.values()) == expected
                and all(row["count"] == lm["n_samples"][task]["effective"] for task, row in rows.items()),
                f"Predecessor {arm}/{name} raw sample counts are incomplete.")
        leaves += len(rows)
    require(leaves == 64, f"Predecessor {arm} full8 leaf coverage changed.")
    scores = read_json(output / "longbench/scores.json")
    require(scores.get("status") == "complete" and scores.get("sample_count") == 4750
            and set(scores.get("tasks", {})) == set(DATASETS)
            and all(scores["tasks"][task]["samples"] == count for task, count in EXPECTED_COUNTS.items())
            and scores["model"]["checkpoint_id"] == checkpoint_id,
            f"Predecessor {arm} full21 sample coverage or identity is incomplete.")
    nine = read_json(output / "nine_task_acceptance.json")
    require(nine.get("quality_target_evaluated") is True and nine.get("checkpoint_id") == checkpoint_id,
            f"Predecessor {arm} has no valid complete nine-term score.")
    # A valid score below the quality target is a completed experiment.
    return {"chain": str(chain_path), "evaluation": str(output / "evaluation.json"),
            "checkpoint_id": checkpoint_id, "full8_samples": 24742, "longbench_samples": 4750,
            "nine_task_score": nine["compressed_nine_task_score"], "quality_passed": nine["passed"]}
