"""SC-CAT complete-nine scoring with its confirmed <=1 point / <=2.6 logical goal."""
from fractions import Fraction
import argparse
import json
from pathlib import Path

from experiments.cat_auto_protocol_cmcc_20261104.contracts import digest, require
from experiments.cat_auto_protocol_cmcc_20261104.ledger import measure_checkpoint
from experiments.cat_auto_protocol_cmcc_20261104.nodes import FullEightScore
from experiments.cat_offline_execution_cmcc_20261104.protocol import load_baseline
from experiments.compare_longbench_v1 import comparison
from experiments.longbench_v1.common import check_hashes, check_stat, sha256
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.protocol import DATASETS
from experiments.sc_cat_live_cmcc_20261106.recovery_io import binding, read_bound_json, read_json, write_json
from train_utils import checkpoint_v6 as v6


def fraction_record(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def exact_nine(task_scores, longbench_tasks):
    require(len(task_scores) == 8 and set(longbench_tasks) == set(DATASETS), "Use all eight tasks plus all21 LongBench task scores.")
    eight = [Fraction(str(value)) for value in task_scores.values()]
    lb = [Fraction(str(value)) for value in longbench_tasks.values()]
    require(all(0 <= value <= 1 for value in eight) and all(0 <= value <= 100 for value in lb), "Keep fraction versus percent metric units explicit.")
    return (sum(eight) * 100 + sum(lb) / 21) / 9


def quality_flags(drop):
    return dict(quality_passed=drop <= 1, research_goal_quality_passed=drop < Fraction(9, 10))


def summarize(request):
    report, inventory = read_bound_json(request["deployment_report"]), read_bound_json(request["inventory_receipt"])["inventory"]
    directory = Path(report["path"]).resolve()
    meta = v6.load_v6_meta(str(directory))
    score_data = read_bound_json(request["full8_score"])
    score = FullEightScore(score_data["task_scores"], score_data["evaluation_id"], score_data["evidence"])
    check_hashes(score.evidence["artifacts"])
    baseline_path = request["baseline"]["path"]
    require(sha256(baseline_path) == request["baseline"]["sha256"], "The original BF16 eight-task reference changed.")
    baseline = load_baseline(baseline_path)
    require(meta["checkpoint_kind"] == "category_boundary" and report["phase"] == "formal"
        and meta["checkpoint_id"] == report["uuid"] == score.evidence["checkpoint_id"]
        and score.evidence["source_checkpoint_id"] == report["export_evidence"]["source_node_id"]
        and score.evidence["phase"] == "formal" and score.evidence["protocol_id"] == baseline["protocol_identity"]
        and baseline["model_identity"] == inventory["model_identity"]
        and not meta["pending_dense_targets"] and not meta["skip_targets"]
        and meta["completed_categories"] == meta["compression_categories"], "Bind one fully compressed formal artifact to its own complete-eight evidence.")
    dense_path, low_path, archive_path = (request[key]["path"] for key in ("baseline_longbench_scores", "compressed_longbench_scores", "archive"))
    for key in ("baseline_longbench_scores", "compressed_longbench_scores"):
        read_bound_json(request[key])
    require(sha256(archive_path) == request["archive"]["sha256"], "The prepared LongBench manifest changed.")
    load_archive(archive_path)
    lb = comparison(dense_path, low_path, archive_path)
    low = lb["models"]["compressed"]
    for model in lb["models"].values():
        require(Path(model["hf_path"]).resolve() == Path(inventory["source_evidence"]["path"]).resolve(),
            "Both LongBench sides must retain this exact original BF16/tokenizer source.")
        check_hashes(model["hf_metadata_sha256"])
        for stat in model["hf_weight_stats"]:
            check_stat(stat)
    check_stat(low["weight_stat"])
    require(Path(low["checkpoint"]).resolve() == directory and low["checkpoint_id"] == report["uuid"]
        and low["metadata_sha256"] == sha256(directory / "checkpoint_meta.json")
        and Path(low["hf_path"]).resolve() == Path(inventory["source_evidence"]["path"]).resolve(),
        "LongBench and complete-eight must describe the same already-evaluated deployment.")
    ledger = measure_checkpoint(directory, inventory)
    require(ledger["final_budget_applicable"], "Partial prefixes cannot claim a final storage result.")
    dense = exact_nine(baseline["task_metrics"], {task: row["dense"] for task, row in lb["tasks"].items()})
    compressed = exact_nine(score.task_scores, {task: row["compressed"] for task, row in lb["tasks"].items()})
    drop = dense - compressed
    flags = quality_flags(drop)
    quality, storage = flags["quality_passed"], ledger["logical_bits"] * 5 <= ledger["N"] * 13
    return dict(schema="sc_cat_complete_nine_v1", status="passed" if quality and storage else "not_met", exit_code=0,
        passed=bool(quality and storage), quality_target_evaluated=True, checkpoint_id=report["uuid"], checkpoint_dir=str(directory),
        baseline_nine_task_score=float(dense), compressed_nine_task_score=float(compressed), mean_drop_points=float(drop),
        baseline_nine_exact=fraction_record(dense), compressed_nine_exact=fraction_record(compressed), mean_drop_exact=fraction_record(drop),
        protocol=dict(terms=9, eight_task_units="fraction converted to percent", longbench_task_macro="21 unrounded task scores",
            maximum_drop_points=1, drop_bound_inclusive=True, maximum_logical_bpw="2.6", max_lora_rank=0),
        quality_passed=bool(quality), budget_passed=bool(storage), actual_ledger=ledger,
        research_goal_quality_passed=flags["research_goal_quality_passed"],
        research_goal_passed=bool(flags["research_goal_quality_passed"] and storage),
        research_goal=dict(maximum_drop_points="0.9", drop_bound_inclusive=False,
            complete_BF16_replay_evaluated=False, cross_model_evaluated=False),
        full8_mean_percent=float(score.mean * 100), longbench_task_macro_mean=lb["task_macro_mean"]["compressed"],
        deployment_report=request["deployment_report"], full8_score=request["full8_score"], baseline=request["baseline"],
        longbench=lb, source=dict(baseline_longbench=request["baseline_longbench_scores"],
            compressed_longbench=request["compressed_longbench_scores"], archive=request["archive"]),
        final_BF16_replay_not_implied=True)


def select_reported_endpoint(endpoints):
    require(1 <= len(endpoints) <= 2, "Only the one or two announced final raw-recipe endpoints may enter S9 selection.")
    rows = []
    for endpoint in endpoints:
        score = read_bound_json(endpoint["nine_task"])
        exact = score["compressed_nine_exact"]
        value = Fraction(exact["numerator"], exact["denominator"])
        require(score["quality_target_evaluated"] and score["exit_code"] == 0 and score["budget_passed"]
            and score["checkpoint_id"] == endpoint["checkpoint_id"], "Choose only a completed same-artifact complete-nine result.")
        rows.append((endpoint, value))
    chosen, value = min(rows, key=lambda row: (-row[1], not row[0]["is_anchor"], row[0]["logical_bits"],
        row[0]["packed_bits"], row[0]["candidate_id"], row[0]["trajectory_id"]))
    result = dict(selected=chosen, nine_score_exact=fraction_record(value), compared=endpoints,
        decision="maximum_unrounded_S9_then_anchor_logical_packed_candidate_trajectory", deliver_existing_artifact=True)
    result["selection_id"] = digest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    report = summarize(request)
    write_json(Path(request["output"]) / "nine_task.json", report)
    print(json.dumps(dict(status=report["status"], compressed_nine_task_score=report["compressed_nine_task_score"])), flush=True)


if __name__ == "__main__":
    main()
