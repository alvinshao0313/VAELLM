"""Complete-nine quality/storage profile: drop <0.9 points and logical <=2.6bpw.

This profile alone does not establish that a compression pipeline is transferable.
Historical acceptance sources stay byte-for-byte intact for existing source pins.
"""
from fractions import Fraction
from pathlib import Path

from experiments.compare_longbench_v1 import compare_reports
from experiments.qwen3_w25.summarize_results import TASK_METRICS, _require
from experiments.w25_eval import nine_task_acceptance as historical
from experiments.w25_eval.acceptance import read_json
from experiments.w25_eval.model_inventory import sha256
from train_utils.checkpoint_bpw import audit_checkpoint

TRANSFER_GOAL = "transfer_2_6_lt_0_9"


def combine_scores(model_type, baseline_tasks, compressed_tasks, longbench, budget):
    """Reuse complete-nine validation, then apply the explicitly separate profile."""
    result = historical.combine_scores(model_type, baseline_tasks, compressed_tasks, longbench, budget)
    # Equality must fail; scores genuinely below the threshold have no exclusion
    # band. Compare the decimal values recorded in the input artifacts exactly.
    exact_drop = (sum((Fraction(str(baseline_tasks[key])) - Fraction(str(compressed_tasks[key]))) * 100
                      for key in TASK_METRICS)
                  + sum(Fraction(str(row["dense"])) - Fraction(str(row["compressed"]))
                        for row in longbench["tasks"].values()) / 21) / 9
    mean_passed = exact_drop < Fraction("0.9")
    bits, weights = budget["user_linear_bits"], budget["original_linear_weights"]
    budget_passed = bool(budget["budget_passed"]) and bits * 5 <= weights * 13
    passed = mean_passed and budget_passed
    failures = []
    if not mean_passed:
        failures.append(f"Nine-term score drop {float(exact_drop):.12g} must be strictly below 0.9 points.")
    if not budget_passed:
        failures.append("Actual projection storage/rank/protection budget does not pass.")
    failures.extend(budget.get("contract_violations", []))
    result.update(schema="w25_transfer_nine_task_acceptance_v1", goal=TRANSFER_GOAL,
        status="passed" if passed else "not_met", passed=passed, mean_passed=mean_passed,
        budget_passed=budget_passed, failure_reasons=failures, mean_drop_points=float(exact_drop),
        mean_drop_points_exact={"numerator": exact_drop.numerator, "denominator": exact_drop.denominator},
        minimum_compressed_nine_task_score=result["baseline_nine_task_score"] - 0.9,
        acceptance_scope="complete_nine_task_quality_and_actual_projection_storage_only",
        transferability_evaluated=False, packed_bpw_reported_separately=budget["packed_linear_bpw"])
    result["protocol"].update(maximum_absolute_drop_points=0.9, strict_score_drop_bound=True,
        minimum_compressed_nine_task_score_is_exclusive=True, score_boundary_roundoff_tolerance=0,
        max_linear_bpw=2.6, strict_bpw_bound=False,
        score_comparison="Exact rational arithmetic on recorded decimal downstream fractions and LongBench task scores.")
    return result


def _sample_artifacts(paths):
    """Retain the aligned evaluator's complete raw downstream sample binding."""
    artifacts, leaf_count, sample_count = {}, 0, 0
    for path in paths:
        result = read_json(path)["evals"]["lm_eval"]
        counts = result["n_samples"]
        samples = result["sample_artifacts"]
        expected = sum(row["effective"] for row in counts.values())
        _require(samples["total_count"] == expected and set(samples["tasks"]) == set(counts),
                 "Incomplete downstream raw sample artifacts.")
        for task, row in samples["tasks"].items():
            artifact = Path(row["path"]).resolve()
            _require(artifact.is_relative_to(Path(path).resolve().parent)
                     and row["count"] == counts[task]["effective"] and str(artifact) not in artifacts
                     and sha256(artifact) == row["sha256"], "Downstream raw sample binding changed.")
            artifacts[str(artifact)] = row["sha256"]
        leaf_count += len(counts)
        sample_count += expected
    _require(leaf_count == 64 and sample_count == 24742, "Require all64 leaf tasks/24742 raw samples.")
    return {"leaf_count": leaf_count, "sample_count": sample_count, "sha256": artifacts}


def summarize(base_model_path, checkpoint_dir, baseline_results, compressed_results,
              longbench_manifest, baseline_longbench_scores, compressed_longbench_scores):
    # The old profile can return not_met; its complete provenance/shape/protocol
    # validation still applies. Its 2.5bpw verdict is not a gate for this profile.
    result = historical.summarize(base_model_path, checkpoint_dir, baseline_results, compressed_results,
                                 longbench_manifest, baseline_longbench_scores, compressed_longbench_scores)
    budget = audit_checkpoint(checkpoint_dir, max_bpw=2.6, max_lora_rank=16)
    _require(all(budget[key] == result["bit_budget"][key] for key in
                 ("checkpoint_id", "components", "user_linear_bits", "packed_linear_bits",
                  "outside_linear_bits", "original_linear_weights", "compressed_target_count")),
             "Actual checkpoint storage changed between provenance and profile audits.")
    reports = []
    for side, path in (("dense_scores", baseline_longbench_scores), ("compressed_scores", compressed_longbench_scores)):
        report = read_json(path)
        _require(sha256(path) == result["longbench_inputs"][side]["sha256"], "LongBench scores changed after validation.")
        reports.append(report)
    longbench = compare_reports(*reports, archive_sha256=reports[0]["archive_sha256"],
                                expected_protocol=reports[0]["protocol"])
    result.update(combine_scores(result["model_type"], result["downstream_baseline"]["task_metrics"],
                                result["downstream_compressed"]["task_metrics"], longbench, budget))
    result.update(downstream_sample_artifacts=_sample_artifacts(compressed_results),
        implementation_sha256=sha256(Path(__file__)),
        historical_validation_implementation_sha256=sha256(Path(historical.__file__)))
    return result
