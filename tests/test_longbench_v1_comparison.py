"""Aggregation contracts only; these synthetic scores are not benchmark results."""
from copy import deepcopy
import math
from pathlib import Path

import pytest

from experiments.compare_longbench_v1 import compare_reports
from experiments.longbench_v1.protocol import CATEGORIES, DATASETS, EXPECTED_COUNTS, protocol_record
from experiments.longbench_v1.scoring import METRICS


def score_report(kind):
    scores = {name: float(i) + (2 if kind == "compressed" and i < 4 else 0)
              for i, name in enumerate(DATASETS)}
    categories = {name: sum(scores[task] for task in tasks) / len(tasks)
                  for name, tasks in CATEGORIES.items()}
    return {"schema": "longbench_v1_scores_v1", "status": "complete", "sample_count": 4750,
            "archive_sha256": "a" * 64, "protocol": protocol_record(),
            "model": {"kind": kind, "hf_path": "same", "hf_metadata_sha256": {"same": "same"},
                      "hf_weight_stats": []},
            "tasks": {name: {"samples": EXPECTED_COUNTS[name], "score": score,
                              "official_rounded_score": round(score, 2), "metric": METRICS[name].__name__}
                      for name, score in scores.items()},
            "categories": categories, "task_macro_mean": sum(scores.values()) / 21,
            "category_macro_mean": sum(categories.values()) / 6}


def test_task_and_six_category_macros_keep_their_different_denominators():
    report = compare_reports(score_report("dense"), score_report("compressed"), archive_sha256="a" * 64)
    assert len(report["tasks"]) == 21 and len(report["categories"]) == 6
    assert report["task_macro_mean"]["dense"] == 10
    assert report["category_macro_mean"]["dense"] == pytest.approx(66.5 / 6)
    assert report["task_macro_mean"]["delta_score_points"] == pytest.approx(8 / 21)
    assert report["category_macro_mean"]["delta_score_points"] == pytest.approx(2 / 6)
    assert report["categories"]["single_document_qa"]["delta_score_points"] == 2
    assert report["tasks"]["repobench-p"]["delta_score_points"] == 0


def test_partial_mismatched_and_corrupt_score_artifacts_are_rejected():
    dense, valid = score_report("dense"), score_report("compressed")
    mutations = [
        lambda value: value.update(status="partial_runtime_budget"),
        lambda value: value.update(sample_count=4749),
        lambda value: value["tasks"].pop("lcc"),
        lambda value: value.update(archive_sha256="b" * 64),
        lambda value: value["protocol"].update(seed=0),
        lambda value: value["tasks"]["qasper"].update(score=math.nan),
        lambda value: value["tasks"]["qasper"].update(metric="accuracy"),
        lambda value: value.update(task_macro_mean=99),
        lambda value: value["model"].update(hf_path="another-original-model"),
    ]
    for mutate in mutations:
        bad = deepcopy(valid)
        mutate(bad)
        with pytest.raises(ValueError):
            compare_reports(dense, bad, archive_sha256="a" * 64)


def test_llama_pair_uses_its_own_real_model_protocol():
    path = Path("/home/star/models/meta-llama/Llama-3.1-8B-Instruct")
    if not path.is_dir():
        pytest.skip("The real local Llama metadata is required for this compatibility check.")
    expected = protocol_record(path)
    assert expected != protocol_record()
    dense, compressed = score_report("dense"), score_report("compressed")
    dense["protocol"] = deepcopy(expected)
    compressed["protocol"] = deepcopy(expected)
    result = compare_reports(dense, compressed, archive_sha256="a" * 64, expected_protocol=expected)
    assert result["protocol"] == expected
    assert result["sample_count_per_model"] == 4750
    with pytest.raises(ValueError):
        compare_reports(dense, compressed, archive_sha256="a" * 64)
    compressed["protocol"] = protocol_record()
    with pytest.raises(ValueError):
        compare_reports(dense, compressed, archive_sha256="a" * 64, expected_protocol=expected)
