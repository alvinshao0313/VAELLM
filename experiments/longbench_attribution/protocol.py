"""Fixed geometry-selected retrieval cases and immutable, already-run controls."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from experiments.longbench_v1.common import check_hashes, sha256, source_pins, write_json
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.models import check_config
from experiments.longbench_v1.protocol import generation_options
from experiments.longbench_v1.runner import read_input
from experiments.longbench_v1.scoring import check_prediction, score_example
from experiments.w25_eval.model_inventory import model_inventory

TASKS = ("passage_retrieval_en", "passage_retrieval_zh")
ARMS = {"qk": ("q_proj", "k_proj"), "ov": ("o_proj", "v_proj"),
        "mlp": ("gate_proj", "up_proj", "down_proj")}


def select_rows(rows):
    chosen = []
    for task in TASKS:
        ordered = sorted((row for row in rows if row["dataset"] == task),
                         key=lambda row: (row["input_tokens"], row["row_index"]))
        if len(ordered) != 200:
            raise ValueError("Expected the complete 200-row retrieval task.")
        chosen.extend(ordered[i * 199 // 7] for i in range(8))
    return chosen


def prepare(longbench_root, output):
    from transformers import AutoTokenizer

    root, output = Path(longbench_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    control_paths = {kind: root / kind / "config.json" for kind in ("compressed", "dense")}
    controls = {kind: json.loads(path.read_text()) for kind, path in control_paths.items()}
    source = controls["compressed"]
    if source["model"]["kind"] != "compressed" or controls["dense"]["model"]["kind"] != "dense":
        raise ValueError("Expected compressed and original BF16 controls.")
    for key in ("archive", "protocol", "package_versions"):
        if controls["dense"][key] != source[key]:
            raise ValueError(f"Control protocols differ: {key}")
    if source["model"]["hf_path"] != controls["dense"]["model"]["hf_path"]:
        raise ValueError("The controls must share the official tokenizer/model origin.")
    for key in ("hf_metadata_sha256", "hf_weight_stats"):
        if source["model"][key] != controls["dense"]["model"][key]:
            raise ValueError(f"Control HF source identities differ: {key}")
    # Historical config bytes remain unchanged. Only this new run binds current
    # compatible source; historical predictions keep their original config SHA.
    config = deepcopy(source)
    config["source_sha256"] = source_pins()
    check_config(config)
    manifest, rows = load_archive(config["archive"]["path"])
    selected = select_rows(rows)
    architecture = model_inventory(config["model"]["hf_path"])
    selected_targets = {arm: [name for name in architecture["projection_shapes"]
                             if name.rsplit(".", 1)[1] in suffixes] for arm, suffixes in ARMS.items()}
    if any(len(selected_targets[arm]) != architecture["num_hidden_layers"] * len(suffixes)
           for arm, suffixes in ARMS.items()):
        raise ValueError("Incomplete actual projection inventory.")
    tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer"]["path"], use_fast=True, local_files_only=True)
    tokens = np.memmap(manifest["inputs"]["path"], dtype="<u4", mode="r")
    evidence = {str(path): sha256(path) for path in control_paths.values()}
    control_records = {}
    for kind, path in control_paths.items():
        by_id = {}
        for task in TASKS:
            predictions = root / kind / "predictions" / (task + ".jsonl")
            evidence[str(predictions)] = sha256(predictions)
            values = [json.loads(line) for line in predictions.read_text().splitlines()]
            task_rows = [row for row in rows if row["dataset"] == task]
            if len(values) != len(task_rows):
                raise ValueError("Historical retrieval control is incomplete.")
            for predicted, row in zip(values, task_rows):
                check_prediction(predicted, row, evidence[str(path)])
                by_id[(task, row["row_index"])] = predicted
        records = []
        for row in selected:
            read_input(tokens, row)
            predicted = by_id[(row["dataset"], row["row_index"])]
            options = generation_options(tokenizer, row["dataset"], row["input_tokens"], row["max_new_tokens"])
            if (predicted["generation_options"] != options or predicted["effective_generation_options"] != options
                    or predicted["use_model_defaults"] is not False or predicted["input_tokens"] != row["input_tokens"]
                    or len(predicted["output_token_ids"]) != predicted["generated_tokens"]
                    or tokenizer.decode(predicted["output_token_ids"], skip_special_tokens=True) != predicted["pred"]):
                raise ValueError("Historical generation settings or decoded output differ.")
            records.append({"source_prediction": predicted,
                            "official_score": score_example(row["dataset"], predicted["pred"], row["answers"], row["all_classes"])})
        control_records[kind] = records
    del tokens
    own_paths = list(Path(__file__).parent.glob("*.py"))
    own_paths.append(Path(__file__).resolve().parents[1] / "w25_eval" / "model_inventory.py")
    new_sources = {str(path.resolve()): sha256(path) for path in own_paths}
    record = {"schema": "longbench_retrieval_restoration_v1", "status": "prepared_not_scheduled",
        "gpu": None, "queued": False, "config": config,
        "selection": "For each task sort (input_tokens,row_index), take floor(i*199/7), i=0..7; no labels/predictions used.",
        "cases": selected, "controls": control_records, "historical_evidence_sha256": evidence,
        "source_sha256": new_sources, "arms": ARMS, "architecture": architecture,
        "selected_targets": selected_targets,
        "scope": "16-case retrieval diagnosis only; mixed BF16 arms have no compression-budget claim or full benchmark acceptance.",
        "historical_source_sha256": source["source_sha256"],
        "historical_selected_inference_seconds": {kind: sum(item["source_prediction"]["elapsed_seconds"]
            for item in records) for kind, records in control_records.items()},
        "historical_to_current_changed_sources": {path: {"historical": digest, "current": config["source_sha256"].get(path)}
            for path, digest in source["source_sha256"].items() if config["source_sha256"].get(path) != digest}}
    write_json(output, record)
    return record


def check_preparation(path):
    record = json.loads(Path(path).read_text())
    if record["schema"] != "longbench_retrieval_restoration_v1" or record["arms"] != {k: list(v) for k, v in ARMS.items()}:
        raise ValueError("Unexpected attribution protocol.")
    check_hashes(record["source_sha256"])
    check_hashes(record["historical_evidence_sha256"])
    check_config(record["config"])
    manifest, rows = load_archive(record["config"]["archive"]["path"])
    if record["cases"] != select_rows(rows):
        raise ValueError("Geometry-selected inputs changed.")
    return record, manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longbench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.longbench_root, args.output)


if __name__ == "__main__":
    main()
