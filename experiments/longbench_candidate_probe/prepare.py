"""Bind a current Qwen candidate to the existing sixteen retrieval cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.longbench_attribution.protocol import check_preparation
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.longbench_v1.models import check_config, make_config


def check_candidate_preparation(path):
    record = json.loads(Path(path).read_text())
    if record["schema"] != "longbench_candidate_retrieval_probe_v1":
        raise ValueError("Unexpected candidate diagnostic schema.")
    check_hashes(record["source_sha256"])
    check_hashes(record["evidence_sha256"])
    reference, manifest = check_preparation(record["reference_preparation"])
    config = record["config"]
    check_config(config)
    if (config["model"]["kind"] != "compressed" or config["mode"] != "validation"
            or config["model_policy"]["model_type"] != "qwen3"
            or record["cases"] != reference["cases"] or len(record["cases"]) != 16):
        raise ValueError("Require an unchanged Qwen candidate and the original sixteen cases.")
    for field in ("archive", "protocol", "package_versions"):
        if config[field] != reference["config"][field]:
            raise ValueError(f"Candidate/control protocol differs: {field}")
    for field in ("hf_path", "hf_metadata_sha256", "hf_weight_stats"):
        if config["model"][field] != reference["config"]["model"][field]:
            raise ValueError(f"Candidate/control origin differs: {field}")
    return record, reference, manifest


def prepare(reference_preparation, checkpoint, audit, terminal_record, output):
    reference_preparation, checkpoint, audit, terminal_record, output = (
        Path(value).resolve() for value in (reference_preparation, checkpoint, audit, terminal_record, output))
    if output.exists():
        raise FileExistsError(output)
    reference, manifest = check_preparation(reference_preparation)
    verified = json.loads(audit.read_text())
    terminal = json.loads(terminal_record.read_text())
    config = make_config(reference["config"]["archive"]["path"], kind="compressed",
                         hf_path=manifest["tokenizer"]["path"], checkpoint=checkpoint, mode="validation")
    if (verified["status"] != "verified" or verified["checkpoint_id"] != config["model"]["checkpoint_id"]
            or Path(verified["checkpoint_dir"]).resolve() != checkpoint
            or verified.get("fresh_strict_reload_state_exact") is not True
            or terminal["status"] != "completed" or terminal["exit_code"] != 0
            or (terminal_record.parent / "run.exit").read_text().strip() != "0"):
        raise ValueError("Candidate needs its completed export, strict audit and terminal run.")
    sources = list(Path(__file__).parent.glob("*.py"))
    sources.extend(Path(__file__).resolve().parents[1] / "longbench_attribution" / name
                   for name in ("protocol.py", "restore.py"))
    record = {"schema": "longbench_candidate_retrieval_probe_v1", "status": "prepared_not_scheduled",
              "gpu": None, "queued": False, "config": config,
              "reference_preparation": str(reference_preparation), "cases": reference["cases"],
              "selection": reference["selection"],
              "source_sha256": {str(path.resolve()): sha256(path) for path in sources},
              "evidence_sha256": {str(path): sha256(path) for path in
                                  (reference_preparation, audit, terminal_record, terminal_record.parent / "run.exit")},
              "scope": "Fixed sixteen-case retrieval diagnosis only; no full LongBench or nine-item acceptance.",
              "model_mutations": False, "model_exported": False, "goal_acceptance": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, record)
    check_candidate_preparation(output)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference_preparation", "checkpoint", "audit", "terminal_record", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    record = prepare(**vars(args))
    print(json.dumps({"checkpoint_id": record["config"]["model"]["checkpoint_id"],
                      "cases": len(record["cases"]), "status": record["status"]}))


if __name__ == "__main__":
    main()
