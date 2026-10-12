"""Authenticate and copy the original task artifact's complete MMLU component."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Dataset
from e2e_common.data import record_to_mcqa_example
from experiments.mmlu_sft_probe.data import sha256


def extract(source_manifest: Path, arrow: Path, info: Path, output_dir: Path) -> dict:
    task = json.loads(source_manifest.read_text())
    source = Path(task["path"])
    if task["samples"] != 176597 or sha256(source) != task["sha256"]:
        raise ValueError("Original task artifact does not match its authenticated manifest.")
    generator = Path(__file__).resolve().parents[2] / "tools/prepare_vaellm_task_mix.py"
    if sha256(generator) != task["generator_sha256"]:
        raise ValueError("The generating code differs from the task manifest.")
    revision = task["source_revision_cache"]["datasets--cais--mmlu"]
    metadata = json.loads(info.read_text())
    expected_url = f"hf://datasets/cais/mmlu@{revision}/auxiliary_train/train-00000-of-00001.parquet"
    if (metadata["config_name"] != "auxiliary_train"
            or set(metadata["download_checksums"]) != {expected_url}
            or metadata["splits"]["train"]["num_examples"] != 99842):
        raise ValueError("HF cache is not the recorded auxiliary_train/train revision.")
    # These are the existing full-cache identities used by the verified LM arm.
    if sha256(arrow) != "af0bbd09f5648bf5841df34708d700065fa7a3eb973953c1c4a1817e52af6bf0":
        raise ValueError("HF auxiliary_train Arrow identity changed.")
    if sha256(info) != "b1504e4672c717df40c34a15eb3427bbfb37b5dcf0c5589f6791d29795ae0916":
        raise ValueError("HF auxiliary_train metadata identity changed.")
    hf = Dataset.from_file(str(arrow))
    if len(hf) != 99842:
        raise ValueError("Unexpected auxiliary_train length.")
    offset = task["samples"] - len(hf)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "mmlu_auxiliary_train_sft.jsonl"
    report_path = output_dir / "manifest.json"
    if output.exists() or report_path.exists():
        raise FileExistsError("Use a fresh extraction directory.")
    partial = output.with_suffix(".jsonl.partial")
    proof = hashlib.sha256()
    matched = 0
    with source.open("rb") as handle, partial.open("xb") as writer:
        for ordinal, raw in enumerate(handle):
            if ordinal < offset:
                continue
            hf_index = ordinal - offset
            if hf_index >= len(hf):
                raise ValueError("Original task artifact contains unexpected trailing rows.")
            example = record_to_mcqa_example(hf[hf_index], text_format="mmlu_mcqa")
            if example is None:
                raise ValueError(f"HF auxiliary_train row {hf_index} is not representable.")
            expected = {"messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": " " + chr(ord("A") + example["answer_index"])},
            ]}
            if json.loads(raw) != expected:
                raise ValueError(f"Original task row {ordinal + 1} does not match HF row {hf_index}.")
            writer.write(raw)
            proof.update(f"{ordinal + 1}:{hf_index}:{hashlib.sha256(raw).hexdigest()}\n".encode())
            matched += 1
    if matched != len(hf):
        raise ValueError("Incomplete MMLU component; no verified output was published.")
    partial.replace(output)
    result = {
        "status": "verified", "rows": matched, "all_rows_matched": True,
        "output_bytes_preserved": True,
        "output": {"path": str(output.resolve()), "sha256": sha256(output), "size": output.stat().st_size},
        "source": {"repo": "cais/mmlu", "revision": revision, "config": "auxiliary_train", "split": "train",
                   "arrow_path": str(arrow.resolve()), "arrow_sha256": sha256(arrow),
                   "dataset_info_path": str(info.resolve()), "dataset_info_sha256": sha256(info)},
        "task_artifact": {"path": str(source), "sha256": task["sha256"], "rows": task["samples"],
                          "manifest_path": str(source_manifest.resolve()), "manifest_sha256": sha256(source_manifest)},
        "mapping": {"task_line_first_1based": offset + 1, "task_line_last_1based": task["samples"],
                    "hf_index_first_0based": 0, "hf_index_last_0based": len(hf) - 1,
                    "rule": "output row j is exact original task raw line offset+j and equals the existing generator's HF row j prompt/answer",
                    "all_row_identity_sha256": proof.hexdigest()},
        "selection": "Complete component, validated row-by-row against fixed HF source; no length/content heuristic, deduplication, reordering or loss-based filtering.",
        "generator": {"path": str(generator), "sha256": sha256(generator)},
        "extractor_sha256": sha256(__file__),
    }
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--hf_arrow", type=Path, required=True)
    parser.add_argument("--hf_info", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.source_manifest, args.hf_arrow, args.hf_info, args.output_dir)
    print(json.dumps({"status": result["status"], "rows": result["rows"], "output": result["output"]}))


if __name__ == "__main__":
    main()
