from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time
import requests
import zipfile

from .common import sha256, write_json
from .protocol import (DATASETS, DATA_REVISION, DATA_ZIP_SHA256, DATA_ZIP_BYTES,
                       EXPECTED_COUNTS, encode_example, input_bytes, official_files, protocol_record)


def download_and_extract(directory, *, resume_download=False):
    """Fetch only the public, revision-pinned official zip; no credentials."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "data.zip"
    if not archive.exists():
        url = f"https://huggingface.co/datasets/THUDM/LongBench/resolve/{DATA_REVISION}/data.zip"
        temporary = directory / "data.zip.part"
        start = temporary.stat().st_size if temporary.exists() else 0
        if start and not resume_download:
            raise FileExistsError("Partial data download exists; request --resume-download explicitly.")
        if start >= DATA_ZIP_BYTES:
            raise ValueError("Partial download has invalid size; inspect it before continuing.")
        headers = {"Range": f"bytes={start}-"} if start else {}
        attempt = {"start_byte": start, "expected_total_bytes": DATA_ZIP_BYTES,
                   "revision": DATA_REVISION, "expected_sha256": DATA_ZIP_SHA256}
        # Do not print redirect URLs, headers, credentials, or signed URLs.
        try:
            with requests.get(url, stream=True, headers=headers, timeout=(30, 180)) as response:
                if response.status_code != (206 if start else 200):
                    raise RuntimeError(f"Official data GET returned HTTP {response.status_code}.")
                content_range = response.headers.get("Content-Range")
                if start and content_range != f"bytes {start}-{DATA_ZIP_BYTES - 1}/{DATA_ZIP_BYTES}":
                    raise ValueError("Server did not honor the exact pinned data byte range.")
                attempt.update(status_code=response.status_code, content_range=content_range,
                               content_length=response.headers.get("Content-Length"))
                with temporary.open("ab" if start else "xb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        output.write(chunk)
            if temporary.stat().st_size != DATA_ZIP_BYTES or sha256(temporary) != DATA_ZIP_SHA256:
                raise ValueError("Downloaded data does not match pinned LFS size/SHA256.")
            attempt["verified"] = True
        except BaseException as error:
            attempt.update(verified=False, error_type=type(error).__name__)
            if isinstance(error, requests.RequestException):
                # HTTP client exceptions may include temporary signed redirect URLs.
                raise RuntimeError(f"Pinned data transfer failed ({type(error).__name__}); partial bytes retained.") from None
            raise
        finally:
            attempt["actual_bytes"] = temporary.stat().st_size if temporary.exists() else 0
            write_json(directory / "download.json", attempt)
        temporary.replace(archive)
    if archive.stat().st_size != DATA_ZIP_BYTES or sha256(archive) != DATA_ZIP_SHA256:
        raise ValueError("Official data.zip does not match its pinned LFS size/SHA256.")
    data_dir = directory / "raw"
    data_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as stream:
        for name in DATASETS:
            matches = [item for item in stream.infolist()
                       if PurePosixPath(item.filename).name == name + ".jsonl"]
            if len(matches) != 1:
                raise ValueError(f"Expected one official zip member for {name}.")
            member = matches[0]
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe zip member.")
            destination = data_dir / (name + ".jsonl")
            with stream.open(member) as source:
                contents = source.read()
            if destination.exists():
                if destination.read_bytes() != contents:
                    raise ValueError(f"Existing official data differs: {destination}")
            else:
                destination.write_bytes(contents)
    return data_dir


def tokenizer_pins(path):
    path = Path(path).resolve()
    names = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json", "generation_config.json")
    pins = {str(path / name): sha256(path / name) for name in names if (path / name).is_file()}
    if not (path / "tokenizer.json").is_file() or not (path / "tokenizer_config.json").is_file():
        raise ValueError("A complete local fast tokenizer is required.")
    return pins


def prepare(directory, tokenizer_path, *, resume_download=False):
    directory = Path(directory).resolve()
    if (directory / "manifest.json").exists():
        raise FileExistsError("A prepared archive already exists; validate and reuse it.")
    started = time.time()
    data_dir = download_and_extract(directory, resume_download=resume_download)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError("Expected the original Qwen3 fast tokenizer with a pad token.")
    pins = tokenizer_pins(tokenizer_path)
    prompts, lengths, upstream = official_files()
    tokens_path, index_path = directory / "inputs.u32", directory / "index.jsonl"
    counts, raw_hashes, offset, stats = {}, {}, 0, {}
    with tokens_path.open("xb") as tokens, index_path.open("x", encoding="utf-8") as index:
        for dataset in DATASETS:
            raw_path = data_dir / (dataset + ".jsonl")
            raw_hashes[str(raw_path)] = sha256(raw_path)
            seen = set()
            stats[dataset] = {"samples": 0, "truncated": 0, "input_tokens": 0,
                              "removed_tokens": 0, "max_input_tokens": 0}
            with raw_path.open("rb") as raw:
                for row_number, line in enumerate(raw):
                    item = json.loads(line)
                    identity = item["_id"]
                    if identity in seen or not isinstance(item["answers"], list) or not item["answers"]:
                        raise ValueError(f"Invalid/repeated official example: {dataset}:{row_number}")
                    seen.add(identity)
                    ids, metadata = encode_example(tokenizer, dataset, item, prompts, lengths)
                    payload = input_bytes(ids)
                    tokens.write(payload)
                    record = {"dataset": dataset, "row_index": row_number, "_id": identity,
                              "source_line_sha256": hashlib.sha256(line).hexdigest(),
                              "token_offset": offset, "answers": item["answers"],
                              "all_classes": item["all_classes"], "length": item["length"], **metadata}
                    index.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    offset += len(ids)
                    stat = stats[dataset]
                    stat["samples"] += 1
                    stat["truncated"] += int(metadata["removed_tokens"] > 0)
                    stat["input_tokens"] += len(ids)
                    stat["removed_tokens"] += metadata["removed_tokens"]
                    stat["max_input_tokens"] = max(stat["max_input_tokens"], len(ids))
            counts[dataset] = len(seen)
            if counts[dataset] != EXPECTED_COUNTS[dataset]:
                raise ValueError(f"Unexpected official sample count for {dataset}: {counts[dataset]}")
            print(json.dumps({"prepared_dataset": dataset, **stats[dataset]}), flush=True)
        tokens.flush()
        os.fsync(tokens.fileno())
        index.flush()
        os.fsync(index.fileno())
    record = {"schema": "longbench_v1_shared_inputs_v1", "status": "prepared",
              "protocol": protocol_record(), "upstream": upstream,
              "data": {"repository": "THUDM/LongBench", "revision": DATA_REVISION,
                       "zip_sha256": DATA_ZIP_SHA256, "raw_sha256": raw_hashes, "counts": counts},
              "tokenizer": {"path": str(Path(tokenizer_path).resolve()), "use_fast": True, "sha256": pins},
              "inputs": {"path": str(tokens_path), "sha256": sha256(tokens_path),
                         "dtype": "uint32_little_endian", "tokens": offset, "bytes": tokens_path.stat().st_size},
              "index": {"path": str(index_path), "sha256": sha256(index_path)},
              "sample_count": sum(counts.values()), "statistics": stats,
              "duration_seconds": time.time() - started}
    write_json(directory / "manifest.json", record)
    return record


def load_archive(manifest_path, *, check_token_file=True):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["schema"] != "longbench_v1_shared_inputs_v1" or manifest["protocol"] != protocol_record():
        raise ValueError("Prepared LongBench protocol mismatch.")
    if manifest["sample_count"] != 4750 or manifest["data"]["counts"] != EXPECTED_COUNTS:
        raise ValueError("Full 21-task archive required.")
    if sha256(manifest["index"]["path"]) != manifest["index"]["sha256"]:
        raise ValueError("Input index changed.")
    if check_token_file and sha256(manifest["inputs"]["path"]) != manifest["inputs"]["sha256"]:
        raise ValueError("Input token bytes changed.")
    with Path(manifest["index"]["path"]).open() as stream:
        rows = [json.loads(line) for line in stream]
    offset, observed = 0, {}
    for row in rows:
        name = row["dataset"]
        if name not in EXPECTED_COUNTS or row["token_offset"] != offset:
            raise ValueError("Broken task or offset in input archive.")
        if row["row_index"] != observed.get(name, 0):
            raise ValueError("Noncanonical per-task sample order.")
        if row["max_new_tokens"] != manifest["protocol"]["max_new_tokens"][name]:
            raise ValueError("Wrong official generation length.")
        if row["input_tokens"] + row["max_new_tokens"] > 32768:
            raise ValueError("Prepared sequence exceeds total window.")
        offset += row["input_tokens"]
        observed[name] = observed.get(name, 0) + 1
    if observed != EXPECTED_COUNTS or len(rows) != 4750 or offset != manifest["inputs"]["tokens"]:
        raise ValueError("Prepared archive coverage mismatch.")
    if Path(manifest["inputs"]["path"]).stat().st_size != offset * 4:
        raise ValueError("Prepared token file has trailing or missing bytes.")
    return manifest, rows
