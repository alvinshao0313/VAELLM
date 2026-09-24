"""Document/window sampling adapted from pinned official LiftQuant datautils.py."""
import hashlib
import json
import random
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets

DATASET = "ZengXiangyu/RedPajama-Data-1T-Sample"
REVISION = "72b3875c770e4579639931fed89dc95e4067edac"


def calibration(args, tokenizer):
    if args.smoke_rows:
        payload = json.loads(Path(args.smoke_rows).read_text())
        if payload.get("dataset") != DATASET:
            raise ValueError("Smoke rows must identify the configured RedPajama dataset.")
        data = Dataset.from_list([row["row"] for row in payload["rows"]])
        source = {"dataset": DATASET, "scope": payload["scope"],
                  "revision": payload["revision"], "source": payload["source"],
                  "file_sha256": hashlib.sha256(Path(args.smoke_rows).read_bytes()).hexdigest()}
    else:
        if not args.redpajama_arrow_dir:
            raise ValueError("Full sampling requires --redpajama-arrow-dir with the original 11 RedPajama Arrow shards.")
        root = Path(args.redpajama_arrow_dir)
        info_path = root / "dataset_info.json"
        info = json.loads(info_path.read_text())
        if info.get("dataset_name") != "red_pajama-data-1_t-sample":
            raise ValueError("Provided Arrow directory is not the upstream RedPajama dataset.")
        files = [root / f"red_pajama-data-1_t-sample-train-{i:05d}-of-00011.arrow" for i in range(11)]
        if any(not path.is_file() for path in files):
            raise FileNotFoundError("All 11 original RedPajama train shards are required for full sampling.")
        data = concatenate_datasets([Dataset.from_file(str(path)) for path in files])
        if len(data) != info["splits"]["train"]["num_examples"] or len(data) != 930514:
            raise ValueError("Incomplete or incorrect RedPajama training set.")
        source = {"dataset": DATASET, "scope": "official full dataset sampling",
                  "arrow_directory": str(root.resolve()), "rows": len(data),
                  "dataset_info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest()}
    data = data.shuffle(seed=args.seed)
    rng = random.Random(args.seed)
    samples = []
    for _ in range(args.nsamples):
        for _attempt in range(10000):
            row = rng.randint(0, len(data) - 1)
            tokens = tokenizer(data[row]["text"], return_tensors="pt").input_ids
            if tokens.shape[1] >= args.seqlen + 1:
                break
        else:
            raise ValueError("No sufficiently long RedPajama document in sampling scope.")
        start = rng.randint(0, tokens.shape[1] - args.seqlen - 1)
        samples.append(tokens[:, start:start + args.seqlen])
    ids = torch.cat(samples)
    source.update(seed=args.seed, nsamples=args.nsamples, seqlen=args.seqlen,
                  dataset_fingerprint=data._fingerprint,
                  input_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())
    return ids, source
