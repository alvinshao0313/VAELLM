"""Fetch the first Arrow record batch of the exact upstream RedPajama mirror."""
import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import requests

from experiments.liftquant_recovery.recovery_data import DATASET

ROOT = "https://hf-mirror.com"
repo = requests.get(f"{ROOT}/api/datasets/{DATASET}", timeout=30)
repo.raise_for_status()
info = repo.json()
paths = sorted(x["rfilename"] for x in info["siblings"]
               if "/red_pajama-data-1_t-sample-train-" in x["rfilename"] and x["rfilename"].endswith(".arrow"))
if not paths:
    raise ValueError("No original RedPajama train shards in the upstream dataset.")
url = f"{ROOT}/datasets/{DATASET}/resolve/{info['sha']}/{paths[0]}"
response = requests.get(url, headers={"Range": "bytes=0-67108863"}, stream=True, timeout=45)
response.raise_for_status()
# Limit the response even if the mirror ignores Range.
chunks, size = [], 0
for chunk in response.iter_content(1024 * 1024):
    chunks.append(chunk)
    size += len(chunk)
    if size >= 64 * 1024 * 1024:
        break
response.close()
raw = b"".join(chunks)
reader = pa.ipc.open_stream(io.BytesIO(raw))
batch = reader.read_next_batch()
rows = batch.slice(0, min(32, batch.num_rows)).to_pylist()
if not rows or any("text" not in row for row in rows):
    raise ValueError("Downloaded Arrow shard has no RedPajama text.")
payload = dict(dataset=DATASET, revision=info["sha"], source=url,
               scope="smoke-only first 32 rows of the original first train shard",
               prefix_sha256=hashlib.sha256(raw).hexdigest(),
               rows=[dict(row_idx=i, row=row) for i, row in enumerate(rows)])
destination = Path("experiments/liftquant_recovery/redpajama_smoke_rows.json")
with destination.open("x") as handle:
    json.dump(payload, handle, ensure_ascii=False)
print(json.dumps(dict(status="PASS", rows=len(rows), downloaded_bytes=size, destination=str(destination))))
