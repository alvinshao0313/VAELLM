"""Fetch and verify the pinned, original eleven Arrow shards used by LiftQuant."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset
import requests
import torch
from transformers import AutoTokenizer

from experiments.liftquant_recovery.recovery_data import DATASET, calibration

REVISION = '4b6d76ca56b821e4f2110204943b7a67927b565c'
ROOT = 'https://hf-mirror.com'
PREFIX = 'plain_text/1.0.0/6ea3bc8ec2e84ec6d2df1930942e9028ace8c5b9d9143823cf911c50bbd92039/'


def fetch_file(spec, destination):
    name = Path(spec['rfilename']).name
    target = destination / name
    temporary = target.with_name(target.name + '.part')
    if target.exists() or temporary.exists():
        raise FileExistsError(f'Refusing to overwrite existing data: {target}')
    url = f'{ROOT}/datasets/{DATASET}/resolve/{REVISION}/{spec["rfilename"]}'
    digest = hashlib.sha256()
    git_digest = hashlib.sha1(f'blob {spec["size"]}\0'.encode())
    size = 0
    print(f'Downloading {name}: expected {spec["size"]} bytes', flush=True)
    with requests.get(url, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        with temporary.open('xb') as stream:
            for chunk in response.iter_content(4 * 1024**2):
                if not chunk:
                    continue
                stream.write(chunk)
                size += len(chunk)
                digest.update(chunk)
                git_digest.update(chunk)
    if size != spec['size']:
        raise ValueError(f'{name}: size mismatch {size} != {spec["size"]}')
    lfs = spec.get('lfs')
    if lfs and digest.hexdigest() != lfs['sha256']:
        raise ValueError(f'{name}: upstream LFS SHA256 mismatch')
    if not lfs and git_digest.hexdigest() != spec['blobId']:
        raise ValueError(f'{name}: upstream Git blob checksum mismatch')
    temporary.rename(target)
    rows = len(Dataset.from_file(str(target))) if name.endswith('.arrow') else None
    result = dict(file=name, bytes=size, sha256=digest.hexdigest(), rows=rows,
                  upstream_path=spec['rfilename'], url=url,
                  upstream_lfs_sha256=lfs['sha256'] if lfs else None,
                  upstream_git_blob=spec['blobId'])
    print(f'Verified {name}: {size} bytes, rows={rows}, SHA256={digest.hexdigest()}', flush=True)
    return result


def run(args):
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    api = f'{ROOT}/api/datasets/{DATASET}/revision/{REVISION}?blobs=true'
    response = requests.get(api, timeout=(30, 60))
    response.raise_for_status()
    repository = response.json()
    if repository['sha'] != REVISION:
        raise ValueError('Dataset revision does not match the pinned source.')
    names = ['dataset_info.json'] + [f'red_pajama-data-1_t-sample-train-{i:05d}-of-00011.arrow' for i in range(11)]
    siblings = {item['rfilename']: item for item in repository['siblings']}
    specs = [siblings[PREFIX + name] for name in names]
    print(json.dumps(dict(dataset=DATASET, revision=REVISION, files=len(specs),
                          expected_bytes=sum(item['size'] for item in specs))), flush=True)
    with ThreadPoolExecutor(max_workers=2) as workers:
        files = list(workers.map(lambda spec: fetch_file(spec, destination), specs))
    info = json.loads((destination / 'dataset_info.json').read_text())
    rows = sum(item['rows'] or 0 for item in files)
    if info['dataset_name'] != 'red_pajama-data-1_t-sample' or rows != 930514 or rows != info['splits']['train']['num_examples']:
        raise ValueError(f'Unexpected dataset identity or row count: {rows}')
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False, local_files_only=True)
    config = SimpleNamespace(smoke_rows=None, redpajama_arrow_dir=str(destination),
                             nsamples=4096, seqlen=2048, seed=42)
    print('Verifying existing production calibration: 4096 x 2048, seed 42.', flush=True)
    ids, metadata = calibration(config, tokenizer)
    if tuple(ids.shape) != (4096, 2048):
        raise ValueError(f'Unexpected calibration shape: {tuple(ids.shape)}')
    torch.save(ids, destination / 'calibration_4096x2048_seed42.pt')
    manifest = dict(status='PASS', dataset=DATASET, revision=REVISION,
                    upstream_api=api, rows=rows, original_files=files,
                    original_bytes=sum(item['bytes'] for item in files),
                    calibration=metadata,
                    calibration_ids='calibration_4096x2048_seed42.pt',
                    sampling='Existing production calibration: shuffle(seed=42), replacement document draw, uniform eligible window.',
                    protocol_changed=False)
    (destination / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', directory=str(destination), rows=rows,
                          calibration_shape=list(ids.shape), token_sha256=metadata['input_sha256'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New dataset directory; existing files are never overwritten.')
    parser.add_argument('--checkpoint', required=True, help='Existing native checkpoint tokenizer for exact calibration verification.')
    run(parser.parse_args())
