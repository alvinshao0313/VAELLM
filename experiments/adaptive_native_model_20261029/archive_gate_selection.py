"""Preserve real CAT gate250/500 mutable endpoints for scheduled read-only pairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
CAT = Path('/data/star/VAELLM/cat_structure_full_20261029')
RAW = Path('/data/star/VAELLM/adaptive_native_model_20261029/cat_gate_prefixes')
AFTER_O = Path('/data/star/VAELLM/qwen_recompression_20261026/C/initial_after_o')
SUFFIX = 'native/home_star_models_Qwen_Qwen3-8B_20261029_134231/training_rounds/0004_gate_proj'
SCHEMA = 'cat_gate_selection_partial_archive_v1'
ARMS = {'A': ('94c8c25b-19dd-4c6d-852c-19f5b12c4d6b', 70518914),
        'B': ('7c0b1199-b9ff-46af-a1b7-c2f5b7dabf7f', 72887426)}
IDS = {('A', 250): '6b99b752', ('A', 500): 'ffef68e6',
       ('B', 250): 'c2d2ca9e', ('B', 500): 'e8a08b0f'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    path = Path(path)
    info = path.stat()
    return dict(path=str(path), bytes=info.st_size, sha256=sha(path))


def copy_original(source, destination):
    """Opening the immutable source protects the small file from concurrent prune."""
    digest = hashlib.sha256()
    with source.open('rb') as origin, destination.open('xb') as target:
        before = os.fstat(origin.fileno())
        while chunk := origin.read(4 * 1024 * 1024):
            digest.update(chunk)
            target.write(chunk)
        after = os.fstat(origin.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Completed source changed during archive: ' + str(source))
    result = dict(path=str(source), bytes=before.st_size, inode=before.st_ino,
                  mtime_ns=before.st_mtime_ns, sha256=digest.hexdigest())
    if destination.stat().st_size != result['bytes'] or sha(destination) != result['sha256']:
        raise ValueError('Original byte copy mismatch: ' + str(destination))
    return result


def check_identity(meta, trainer, arm, step, raw_meta):
    if not (meta['format'] == 'vaellm_model_checkpoint_v6' and meta['schema_version'] == 6
            and meta['checkpoint_kind'] == 'training_step'
            and meta['checkpoint_id'].startswith(IDS[arm, step])
            and meta['round_base_checkpoint_id'] == ARMS[arm][0]
            and raw_meta['checkpoint_id'] == ARMS[arm][0]
            and trainer['global_step'] == step and trainer['max_steps'] == 500
            and meta['after_category_mode'] == 'remaining_lora_prefix_decoder'):
        raise ValueError(f'{arm}{step}: wrong completed training endpoint or raw parent')
    if len(meta['compressed_targets']) != 180 or len(meta['pending_dense_targets']) != 72:
        raise ValueError(f'{arm}{step}: unexpected original inventory')
    if set(meta['compressed_targets']) != set(raw_meta['compressed_targets']):
        raise ValueError(f'{arm}{step}: compressed inventory differs from raw parent')
    if set(meta['pending_dense_targets']) != set(raw_meta['pending_dense_targets']):
        raise ValueError(f'{arm}{step}: future dense inventory differs from raw parent')


def digest_tensors(state):
    import torch
    aggregate = hashlib.sha256()
    table = {}
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
            raise ValueError('Mutable state must be CPU named tensors')
        tensor = value.detach().contiguous()
        data = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
        aggregate.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        aggregate.update(data)
        table[name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                           bytes=tensor.numel() * tensor.element_size(),
                           sha256=hashlib.sha256(data).hexdigest())
    return aggregate.hexdigest(), table


def inspect_payload(directory, meta):
    import torch
    state = torch.load(directory / 'training_model_state.pt', weights_only=True, map_location='cpu', mmap=True)
    entries = meta['mutable_state_manifest']
    names = [item['name'] for item in entries]
    if len(names) != len(set(names)) or set(state) != set(names):
        raise ValueError('Mutable keys differ from original manifest')
    components = {}
    for item in entries:
        tensor = state[item['name']]
        if list(tensor.shape) != item['shape'] or str(tensor.dtype).removeprefix('torch.') != item['dtype']:
            raise ValueError('Mutable shape/dtype differs from original manifest: ' + item['name'])
        label = item['component_class']
        components[label] = components.get(label, 0) + 1
    digest, table = digest_tensors(state)
    return dict(tensor_count=len(state), tensor_bytes=sum(item['bytes'] for item in table.values()),
                component_counts=components, tensor_digest=digest, tensor_table=table)


def fresh(output):
    rows = []
    # This subprocess reads only the partial archives, not their prunable sources.
    for arm in ARMS:
        for step in (250, 500):
            directory = output / arm / f'gate{step}'
            manifest = read(directory / 'manifest.json')
            if manifest['schema'] != SCHEMA or manifest['independently_loadable_model']:
                raise ValueError('Wrong partial archive schema')
            for filename, item in manifest['files'].items():
                path = directory / filename
                if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                    raise ValueError('Fresh archived file mismatch: ' + str(path))
            meta, trainer = read(directory / 'checkpoint_meta.json'), read(directory / 'trainer_state.json')
            if meta['checkpoint_id'] != manifest['checkpoint_id'] or trainer['global_step'] != step:
                raise ValueError('Fresh identity/step mismatch')
            actual = inspect_payload(directory, meta)
            if actual != manifest['mutable_state']:
                raise ValueError('Fresh tensor digest/table mismatch')
            rows.append(dict(arm=arm, step=step, checkpoint_id=meta['checkpoint_id'],
                             tensor_digest=actual['tensor_digest'], source_model_read=False))
    result = dict(status='fresh_partial_bytes_fields_exact', pid=os.getpid(), nodes=rows,
                  model_constructed=False, cuda_initialized=False)
    write(output / 'fresh.json', result)
    return result


def archive(output):
    started = time.time()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='archiving_original_small_endpoints', pid=os.getpid(), start_unix=started,
                  command=sys.argv, source_script=binding(Path(__file__)), nodes=[], exit_code=None,
                  environment=dict(python=sys.executable, conda_prefix=os.environ.get('CONDA_PREFIX'),
                                   CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    write(output / 'report.json', report)
    try:
        # Copy all irreplaceable files before importing torch or inspecting tensors.
        copies = {}
        for arm in ARMS:
            for step in (250, 500):
                source = CAT / arm / SUFFIX / 'trainer_state' / f'checkpoint-{step}'
                destination = output / arm / f'gate{step}'
                destination.mkdir(parents=True)
                originals = {}
                for filename in ('training_model_state.pt', 'checkpoint_meta.json', 'trainer_state.json'):
                    originals[filename] = copy_original(source / filename, destination / filename)
                originals['raw_config.json'] = copy_original(RAW / arm / 'raw_config.json', destination / 'raw_config.json')
                copies[arm, step] = originals
                print(json.dumps(dict(event='original_small_endpoint_preserved', arm=arm, step=step,
                                      bytes=sum(item['bytes'] for item in originals.values()))), flush=True)
        import torch
        torch.set_num_threads(4)
        if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('This task requires hidden CUDA and no CUDA initialization')
        for arm in ARMS:
            raw_directory = RAW / arm
            raw_manifest = read(raw_directory / 'manifest.json')
            raw_meta = read(raw_directory / 'raw_checkpoint_meta.json')
            if raw_manifest['source_raw']['checkpoint_id'] != ARMS[arm][0] or not raw_manifest['after_o_complement_exact']:
                raise ValueError('Wrong preserved raw parent archive')
            raw_files = {name: binding(raw_directory / name) for name in
                         ('manifest.json', 'raw_checkpoint_meta.json', 'raw_config.json')}
            for step in (250, 500):
                directory = output / arm / f'gate{step}'
                meta = read(directory / 'checkpoint_meta.json')
                trainer = read(directory / 'trainer_state.json')
                check_identity(meta, trainer, arm, step, raw_meta)
                if copies[arm, step]['training_model_state.pt']['bytes'] != ARMS[arm][1]:
                    raise ValueError('Unexpected original mutable file size')
                mutable = inspect_payload(directory, meta)
                files = {name: dict(bytes=item['bytes'], sha256=item['sha256']) for name, item in copies[arm, step].items()}
                manifest = dict(schema=SCHEMA, independently_loadable_model=False, arm=arm, step=step,
                    checkpoint_id=meta['checkpoint_id'], source_directory=str(CAT / arm / SUFFIX / 'trainer_state' / f'checkpoint-{step}'),
                    source_files=copies[arm, step], files=files, source_metadata_unchanged=True,
                    raw_parent_checkpoint_id=meta['round_base_checkpoint_id'], original_round_base_ref=meta['round_base_ref'],
                    preserved_raw_archive=dict(directory=str(raw_directory), files=raw_files,
                        source_raw_tensor_digest=raw_manifest['source_raw_state_digest'],
                        gate_state=raw_manifest['files']['gate_state.pt'], runtime=raw_manifest['files']['cat_runtime_state.pt']),
                    common_after_o=raw_manifest['common_after_o'], mutable_state=mutable,
                    trainer_global_step=trainer['global_step'], trainer_max_steps=trainer['max_steps'],
                    purpose='Scheduled read-only gate250/500 paired eight-task selection-candidate comparison; current A/B fixed500 continues unchanged.',
                    boundary='Partial immutable archive of actual mutable endpoint plus original metadata/trainer state and raw config. Rebuild from preserved after_o + corresponding raw36gate, then original PEFT/native export/merge rules. No independent model, merged-export, GPU, quality, next-category selection or optimizer resume is claimed.',
                    excluded_files=['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'full raw weight payload'])
                write(directory / 'manifest.json', manifest)
                report['nodes'].append(dict(arm=arm, step=step, checkpoint_id=meta['checkpoint_id'],
                    raw_parent_checkpoint_id=meta['round_base_checkpoint_id'], mutable_tensor_count=mutable['tensor_count'],
                    mutable_tensor_bytes=mutable['tensor_bytes'], component_counts=mutable['component_counts'],
                    tensor_digest=mutable['tensor_digest'], files=files,
                    archive_total_bytes=sum(path.stat().st_size for path in directory.iterdir())))
                write(output / 'report.json', report)
        process = subprocess.run([sys.executable, '-B', '-u', str(Path(__file__)), '--output', str(output), '--fresh'],
                                 check=True, cwd=ROOT)
        report.update(independent_fresh=read(output / 'fresh.json'), status='four_completed_gate_endpoints_preserved_and_fresh_exact',
                      exit_code=process.returncode, seconds=time.time() - started, cuda_initialized=torch.cuda.is_initialized(),
                      model_constructed=False, original_sources_modified=False,
                      archive_node_total_bytes=sum(row['archive_total_bytes'] for row in report['nodes']))
        write(output / 'report.json', report)
        (output / 'fresh.json').unlink()
        print(json.dumps(dict(status=report['status'], seconds=report['seconds'], bytes=report['archive_node_total_bytes'])), flush=True)
    except BaseException:
        report.update(status='failed_partial_archive_retained', exit_code=1, seconds=time.time() - started,
                      error=traceback.format_exc())
        write(output / 'report.json', report)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    if args.fresh:
        result = fresh(args.output)
        print(json.dumps(dict(status=result['status'], pid=result['pid'])), flush=True)
    else:
        archive(args.output)


if __name__ == '__main__':
    main()
