"""Preserve A/B's actual pre-recovery raw36gate states as partial archives."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.adaptive_native_model_20261029 import native_prefix as prefix
from experiments.adaptive_native_model_20261029.consumer import CompactRVQLinear
from experiments.adaptive_native_model_20261029.validate_native_prefix import check_fields, numerical
from experiments.adaptive_vector_rvq import common as c
import torch

BASE = Path('/data/star/VAELLM/qwen_recompression_20261026/C/initial_after_o')
BASE_ID = 'b63bfe79-3889-40c6-bf66-1d002893f874'
RAW_ROOT = Path('/data/star/VAELLM/cat_structure_full_20261029')
RAW_SUFFIX = 'native/home_star_models_Qwen_Qwen3-8B_20261029_134231/training_rounds/0004_gate_proj/round_base'
ARMS = {'A': dict(checkpoint_id='94c8c25b-19dd-4c6d-852c-19f5b12c4d6b', dim=32, bits=32, bytes=476068608),
        'B': dict(checkpoint_id='7c0b1199-b9ff-46af-a1b7-c2f5b7dabf7f', dim=64, bits=64, bytes=477252864)}
COMPLETED = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
SCHEMA = 'cat_gate_raw_prefix_archive_v1'


def event(name, **fields):
    print(json.dumps(dict(event=name, **fields)), flush=True)


def subset(state, names, keep=True):
    starts = tuple(name + '.' for name in names)
    return {name: tensor for name, tensor in state.items() if name.startswith(starts) == keep}


def tensor_table(state):
    return {name: dict(dtype=str(tensor.dtype), shape=list(tensor.shape), bytes=tensor.numel() * tensor.element_size())
            for name, tensor in state.items()}


def equal_fields(actual, expected):
    differences = [dict(name=name, difference='missing_from_actual') for name in sorted(set(expected) - set(actual))]
    differences += [dict(name=name, difference='unexpected_in_actual') for name in sorted(set(actual) - set(expected))]
    for name in sorted(set(actual) & set(expected)):
        left, right = actual[name], expected[name]
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            differences.append(dict(name=name, difference='dtype_shape_or_value', actual_dtype=str(left.dtype),
                expected_dtype=str(right.dtype), actual_shape=list(left.shape), expected_shape=list(right.shape)))
    return differences


def binding(path):
    return dict(**c.stat(path), sha256=c.sha(path))


def load_archive(directory):
    """Read only the partial gate payload and its untouched source metadata."""
    directory = Path(directory)
    manifest = c.read(directory / 'manifest.json')
    c.require(manifest['schema'] == SCHEMA and not manifest['independently_loadable_model'],
              'This is a partial raw-gate archive, never an independently loadable native model.')
    for filename, item in manifest['files'].items():
        path = directory / filename
        c.require(not path.is_symlink() and path.stat().st_size == item['bytes'] and c.sha(path) == item['sha256'],
                  'Archived actual payload/metadata/runtime changed: ' + str(path))
    raw_meta = c.read(directory / 'raw_checkpoint_meta.json')
    c.require(raw_meta['checkpoint_id'] == manifest['source_raw']['checkpoint_id'], 'Archived original raw metadata identity changed.')
    specs = [item for item in raw_meta['converted_modules'] if item['name'] in manifest['gate_names']]
    state = torch.load(directory / 'gate_state.pt', map_location='cpu', weights_only=True, mmap=True)
    c.require(len(specs) == 36 and len(state) == 468 and tensor_table(state) == manifest['gate_tensor_table']
              and c.digest(state) == manifest['source_gate_tensor_digest'], 'Fresh partial gate state differs from its actual source fields.')
    return state, specs, manifest


def base_source():
    meta = c.read(BASE / 'checkpoint_meta.json')
    c.require(meta['checkpoint_id'] == BASE_ID and meta['completed_categories'] == COMPLETED
              and len(meta['compressed_targets']) == 144 and len(meta['pending_dense_targets']) == 108,
              'Use the assigned real common after_o source.')
    state = torch.load(BASE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    names = [name for name in meta['pending_dense_targets'] if name.endswith('.gate_proj')]
    targets = subset(state, names)
    c.require(len(names) == 36 and set(targets) == {name + '.weight' for name in names},
              'The common source must contain exactly the 36 actual dense gate target weights.')
    return meta, state, names, targets


def archive_one(arm, output, base_state, names, base_binding):
    source = RAW_ROOT / arm / RAW_SUFFIX
    expected = ARMS[arm]
    meta = c.read(source / 'checkpoint_meta.json')
    c.require(meta['checkpoint_id'] == expected['checkpoint_id'] and meta['checkpoint_kind'] == 'round_base'
        and meta['completed_categories'] == COMPLETED and len(meta['compressed_targets']) == 180
        and len(meta['pending_dense_targets']) == 72 and meta['converted_module_count'] == 180
        and 'distill_stage' in meta['extra_meta'] and meta['extra_meta']['distill_stage'] is None,
        'Use only the actual raw36gate boundary saved before its500 recovery.')
    specs = [item for item in meta['converted_modules'] if item['name'].endswith('.gate_proj')]
    c.require(len(specs) == 36 and {item['name'] for item in specs} == set(names), 'Raw/base gate inventories differ.')
    for spec in specs:
        c.require(spec['residual_stages'] == 2 and spec['stage_codebook_dims'] == [expected['dim']] * 2
            and all(stage['in_dim'] == expected['bits'] and stage['out_dim'] == expected['dim']
                    and stage['decoder_type'] == 'asymmetric' for stage in spec['stage_decoders']),
            'The archived actual gate must retain this arm\'s real asymmetric two-stage D/K.')
    raw_weight = source / meta['state_dict_file']
    source_binding = dict(checkpoint_id=meta['checkpoint_id'], directory=str(source),
        metadata=binding(source / 'checkpoint_meta.json'), config=binding(source / 'config.json'),
        runtime=binding(source / 'cat_runtime_state.pt'), weight_stat=c.stat(raw_weight))
    state = torch.load(raw_weight, map_location='cpu', weights_only=True, mmap=True)
    original_gate = subset(state, names)
    c.require(len(original_gate) == 468 and all(len(subset(original_gate, [name])) == 13 for name in names),
              'Preserve exactly all13 native fields of each of36 gate projections.')
    gate = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in original_gate.items()}
    payload_bytes = sum(c.compact(tensor) for tensor in gate.values())
    c.require(payload_bytes == expected['bytes'] and not equal_fields(gate, original_gate), 'Actual extracted gate fields/bytes changed.')
    source_gate_digest = c.digest(original_gate)
    c.require(c.digest(gate) == source_gate_digest, 'Extraction must preserve the complete original gate tensor digest.')
    directory = output / arm
    directory.mkdir()
    # Save the irreplaceable small raw state before optional consumer work.
    with (directory / 'gate_state.pt').open('xb') as stream:
        torch.save(gate, stream)
    for original, saved in (('checkpoint_meta.json', 'raw_checkpoint_meta.json'), ('config.json', 'raw_config.json'),
                            ('cat_runtime_state.pt', 'cat_runtime_state.pt')):
        with (directory / saved).open('xb') as destination, (source / original).open('rb') as origin:
            shutil.copyfileobj(origin, destination)
    files = {name: dict(bytes=(directory / name).stat().st_size, sha256=c.sha(directory / name))
             for name in ('gate_state.pt', 'raw_checkpoint_meta.json', 'raw_config.json', 'cat_runtime_state.pt')}
    for filename, key in (('raw_checkpoint_meta.json', 'metadata'), ('raw_config.json', 'config'), ('cat_runtime_state.pt', 'runtime')):
        c.require(files[filename]['sha256'] == source_binding[key]['sha256'], 'Original metadata/config/runtime copy must remain byte-exact.')
    manifest = dict(schema=SCHEMA, arm=arm, independently_loadable_model=False, source_raw=source_binding,
        common_after_o=base_binding, gate_names=names, gate_tensor_table=tensor_table(gate),
        source_gate_tensor_digest=source_gate_digest, archived_gate_tensor_digest=c.digest(gate), files=files,
        payload_tensor_bytes=payload_bytes, serialization_overhead_bytes=files['gate_state.pt']['bytes'] - payload_bytes,
        source_completed_categories=meta['completed_categories'], source_distill_stage=meta['extra_meta']['distill_stage'],
        gate_spec_location='raw_checkpoint_meta.json: actual converted_modules entries named by gate_names',
        retained_use='Scheduled same-parent adaptive-vs-Rice raw2 comparison; original raw runtime for subsequent500 recovery/continued live categories.',
        boundary='Partial archive of original raw36gate native fields. Reconstruct model state with the pinned after_o non-gate complement. No new full native schema, end-model, optimizer resume or quality result is claimed.')
    c.write(directory / 'manifest.json', manifest)
    raw_complement, base_complement = subset(state, names, False), subset(base_state, names, False)
    differences = equal_fields(raw_complement, base_complement)
    if differences:
        manifest.update(after_o_complement_exact=False, complement_differences=differences)
        c.write(directory / 'manifest.json', manifest)
        raise ValueError(f'{arm}: raw non-gate fields differ from after_o; saved gate archive retained, no equivalence claimed: {differences[:8]}')
    manifest.update(after_o_complement_exact=True, non_gate_fields_compared=len(raw_complement),
        source_raw_state_digest=c.digest(state), source_non_gate_tensor_digest=c.digest(raw_complement),
        source_raw_tensor_count=len(state), source_raw_tensor_bytes=sum(value.numel() * value.element_size() for value in state.values()))
    c.require(manifest['source_non_gate_tensor_digest'] == base_binding['non_gate_tensor_digest'],
              'Exact fields and common non-gate digest disagree.')
    raw_config, initial_config = c.read(source / 'config.json'), c.read(BASE / 'config.json')
    manifest['raw_vs_after_o_config_differences'] = {key: dict(raw=raw_config.get(key), after_o=initial_config.get(key),
        raw_present=key in raw_config, after_o_present=key in initial_config)
        for key in sorted(set(raw_config) | set(initial_config)) if (key in raw_config) != (key in initial_config) or raw_config.get(key) != initial_config.get(key)}
    c.write(directory / 'manifest.json', manifest)
    first = specs[0]
    name = first['name']
    relative = {key[len(name) + 1:]: value for key, value in gate.items() if key.startswith(name + '.')}
    provenance = dict(source_raw_checkpoint_id=meta['checkpoint_id'], source_module=name,
        actual_archive=str(directory), purpose='Bounded real D/K asymmetric CPU consumer check; no live target inferred')
    bundle, lengths, resolved = prefix.from_native(first, relative, provenance)
    module = prefix.load_native(first, relative)
    consumer = CompactRVQLinear(bundle, lengths)
    fields = check_fields(module, relative, bundle, lengths, resolved)
    ids = torch.tensor([0, len(lengths) // 2, len(lengths) - 1], dtype=torch.long)
    values, sample_digest = numerical(module, consumer, ids)
    for key, filename in (('metadata', 'checkpoint_meta.json'), ('config', 'config.json'), ('runtime', 'cat_runtime_state.pt')):
        c.require(c.sha(source / filename) == source_binding[key]['sha256'], 'An active original raw source changed during archiving.')
    c.require(c.stat(raw_weight) == source_binding['weight_stat'], 'Original raw weights changed during archiving.')
    row = dict(archive=str(directory), source_raw_checkpoint_id=meta['checkpoint_id'], gate_tensor_count=468,
        payload_tensor_bytes=payload_bytes, actual_files={path.name: path.stat().st_size for path in directory.iterdir()},
        archive_total_bytes=sum(path.stat().st_size for path in directory.iterdir()),
        source_gate_tensor_digest=source_gate_digest, source_raw_state_digest=manifest['source_raw_state_digest'],
        after_o_non_gate_fields_exact=len(raw_complement), raw_vs_after_o_config_differences=manifest['raw_vs_after_o_config_differences'],
        representative=dict(module=name, fields=fields, numerical=values, compact_selected_digest=sample_digest),
        original_sources_unchanged=True, temporary_compact_bundle_persisted=False)
    event('raw_gate_archive_saved_parent_verified', arm=arm, tensors=468, payload_bytes=payload_bytes, non_gate_exact=len(raw_complement))
    return row


def fresh(output):
    reads = []
    torch_load = torch.load
    allowed = {str((BASE / 'pytorch_model.bin').resolve()),
               *(str((output / arm / 'gate_state.pt').resolve()) for arm in ARMS)}

    def only_effective_sources(path, *args, **kwargs):
        actual = str(Path(path).resolve())
        c.require(actual in allowed, 'Independent reconstruction attempted an unassigned full raw weight read.')
        reads.append(actual)
        return torch_load(path, *args, **kwargs)

    torch.load = only_effective_sources
    base_meta, base_state, names, targets = base_source()
    rows = {}
    for arm in ARMS:
        gate, specs, manifest = load_archive(output / arm)
        c.require(c.stat(BASE / base_meta['state_dict_file']) == manifest['common_after_o']['weight_stat'], 'The retained common source weight identity changed.')
        reconstructed = {**subset(base_state, names, False), **gate}
        digest = c.digest(reconstructed)
        c.require(len(reconstructed) == manifest['source_raw_tensor_count'] and digest == manifest['source_raw_state_digest'],
                  'Independent after_o + actual archived raw36gate does not reconstruct the original raw tensor state.')
        rows[arm] = dict(status='independent_partial_archive_verified', gate_tensors=len(gate), gate_specs=len(specs),
            archived_gate_tensor_digest=c.digest(gate), reconstructed_raw_state_digest=digest,
            reconstructed_raw_tensor_count=len(reconstructed), runtime_byte_exact=True,
            effective_weight_sources=[str(BASE / base_meta['state_dict_file']), str(output / arm / 'gate_state.pt')],
            raw_full_weight_file_read=False, full_model_constructed=False)
        event('independent_partial_archive_verified', arm=arm, gate_tensors=len(gate), raw_tensors=len(reconstructed))
        del gate, reconstructed
        gc.collect()
    c.write(output / 'fresh.json', dict(status='independent_A_B_raw_state_reconstruction_verified', pid=os.getpid(), arms=rows,
        cuda_initialized=torch.cuda.is_initialized(), common_target_tensor_count=len(targets),
        common_target_digest=c.digest(targets), actual_weight_reads=reads))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU archive only; hide CUDA before imports.')
    torch.set_num_threads(8)
    output = args.output.resolve()
    if args.fresh:
        fresh(output)
        return
    c.require(not output.exists(), 'Never overwrite an archive, existing model or prior evidence.')
    output.mkdir(parents=True)
    report = dict(status='running', schema=SCHEMA, pid=os.getpid(), command=[sys.executable, *sys.argv],
        started_unix=time.time(), cuda_initialized=False, arms={},
        retained_use='Scheduled same-parent raw gate adaptive/Rice comparison, native500 recovery and continued live categories; preserve common after_o.')
    c.write(output / 'report.json', report)
    try:
        base_meta, base_state, names, targets = base_source()
        base_binding = dict(checkpoint_id=BASE_ID, directory=str(BASE), metadata=binding(BASE / 'checkpoint_meta.json'),
            config=binding(BASE / 'config.json'), weight_stat=c.stat(BASE / base_meta['state_dict_file']),
            dense_gate_target_fields=list(targets), dense_gate_target_tensor_bytes=sum(value.numel() * value.element_size() for value in targets.values()),
            dense_gate_target_tensor_digest=c.digest(targets), non_gate_tensor_digest=c.digest(subset(base_state, names, False)),
            retained_use='Actual shared dense gate target and all non-gate reconstruction state for the scheduled same-parent adaptive/Rice comparison.')
        report['common_after_o'] = base_binding
        c.write(output / 'report.json', report)
        for arm in ARMS:
            report['arms'][arm] = archive_one(arm, output, base_state, names, base_binding)
            c.write(output / 'report.json', report)
            gc.collect()
        subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--output', str(output), '--fresh'], cwd=ROOT, check=True)
        verified = c.read(output / 'fresh.json')
        c.require(not verified['cuda_initialized'] and verified['common_target_digest'] == base_binding['dense_gate_target_tensor_digest'],
                  'Independent raw-state reconstruction changed the retained real dense targets or initialized CUDA.')
        c.require(c.stat(BASE / base_meta['state_dict_file']) == base_binding['weight_stat']
            and c.sha(BASE / 'checkpoint_meta.json') == base_binding['metadata']['sha256'], 'Common retained after_o source changed.')
        report.update(status='A_B_raw_gate_archives_saved_independently_verified', exit_code=0, independent_fresh=verified,
            source_sha256={str(path): c.sha(path) for path in (Path(__file__), Path(prefix.__file__),
                Path(__file__).with_name('spec.py'), Path(__file__).with_name('consumer.py'), Path(__file__).with_name('validate_native_prefix.py'))},
            full_original_models_copied=False, dense_targets_copied=False, raw_sources_deleted=False,
            partial_archive_not_independent_model=True,
            boundary='Tensor-state reconstruction from after_o + raw36gate is exact; raw metadata/config/runtime are retained verbatim. This does not assert full runtime/generation semantics or optimizer/RNG resume equivalence.')
    except BaseException:
        report.update(status='failed_archives_kept', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - report['started_unix'])
        c.write(output / 'report.json', report)
    event(report['status'], seconds=report['elapsed_seconds'])


if __name__ == '__main__':
    main()
