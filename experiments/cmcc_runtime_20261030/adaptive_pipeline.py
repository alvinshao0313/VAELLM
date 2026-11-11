"""Rebind migrated adaptive inputs, then run the unchanged comparison protocol."""
import argparse
from functools import partial
from pathlib import Path

from .paths import ARTIFACTS, MODELS, resolve_legacy_path


def rebind_paths(value):
    if isinstance(value, dict):
        return {key: rebind_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rebind_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(rebind_paths(item) for item in value)
    if isinstance(value, str) and value.startswith(('/data/star/', '/home/star/models/', '/home/star/VAELLM/')):
        return str(resolve_legacy_path(value))
    return value


def prepare(output):
    import torch
    from experiments.adaptive_native_model_20261029 import checkpoint, recovery
    from experiments.adaptive_native_model_20261029 import archive_cat_gate_prefixes as archive
    from experiments.adaptive_vector_rvq import common as c

    c.require(not output.exists(), 'Prepare migration identity in a fresh directory.')
    c.require(not torch.cuda.is_initialized(), 'Migration preparation requires hidden CUDA.')
    source = resolve_legacy_path(recovery.PREPARED)
    gate = resolve_legacy_path(recovery.ARCHIVE)
    base = resolve_legacy_path(archive.BASE)
    original = c.read(source)
    raw_runtime = torch.load(gate / 'cat_runtime_state.pt', map_location='cpu', weights_only=True)
    runtime = rebind_paths(raw_runtime)
    native_argv = rebind_paths(original['arms']['B']['native_argv'])
    output.mkdir(parents=True)
    c.write(output / 'structure_prepared.json', dict(arms={'B': dict(native_argv=native_argv)},
        migration_source_prepared=str(source), migration_source_sha256=c.sha(source)))
    torch.save(runtime, output / 'cat_runtime_state.pt')
    pipeline = setup(output, verify=False)
    cat, hf, training, vae, cfgs, loaded_runtime, stage = recovery.recipe(output / 'recipe_probe')
    c.require(vae.model_path == str(MODELS / 'Qwen/Qwen3-8B') and cat.seed == 31
        and loaded_runtime['runtime_identity'] == runtime['runtime_identity'], 'Migration changed the original recipe identity.')
    meta = c.read(base / 'checkpoint_meta.json')
    c.require(meta['checkpoint_id'] == archive.BASE_ID, 'Migrated common after_o UUID changed.')
    artifact_roots = ARTIFACTS / 'adaptive_class_20261030'
    artifact_counts = {}
    reports = []
    for phase, names in (('short', ('short01', 'short36')),
                         ('formal', ('formal_gpu1', 'formal_gpu2', 'formal_gpu3'))):
        roots = [artifact_roots / name for name in names]
        reports.extend(root / 'report.json' for root in roots)
        artifact_counts[phase] = {arm: len(pipeline.flow.layer_artifacts(roots, arm, short=phase == 'short'))
                                 for arm in ('dynamic', 'rice')}
    source_files = (source, gate / 'manifest.json', gate / 'cat_runtime_state.pt', gate / 'raw_checkpoint_meta.json',
                    gate / 'raw_config.json', base / 'checkpoint_meta.json', *reports)
    current_files = (output / 'structure_prepared.json', output / 'cat_runtime_state.pt')
    source_hashes = {str(path.resolve()): c.sha(path) for path in source_files}
    code_hashes = {str(path.resolve()): c.sha(path) for path in
        (Path(__file__), Path(__file__).with_name('paths.py'), Path(recovery.__file__), Path(checkpoint.__file__))}
    record = dict(status='cpu_verified_migrated_recipe', exact_resume_implemented=False,
        original_source_files_sha256=source_hashes,
        prepared_files_sha256={str(path.resolve()): c.sha(path) for path in current_files},
        source_sha256=code_hashes, source_after_o_checkpoint_id=archive.BASE_ID,
        source_after_o_weight_stat=c.stat(base / meta['state_dict_file']),
        actual_artifact_counts=artifact_counts,
        actual_raw_archive=str(gate), model_path=vae.model_path,
        original_runtime_identity_unchanged=raw_runtime['runtime_identity'] == runtime['runtime_identity'],
        migration='Only command/input path strings rebound; historical JSON, metadata and runtime files unchanged.',
        recovery_steps=stage.config.opt.steps, recovery_batch_size=stage.config.opt.batch_size,
        recovery_model_max_length=stage.config.data.model_max_length,
        cuda_initialized=torch.cuda.is_initialized())
    c.write(output / 'prepared.json', record)
    print(dict(status=record['status'], output=str(output)), flush=True)


def setup(prepared, *, verify=True):
    from experiments.adaptive_native_model_20261029 import recovery
    from experiments.adaptive_native_model_20261029 import archive_cat_gate_prefixes as archive
    from experiments.adaptive_class_20261030 import continuation, run_pipeline, sources
    from experiments.adaptive_vector_rvq import common as c

    if verify:
        record = c.read(prepared / 'prepared.json')
        c.require(record['status'] == 'cpu_verified_migrated_recipe', 'Use the new-machine verified preparation.')
        for inventory in ('original_source_files_sha256', 'prepared_files_sha256', 'source_sha256'):
            for path, digest in record[inventory].items():
                c.require(c.sha(path) == digest, 'Migrated input/code changed: ' + path)
        c.require(c.stat(record['source_after_o_weight_stat']['path']) == record['source_after_o_weight_stat'],
            'Migrated after_o weight identity changed.')
    archive.BASE = resolve_legacy_path(archive.BASE)
    recovery.BASE = archive.BASE
    recovery.ARCHIVE = resolve_legacy_path(recovery.ARCHIVE)
    recovery.PREPARED = prepared / 'structure_prepared.json'
    recovery.RUNTIME_STATE = prepared / 'cat_runtime_state.pt'
    sources.BASE = archive.BASE
    sources.ARCHIVE = recovery.ARCHIVE
    sources.OUTPUT = ARTIFACTS / 'adaptive_class_cmcc_20261030'
    c.HF = MODELS / 'Qwen/Qwen3-8B'
    continuation.layer_artifacts = partial(continuation.layer_artifacts, resolve_path=resolve_legacy_path)
    run_pipeline.ENTRY_MODULE = 'experiments.cmcc_runtime_20261030.adaptive_pipeline'
    run_pipeline.ENTRY_ARGUMENTS = ('--prepared', str(prepared))
    run_pipeline.EXTRA_SOURCE_PATHS = (Path(__file__), Path(__file__).with_name('paths.py'), prepared / 'prepared.json',
        prepared / 'structure_prepared.json', prepared / 'cat_runtime_state.pt')
    return run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--prepared', type=Path)
    args, remaining = parser.parse_known_args()
    if args.prepare:
        if remaining or args.prepared:
            parser.error('--prepare does not take pipeline arguments.')
        prepare(args.prepare.resolve())
    else:
        if args.prepared is None:
            parser.error('--prepared is required to run or fresh-load.')
        pipeline = setup(args.prepared.resolve())
        pipeline.main(remaining)


if __name__ == '__main__':
    main()
