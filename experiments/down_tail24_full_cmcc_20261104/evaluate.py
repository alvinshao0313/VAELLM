"""Strict local raw-stage provenance, with unchanged complete full8 consumption."""
import argparse
import importlib
from pathlib import Path
import sys
import time

from . import bindings as b
from experiments.adaptive_class_evaluation_20261030 import bindings as e
from experiments.cmcc_runtime_20261030 import adaptive_evaluation as cmcc


def inspect_topology(meta, budget, arm):
    b.c.require(meta['checkpoint_kind'] == 'mixed_final_model' and len(meta['native_modules']) == 180
        and len(meta['compact_modules']) == 72 and not meta['pending_dense_modules']
        and budget['budget_passed'] and budget['final_budget_applicable'] and budget['user_linear_bpw'] <= 2.6,
        'This stage requires actual180native+72compact+0pending and full logical<=2.6.')
    gates = {name.replace('down_proj', 'gate_proj') for name in b.NAMES}
    b.c.require({item['name'] for item in meta['compact_modules']} == set(b.NAMES) | gates,
        'The72 compact modules must be exactly36 unchanged gates and36 downs.')
    for item in meta['compact_modules']:
        if item['name'] not in b.NAMES:
            continue
        spec = item['spec']
        b.c.require(len(spec['stages']) == (2 if arm == 'control' else 3)
            and spec['vector_dim'] == 16 and spec['n_vectors'] == b.N
            and all(row['decoder_spec']['in_dim'] == 24 for row in spec['stages']), 'Actual down topology differs from its arm.')


def binding(directory, pipeline_report):
    directory, path = Path(directory).resolve(), Path(pipeline_report).resolve()
    record = b.c.read(path)
    arm = record.get('arm')
    expected = {'control': 'completed_full_compact2_control', 'k24': 'completed_full_K24_raw'}
    b.c.require(record.get('schema') == b.SCHEMA and record.get('phase') == 'formal' and arm in expected
        and record.get('status') == expected[arm] and record.get('exit_code') == 0
        and directory == path.parent / 'model' and directory.is_relative_to(b.OUTPUT.resolve()),
        'Reject CPU fixtures, partial fits, old formal/500 reports and external checkpoints.')
    prepared = b.c.read(b.PREPARED)
    b.check_pins(prepared['source_sha256'])
    b.c.require(record['source_binding'] == prepared['source_binding'] and record['source_sha256'] == prepared['source_sha256']
        and record['all_frozen_fields_exact'], 'The raw-stage source, static code pins or frozen complement differs.')
    if arm == 'k24':
        fits = record['group_training']
        b.c.require(len(fits) == 12 and [row['names'] for row in fits] == b.GROUPS
            and all(row['actual_updates'] == row['schedule_steps'] == 10000 and row['warmup_steps'] == 0
                and row['batch_shape'] == [16384, 3, 16] and row['init_seed_per_layer'] == row['shuffle_seed_per_layer'] == 2031
                and row['actual_update_checks'][-1]['actual_step'] == 10000 for row in fits),
            'Require the completed fresh36x10000 stage, never a short bank or partial report.')
    meta, budget, hashes = e.inspect(directory, final=False)
    inspect_topology(meta, budget, arm)
    b.c.require(meta['checkpoint_id'] == record['checkpoint_id'] and meta['provenance'] == record['provenance']
        and meta['provenance']['schema'] == b.SCHEMA and meta['provenance']['phase'] == 'formal'
        and meta['provenance']['source_raw_id'] == b.RAW_ID and meta['provenance']['live_target_id'] == b.LIVE_ID
        and meta['provenance']['actual_stage3_updates_per_layer'] == (0 if arm == 'control' else 10000),
        'Actual UUID/storage provenance is not this completed raw-stage arm.')
    strict_path = path.parent / 'fresh_report.json'
    strict = b.c.read(strict_path)
    b.c.require(strict['status'] == 'strict_full_state_and_BF16_logits_verified' and strict['exit_code'] == 0
        and strict['checkpoint_id'] == meta['checkpoint_id'] and strict['same_input_logits_exact']
        and strict['persistent_fields'] == len(meta['tensor_table'])
        and strict['effective_weight_files'] == [str(directory / b.checkpoint.STATE_FILE)]
        and all(strict['actual_budget'][key] == budget[key] for key in e.BUDGET_FIELDS),
        'The actual complete model needs its own successful independent same-path fresh proof.')
    hashes.update({str(path): b.c.sha(path), str(strict_path): b.c.sha(strict_path)})
    return dict(checkpoint_dir=str(directory), checkpoint_id=meta['checkpoint_id'], checkpoint_format=e.FORMAT,
        metadata_sha256=hashes[str(directory / b.checkpoint.META_FILE)], checkpoint_files_sha256=hashes,
        checkpoint_weight_stat=e.file_stat(directory / b.checkpoint.STATE_FILE), bit_budget=budget,
        pipeline_report=str(path), pipeline_provenance=meta['provenance']), meta


def setup():
    cmcc.setup(b.SUPPORT)
    previous = e.source_pins
    e.source = binding
    e.source_pins = lambda value: {**previous(value), **b.c.read(b.PREPARED)['source_sha256']}


def full8(root, arm):
    setup()
    directory = root / arm / 'model'
    bound, _meta = binding(directory, root / arm / 'report.json')
    pins = e.source_pins(bound)
    config = e.make_config(bound, pins)
    output = root / arm / 'full8'
    b.c.require(not output.exists(), 'Do not overwrite or automatically repeat an evaluated arm.')
    output.mkdir()
    b.c.write(output / 'config.json', config)
    return output, config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=('mmlu', 'seven', 'summary'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=('control', 'k24'), required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    args = parser.parse_args()
    b.gpu_environment()
    b.c.require(time.time() < args.deadline_unix, 'The one common4h deadline already expired.')
    setup()
    output = args.root.resolve() / args.arm / 'full8'
    config = b.c.read(output / 'config.json')
    # Recheck the immutable raw/strict report before each actual shard.
    binding(args.root.resolve() / args.arm / 'model', args.root.resolve() / args.arm / 'report.json')
    b.torch.set_num_threads(8)
    b.torch.backends.cuda.matmul.allow_tf32 = False
    b.torch.backends.cudnn.allow_tf32 = False
    if args.component == 'summary':
        from experiments.adaptive_class_evaluation_20261030.summarize import full8_summary
        full8_summary(config, [output / 'mmlu.json', output / 'seven.json'], output / 'full8_summary.json')
    else:
        sys.argv = [sys.argv[0], '--config', str(output / 'config.json'), '--shard', args.component,
            '--output', str(output / (args.component + '.json'))]
        importlib.import_module('experiments.adaptive_class_evaluation_20261030.full8').main()
    b.c.require(time.time() < args.deadline_unix, 'The one common4h deadline expired during the shard.')
