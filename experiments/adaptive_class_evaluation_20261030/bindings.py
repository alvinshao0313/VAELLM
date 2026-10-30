"""Real mixed-format identity/storage checks; no v6 metadata substitution."""
from __future__ import annotations

from copy import deepcopy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

import torch

from experiments.adaptive_native_model_20261029 import checkpoint, RUNTIME
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.longbench_v1.protocol import model_policy, protocol_record
from experiments.w25_eval.model_inventory import model_inventory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOPIC = ROOT / 'result/qwen3_8b_mmlu65_20261004'
OUTPUT = TOPIC / 'adaptive_class_evaluation_20261030'
PYTHON = Path('/home/star/miniconda3/envs/bitvae/bin/python')
HELPERS = TOPIC / 'qwen_aux_retention32_batch_20261011/evaluate.py'
SAMPLE_DIR = TOPIC / 'lm_eval_sample_artifacts_20261010'
REFERENCE = TOPIC / 'qwen_a0e5_baseline_s400_longbench21/config.json'
TEMPLATE = REFERENCE.with_name('merge_plan.json')
BASE = Path('/home/star/models/Qwen/Qwen3-8B')
FORMAT = checkpoint.FORMAT
SEVEN = 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa'
BUDGET_FIELDS = ('components', 'user_linear_bits', 'packed_linear_bits', 'outside_linear_bits',
                 'original_linear_weights', 'code_padding_bits', 'full_model_tensor_bytes',
                 'complete_directory_bytes', 'native_target_count', 'compact_target_count', 'pending_dense_target_count')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def module(path, name):
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(Path(path).parent))
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value
    finally:
        sys.path[:] = previous


def cpu_environment():
    require(Path(sys.executable).resolve() == PYTHON.resolve()
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'Use caller-activated bitvae with CUDA_VISIBLE_DEVICES empty for the CPU driver.')


def inspect(directory, *, final=True):
    directory = Path(directory).resolve()
    meta = read(directory / checkpoint.META_FILE)
    require(meta['format'] == FORMAT and meta['schema_version'] == 2
            and meta['state_dict_file'] == checkpoint.STATE_FILE, 'Require the actual mixed v2 format.')
    uuid.UUID(meta['checkpoint_id'])
    require(meta['runtime']['native_code_sha256'] == checkpoint.runtime_identity(), 'Native runtime binding changed.')
    files = list(directory.iterdir())
    require({path.name for path in files} == set(meta['files'])
            and all(path.is_file() and not path.is_symlink() for path in files), 'Require a self-contained complete checkpoint.')
    hashes = {str(path): sha256(path) for path in files}
    require(hashes[str(directory / checkpoint.STATE_FILE)] == meta['state_sha256']
            and hashes[str(directory / 'config.json')] == meta['config_sha256'], 'Actual model payload/config SHA differs.')
    state = torch.load(directory / checkpoint.STATE_FILE, map_location='cpu', weights_only=True, mmap=True)
    budget = checkpoint.ledger(meta, state, directory)
    inventory = model_inventory(BASE)
    shapes = inventory['projection_shapes']
    specs = meta['native_modules'] + meta['compact_modules'] + meta['pending_dense_modules']
    require(len(specs) == len(shapes) and {item['name'] for item in specs} == set(shapes)
            and budget['original_linear_weights'] == inventory['original_linear_weights'], 'Actual projection inventory differs.')
    for item in specs:
        spec = item.get('spec', item)
        shape = spec['geometry']['original_shape'] if 'spec' in item else [spec['out_features'], spec['in_features']]
        require(list(shape) == shapes[item['name']], 'Projection geometry differs: ' + item['name'])
    if final:
        require(meta['checkpoint_kind'] == 'mixed_final_model' and len(meta['native_modules']) == 216
                and len(meta['compact_modules']) == 36 and not meta['pending_dense_modules']
                and all(item['name'].endswith('.gate_proj') for item in meta['compact_modules'])
                and budget['budget_passed'] and budget['final_budget_applicable'],
                'Final quality evaluation requires real216 native+36 compact+0 dense and logical<=2.6.')
    return meta, budget, hashes


def source(directory, pipeline_report):
    directory, report_path = Path(directory).resolve(), Path(pipeline_report).resolve()
    report = read(report_path)
    meta, budget, hashes = inspect(directory)
    require(report['status'] == 'completed_whole_class_pipeline' and report['phase'] == 'formal' and report['exit_code'] == 0
            and report['final_checkpoint_id'] == meta['checkpoint_id']
            and Path(report['final_model']).resolve() == directory
            and report['actual_recovery_steps_per_category'] == report['planned_recovery_steps_per_category'] == 500
            and report['actual_native_fit_updates_per_stage'] == report['planned_native_fit_updates_per_stage'] == 10000,
            'Require the real completed whole-class terminal report with this final UUID.')
    provenance = meta['provenance']
    require(provenance['phase'] == 'formal' and provenance['arm'] == report['arm']
            and provenance['actual_category'] == 'down_proj' and provenance['actual_recovery_steps'] == 500
            and provenance['stage3_fit_actual_updates'] == 10000
            and provenance['source_raw_manifest'] == report['source_raw_manifest']
            and provenance['source_raw_manifest_sha256'] == report['source_raw_manifest_sha256']
            and provenance['source_after_o'] == report['source_after_o'], 'Final raw-training/parent provenance differs.')
    require(report['arm'] in ('dynamic', 'rice')
            and all(item['spec'].get('consumer') == ('fixed2_rice7' if report['arm'] == 'rice' else None)
                    for item in meta['compact_modules']), 'Actual compact consumers differ from the completed arm.')
    raw_manifest = Path(report['source_raw_manifest'])
    require(sha256(raw_manifest) == report['source_raw_manifest_sha256'], 'Actual raw-training source manifest changed.')
    strict_path = Path(report['final_strict_reload_report'])
    strict = read(strict_path)
    require(strict['status'] == 'strict_full_state_and_BF16_logits_verified' and strict['exit_code'] == 0
            and strict['checkpoint_id'] == meta['checkpoint_id'] and strict['same_input_logits_exact']
            and strict['persistent_fields'] == len(meta['tensor_table'])
            and strict['effective_weight_files'] == [str(directory / checkpoint.STATE_FILE)]
            and all(strict['actual_budget'][key] == budget[key] for key in BUDGET_FIELDS),
            'Final independent deployed-consumer reload evidence is missing.')
    for path in (raw_manifest, strict_path):
        hashes[str(path)] = sha256(path)
    hashes[str(report_path)] = sha256(report_path)
    return dict(checkpoint_dir=str(directory), checkpoint_id=meta['checkpoint_id'], checkpoint_format=FORMAT,
        metadata_sha256=hashes[str(directory / checkpoint.META_FILE)], checkpoint_files_sha256=hashes,
        checkpoint_weight_stat=file_stat(directory / checkpoint.STATE_FILE), bit_budget=budget,
        pipeline_report=str(report_path), pipeline_provenance=meta['provenance']), meta


def source_pins(binding):
    paths = set(HERE.glob('*.py'))
    for folder, names in (
        ('adaptive_native_model_20261029', ('__init__.py', 'checkpoint.py', 'consumer.py', 'codec.py', 'spec.py',
            'native_prefix.py', 'archive_cat_gate_prefixes.py', 'validate_native_prefix.py', 'validate_cpu.py')),
        ('adaptive_class_20261030', ('__init__.py', 'sources.py', 'sparse.py')),
        ('adaptive_vector_rvq', ('common.py', 'codec.py', 'length_codec_probe.py'))):
        paths.update(ROOT / 'experiments' / folder / name for name in names)
    paths.update((ROOT / 'experiments/longbench_v1').glob('*.py'))
    for folder in ('rotation', 'litebsq', 'train_utils', 'e2e_common'):
        paths.update((RUNTIME / folder).rglob('*.py'))
    paths.add(RUNTIME / 'tools/cat_eval.py')
    paths.update(ROOT / 'experiments/w25_eval' / name for name in
                 ('transfer_nine_task_acceptance.py', 'nine_task_acceptance.py', 'acceptance.py', 'model_inventory.py'))
    paths.update((HELPERS, REFERENCE, TEMPLATE,
        TOPIC / 'llama_fullqk_qkonly2000_longbench21/bind_terminal.py',
        TOPIC / 'llama_fullqk_qkonly2000_longbench21/merge_full21.py',
        TOPIC / 'qwen_sparse005_lossless_index_rice7_20261024/rice7.py',
        ROOT / 'experiments/cat_aligned_evaluation_20261028/evaluate.py',
        ROOT / 'experiments/cat_aligned_evaluation_20261028/bindings.py',
        ROOT / 'experiments/qwen3_w25/summarize_results.py',
        ROOT / 'experiments/qwen3_w25/step_export_callback.py', ROOT / 'experiments/compare_longbench_v1.py',
        SAMPLE_DIR / 'eval_utils_with_samples.py', SAMPLE_DIR / 'sample_artifacts.py'))
    baseline, lb = template_inputs()
    paths.update(map(Path, (*baseline, lb)))
    return {**binding['checkpoint_files_sha256'], **{str(path.resolve()): sha256(path) for path in paths}}


def make_config(binding, pins, *, mode='full'):
    config = deepcopy(read(REFERENCE))
    config['mode'] = mode
    model = {key: config['model'][key] for key in ('hf_path', 'hf_metadata_sha256', 'hf_weight_stats')}
    model.update(kind='compressed', checkpoint=binding['checkpoint_dir'], checkpoint_id=binding['checkpoint_id'],
        checkpoint_format=FORMAT, metadata_file=checkpoint.META_FILE, metadata_sha256=binding['metadata_sha256'],
        weight_stat=binding['checkpoint_weight_stat'], checkpoint_files_sha256=binding['checkpoint_files_sha256'],
        mixed_runtime=dict(max_linear_bpw=2.6, max_lora_rank=0, bit_budget=binding['bit_budget'], source_sha256=pins))
    config['model'] = model
    check_config(config)
    return config


def check_config(config):
    record = config['model']
    require(config['schema'] == 'longbench_v1_run_v1' and config['mode'] in ('full', 'validation')
            and config['protocol'] == protocol_record(record['hf_path'])
            and config['model_policy'] == model_policy(record['hf_path'])
            and config['prewarm_group_size'] == 8 and config['device'] == 'cuda:0', 'Canonical generation policy changed.')
    require(sha256(config['archive']['path']) == config['archive']['sha256'], 'Archive manifest changed.')
    check_hashes(config['source_sha256']); check_hashes(record['hf_metadata_sha256'])
    for stat in record['hf_weight_stats']:
        check_stat(stat)
    require(record['kind'] == 'compressed' and record['checkpoint_format'] == FORMAT
            and record['metadata_file'] == checkpoint.META_FILE, 'Use the explicit mixed model identity.')
    check_hashes(record['checkpoint_files_sha256']); check_stat(record['weight_stat'])
    meta = read(Path(record['checkpoint']) / checkpoint.META_FILE)
    require(meta['format'] == record['checkpoint_format'] and meta['checkpoint_id'] == record['checkpoint_id']
            and sha256(Path(record['checkpoint']) / checkpoint.META_FILE) == record['metadata_sha256'], 'Mixed metadata changed.')
    check_hashes(record['mixed_runtime']['source_sha256'])
    for name, version in config['package_versions'].items():
        require(importlib.metadata.version(name) == version, 'Installed dependency changed: ' + name)


def template_inputs():
    argv = read(TEMPLATE)['nine_task_argv']
    index = argv.index('--baseline_results')
    return argv[index + 1:index + 3], argv[argv.index('--baseline_longbench_scores') + 1]
