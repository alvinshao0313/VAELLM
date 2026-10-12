"""CPU-only joint-path preparation; formal execution remains blocked by the genuine short proof."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.qwen_decoder_lora_fallback.callback import stage_commands
from experiments.qwen_decoder_lora_fallback.config import (
    PUBLIC, ROOT, CONTROL, SHORT, FORMAL, check_inputs, configuration, read, require, training_arguments)
from experiments.w25_eval.lora_state import lora_storage, validate_source
from experiments.w25_eval.subset_state import validate_recipe
from train_utils.runtime_snapshot import to_runtime_jsonable


def prepare():
    require(Path.cwd() == ROOT and Path(__file__).resolve().is_relative_to(ROOT)
            and os.environ.get('PYTHONPATH') == str(ROOT) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized(), 'CPU-only bitvae in S required.')
    require(not SHORT.exists() and not FORMAL.exists(), 'Do not overwrite a previous preparation.')
    control = read(CONTROL)
    require(control['status'] == 'completed' and control['completed_steps'] == [400, 800, 1200, 1600, 2000]
            and control['source_checkpoint_id'] == 'f927ffda-dd5b-4d00-b63a-5e839a182d70', 'Original matched D-only control incomplete.')
    source = Path(control['source_checkpoint']); meta = read(source / 'checkpoint_meta.json')
    architecture = validate_source(meta); validate_recipe(meta, architecture)
    require(architecture == control['architecture'] and meta['checkpoint_id'] == control['source_checkpoint_id'], 'Source differs.')
    storage = lora_storage(architecture['projection_shapes'], 2)
    require(storage == {'parameters': 5455872, 'tensors': 504, 'storage_bits': 87293952}, 'Actual source geometry differs.')
    expected = deepcopy(control['expected_inventory'])
    require(expected['all_decoder_tensors'] == 1512 and expected['selected_decoder_elements'] == 4633344
            and expected['serialized_tensors'] == 3423 and expected['frozen_tensors'] == 1911, 'Verified source inventory differs.')
    expected.update(lora_tensors=504, lora_elements=5455872, mutable_tensors=2016,
                    mutable_elements=10089216, deployed_tensors=3927)
    budget = control['source_bit_budget']
    require(budget['user_linear_bits'] == budget['packed_linear_bits'] == 17065672704
            and budget['original_linear_weights'] == 6945767424, 'Source ledger differs.')
    proposed = {'user_linear_bits': 17152966656, 'packed_linear_bits': 17152966656,
        'original_linear_weights': 6945767424, 'user_linear_bpw': 17152966656 / 6945767424,
        'added_lora_parameters': storage['parameters'], 'added_lora_bits': storage['storage_bits'],
        'measurement_boundary': 'Exact source-geometry estimate; a new deployed artifact has not been measured.'}
    sources, evidence = {}, {str(CONTROL): sha256(CONTROL)}
    for raw, digest in control['source_sha256'].items():
        original = Path(raw)
        mapped = ROOT / original.relative_to(PUBLIC) if original.is_relative_to(PUBLIC) else original
        chosen = mapped if mapped.is_file() else original
        require(sha256(chosen) == digest, 'Original runtime source drifted: ' + str(chosen))
        (sources if mapped.is_file() else evidence)[str(chosen)] = digest
    pending, seen = list(Path(__file__).parent.glob('*.py')), set()
    while pending:
        path = pending.pop().resolve()
        if path in seen: continue
        seen.add(path); sources[str(path)] = sha256(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else
                     [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                local = ROOT / (name.replace('.', '/') + '.py')
                if local.is_file(): pending.append(local)
    for path in [source / 'checkpoint_meta.json', source / 'config.json', Path(control['data']['manifest']),
                 Path(control['data']['canonical_statistics']), *map(Path, control['control_result_paths'])]:
        evidence[str(path)] = sha256(path)
    check_hashes(sources); check_hashes(evidence)
    for path in Path(__file__).parent.glob('*.py'):
        require(sha256(path) == sha256(PUBLIC / path.relative_to(ROOT)), 'New public/S source differs.')
    environment = deepcopy(control['environment']); environment['PYTHONPATH'] = str(ROOT)
    records = []
    for output, short in ((SHORT, True), (FORMAL, False)):
        record = {'schema': 'qwen_f927_decoder_lora_rank2_v1', 'kind': 'validate_s4' if short else 'formal_s2000',
            'status': 'prepared_unbound' if short else 'cpu_prepared_pending_joint_short',
            'gpu': None, 'export_gpu': None, 'queued': False, 'automatic_retry': False,
            'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
            **{k: deepcopy(control[k]) for k in ('source_checkpoint', 'source_checkpoint_id', 'source_weight_stat',
                'source_bit_budget', 'hf_source', 'input_data_stat', 'data', 'control_result_paths', 'control_mean_percent')},
            'architecture': architecture, 'expected_inventory': expected, 'proposed_bit_budget': proposed,
            'control_launch': str(CONTROL), 'source_sha256': sources, 'input_evidence_sha256': evidence,
            'training_argv': training_arguments(output, short), 'environment': environment,
            'argv': [sys.executable, '-u', '-m', 'experiments.qwen_decoder_lora_fallback.train',
                     '--launch_record', str(output / 'launch.json')],
            'dispatch_argv': [sys.executable, '-u', '-m', 'experiments.qwen_decoder_lora_fallback.run',
                     '--launch_record', str(output / 'launch.json')],
            'readiness': {'cpu': 'pending', 'gpu': 'not_scheduled', 'short_validation': {
                'path': str(SHORT / 'verification.json'), 'sha256': None, 'status': 'not_run'}},
            'schedule': {'max_steps': 2000, 'warmup_steps': 20, 'save_steps': 2 if short else 400,
                'save_total_limit': 1, 'actual_stop_step': 4 if short else 2000,
                'independent_export_steps': [4] if short else [400, 800, 1200, 1600, 2000],
                'full8_steps': [] if short else [400, 800, 1200, 1600, 2000]},
            'resource_budget_seconds': 1200 if short else 43200, 'termination_grace_seconds': 20,
            'estimate': {'training_peak_gib_estimate': [40, 70], 'short_elapsed_seconds_estimate': [250, 650],
                'disk_bytes_reserved': 20000000000 if short else 40000000000,
                'disk_free_at_prepare': shutil.disk_usage(output.parent).free,
                'boundary': 'Joint full-model resource cost is unmeasured; native short must establish it before formal binding.'},
            'scope': 'Existing native decoder_lora only; isolated saved-state/export/audit adaptation, no changes to existing runtime.',
            'scientific_boundary': 'Matched same-source FineWeb D-only recipe; adding rank2 also changes initialization RNG consumption. No quality claim.',
            'longbench_boundary': 'Not evaluated. Ordinary low_rank_a/b are component lora, not residual_lora; existing LongBench source is untouched.',
            'validation_plan': 'Four native B4/max1024 updates under the original2000 cosine/warm20 horizon; save2/4; training exits before step4 export/audit on the same card. Verify decoder/A/B gradients and updates with zero-B/warmup semantics,504 frozen banks,1911 complement,3927 save/fresh and native-vs-independent exact state. No short full8/LongBench.',
            'callback_reuse': 'Existing synchronous save subprocess protocol/full8 commands; new boundary audits the decoder and ordinary LoRA union.'}
        cfg, hf, args, names, _ = configuration(record)
        record['cpu_configuration'] = {'canonical_config': to_runtime_jsonable(cfg), 'hf_args': to_runtime_jsonable(hf),
            'training_args': to_runtime_jsonable(args), 'native_data_config_matches_control':
            to_runtime_jsonable(cfg.data) == control['cpu_configuration']['canonical_config']['data']}
        record['stage_templates'] = stage_commands(output / 'checkpoint_PLACEHOLDER', source,
                                                   output / 'step_PLACEHOLDER', diagnostic=short)
        check_inputs(record)
        record['readiness']['cpu'] = 'passed'; record['cuda_initialized'] = torch.cuda.is_initialized()
        require(not record['cuda_initialized'], 'Unexpected CUDA initialization.')
        records.append((output, 'launch.prepared.json' if short else 'launch.pending.json', record))
    for output, filename, record in records:
        output.mkdir(); write_json(output / filename, record)
        print({'path': str(output / filename), 'sha256': sha256(output / filename), 'status': record['status'],
               'sources': len(sources), 'evidence': len(evidence), 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    prepare()
