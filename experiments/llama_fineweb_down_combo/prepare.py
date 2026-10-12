"""CPU binding of completed Llama FineWeb800 and preserved fresh D16 down."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import os
import sys

import torch

from experiments.full_rht_gate_pair.compare import score
from experiments.llama_fineweb_down_combo.run import check, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.w25_eval.assemble_categories import validate_sources
from tools.cat_eval import build_parser

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / 'result/qwen3_8b_mmlu65_20261004'
BASE = R2 / 'general_lm_recovery_s2000/llama/step_0800'
DONOR = R2 / 'llama_down_dimension/block32'
SOURCE_ID = '35776715-2707-45a9-9fb4-2da1efef3f0e'
BASE_ID = '2fd16aa0-28e2-451b-ac70-91950ceeff84'
DONOR_ID = '8a4633a7-93bd-429a-8a0e-a519a2ae30ae'


def file_stat(path):
    path = Path(path); st = path.stat()
    return {'path': str(path), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'inode': st.st_ino}


def prepare(output):
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and Path.cwd() == ROOT and not output.exists(), 'Prepare a new output with CUDA hidden in the public repository.')
    parent = read(BASE.parent / 'launch.json'); donor_run = read(DONOR / 'launch.json')
    ba, da = read(BASE / 'audit.json'), read(DONOR / 'audit.json')
    review = read(BASE / 'independent_review.json'); old_assembly = read(DONOR / 'final_model/assembly_report.json')
    _require(review['status'] == ba['status'] == da['status'] == old_assembly['status'] == 'verified'
             and review['checkpoint_id'] == ba['checkpoint_id'] == BASE_ID
             and da['checkpoint_id'] == old_assembly['checkpoint_id'] == donor_run['checkpoint_id'] == DONOR_ID
             and ba['source_checkpoint_id'] == da['base_checkpoint_id'] == parent['source_checkpoint_id'] == SOURCE_ID
             and ba['selected_global_step'] == 800 and ba['train_mode'] == 'decoder'
             and ba['rank_zero'] and ba['non_decoder_non_code_bytes_unchanged']
             and ba['frozen_bytes_unchanged'] and ba['changed_code_bits'] == 0
             and ba['fresh_strict_reload_state_exact'] and ba['serialized_tensors_compared'] == ba['reloaded_tensors_compared'] == 2979,
             'Require actual completed decoder-only800 with the original3577 nondecoder state and the retained donor.')
    _require(donor_run['status'] == 'completed' and donor_run['exit_code'] == 0
             and old_assembly['sources']['base_checkpoint_id'] == SOURCE_ID
             and old_assembly['sources']['categories'] == ['down_proj']
             and da['source_union_save_fresh_tensors'] == 2979 and da['preserved_projection_count'] == 192
             and da['assembly_report_sha256'] == sha256(DONOR / 'final_model/assembly_report.json'),
             'The D16 donor must preserve original3577 backbone and unselected projections.')
    evidence = [BASE / name for name in ('independent_review.json', 'audit.json', 'step_export.json', 'full8_summary.json')]
    evidence += [DONOR / name for name in ('launch.json', 'audit.json', 'full8_summary.json')]
    evidence += [DONOR / 'final_model/assembly_report.json']
    for directory, stages in ((BASE, ('export', 'audit', 'mmlu', 'seven')),
                              (DONOR, ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer'))):
        for stage in stages:
            path = directory / (stage + '.exit'); _require(path.read_text().strip() == '0', 'Missing successful stage: ' + str(path)); evidence.append(path)
    bm, dm, names, architecture, _, runtime, differences = validate_sources(BASE / 'final_model', DONOR / 'final_model', ['down_proj'])
    _require(bm['checkpoint_id'] == BASE_ID and dm['checkpoint_id'] == DONOR_ID
             and architecture['model_type'] == 'llama' and architecture['projection_count'] == 224 and len(names) == 32,
             'Actual model-aware assembly inventory differs.')
    for spec in dm['converted_modules']:
        if spec['name'] not in names:
            continue
        _require(spec['codebook_dim'] == 16 and spec['stage_codebook_dims'] == [16, 16]
                 and spec['transpose'] and spec['residual_stages'] == 2
                 and spec['weight_rotation']['block_size'] == 32
                 and spec['protected_input_indices'] == {'shape': [256], 'dtype': 'int64'}
                 and all(d['in_dim'] == 24 and d['out_dim'] == 16 and d['hidden_dim'] == 128
                         and d['num_res_blocks'] == 0 and d['param_dtype'] == 'bfloat16' for d in spec['stage_decoders']),
                 'Require the retained D16/K24/S2/H128R0/block32/input256 geometry.')
    bb, db = ba['bit_budget'], da['bit_budget']
    bits = sum((db if name in names else bb)['per_target'][name]['budget_bits'] for name in architecture['projection_shapes'])
    _require(bits == 17095180288 and bits / architecture['original_linear_weights'] < 2.5,
             'Selected audited per-target storage projection differs.')
    controls = {}; raw_controls = []
    for label, directory, checkpoint in (
            ('fineweb800', BASE, BASE / 'final_model'), ('down_d16', DONOR, DONOR / 'final_model'),
            ('original3577', Path(parent['control_result_paths'][0]).parent, Path(parent['source_checkpoint']))):
        paths = [directory / (name + '.json') for name in ('mmlu', 'seven')]
        value, raw = score(paths, architecture, checkpoint); raw_controls.append(raw)
        controls[label] = {'checkpoint_dir': str(checkpoint), 'checkpoint_id': value['checkpoint_id'],
                           'result_paths': list(map(str, paths)), 'mean_percent': value['mean_percent'],
                           'mmlu_correct': value['integer_counts']['mmlu']['correct']}
        evidence += paths + [checkpoint / name for name in ('checkpoint_meta.json', 'config.json')]
    for raw in raw_controls[1:]:
        for current, old in zip(raw, raw_controls[0], strict=True):
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                _require(current[key] == old[key], 'Control protocol differs: ' + key)
    old = read(ROOT / 'result/llama31_8b_w25_20261004/exact_ce2000_reproduction/ce50_attention_ce10_mlp_step2000/launch.json')
    commands = [{'name': 'assembly', 'argv': [sys.executable, '-m', 'experiments.w25_eval.assemble_categories',
        '--base', str(BASE / 'final_model'), '--donor', str(DONOR / 'final_model'), '--categories', 'down_proj', '--output', str(output / 'final_model')]}]
    for name, tasks, batch in (('mmlu', 'mmlu', '64'), ('seven', 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa', '32')):
        argv = [sys.executable, 'tools/cat_eval.py', '--checkpoint_dir', str(output / 'final_model'), '--strict', '--eval_lm_eval',
            '--eval_device', 'cuda:0', '--tasks', tasks, '--num_fewshot', '0', '--lm_batch_size', batch, '--tokenizer_use_fast', 'true',
            '--prewarm_group_size', '8', '--eval_log_dir', str(output / (name + '_logs')), '--output_json', str(output / (name + '.json'))]
        parsed = build_parser().parse_args(argv[2:]); _require(parsed.num_fewshot == 0 and str(parsed.lm_batch_size) == batch, 'Native evaluation parser differs.')
        commands.append({'name': name, 'argv': argv})
    sources = dict(parent['source_sha256'])
    for module in tuple(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file:
            path = Path(file).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'result') and path.suffix == '.py':
                sources[str(path)] = sha256(path)
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    hf = Path(architecture['base_model_path'])
    evidence += [BASE / 'final_model/export_record.json', hf / 'config.json', hf / 'tokenizer_config.json']
    stats = [file_stat(path / 'pytorch_model.bin') for path in (BASE / 'final_model', DONOR / 'final_model', Path(parent['source_checkpoint']))]
    stats += [file_stat(path) for path in hf.glob('*.safetensors')]
    record = {'schema': 'llama_fineweb800_down_d16_combo_v1', 'status': 'prepared_unbound', 'gpu': None,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'queued': False, 'automatic_retry': False,
        'base_checkpoint_id': BASE_ID, 'donor_checkpoint_id': DONOR_ID, 'original_checkpoint_id': SOURCE_ID,
        'architecture': architecture, 'selected_targets': names, 'preserved_projection_count': 192,
        'source_sha256': sources, 'input_sha256': {str(p): sha256(p) for p in evidence}, 'protected_stats': stats,
        'controls': controls, 'stage_commands': commands, 'environment': old['environment'],
        'expected_user_linear_bits': int(bits), 'projected_user_linear_bpw': bits / architecture['original_linear_weights'],
        'budget_boundary': 'Projection from the two existing audited per-target inventories; actual new checkpoint tensor budget remains required.',
        'runtime_config_compatibility': runtime, 'selected_spec_differences': differences,
        'frozen_backbone_evidence': 'FineWeb800 native audit freezes all nondecoder state versus3577; D16 donor native union preserves3577 backbone. Existing assembler additionally compares both complete outside-projection states byte-exactly before assembly.',
        'scientific_difference': 'Retain FineWeb800 attention/gate/up192 and original3577 backbone/norm/head; replace all32 down with retained fresh-HF D16/K24/S2/H128R0/block32/input256. This discards FineWeb800 D32 down decoder updates; no new fitting, LoRA, code search or additive-gain claim.',
        'resource_budget_seconds': 1500, 'termination_grace_seconds': 20,
        'validation_reuse': 'Native D16 donor full8+strict2979 and FineWeb800 full8+strict2979; same generic category assembler used by prior Llama CE combo. No new GPU smoke.',
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated',
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_fineweb_down_combo.run', '--launch-record', str(output / 'launch.json')],
        'cuda_initialized': torch.cuda.is_initialized()}
    check(record); _require(not record['cuda_initialized'], 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print({'status': record['status'], 'sources': len(sources), 'inputs': len(record['input_sha256']),
           'launch_sha256': sha256(output / 'launch.json'), 'bits_projected': int(bits), 'controls': controls}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=R2 / 'llama_fineweb800_down_d16_combo')
    prepare(parser.parse_args().output.resolve())
