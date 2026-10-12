"""Bind K40 gate/down and completed FineWeb800 attention/up for one native union."""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.compare import score
from experiments.llama_ce_reproduction.evaluate_assembly import check, read
from experiments.longbench_v1.common import file_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.w25_eval.assemble_categories import validate_sources
from tools.cat_eval import build_parser

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / 'result/qwen3_8b_mmlu65_20261004'
BASE = R2 / 'llama_gate_down_balanced_bits/combined'
DONOR = R2 / 'general_lm_recovery_s2000/llama/step_0800'
REFERENCE = R2 / 'llama_fineweb800_down_d16_combo'
BASE_ID = '334eb3a1-d532-4df9-b938-d3e0d3c9d994'
DONOR_ID = '2fd16aa0-28e2-451b-ac70-91950ceeff84'
REFERENCE_ID = 'e985bada-3960-4a5a-82af-4652ef94ca6c'
SOURCE_ID = '35776715-2707-45a9-9fb4-2da1efef3f0e'
CATEGORIES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj']


def prepare(output):
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and Path.cwd() == ROOT and not output.exists(),
             'Prepare a new output with CUDA hidden in the public repository.')
    base_run = read(BASE / 'launch.json'); parent = read(DONOR.parent / 'launch.json')
    ba, da = read(BASE / 'audit.json'), read(DONOR / 'audit.json')
    br, dr = read(BASE / 'independent_review.json'), read(DONOR / 'independent_review.json')
    ds = read(DONOR / 'step_export.json'); rr = read(REFERENCE / 'review.json')
    reference_run = read(REFERENCE / 'launch.json')
    _require(ba['status'] == da['status'] == br['status'] == dr['status'] == rr['status'] == 'verified'
             and ba['checkpoint_id'] == br['checkpoint_id'] == base_run['checkpoint_id'] == BASE_ID
             and da['checkpoint_id'] == dr['checkpoint_id'] == ds['checkpoint_id'] == DONOR_ID
             and rr['checkpoint_id'] == reference_run['checkpoint_id'] == REFERENCE_ID,
             'Require the completed, independently reviewed actual three endpoints.')
    _require(base_run['status'] == reference_run['status'] == 'completed'
             and base_run['exit_code'] == reference_run['exit_code'] == 0
             and ba['base_checkpoint_id'] == da['source_checkpoint_id'] == parent['source_checkpoint_id'] == SOURCE_ID
             and ba['replaced_projection_count'] == 64 and ba['preserved_projection_count'] == 160
             and ba['source_union_save_fresh_tensors'] == 2979 and ba['historical_protection_sign_exact'] and ba['rank_zero'],
             'K40 must preserve original3577 backbone and all160 unselected projections.')
    _require(ds['status'] == 'evaluated' and ds['global_step'] == da['selected_global_step'] == 800
             and da['train_mode'] == 'decoder' and da['selected_decoder_tensor_count'] == 1344
             and da['frozen_tensors_compared'] == 1635 and da['non_decoder_non_code_bytes_unchanged']
             and da['frozen_bytes_unchanged'] and da['code_bank_count'] == 448 and da['changed_code_bits'] == 0
             and da['rank_zero'] and da['fresh_strict_reload_state_exact']
             and da['serialized_tensors_compared'] == da['reloaded_tensors_compared'] == 2979,
             'FineWeb800 must retain original codes/norm/head/backbone with only decoder updates.')
    evidence = [BASE / name for name in ('launch.json', 'audit.json', 'independent_review.json', 'full8_summary.json')]
    evidence += [DONOR / name for name in ('audit.json', 'independent_review.json', 'step_export.json', 'full8_summary.json')]
    evidence += [REFERENCE / name for name in ('launch.json', 'review.json', 'full8_summary.json')]
    evidence += [BASE / 'final_model/assembly_report.json', BASE / 'down_base/assembly_report.json',
                 DONOR / 'final_model/export_record.json', REFERENCE / 'final_model/assembly_report.json']
    for directory, stages in (
            (BASE, ('assemble_down_proj', 'assemble_gate_proj', 'mmlu', 'seven', 'dispatcher', 'outer')),
            (DONOR, ('export', 'audit', 'mmlu', 'seven')),
            (REFERENCE, ('assembly', 'mmlu', 'seven', 'dispatcher', 'outer'))):
        for stage in stages:
            path = directory / (stage + '.exit')
            _require(path.read_text().strip() == '0', 'Successful original stage required: ' + str(path))
            evidence.append(path)
    bm, dm, names, architecture, _, runtime, differences = validate_sources(
        BASE / 'final_model', DONOR / 'final_model', CATEGORIES)
    _require(bm['checkpoint_id'] == BASE_ID and dm['checkpoint_id'] == DONOR_ID
             and architecture['model_type'] == 'llama' and architecture['projection_count'] == 224
             and len(names) == 160 and all(value == [] for value in differences.values()),
             'All160 selected native projection specifications must be identical across the two sources.')
    preserved = [s for s in bm['converted_modules'] if s['name'] not in names]
    _require(len(preserved) == 64 and all(s['name'].endswith(('.gate_proj', '.down_proj'))
             and s['codebook_dim'] == 32 and s['stage_codebook_dims'] == [32, 32]
             and s['residual_stages'] == 2 and s['weight_rotation']['block_size'] == 32
             and all(d['in_dim'] == 40 and d['out_dim'] == 32 and d['hidden_dim'] == 128
                     and d['num_res_blocks'] == 0 and d['param_dtype'] == 'bfloat16' for d in s['stage_decoders'])
             for s in preserved), 'Preserve all64 actual K40/D32/S2/H128R0 gate/down projections.')
    bb, db = ba['bit_budget'], da['bit_budget']
    for name in names:
        _require(bb['per_target'][name] == db['per_target'][name], 'Selected donor storage differs: ' + name)
    bits = sum((db if name in names else bb)['per_target'][name]['budget_bits']
               for name in architecture['projection_shapes'])
    _require(bits == bb['user_linear_bits'] == 17102536704
             and architecture['original_linear_weights'] == 6979321856
             and 2 * bits < 5 * architecture['original_linear_weights'], 'Audited integer storage projection differs.')
    controls = {}; raw_controls = []
    for label, directory in (('k40', BASE), ('fineweb800', DONOR), ('fineweb800_down_d16', REFERENCE)):
        checkpoint = directory / 'final_model'; paths = [directory / (n + '.json') for n in ('mmlu', 'seven')]
        value, raw = score(paths, architecture, checkpoint); raw_controls.append(raw)
        controls[label] = {'checkpoint_dir': str(checkpoint), 'checkpoint_id': value['checkpoint_id'],
            'result_paths': list(map(str, paths)), 'mean_percent': value['mean_percent'],
            'mmlu_correct': value['integer_counts']['mmlu']['correct']}
        evidence += paths + [checkpoint / n for n in ('checkpoint_meta.json', 'config.json')]
    for raw in raw_controls[1:]:
        for current, old in zip(raw, raw_controls[0], strict=True):
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                _require(current[key] == old[key], 'Historical full8 protocol differs: ' + key)
    commands = [{'name': 'assembly', 'argv': [sys.executable, '-m', 'experiments.w25_eval.assemble_categories',
        '--base', str(BASE / 'final_model'), '--donor', str(DONOR / 'final_model'), '--categories', ','.join(CATEGORIES),
        '--output', str(output / 'final_model')]}]
    for name, tasks, batch in (('mmlu', 'mmlu', '64'),
            ('seven', 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa', '32')):
        argv = [sys.executable, 'tools/cat_eval.py', '--checkpoint_dir', str(output / 'final_model'), '--strict',
            '--eval_lm_eval', '--eval_device', 'cuda:0', '--tasks', tasks, '--num_fewshot', '0', '--lm_batch_size', batch,
            '--tokenizer_use_fast', 'true', '--prewarm_group_size', '8', '--eval_log_dir', str(output / (name + '_logs')),
            '--output_json', str(output / (name + '.json'))]
        parsed = build_parser().parse_args(argv[2:])
        _require(parsed.num_fewshot == 0 and str(parsed.lm_batch_size) == batch and parsed.lm_limit is None,
                 'Native evaluation parser differs.')
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
    _require((Path(__file__).resolve().parent / 'run.py').is_file(), 'Complete the thin runner before pinning.')
    hf = Path(architecture['base_model_path']); evidence += [hf / n for n in ('config.json', 'tokenizer_config.json')]
    stats = [file_stat(path / 'final_model/pytorch_model.bin') for path in (BASE, DONOR, REFERENCE)]
    stats += [file_stat(path) for path in sorted(hf.glob('*.safetensors'))]
    record = {'schema': 'llama_fineweb800_gate_down_k40_combo_v1', 'status': 'prepared_unbound', 'gpu': None,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'queued': False, 'automatic_retry': False,
        'base_checkpoint_id': BASE_ID, 'donor_checkpoint_id': DONOR_ID, 'original_checkpoint_id': SOURCE_ID,
        'architecture': architecture, 'categories': CATEGORIES, 'selected_targets': names, 'preserved_projection_count': 64,
        'source_sha256': sources, 'input_sha256': {str(p): sha256(p) for p in evidence}, 'protected_stats': stats,
        'controls': controls, 'primary_control': 'k40', 'stage_commands': commands, 'environment': reference_run['environment'],
        'expected_user_linear_bits': int(bits), 'expected_budget_components': bb['components'],
        'projected_user_linear_bpw': bits / architecture['original_linear_weights'],
        'budget_boundary': 'Same per-target geometry and physical storage as K40; actual assembled tensor budget must equal every K40 component.',
        'runtime_config_compatibility': runtime, 'selected_spec_differences': differences,
        'frozen_backbone_evidence': 'K40 two native unions preserve3577 backbone; FineWeb800 freezes all nondecoder state/codes/norm/head. Assembler compares full outside-projection source states and exact union of2979 tensors before saving/fresh reload.',
        'scientific_difference': 'Keep K40 gate/down64 and3577 backbone; transplant only FineWeb800 attention(q/k/v/o)+up160. No training, code search, LoRA or fitting. Primary comparison is K40 at the same geometry; FineWeb800 and e985 D16 combo have different compression/recovery histories and are secondary endpoint references.',
        'resource_budget_seconds': 1200, 'termination_grace_seconds': 20,
        'resource_estimate': {'same_native_one_union_full8_reference': str(REFERENCE / 'launch.json'),
            'reference_seconds': reference_run['finished_unix'] - reference_run['started_unix'],
            'note': 'One native union followed by the same64/32 full8. Actual wall time remains measured; no queue or retry.'},
        'validation_reuse': 'K40 and FineWeb800 actual2979 exact native source-union/deployment plus complete64leaf raw; generic assembler already exercised by e985. No new GPU short validation.',
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated',
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_fineweb_k40_combo.run', '--launch-record', str(output / 'launch.json')],
        'cuda_initialized': torch.cuda.is_initialized()}
    check(record); _require(not record['cuda_initialized'], 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print({'status': record['status'], 'sources': len(sources), 'inputs': len(record['input_sha256']),
           'launch_sha256': sha256(output / 'launch.json'), 'projected_bits': int(bits), 'controls': controls}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=R2 / 'llama_fineweb800_gate_down_k40_combo')
    prepare(parser.parse_args().output.resolve())
