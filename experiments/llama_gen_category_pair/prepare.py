"""Prepare the two fixed e767/Gen800 decoder-category unions without loading weights."""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.full_rht_gate_pair.common import SHARDS, eval_argv, read
from experiments.full_rht_gate_pair.compare import score
from experiments.longbench_candidate_probe.prepare_llama_gen800 import endpoint
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from experiments.w25_eval.assemble_categories import validate_sources

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / 'result/qwen3_8b_mmlu65_20261004'
BASE = R2 / 'llama_fineweb1600_down_d16_combo'
GEN = R2 / 'general_lm_llama_gen_s2000/step_0800'
RETRIEVAL = R2 / 'llama_gen800_retrieval400'
BASE_ID = 'e767c232-56fd-4b03-a4ae-dd0781b6e047'
GEN_ID = '4cdf6c5c-f6cf-44cf-aa39-e76d65102c1e'
ARMS = {'gen_attention': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'gen_mlp': ['gate_proj', 'up_proj', 'down_proj']}
BITS = 17095180288
DENOMINATOR = 6979321856


def prepare(output):
    output = Path(output).resolve()
    _require(Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
             and not torch.cuda.is_initialized() and not output.exists(),
             'Use CUDA-hidden bitvae and a fresh output directory.')
    candidate, gen_launch, inputs = endpoint()
    review, audit = (read(GEN / name) for name in ('independent_review.json', 'audit.json'))
    base_review, template = (read(BASE / name) for name in ('review.json', 'launch.prepared.json'))
    _require(base_review['status'] == 'verified' and base_review['checkpoint_id'] == BASE_ID
             and review['checkpoint_id'] == candidate['checkpoint_id'] == GEN_ID
             and review['source_checkpoint_id'] == BASE_ID
             and review['evidence_sha256'][str(GEN / 'audit.json')] == sha256(GEN / 'audit.json'),
             'Require the reviewed e767 source and exact D-only Gen800 descendant.')
    _require(audit['frozen_tensors_compared'] == 1635 and audit['frozen_bytes_unchanged']
             and audit['non_decoder_non_code_bytes_unchanged'] and audit['packed_codes_unchanged']
             and audit['code_bank_count'] == 448 and audit['changed_code_bits'] == 0
             and audit['tensor_schema_unchanged'] and audit['actual_tensor_storage_unchanged']
             and review['frozen_norm_tensors'] == 65 and review['frozen_head_embedding']
             and review['full_saved_tensor_count'] == review['native_strict_reload_tensors_reused'] == 2979,
             'Whole-module unions are decoder-only here because all1635 other tensors are byte-identical.')
    budget = candidate['bit_budget']
    for key in ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'components'):
        _require(budget[key] == audit['source_bit_budget'][key], 'Source storage differs: ' + key)
    _require(budget['user_linear_bits'] == BITS and budget['original_linear_weights'] == DENOMINATOR,
             'Unexpected mixed-D16/D32 source budget.')
    inputs.update(base_review['evidence_sha256'])
    inputs[str(BASE / 'review.json')] = sha256(BASE / 'review.json')
    inputs.update(gen_launch['hf_source']['metadata_sha256'])
    stats = review['state_stats'][:2] + gen_launch['hf_source']['weight_files']
    _require({row['path'] for row in stats[:2]} ==
             {str(BASE / 'final_model/pytorch_model.bin'), str(GEN / 'final_model/pytorch_model.bin')},
             'Reuse exactly the two source weight identities from the independent Gen800 audit.')
    for row in stats:
        check_stat(row)
    retrieval = read(RETRIEVAL / 'review.json')
    _require(sha256(RETRIEVAL / 'review.json') ==
             '424cb0699c86c1d573bcc56e5462a4f560bc8bbdb1b315de300be0bb143f71bc'
             and retrieval['status'] == 'verified' and retrieval['checkpoint_id'] == GEN_ID
             and retrieval['samples'] == 400 and retrieval['execution_exit_code'] == 0,
             'Use the completed official400 retrieval result only as selection evidence.')
    for path in [RETRIEVAL / 'review.json', RETRIEVAL / 'scores.json',
                 *(RETRIEVAL / ('run/predictions/' + task + '.jsonl')
                   for task in ('passage_retrieval_en', 'passage_retrieval_zh'))]:
        digest = sha256(path)
        if str(path) in retrieval['evidence_sha256']:
            _require(digest == retrieval['evidence_sha256'][str(path)], 'Retrieval evidence changed.')
        inputs[str(path)] = digest
    _require(inputs[str(RETRIEVAL / 'scores.json')] == retrieval['scores_sha256'],
             'Official400 scores changed after independent review.')
    sources = dict(template['source_sha256'])
    additions = ['experiments/longbench_candidate_probe/prepare_llama_gen800.py',
                 'experiments/longbench_candidate_probe/prepare_llama_decoder1200_pair.py',
                 'experiments/longbench_candidate_probe/prepare_llama_best1600_d16.py']
    for path in [*(ROOT / name for name in additions), *sorted(Path(__file__).parent.glob('*.py'))]:
        ast.parse(path.read_text())
        sources[str(path)] = sha256(path)
    check_hashes(sources); check_hashes(inputs)
    controls, raw_controls, records = {}, {}, {}
    for arm, categories in ARMS.items():
        base_meta, donor_meta, names, architecture, _, runtime, differences = validate_sources(
            BASE / 'final_model', GEN / 'final_model', categories)
        _require(base_meta['checkpoint_id'] == BASE_ID and donor_meta['checkpoint_id'] == GEN_ID
                 and architecture == audit['architecture'] and architecture['num_hidden_layers'] == 32
                 and architecture['projection_count'] == 224 and not any(differences.values()),
                 'Actual dynamic Llama inventory or selected module specifications differ.')
        if not controls:
            for label, directory, expected_id, measured in (
                    ('e767', BASE, BASE_ID, base_review), ('gen800', GEN, GEN_ID, review['candidate'])):
                paths = [directory / (name + '.json') for name, _, _ in SHARDS]
                result, rows = score(paths, architecture, directory / 'final_model')
                _require(result['checkpoint_id'] == expected_id and result['task_metrics'] == measured['task_metrics']
                         and result['mean_percent'] == measured['mean_percent'], 'Raw full8 control changed.')
                controls[label] = {'checkpoint_dir': str(directory / 'final_model'), 'checkpoint_id': expected_id,
                    'result_paths': [str(p) for p in paths], 'mean_percent': result['mean_percent'],
                    'mmlu_correct': result['integer_counts']['mmlu']['correct']}
                raw_controls[label] = rows
            for left, right in zip(raw_controls['e767'], raw_controls['gen800'], strict=True):
                _require(all(left[k] == right[k] for k in ('configs', 'versions', 'n_shot', 'n_samples')),
                         'The two existing complete controls use different task protocols.')
        directory = output / arm
        commands = [{'name': 'assembly', 'argv': [sys.executable, '-u', '-m',
            'experiments.w25_eval.assemble_categories', '--base', str(BASE / 'final_model'),
            '--donor', str(GEN / 'final_model'), '--categories', ','.join(categories),
            '--output', str(directory / 'final_model')]}]
        commands += [{'name': name, 'argv': eval_argv(directory / 'final_model', directory / (name + '.json'),
                       tasks=tasks, batch=batch, python=sys.executable)} for name, tasks, batch in SHARDS]
        records[arm] = {'schema': 'llama_gen800_category_union_v1', 'status': 'prepared_unbound', 'gpu': None,
            'arm': arm, 'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(directory),
            'queued': False, 'automatic_retry': False, 'base_dir': str(BASE / 'final_model'),
            'donor_dir': str(GEN / 'final_model'), 'base_checkpoint_id': BASE_ID, 'donor_checkpoint_id': GEN_ID,
            'architecture': architecture, 'categories': categories, 'selected_targets': names,
            'selected_decoder_tensors': 6 * len(names), 'e767_complement_tensor_count': 2979 - 6 * len(names),
            'frozen_nondecoder_tensors': 1635, 'frozen_code_banks': 448,
            'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': stats, 'controls': controls,
            'source_bit_budget': budget, 'runtime_config_compatibility': runtime,
            'stage_commands': commands, 'environment': gen_launch['environment'],
            'timeout_seconds': 900, 'termination_grace_seconds': 20,
            'resource_estimate': {'single_gpu_seconds': [450, 600], 'new_disk_bytes': 4_300_000_000,
                'historical_assembly_seconds': 37.84, 'historical_gen800_callback_seconds': review['callback_seconds'],
                'boundary': 'Estimate only; one native union save/fresh and two complete eval shards. No training or retry.'},
            'validation_reuse': {'gen800_review': str(GEN / 'independent_review.json'),
                'frozen_contract': 'Gen800 has only1344 decoder trainables; all1635 other tensors byte-exact to e767.',
                'assembly_contract': 'Existing dynamic assembler verifies all2979 source-union tensors after save/fresh strict.',
                'large_weights_reloaded_in_preparation': False, 'new_gpu_short': False},
            'selection_evidence': {'gen800_retrieval_review': str(RETRIEVAL / 'review.json'),
                'task_results': retrieval['task_results'], 'full8_vs_e767': review['vs_e767']},
            'protocol': {'mmlu_batch': 64, 'seven_batch': 32, 'num_fewshot': 0, 'limit': None,
                         'leaf_count': 64, 'sample_count': 24742, 'prewarm_group_size': 8},
            'comparison_boundary': 'Complementary decoder source unions; no training, new codes or LoRA. '
                'Conditional category effects need not add. Source retrieval gains do not establish union gains.',
            'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated', 'cuda_initialized': False,
            'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_gen_category_pair.run',
                              '--launch-record', str(directory / 'launch.json')]}
    _require(shutil.disk_usage(output.parent).free >= 8_600_000_000 and not torch.cuda.is_initialized(),
             'Require space for both independent outputs and no CUDA initialization.')
    for arm, record in records.items():
        directory = output / arm; directory.mkdir(parents=True)
        write_json(directory / 'launch.prepared.json', record)
        print({'arm': arm, 'status': record['status'], 'source_count': len(sources), 'input_count': len(inputs),
               'stat_count': len(stats), 'prepared_sha256': sha256(directory / 'launch.prepared.json')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=R2 / 'llama_gen800_category_pair')
    prepare(parser.parse_args().output_dir)
