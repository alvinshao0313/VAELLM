"""Prepare two unbound CPU records, sharing actual f927 plans and retained HF calibration."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import sys

import torch

from experiments.qwen_attention_decoded_target_pair.common import (
    ROOT, PUBLIC, RESULT, SOURCE_ID, CATEGORIES, BITS, NAMES, ARMS, read, check, paired_recipe, _require)
from experiments.qwen_attention_decoded_target_pair.core import parsed
from experiments.qwen_attention_int4_pair.common import check as check_template
from experiments.longbench_v1.common import sha256, write_json, file_stat
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_runtime_state_v6 import restore_cat_runtime_state
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import load_v6_cat_runtime_state


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != PUBLIC and Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
             and not output.exists(), 'Use a fresh isolated output in the CPU-only snapshot.')
    template_path = RESULT / 'qwen_attention_int4_pair/candidate/launch.json'
    control_path = RESULT / 'qwen_attention_int4_pair/control/launch.json'
    template = read(template_path); control = read(control_path)
    check_template(template); check_template(control)
    _require(all(r['status'] == 'completed' and r['exit_code'] == 0 for r in (template, control))
             and template['base_checkpoint_id'] == control['base_checkpoint_id'] == SOURCE_ID
             and template['codebook_bits'] == BITS, 'Require the completed actual all-attention pair.')
    base = Path(template['base_dir']); meta = read(base / 'checkpoint_meta.json')
    state = torch.load(base / meta['state_dict_file'], map_location='cpu', mmap=True, weights_only=True)
    specs = {row['name']: row for row in meta['converted_modules']}
    _require(meta['checkpoint_id'] == SOURCE_ID and len(specs) == 252 and len(state) == 3423,
             'Unexpected actual f927 source inventory.')
    plan = {}
    for name in NAMES:
        spec = specs[name]; indices = state[name + '.protected_input_indices']
        category = name.rsplit('.', 1)[1]
        _require(spec['transpose'] == (category != 'k_proj') and spec['residual_stages'] == 2
                 and spec['stage_codebook_dims'] == [32, 32] and spec['parallel_parts'] == 1
                 and spec['weight_rotation']['type'] == 'two_sided_hadamard'
                 and spec['weight_rotation']['block_size'] == 32 and spec['protected_channel_quant_format'] == 'int8'
                 and indices.dtype == torch.int64 and indices.shape == (64,) and indices.unique().numel() == 64
                 and int(indices.min()) >= 0 and int(indices.max()) < spec['in_features'], 'Unexpected fixed plan geometry.')
        plan[name] = {'protected_input_indices': tensor_record(indices),
            'signs': {field: tensor_record(state[name + '.weight_rotation.' + field])
                      for field in ('input_signs', 'output_signs')}}
    activation_dir = Path(control['donor_checkpoint_dir'])
    runtime = load_v6_cat_runtime_state(activation_dir, required=True, map_location='cpu')
    activation, _, _ = restore_cat_runtime_state(runtime)
    _require(activation['dataset'] == 'alpaca=1' and activation['nsamples'] == 512
             and activation['seqlen'] == 256 and activation['seed'] == 31
             and set(activation['stats_by_linear']) == set(NAMES), 'Wrong completed control HF calibration.')
    dependencies = ('experiments/qwen_qk_decoded_target_probe/core.py',
        'experiments/rotation_factor_probe/capture.py', 'train_utils/cat_runtime_state_v6.py',
        'train_utils/v6_model_loader.py', 'train_utils/checkpoint_v6.py', 'train_utils/distill_precision.py',
        'litebsq/vae_linear_prewarm.py', 'experiments/qwen_attention_int4_pair/common.py',
        'experiments/qwen_attention_int4_pair/combination.py')
    sources = dict(template['source_sha256'])
    for path in [*sorted(Path(__file__).parent.glob('*.py')), *(ROOT / p for p in dependencies)]:
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
        _require(sha256(PUBLIC / path.relative_to(ROOT)) == sha256(path), 'Public/snapshot dependency differs: ' + str(path))
    negative = RESULT / 'qwen_qk_decoded_target_pair/independent_review.json'
    positive = template_path.parent / 'independent_review.json'
    _require(sha256(negative) == 'c26d303e2b1667a82f582a2b09f68e724a58eab7175e565ee9149fdaa84c0d6c'
             and sha256(positive) == 'a737a767fa4fc6e30a176068b2c2f1520730f313fc9e391052e0406940be66c0',
             'Historical result provenance differs.')
    inputs = dict(template['input_sha256'])
    evidence = [template_path, control_path, positive, negative,
        activation_dir / 'cat_runtime_state.pt', activation_dir / 'checkpoint_meta.json',
        control_path.parent / 'independent_review.json',
        *[directory / name for directory in (template_path.parent, control_path.parent)
          for name in ('launch.prepared.json', 'report.json', 'lm_eval_results_mmlu.json', 'lm_eval_results_seven.json')]]
    inputs.update({str(path): sha256(path) for path in evidence})
    endpoint_bytes = sum(value.untyped_storage().nbytes() for key, value in state.items()
                         if key.startswith(tuple(name + '.' for name in NAMES)))
    endpoint_bytes += (template['forecast']['code_delta_bits'] + template['forecast']['decoder_delta_bits']) // 8
    # Only native attention endpoint tensors are saved: no dense partial, full union, optimizer or tokenizer copy.
    common = {key: deepcopy(template[key]) for key in ('base_dir', 'base_checkpoint_id', 'hf_path', 'architecture',
        'model_policy', 'source_native_mean_percent', 'control_result_paths', 'source_scale_control', 'environment',
        'package_versions', 'forecast', 'protocol', 'protected_stats')}
    common.update(schema='qwen_attention_decoded_target_pair_v1', status='prepared_unbound', gpu=None,
        queued=False, automatic_retry=False, cwd=str(ROOT), python_executable=sys.executable,
        categories=list(CATEGORIES), codebook_bits=BITS, steps_per_stage=10000,
        source_sha256=sources, input_sha256=inputs, source_plan=plan,
        activation_checkpoint_dir=str(activation_dir), timeout_seconds=5400, termination_grace_seconds=20,
        cpu_prepared=True, cuda_initialized=False,
        target_contract={'decoded': 'Actual complete f927 BF16 weight via native group8, including protection and inverse rotation once.',
            'hf': 'Original BF16 safetensors weight.', 'protection': 'Both use actual144 f927 input64 indices; each target is natively INT8 encoded.',
            'activation': 'Existing original-HF Alpaca512x256 seed31 cache. No recalibration, reranking or source-dependent plan.',
            'rng': 'Allocate all144 targets/Linear modules, then reset seed31 once; record global/category entry and exit states.',
            'initialization': 'Fresh native VAE/decoders; no checkpoint decoder continuation.',
            'complement': 'Exact source108MLP plus147 backbone tensors.',
            'endpoint': '1872 attention tensors plus specs; fresh native strict3423 union and actual group8 BF16 equality.'},
        historical_evidence={'qk_only_decoded_block32_mean_percent': 68.70785773314,
            'qk_only_boundary': 'Existing negative: no guarantee that restored-target re-encoding preserves recovery. Only72 Q/K and older K, not this144 K+8 INT4 pair.',
            'fresh_all144_kplus8_delta_vs_fresh_control_pp': 1.2850613748785804,
            'fresh_all144_kplus8_mean_percent': 66.3949388693221,
            'fresh_pair_boundary': 'Secondary reference only: vL4/vL26 source plans and global reset differ from this pair.',
            'fixed_plan_differences': {'model.layers.4.self_attn.v_proj': {'f927_only': [2178], 'old_fresh_only': [1984]},
                'model.layers.26.self_attn.v_proj': {'f927_only': [1773], 'old_fresh_only': [3327]}}},
        comparison_boundary='Primary decoded minus NEW hf, matched plan/RNG/K. Assess net complete-task accuracy versus recovered INT8 source. No LongBench/nine-item acceptance.',
        int4_decision_boundary='INT8 remains default. This is the one planned recovery-target INT4 comparison; no net quality benefit means stop this INT4 recipe, no automatic expansion or main-framework format change.',
        resource_estimate={'seconds_per_arm': [3000, 3600], 'hard_timeout_seconds': 5400,
            'gpu_requirement': 'One explicitly bound80GiB GPU; no automatic queue.', 'host_target_bytes': 3019898880,
            'endpoint_tensor_bytes': endpoint_bytes, 'single_arm_disk_provision_bytes': endpoint_bytes + 2 * 1024**3,
            'validation_boundary': 'CPU provenance/geometry/parser only. New144 materialization and endpoint native-rebuild GPU checks run inline; CPU preparation is not a claim they passed.'})
    common['protected_stats'].append(file_stat(activation_dir / 'cat_runtime_state.pt'))
    records = {}
    for arm in ARMS:
        directory = output / arm
        argv = list(template['training_argv'][2:])
        _require(argv.count('--output_dir') == 1, 'Ambiguous native output.')
        argv[argv.index('--output_dir') + 1] = str(directory / 'native')
        cat, _, training, vae = parse_cat_runtime_args(argv); training.seed = int(cat.seed)
        normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae, training_args=training,
            resolved_category_cfgs=resolve_category_runtime_configs(cat, vae, CATEGORIES))
        record = dict(deepcopy(common), arm=arm, target_kind=arm, output_dir=str(directory),
                      native_argv=argv, normalized_expected=normalized)
        record['launcher_argv'] = [sys.executable, '-u', '-m', 'experiments.qwen_attention_decoded_target_pair.run',
                                   '--launch-record', str(directory / 'launch.json')]
        record['bounded_launcher_argv'] = ['timeout', '--signal=TERM', '--kill-after=20', '5400', *record['launcher_argv']]
        parsed(record); check(record); records[arm] = record
    _require(paired_recipe(records['hf']) == paired_recipe(records['decoded']) and not torch.cuda.is_initialized(),
             'CPU pairing changed scientific parameters or initialized CUDA.')
    for arm, record in records.items():
        directory = Path(record['output_dir']); directory.mkdir(parents=True)
        write_json(directory / 'launch.prepared.json', record); write_json(directory / 'launch.json', record)
        print({'arm': arm, 'status': record['status'], 'prepared_sha256': sha256(directory / 'launch.prepared.json'),
               'source_pins': len(sources), 'input_pins': len(inputs), 'stats': len(record['protected_stats']),
               'resource_estimate': record['resource_estimate']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
