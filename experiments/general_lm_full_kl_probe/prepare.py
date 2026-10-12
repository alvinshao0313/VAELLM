"""CPU preparation for one unchanged FineWeb recipe with full KL and four updates."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.general_lm_full_kl_probe.train import configuration, training_arguments
from experiments.general_lm_recovery.prepare import R2, ROOT, data_contract
from experiments.llama_ce_reproduction.train import check_inputs, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.w25_eval.acceptance import validate_checkpoint
from experiments.w25_eval.validate_decoder_subset_training import expected_inventory
from train_utils.runtime_snapshot import to_runtime_jsonable

CONTROL = R2 / 'general_lm_recovery_s2000/qwen3'


def prepare(output):
    output = Path(output).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and os.environ.get('ACCELERATE_GRADIENT_ACCUMULATION_STEPS', '1') == '1', 'CPU preparation only.')
    require(not output.exists() and shutil.disk_usage(output.parent).free >= 25_000_000_000,
            'Use a fresh directory and reserve native plus independent deployment storage.')
    control = read(CONTROL / 'launch.json')
    check_inputs(control)
    normalized = list(CONTROL.glob('home_*/normalized_e2e_runtime_args.json'))
    require(len(normalized) == 1, 'Require the actual single native FineWeb control snapshot.')
    snapshot = read(normalized[0])
    require(snapshot['canonical_config'] == control['cpu_configuration']['canonical_config'],
            'Actual control canonical configuration differs from its prepared science.')
    baseline = read(CONTROL / 'step_0400/audit.json')
    require(baseline['status'] == 'verified' and baseline['selected_global_step'] == 400
            and baseline['source_checkpoint_id'] == control['source_checkpoint_id'],
            'The matching source/data/decoder control has no verified actual first400 state.')
    manifest, stats, train_file = data_contract()
    require(str(train_file) == control['data']['train_file']
            and snapshot['canonical_config']['data'] == stats['data_config'], 'Do not change data or sampler.')
    source, meta, budget = validate_checkpoint(control['source_checkpoint'], control['architecture'])
    require(meta['checkpoint_id'] == control['source_checkpoint_id'] == 'f927ffda-dd5b-4d00-b63a-5e839a182d70'
            and budget['user_linear_bits'] == 17065672704 and budget['original_linear_weights'] == 6945767424,
            'Require the original f927 source and its actual unchanged storage budget.')
    sources, inputs = deepcopy(control['source_sha256']), deepcopy(control['input_evidence_sha256'])
    additional = ('experiments/w25_eval/validate_wide_subset_training.py',
                  'experiments/w25_eval/validate_decoder_subset_training.py',
                  'experiments/qwen3_w25/aux_step_export_callback.py',
                  'experiments/qwen3_w25/subset_step_export_callback.py',
                  'experiments/general_lm_recovery/prepare.py', 'experiments/general_lm_recovery/train.py')
    for path in [*(ROOT / x for x in additional), *sorted(Path(__file__).parent.glob('*.py'))]:
        ast.parse(path.read_text()); digest = sha256(path)
        require(str(path) not in sources or sources[str(path)] == digest, 'A reused source changed.')
        sources[str(path)] = digest
    for path in (normalized[0], CONTROL / 'step_0400/audit.json', CONTROL / 'step_0400/independent_review.json'):
        inputs[str(path)] = sha256(path)
    record = {key: deepcopy(control[key]) for key in ('source_checkpoint', 'source_checkpoint_id',
        'source_weight_stat', 'architecture', 'expected_inventory', 'hf_source', 'input_data_stat', 'data', 'environment')}
    record.update(schema='general_lm_full_kl_resource_s4_v1', status='prepared_unbound', gpu=None,
        queued=False, automatic_retry=False, cwd=str(ROOT), python_executable=sys.executable,
        output_dir=str(output), source_bit_budget=budget, source_sha256=sources, input_evidence_sha256=inputs,
        control_normalized=str(normalized[0]), control_dir=str(CONTROL),
        control_launch_observed_sha256=sha256(CONTROL / 'launch.json'),
        source_control_note='The live launch is provenance only; immutable normalized/data/source evidence is pinned.',
        selected_targets=list(control['architecture']['projection_shapes']),
        timeout_seconds=1200, resource_budget_seconds=1200, termination_grace_seconds=20,
        schedule={'max_steps':2000,'warmup_steps':20,'save_steps':400,'actual_stop_step':4,
                  'forced_native_save_steps':[4],'full8_steps':[]},
        protocol={'science_change':'loss_type kl_top_mass -> kl; all other native canonical fields except output unchanged.',
            'diagnostic_only':'Four genuine optimizer updates, per-step native logging and forced save/stop at4; no resume or quality evaluation.',
            'execution':'One GPU: training child exits completely before independent native subset export and CPU audit.',
            'observations':'Native loss, post-clip gradients, actual FP32 update and Adam moments; no extra forward or backward.',
            'resource_boundary':'Full-KL allocation is unmeasured. Actual four dynamic B4<=1024 batches do not establish every future batch peak.',
            'no_followup':'No automatic formal training, retry, queue, full8 or LongBench.'},
        estimate={'seconds':[300,900],'hard_budget_seconds':1200,
            'one_fp32_shifted_full_vocab_tensor_bytes':4*1023*151936*4,
            'observation_previous_fp32_parameter_bytes':control['expected_inventory']['selected_decoder_elements']*4,
            'peak_cuda_bytes':'unmeasured; full softmax/logsoftmax/elementwise KL and backward add dense tensors',
            'disk_free_bytes':shutil.disk_usage(output.parent).free})
    record['training_argv'] = training_arguments(record)
    record['argv'] = [sys.executable, '-u', '-m', 'experiments.general_lm_full_kl_probe.train',
                      '--launch-record', str(output / 'launch.json')]
    record['launcher_argv'] = [sys.executable, '-u', '-m', 'experiments.general_lm_full_kl_probe.run',
                               '--launch-record', str(output / 'launch.json')]
    cfg, hf, args, names, architecture = configuration(record)
    require(expected_inventory(source, architecture, names) == record['expected_inventory'], 'Real decoder inventory differs.')
    record['cpu_configuration'] = {'canonical_config':to_runtime_jsonable(cfg), 'hf_args':to_runtime_jsonable(hf),
        'training_args':to_runtime_jsonable(args), 'only_canonical_differences':['run_root_dir','loss.loss_type'],
        'source_actual_step400_verified':True}
    check_inputs(record)
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['cuda_initialized'] = False
    output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'launch':str(output/'launch.json'), 'sha256':sha256(output/'launch.json'),
           'sources':len(sources),'inputs':len(inputs),'cuda_initialized':False}, flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=R2/'general_lm_full_kl_probe')
    prepare(parser.parse_args().output_dir)
