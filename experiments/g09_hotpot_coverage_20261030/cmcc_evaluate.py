"""Bind H192 to cmcc and run the unchanged complete full8 + 21/4750 protocol."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from experiments.g09_hotpot_coverage_20261030 import cmcc_export as c

# A direct script invocation must not shadow the installed Hugging Face
# evaluate package with this experiment's historical evaluate.py.
sys.path[:] = [path for path in sys.path if Path(path or os.getcwd()).resolve() != Path(__file__).resolve().parent]

R = c.ROOT / 'result/qwen3_8b_mmlu65_20261004'
SUPPORT = c.OUT / 'evaluation_support'
MANIFEST = c.ROOT / 'result/prepare_eval_20261030/longbench/manifest.json'
# A source-preserving migration binding supplies these baseline artifacts.
BASELINE = Path('/data1/cmcc/VAELLM/artifacts/cmcc_baselines_20261030/qwen3_8b')


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


def support():
    from experiments.cmcc_runtime_20261030.paths import resolve_legacy_path
    original = R / 'qwen_a0e5_baseline_s400_longbench21/merge_plan.json'
    baseline = c.read(BASELINE / 'migration_binding.json')
    c.require(baseline['schema'] == 'cmcc_qwen3_bf16_baseline_migration_v1'
              and baseline['status'] == 'verified', 'Require the verified source-preserving baseline migration.')
    for item in baseline['bindings'].values():
        c.require(c.sha(item['path']) == item['sha256'], 'Changed rebound baseline: ' + item['path'])
    plan = c.read(original)
    argv = [str(resolve_legacy_path(item)) if item.startswith(('/home/star/', '/data/star/')) else item
            for item in plan['nine_task_argv']]
    argv[0] = sys.executable
    argv[argv.index('--base_model_path') + 1] = str(c.HF)
    argv[argv.index('--longbench_manifest') + 1] = str(MANIFEST)
    index = argv.index('--baseline_results')
    argv[index + 1:index + 3] = [str(BASELINE / 'mmlu.json'), str(BASELINE / 'seven.json')]
    argv[argv.index('--baseline_longbench_scores') + 1] = str(BASELINE / 'longbench/scores.json')
    plan['nine_task_argv'] = argv
    binding = dict(original_template=c.bind(original), original_config=c.bind(original.parent / 'config.json'),
        archive=c.bind(MANIFEST), baseline_binding=c.bind(BASELINE / 'migration_binding.json'),
        scope='Only current absolute paths and identities rebound; benchmark results and protocols remain original.')
    destination = SUPPORT / 'template.json'
    if destination.exists():
        c.require(c.read(destination) == plan and c.read(SUPPORT / 'bindings.json') == binding,
                  'Changed evaluation support binding.')
    else:
        c.write(destination, plan)
        c.write(SUPPORT / 'bindings.json', binding)
    return destination


def source(profile, stage, checkpoint=None, native_audit=None):
    c.require(profile == 'Muon' and stage == '' and checkpoint is native_audit is None,
              'This evaluation binds only the completed cmcc H192 endpoint.')
    from experiments.longbench_v1.common import file_stat
    from train_utils.checkpoint_bpw import audit_checkpoint
    audit_path = c.OUT / 'native_audit.json'
    audit = c.read(audit_path)
    directory = c.OUT / 'native_final/final_model'
    meta = c.read(directory / 'checkpoint_meta.json')
    c.require((c.OUT / 'export.exit').read_text().strip() == '0'
              and audit['status'] == 'verified' and audit['schema'] == 'cmcc_H192_completed_endpoint_native_audit_v1'
              and audit['source_checkpoint_id'] == c.SOURCE_ID and audit['checkpoint_id'] == meta['checkpoint_id'] != c.SOURCE_ID
              and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 3423
              and audit['fresh_strict_reload_state_exact'] and audit['frozen_tensors_compared'] == 1911
              and audit['code_bank_count'] == 504 and audit['source_geometry_unchanged']
              and audit['endpoint_report'] == c.bind(c.OLD / 'formal32/report.json')
              and audit['prepared_binding'] == c.bind(c.PREPARED)
              and audit['checkpoint_meta_sha256'] == c.sha(directory / 'checkpoint_meta.json')
              and audit['checkpoint_weight_stat'] == file_stat(directory / meta['state_dict_file'])
              and c.read(c.OUT / 'frozen_content_audit.json')['status'] == 'verified',
              'Require this H1923423/fresh strict/1911-frozen/504-bank complete native endpoint.')
    budget = audit_checkpoint(directory, max_bpw=2.6, max_lora_rank=0)
    c.require(budget['budget_passed'] and not budget['contract_violations']
              and all(budget[key] == audit['bit_budget'][key] == audit['source_bit_budget'][key] for key in c.FIELDS),
              'Actual native storage changed after export.')
    pins = {str(path.resolve()): c.sha(path) for path in (audit_path, directory / 'checkpoint_meta.json',
            c.PREPARED, c.OUT / 'export.exit', c.OUT / 'frozen_content_audit.json', c.OLD / 'formal32/report.json')}
    return dict(profile='G', stage='Muon', original_phase='formal32', checkpoint_dir=str(directory),
        checkpoint_id=meta['checkpoint_id'], source_checkpoint_id=c.SOURCE_ID, native_audit=str(audit_path),
        native_audit_sha256=c.sha(audit_path), checkpoint_weight_stat=audit['checkpoint_weight_stat'],
        bit_budget=budget, evidence_sha256=pins), meta


def runtime():
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Keep the orchestration process on CPU.')
    _, native_imports = c.native()
    template = support()
    original = module(c.ROOT / 'experiments/cat_aligned_evaluation_20261028/evaluate.py', '_cmcc_H192_fullnine')
    api = module(R / 'qwen_mlp_night_eval/evaluate.py', '_cmcc_H192_LB_helpers')
    full8_api = module(R / 'qwen_aux_retention32_batch_20261011/evaluate.py', '_cmcc_H192_full8_helpers')
    recovery = module(c.ROOT / 'experiments/night_sequence_20261026/recovery_eval.py', '_cmcc_H192_full8')
    for helper in (api, full8_api):
        helper.PYTHON, helper.S, helper.PUBLIC = Path(sys.executable), c.S, c.ROOT
        helper.TEMPLATE = template
    old_module_from = recovery.module_from

    def recovery_module(path, name):
        if Path(path).resolve() == R / 'qwen_aux_retention32_batch_20261011/evaluate.py':
            return full8_api
        return old_module_from(path, name)

    recovery.module_from = recovery_module

    def source_pins(binding):
        paths = [Path(__file__), Path(c.__file__), Path(c.__file__).with_name('cmcc_native_loader.py'),
                 template, SUPPORT / 'bindings.json', BASELINE / 'migration_binding.json',
                 BASELINE / 'mmlu.json', BASELINE / 'seven.json', BASELINE / 'longbench/scores.json',
                 BASELINE / 'longbench/config.json', api.HERE, api.BINDER, api.MERGER, full8_api.HERE,
                 full8_api.SAMPLE_EVAL, full8_api.SAMPLE_EVAL.parent / 'sample_artifacts.py',
                 full8_api.SAMPLE_EVAL.parent / 'eval_utils_with_samples.py', Path(original.__file__),
                 c.ROOT / 'experiments/cat_aligned_evaluation_20261028/predict.py',
                 c.ROOT / 'experiments/cat_recompression/longbench.py',
                 c.ROOT / 'experiments/w25_eval/summarize_transfer_nine_task.py',
                 c.ROOT / 'experiments/w25_eval/transfer_nine_task_acceptance.py',
                 c.ROOT / 'experiments/night_sequence_20261026/recovery_eval.py',
                 c.ROOT / 'experiments/night_sequence_20261026/recovery_core.py']
        from experiments.longbench_v1.common import source_pins as lb_pins
        return {**binding['evidence_sha256'], **lb_pins(),
                **c.read(BASELINE / 'longbench/scores.json')['prediction_sha256'],
                **{item['path']: item['sha256'] for item in native_imports.values()},
                **{str(path.resolve()): c.sha(path) for path in paths}}

    # Both mature full8 call sites use only the current executable/template/source list.
    full8_api.source_paths = lambda: [Path(path) for path in source_pins(dict(evidence_sha256={}))]

    def make_config(binding, meta, pins):
        from experiments.longbench_v1.models import make_config as native_config, check_config
        reference = c.read(R / 'qwen_a0e5_baseline_s400_longbench21/config.json')
        config = native_config(MANIFEST, kind='compressed', hf_path=c.HF, checkpoint=binding['checkpoint_dir'], mode='full')
        c.require(all(config[key] == reference[key] for key in ('schema', 'mode', 'protocol', 'model_policy',
                    'prewarm_group_size', 'device')), 'Keep the original canonical32K/BF16/seed42 generation protocol.')
        config['model']['recompression'] = dict(max_linear_bpw=2.6, max_lora_rank=0,
                                                bit_budget=binding['bit_budget'], source_sha256=pins)
        check_config(config)
        return config

    b = SimpleNamespace(**vars(original.b))
    b.ROOT, b.SNAPSHOT, b.PYTHON = c.ROOT, c.S, Path(sys.executable)
    b.OUTPUT = c.OUT / 'evaluation'
    b.source, b.source_pins, b.make_config = source, source_pins, make_config
    b.helpers, b.recovery = lambda: api, lambda: recovery
    b.cpu_environment = lambda: c.require(os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and not sys.modules['torch'].cuda.is_initialized(), 'Keep bitvae orchestration CPU-only.')
    original.b = b
    mature_plan = original.make_plan

    def make_plan(*plan_args):
        plan = mature_plan(*plan_args)
        argv = plan['nine_task_argv']
        plan['nine_task_argv'] = [sys.executable, '-B', '-m', 'experiments.w25_eval.summarize_transfer_nine_task', *argv[2:]]
        return plan

    original.make_plan = make_plan
    return original


def finish(record, output):
    result = c.read(record['nine_task_report'])
    c.require(result['quality_target_evaluated'] and result['protocol']['maximum_absolute_drop_points'] == .9
              and result['protocol']['strict_score_drop_bound'] and result['protocol']['max_linear_bpw'] == 2.6
              and abs(result['baseline_nine_task_score'] - 68.04714491790848) < 1e-12,
              'Keep original BF16 baseline and the strict current acceptance profile.')
    c.write(Path(output) / 'current_goal_result.json', dict(status='complete_full8_full21_current_goal',
        checkpoint_id=record['checkpoint_id'], source_checkpoint_id=c.SOURCE_ID,
        acceptance_report=record['nine_task_report'], score=result['compressed_nine_task_score'], passed=result['passed']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    original = runtime()
    record = original.evaluate(SimpleNamespace(profile='Muon', stage='', checkpoint=None, native_audit=None,
        check_only=args.check_only, gpus=original.gpu_list(args.gpus)))
    if args.check_only:
        return
    finish(record, c.OUT / 'evaluation/Muon')


if __name__ == '__main__':
    main()
