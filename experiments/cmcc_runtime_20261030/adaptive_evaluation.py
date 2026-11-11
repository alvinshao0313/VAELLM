"""Use the existing mixed-format evaluators with explicit cmcc identities."""
import argparse
import importlib
from pathlib import Path
import sys

from .paths import ARTIFACTS, EVALUATION, MODELS, REPO, TOPIC, resolve_legacy_path

BASELINE = ARTIFACTS / 'cmcc_baselines_20261030/qwen3_8b'
BASE = MODELS / 'Qwen/Qwen3-8B'
MANIFEST = EVALUATION / 'longbench/manifest.json'


def prepare(output):
    from experiments.adaptive_class_evaluation_20261030 import bindings as b
    from experiments.longbench_v1.models import make_config, check_config

    b.require(not output.exists(), 'Use a fresh evaluation-support directory.')
    original = TOPIC / 'qwen_a0e5_baseline_s400_longbench21'
    baseline = b.read(BASELINE / 'migration_binding.json')
    b.require(baseline['schema'] == 'cmcc_qwen3_bf16_baseline_migration_v1' and baseline['status'] == 'verified',
        'Require the verified source-preserving BF16 baseline migration.')
    for item in baseline['bindings'].values():
        b.require(b.sha256(item['path']) == item['sha256'], 'Rebound BF16 baseline changed: ' + item['path'])
    reference = make_config(MANIFEST, kind='dense', hf_path=BASE, mode='full')
    historical = b.read(original / 'config.json')
    b.require(all(reference[key] == historical[key] for key in
        ('schema', 'mode', 'protocol', 'model_policy', 'prewarm_group_size', 'device')),
        'Keep the original canonical BF16/32K/generation protocol.')
    check_config(reference)
    template = b.read(original / 'merge_plan.json')
    argv = [str(resolve_legacy_path(item)) if item.startswith(('/home/star/VAELLM/', '/home/star/models/', '/data/star/'))
            else item for item in template['nine_task_argv']]
    argv[0] = sys.executable
    argv[argv.index('--base_model_path') + 1] = str(BASE)
    argv[argv.index('--longbench_manifest') + 1] = str(MANIFEST)
    index = argv.index('--baseline_results')
    argv[index + 1:index + 3] = [str(BASELINE / 'mmlu.json'), str(BASELINE / 'seven.json')]
    argv[argv.index('--baseline_longbench_scores') + 1] = str(BASELINE / 'longbench/scores.json')
    template['nine_task_argv'] = argv
    output.mkdir(parents=True)
    b.write_json(output / 'config.json', reference)
    b.write_json(output / 'template.json', template)
    paths = (original / 'config.json', original / 'merge_plan.json', BASELINE / 'migration_binding.json', MANIFEST,
             output / 'config.json', output / 'template.json', Path(__file__), REPO / 'experiments/cmcc_runtime_20261030/paths.py')
    b.write_json(output / 'prepared.json', dict(status='verified_cmcc_mixed_evaluation_support',
        source_sha256={str(path.resolve()): b.sha256(path) for path in paths},
        baseline_files=baseline['bindings'], model_path=str(BASE), archive=str(MANIFEST),
        protocol=reference['protocol'], full21_samples=4750,
        boundary='Path/stat/code identities rebound only; original BF16 results and full8/full21 acceptance protocol retained.'))
    print(dict(status='verified_cmcc_mixed_evaluation_support', output=str(output)), flush=True)


def setup(support):
    from experiments.adaptive_class_evaluation_20261030 import bindings as b

    record = b.read(support / 'prepared.json')
    b.require(record['status'] == 'verified_cmcc_mixed_evaluation_support', 'Use verified cmcc evaluation support.')
    b.check_hashes(record['source_sha256'])
    for item in record['baseline_files'].values():
        b.require(b.sha256(item['path']) == item['sha256'], 'Rebound BF16 baseline changed: ' + item['path'])
    b.PYTHON, b.BASE = Path(sys.executable), BASE
    b.REFERENCE, b.TEMPLATE = support / 'config.json', support / 'template.json'
    original_pins = b.source_pins

    def source_pins(binding):
        return {**original_pins(binding), **record['source_sha256'],
                str(support / 'prepared.json'): b.sha256(support / 'prepared.json')}

    def command(component, *arguments):
        return [str(b.PYTHON), '-B', '-u', '-m', 'experiments.cmcc_runtime_20261030.adaptive_evaluation',
                '--support', str(support), '--component', component, *arguments]

    b.source_pins, b.command = source_pins, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--support', type=Path)
    parser.add_argument('--component', choices=('evaluate', 'full8', 'predict', 'summarize', 'merge', 'validate_cpu', 'validate_gpu'), default='evaluate')
    args, remaining = parser.parse_known_args()
    if args.prepare:
        if args.support or remaining:
            parser.error('--prepare does not take evaluator arguments.')
        prepare(args.prepare.resolve())
        return
    if args.support is None:
        parser.error('--support is required for evaluation.')
    setup(args.support.resolve())
    from experiments.adaptive_class_evaluation_20261030 import bindings as b
    b.CLI_ARGUMENTS = list(sys.argv)
    component = importlib.import_module('experiments.adaptive_class_evaluation_20261030.' + args.component)
    sys.argv = [sys.argv[0], *remaining]
    return component.main()


if __name__ == '__main__':
    raise SystemExit(main())
