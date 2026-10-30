"""Bind the completed paired native endpoint to mature Llama full21 evidence."""
import argparse
from pathlib import Path
import qat_common as c
from experiments.longbench_v1.models import make_config, check_config
from experiments.w25_eval.acceptance import collect_results


def prepare(arm):
    record = c.prepared()
    output = c.HERE / arm
    report = c.read(output / 'formal64/report.json')
    final = c.DATA / arm / 'formal64/native_model'
    exported = report['export']
    meta = c.read(final / 'checkpoint_meta.json')
    c.require(report['status'] == 'complete' and report['exit_code'] == 0 and report['training']['steps'] == 64
        and report['training']['group_indices'] == list(range(32)) * 2 and report['source_checkpoint_id'] == c.SOURCE_ID
        and report['prepared_sha256'] == c.sha(c.HERE / 'prepared.json') and exported['status'] == 'verified'
        and exported['checkpoint_id'] == meta['checkpoint_id'] != c.SOURCE_ID
        and exported['serialized_tensors_compared'] == exported['reloaded_tensors_compared'] == 3619
        and exported['sources']['base_checkpoint_id'] == c.SOURCE_ID
        and exported['sources']['frozen_tensors_compared'] == record['frozen_state'][arm]['tensors'] and exported['sources']['projection_count'] == 224,
        'Require the actual fixed64 native sparse005 paired export.')
    dense = Path(record['full9']['dense'])
    reference = c.read(dense / 'config.json')
    config = make_config(reference['archive']['path'], kind='compressed', hf_path=str(c.HF), checkpoint=final, mode='full')
    c.require(all(config[key] == reference[key] for key in ('schema', 'mode', 'protocol', 'archive',
        'package_versions', 'model_policy', 'prewarm_group_size', 'device')), 'Original Llama full21 BF16 protocol changed.')
    config['model']['Llama_QAT_authorized_budget'] = {'max_linear_bpw': 2.6, 'max_lora_rank': 0,
        'native_tensors': 3619, 'native_projections': 224, 'native_main_banks': 512,
        'export_audit': str(final / 'assembly_report.json'), 'export_audit_sha256': c.sha(final / 'assembly_report.json')}
    config['model']['evaluation_adapter_sha256'] = {str(path): c.sha(path) for path in
        (c.CODE / 'native_core.py', c.CODE / 'qat_longbench.py', c.previous.SPARSE / 'native_export/runtime.py')}
    check_config(config)
    c.require(not (output / 'longbench_config.json').exists(), 'Fresh full21 configuration required.')
    c.write(output / 'longbench_config.json', config)


def full8_summary(arm):
    record = c.prepared()
    output = c.HERE / arm
    final = c.DATA / arm / 'formal64/native_model'
    architecture = c.native.inventory.model_inventory(str(c.HF))
    paths = [output / 'mmlu.json', output / 'seven.json']
    result = collect_results(paths, architecture, checkpoint=(final, c.read(final / 'checkpoint_meta.json')))
    baseline_paths = [Path(item['path']) for item in c.read(Path(record['full9']['dense']) / 'score_verification.json')['baseline_eight_raw_inputs']]
    baseline = collect_results(baseline_paths, architecture)
    c.require(result['versions'] == baseline['versions'], 'Paired full8 task versions differ.')
    for path, reference in zip(paths, baseline['input_results'], strict=True):
        c.require(c.read(path)['evals']['lm_eval']['configs'] == c.read(reference['path'])['configs'], 'Paired full8 task configs differ.')
    c.write(output / 'full8_summary.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'full8_summary'))
    parser.add_argument('--arm', choices=('D', 'DQ'), required=True)
    args = parser.parse_args()
    (prepare if args.phase == 'prepare' else full8_summary)(args.arm)
