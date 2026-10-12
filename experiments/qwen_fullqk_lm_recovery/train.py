"""Bind successful full-QK backward evidence before the unchanged native recovery."""
import argparse
from pathlib import Path
import sys

from experiments.general_lm_recovery.train import run_record
from experiments.longbench_v1.common import check_hashes, sha256
from experiments.long_context_recovery.contract import require
from experiments.qwen_fullqk_lm_validation.config import OUTPUT as SHORT, SOURCE, SOURCE_ID, check_inputs, read


def completed_proof():
    """Read completed evidence only; never load short-run weights for formal training."""
    paths = [SHORT / p for p in ('launch.json', 'verification.json', 'training.json', 'step_0004/audit.json')]
    launch, proof, trained, audit = map(read, paths)
    require(launch['status'] == proof['status'] == 'verified' and launch['dispatcher_exit_code'] == 0
            and all((SHORT / f'{stage}.exit').read_text().strip() == '0'
                    for stage in ('training', 'export', 'audit', 'dispatcher', 'outer'))
            and trained['status'] == 'trained' and trained['training_result']['global_step'] == 4
            and launch['source_checkpoint_id'] == proof['source_checkpoint_id'] == audit['source_checkpoint_id'] == SOURCE_ID
            and launch['source_checkpoint'] == str(SOURCE) and proof['native_vs_independent_tensors_compared'] == 3423,
            'Require naturally completed four-step full-QK validation and all actual exits0.')
    check_hashes(proof['evidence_sha256'])
    require(sha256(SHORT / 'verification.json') == launch['verification_sha256']
            and proof['expected_inventory'] == launch['expected_inventory']
            and audit['selected_decoder_tensor_count'] == 1512 and audit['frozen_tensors_compared'] == 1911
            and audit['code_bank_count'] == 504 and audit['changed_code_bits'] == 0
            and audit['fresh_strict_reload_state_exact'] and audit['frozen_bytes_unchanged'] and audit['rank_zero'],
            'Actual all-decoder/frozen-code/native-export proof differs.')
    observed = trained['observations']
    for field in ('qk_gradients', 'qk_updates'):
        require([item['step'] for item in observed[field]] == [1, 2, 3, 4]
                and all(len(item['stages']) == 144 for item in observed[field]), 'Incomplete actual Q/K stage observations.')
    require(all(value['finite'] and value['l2'] > 0 for item in observed['qk_gradients'] for value in item['stages'].values())
            and all(value['changed_fp32_elements_this_step'] > 0 for item in observed['qk_updates']
                    if item['step'] > 1 for value in item['stages'].values()), 'Missing native stage gradient/update.')
    require(audit['bit_budget']['user_linear_bits'] == audit['bit_budget']['packed_linear_bits'] == 17065672704,
            'The short proof changed persistent storage.')
    evidence = {str(path): sha256(path) for path in paths}
    evidence.update({str(SHORT / f'{stage}.exit'): sha256(SHORT / f'{stage}.exit')
                     for stage in ('training', 'export', 'audit', 'dispatcher', 'outer')})
    return {'status': 'verified', 'path': str(SHORT / 'verification.json'),
            'verification_sha256': sha256(SHORT / 'verification.json'), 'evidence_sha256': evidence}


def train(path):
    record = read(path)
    require(record['status'] == 'bound_ready' and record['resource_validation']['status'] == 'verified'
            and record['source_checkpoint'] == str(SOURCE) and record['source_checkpoint_id'] == SOURCE_ID,
            'Bind a fresh43b0 run only after the real short proof; never consume a short endpoint.')
    check_inputs(record)
    actual = completed_proof()
    require(actual == record['resource_validation'], 'Completed proof differs from the CPU-bound evidence.')
    return run_record(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', required=True, type=Path)
    sys.exit(train(parser.parse_args().launch_record))
