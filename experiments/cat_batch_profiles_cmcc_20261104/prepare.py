"""Read actual source/data/recipes and emit only an unapproved proposal plus CPU evidence."""
import argparse
import ast
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
import time

from . import bindings as b
from experiments.cat_auto_protocol_cmcc_20261104.contracts import require, digest
from experiments.cat_auto_protocol_cmcc_20261104.identity import semantic_seed


def rejects(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, KeyError, TypeError, FileNotFoundError):
        return
    raise AssertionError('An unauthorized or mismatched proposal/receipt was accepted.')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--batch-root', required=True); parser.add_argument('--report', required=True)
    args = parser.parse_args(); started = time.time()
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU preparation requires caller bitvae and CUDA hidden.')
    import torch
    torch.set_num_threads(2)
    batch = Path(args.batch_root).resolve(); require(not batch.exists(), 'Use a unique new batch proposal root.')
    original, inventory, identity = b.common(); proposal = b.make_proposal(batch)
    source = Path(inventory['source_evidence']['path'])
    for name, record in inventory['source_evidence']['files'].items():
        require(b.sha(source/name) == record['sha256'], 'Actual original BF16/model/tokenizer bytes changed: '+name)
    require(b.sha(original['data_plan']) == original['data_plan_sha256'], 'Actual frozen DataPlan changed.')
    data = torch.load(original['data_plan'], map_location='cpu', weights_only=False)
    require(digest(data['manifest']) == data['data_identity'] == identity['data_identity']
        and data['manifest']['model_identity'] == inventory['model_identity'] and data['manifest']['search_seed'] == 31
        and len(data['calibration']) == len(data['proposal']) == 512
        and all(tuple(value.shape) == (1, 256) and value.device.type == 'cpu' for value in data['calibration']), 'Actual CPU root data state/shape/semantic seed differs.')
    ref = dict(path=str(batch/'proposal.json'), sha256='CPU_nonexistent_proposal_hash')
    rejects(b.validate_authorization_value, proposal, ref, {})
    rejects(b.validate_authorization_value, proposal, ref, dict(user_approved=False))
    invalid_template = dict(reference='CPU_INVALID_COUNTEREXAMPLE_ONLY', proposal_identity=proposal['proposal_identity'],
        proposal_sha256=ref['sha256'], batch_root=proposal['batch_root'], labels=list(b.CARDS), cards=deepcopy(b.CARDS),
        batch_GPU_hours=3.75, technical_seconds=900, formal_seconds=3600, chain_seconds=4500)
    for key, value in [('proposal_identity', 'wrong'), ('proposal_sha256', 'wrong'), ('batch_root', '/CPU_wrong_root'),
            ('labels', ['k32']), ('cards', {}), ('batch_GPU_hours', 99), ('technical_seconds', 901), ('formal_seconds', 3601), ('chain_seconds', 4501)]:
        invalid = dict(invalid_template, **{key: value})
        rejects(b.validate_authorization_value, proposal, ref, dict(user_approved=True, user_authorization=invalid))
    rejects(b.validate_proposal, proposal, batch.with_name(batch.name+'_foreign'))
    wrong = deepcopy(proposal); wrong['cards']['k32']['gpu'] = 5
    rejects(b.validate_proposal, wrong, batch)
    for label, card in b.CARDS.items():
        rejects(b.technical_evidence, proposal, label, b.old.PILOT/'completed.json')
        require(proposal['recipes'][label]['recipe'] == b.old.recipe_for(label)[0], 'Only source identity/routing changes; original recipes remain exact.')
    require(b.stage_window(100., 200., 'technical_validation') == (900, 1100.)
        and b.stage_window(100., 1100., 'formal') == (3600, 4600.), 'Phase caps must remain within the original absolute chain4500.')
    rejects(b.stage_window, 100., 4600., 'formal')
    require(sum(b.TECHNICAL_SECONDS+b.FORMAL_SECONDS for _ in b.CARDS)/3600 == 3.75, 'Actual three fixed chain cap arithmetic changed.')
    for path in b.HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    cli = subprocess.run([sys.executable, '-B', '-m', 'experiments.cat_batch_profiles_cmcc_20261104.run', '--help'],
        capture_output=True, text=True, check=True)
    require('--authorization' in cli.stdout and '--label' in cli.stdout, 'Actual finite CLI must require independent authorization.')
    require(not torch.cuda.is_initialized(), 'CPU proposal preparation must never initialize CUDA.')
    batch.mkdir(parents=True); b.write_new(batch/'proposal.json', proposal)
    report = dict(status='CPU_unapproved_three_chain_proposal_source_and_shape_verified', exit_code=0, CUDA_initialized=False,
        elapsed_seconds=time.time()-started, proposal=b.bound(batch/'proposal.json'), source_sha256=b.pins(),
        actual_model_source_files_rehashed=len(inventory['source_evidence']['files']), actual_DataPlan=b.bound(original['data_plan']),
        calibration_shape=[512, 1, 256], proposal_records=512, cards=b.CARDS, batch_GPU_hours=3.75,
        fit_code_identity=identity['code_identity'], old_technical_receipts_rejected=True,
        unapproved_wrong_hash_cards_budget_and_output_root_rejected=True, actual_CLI_help_exit_code=cli.returncode,
        semantic_stage_seeds=[dict(stage=stage, init=semantic_seed(identity['model'], 'up_proj', 'residual_stage_'+str(stage), 'vae_initialization', 31),
            shuffle=semantic_seed(identity['model'], 'up_proj', 'residual_stage_'+str(stage), 'vae_shuffle', 31)) for stage in (1, 2)],
        prior_actual_chain_wall_GPUh=proposal['original_batch']['actual_chain_wall_GPUh'],
        actual_executable_requests_created=False, valid_user_authorization_created=False, new_GPU_execution=False,
        boundary='Source/model/data/recipe/state/budget/CLI CPU preparation only. New helper GPU technical2/formal10K and actual throughput remain unverified.')
    Path(args.report).parent.mkdir(parents=True, exist_ok=True); b.write_new(args.report, report)
    print({key: report[key] for key in ('status', 'elapsed_seconds', 'CUDA_initialized')}, flush=True)


if __name__ == '__main__':
    main()
