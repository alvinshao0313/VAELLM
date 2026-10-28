"""Native endpoint consumption; generation keeps the existing LongBench protocol."""
from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import export as native

import torch
from experiments.longbench_v1 import runner
from experiments.longbench_v1.common import check_hashes
from experiments.longbench_v1.models import check_config
from experiments.longbench_v1.protocol import DATASETS, model_policy
from tools.cat_eval import _prepare_model_for_eval
from train_utils.v6_model_loader import load_v6_model_checkpoint
from train_utils.checkpoint_bpw import audit_state_dict


def load_checkpoint(checkpoint, audit_path, logger):
    audit = native.read(audit_path)
    meta = native.read(checkpoint / 'checkpoint_meta.json')
    native.require(audit['status'] == 'verified' and audit['checkpoint_id'] == meta['checkpoint_id']
        and Path(audit['checkpoint_dir']).resolve() == checkpoint
        and audit['serialized_tensors_compared'] == audit['fresh_strict_tensors_compared'] == 4071
        and meta['extra_meta'][native.META_KEY] == audit['sources'], 'Use the actual verified native endpoint.')
    model, actual, loaded = load_v6_model_checkpoint(str(checkpoint),
        base_model_path=meta['base_model_path'], map_location='cpu', strict=True)
    native.require(actual == meta and not loaded.missing_keys and not loaded.unexpected_keys
        and len(model.state_dict()) == 4071, 'Fresh strict endpoint loading failed.')
    budget = audit_state_dict(meta, model.state_dict(), max_bpw=2.6, max_lora_rank=0)
    native.require(budget['budget_passed'] and all(budget[key] == audit['bit_budget'][key]
        for key in ('user_linear_bits', 'packed_linear_bits')), 'Loaded endpoint bit budget changed.')
    model.eval().requires_grad_(False)
    warm = _prepare_model_for_eval(model, 'cuda:0', 8, logger, {})
    native.require(warm['warmed'] == warm['total'] == 252 and not warm['failed'] and not warm['skipped'],
                   'All actual native projections must be consumed.')
    target = model.get_submodule(native.TARGET)
    weight_record = native.tensor_record(target._cached_weight)
    native.require(weight_record == audit['native_weight_record'],
                   'Fresh deployed target weight differs from the real training endpoint.')
    policy = model_policy(meta['base_model_path'])
    native.require(model.config.model_type == policy['model_type']
        and model.config.rope_theta == policy['rope_theta'] and model.config.rope_scaling == policy['rope_scaling']
        and model.config._attn_implementation == policy['attn_implementation']
        and model.get_input_embeddings().weight.dtype == torch.bfloat16, 'Original inference policy changed.')
    return model, {'checkpoint_id': meta['checkpoint_id'], 'strict_missing': [], 'strict_unexpected': [],
        'persistent_tensors': 4071, 'native_main_banks': 504, 'bit_budget': budget, 'warmup': warm,
        'model_policy': policy, 'native_weight_record': weight_record,
        'endpoint_audit': native.binding(audit_path), 'target_deployment_equal': True}


def load_model(config, logger):
    check_config(config)
    identity = config['model']
    check_hashes(identity['deoscillation_source_sha256'])
    checkpoint = Path(identity['checkpoint']).resolve()
    model, proof = load_checkpoint(checkpoint, Path(identity['deoscillation_audit']), logger)
    native.require(proof['checkpoint_id'] == identity['checkpoint_id'], 'Generation UUID changed.')
    return model, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--checkpoint', type=Path, required=True)
    verify.add_argument('--audit', type=Path, required=True)
    verify.add_argument('--output', type=Path, required=True)
    predict = commands.add_parser('predict')
    predict.add_argument('--config', type=Path, required=True)
    predict.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    native.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Use one explicitly assigned GPU.')
    if args.command == 'verify':
        native.require(not args.output.exists(), 'Never overwrite verification evidence.')
        model, proof = load_checkpoint(args.checkpoint.resolve(), args.audit, logging.getLogger(__name__))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        proof.update(status='verified', GPU_allocated_after_release=torch.cuda.memory_allocated())
        native.write(args.output, proof)
    else:
        config = native.read(args.config)
        native.require(config['mode'] == 'full' and not args.output.exists(), 'Only a fresh complete LongBench run.')
        runner.load_model = load_model
        report = runner.run(args.config, args.output, datasets=list(DATASETS), resume=False,
                            max_runtime_seconds=86400)
        native.require(report['status'] == 'partition_complete' and report['exit_code'] == 0
                       and report['new_predictions'] == 4750 and report['already_present'] == 0,
                       'Require every official sample; no partial score or retry.')


if __name__ == '__main__':
    main()
