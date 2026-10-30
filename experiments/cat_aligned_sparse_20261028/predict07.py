"""Original official generation with the proven raw-C/G abs005 Rice bundle loader."""
import argparse
from pathlib import Path
import sparse07_bindings as b
from experiments.longbench_v1 import runner
from experiments.longbench_v1.common import check_hashes
from experiments.longbench_v1.models import check_config
from experiments.longbench_v1.protocol import DATASETS, model_policy


def load_model(config, logger):
    model, proof = b.runtime.load_model(config, logger)
    policy = model_policy(config['model']['hf_path'])
    b.require(policy == config['model_policy'] and model.config.model_type == policy['model_type']
        and model.config.rope_theta == policy['rope_theta'] and model.config.rope_scaling == policy['rope_scaling']
        and model.config._attn_implementation == policy['attn_implementation']
        and model.get_input_embeddings().weight.dtype == b.torch.bfloat16,
        'Actual Rice model inference policy differs from the canonical/configured model policy.')
    proof['model_policy'] = policy
    return model, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, required=True)
    parser.add_argument('--max-runtime-seconds', type=int, required=True)
    args = parser.parse_args()
    checkpoint, audit, meta = b.check_export(quality=True)
    config = b.read(args.config)
    check_config(config)
    check_hashes(config['model']['raw_abs005_Rice7']['source_sha256'])
    b.require(config['mode'] == 'full' and Path(config['model']['checkpoint']).resolve() == checkpoint
        and config['model']['checkpoint_id'] == meta['checkpoint_id'] and not args.output.exists(),
        'One fresh complete partition of the actual current Rice07 bundle.')
    b.runtime.check_gpu(checkpoint)
    runner.load_model = load_model
    report = runner.run(args.config, args.output, datasets=args.datasets, resume=False, max_runtime_seconds=args.max_runtime_seconds)
    b.require(report['status'] == 'partition_complete' and report['exit_code'] == 0
        and report['already_present'] == 0, 'No retry/resume/partial scoring.')
    b.runtime.check_loaded(report['load_audit'], meta['checkpoint_id'], audit['bit_budget'])


if __name__ == '__main__':
    main()
