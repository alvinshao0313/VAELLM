"""Real mixed-model HFLM/logits plus the original three LongBench diagnostic cases."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from experiments.adaptive_class_evaluation_20261030 import bindings as b
from experiments.adaptive_class_evaluation_20261030.loading import load_model
from experiments.longbench_v1 import runner
from experiments.longbench_v1.data import load_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    b.require(not args.output.exists(), 'Never overwrite a GPU diagnostic.')
    meta, budget, hashes = b.inspect(args.checkpoint, final=False)
    args.output.mkdir(parents=True)
    binding = dict(checkpoint_dir=str(args.checkpoint.resolve()), checkpoint_id=meta['checkpoint_id'],
        metadata_sha256=hashes[str(args.checkpoint.resolve() / 'adaptive_model_meta.json')],
        checkpoint_files_sha256=hashes, checkpoint_weight_stat=b.file_stat(args.checkpoint / 'pytorch_model.bin'), bit_budget=budget)
    config = b.make_config(binding, b.source_pins(binding), mode='validation')
    path = args.output / 'config.json'
    b.write_json(path, config)

    def load_and_check(config, logger):
        model, audit = load_model(config, logger)
        manifest, rows = load_archive(config['archive']['path'])
        import numpy as np
        tokens = np.memmap(manifest['inputs']['path'], dtype='<u4', mode='r')
        ids = runner.read_input(tokens, rows[0])[:, :8].to('cuda:0')
        with torch.inference_mode():
            logits = model(ids, use_cache=False, logits_to_keep=1).logits
        b.require(bool(torch.isfinite(logits).all()), 'Real model logits must be finite.')
        del tokens, ids, logits
        tokenizer = AutoTokenizer.from_pretrained(str(b.BASE), use_fast=True, local_files_only=True)
        model.config.use_cache = False
        api = b.module(b.SAMPLE_DIR / 'eval_utils_with_samples.py', '_adaptive_diagnostic_existing_hflm')
        lm_args = SimpleNamespace(tasks='rte', num_fewshot=0, batch_size='32', lm_limit=2, model_path=str(b.BASE),
            eval_log_dir=None, eval_run_ts=None, mmlu_debug_samples=0,
            mmlu_debug_log_dir=str(args.output / 'hflm_logs'), mmlu_debug_run_ts='diagnostic')
        result = api.run_lm_eval(model, tokenizer, lm_args)
        b.require(result['n_samples']['rte']['effective'] == 2 and result['sample_artifacts']['total_count'] == 2,
                  'Real two-example likelihood loop and raw samples must complete.')
        audit['compact_cache_builds_after_logits_and_likelihood'] = {
            item['name']: model.get_submodule(item['name']).cache_builds for item in meta['compact_modules']}
        b.require(all(count > 0 for count in audit['compact_cache_builds_after_logits_and_likelihood'].values()),
                  'Every actual compact consumer must have decoded its real persistent payload.')
        b.write_json(args.output / 'hflm_diagnostic.json', dict(checkpoint_id=meta['checkpoint_id'],
            checkpoint_format=b.FORMAT, diagnostic=True, quality_target_evaluated=False, finite_real_logits=True,
            load_audit=audit, lm_eval=result))
        model.config.use_cache = True
        return model, audit

    runner.check_config, runner.load_model = b.check_config, load_and_check
    report = runner.run(path, args.output / 'longbench')
    b.require(report['status'] == 'diagnostic_complete' and report['new_predictions'] == 3, 'Original real32K/KV/summary/SAMSum diagnostic failed.')
    b.write_json(args.output / 'report.json', dict(status='verified_gpu_mixed_inference_diagnostic',
        checkpoint_dir=str(args.checkpoint.resolve()), checkpoint_id=meta['checkpoint_id'], checkpoint_format=b.FORMAT,
        checkpoint_files_sha256=hashes, bit_budget=budget, longbench_report=report,
        hflm_diagnostic=str(args.output / 'hflm_diagnostic.json'), quality_target_evaluated=False,
        boundary='Finite real logits; two real RTE likelihood examples; canonical longest/512-token summary/SAMSum generated cases. This is no complete full8/full21 or final budget acceptance.'))
    print(json.dumps(dict(status='verified_gpu_mixed_inference_diagnostic', checkpoint_id=meta['checkpoint_id'])), flush=True)


if __name__ == '__main__':
    main()
