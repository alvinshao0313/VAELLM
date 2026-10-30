"""The established HFLM+raw-sample evaluator with an explicit mixed v2 loader."""
import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
import time

from transformers import AutoTokenizer

from experiments.adaptive_class_evaluation_20261030 import bindings as b
from experiments.adaptive_class_evaluation_20261030.loading import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--shard', required=True, choices=('mmlu', 'seven'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    b.require(not args.output.exists(), 'Never overwrite a full8 shard.')
    config = b.read(args.config)
    b.require(config['mode'] == 'full', 'Full8 has no limit or diagnostic mode.')
    logger = logging.getLogger('adaptive_full8')
    model, audit = load_model(config, logger)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(str(b.BASE), use_fast=True, trust_remote_code=True, local_files_only=True)
    api = b.module(b.SAMPLE_DIR / 'eval_utils_with_samples.py', '_adaptive_existing_hflm_samples')
    logs = args.output.parent / (args.shard + '_logs')
    lm_args = SimpleNamespace(tasks='mmlu' if args.shard == 'mmlu' else b.SEVEN, num_fewshot=0,
        batch_size='64' if args.shard == 'mmlu' else '32', lm_limit=None, model_path=str(b.BASE),
        eval_log_dir=None, eval_run_ts=None, mmlu_debug_samples=0,
        mmlu_debug_log_dir=str(logs), mmlu_debug_run_ts=time.strftime('%Y%m%d_%H%M%S'))
    result = api.run_lm_eval(model, tokenizer, lm_args)
    payload = dict(checkpoint_dir=config['model']['checkpoint'], checkpoint_id=audit['checkpoint_id'],
        checkpoint_loader=b.FORMAT, checkpoint_format=b.FORMAT, base_model_path=str(b.BASE), strict_load=True,
        adapter_dir=None, eval_hif4_act=False, tokenizer_class=type(tokenizer).__name__, tokenizer_is_fast=tokenizer.is_fast,
        checkpoint_files_sha256=config['model']['checkpoint_files_sha256'], load_audit=audit,
        inference=dict(use_cache=False, model_seqlen=model.seqlen,
            max_position_embeddings=model.config.max_position_embeddings, num_fewshot=0, limit=None),
        evals=dict(lm_eval=result))
    b.check_config(config)
    b.write_json(args.output, payload)


if __name__ == '__main__':
    main()
