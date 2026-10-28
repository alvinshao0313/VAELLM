"""Use the original endpoint callback and immutable 2000-step training schedule."""
from __future__ import annotations

import argparse
from pathlib import Path

import common as c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['C', 'G'], required=True)
    parser.add_argument('--recipe', choices=['decoder', 'last_joint'], required=True)
    parser.add_argument('--source')
    args = parser.parse_args()
    c.check_prepared()
    expected = c.SOURCES[args.profile] if args.recipe == 'decoder' else c.DATA / args.profile / '08_joint_mlp_decoder_attention/final_model'
    actual = c.SOURCES[args.profile] if args.source is None else Path(args.source).resolve()
    c.require(actual == expected, 'Each fixed recipe must use its actual same-source05 or08 endpoint.')
    from experiments.qwen3_w25.train_reproduction_stage import main as native_stage
    c.native_context()
    return native_stage(['--finish_step', '2000' if args.recipe == 'decoder' else '1200',
                         *c.training_argv(args.profile, args.recipe, actual)])


if __name__ == '__main__':
    main()
