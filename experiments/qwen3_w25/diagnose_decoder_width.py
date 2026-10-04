"""Run one isolated H128/H256 decoder-capacity profile or matched KL arm."""
import argparse

from experiments.qwen3_w25.decoder_width_diagnostic import run_decoder_width_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal_launch", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_cache", required=True)
    parser.add_argument("--hidden_dim", required=True, type=int, choices=(128, 256))
    parser.add_argument("--steps", type=int, choices=(8, 256), default=8)
    args = parser.parse_args()
    run_decoder_width_diagnostic(args.formal_launch, args.output_dir, hidden_dim=args.hidden_dim,
                                 data_cache=args.data_cache, steps=args.steps)


if __name__ == "__main__":
    main()
