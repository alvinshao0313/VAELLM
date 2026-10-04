"""Run the isolated full-model KL batch-gradient diagnostic."""
import argparse
from experiments.qwen3_w25.kl_gradient_diagnostic import run_kl_gradient_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal_launch", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    run_kl_gradient_diagnostic(args.formal_launch, args.output_dir)


if __name__ == "__main__":
    main()
