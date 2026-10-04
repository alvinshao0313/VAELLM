"""Run the isolated decoder-only or joint L0-down KL bridge."""
import argparse

from experiments.qwen3_w25.kl_joint_diagnostic import run_joint_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal_launch", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_cache", required=True)
    parser.add_argument("--pure_bit_reference", required=True)
    parser.add_argument("--mode", required=True, choices=("decoder", "joint"))
    parser.add_argument("--steps", type=int, choices=(8, 256), default=256)
    args = parser.parse_args()
    run_joint_diagnostic(args.formal_launch, args.output_dir, args.data_cache, args.pure_bit_reference,
                         joint=args.mode == "joint", steps=args.steps)


if __name__ == "__main__":
    main()
