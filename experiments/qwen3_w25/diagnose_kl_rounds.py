"""Profile or observe the isolated existing pure-bit KL round lifecycle."""
import argparse
from experiments.qwen3_w25.kl_round_diagnostic import run_kl_round_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal_launch", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, choices=(8, 256), default=8)
    parser.add_argument("--round_steps", type=int, choices=(128, 512), default=128)
    parser.add_argument("--data_cache")
    parser.add_argument("--save_final_codes", action="store_true")
    args = parser.parse_args()
    run_kl_round_diagnostic(args.formal_launch, args.output_dir, steps=args.steps,
                            round_steps=args.round_steps, data_cache=args.data_cache,
                            save_final_codes=args.save_final_codes)


if __name__ == "__main__":
    main()
