"""Export a genuine deterministic KL replay's two final code banks."""
import argparse
import json

from experiments.qwen3_w25.kl_round_export import export_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--reference_diagnostic", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = export_replay(args.diagnostic, args.reference_diagnostic, args.output)
    print(json.dumps({"status": result["status"], "checkpoint_id": result["checkpoint_id"],
                      "user_linear_bpw": result["bit_budget"]["user_linear_bpw"]}), flush=True)


if __name__ == "__main__":
    main()
