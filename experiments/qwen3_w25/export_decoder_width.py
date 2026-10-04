"""Export a saved local decoder-width endpoint as a complete BF16 model."""
import argparse
import json

from experiments.qwen3_w25.decoder_width_export import export_width_endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = export_width_endpoint(args.diagnostic, args.output)
    print(json.dumps({"status": result["status"], "checkpoint_id": result["checkpoint_id"],
                      "user_linear_bpw": result["bit_budget"]["user_linear_bpw"]}), flush=True)


if __name__ == "__main__":
    main()
