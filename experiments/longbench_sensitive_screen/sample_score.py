"""CPU H50 partial scorer; run in a fresh PUBLIC process."""
import argparse
import json

from sample_core import score_directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--prediction-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = score_directory(args.manifest, args.prediction_dir, args.output)
    print(json.dumps({key: report[key] for key in ('status', 'kind', 'is_full_benchmark', 'sample_count',
        'h50_macro_mean', 'paired_source_h50_delta_pp', 'duration_seconds')}))
