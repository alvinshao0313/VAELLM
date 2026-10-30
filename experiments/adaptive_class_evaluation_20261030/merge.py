"""Use the established full21 merger with actual mixed config validation."""
import argparse

from experiments.adaptive_class_evaluation_20261030 import bindings as b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    api = b.module(b.TOPIC / 'llama_fullqk_qkonly2000_longbench21/merge_full21.py', '_adaptive_canonical_full21_merge')
    api.check_config = b.check_config
    api.main(args.plan)


if __name__ == '__main__':
    main()
