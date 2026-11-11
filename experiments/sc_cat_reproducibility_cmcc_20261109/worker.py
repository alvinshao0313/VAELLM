"""Run the frozen legacy worker with deterministic FA2 backward bound at import."""
from .execution import configure_deterministic_backward


def main():
    policy = configure_deterministic_backward()
    print({'reproducibility_execution': policy}, flush=True)
    from experiments.sc_cat_finish_cmcc_20261108.worker import main as frozen_worker
    frozen_worker()


if __name__ == '__main__':
    main()
