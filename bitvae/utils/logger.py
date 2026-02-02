# from https://github.com/FoundationVision/LlamaGen/blob/main/utils/logger.py
import logging
import glob
import os
import torch.distributed as dist


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()

    if rank == 0:  # real logger
        existing_logs = glob.glob(os.path.join(logging_dir, 'log_*.txt'))
        log_numbers = [int(log.split('.txt')[0].split('_')[-1]) for log in existing_logs]
        next_log_number = max(log_numbers) + 1 if log_numbers else 1
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log_{next_log_number}.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger
