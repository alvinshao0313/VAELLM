# Copyright (c) Foundationvision, Inc. All Rights Reserved

import torch
import torch.distributed as dist
import imageio
import os
import random

import math
import numpy as np
# import skvideo.io
from einops import rearrange
import torch.optim as optim

from contextlib import contextmanager

ptdtype = {None: torch.float32, 'fp32': torch.float32, 'bf16': torch.bfloat16}


def rank_zero_only(fn):
    def wrapped_fn(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return fn(*args, **kwargs)
    return wrapped_fn


def is_torch_optimizer(obj):
    return isinstance(obj, optim.Optimizer)


def rearranged_forward(x, func):
    x = rearrange(x, "B C H W -> B H W C")
    x = func(x)
    x = rearrange(x, "B H W C -> B C H W")
    return x


def is_dtype_16(data):
    return data.dtype == torch.float16 or data.dtype == torch.bfloat16


@contextmanager
def set_tf32_flags(flag):
    old_matmul_flag = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_flag = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = flag
    torch.backends.cudnn.allow_tf32 = flag
    try:
        yield
    finally:
        # Restore the original flags
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_flag
        torch.backends.cudnn.allow_tf32 = old_cudnn_flag


def get_last_ckpt(root_dir):
    if not os.path.exists(root_dir):
        return None
    ckpt_files = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.ckpt'):
                num_iter = int(filename.split('.ckpt')[0].split('_')[-1])
                ckpt_files[num_iter] = os.path.join(dirpath, filename)
    iter_list = list(ckpt_files.keys())
    if len(iter_list) == 0:
        return None
    max_iter = max(iter_list)
    return ckpt_files[max_iter]
