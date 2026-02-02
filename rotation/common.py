import numpy as np
import torch
import argparse
import logging
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif v.lower() in ('None'):
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(v):
    if v is None or v.lower() in ('none'):
        return []
    vv = v.split(',')
    ret = []
    for vvv in vv:
        ret.append(vvv)
    return ret


def str2intlist(v):
    vv = v.split(',')
    ret = []
    for vvv in vv:
        ret.append(int(vvv))
    return ret


def str2int(v):
    if v.lower() in ('none'):
        return None
    else:
        return int(v)


def str2path(v):
    if v is None or v.lower() in ('none'):
        return None
    else:
        return str(v)


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def distribute_model(model) -> None:
    """Distribute the model across available GPUs. NB: only implemented for Llama-2."""
    from rotation import model_utils
    if model_utils.get_model_type(model) == model_utils.LLAMA_MODEL:
        no_split_module_classes = ['LlamaDecoderLayer']
    elif model_utils.get_model_type(model) == model_utils.MISTRAL_MODEL:
        no_split_module_classes = ['MistralDecoderLayer']
    elif model_utils.get_model_type(model) == model_utils.QWEN2_MODEL:
        no_split_module_classes = ['Qwen2DecoderLayer']
    elif model_utils.get_model_type(model) == model_utils.QWEN3_MODEL:
        no_split_module_classes = ['Qwen3DecoderLayer']
    else:
        raise ValueError(f"Unsupported model type: {model_utils.get_model_type(model)}")
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    cleanup_memory()


import os
from tqdm import tqdm


def save_model_in_parts(model, save_qmodel_path, prefix='model_part', num_digits=5, target_file_size=10 * 1024**3):
    """
    将模型按文件大小（例如 10GB）分块保存为多个文件。
    :param model: 要保存的 PyTorch 模型
    :param save_qmodel_path: 模型保存路径
    :param target_file_size: 每个文件的目标大小（单位：字节，默认10GB）
    :param prefix: 文件名前缀（默认 'model_part'）
    :param num_digits: 文件序号的位数（默认 5 位数）
    """
    state_dict = model.state_dict()

    # 计算模型每个参数的大小（根据参数的数据类型自动计算）
    total_size = 0
    for param in state_dict.values():
        param_size = param.element_size() * param.numel()  # 获取参数的大小（单位字节）
        total_size += param_size

    # 计算分块数量
    num_parts = (total_size + target_file_size - 1) // target_file_size  # 向上取整

    logging.info(
        f"模型总大小: {total_size / (1024**3):.2f} GB, 分为 {num_parts} 部分，每个部分约 {target_file_size / (1024**3):.2f} GB")

    # 分块保存模型
    idx = 0
    current_part_size = 0  # 当前分块的实际字节大小
    part = {}

    for name, param in state_dict.items():
        param_size = param.element_size() * param.numel()  # 计算当前参数的字节数
        current_part_size += param_size  # 累加当前分块的大小
        part[name] = param  # 添加当前的参数到部分分块中

        # 如果当前块的大小超过目标大小，保存并开始新的块
        if current_part_size >= target_file_size:
            # 格式化文件名，确保序号是 5 位数
            part_filename = f"{prefix}_{str(idx).zfill(num_digits)}.pth"
            torch.save(part, os.path.join(save_qmodel_path, part_filename))
            logging.info(f"保存了模型的第 {idx + 1} 部分：{part_filename}，共 {num_parts} 部分。")
            part = {}  # 清空当前部分，开始下一个分块
            current_part_size = 0  # 重置当前分块的大小
            idx += 1

    # 最后一块
    if part:
        part_filename = f"{prefix}_{str(idx).zfill(num_digits)}.pth"
        torch.save(part, os.path.join(save_qmodel_path, part_filename))
        logging.info(f"保存了模型的第 {idx + 1} 部分：{part_filename}，共 {num_parts} 部分。")

    logging.info("模型分块保存完成。")


def load_model_in_parts(model, folder_path):
    """
    将模型的多个部分加载并逐块赋值给模型。
    :param model: 要加载的 PyTorch 模型
    :param folder_path: 存储分块模型文件的文件夹路径
    """
    # 获取文件夹中的所有 .pth 文件，按名称排序（确保加载顺序正确）
    model_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pth')])

    # 逐个加载分块
    with tqdm(total=len(model_files), desc="Loading Model Parts", unit="part") as pbar:
        for file_name in model_files:
            part = torch.load(os.path.join(folder_path, file_name), map_location='cpu')  # 加载分块

            model.load_state_dict(part, strict=False)  # 更新模型的参数（实时赋值）

            del part  # 释放已加载分块的内存
            pbar.update(1)  # 更新进度条

    logging.info("模型加载完成。")

# 分离embeddings和lm_head的函数


def separate_embeddings_and_lm_head(model):
    import torch.nn as nn
    model.config.tie_word_embeddings = False
    if hasattr(model, 'lm_head') and hasattr(model, 'model'):
        if hasattr(model.model, 'embed_tokens'):
            # 获取嵌入层的权重
            embed_weight = model.model.embed_tokens.weight.data.clone()

            # 确保 lm_head 有独立的权重
            if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
                # 权重共享，需要断开
                model.lm_head.weight = nn.Parameter(embed_weight.clone())
                logging.info("Successfully disconnected tied word embeddings")
            else:
                logging.info("Word embeddings are already disconnected")
        else:
            logging.warning("Could not find embed_tokens layer")
    else:
        logging.warning("Could not find lm_head or model layers")
