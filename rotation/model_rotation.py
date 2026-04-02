import logging
from rotation.common import separate_embeddings_and_lm_head
from rotation import rotation_utils
from rotation import hadamard_utils
import math
import tqdm
try:
    import fast_hadamard_transform
except ImportError:
    fast_hadamard_transform = None


def prepare_model(model, rot_block_size=0):
    if model.config.tie_word_embeddings:  # 断开权重共享 针对 Llama-3.2
        logging.info("Tying word embeddings is not supported for rotation, disabling it.")
        separate_embeddings_and_lm_head(model)

    rotation_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model, rot_block_size=rot_block_size)
    rotation_utils.cleanup_memory(verbos=True)

    return model
