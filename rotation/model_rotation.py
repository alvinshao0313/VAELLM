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


def rotate_pre_hook(rotate_mode):
    def hook(module, inp):
        # Hadamard transform (QuaRot)
        if rotate_mode.get("online_full_had", False):
            inp[0].data = hadamard_utils.matmul_hadU_cuda(
                inp[0].data, rotate_mode["had_K"], rotate_mode["K"])
        elif rotate_mode.get("online_partial_had", False):
            init_shape = inp[0].shape
            if rotate_mode["K"] == 1:
                inp[0].data = fast_hadamard_transform.hadamard_transform(
                    inp[0].data.reshape(-1, init_shape[-1] // rotate_mode["had_dim"],
                                        rotate_mode["had_dim"]).transpose(1, 2),
                    scale=1 /
                    math.sqrt(
                        init_shape[-1] // rotate_mode["had_dim"])
                ).transpose(1, 2)
            else:
                inp[0].data = (rotate_mode["had_K"].to(inp[0].dtype).to(inp[0].device) @ inp[0].data.reshape(-1,
                                                                                                             init_shape[-1] // rotate_mode["had_dim"], rotate_mode["had_dim"])) / math.sqrt(init_shape[-1] // rotate_mode["had_dim"])
            inp[0].data = inp[0].data.reshape(init_shape)
        elif rotate_mode.get("online_group_had", False):
            assert rotate_mode["had_dim"] > 0 and rotate_mode[
                "K"] == 1, "Group Hadamard transform requires had_dim > 0 and K == 1"
            # Group Hadamard transform
            init_shape = inp[0].shape
            inp[0].data = fast_hadamard_transform.hadamard_transform(
                inp[0].data.reshape(-1, init_shape[-1] //
                                    rotate_mode["had_dim"], rotate_mode["had_dim"]),
                scale=1 / math.sqrt(rotate_mode["had_dim"]))
            inp[0].data = inp[0].data.reshape(init_shape)
        return inp
    return hook


def register_rotate_hook(model, args):
    rotate_handles = []
    for name, mod in tqdm.tqdm(model.named_modules(), desc="Registering Rotate Hooks"):
        rotate_mode = {}
        if 'down_proj' in name and args.online_down_had:
            if args.rotate_mode == 'hadamard':
                had_K, K = hadamard_utils.get_hadK(
                    model.config.intermediate_size)
                rotate_mode["online_full_had"] = True
            elif args.rotate_mode == 'group_hadamard':
                had_K, K = hadamard_utils.get_hadK(
                    args.block_size_linear)
                rotate_mode["online_group_had"] = True
                rotate_mode["had_dim"] = args.block_size_linear
            rotate_mode["had_K"] = had_K
            rotate_mode["K"] = K
            rotate_handles.append(mod.register_forward_pre_hook(
                rotate_pre_hook(rotate_mode)))
        elif 'o_proj' in name and args.online_partial_had:
            had_K, K = hadamard_utils.get_hadK(
                model.config.num_attention_heads)
            rotate_mode["online_partial_had"] = True
            rotate_mode["had_K"] = had_K
            rotate_mode["K"] = K
            rotate_mode["had_dim"] = model.config.hidden_size // model.config.num_attention_heads
            rotate_handles.append(mod.register_forward_pre_hook(
                rotate_pre_hook(rotate_mode)))
    return rotate_handles


def prepare_model(model, args):
    if model.config.tie_word_embeddings:  # 断开权重共享 针对 Llama-3.2
        logging.info("Tying word embeddings is not supported for rotation, disabling it.")
        separate_embeddings_and_lm_head(model)

    rotation_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model, args)
    rotation_utils.cleanup_memory(verbos=True)
    rotate_handles = register_rotate_hook(model, args)

    return model, rotate_handles


def prepare_model4eval(model, args):
    if model.config.tie_word_embeddings:  # 断开权重共享 针对 Llama-3.2
        logging.info("Tying word embeddings is not supported for rotation, disabling it.")
        separate_embeddings_and_lm_head(model)
    rotate_handles = register_rotate_hook(model, args)

    return model, rotate_handles


def remove_rotate_hooks(rotate_handles):
    for handle in rotate_handles:
        handle.remove()
