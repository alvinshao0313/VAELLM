import transformers
import transformers.models.llama.modeling_llama as llama_module
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rotary_pos_emb
import functools

# 保存原始函数
_original_apply_rotary_pos_emb = llama_apply_rotary_pos_emb


def capture_q_states_wrapper(original_func, q_states_storage, layer_idx):
    """
    包装apply_rotary_pos_emb函数来捕获q_states
    """
    def wrapper(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        # 调用原始函数
        q_embed, k_embed = original_func(q, k, cos, sin, position_ids, unsqueeze_dim)

        # 捕获q_state (经过位置编码后的q)
        if layer_idx is not None and f'layer_{layer_idx}_act_avg' in q_states_storage:
            # q_states_storage[f'layer_{layer_idx}_act_avg'].append(q_embed.clone().detach().cpu())
            bs, head_num, seq_len, hidden_size = q_embed.shape
            q_embed_reshape = q_embed.transpose(1, 2).reshape(bs, seq_len, -1)
            q_states_storage[f'layer_{layer_idx}_act_avg'] = (q_states_storage[f'layer_{layer_idx}_act_avg'] *
                                                              q_states_storage[f'layer_{layer_idx}_idx'] +
                                                              q_embed_reshape.view(-1, head_num * hidden_size).sum(dim=0, keepdim=True)) / \
                (q_states_storage[f'layer_{layer_idx}_idx'] + seq_len)
            q_states_storage[f'layer_{layer_idx}_idx'] += seq_len
        return q_embed, k_embed

    return wrapper


def apply_rotary_pos_emb_for_q_capture(q_states_storage, current_layer_idx):
    """
    临时patch apply_rotary_pos_emb函数
    """
    # 创建wrapper
    wrapper = capture_q_states_wrapper(
        _original_apply_rotary_pos_emb,
        q_states_storage,
        current_layer_idx
    )

    # 替换全局函数
    llama_module.apply_rotary_pos_emb = wrapper

    return wrapper
