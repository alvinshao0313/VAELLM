from rotation import model_utils
import torch
import typing
import transformers
import tqdm
import math
from rotation.hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2
from fast_hadamard_transform import hadamard_transform
import logging


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)


def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model):

    model_type = model_utils.get_model_type(model)

    kwargs = {'model': model, 'model_type': model_type}

    # Embedding fusion
    for W in model_utils.get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = model_utils.get_transformer_layers(**kwargs)

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:

        # fuse the input layernorms into the linear layers
        if model_type in model_utils.SAME_TYPE_MODELS:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj,
                           layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj,
                           layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')

        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)

    fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])

    if model_type == model_utils.LLAMA_MODEL:
        model_utils.replace_modules(
            model,
            transformers.models.llama.modeling_llama.LlamaRMSNorm,
            lambda _: model_utils.RMSN(model.config.hidden_size),
            replace_layers=False,
        )
    elif model_type == model_utils.MISTRAL_MODEL:
        model_utils.replace_modules(
            model,
            transformers.models.mistral.modeling_mistral.MistralRMSNorm,
            lambda _: model_utils.RMSN(model.config.hidden_size),
            replace_layers=False,
        )
    elif model_type == model_utils.QWEN2_MODEL:
        model_utils.replace_modules(
            model,
            transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm,
            lambda _: model_utils.QwenRMSN(model.config.hidden_size),
            replace_layers=False,
        )
    elif model_type == model_utils.QWEN3_MODEL:
        model_utils.replace_modules(
            model,
            transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm,
            lambda _: model_utils.QwenRMSN(model.config.hidden_size),
            replace_layers=False,
            expect_list=['q_norm', 'k_norm']  # do not replace these small RMSNorms
        )
    else:
        model_utils.replace_modules(
            model,
            torch.nn.LayerNorm,
            lambda _: model_utils.RMSN(model.config.hidden_size),
            replace_layers=False,
        )


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def group_hadamard_matrix(full_size, had_dim, device):
    assert is_pow2(had_dim), "group_size must be a power of 2"
    had = random_hadamard_matrix(had_dim, device)
    group_had = torch.zeros((full_size, full_size), dtype=torch.float64, device=device)
    for i in range(0, full_size, had_dim):
        group_had[i:i + had_dim, i:i + had_dim] = had
    return group_had


def get_orthogonal_matrix(size, mode, device="cuda:0", **kwargs):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    elif mode == 'group_hadamard':
        assert kwargs.get('had_dim', False), "had_dim must be specified for group hadamard mode"
        had_dim = kwargs['had_dim']
        return group_hadamard_matrix(size, had_dim, device)
    elif mode == 'identity':
        return torch.eye(size, dtype=torch.float64, device=device)
    else:
        raise ValueError(f'Unknown mode {mode}')


def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda:0", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device="cuda:0", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type in model_utils.SAME_TYPE_MODELS:
        W = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda:0", dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda:0", dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, Q, model_type, **kwargs):
    # Rotate the MLP input weights.
    if model_type in model_utils.SAME_TYPE_MODELS:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device="cuda:0", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q)
        W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, Q, model_type, args, **kwargs):
    # Rotate the MLP output weights and bias.
    if model_type in model_utils.SAME_TYPE_MODELS:
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda:0", dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_)
    if not args.online_down_had:
        pass
    elif args.rotate_mode == 'hadamard':
        # apply exact (inverse) hadamard on the weights of mlp output
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
    elif args.rotate_mode == 'group_hadamard':
        W_ = W.weight.data.to(device="cuda:0", dtype=torch.float32)
        init_shape = W_.shape
        had_dim = args.had_dim
        W.weight.data = hadamard_transform(W_.reshape(-1, init_shape[-1] // had_dim,
                                                      had_dim), scale=1 / math.sqrt(had_dim)).reshape(init_shape)
    W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda:0", dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def matmul_hadU_cuda_had(X, hadK, transpose=False):
    '''
    Apply hadamard transformation.
    It reshapes X and applies Walsh-Hadamard transform to the last dimension.
    Then, it will multiply the retult by another hadamard matrix.
    '''
    from fast_hadamard_transform import hadamard_transform
    from rotation.hadamard_utils import get_had172
    n = X.shape[-1]
    K = hadK.shape[-1]

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().cuda().view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), scale=1 / math.sqrt(n))
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.to(X.device).to(X.dtype).reshape(
        X.shape)

# def rotate_faster_down_proj(layer, model_type, hardK):
#    from fast_hadamard_transform import hadamard_transform
#    if model_type == model_utils.LLAMA_MODEL:
#        W = layer.mlp.down_proj
#    elif model_type == model_utils.QWEN2_MODEL:
#        W = layer.mlp.down_proj
#    else:
#        raise ValueError(f'Faster MLP is onlu supported for LLaMa models!')
#
#    dtype = W.weight.data.dtype
#    W.weight.data = matmul_hadU_cuda_had(W.weight.data.float().cuda(), hardK)
#    W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)


def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda:0", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, model_type, head_num, head_dim, args, **kwargs):
    v_proj = layer.self_attn.v_proj
    if model_type in model_utils.SAME_TYPE_MODELS:
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        o_proj = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    if args.online_partial_had:
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
        apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)
    else:
        assert kwargs.get('Q2', None) is not None, "Q2 must be specified for group hadamard mode"
        Q2 = kwargs['Q2']
        apply_multi_head_rotate(v_proj, Q2, head_dim, head_num, output=True, **kwargs)
        apply_multi_head_rotate(o_proj, Q2, head_dim, head_num, output=False, **kwargs)


def apply_multi_head_rotate(module, Q, head_dim, head_num, output=False, **kwargs):
    assert isinstance(module, torch.nn.Linear)
    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    W_ = W_.to(device="cuda:0", dtype=torch.float64)

    if output:
        W_ = W_.t()
        transposed_shape = W_.shape
        W_ = W_.reshape(-1, head_num, head_dim).transpose(0, 1)
        W_ = torch.matmul(W_, Q)
        W_ = W_.transpose(0, 1).reshape(transposed_shape).t()
        if module.bias is not None:
            b = module.bias.data.to(device="cuda:0", dtype=torch.float64)
            # b = b[kwargs['sorting_idx']] if kwargs.get('reflow', False) else b
            b_ = b.reshape(head_num, head_dim)
            b_ = torch.matmul(b_, Q)
            b_ = b_.reshape(-1)
            module.bias.data = b_.to(device=dev, dtype=dtype)
    else:
        W_ = W_.reshape(-1, init_shape[1] // head_dim,
                        head_dim).transpose(0, 1)
        W_ = torch.matmul(W_, Q)
        W_ = W_.transpose(0, 1).reshape(init_shape)

    module.weight.data = W_.to(device=dev, dtype=dtype)


@torch.inference_mode()
def rotate_model(model, args):
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    kv_head = config.num_key_value_heads

    kwargs = {'had_dim': args.had_dim} if args.rotate_mode == 'group_hadamard' else {}
    if args.r1_path is not None:
        logging.info(f"Loading R1 from {args.r1_path}")
        Q1 = torch.load(args.r1_path, map_location=torch.device('cuda:0'))['R1'].to(dtype=torch.float64)
    else:
        Q1 = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode, device="cuda:0", **kwargs)

    Q2 = get_orthogonal_matrix(head_dim, args.rotate_mode, device="cuda:0", **
                               kwargs) if not args.online_partial_had else None

    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q1)
    rotate_head(model, Q1)
    cleanup_memory()
    layers = model_utils.get_transformer_layers(model,
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Fuse and Rotate Layers")):
        rotate_attention_inputs(layers[idx], Q1, model_type)
        rotate_attention_output(layers[idx], Q1, model_type)
        rotate_mlp_input(layers[idx], Q1, model_type,)
        rotate_mlp_output(layers[idx], Q1, model_type, args,)
        rotate_ov_proj(layers[idx], model_type, kv_head, head_dim, args,
                       **{'Q2': Q2, })


@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]


def register_online_rotation(module, Q: torch.Tensor):
    assert not hasattr(module, 'Q')
    module.register_buffer('Q', Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)


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


class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, config, had_dim: int = -1, *args, **kwargs):
        super().__init__()
        self.config = config
        self.func = func
        self.had_dim = had_dim

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q_shape = q.shape
        (bsz, num_heads, seq_len, head_dim) = k.shape
        if self.had_dim > 0:
            # Apply hadamard transform to q and k
            q = hadamard_transform(q.reshape(bsz, -1, seq_len, head_dim // self.had_dim,
                                   self.had_dim).float(), scale=1 / math.sqrt(self.had_dim)).to(dtype).reshape(q_shape)
            k = hadamard_transform(k.reshape(bsz, -1, seq_len, head_dim // self.had_dim,
                                   self.had_dim).float(), scale=1 / math.sqrt(self.had_dim)).to(dtype).reshape(bsz, num_heads, seq_len, head_dim)
        else:
            # Apply standard hadamard transform
            q = hadamard_transform(q.float(), scale=1 / math.sqrt(q.shape[-1])).to(dtype)
            k = hadamard_transform(k.float(), scale=1 / math.sqrt(k.shape[-1])).to(dtype)
        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    from rotation import monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module, "forward", function_name, functools.partial(QKRotationWrapper, *args, **kwargs)
    )
    setattr(module, attr_name, wrapper)
