import transformers
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaDecoderLayer, LlamaModel, LlamaForCausalLM, LlamaMLP
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, Qwen3MLP, Qwen3DecoderLayer, Qwen3Model, Qwen3ForCausalLM
)
from llm_models._forward import (
    sa_forward, fclm_forward, mlp_forward, model_forward, decoder_forward,
    QWEN_TAG
)
from types import MethodType


def rebuild_llama_forward(model):
    QWEN_TAG = False
    assert isinstance(model, LlamaForCausalLM), "Model must be LlamaForCausalLM"
    if isinstance(model, LlamaForCausalLM):
        model.forward = MethodType(fclm_forward, model)
        model = model.model
    model.forward = MethodType(model_forward, model)
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            module.forward = MethodType(decoder_forward, module)
        elif isinstance(module, LlamaAttention):
            module.forward = MethodType(sa_forward, module)
        elif isinstance(module, LlamaMLP):
            module.forward = MethodType(mlp_forward, module)


def rebuild_qwen3_forward(model):
    QWEN_TAG = True
    assert isinstance(model, Qwen3ForCausalLM), "Model must be Qwen3ForCausalLM"
    if isinstance(model, Qwen3ForCausalLM):
        model.forward = MethodType(fclm_forward, model)
        model = model.model
    model.forward = MethodType(model_forward, model)
    for name, module in model.named_modules():
        if isinstance(module, Qwen3DecoderLayer):
            module.forward = MethodType(decoder_forward, module)
        elif isinstance(module, Qwen3Attention):
            module.forward = MethodType(sa_forward, module)
        elif isinstance(module, Qwen3MLP):
            module.forward = MethodType(mlp_forward, module)
