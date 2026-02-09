import torch
import torch.nn as nn
import torch.nn.functional as F
from litebsq.llm_vae import MultiLayerVAE
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM, LlamaDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3ForCausalLM, Qwen3DecoderLayer


class BSQLinear(nn.Module):
    def __init__(self, linear: nn.Linear, vae_params: dict, transpose=False):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.transpose = transpose  # nn.Linear 的权重默认是 (Out, In)
        parallel_layers = getattr(vae_params, "parallel_layers", 1)
        distil_loss_type = getattr(vae_params, "distil_loss_type", "mse")
        distil_loss_weight = getattr(vae_params, "distil_loss_weight", 1.0)
        norm_weight = bool(getattr(vae_params, "normalize_weight", True))
        self.parallel_layers = parallel_layers
        if transpose:
            assert self.in_features % parallel_layers == 0, "In features must be divisible by parallel_layers for transposed weights."
        else:
            assert self.out_features % parallel_layers == 0, "Out features must be divisible by parallel_layers for non-transposed weights."
        self._out_per = self.out_features // parallel_layers
        self._in_per = self.in_features // parallel_layers
        self._weight_view_shape = (
            parallel_layers,
            self._out_per if not transpose else self._in_per,
            self.in_features if not transpose else self.out_features,
        )
        self._transpose = transpose
        self.weight = linear.weight
        self.weight.requires_grad = False  # 确保原始权重不被训练

        if linear.bias is not None:
            self.bias = linear.bias
            self.bias.requires_grad = False  # 确保偏置不被训练
        else:
            self.register_parameter('bias', None)
        self.distil_loss_type = distil_loss_type
        self.distil_loss_weight = distil_loss_weight
        self.w_input_batches = max(1, int(getattr(vae_params, "w_input_batches", 1)))

        self.vae = MultiLayerVAE(vae_params).to(self.weight.device)
        assert self.vae.num_models == self.parallel_layers, "VAE num_models must match parallel_layers."
        self.register_buffer('codebook_dim', torch.tensor(self.vae.model.chunk_size))

        w_flatten_len = self._weight_view_shape[1] * self._weight_view_shape[2]
        assert w_flatten_len % int(self.codebook_dim.item(
        )) == 0, "Weight size not compatible with VAE input dimension."

        if norm_weight:
            self._normalize_weight(self.weight)
        else:
            self.register_buffer('d_mean', torch.zeros(self.parallel_layers, 1, dtype=self.weight.dtype))
            self.register_buffer('d_std', torch.ones(self.parallel_layers, 1, dtype=self.weight.dtype))

    def parallel_weight(self, weight):
        if self._transpose:
            weight = weight.t()
        weight = weight.reshape(self._weight_view_shape)
        w_flatten = weight.reshape(self.parallel_layers, -1)
        stacked_flat = (w_flatten - self.d_mean) / (self.d_std + 1e-6)
        stacked_data = stacked_flat.reshape(stacked_flat.shape[0], -1, self.codebook_dim).permute(1, 0, 2)
        return stacked_data

    def recover_weight(self, stacked_data):
        stacked_flat = stacked_data.permute(1, 0, 2).contiguous()  # stacked_data: (N_block, P, codebook_dim)
        w_flatten = stacked_flat.reshape(self.parallel_layers, -1)  # shape: (P, N_block, codebook_dim)
        w_flatten = w_flatten * (self.d_std + 1e-6) + self.d_mean      # undo normalization

        if self._transpose:
            weight = w_flatten.reshape(
                self.parallel_layers,
                self._in_per,
                self.out_features
            )
            weight = weight.reshape(self.in_features, self.out_features)
            weight = weight.t()
        else:
            weight = w_flatten.reshape(
                self.parallel_layers,
                self._out_per,
                self.in_features
            )
            weight = weight.reshape(self.out_features, self.in_features)

        return weight

    def _set_mode(self, is_train: bool):
        self.is_train = is_train
        if not is_train:
            self.vae.eval()
            with torch.no_grad():
                w_input = self.parallel_weight(self.weight)
                if self.w_input_batches > 1 and w_input.shape[0] > 1:
                    vq_chunks = []
                    total = w_input.shape[0]
                    chunk_size = (total + self.w_input_batches - 1) // self.w_input_batches
                    for start in range(0, total, chunk_size):
                        _, vq_chunk = self.vae(w_input[start:start + chunk_size], is_train=False)
                        vq_chunks.append(vq_chunk)
                    vq_w = torch.cat(vq_chunks, dim=0)
                else:
                    _, vq_w = self.vae(w_input, is_train=False)
            self.register_buffer('vq_weight', vq_w)
            self.decoder = self.vae.model.decoder
            self.decoder._fuse_q_scale()
            del self.vae, self.weight

    def _normalize_weight(self, weight):
        if self.transpose:
            weight = weight.t()
        weight = weight.reshape(self.parallel_layers, weight.shape[0] // self.parallel_layers, weight.shape[1])
        w_flatten = weight.reshape(self.parallel_layers, -1)
        self.register_buffer('d_mean', w_flatten.mean(dim=1, keepdim=True).to(weight.dtype))
        self.register_buffer('d_std', w_flatten.std(dim=1, keepdim=True).to(weight.dtype))

    def forward(self, x):
        # 执行线性变换
        if self.is_train:
            w_input = self.parallel_weight(self.weight)
            if self.w_input_batches > 1 and w_input.shape[0] > 1:
                recons = []
                loss_dict = {}
                total = w_input.shape[0]
                chunk_size = (total + self.w_input_batches - 1) // self.w_input_batches
                for start in range(0, total, chunk_size):
                    chunk = w_input[start:start + chunk_size]
                    w_recon_chunk, loss_chunk = self.vae(chunk, is_train=self.is_train)
                    recons.append(w_recon_chunk)
                    for key, value in loss_chunk.items():
                        if key in loss_dict:
                            loss_dict[key] += value
                        else:
                            loss_dict[key] = value
                w_recon = torch.cat(recons, dim=0)
            else:
                w_recon, loss_dict = self.vae(w_input, is_train=self.is_train)
        else:
            w_recon = self.decoder(self.vq_weight.to(x.dtype))

        w_recon = self.recover_weight(w_recon)

        out = F.linear(x, w_recon, self.bias)
        # TODO: 目前推理时还需保持LLM内部forward是替换后的，因此还需返回一个空的loss_dict
        if not self.is_train:
            return out, {}

        with torch.no_grad():
            orgi_out = F.linear(x, self.weight, self.bias)

        if self.distil_loss_type == 'mse':
            distil_loss = F.mse_loss(orgi_out, out)
        else:
            distil_loss = 0.

        loss_dict.update({'distil_loss': distil_loss})
        loss_dict["loss"] += distil_loss * self.distil_loss_weight
        return out, loss_dict  # 输出重建out用于蒸馏


def set_module_by_name(model, name, new_module):
    """Helper function to set a module by its full name (e.g., 'layers.0.self_attn.q_proj')."""
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def insert_bsq_linear(llm, transpose_modules, args):
    params_list = []
    if isinstance(llm, LlamaForCausalLM) or isinstance(llm, Qwen3ForCausalLM):
        model = llm.model
    elif isinstance(llm, LlamaModel) or isinstance(llm, Qwen3Model):
        model = llm
    elif isinstance(llm, LlamaDecoderLayer) or isinstance(llm, Qwen3DecoderLayer):
        model = llm
    else:
        raise ValueError("Model must be LlamaForCausalLM, Qwen3ForCausalLM, LlamaModel, or Qwen3Model.")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name.split('.')[-1] not in transpose_modules:
                transpose = False
            else:
                transpose = True

            bsq_linear = BSQLinear(
                linear=module,
                vae_params=args,
                transpose=transpose,
            )
            bsq_linear._set_mode(is_train=True)
            set_module_by_name(model, name, bsq_linear)
            params_list.extend(list(bsq_linear.vae.parameters()))

    return params_list


def bsq_turn2infra(llm):
    if isinstance(llm, LlamaForCausalLM) or isinstance(llm, Qwen3ForCausalLM):
        model = llm.model
    elif isinstance(llm, LlamaModel) or isinstance(llm, Qwen3Model):
        model = llm
    elif isinstance(llm, LlamaDecoderLayer) or isinstance(llm, Qwen3DecoderLayer):
        model = llm
    else:
        raise ValueError("Model must be LlamaForCausalLM, Qwen3ForCausalLM, LlamaModel, or Qwen3Model.")
    for module in model.modules():
        if isinstance(module, BSQLinear):
            module._set_mode(is_train=False)
    return
