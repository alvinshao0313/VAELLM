import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from bitvae.models.llm_vae import MultiLayerVAE
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3ForCausalLM


class BSQLinear(nn.Module):
    def __init__(self, linear: nn.Linear, transpose=False,
                 parallel_layers=1, distil_loss_type='mse',
                 distil_loss_weight=1.0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.transpose = transpose  # nn.Linear 的权重默认是 (Out, In)
        self.parallel_layers = parallel_layers
        if transpose:
            assert self.out_features % parallel_layers == 0, "Out features must be divisible by parallel_layers for transposed weights."
        else:
            assert self.in_features % parallel_layers == 0, "In features must be divisible by parallel_layers for non-transposed weights."
        # 将原始权重注册为 Buffer，确保它能随模型移动设备 (to cuda)，但不作为可训练参数
        self.register_buffer('weight', linear.weight.data.clone())

        self.register_parameter('bias', linear.bias.data.clone() if linear.bias is not None else None)
        self.distil_loss_type = distil_loss_type
        self.distil_loss_weight = distil_loss_weight
        self.vae = None

    def set_vae(self, vae, normalize_weight=True):
        assert self.vae is None, "VAE has already been initialized."
        assert vae.num_models == self.parallel_layers, "VAE num_models must match parallel_layers."
        self.vae = vae
        self.register_buffer('codebook_dim', torch.tensor(self.vae.model.chunk_size))
        if normalize_weight:
            self._normalize_weight(self.weight)
        else:
            self.register_buffer('d_mean', torch.zeros(self.parallel_layers, 1))
            self.register_buffer('d_std', torch.ones(self.parallel_layers, 1))

    def parallel_weight(self, weight):
        if self.transpose:
            weight = weight.t()
        weight = weight.reshape(self.parallel_layers, weight.shape[0] // self.parallel_layers, weight.shape[1])
        w_flatten = weight.reshape(self.parallel_layers, -1)
        stacked_flat = (w_flatten - self.d_mean) / (self.d_std + 1e-6)
        assert w_flatten.shape[1] % self.codebook_dim == 0, "Weight size not compatible with VAE input dimension."
        stacked_data = stacked_flat.reshape(stacked_flat.shape[0], -1, self.codebook_dim).permute(1, 0, 2)
        return stacked_data

    def recover_weight(self, stacked_data):
        # stacked_data: (N_block, P, codebook_dim)
        stacked_flat = stacked_data.permute(1, 0, 2).contiguous()
        # shape: (P, N_block, codebook_dim)
        w_flatten = stacked_flat.reshape(self.parallel_layers, -1)
        # shape: (P, (O/P)*I)

        # 3. undo normalization
        w_flatten = w_flatten * (self.d_std + 1e-6) + self.d_mean
        # 4. reshape back to parallel weight
        weight = w_flatten.reshape(
            self.parallel_layers,
            self.out_features // self.parallel_layers,
            self.in_features
        )
        # shape: (P, O/P, I)

        # 5. merge parallel layers
        weight = weight.reshape(self.out_features, self.in_features)

        # 6. undo transpose if needed
        if self.transpose:
            weight = weight.t()

        return weight

    def _set_mode(self, is_train: bool):
        self.is_train = is_train
        if not is_train:
            self.vae.eval()
            with torch.no_grad():
                _, vq_w = self.vae(self.parallel_weight(self.weight), is_train=False)
            self.register_buffer('vq_weight', vq_w)
            self.decoder = self.vae.decoder
            self.decoder._fuse_q_scale()

    def _normalize_weight(self, weight):
        if self.transpose:
            weight = weight.t()
        weight = weight.reshape(self.parallel_layers, weight.shape[0] // self.parallel_layers, weight.shape[1])
        w_flatten = weight.reshape(self.parallel_layers, -1)
        self.register_buffer('d_mean', w_flatten.mean(dim=1, keepdim=True))
        self.register_buffer('d_std', w_flatten.std(dim=1, keepdim=True))

    def forward(self, x):
        # 执行线性变换
        if self.is_train:
            w_recon, _, loss_dict = self.vae(self.parallel_weight(self.weight), is_train=self.is_train)
        else:
            w_recon = self.decoder(self.vq_weight)

        w_recon = self.recover_weight(w_recon)

        out = F.linear(x, w_recon, self.bias)

        if self.distil_loss_type == 'mse':
            distil_loss = F.mse_loss(F.linear(x, self.weight, self.bias), out)
            loss_dict.update({'distil_loss': distil_loss})
        else:
            distil_loss = 0.
            loss_dict.update({'distil_loss': distil_loss})
        loss_dict["loss"] += distil_loss * self.distil_loss_weight

        if self.is_train:
            return out, loss_dict

        return out


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
                transpose=transpose,
                parallel_layers=args.parallel_layers,
                distil_loss_type=args.distil_loss_type,
                distil_loss_weight=args.distil_loss_weight
            )
            multi_vae = MultiLayerVAE(args).to(bsq_linear.weight.device)
            bsq_linear.set_vae(multi_vae)
            bsq_linear._set_mode(is_train=True)
            set_module_by_name(model, name, bsq_linear)
            params_list.extend(list(multi_vae.parameters()))

    return model, params_list


def bsq_turn2infra(llm):
    if isinstance(llm, LlamaForCausalLM) or isinstance(llm, Qwen3ForCausalLM):
        model = llm.model
    elif isinstance(llm, LlamaModel) or isinstance(llm, Qwen3Model):
        model = llm
    else:
        raise ValueError("Model must be LlamaForCausalLM, Qwen3ForCausalLM, LlamaModel, or Qwen3Model.")
    for module in model.modules():
        if isinstance(module, BSQLinear):
            module._set_mode(is_train=False)
    return
