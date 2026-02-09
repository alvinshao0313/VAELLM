import argparse
import os
import sys
from functools import partial
from logging import Logger
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers import LlamaTokenizerFast
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from litebsq.bsq_linear import bsq_turn2infra
from train_utils.train_args import process_all_args, create_optimizer
from train_utils.utils import get_logger, pt_fsdp_state_dict
from train_utils.llm_eval import calculate_ppl

log: Logger = get_logger("bsqLLM")


def _parse_layer_indices(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [int(v) for v in items]


def _to_cuda(obj):
    return obj.to("cuda")


def _to_cpu(obj):
    return obj.cpu()


class Catcher(nn.Module):
    def __init__(self, module: nn.Module, orgi_inps: torch.Tensor, cache: dict):
        super().__init__()
        self.module = module
        self.orgi_inps = orgi_inps
        self.cache = cache

    def forward(self, inp, **kwargs):
        self.orgi_inps[self.cache["i"]] = inp.detach().to("cpu")
        self.cache["i"] += 1
        self.cache["attention_mask"] = kwargs["attention_mask"]
        self.cache["position_ids"] = kwargs["position_ids"]
        raise ValueError


def _prepare_layer_inputs(
    model: LlamaForCausalLM,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
):
    position_ids_cuda = position_ids.to(hidden_states.device)
    attention_mask_cuda = attention_mask.to(hidden_states.device) if attention_mask is not None else None
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids_cuda)
    return position_ids_cuda, attention_mask_cuda, position_embeddings


def _eval_ppl(model: LlamaForCausalLM, vae_args, tag: str):
    log.info(f"Start PPL eval after {tag}...")
    model.eval()
    model.to("cuda")
    with torch.no_grad():
        results = calculate_ppl(model, vae_args)
    model.to("cpu")
    torch.cuda.empty_cache()
    log.info(f"PPL after {tag}: {results.get('wiki_ppl'):.2f}")


def _layer_forward_for_checkpoint(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask_cuda: Optional[torch.Tensor],
    position_ids_cuda: torch.Tensor,
    position_embeddings: torch.Tensor,
):
    outputs, loss_dict = layer(
        hidden_states,
        attention_mask=attention_mask_cuda,
        position_ids=position_ids_cuda,
        position_embeddings=position_embeddings,
    )
    return outputs, loss_dict["loss"]


def _layer_forward_with_loss_dict(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask_cuda: Optional[torch.Tensor],
    position_ids_cuda: torch.Tensor,
    position_embeddings: torch.Tensor,
):
    outputs, loss_dict = layer(
        hidden_states,
        attention_mask=attention_mask_cuda,
        position_ids=position_ids_cuda,
        position_embeddings=position_embeddings,
    )
    return outputs, loss_dict


def _compute_layer_loss(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask_cuda: Optional[torch.Tensor],
    position_ids_cuda: torch.Tensor,
    position_embeddings: torch.Tensor,
    use_checkpoint: bool,
):
    if use_checkpoint:
        from torch.utils.checkpoint import checkpoint

        layer_forward = partial(_layer_forward_for_checkpoint, layer)
        return checkpoint(
            layer_forward,
            hidden_states,
            attention_mask_cuda,
            position_ids_cuda,
            position_embeddings,
        ), None

    outputs, loss_dict = _layer_forward_with_loss_dict(
        layer,
        hidden_states,
        attention_mask_cuda,
        position_ids_cuda,
        position_embeddings,
    )
    return (outputs, loss_dict["loss"]), loss_dict


def _format_loss_items(loss_dict: dict) -> str:
    pieces = []
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            scalar = value.detach().float().item()
        else:
            scalar = float(value)
        pieces.append(f"{key}={scalar:.6f}")
    return " ".join(pieces)


def train() -> None:
    lbl_args, hf_args, training_args, vae_args = process_all_args(sys.argv[1:])
    log.info("Args: lbl=%s hf=%s training=%s vae=%s", lbl_args, hf_args, training_args, vae_args)

    config = transformers.AutoConfig.from_pretrained(
        vae_args.model_path, token=hf_args.access_token
    )

    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=vae_args.model_path,
        config=config,
        torch_dtype=dtype,
        token=hf_args.access_token,
        device_map="cpu",
    )
    model.config._attn_implementation = "sdpa"
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    for param in model.parameters():
        param.requires_grad = False

    transpose_modules = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    if lbl_args.disable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        training_args.gradient_checkpointing = False

    log.info("Model init completed for training {}".format(model))

    model.seqlen = training_args.model_max_length

    layer_indices = _parse_layer_indices(lbl_args.layer_indices)
    if layer_indices is None:
        num_layers = model.config.num_hidden_layers
        layer_indices = list(range(num_layers))

    if lbl_args.max_layers is not None:
        layer_indices = layer_indices[: lbl_args.max_layers]

    from train_utils.data_utils import get_wikitext2
    data_loader = get_wikitext2(
        nsamples=lbl_args.nsamples,
        seed=0,
        seqlen=model.seqlen,
        model=vae_args.model_path,
    )

    # GPTQ 方式
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = _to_cuda(model.model.embed_tokens)
    model.model.norm = _to_cuda(model.model.norm)
    layers[0] = _to_cuda(layers[0])

    dtype = next(iter(model.parameters())).dtype
    orgi_inps = torch.zeros(
        (lbl_args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    cache = {'i': 0, 'attention_mask': None}

    layers[0] = Catcher(layers[0], orgi_inps, cache)
    for batch in data_loader:
        try:
            model(batch[0].to('cuda'))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = _to_cpu(layers[0])
    model.model.embed_tokens = _to_cpu(model.model.embed_tokens)
    model.model.norm = _to_cpu(model.model.norm)
    torch.cuda.empty_cache()

    orgi_outs = torch.zeros_like(orgi_inps, device='cpu')
    attention_mask = cache['attention_mask']
    position_ids = cache.get('position_ids', None)
    if position_ids is None:
        position_ids = torch.arange(0, model.seqlen, device='cpu').unsqueeze(0)

    use_amp = training_args.fp16 or training_args.bf16
    use_fp16 = bool(training_args.fp16)
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    logging_steps = getattr(training_args, "logging_steps", 1)
    global_step = 0

    for idx in layer_indices:
        layer = _to_cuda(model.model.layers[idx])

        with torch.no_grad():
            for j in range(lbl_args.nsamples):  # 获取原本第idx层的输出
                hidden_states = _to_cuda(orgi_inps[j].unsqueeze(0))
                position_ids_cuda, attention_mask_cuda, position_embeddings = _prepare_layer_inputs(
                    model, hidden_states, position_ids, attention_mask
                )
                orgi_outs[j] = layer(
                    hidden_states,
                    attention_mask=attention_mask_cuda,
                    position_ids=position_ids_cuda,
                    position_embeddings=position_embeddings,
                )[0].detach().to('cpu')

        from train_utils.instead_forward import rebuild_llama_forward
        rebuild_llama_forward(layer)
        from litebsq.bsq_linear import insert_bsq_linear
        learrnable_params = insert_bsq_linear(layer, transpose_modules, vae_args)

        optimizer = create_optimizer(learrnable_params, vae_args, vae_args.lr)
        if vae_args.lr_scheduler != "none":
            lr_scheduler = transformers.get_scheduler(
                vae_args.lr_scheduler,
                optimizer,
                num_warmup_steps=vae_args.lr_warmup_steps,
                num_training_steps=lbl_args.steps_per_layer
                if lbl_args.steps_per_layer is not None
                else len(data_loader) * lbl_args.num_train_epochs,
            )

        for epoch in range(lbl_args.num_train_epochs):
            for j in range(lbl_args.nsamples):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    hidden_states = _to_cuda(orgi_inps[j].unsqueeze(0))
                    hidden_states.requires_grad_(True)
                    position_ids_cuda, attention_mask_cuda, position_embeddings = _prepare_layer_inputs(
                        model, hidden_states, position_ids, attention_mask
                    )
                    (outputs, loss), loss_dict = _compute_layer_loss(
                        layer,
                        hidden_states,
                        attention_mask_cuda,
                        position_ids_cuda,
                        position_embeddings,
                        lbl_args.layer_checkpointing,
                    )
                    if lbl_args.use_output_mse_loss:
                        target_out = orgi_outs[j].unsqueeze(0).to(device=outputs.device, dtype=outputs.dtype)
                        mse_loss = F.mse_loss(outputs, target_out)
                        loss = loss + lbl_args.output_mse_loss_weight * mse_loss
                        if loss_dict is not None:
                            loss_dict = dict(loss_dict)
                            loss_dict["train/output_mse_loss"] = mse_loss
                            loss_dict["loss"] = loss

                optimizer.zero_grad()
                if use_fp16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if vae_args.lr_scheduler != "none":
                    lr_scheduler.step()
                global_step += 1
                if logging_steps > 0 and global_step % logging_steps == 0:
                    if loss_dict is None:
                        with torch.no_grad():
                            _, log_loss_dict = _layer_forward_with_loss_dict(
                                layer,
                                hidden_states.detach(),
                                attention_mask_cuda,
                                position_ids_cuda,
                                position_embeddings,
                            )
                        loss_dict = dict(log_loss_dict)
                        if lbl_args.use_output_mse_loss:
                            loss_dict["train/output_mse_loss"] = mse_loss.detach()
                            loss_dict["loss"] = loss.detach()

                    loss_items = _format_loss_items(loss_dict)
                    log.info(
                        "layer=%d epoch=%d step=%d loss=%.6f %s",
                        idx,
                        epoch,
                        global_step,
                        loss.detach().float().item(),
                        loss_items,
                    )
        orgi_inps = orgi_outs
        layer = _to_cpu(layer)
        model.model.layers[idx] = layer
        bsq_turn2infra(layer)
        _eval_ppl(model, vae_args, f"layer {idx}")

    if training_args.fsdp != "" and training_args.fsdp != []:
        cpu_state = pt_fsdp_state_dict(model)
    else:
        cpu_state = model.state_dict()

    output_dir = getattr(vae_args, "output_dir", "./output_model")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(cpu_state, os.path.join(output_dir, "pytorch_model.bin"))
    model.config.save_pretrained(output_dir)
    tokenizer = LlamaTokenizerFast.from_pretrained(
        vae_args.model_path,
        use_fast=False,
        torch_dtype=dtype,
        token=hf_args.access_token,
    )
    tokenizer.save_pretrained(output_dir)
    log.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    train()
