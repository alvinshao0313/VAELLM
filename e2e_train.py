import datetime
import os
from logging import Logger

import datasets
import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast, Trainer, default_data_collator
import transformers
from llm_models.fsdp_trainer import FSDPTrainer
from llm_models.train_args import process_args, create_optimizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from llm_models.data_utils import CustomJsonDataset
from llm_models.utils import get_local_rank, get_logger, pt_fsdp_state_dict

log: Logger = get_logger("spinquant")


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    hf_args, training_args, vae_args = process_args()
    local_rank = get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        vae_args.model_path, token=hf_args.access_token
    )

    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
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
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    # model = prepare_model(ptq_args, model)
    for param in model.parameters():
        param.requires_grad = False
    model.enable_input_require_grads()
    from llm_models.instead_forward import rebuild_llama_forward, rebuild_qwen3_forward
    if 'qwen3' in vae_args.model_path.lower():
        rebuild_qwen3_forward(model)
    else:
        rebuild_llama_forward(model)
    transpose_modules = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    from bitvae.models.bsq_linear import insert_bsq_linear
    trainable_parameters = insert_bsq_linear(model, transpose_modules, vae_args)

    if local_rank == 0:
        log.info("Model init completed for training {}".format(model))
        log.info("Start to load tokenizer...")
    if 'llama' in vae_args.model_path.lower():
        tokenizer = LlamaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=vae_args.model_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            add_eos_token=False,
            add_bos_token=False,
            token=hf_args.access_token,
        )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False
    calibration_datasets = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1"
    )

    TARGET_FINAL_SAMPLES = None

    train_data = CustomJsonDataset(
        calibration_datasets["train"],
        tokenizer,
        block_size=min(training_args.model_max_length, 2048),
        max_samples=TARGET_FINAL_SAMPLES,
    )

    if local_rank == 0:
        log.info(
            f"Final training dataset contains {len(train_data)} samples of length {min(training_args.model_max_length, 2048)}")

    model.seqlen = training_args.model_max_length
    optimizer = create_optimizer(trainable_parameters, vae_args, vae_args.lr)

    # Add LR Scheduler
    lr_scheduler = None
    if hasattr(vae_args, 'lr_scheduler') and vae_args.lr_scheduler != 'none':
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name=vae_args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=getattr(vae_args, 'warmup_steps', 0),
            num_training_steps=training_args.max_steps,
        )

    MyTrainer = Trainer
    # Use FSDP for 70B rotation training
    if training_args.fsdp != "" and training_args.fsdp != []:
        MyTrainer = FSDPTrainer

    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=None,
        data_collator=default_data_collator,
        optimizers=(optimizer, lr_scheduler),
    )
    torch.distributed.barrier()

    trainer.train()

    from bitvae.models.bsq_linear import bsq_turn2infra
    bsq_turn2infra(trainer.model)

    if training_args.fsdp != "" and training_args.fsdp != []:
        cpu_state = pt_fsdp_state_dict(trainer.model)
    else:
        cpu_state = trainer.model.state_dict()

    # Save the model
    if local_rank == 0:
        output_dir = getattr(vae_args, 'output_dir', './output_model')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(cpu_state, os.path.join(output_dir, 'pytorch_model.bin'))
        trainer.model.config.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        log.info(f"Model saved to {output_dir}")

    llm = trainer.model
    from llm_eval import calculate_ppl
    if local_rank == 0:
        log.info("Start to evaluate PPL before reconstruction...")
        llm.eval().to('cuda:0')
        setattr(vae_args, 'limit', -1)
        with torch.no_grad():
            ppl = calculate_ppl(llm, vae_args)
        llm.cpu()
        print(f"After reconstruction PPL evaluated: {ppl}")

    dist.barrier()


if __name__ == "__main__":
    train()
