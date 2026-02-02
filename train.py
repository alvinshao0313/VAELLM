import os
import argparse
import math
import glob
import time
import logging
from copy import deepcopy
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from timm.scheduler.cosine_lr import CosineLRScheduler

from bitvae.utils.distributed import init_distributed_mode, reduce_losses, average_losses
from bitvae.utils.logger import create_logger

from bitvae.models import ImageDiscriminator
from bitvae.data import ImageData
from bitvae.modules.loss import get_disc_loss, adopt_weight
from bitvae.utils.misc import get_last_ckpt
from bitvae.utils.init_models import resume_from_ckpt
from bitvae.utils.arguments import MainArgs, add_model_specific_args

logger = logging.getLogger(__name__)

def lecam_reg_zero(real_pred, fake_pred, thres=0.1):
    # avoid logits get too high
    assert real_pred.ndim == 0
    reg = torch.mean(F.relu(torch.abs(real_pred) - thres).pow(2)) + \
    torch.mean(F.relu(torch.abs(fake_pred) - thres).pow(2))
    return reg


def main():
    parser = argparse.ArgumentParser()
    parser = MainArgs.add_main_args(parser)
    parser = ImageData.add_data_specific_args(parser)
    args, unknown = parser.parse_known_args()
    args, parser, d_vae_model = add_model_specific_args(args, parser)
    args = parser.parse_args()

    args.resolution = (args.resolution[0], args.resolution[0]) if len(args.resolution) == 1 else args.resolution # init resolution

    print(f"{args.default_root_dir=}")

    # Setup DDP:
    init_distributed_mode(args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.default_root_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders
        checkpoint_dir = f"{args.default_root_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(args.default_root_dir)
        logger.info(f"Experiment directory created at {args.default_root_dir}")

        import wandb
        wandb_project = "VQVAE"
        wandb.init(
            project=wandb_project,
            name=os.path.basename(os.path.normpath(args.default_root_dir)),
            dir=args.default_root_dir,
            config=args,
            mode="offline" if args.debug else "online"
        )
    else:
        logger = create_logger(None)
    
    # init dataloader
    data = ImageData(args)
    dataloaders = data.train_dataloader()
    dataloader_iters = [iter(loader) for loader in dataloaders]
    data_epochs = [0 for _ in dataloaders]
    
    # init model    
    d_vae = d_vae_model(args).to(device)
    d_vae.logger = logger
    image_disc = ImageDiscriminator(args).to(device)

    # init optimizers and schedulers
    if args.optim_type == "Adam":
        optim = torch.optim.Adam
    elif args.optim_type == "AdamW":
        optim = torch.optim.AdamW
    if args.disc_optim_type is None:
        disc_optim = optim
    elif args.disc_optim_type == "rmsprop":
        disc_optim = torch.optim.RMSprop
    opt_vae = optim(d_vae.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    if disc_optim == torch.optim.RMSprop:
        opt_image_disc = disc_optim(image_disc.parameters(), lr=args.lr * args.dis_lr_multiplier)
    else:
        opt_image_disc = disc_optim(image_disc.parameters(), lr=args.lr * args.dis_lr_multiplier, betas=(args.beta1, args.beta2))

    lr_min = args.lr_min
    train_iters = args.max_steps
    warmup_steps = args.warmup_steps
    warmup_lr_init = args.warmup_lr_init

    if args.disable_sch:
        # scheduler_list = [None, None]
        sch_vae, sch_image_disc = None, None

    model_optims = {
        "vae" : d_vae,
        "image_disc" : image_disc,
        "opt_vae" : opt_vae,
        "opt_image_disc" : opt_image_disc,
        "sch_vae" : sch_vae,
        "sch_image_disc" : sch_image_disc,
    }

    # resume from default_root_dir
    ckpt_path = None
    assert not args.default_root_dir is None # required argument
    ckpt_path = get_last_ckpt(args.default_root_dir)
    init_step = 0
    load_optimizer = not args.not_load_optimizer
    if ckpt_path:
        logger.info(f"Resuming from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model_optims, init_step = resume_from_ckpt(state_dict, model_optims, load_optimizer=True)
    # load pretrained weights
    elif args.pretrained is not None:
        state_dict = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        if args.pretrained_mode == "full":
            model_optims, _ = resume_from_ckpt(state_dict, model_optims, load_optimizer=load_optimizer)
        logger.info(f"Successfully loaded ckpt {args.pretrained}, pretrained_mode {args.pretrained_mode}")

    d_vae = DDP(d_vae.to(device), device_ids=[args.gpu], bucket_cap_mb=args.bucket_cap_mb)
    image_disc = DDP(image_disc.to(device), device_ids=[args.gpu], bucket_cap_mb=args.bucket_cap_mb)
    disc_loss = get_disc_loss(args.disc_loss_type) # hinge loss by default

    if args.multiscale_training:
        scale_idx_list = np.load('bitvae/utils/random_numbers.npy') # load pre-computed scale_idx in each iteration

    start_time = time.time()
    for global_step in range(init_step, args.max_steps):
        loss_dicts = []
        
        if global_step == args.discriminator_iter_start - args.disc_pretrain_iter:
            logging.info(f"discriminator begins pretraining ")
        if global_step == args.discriminator_iter_start:
            log_str = "add GAN loss into training"
            if args.disc_pretrain_iter > 0:
                log_str += ", discriminator ends pretraining"
            logging.info(log_str)
        
        for idx in range(len(dataloader_iters)):
            try:
                _batch = next(dataloader_iters[idx])
            except StopIteration:
                data_epochs[idx] += 1
                logger.info(f"Reset the {idx}th dataloader as epoch {data_epochs[idx]}")
                dataloaders[idx].sampler.set_epoch(data_epochs[idx])
                dataloader_iters[idx] = iter(dataloaders[idx]) # update dataloader iter
                _batch = next(dataloader_iters[idx])
            except Exception as e:
                raise e 
            x = _batch["image"]
            _type = _batch["type"][0]

            if args.multiscale_training:
                # data processing for multi-scale training
                scale_idx = scale_idx_list[global_step]
                if scale_idx == 0:
                    # 256x256 batch=8
                    x = F.interpolate(x, size=(256, 256), mode='area')
                elif scale_idx == 1:
                    # 512x512 batch=4
                    rdn_idx = torch.randperm(len(x))[:4] # without replacement
                    x = x[rdn_idx]
                    x = F.interpolate(x, size=(512, 512), mode='area')
                elif scale_idx == 2:
                    # 1024x1024 batch=2
                    rdn_idx = torch.randperm(len(x))[:2] # without replacement
                    x = x[rdn_idx]
                else:
                    raise ValueError(f"scale_idx {scale_idx} is not supported")

            if _type == "image":
                x_recon, flat_frames, flat_frames_recon, vae_loss_dict = d_vae(x, global_step, image_disc=image_disc)
            g_loss = sum(vae_loss_dict.values())
            opt_vae.zero_grad()
            g_loss.backward()

            if not ((global_step+1) % args.ckpt_every) == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(d_vae.parameters(), args.max_grad_norm)
                if not sch_vae is None:
                    sch_vae.step(global_step)
                elif args.lr_drop and global_step in args.lr_drop:
                    logger.info(f"multiply lr of VQ-VAE by {args.lr_drop_rate} at iteration {global_step}")
                    for opt_vae_param_group in opt_vae.param_groups:
                        opt_vae_param_group["lr"] = opt_vae_param_group["lr"] * args.lr_drop_rate
                opt_vae.step()
            opt_vae.zero_grad() # free memory

            disc_loss_dict = {}
            # disc_factor = 0 before (args.discriminator_iter_start - args.disc_pretrain_iter)
            disc_factor = adopt_weight(global_step, threshold=args.discriminator_iter_start - args.disc_pretrain_iter)
            discloss = d_image_loss = torch.tensor(0.).to(x.device)
            ### enable pool warmup
            for disc_step in range(args.disc_optim_steps): # train discriminator
                require_optim = False
                if _type == "image" and args.image_disc_weight > 0: # train image discriminator
                    require_optim = True
                    logits_image_real = image_disc(x, pool_name="real")
                    logits_image_fake = image_disc(x_recon.detach(), pool_name="fake")
                    d_image_loss = disc_loss(logits_image_real, logits_image_fake)
                    disc_loss_dict["train/logits_image_real"] = logits_image_real.mean().detach()
                    disc_loss_dict["train/logits_image_fake"] = logits_image_fake.mean().detach()
                    disc_loss_dict["train/d_image_loss"] = d_image_loss.mean().detach()
                    discloss = d_image_loss * args.image_disc_weight
                    opt_discs, sch_discs = [opt_image_disc], [sch_image_disc]
                    if global_step >= args.discriminator_iter_start and args.use_lecam_reg_zero:
                        lecam_zero_loss = lecam_reg_zero(logits_image_real.mean(), logits_image_fake.mean())
                        disc_loss_dict["train/lecam_zero_loss"] = lecam_zero_loss.mean().detach()
                        discloss += lecam_zero_loss * args.lecam_weight
                discloss = disc_factor * discloss

                if require_optim:
                    for opt_disc in opt_discs:
                        opt_disc.zero_grad()
                    discloss.backward()

                    if not ((global_step+1) % args.ckpt_every) == 0:
                        if args.max_grad_norm_disc > 0: # by default, 1.0
                            torch.nn.utils.clip_grad_norm_(image_disc.parameters(), args.max_grad_norm_disc)
                        for sch_disc in sch_discs:
                            if not sch_disc is None:
                                sch_disc.step(global_step)
                            elif args.lr_drop and global_step in args.lr_drop:
                                for opt_disc in opt_discs:
                                    logger.info(f"multiply lr of discriminator by {args.lr_drop_rate} at iteration {global_step}")
                                    for opt_disc_param_group in opt_disc.param_groups:
                                        opt_disc_param_group["lr"] = opt_disc_param_group["lr"] * args.lr_drop_rate
                        for opt_disc in opt_discs:
                            opt_disc.step()
                    for opt_disc in opt_discs:
                        opt_disc.zero_grad() # free memory

            loss_dict = {**vae_loss_dict, **disc_loss_dict}
            if (global_step+1) % args.log_every == 0:
                reduced_loss_dict = reduce_losses(loss_dict)
            else:
                reduced_loss_dict = {}
            loss_dicts.append(reduced_loss_dict)

        if (global_step+1) % args.log_every == 0:
            avg_loss_dict = average_losses(loss_dicts)
            torch.cuda.synchronize()
            end_time = time.time()
            iter_speed = (end_time - start_time) / args.log_every
            if rank == 0:
                for key, value in avg_loss_dict.items():
                    wandb.log({key: value}, step=global_step)
                # writing logs
                logger.info(f'global_step={global_step}, precepetual_loss={avg_loss_dict.get("train/perceptual_loss",0):.4f}, recon_loss={avg_loss_dict.get("train/recon_loss",0):.4f}, commitment_loss={avg_loss_dict.get("train/commitment_loss",0):.4f}, logit_r={avg_loss_dict.get("train/logits_image_real",0):.4f}, logit_f={avg_loss_dict.get("train/logits_image_fake",0):.4f}, L_disc={avg_loss_dict.get("train/d_image_loss",0):.4f}, iter_speed={iter_speed:.2f}s')
            start_time = time.time()
        
        if (global_step+1) % args.ckpt_every == 0 and global_step != init_step:
            if rank == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'model_step_{global_step}.ckpt')
                save_dict = {}
                for k in model_optims:
                    save_dict[k] = None if model_optims[k] is None \
                        else model_optims[k].module.state_dict() if hasattr(model_optims[k], "module") \
                        else model_optims[k].state_dict()
                torch.save({
                    'step': global_step,
                    **save_dict,
                }, checkpoint_path)
                logger.info(f'Checkpoint saved at step {global_step}')

            
if __name__ == '__main__':
    main()
