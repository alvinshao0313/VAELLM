import argparse

from bitvae.models import d_vae

def add_model_specific_args(args, parser):
    if args.tokenizer == "flux":
        parser = d_vae.add_model_specific_args(parser) # flux config
        d_vae_model = d_vae
    else:
        raise NotImplementedError
    return args, parser, d_vae_model

class MainArgs:
    @staticmethod
    def add_main_args(parser):
        # training
        parser.add_argument('--max_steps', type=int, default=1e6)
        parser.add_argument('--log_every', type=int, default=1)
        parser.add_argument('--visu_every', type=int, default=1000)
        parser.add_argument('--ckpt_every', type=int, default=1000)
        parser.add_argument('--default_root_dir', type=str, required=True)
        parser.add_argument('--multiscale_training', action="store_true")

        # optimization
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.95)
        parser.add_argument('--warmup_steps', type=int, default=0)
        parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "AdamW"])
        parser.add_argument('--disc_optim_type', type=str, default=None, choices=[None, "rmsprop"])
        parser.add_argument('--lr_min', type=float, default=0.)
        parser.add_argument('--warmup_lr_init', type=float, default=0.)
        parser.add_argument('--max_grad_norm', type=float, default=1.0)
        parser.add_argument('--max_grad_norm_disc', type=float, default=1.0)
        parser.add_argument('--disable_sch', action="store_true")

        # basic d_vae config
        parser.add_argument('--patch_size', type=int, default=8)
        parser.add_argument('--codebook_dim', type=int, default=16)
        parser.add_argument('--quantizer_type', type=str, default='MultiScaleLFQ')

        parser.add_argument('--new_quant', action="store_true") # use new quantization (fix the potential bugs of the old quantizer)
        parser.add_argument('--use_decay_factor', action="store_true")
        parser.add_argument('--use_stochastic_depth', action="store_true")
        parser.add_argument("--drop_rate", type=float, default=0.0)
        parser.add_argument('--schedule_mode', type=str, default="original", choices=["original", "dynamic", "dense", "same1", "same2", "same3", "half", "dense_f8"])
        parser.add_argument('--lr_drop', nargs='*', type=int, default=None, help="A list of numeric values. Example: --values 270 300")
        parser.add_argument('--lr_drop_rate', type=float, default=0.1)
        parser.add_argument('--keep_first_quant', action="store_true")
        parser.add_argument('--keep_last_quant', action="store_true")
        parser.add_argument('--remove_residual_detach', action="store_true")
        parser.add_argument('--use_out_phi', action="store_true")
        parser.add_argument('--use_out_phi_res', action="store_true")
        parser.add_argument('--lecam_weight', type=float, default=0.05)
        parser.add_argument('--perceptual_model', type=str, default="vgg16", choices=["vgg16"])
        parser.add_argument('--base_ch_disc', type=int, default=64)
        parser.add_argument('--random_flip', action="store_true")
        parser.add_argument('--flip_prob', type=float, default=0.5)
        parser.add_argument('--flip_mode', type=str, default="stochastic", choices=["stochastic"])
        parser.add_argument('--max_flip_lvl', type=int, default=1)
        parser.add_argument('--not_load_optimizer', action="store_true")
        parser.add_argument('--use_lecam_reg_zero', action="store_true")
        parser.add_argument('--rm_downsample', action="store_true")
        parser.add_argument('--random_flip_1lvl', action="store_true")
        parser.add_argument('--flip_lvl_idx', type=int, default=0)
        parser.add_argument('--drop_when_test', action="store_true")
        parser.add_argument('--drop_lvl_idx', type=int, default=None)
        parser.add_argument('--drop_lvl_num', type=int, default=0)
        parser.add_argument('--compute_all_commitment', action="store_true")
        parser.add_argument('--disable_codebook_usage', action="store_true")
        parser.add_argument('--random_short_schedule', action="store_true")
        parser.add_argument('--short_schedule_prob', type=float, default=0.5)
        parser.add_argument('--disable_flip_prob', type=float, default=0.0)
        parser.add_argument('--zeta', type=float, default=1.0) # entropy penalty weight
        parser.add_argument('--disable_codebook_usage_bit', action="store_true")
        parser.add_argument('--gamma', type=float, default=1.0) # loss weight of H(E[p(c|u)])
        parser.add_argument('--uniform_short_schedule', action="store_true")

        # discriminator config
        parser.add_argument('--dis_warmup_steps', type=int, default=0)
        parser.add_argument('--dis_lr_multiplier', type=float, default=1.)
        parser.add_argument('--dis_minlr_multiplier', action="store_true")
        parser.add_argument('--disc_layers', type=int, default=3)
        parser.add_argument('--discriminator_iter_start', type=int, default=0)
        parser.add_argument('--disc_pretrain_iter', type=int, default=0)
        parser.add_argument('--disc_optim_steps', type=int, default=1)
        parser.add_argument('--disc_warmup', type=int, default=0)
        parser.add_argument('--disc_pool', type=str, default="no", choices=["no", "yes"])
        parser.add_argument('--disc_pool_size', type=int, default=1000)

        # loss
        parser.add_argument("--recon_loss_type", type=str, default='l1', choices=['l1', 'l2'])
        parser.add_argument('--image_gan_weight', type=float, default=1.0)
        parser.add_argument('--image_disc_weight', type=float, default=0.)
        parser.add_argument('--l1_weight', type=float, default=4.0)
        parser.add_argument('--gan_feat_weight', type=float, default=0.0)
        parser.add_argument('--perceptual_weight', type=float, default=0.0)
        parser.add_argument('--kl_weight', type=float, default=0.)
        parser.add_argument('--lfq_weight', type=float, default=0.)
        parser.add_argument('--entropy_loss_weight', type=float, default=0.1)
        parser.add_argument('--commitment_loss_weight', type=float, default=0.25)
        parser.add_argument('--diversity_gamma', type=float, default=1)
        parser.add_argument('--norm_type', type=str, default='group', choices=['batch', 'group', "no"])
        parser.add_argument('--disc_loss_type', type=str, default='hinge', choices=['hinge', 'vanilla'])

        # acceleration
        parser.add_argument('--use_checkpoint', action="store_true")
        parser.add_argument('--precision', type=str, default="fp32", choices=['fp32', 'bf16']) # disable fp16
        parser.add_argument('--encoder_dtype', type=str, default="fp32", choices=['fp32', 'bf16']) # disable fp16
        parser.add_argument('--upcast_tf32', action="store_true")

        # initialization
        parser.add_argument('--tokenizer', type=str, default='flux', choices=["flux"])
        parser.add_argument('--pretrained', type=str, default=None)
        parser.add_argument('--pretrained_mode', type=str, default="full", choices=['full'])

        # misc
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--seed', type=int, default=1234)
        parser.add_argument('--bucket_cap_mb', type=int, default=40) # DDP
        parser.add_argument('--manual_gc_interval', type=int, default=1000) # DDP

        return parser