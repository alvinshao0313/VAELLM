import argparse
from rotation.common import *


class LLMArgs:
    @staticmethod
    def add_llm_args(parser):
        # Training
        parser.add_argument("--default_root_dir", type=str, default="./results/llm_vae_test")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--max_steps", type=int, default=10000)
        parser.add_argument("--log_every", type=int, default=100)
        parser.add_argument("--ckpt_every", type=int, default=1000)
        parser.add_argument("--eval_every", type=int, default=100,
                            help="Evaluate full matrix MSE every these many steps")
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
        parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
        parser.add_argument("--wandb_project", type=str, default="llm_vae_experiment", help="Wandb project name")
        parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")

        # Optimizer
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.95)
        parser.add_argument("--weight_decay", type=float, default=1e-2)
        parser.add_argument("--optimizer", type=str, default='adamw', choices=['adam', 'adamw', 'sgd', 'rmsprop'])
        parser.add_argument("--lr_scheduler", type=str, default='none', choices=['none', 'linear', 'cosine'],
                            help="Learning rate scheduler")
        parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps for scheduler")
        # 权重初始化
        # Rotation and Hadamard Transform
        parser.add_argument('--rotate', action='store_true', default=False)
        parser.add_argument('--rotate_mode', type=str, default='hadamard',
                            choices=['hadamard', 'group_hadamard', 'identity'])
        parser.add_argument('--online_partial_had', action='store_true', default=False)
        parser.add_argument('--online_down_had', action='store_true', default=True)
        parser.add_argument('--r1_path', type=str, default=None,
                            help='''Path to the R1 rotation matrix. Deafult is None.
                            If not specified, R1 will generated as "rotate_mode".''')

        # Training Specific
        parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf",
                            help="Path or HuggingFace ID of the LLM")

        # Data Preprocessing
        parser.add_argument("--normalize_data", action="store_true", help="Normalize data (z-score) before training")
        parser.add_argument("--clip_threshold", type=float, default=-1.0,
                            help="Clip data at standard deviation threshold (e.g. 3.0). < 0 to disable.")

        parser.add_argument("--recon_loss_type", type=str, default='mse',
                            choices=['mse', 'l1', 'huber',
                                     'relative_l1', 'top_k_mse'],
                            help="Type of reconstruction loss to use")
        parser.add_argument("--distil_loss_type", type=str, default='mse',
                            choices=['mse', 'none'],
                            help="Type of distillation loss to use between original and reconstructed weights")
        parser.add_argument("--distil_loss_weight", type=float, default=1.0,
                            help="Weight of the distillation loss")
        parser.add_argument("--l1_weight", type=float, default=1.0)
        parser.add_argument("--lfq_weight", type=float, default=1.0)
        parser.add_argument("--commitment_loss_weight", type=float, default=0.25)
        parser.add_argument("--entropy_loss_weight", type=float, default=0.1)
        parser.add_argument("--diversity_gamma", type=float, default=1.0)
        # parser.add_argument("--compute_all_commitment", action="store_true")
        parser.add_argument("--encoder_dtype", type=str, default="fp32")
        parser.add_argument("--use_checkpoint", action="store_true")
        parser.add_argument("--new_quant", action="store_true")

        # GAN / Discriminator
        parser.add_argument("--image_gan_weight", type=float, default=0.0, help="GAN loss weight, set 0 to disable")
        parser.add_argument("--discriminator_iter_start", type=int, default=1000)
        parser.add_argument("--disc_lr", type=float, default=1e-4)
        parser.add_argument("--disc_hidden_dim", type=int, default=256)
        parser.add_argument("--disc_loss_type", type=str, default='hinge', choices=['hinge', 'vanilla'])

        return parser
