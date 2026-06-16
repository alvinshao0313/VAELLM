from vae_e2e_fintuning.runtime import run as _run_vae_e2e


def run(args, hf_args, training_args):
    args.e2e_stage = "compressed_e2e_fintuning"
    args.e2e_args_key = "compressed_e2e_args"
    if not hasattr(args, "finetune_mode"):
        args.finetune_mode = "decoder"
    args.internal_vae_train_mode = str(args.vae_train_mode)
    return _run_vae_e2e(args, hf_args, training_args)
