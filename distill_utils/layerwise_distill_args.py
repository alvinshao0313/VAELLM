import argparse
from typing import List, Optional


def parse_layer_indices(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layer-wise distillation for cat_train quantized checkpoints.")
    parser.add_argument("--student_checkpoint_dir", type=str, required=True, help="Quantized checkpoint dir (or run dir).")
    parser.add_argument("--teacher_model_path", type=str, default=None, help="Teacher model path. Defaults to checkpoint meta base_model_path.")
    parser.add_argument("--access_token", type=str, default=None)
    parser.add_argument("--map_location", type=str, default="cpu")
    parser.add_argument("--strict", dest="strict", action="store_true")
    parser.add_argument("--no_strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)

    parser.add_argument("--student_device", type=str, default="cuda")
    parser.add_argument("--teacher_device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no_shuffle", dest="shuffle", action="store_false")

    parser.add_argument("--layer_indices", type=str, default=None, help="Comma-separated layer ids, e.g. 0,1,2")
    parser.add_argument("--max_layers", type=int, default=None, help="When layer_indices is empty, only train first N layers.")
    parser.add_argument("--epochs_per_layer", type=int, default=1)
    parser.add_argument("--steps_per_layer", type=int, default=0, help=">0 overrides epochs_per_layer.")

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd", "rmsprop"])
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--lambda_blk", type=float, default=0.70)
    parser.add_argument("--lambda_res", type=float, default=0.25)
    parser.add_argument("--lambda_anchor", type=float, default=0.05)
    parser.add_argument("--use_norm_loss", action="store_true", default=False)
    parser.add_argument("--lambda_norm", type=float, default=0.10)

    parser.add_argument("--train_bias", action="store_true", default=False)
    parser.add_argument("--train_layernorm_weight", action="store_true", default=False)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./output_layerwise_distill")
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--save_tokenizer", action="store_true", default=False)
    parser.add_argument(
        "--unload_vae_original_weights_on_save",
        action="store_true",
        default=False,
        help="Unload VAELinear original weights before save to shrink checkpoint.",
    )
    return parser
