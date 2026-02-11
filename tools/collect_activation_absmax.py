import argparse
import os
import re
import sys
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rotation.model_utils import get_model
from train_utils.data_utils import get_wikitext2
from train_utils.utils import get_logger, set_seed


log = get_logger("collect_activation_absmax")


def _split_csv(value: str) -> List[str]:
    raw = str(value).strip()
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _safe_token(text: str) -> str:
    token = str(text).strip().replace("\\", "/").rstrip("/")
    token = token.split("/")[-1] if token else "model"
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token)
    return token or "model"


def _collect_target_linears(
    model: nn.Module,
    projection_suffixes: Sequence[str],
) -> List[Tuple[str, nn.Linear]]:
    suffix_set = set(projection_suffixes)
    out: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        category = name.split(".")[-1]
        if category not in suffix_set:
            continue
        out.append((name, module))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect per-channel abs-max of input activations for decoder projection linears "
            "(q/k/v/o/gate/up/down) on WikiText-2 calibration set."
        )
    )
    parser.add_argument("--model_path", type=str, required=True, help="HF model id or local path.")
    parser.add_argument("--access_token", type=str, default=None, help="HuggingFace access token.")
    parser.add_argument("--device", type=str, default="cuda", help="Run device, e.g. cuda or cpu.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=512, help="Number of calibration samples.")
    parser.add_argument("--seqlen", type=int, default=512, help="Calibration sequence length.")
    parser.add_argument(
        "--projection_suffixes",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Target decoder projection suffixes.",
    )
    parser.add_argument("--output_dir", type=str, default="./prepares", help="Directory to store output file.")
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Optional output filename. Default: <model_name>_activation_weight.pt",
    )
    parser.add_argument("--log_every", type=int, default=50)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.access_token is not None and not str(args.access_token).strip():
        args.access_token = None
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available, fallback to cpu.")
        args.device = "cpu"

    set_seed(int(args.seed))
    suffixes = _split_csv(args.projection_suffixes)
    if not suffixes:
        raise ValueError("--projection_suffixes is empty.")

    log.info("Loading model: %s", args.model_path)
    model = get_model(args.model_path, args.access_token)
    model.eval()
    model.to(args.device)

    log.info("Loading tokenizer and WikiText-2 calibration set (nsamples=%d, seqlen=%d)", args.nsamples, args.seqlen)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
        token=args.access_token,
    )
    calib_data = get_wikitext2(
        nsamples=int(args.nsamples),
        seed=int(args.seed),
        seqlen=int(args.seqlen),
        model=args.model_path,
        tokenizer=tokenizer,
        eval_mode=False,
    )

    target_linears = _collect_target_linears(model, suffixes)
    if not target_linears:
        raise RuntimeError("No target decoder projection linears found in model.")
    log.info("Found %d target linears.", len(target_linears))

    absmax_by_linear: Dict[str, torch.Tensor] = {
        name: torch.zeros(module.in_features, dtype=torch.float32)
        for name, module in target_linears
    }

    handles = []

    def _hook_factory(name: str, in_features: int):
        def _hook(_module: nn.Module, inputs, _output):
            x = inputs[0]
            if x.numel() == 0:
                return
            last_dim = int(x.shape[-1])
            if last_dim != in_features:
                log.warning("Skip %s once due to feature mismatch: got %d expected %d", name, last_dim, in_features)
                return
            cur = x.detach().reshape(-1, last_dim).abs().amax(dim=0).to(dtype=torch.float32, device="cpu")
            absmax_by_linear[name] = torch.maximum(absmax_by_linear[name], cur)

        return _hook

    for name, module in target_linears:
        handles.append(module.register_forward_hook(_hook_factory(name, int(module.in_features))))

    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False

    with torch.no_grad():
        total = len(calib_data)
        for i, (inp, _tar) in enumerate(calib_data, start=1):
            input_ids = inp.to(args.device)
            _ = model(input_ids=input_ids)
            if i % max(1, int(args.log_every)) == 0 or i == total:
                log.info("Progress: %d/%d", i, total)

    for h in handles:
        h.remove()
    if use_cache is not None:
        model.config.use_cache = use_cache

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = _safe_token(args.model_path)
    output_filename = args.output_filename or f"{model_name}_activation_weight.pt"
    output_path = os.path.join(args.output_dir, output_filename)

    # dict: {linear_name: per-channel abs-max vector}
    torch.save(absmax_by_linear, output_path)
    log.info("Saved activation abs-max dictionary to: %s", output_path)
    log.info("Dictionary size: %d", len(absmax_by_linear))


if __name__ == "__main__":
    main()
