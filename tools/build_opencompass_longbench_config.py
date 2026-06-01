import argparse
import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _model_dict_to_code(model: dict) -> str:
    lines = ["    dict("]
    for key, value in model.items():
        if key == "type":
            lines.append(f"        type={value},")
        else:
            lines.append(f"        {key}={value!r},")
    lines.append("    ),")
    return "\n".join(lines)


def build_config() -> str:
    dense_hf_path = _env("DENSE_HF_PATH", "Qwen/Qwen3-8B")
    checkpoint_dir = _env("CHECKPOINT_DIR", ".result/final_model")
    adapter_dir = _env("ADAPTER_DIR", ".result/dense_e2e_fintuning/final_model_20260427_020212/final_adapter")
    base_model_path = _env("BASE_MODEL_PATH", "Qwen/Qwen3-8B")
    access_token = _env("ACCESS_TOKEN", "")
    max_seq_len = int(_env("MAX_SEQ_LEN", "8192"))
    max_out_len = int(_env("MAX_OUT_LEN", "1024"))
    batch_size = int(_env("BATCH_SIZE", "1"))
    hf_num_gpus = int(_env("HF_NUM_GPUS", "1"))
    eval_device = _env("EVAL_DEVICE", "cuda")
    prewarm_group_size = int(_env("PREWARM_GROUP_SIZE", "8"))

    tokenizer_kwargs = dict(trust_remote_code=True, use_fast=False)
    generation_kwargs = dict(do_sample=False)
    models = []

    if dense_hf_path:
        models.append(
            dict(
                type="HuggingFaceBaseModel",
                abbr=_env("DENSE_ABBR", "qwen3-8b-dense"),
                path=dense_hf_path,
                tokenizer_path=dense_hf_path,
                tokenizer_kwargs=tokenizer_kwargs,
                model_kwargs=dict(device_map="auto", trust_remote_code=True, torch_dtype="auto"),
                generation_kwargs=generation_kwargs,
                max_seq_len=max_seq_len,
                max_out_len=max_out_len,
                batch_size=batch_size,
                run_cfg=dict(num_gpus=hf_num_gpus),
            )
        )

    if checkpoint_dir:
        models.append(
            dict(
                type="VAELLMOpenCompassModel",
                abbr=_env("CHECKPOINT_ABBR", "vaellm-compressed"),
                path=base_model_path,
                checkpoint_dir=checkpoint_dir,
                base_model_path=base_model_path,
                access_token=access_token or None,
                tokenizer_path=base_model_path,
                tokenizer_kwargs=tokenizer_kwargs,
                model_kwargs=dict(),
                generation_kwargs=generation_kwargs,
                max_seq_len=max_seq_len,
                max_out_len=max_out_len,
                batch_size=batch_size,
                eval_device=eval_device,
                prewarm_group_size=prewarm_group_size,
                run_cfg=dict(num_gpus=hf_num_gpus),
            )
        )

    if checkpoint_dir and adapter_dir:
        models.append(
            dict(
                type="VAELLMOpenCompassModel",
                abbr=_env("ADAPTER_ABBR", "vaellm-compressed-adapter"),
                path=base_model_path,
                checkpoint_dir=checkpoint_dir,
                adapter_dir=adapter_dir,
                base_model_path=base_model_path,
                access_token=access_token or None,
                tokenizer_path=base_model_path,
                tokenizer_kwargs=tokenizer_kwargs,
                model_kwargs=dict(),
                generation_kwargs=generation_kwargs,
                max_seq_len=max_seq_len,
                max_out_len=max_out_len,
                batch_size=batch_size,
                eval_device=eval_device,
                prewarm_group_size=prewarm_group_size,
                run_cfg=dict(num_gpus=hf_num_gpus),
            )
        )

    if not models:
        raise ValueError("No model configured. Set DENSE_HF_PATH or CHECKPOINT_DIR.")

    rendered_models = "\n".join(_model_dict_to_code(model) for model in models)
    return (
        "from opencompass.models import HuggingFaceBaseModel\n"
        "from scripts.opencompass_vaellm_model import VAELLMOpenCompassModel\n\n"
        "models = [\n"
        f"{rendered_models}\n"
        "]\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_config(), encoding="utf-8")


if __name__ == "__main__":
    main()
