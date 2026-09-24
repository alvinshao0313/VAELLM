"""Strict completed-block restart state; never a mid-block optimizer checkpoint.

A boundary stores only completed blocks' permitted native tensors. Replaying the
FP teacher prefix is deterministic; the caller restores RNG at the saved boundary,
not before loading models or replaying earlier blocks.
"""
import hashlib
import json
from importlib.metadata import version
import os
from pathlib import Path
import random

import numpy as np
import torch


FORMAT = "vaellm_stage_b_completed_blocks_v1"
TRAINING_FILES = (
    "recover.py", "recovery_resume.py", "recovery_data.py", "recovery_runtime.py",
    "block_train.py", "all_bits.py", "proxy_coordinates.py", "liftquant_optimizer.py",
    "verify_recovery.py",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(path):
    """Hash actual native weights and model/tokenizer inputs, not prior hash files."""
    root = Path(path).resolve()
    required = ("checkpoint_meta.json", "config.json")
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    names = set(required)
    metadata = json.loads((root / "checkpoint_meta.json").read_text())
    state_file = metadata.get("state_dict_file", "pytorch_model.bin")
    if not isinstance(state_file, str) or Path(state_file).is_absolute() or ".." in Path(state_file).parts:
        raise ValueError("Native state_dict_file must be a relative checkpoint payload path.")
    if not (root / state_file).is_file():
        raise FileNotFoundError(root / state_file)
    names.add(state_file)
    names.update(p.name for pattern in ("*.bin", "*.safetensors", "*.index.json")
                 for p in root.glob(pattern) if p.is_file())
    names.update(name for name in (
        "generation_config.json", "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
        "tokenizer.model", "spiece.model",
    ) if (root / name).is_file())
    return {name: dict(bytes=(root / name).stat().st_size, sha256=file_sha256(root / name))
            for name in sorted(names)}


def teacher_fingerprint(path):
    """Hash the local HF config and actual weight payloads, including indexed shards."""
    root = Path(path).resolve()
    names = {"config.json"}
    weights = {str(p.relative_to(root)) for pattern in ("*.bin", "*.safetensors")
               for p in root.glob(pattern) if p.is_file()}
    for index in root.glob("*.index.json"):
        mapping = json.loads(index.read_text()).get("weight_map")
        if not isinstance(mapping, dict):
            raise ValueError(f"HF weight index has no weight_map: {index}")
        names.add(index.name)
        for name in mapping.values():
            if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("HF weight_map must reference relative payload paths.")
            weights.add(name)
    if not weights:
        raise ValueError(f"No local HF teacher weights found: {root}")
    names.update(weights)
    if (root / "generation_config.json").is_file():
        names.add("generation_config.json")
    for name in names:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    return {name: dict(bytes=(root / name).stat().st_size, sha256=file_sha256(root / name))
            for name in sorted(names)}


def training_fingerprint(project_root):
    """Conservative source identity including shared native/kernel dependencies."""
    root = Path(project_root).resolve()
    paths = [root / "experiments/liftquant_recovery" / name for name in TRAINING_FILES]
    for directory in ("litebsq", "sparse_bit_tuning", "rotation", "train_utils"):
        paths.extend(sorted((root / directory).rglob("*.py")))
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(set(paths))}


def boundary_identity(args, selected, source_fingerprint, code_fingerprint, ids, *, teacher_fingerprint=None):
    # Paths and the memory cap do not alter a training trajectory. Actual source
    # contents and sampled IDs are checked independently; all other CLI values
    # stay strict, including future training options added to the entry point.
    config = {key: value for key, value in vars(args).items() if key not in {
        "output", "resume", "checkpoint", "blocks", "gpu_memory_gib",
        "smoke_rows", "redpajama_arrow_dir",
    }}
    config["holdout"] = args.holdout if args.holdout is not None else args.nsamples // 32
    value = ids.detach().cpu().contiguous()
    return dict(
        source=source_fingerprint, teacher=teacher_fingerprint, training_code=code_fingerprint,
        config=config, selected_blocks=list(selected),
        calibration=dict(shape=list(value.shape), dtype=str(value.dtype),
                         sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest()),
        runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                     cpu_threads=torch.get_num_threads(), numpy=np.__version__,
                     transformers=version("transformers"), flash_attn=version("flash_attn"),
                     triton=version("triton")),
    )


def mutable_names_by_block(allowed_names, selected):
    result = {index: sorted(name for name in allowed_names
                            if name.startswith(f"model.layers.{index}.")) for index in selected}
    if any(not names for names in result.values()) or set().union(*map(set, result.values())) != set(allowed_names):
        raise ValueError("Every mutable native state must belong to exactly one selected block.")
    return result


def capture_rng(device=None):
    np_state = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=(np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=None if device is None else torch.cuda.get_rng_state(device).cpu(),
    )


def restore_rng(state, device=None):
    if (state["torch_cuda"] is None) != (device is None):
        raise ValueError("Boundary RNG device type differs from the current run.")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         numpy_state[2], numpy_state[3], numpy_state[4]))
    torch.set_rng_state(state["torch_cpu"])
    if device is not None:
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def _validate(payload, identity, names_by_block):
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("Expected a completed-block recovery boundary, not a native/mid-block checkpoint.")
    if payload.get("identity") != identity:
        differences = [key for key in identity if payload.get("identity", {}).get(key) != identity[key]]
        raise ValueError(f"Boundary identity mismatch: {differences}")
    completed = payload.get("completed")
    selected = identity["selected_blocks"]
    if not isinstance(completed, list) or not completed or completed != selected[:len(completed)]:
        raise ValueError("Completed blocks must be a nonempty prefix of the selected blocks.")
    expected = {name for index in completed for name in names_by_block[index]}
    if set(payload["state"]) != expected:
        raise ValueError("Boundary state must contain exactly the completed blocks' allowed native tensors.")
    if set(payload["records"]) != {str(index) for index in completed}:
        raise ValueError("Boundary metrics do not match the completed block prefix.")
    if set(payload["reload_inputs"]) != set(completed) or set(payload["reload_outputs"]) != set(completed):
        raise ValueError("Boundary strict-reload samples do not match completed blocks.")
    config = identity["config"]
    expected_steps = config["epochs"] * ((config["nsamples"] - config["holdout"]) // config["batch_size"])
    for record in payload["records"].values():
        if record.get("optimizer_steps") != expected_steps:
            raise ValueError("Boundary contains an incomplete or different-budget block.")


def save_boundary(path, model, identity, names_by_block, completed, records,
                  reload_inputs, reload_outputs, device=None):
    """Atomically replace this run's single latest state, never an earlier run."""
    path = Path(path)
    model_state = model.state_dict()
    names = {name for index in completed for name in names_by_block[index]}
    payload = dict(
        format=FORMAT, identity=identity, completed=list(completed),
        state={name: model_state[name].detach().cpu().clone() for name in sorted(names)},
        records=records, reload_inputs=reload_inputs, reload_outputs=reload_outputs,
        rng=capture_rng(device),
    )
    _validate(payload, identity, names_by_block)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(path=str(path), bytes=path.stat().st_size, completed=list(completed))


def load_boundary(path, model, identity, names_by_block):
    """Validate before copying tensors; caller rebuilds native derived decode plans.

    This generic CPU persistence helper does not import the GPU model stack.
    VAELLM callers must immediately invoke the existing
    refresh_vae_linear_runtime_after_state_load before evaluating restored blocks.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    _validate(payload, identity, names_by_block)
    native = model.state_dict()
    for name, value in payload["state"].items():
        if not isinstance(value, torch.Tensor) or name not in native:
            raise ValueError(f"Invalid native state tensor: {name}")
        if value.shape != native[name].shape or value.dtype != native[name].dtype:
            raise ValueError(f"Native state shape/dtype mismatch: {name}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite native state tensor: {name}")
    with torch.no_grad():
        for name, value in payload["state"].items():
            native[name].copy_(value)
    # State tensors are now held by the model; avoid retaining a duplicate payload.
    del payload["state"]
    return payload
