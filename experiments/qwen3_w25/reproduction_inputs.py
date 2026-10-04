"""Bind fresh Qwen3 reproduction inputs without historical result dependencies."""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

from experiments.qwen3_w25.reproduction_recipe import DATA_FILES, REPO, format_protocol


_HF_METADATA = (
    "config.json", "model.safetensors.index.json", "tokenizer_config.json",
    "tokenizer.json", "vocab.json", "merges.txt", "generation_config.json",
)
_SOURCE_DIRS = (
    "compressed_e2e_fintuning", "e2e_common", "train_utils", "litebsq",
    "sparse_bit_tuning", "rotation", "experiments/qwen3_w25", "experiments/liftquant_recovery",
)
_SOURCE_FILES = (
    "tools/cat_train.py", "tools/cat_eval.py", "tools/checkpoint_bpw.py",
    "scripts/cat_w25_remaining.sh",
)
_RECORDED_ENVIRONMENT = (
    "PYTHONPATH", "PYTORCH_CUDA_ALLOC_CONF", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG",
    "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE",
    "WANDB_MODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
    "FLASH_ATTENTION_DETERMINISTIC", "NVIDIA_TF32_OVERRIDE",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required reproduction input is missing: {path}")
    stat = path.stat()
    if stat.st_size == 0:
        raise ValueError(f"Required reproduction input is empty: {path}")
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def _write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _require_root(root: Path) -> Path:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"The caller must create a fresh run root first: {root}")
    return root


def _source_pins() -> dict[str, str]:
    # Pin the core training/export implementation as well as this experiment.
    paths = {path for name in _SOURCE_DIRS for path in (REPO / name).rglob("*.py")}
    paths.update(REPO / name for name in _SOURCE_FILES)
    return {str(path.resolve()): _sha(path) for path in sorted(paths)}


def _hf_inputs(hf_model: Path) -> dict:
    hf_model = Path(hf_model).resolve()
    if (hf_model / "checkpoint_meta.json").exists():
        raise ValueError("Start from the original BF16 HF Qwen3-8B, not a v6 compressed checkpoint.")
    config = json.loads((hf_model / "config.json").read_text(encoding="utf-8"))
    expected = {"model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 4096,
                "intermediate_size": 12288, "num_attention_heads": 32,
                "num_key_value_heads": 8, "vocab_size": 151936}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("The HF source must have the original Qwen3-8B model geometry.")
    if config.get("torch_dtype", config.get("dtype")) != "bfloat16" or config.get("quantization_config"):
        raise ValueError("The HF source must declare BF16 weights without a quantization configuration.")
    index = json.loads((hf_model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("The original HF source requires a nonempty safetensors weight index.")
    shards = sorted(set(weight_map.values()))
    if any(not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors")
           for name in shards):
        raise ValueError("HF weight shards must be safetensors files inside the original model directory.")
    weights = [_stat(hf_model / name) for name in shards]
    for name in _HF_METADATA:
        _stat(hf_model / name)
    return {"path": str(hf_model), "config": config,
            "metadata_sha256": {str(hf_model / name): _sha(hf_model / name) for name in _HF_METADATA},
            "weight_files": weights,
            "weight_identity_scope": "Existing shard paths/size/mtime/inode; large HF shards are not rehashed."}


def prepare_inputs(root: Path, hf_model: Path) -> dict:
    """Check the fixed data/model inputs and publish the new task manifest."""
    from experiments.qwen3_w25.format_training_data import SOURCE_ROWS, SOURCE_SHA

    root = _require_root(root)
    manifest_path = root / "task_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to replace an existing task manifest: {manifest_path}")
    hf_source = _hf_inputs(hf_model)
    data_stats = [_stat(REPO / name) for name in DATA_FILES]
    task_source = (REPO / DATA_FILES[-1]).resolve()
    before = _stat(task_source)
    digest, rows = hashlib.sha256(), 0
    with task_source.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            rows += 1
    if _stat(task_source) != before:
        raise RuntimeError("The task data changed while its identity was being checked.")
    if rows != SOURCE_ROWS or digest.hexdigest() != SOURCE_SHA:
        raise ValueError("Expected the fixed repaired task data: 176597 rows and the recorded SHA256.")
    manifest = {"path": str(task_source), "samples": rows,
                "size_bytes": before["size"], "sha256": digest.hexdigest(),
                "split_policy": "Task training splits only; validation/test evaluation splits excluded."}
    versions = {"python": platform.python_version(), "python_executable": sys.executable}
    for package in ("torch", "transformers", "accelerate", "datasets", "lm_eval", "triton", "flash_attn", "safetensors"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    pins = _source_pins()
    _write_new(manifest_path, manifest)
    return {"task_source": str(task_source), "task_manifest": str(manifest_path),
            "task_manifest_sha256": _sha(manifest_path), "input_data_stat": data_stats,
            "hf_source": hf_source, "versions": versions, "source_sha256": pins}


def prepare_format_launch(*, root: Path, source_dir: Path, hf_model: Path,
                          archive_dir: Path, environment: dict, formal_command: list[str]) -> Path:
    """Bind a fresh assembled source to the unchanged fixed256 recovery loop."""
    from experiments.qwen3_w25.format_training_data import FORMAT, SOURCE_ROWS, SOURCE_SHA

    root = _require_root(root)
    source_dir, archive_dir = Path(source_dir).resolve(), Path(archive_dir).resolve()
    reference_path, launch_path = root / "format_reference.json", root / "format_launch.json"
    observations = root / "format_observations"
    if any(path.exists() for path in (reference_path, launch_path, observations)):
        raise FileExistsError("Use fresh format-reference, launch, and observation paths.")
    meta_path = source_dir / "checkpoint_meta.json"
    source_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_id = source_meta.get("checkpoint_id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("The fresh assembled source lacks a checkpoint identity.")
    hf_source = _hf_inputs(hf_model)
    if Path(source_meta["base_model_path"]).resolve() != Path(hf_source["path"]):
        raise ValueError("The assembled source and teacher must reference the same original HF model.")
    preparation_path = archive_dir / "preparation.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    archive = preparation["archive"]
    archive_path = archive_dir / "batches.pt"
    source = preparation["source"]
    audit = preparation["audit"]
    if (preparation["status"] != "completed_verified" or preparation["format"] != FORMAT
            or Path(archive["path"]).resolve() != archive_path
            or archive["sha256"] != _sha(archive_path)
            or archive["size_bytes"] != _stat(archive_path)["size"]
            or source["sha256"] != SOURCE_SHA or source["rows"] != SOURCE_ROWS
            or audit["training_sequences"] != 1024 or audit["heldout_sequences"] != 32
            or audit["train_heldout_raw_and_content_intersection"] != 0
            or not audit["source_replay_exact"]):
        raise ValueError("The fixed256 archive does not match the verified paired task1024+32 protocol.")
    for path, digest in preparation["tokenizer"]["files_sha256"].items():
        if _sha(Path(path)) != digest or hf_source["metadata_sha256"].get(path) != digest:
            raise ValueError(f"Archive tokenizer differs from the original HF tokenizer: {path}")
    for path, digest in preparation["source_sha256"].items():
        if _sha(Path(path)) != digest:
            raise ValueError(f"Archive preparation source changed: {path}")
    task_stat = _stat(Path(source["path"]))
    if (task_stat["size"], task_stat["mtime_ns"], task_stat["inode"]) != tuple(
            source[key] for key in ("size_bytes", "mtime_ns", "inode")):
        raise ValueError("The task data differs from the verified archive input.")
    manifest_path = Path(source["manifest"]["path"])
    if _sha(manifest_path) != source["manifest"]["sha256"]:
        raise ValueError("The task manifest differs from the verified archive input.")
    if (formal_command[1:3] != ["-m", "experiments.qwen3_w25.train_with_step_exports"]
            or formal_command[formal_command.index("--student_checkpoint_dir") + 1] != str(source_dir)
            or Path(formal_command[formal_command.index("--teacher_model_path") + 1]).resolve() != Path(hf_source["path"])):
        raise ValueError("The optimizer reference must bind the fresh source and original teacher.")
    reference = {"status": "configuration_only", "executed": False,
                 "purpose": "Optimizer/runtime reference for fixed256; this full training command is not executed.",
                 "actual_argv": list(formal_command), "source_checkpoint": str(source_dir),
                 "source_checkpoint_id": source_id}
    record = {
        "status": "prepared", "cwd": str(REPO), "python_executable": sys.executable,
        "training_format": "chatml", "mode": "joint", "steps": 256,
        "training_label_region": "response", "output_dir": str(observations),
        "source_checkpoint": str(source_dir), "source_checkpoint_id": source_id,
        "environment": {key: environment[key] for key in _RECORDED_ENVIRONMENT if key in environment},
        "protocol": format_protocol(),
        "source_sha256": _source_pins(),
        "input_metadata_sha256": {
            str(meta_path): _sha(meta_path), str(source_dir / "config.json"): _sha(source_dir / "config.json"),
            **hf_source["metadata_sha256"],
        },
        "input_evidence_sha256": {
            str(preparation_path): _sha(preparation_path), str(manifest_path): _sha(manifest_path),
        },
        "input_stats": [_stat(source_dir / "pytorch_model.bin"), *hf_source["weight_files"], task_stat],
        "data_archive": {
            "path": str(archive_path), "sha256": archive["sha256"], "format": FORMAT,
            "training_sequences": 1024, "heldout_sequences": 32, "batch_size": 4, "max_length": 1024,
            "size_bytes": archive["size_bytes"], "preparation_path": str(preparation_path),
            "preparation_sha256": _sha(preparation_path),
        },
        "data_audit": audit,
        "loss_recipe": {"name": "ce50_prompt0", "loss_type": "kd_top_mass", "alpha": 0.5,
                        "alpha_applies": True, "top_k": 100, "temperature": 1.0,
                        "prompt_loss_weight": 0.0, "reported_loss": "0.5 response CE + 0.5 response mass-KL"},
    }
    _write_new(reference_path, reference)
    record["formal_reference"] = {"path": str(reference_path), "sha256": _sha(reference_path),
                                  "scope": reference["purpose"]}
    _write_new(launch_path, record)
    return launch_path
