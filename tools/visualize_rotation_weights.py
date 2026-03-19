import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from json import JSONDecoder
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

log = logging.getLogger("visualize_rotation_weights")
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
log.setLevel(logging.INFO)
log.propagate = False

_DEFAULT_CATEGORIES = ("q_proj", "k_proj", "v_proj", "o_proj")
_DEFAULT_OUTPUT_ROOT = os.path.join(".result", "rotation_viz")
_LAYER_IDX_PATTERNS = (
    re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\."),
    re.compile(r"(?:^|\.)(?:model\.)?decoder\.layers\.(\d+)\."),
)


@dataclass(frozen=True)
class CandidateLinear:
    full_name: str
    layer_idx: Optional[int]
    category: str
    module: nn.Linear


def _extract_layer_idx(name: str) -> Optional[int]:
    for pattern in _LAYER_IDX_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def _safe_token(text: str) -> str:
    token = str(text).strip().replace("\\", "/").rstrip("/")
    token = token.split("/")[-1] if token else "item"
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token)
    token = token.strip("._")
    return token or "item"


def _parse_csv_names(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    raw = str(value).strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_csv_ints(value: Optional[str]) -> List[int]:
    values = _parse_csv_names(value)
    if not values:
        return []
    out: List[int] = []
    for item in values:
        try:
            out.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid layer index '{item}'. Expected comma-separated integers.") from exc
    return out


def _extract_json_after_key(text: str, key: str) -> dict:
    marker = f"{key}="
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Could not find '{marker}' in log.")
    start += len(marker)
    while start < len(text) and text[start].isspace():
        start += 1
    decoder = JSONDecoder()
    obj, _end = decoder.raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object after '{marker}'.")
    return obj


def _parse_train_log_config(path: str) -> Dict[str, object]:
    log_path = Path(path)
    if not log_path.is_file():
        raise FileNotFoundError(f"Train log not found: {path}")
    text = log_path.read_text(encoding="utf-8")
    script_args = _extract_json_after_key(text, "script")
    vae_args = _extract_json_after_key(text, "vae")
    model_path = vae_args.get("model_path")
    if not model_path:
        raise ValueError(f"Missing vae.model_path in train log: {path}")
    return {
        "model_path": str(model_path),
        "rot_llm": script_args.get("rot_llm"),
        "script_args": script_args,
        "vae_args": vae_args,
        "path": str(log_path),
    }


def _collect_candidate_linears(model: nn.Module) -> List[CandidateLinear]:
    candidates: List[CandidateLinear] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        layer_idx = _extract_layer_idx(name)
        if layer_idx is None:
            continue
        candidates.append(
            CandidateLinear(
                full_name=name,
                layer_idx=layer_idx,
                category=name.split(".")[-1],
                module=module,
            )
        )
    candidates.sort(key=lambda item: (item.layer_idx if item.layer_idx is not None else -1, item.category, item.full_name))
    return candidates


def _format_candidate_table(candidates: Sequence[CandidateLinear]) -> str:
    lines = []
    for item in candidates:
        lines.append(f"layer={item.layer_idx} category={item.category} full_name={item.full_name}")
    return "\n".join(lines)


def _resolve_targets(
    candidates: Sequence[CandidateLinear],
    *,
    layers: Sequence[int],
    categories: Sequence[str],
    full_names: Sequence[str],
) -> List[CandidateLinear]:
    candidate_by_name = {item.full_name: item for item in candidates}
    if full_names:
        missing = [name for name in full_names if name not in candidate_by_name]
        if missing:
            available = _format_candidate_table(candidates)
            raise ValueError(
                "Target linear not found for full_names:\n"
                + "\n".join(missing)
                + ("\nAvailable candidates:\n" + available if available else "\nNo decoder-layer nn.Linear candidates found.")
            )
        return [candidate_by_name[name] for name in full_names]

    if not layers:
        raise ValueError("--layers is required unless --full_names is provided.")

    category_set = set(categories)
    layer_set = set(int(layer) for layer in layers)
    selected = [
        item
        for item in candidates
        if item.layer_idx in layer_set and item.category in category_set
    ]
    if not selected:
        available = _format_candidate_table(candidates)
        raise ValueError(
            "No target linears matched the requested layers/categories. "
            f"layers={sorted(layer_set)}, categories={sorted(category_set)}"
            + ("\nAvailable candidates:\n" + available if available else "\nNo decoder-layer nn.Linear candidates found.")
        )
    return selected


def _snapshot_weights(targets: Sequence[CandidateLinear]) -> Dict[str, torch.Tensor]:
    snapshots: Dict[str, torch.Tensor] = {}
    for item in targets:
        snapshots[item.full_name] = item.module.weight.detach().to(device="cpu", dtype=torch.float32).clone()
    return snapshots


def _to_numpy(matrix: torch.Tensor) -> np.ndarray:
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(matrix)}")
    return matrix.detach().to(device="cpu", dtype=torch.float32).numpy()


def _check_finite(name: str, matrix: np.ndarray) -> None:
    if not np.isfinite(matrix).all():
        nan_count = int(np.isnan(matrix).sum())
        inf_count = int(np.isinf(matrix).sum())
        raise ValueError(f"{name} contains non-finite values: nan={nan_count}, inf={inf_count}")


def _matrix_stats(matrix: np.ndarray) -> Dict[str, object]:
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={matrix.shape}")
    flat = matrix.reshape(-1).astype(np.float64, copy=False)
    row_norms = np.linalg.norm(matrix.astype(np.float64, copy=False), axis=1)
    col_norms = np.linalg.norm(matrix.astype(np.float64, copy=False), axis=0)
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "abs_mean": float(np.abs(flat).mean()),
        "abs_max": float(np.abs(flat).max()),
        "p01": float(np.percentile(flat, 1.0)),
        "p50": float(np.percentile(flat, 50.0)),
        "p99": float(np.percentile(flat, 99.0)),
        "fro_norm": float(np.linalg.norm(flat)),
        "row_norm_mean": float(row_norms.mean()),
        "row_norm_std": float(row_norms.std()),
        "row_norm_max": float(row_norms.max()),
        "col_norm_mean": float(col_norms.mean()),
        "col_norm_std": float(col_norms.std()),
        "col_norm_max": float(col_norms.max()),
        "has_nan": bool(np.isnan(flat).any()),
        "has_inf": bool(np.isinf(flat).any()),
    }


def _build_summary(
    *,
    target: CandidateLinear,
    before: np.ndarray,
    after: np.ndarray,
    source_model_path: str,
    train_log_rot_llm: Optional[object],
) -> Dict[str, object]:
    if before.shape != after.shape:
        raise ValueError(f"Shape mismatch: before={before.shape}, after={after.shape}")
    delta = after - before
    before_flat = before.reshape(-1).astype(np.float64, copy=False)
    after_flat = after.reshape(-1).astype(np.float64, copy=False)
    delta_flat = delta.reshape(-1).astype(np.float64, copy=False)
    delta_rms = float(np.sqrt(np.mean(np.square(delta_flat))))
    before_rms = float(np.sqrt(np.mean(np.square(before_flat))))
    denom = before_rms if before_rms > 0.0 else None
    cosine_denom = float(np.linalg.norm(before_flat) * np.linalg.norm(after_flat))
    cosine = None if cosine_denom == 0.0 else float(np.dot(before_flat, after_flat) / cosine_denom)
    return {
        "shape": [int(before.shape[0]), int(before.shape[1])],
        "source_model_path": str(source_model_path),
        "layer_idx": target.layer_idx,
        "category": target.category,
        "full_name": target.full_name,
        "train_log_rot_llm": train_log_rot_llm,
        "delta_rms": delta_rms,
        "delta_rel_rms": None if denom is None else float(delta_rms / denom),
        "flat_cosine_similarity": cosine,
        "before": _matrix_stats(before),
        "after": _matrix_stats(after),
        "delta": _matrix_stats(delta),
    }


def _downsample_matrix(matrix: np.ndarray, max_points: int) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={matrix.shape}")
    limit = int(max_points)
    if limit < 1:
        raise ValueError(f"max_points must be >= 1, got {max_points}")
    rows, cols = matrix.shape
    row_step = max(1, int(np.ceil(rows / limit)))
    col_step = max(1, int(np.ceil(cols / limit)))
    return matrix[::row_step, ::col_step]


def _sample_axis_indices(length: int, max_points: int) -> np.ndarray:
    size = int(length)
    if size < 1:
        raise ValueError(f"Axis length must be >= 1, got {length}")
    limit = int(max_points)
    if limit <= 0 or limit >= size:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, num=limit, dtype=np.int64)


def _sample_matrix_for_surface(
    matrix: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={matrix.shape}")
    row_idx = _sample_axis_indices(matrix.shape[0], max_points)
    col_idx = _sample_axis_indices(matrix.shape[1], max_points)
    sampled = matrix[np.ix_(row_idx, col_idx)]
    return sampled, row_idx, col_idx


def _max_abs_limit(matrices: Iterable[np.ndarray]) -> float:
    values = [np.abs(matrix.reshape(-1)) for matrix in matrices if matrix.size > 0]
    if not values:
        return 1.0
    merged = np.concatenate(values, axis=0)
    limit = float(np.max(merged))
    if not np.isfinite(limit) or limit <= 0.0:
        return 1.0
    return limit


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        return plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate visualization outputs. "
            "Please install matplotlib in the active Python environment."
        ) from exc


def _plot_3d_matrix_on_axis(
    ax,
    matrix: np.ndarray,
    *,
    title: str,
    x_indices: Optional[np.ndarray] = None,
    y_indices: Optional[np.ndarray] = None,
    cmap: str = "coolwarm",
):
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={matrix.shape}")
    rows, cols = matrix.shape
    x = np.arange(cols, dtype=np.int64) if x_indices is None else np.asarray(x_indices, dtype=np.int64)
    y = np.arange(rows, dtype=np.int64) if y_indices is None else np.asarray(y_indices, dtype=np.int64)
    if x.shape[0] != cols:
        raise ValueError(f"x_indices length mismatch: got {x.shape[0]}, expected {cols}")
    if y.shape[0] != rows:
        raise ValueError(f"y_indices length mismatch: got {y.shape[0]}, expected {rows}")
    xx, yy = np.meshgrid(x, y)
    abs_matrix = np.abs(matrix)
    ax.plot_surface(
        xx,
        yy,
        abs_matrix,
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("Input Dimension")
    ax.set_ylabel("Output Dimension")
    ax.set_zlabel("Absolute Weight")


def plot_3d_matrix(
    matrix,
    title: str = "3D Surface Plot",
    xlabel: str = "Input Dimension",
    ylabel: str = "Output Dimension",
    zlabel: str = "Absolute Intensity",
    cmap: str = "coolwarm",
    max_surface_points: int = 0,
):
    plt = _require_matplotlib()
    if isinstance(matrix, torch.Tensor):
        matrix = _to_numpy(matrix)
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix (numpy array or torch tensor).")
    sampled_matrix, row_idx, col_idx = _sample_matrix_for_surface(matrix, max_surface_points)
    abs_matrix = np.abs(sampled_matrix)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xx, yy = np.meshgrid(col_idx, row_idx)
    ax.plot_surface(
        xx,
        yy,
        abs_matrix,
        cmap=cmap,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.title.set_text(title)
    return fig


def _plot_surface_triptych(
    before: np.ndarray,
    after: np.ndarray,
    delta: np.ndarray,
    output_path: Path,
    *,
    max_surface_points: int,
    clip_percentile: float,
    dpi: int,
) -> None:
    plt = _require_matplotlib()
    before_surface, before_rows, before_cols = _sample_matrix_for_surface(before, max_surface_points)
    after_surface, after_rows, after_cols = _sample_matrix_for_surface(after, max_surface_points)
    delta_surface, delta_rows, delta_cols = _sample_matrix_for_surface(delta, max_surface_points)

    fig = plt.figure(figsize=(18, 5))
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]
    _plot_3d_matrix_on_axis(
        axes[0],
        before_surface,
        title="|Before Rotation|",
        x_indices=before_cols,
        y_indices=before_rows,
    )
    _plot_3d_matrix_on_axis(
        axes[1],
        after_surface,
        title="|After Rotation|",
        x_indices=after_cols,
        y_indices=after_rows,
    )
    _plot_3d_matrix_on_axis(
        axes[2],
        delta_surface,
        title="|After - Before|",
        x_indices=delta_cols,
        y_indices=delta_rows,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _sorted_norms(matrix: np.ndarray, axis: int) -> np.ndarray:
    norms = np.linalg.norm(matrix.astype(np.float64, copy=False), axis=axis)
    return np.sort(norms)[::-1]


def _plot_distribution_panel(
    before: np.ndarray,
    after: np.ndarray,
    output_path: Path,
    *,
    dpi: int,
) -> None:
    plt = _require_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    before_flat = before.reshape(-1)
    after_flat = after.reshape(-1)

    axes[0, 0].hist(before_flat, bins=120, alpha=0.6, label="before", density=True)
    axes[0, 0].hist(after_flat, bins=120, alpha=0.6, label="after", density=True)
    axes[0, 0].set_title("Value Distribution")
    axes[0, 0].legend()

    axes[0, 1].hist(np.abs(before_flat), bins=120, alpha=0.6, label="before", density=True)
    axes[0, 1].hist(np.abs(after_flat), bins=120, alpha=0.6, label="after", density=True)
    axes[0, 1].set_title("Absolute Value Distribution")
    axes[0, 1].legend()

    before_row = _sorted_norms(before, axis=1)
    after_row = _sorted_norms(after, axis=1)
    axes[1, 0].plot(before_row, label="before")
    axes[1, 0].plot(after_row, label="after")
    axes[1, 0].set_title("Sorted Row L2 Norms")
    axes[1, 0].legend()

    before_col = _sorted_norms(before, axis=0)
    after_col = _sorted_norms(after, axis=0)
    axes[1, 1].plot(before_col, label="before")
    axes[1, 1].plot(after_col, label="after")
    axes[1, 1].set_title("Sorted Column L2 Norms")
    axes[1, 1].legend()

    for axis in axes.reshape(-1):
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_heatmap(
    delta: np.ndarray,
    output_path: Path,
    *,
    clip_percentile: float,
    dpi: int,
) -> None:
    plt = _require_matplotlib()
    limit = _max_abs_limit([delta])
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        delta,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        origin="lower",
        interpolation="nearest",
        extent=(0, delta.shape[1], 0, delta.shape[0]),
    )
    ax.set_title("Delta Heatmap (After - Before)")
    ax.set_xlabel("Input Dimension")
    ax.set_ylabel("Output Dimension")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_target_report(
    target: CandidateLinear,
    *,
    before: np.ndarray,
    after: np.ndarray,
    source_model_path: str,
    output_dir: Path,
    plot_bundle: str,
    max_surface_points: int,
    clip_percentile: float,
    dpi: int,
    save_npz: bool,
    train_log_rot_llm: Optional[object],
) -> Dict[str, object]:
    delta = after - before
    summary = _build_summary(
        target=target,
        before=before,
        after=after,
        source_model_path=source_model_path,
        train_log_rot_llm=train_log_rot_llm,
    )
    _write_json(output_dir / "summary.json", summary)

    has_non_finite = not (np.isfinite(before).all() and np.isfinite(after).all() and np.isfinite(delta).all())
    if has_non_finite:
        return {
            "full_name": target.full_name,
            "layer_idx": target.layer_idx,
            "category": target.category,
            "output_dir": str(output_dir),
            "summary_path": str(output_dir / "summary.json"),
            "plots_skipped": True,
            "reason": "non_finite_values",
        }

    if plot_bundle in {"full", "surface_only"}:
        _plot_surface_triptych(
            before,
            after,
            delta,
            output_dir / "surface_triptych.png",
            max_surface_points=max_surface_points,
            clip_percentile=clip_percentile,
            dpi=dpi,
        )
    if plot_bundle in {"full", "stats_only"}:
        _plot_distribution_panel(
            before,
            after,
            output_dir / "distribution_panel.png",
            dpi=dpi,
        )
        _plot_delta_heatmap(
            delta,
            output_dir / "delta_heatmap.png",
            clip_percentile=clip_percentile,
            dpi=dpi,
        )
    if save_npz:
        np.savez_compressed(output_dir / "matrices.npz", before=before, after=after, delta=delta)

    return {
        "full_name": target.full_name,
        "layer_idx": target.layer_idx,
        "category": target.category,
        "output_dir": str(output_dir),
        "summary_path": str(output_dir / "summary.json"),
        "surface_triptych_path": str(output_dir / "surface_triptych.png")
        if plot_bundle in {"full", "surface_only"}
        else None,
        "distribution_panel_path": str(output_dir / "distribution_panel.png")
        if plot_bundle in {"full", "stats_only"}
        else None,
        "delta_heatmap_path": str(output_dir / "delta_heatmap.png")
        if plot_bundle in {"full", "stats_only"}
        else None,
        "matrices_path": str(output_dir / "matrices.npz") if save_npz else None,
        "plots_skipped": False,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize weight-distribution changes for selected decoder linear layers "
            "before and after offline rotation fusion."
        )
    )
    parser.add_argument("--model_path", type=str, default=None, help="HF model id or local path.")
    parser.add_argument(
        "--train_log",
        type=str,
        default=None,
        help="Optional linear_by_category.log path. Used to recover model_path and record rot_llm in summary.",
    )
    parser.add_argument("--access_token", type=str, default=None, help="HuggingFace access token.")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices, e.g. 2,10,18.")
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(_DEFAULT_CATEGORIES),
        help="Comma-separated target categories. Default: q_proj,k_proj,v_proj,o_proj.",
    )
    parser.add_argument(
        "--full_names",
        type=str,
        default=None,
        help="Comma-separated full module names. Overrides --layers/--categories when provided.",
    )
    parser.add_argument("--list_targets", action="store_true", help="List decoder-layer nn.Linear targets and exit.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output root directory. Default: .result/rotation_viz/<model_token>/",
    )
    parser.add_argument(
        "--plot_bundle",
        choices=("full", "surface_only", "stats_only"),
        default="full",
        help="Which figures to generate.",
    )
    parser.add_argument(
        "--max_surface_points",
        type=int,
        default=0,
        help=(
            "Maximum rendered points per surface dimension. "
            "Sampling is only for display; axis coordinates still reflect the original input/output dimensions. "
            "<=0 means render every point."
        ),
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.5,
        help="Deprecated and ignored. Visualization no longer clips outliers.",
    )
    parser.add_argument("--save_npz", action="store_true", help="Save before/after/delta matrices as matrices.npz.")
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    return parser


def _load_model(model_path: str, access_token: Optional[str]):
    try:
        from rotation.model_utils import get_model

        return get_model(model_path, access_token)
    except Exception as exc:
        raise RuntimeError(
            f"Model load or authentication failed while loading '{model_path}': {exc}"
        ) from exc


def _apply_rotation(model):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to generate rotated snapshots because rotation.prepare_model() uses cuda:0 internally."
        )
    try:
        from rotation.model_rotation import prepare_model

        return prepare_model(model)
    except Exception as exc:
        raise RuntimeError(f"Rotation fusion failed while preparing the model: {exc}") from exc


def _ensure_writable_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to create or write output directory '{path}': {exc}") from exc


def _resolve_model_source(args) -> Tuple[str, Optional[object], Optional[str]]:
    train_log_info = None
    train_log_rot_llm = None
    train_log_path = None
    if args.train_log:
        train_log_info = _parse_train_log_config(args.train_log)
        train_log_rot_llm = train_log_info.get("rot_llm")
        train_log_path = str(train_log_info.get("path"))
    model_path = args.model_path or (train_log_info["model_path"] if train_log_info is not None else None)
    if not model_path:
        raise ValueError("Either --model_path or --train_log must be provided.")
    return str(model_path), train_log_rot_llm, train_log_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.access_token is not None and not str(args.access_token).strip():
        args.access_token = None

    model_path, train_log_rot_llm, train_log_path = _resolve_model_source(args)
    categories = _parse_csv_names(args.categories) or list(_DEFAULT_CATEGORIES)
    full_names = _parse_csv_names(args.full_names)
    layers = _parse_csv_ints(args.layers)

    model = _load_model(model_path, args.access_token)
    candidates = _collect_candidate_linears(model)

    if args.list_targets:
        print(_format_candidate_table(candidates))
        return

    targets = _resolve_targets(
        candidates,
        layers=layers,
        categories=categories,
        full_names=full_names,
    )

    before_snapshots = _snapshot_weights(targets)
    log.info("Captured %d target matrices before rotation.", len(before_snapshots))
    model = _apply_rotation(model)
    after_snapshots = _snapshot_weights(targets)
    log.info("Captured %d target matrices after rotation.", len(after_snapshots))

    output_root = Path(args.output_dir or os.path.join(_DEFAULT_OUTPUT_ROOT, _safe_token(model_path)))
    _ensure_writable_dir(output_root)

    index_entries: List[Dict[str, object]] = []
    for target in targets:
        before = _to_numpy(before_snapshots[target.full_name])
        after = _to_numpy(after_snapshots[target.full_name])
        target_dir = output_root / f"{target.layer_idx}__{target.category}__{_safe_token(target.full_name)}"
        _ensure_writable_dir(target_dir)
        report = _write_target_report(
            target,
            before=before,
            after=after,
            source_model_path=model_path,
            output_dir=target_dir,
            plot_bundle=args.plot_bundle,
            max_surface_points=int(args.max_surface_points),
            clip_percentile=float(args.clip_percentile),
            dpi=int(args.dpi),
            save_npz=bool(args.save_npz),
            train_log_rot_llm=train_log_rot_llm,
        )
        index_entries.append(report)
        log.info("Wrote report for %s -> %s", target.full_name, target_dir)

    index_payload = {
        "source_model_path": model_path,
        "train_log_path": train_log_path,
        "train_log_rot_llm": train_log_rot_llm,
        "plot_bundle": args.plot_bundle,
        "categories": categories,
        "layers": layers,
        "full_names": full_names,
        "entries": index_entries,
    }
    _write_json(output_root / "index.json", index_payload)
    log.info("Completed. Index written to %s", output_root / "index.json")


if __name__ == "__main__":
    main()
