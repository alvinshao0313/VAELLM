from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LinearPrepRef:
    name: str
    weight: torch.Tensor
    in_features: int
    out_features: int
    transpose: bool


@dataclass(frozen=True)
class WAMSEPartMeta:
    linear_name: str
    transpose: bool
    cols: int
    row_offset: int
    rows_part: int
    col_offset: int
    act_max: torch.Tensor  # [in_features], float32 on train device
    col_index_map: Optional[torch.Tensor] = None  # [cols_part], long on train device; sorted-col -> original channel idx


@dataclass(frozen=True)
class LinearSplitMeta:
    linear_name: str
    transpose: bool
    sort_mode: str
    parallel_rows: int
    parallel_cols: int
    restore_row_indices: Optional[torch.Tensor]  # legacy global restore rows for old checkpoints
    restore_col_indices: Optional[torch.Tensor]  # legacy global restore cols for old checkpoints
    part_restore_col_indices: Optional[torch.Tensor]  # [parallel_parts, cols_per_part], long on cpu
    compressed_in_features: int
    compressed_out_features: int
    protected_input_indices: Optional[torch.Tensor]  # [num_protected], long on cpu; original input channel indices
    protected_input_weight: Optional[torch.Tensor]  # [num_protected, out_features], same dtype as original weight on cpu
    protected_output_indices: Optional[torch.Tensor]  # [num_protected], long on cpu; original output channel indices
    protected_output_weight: Optional[torch.Tensor]  # [num_protected, in_features], same dtype as original weight on cpu


@dataclass(frozen=True)
class PreparedLinearWeight:
    split_weight: torch.Tensor  # [compressed_out_features, compressed_in_features] in original Linear layout
    compressed_in_features: int
    compressed_out_features: int
    activation_weight: Optional[torch.Tensor]  # input-axis act vector used by act_spectral_cosine / wa_mse, float32 on cpu
    protected_input_indices: Optional[torch.Tensor]  # [num_protected], long on cpu
    protected_input_weight: Optional[torch.Tensor]  # [num_protected, out_features], original dtype on cpu
    protected_output_indices: Optional[torch.Tensor]  # [num_protected], long on cpu
    protected_output_weight: Optional[torch.Tensor]  # [num_protected, in_features], original dtype on cpu


@dataclass
class GroupDataPrepResult:
    num_models: int
    codebook_dim: int
    stacked_data: torch.Tensor  # [N_blocks, P, codebook_dim]
    d_mean: torch.Tensor  # [P, 1]
    d_std: torch.Tensor  # [P, 1]
    train_loader: torch.utils.data.DataLoader
    eval_loader: torch.utils.data.DataLoader
    use_wa_mse: bool
    part_metas: List[WAMSEPartMeta]
    split_metas: List[LinearSplitMeta]


_INTRA_PART_SORT_MODE_CHOICES = {"none", "spectral_cosine", "act_spectral_cosine"}
_INTRA_PART_SORT_MODE_HELP = "Expected one of: none,spectral_cosine,act_spectral_cosine."
_SPECTRAL_TOPK_NEIGHBORS = 32
_SPECTRAL_SIM_CHUNK = 256
_SPECTRAL_REFINE_SWEEPS = 4


def _normalize_sort_mode_choice(value: object, *, arg_name: str) -> str:
    raw = str(value).strip().lower()
    raw = raw.strip(" \t\r\n'\"()[]")
    if raw not in _INTRA_PART_SORT_MODE_CHOICES:
        raise ValueError(
            f"Unsupported {arg_name}={value}. "
            + _INTRA_PART_SORT_MODE_HELP
        )
    return raw


def normalize_intra_part_sort_mode(
    sort_mode: Union[str, Sequence[str]],
    *,
    arg_name: str = "intra_part_sort_mode",
) -> str:
    if isinstance(sort_mode, (list, tuple)):
        items = list(sort_mode)
        if len(items) == 0:
            raise ValueError(f"{arg_name} cannot be empty.")
        if len(items) == 1:
            return _normalize_sort_mode_choice(items[0], arg_name=arg_name)
        raise ValueError(
            f"Unsupported {arg_name}={sort_mode}. "
            "Expected exactly one scalar mode."
        )

    raw = str(sort_mode).strip()
    if not raw:
        raise ValueError(f"{arg_name} cannot be empty.")
    if any(sep in raw for sep in (",", "|", ":")):
        raise ValueError(
            f"Unsupported {arg_name}={sort_mode}. "
            "Only single-mode syntax is supported."
        )
    return _normalize_sort_mode_choice(raw, arg_name=arg_name)


def format_intra_part_sort_mode(sort_mode: Union[str, Sequence[str]]) -> str:
    return normalize_intra_part_sort_mode(sort_mode)


def resolve_intra_parallel(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"intra_parallel must be >= 1, got {value}")
        return int(value), 1

    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"intra_parallel must be int or list/tuple with length 1/2, got {type(value)}"
        )
    parts = [int(v) for v in value]
    if len(parts) == 1:
        if parts[0] < 1:
            raise ValueError(f"intra_parallel must be >= 1, got {parts[0]}")
        return parts[0], 1
    if len(parts) != 2:
        raise ValueError(
            f"intra_parallel list/tuple must have length 1 or 2, got {len(parts)}"
        )
    if parts[0] < 1 or parts[1] < 1:
        raise ValueError(f"intra_parallel factors must be >= 1, got {parts}")
    return int(parts[0]), int(parts[1])


def _build_restore_indices(sorted_indices: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if sorted_indices is None:
        return None
    restore = torch.empty_like(sorted_indices)
    restore[sorted_indices] = torch.arange(
        int(sorted_indices.numel()),
        device=sorted_indices.device,
        dtype=sorted_indices.dtype,
    )
    return restore


def _sum_block_variance(part: torch.Tensor, codebook_dim: int) -> float:
    blocks = part.contiguous().view(-1, int(codebook_dim))
    centered = blocks - blocks.mean(dim=1, keepdim=True)
    return float((centered * centered).sum().item())


def _build_sparse_affinity_from_embeddings(
    embeddings: torch.Tensor,
    *,
    importance: Optional[torch.Tensor],
) -> sp.csr_matrix:
    num_cols = int(embeddings.shape[0])
    if num_cols <= 1:
        return sp.csr_matrix((num_cols, num_cols), dtype=np.float32)
    k = min(_SPECTRAL_TOPK_NEIGHBORS, num_cols - 1)
    row_idx_list: List[np.ndarray] = []
    col_idx_list: List[np.ndarray] = []
    value_list: List[np.ndarray] = []
    importance_cpu = None
    if importance is not None:
        importance_cpu = importance.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if int(importance_cpu.numel()) != num_cols:
            raise ValueError(
                f"importance size mismatch: got {int(importance_cpu.numel())}, expected {num_cols}."
            )
    embeddings = embeddings.detach().to(device="cpu", dtype=torch.float32).contiguous()
    for start in range(0, num_cols, _SPECTRAL_SIM_CHUNK):
        end = min(start + _SPECTRAL_SIM_CHUNK, num_cols)
        sim = embeddings[start:end] @ embeddings.t()
        sim = sim.clamp_min_(0.0)
        if importance_cpu is not None:
            sim = sim * importance_cpu[start:end].view(-1, 1)
            sim = sim * importance_cpu.view(1, -1)
        local_rows = torch.arange(start, end, dtype=torch.long).view(-1, 1)
        sim.scatter_(1, local_rows, 0.0)
        top_vals, top_idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)
        keep_mask = top_vals > 0
        if not torch.any(keep_mask):
            continue
        kept_rows = torch.arange(start, end, dtype=torch.long).unsqueeze(1).expand(-1, k)[keep_mask]
        row_idx_list.append(kept_rows.cpu().numpy().astype(np.int64, copy=False))
        col_idx_list.append(top_idx[keep_mask].cpu().numpy().astype(np.int64, copy=False))
        value_list.append(top_vals[keep_mask].cpu().numpy().astype(np.float32, copy=False))
    if not value_list:
        return sp.csr_matrix((num_cols, num_cols), dtype=np.float32)
    affinity = sp.coo_matrix(
        (
            np.concatenate(value_list, axis=0),
            (np.concatenate(row_idx_list, axis=0), np.concatenate(col_idx_list, axis=0)),
        ),
        shape=(num_cols, num_cols),
        dtype=np.float32,
    ).tocsr()
    affinity = affinity.maximum(affinity.transpose())
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()
    return affinity


def _spectral_order_from_affinity(
    affinity: sp.csr_matrix,
    *,
    linear_name: str,
) -> torch.Tensor:
    num_cols = int(affinity.shape[0])
    if num_cols <= 1:
        return torch.arange(num_cols, dtype=torch.long)
    if affinity.nnz == 0:
        return torch.arange(num_cols, dtype=torch.long)
    if num_cols == 2:
        return torch.arange(num_cols, dtype=torch.long)
    lap = sparse_laplacian(affinity, normed=True)
    if num_cols <= 32:
        eigvals, eigvecs = np.linalg.eigh(lap.toarray())
        if eigvecs.shape[1] < 2:
            raise RuntimeError(f"{linear_name}: spectral sort failed because laplacian eigvec count < 2.")
        fiedler = eigvecs[:, 1]
    else:
        eigvals, eigvecs = eigsh(lap, k=2, which="SM")
        order = np.argsort(eigvals)
        eigvecs = eigvecs[:, order]
        fiedler = eigvecs[:, 1]
    return torch.from_numpy(np.argsort(fiedler, kind="mergesort").astype(np.int64, copy=False)).contiguous()


def _refine_column_order(
    *,
    part: torch.Tensor,
    order: torch.Tensor,
    codebook_dim: int,
) -> torch.Tensor:
    num_cols = int(order.numel())
    if num_cols <= 1:
        return order
    current_order = order.detach().to(device=part.device, dtype=torch.long).contiguous()
    reverse_order = torch.flip(current_order, dims=[0]).contiguous()
    current_obj = _sum_block_variance(part.index_select(1, current_order), codebook_dim)
    reverse_obj = _sum_block_variance(part.index_select(1, reverse_order), codebook_dim)
    if reverse_obj < current_obj:
        current_order = reverse_order
        current_obj = reverse_obj
    for _ in range(_SPECTRAL_REFINE_SWEEPS):
        improved = False
        for swap_idx in range(num_cols - 1):
            candidate_order = current_order.clone()
            left = candidate_order[swap_idx].clone()
            candidate_order[swap_idx] = candidate_order[swap_idx + 1]
            candidate_order[swap_idx + 1] = left
            candidate_obj = _sum_block_variance(part.index_select(1, candidate_order), codebook_dim)
            if candidate_obj + 1e-12 < current_obj:
                current_order = candidate_order
                current_obj = candidate_obj
                improved = True
        if not improved:
            break
    return current_order.to(device="cpu", dtype=torch.long).contiguous()


def _build_part_col_sort_permutation(
    *,
    part: torch.Tensor,
    transpose: bool,
    sort_mode: str,
    activation_weight: Optional[torch.Tensor],
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    linear_name: str,
    codebook_dim: int,
) -> Optional[torch.Tensor]:
    mode = normalize_intra_part_sort_mode(sort_mode, arg_name="intra_part_sort_mode")
    if mode == "none":
        return None
    embeddings = part.detach().to(device="cpu", dtype=torch.float32).t().contiguous()
    importance = None
    if mode == "act_spectral_cosine":
        if activation_weight is None:
            raise ValueError(f"{linear_name}: sort_mode=act_spectral_cosine requires activation vector.")
        act = activation_weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if transpose:
            if int(act.numel()) < int(row_end):
                raise ValueError(
                    f"{linear_name}: activation size mismatch for transpose split, "
                    f"got={int(act.numel())}, expected at least rows={int(row_end)}."
                )
            embeddings = embeddings * act[row_start:row_end].view(1, -1)
        else:
            if int(act.numel()) < int(col_end):
                raise ValueError(
                    f"{linear_name}: activation size mismatch for non-transpose split, "
                    f"got={int(act.numel())}, expected at least cols={int(col_end)}."
                )
            importance = act[col_start:col_end].contiguous()
    embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-12)
    affinity = _build_sparse_affinity_from_embeddings(embeddings, importance=importance)
    initial_order = _spectral_order_from_affinity(affinity, linear_name=linear_name)
    return _refine_column_order(part=part, order=initial_order, codebook_dim=codebook_dim)


def split_linear_into_parts_with_sort(
    weight: torch.Tensor,
    transpose: bool,
    intra_parallel: Union[int, Sequence[int]],
    *,
    sort_mode: Union[str, Sequence[str]] = "none",
    activation_weight: Optional[torch.Tensor] = None,
    linear_name: str = "<unknown>",
    codebook_dim: Optional[int] = None,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
]:
    """
    Split a linear weight into chunks after optional transpose.
    Supports:
      - intra_parallel=int(n): split rows into n parts (legacy behavior)
      - intra_parallel=(row_parts, col_parts): split rows/cols into 2D parts
    Returns:
      - flat parts [row_parts*col_parts, -1]
      - part_restore_col_indices [row_parts*col_parts, cols_per_part], optional
    """
    row_parts, col_parts = resolve_intra_parallel(intra_parallel)
    w = weight.detach().float()
    if transpose:
        w = w.t()
    if w.shape[0] % row_parts != 0:
        raise ValueError(
            f"weight dim0={w.shape[0]} not divisible by row_parts={row_parts} (transpose={transpose})"
        )
    if w.shape[1] % col_parts != 0:
        raise ValueError(
            f"weight dim1={w.shape[1]} not divisible by col_parts={col_parts} (transpose={transpose})"
        )

    resolved_sort_mode = normalize_intra_part_sort_mode(
        sort_mode,
        arg_name="intra_part_sort_mode",
    )
    if resolved_sort_mode != "none" and codebook_dim is None:
        raise ValueError(f"{linear_name}: codebook_dim is required when intra_part_sort_mode={resolved_sort_mode}.")
    if resolved_sort_mode == "act_spectral_cosine":
        if activation_weight is None:
            raise ValueError(f"{linear_name}: sort_mode=act_spectral_cosine requires activation vector.")
        act = activation_weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
        expected_act = int(w.shape[0] if transpose else w.shape[1])
        if int(act.numel()) != expected_act:
            raise ValueError(
                f"{linear_name}: activation size mismatch for intra_part_sort_mode=act_spectral_cosine, "
                f"got={int(act.numel())}, expected={expected_act}."
            )
    rows_per_part = w.shape[0] // row_parts
    cols_per_part = w.shape[1] // col_parts
    parts: List[torch.Tensor] = []
    part_restore_list: List[torch.Tensor] = []
    for row_idx in range(row_parts):
        row_start = row_idx * rows_per_part
        row_end = row_start + rows_per_part
        for col_idx in range(col_parts):
            col_start = col_idx * cols_per_part
            col_end = col_start + cols_per_part
            part = w[row_start:row_end, col_start:col_end].contiguous()
            part_sorted_col_indices = _build_part_col_sort_permutation(
                part=part,
                transpose=transpose,
                sort_mode=resolved_sort_mode,
                activation_weight=activation_weight,
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                linear_name=linear_name,
                codebook_dim=int(codebook_dim) if codebook_dim is not None else 0,
            )
            if part_sorted_col_indices is not None:
                part = part.index_select(1, part_sorted_col_indices.to(device=part.device, dtype=torch.long))
                part_restore = _build_restore_indices(part_sorted_col_indices)
                if part_restore is None:
                    raise RuntimeError(f"{linear_name}: failed to build part restore indices.")
                part_restore_list.append(part_restore.to(device="cpu", dtype=torch.long).contiguous())
            parts.append(part.contiguous().view(-1))
    stacked_parts = torch.stack(parts, dim=0)
    if not part_restore_list:
        return stacked_parts, None
    if len(part_restore_list) != int(stacked_parts.shape[0]):
        raise RuntimeError(
            f"{linear_name}: part restore count mismatch: {len(part_restore_list)} vs {int(stacked_parts.shape[0])}."
        )
    return stacked_parts, torch.stack(part_restore_list, dim=0)


def split_linear_into_parts(
    weight: torch.Tensor,
    transpose: bool,
    intra_parallel: Union[int, Sequence[int]],
) -> torch.Tensor:
    """
    Split a linear weight into parts after optional transpose.
    Returns shape [parts, -1].
    """
    parts, _part_restore_cols = split_linear_into_parts_with_sort(
        weight,
        transpose,
        intra_parallel,
        sort_mode="none",
        codebook_dim=None,
    )
    return parts


def _prepare_linear_weight_for_outlier_protection(
    *,
    weight: torch.Tensor,
    linear_name: str,
    activation_weight: Optional[torch.Tensor],
    outlier_protect_count: int,
    outlier_protect_axis: str,
) -> PreparedLinearWeight:
    axis = str(outlier_protect_axis).strip().lower()
    if axis not in {"input", "output"}:
        raise ValueError(f"Unsupported outlier_protect_axis={outlier_protect_axis}. Expected input or output.")
    original_in_features = int(weight.shape[1])
    original_out_features = int(weight.shape[0])
    act_cpu = None
    if activation_weight is not None:
        act_cpu = activation_weight.detach().to(dtype=torch.float32, device="cpu").contiguous()
        if int(act_cpu.numel()) != original_in_features:
            raise ValueError(
                f"Activation vector size mismatch for {linear_name}: "
                f"got {int(act_cpu.numel())}, expected in_features={original_in_features}"
            )

    protect_count = int(outlier_protect_count)
    if protect_count < 0:
        raise ValueError(f"{linear_name}: outlier_protect_count must be >= 0, got {protect_count}.")
    if protect_count == 0:
        return PreparedLinearWeight(
            split_weight=weight,
            compressed_in_features=original_in_features,
            compressed_out_features=original_out_features,
            activation_weight=act_cpu,
            protected_input_indices=None,
            protected_input_weight=None,
            protected_output_indices=None,
            protected_output_weight=None,
        )

    if act_cpu is None:
        raise ValueError(
            f"{linear_name}: outlier_protect_count={protect_count} requires activation vector."
        )

    if axis == "input" and protect_count >= original_in_features:
        raise ValueError(
            f"{linear_name}: outlier_protect_count={protect_count} "
            f"protects too many input channels, "
            f"which is not smaller than in_features={original_in_features}."
        )
    if axis == "output" and protect_count >= original_out_features:
        raise ValueError(
            f"{linear_name}: outlier_protect_count={protect_count} "
            f"protects too many output channels, "
            f"which is not smaller than out_features={original_out_features}."
        )

    weight_device = weight.device
    act_dev = act_cpu.to(device=weight_device, dtype=torch.float32, non_blocking=True)
    weight_f = weight.detach().to(device=weight_device, dtype=torch.float32).contiguous()
    if axis == "input":
        weight_in_major = weight_f.t().contiguous()  # [in, out]
        weighted_rows = weight_in_major * act_dev.view(-1, 1)
        channel_norm = torch.norm(weighted_rows, p=2, dim=1)
        protected_idx = torch.topk(channel_norm, k=protect_count, largest=True).indices
        protected_idx = torch.sort(protected_idx).values.contiguous()

        keep_mask = torch.ones(original_in_features, dtype=torch.bool, device=weight_device)
        keep_mask[protected_idx] = False
        compressed_idx = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
        if int(compressed_idx.numel()) < 1:
            raise ValueError(f"{linear_name}: no input channels remain after outlier protection.")

        protected_weight = weight_in_major.index_select(0, protected_idx)
        compressed_weight = weight.index_select(1, compressed_idx).contiguous()
        compressed_act = act_dev.index_select(0, compressed_idx)

        return PreparedLinearWeight(
            split_weight=compressed_weight,
            compressed_in_features=int(compressed_idx.numel()),
            compressed_out_features=original_out_features,
            activation_weight=compressed_act.to(device="cpu", dtype=torch.float32).contiguous(),
            protected_input_indices=protected_idx.to(device="cpu", dtype=torch.long).contiguous(),
            protected_input_weight=protected_weight.to(device="cpu", dtype=weight.dtype).contiguous(),
            protected_output_indices=None,
            protected_output_weight=None,
        )

    weighted_weight = weight_f * act_dev.view(1, -1)
    channel_norm = torch.norm(weighted_weight, p=2, dim=1)
    protected_idx = torch.topk(channel_norm, k=protect_count, largest=True).indices
    protected_idx = torch.sort(protected_idx).values.contiguous()

    keep_mask = torch.ones(original_out_features, dtype=torch.bool, device=weight_device)
    keep_mask[protected_idx] = False
    compressed_idx = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
    if int(compressed_idx.numel()) < 1:
        raise ValueError(f"{linear_name}: no output channels remain after outlier protection.")

    protected_weight = weight.index_select(0, protected_idx).contiguous()
    compressed_weight = weight.index_select(0, compressed_idx).contiguous()

    return PreparedLinearWeight(
        split_weight=compressed_weight,
        compressed_in_features=original_in_features,
        compressed_out_features=int(compressed_idx.numel()),
        activation_weight=act_cpu,
        protected_input_indices=None,
        protected_input_weight=None,
        protected_output_indices=protected_idx.to(device="cpu", dtype=torch.long).contiguous(),
        protected_output_weight=protected_weight.to(device="cpu", dtype=weight.dtype).contiguous(),
    )


def _build_wa_mse_part_metas(
    group_refs: Sequence[LinearPrepRef],
    intra_parallel: Union[int, Sequence[int]],
    activation_weight_by_linear: Dict[str, torch.Tensor],
    train_device: str,
    part_restore_col_indices_by_linear: Dict[str, Optional[torch.Tensor]],
) -> List[WAMSEPartMeta]:
    row_parts, col_parts = resolve_intra_parallel(intra_parallel)
    metas: List[WAMSEPartMeta] = []
    for r in group_refs:
        if r.name not in activation_weight_by_linear:
            raise KeyError(
                f"Missing activation vector for linear '{r.name}'. "
                "Please regenerate activation_weight file for this base model."
            )
        act_vec = activation_weight_by_linear[r.name]
        if int(act_vec.numel()) != int(r.in_features):
            raise ValueError(
                f"Activation vector size mismatch for {r.name}: "
                f"got {int(act_vec.numel())}, expected in_features={int(r.in_features)}"
            )
        act_dev = act_vec.to(device=train_device, dtype=torch.float32, non_blocking=True)

        if r.transpose:
            if int(r.in_features) % row_parts != 0:
                raise ValueError(
                    f"{r.name}: in_features={int(r.in_features)} not divisible by row_parts={row_parts}"
                )
            if int(r.out_features) % col_parts != 0:
                raise ValueError(
                    f"{r.name}: out_features={int(r.out_features)} not divisible by col_parts={col_parts}"
                )
            rows_part = int(r.in_features) // row_parts
            cols_part = int(r.out_features) // col_parts
            for row_part_idx in range(row_parts):
                row_offset = row_part_idx * rows_part
                for col_part_idx in range(col_parts):
                    metas.append(
                        WAMSEPartMeta(
                            linear_name=r.name,
                            transpose=True,
                            cols=cols_part,
                            row_offset=row_offset,
                            rows_part=rows_part,
                            col_offset=col_part_idx * cols_part,
                            act_max=act_dev,
                            col_index_map=None,
                        )
                    )
        else:
            if int(r.out_features) % row_parts != 0:
                raise ValueError(
                    f"{r.name}: out_features={int(r.out_features)} not divisible by row_parts={row_parts}"
                )
            if int(r.in_features) % col_parts != 0:
                raise ValueError(
                    f"{r.name}: in_features={int(r.in_features)} not divisible by col_parts={col_parts}"
                )
            rows_part = int(r.out_features) // row_parts
            cols_part = int(r.in_features) // col_parts
            part_restore = part_restore_col_indices_by_linear.get(r.name)
            part_restore_dev = None
            if part_restore is not None:
                if tuple(part_restore.shape) != (int(row_parts * col_parts), int(cols_part)):
                    raise ValueError(
                        f"{r.name}: part_restore_col_indices shape mismatch, got={tuple(part_restore.shape)}, "
                        f"expected=({int(row_parts * col_parts)}, {int(cols_part)})."
                    )
                part_restore_dev = part_restore.to(device=train_device, dtype=torch.long, non_blocking=True).contiguous()

            for row_part_idx in range(row_parts):
                for col_part_idx in range(col_parts):
                    part_idx = row_part_idx * col_parts + col_part_idx
                    col_offset = col_part_idx * cols_part
                    col_map = None
                    if part_restore_dev is not None:
                        local_restore = part_restore_dev[part_idx]
                        local_sorted = torch.argsort(local_restore).to(dtype=torch.long).contiguous()
                        col_map = (local_sorted + int(col_offset)).contiguous()
                        col_offset = 0
                    metas.append(
                        WAMSEPartMeta(
                            linear_name=r.name,
                            transpose=False,
                            cols=cols_part,
                            row_offset=row_part_idx * rows_part,
                            rows_part=rows_part,
                            col_offset=col_offset,
                            act_max=act_dev,
                            col_index_map=col_map,
                        )
                    )
    return metas


def gather_wa_mse_act_max_batch(
    block_idx_batch: torch.Tensor,
    part_metas: Sequence[WAMSEPartMeta],
    codebook_dim: int,
    train_device: str,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    block_idx = block_idx_batch.to(device=train_device, dtype=torch.long, non_blocking=True)
    bsz = int(block_idx.shape[0])
    offsets = torch.arange(codebook_dim, device=train_device, dtype=torch.long).unsqueeze(0)
    flat_pos = block_idx.unsqueeze(1) * int(codebook_dim) + offsets  # [B, codebook_dim]

    # Many part metas share identical channel mapping (e.g. 2D split repeats).
    # Cache per-step mapping and gathered act_max slices to reduce repeated indexing.
    channel_idx_cache: Dict[Tuple[object, ...], torch.Tensor] = {}
    part_value_cache: Dict[Tuple[int, Tuple[object, ...]], torch.Tensor] = {}
    part_batches: List[torch.Tensor] = []
    for meta in part_metas:
        if meta.transpose:
            idx_key = ("t_off", int(meta.cols), int(meta.row_offset))
            channel_idx = channel_idx_cache.get(idx_key)
            if channel_idx is None:
                row_idx = torch.div(flat_pos, int(meta.cols), rounding_mode="floor")
                channel_idx = row_idx + int(meta.row_offset)
                channel_idx_cache[idx_key] = channel_idx
        else:
            if meta.col_index_map is not None:
                col_map = meta.col_index_map
                idx_key = ("n_map", int(meta.cols), int(col_map.data_ptr()))
                channel_idx = channel_idx_cache.get(idx_key)
                if channel_idx is None:
                    col_idx = torch.remainder(flat_pos, int(meta.cols))
                    channel_idx = col_map.index_select(0, col_idx.reshape(-1)).view(bsz, int(codebook_dim))
                    channel_idx_cache[idx_key] = channel_idx
            else:
                idx_key = ("n_off", int(meta.cols), int(meta.col_offset))
                channel_idx = channel_idx_cache.get(idx_key)
                if channel_idx is None:
                    col_idx = torch.remainder(flat_pos, int(meta.cols))
                    channel_idx = col_idx + int(meta.col_offset)
                    channel_idx_cache[idx_key] = channel_idx
        act_ptr = int(meta.act_max.data_ptr())
        part_key = (act_ptr, idx_key)
        part = part_value_cache.get(part_key)
        if part is None:
            part = meta.act_max.index_select(0, channel_idx.reshape(-1)).view(bsz, int(codebook_dim))
            part_value_cache[part_key] = part
        part_batches.append(part)
    act_max_batch = torch.stack(part_batches, dim=1)  # [B, P, codebook_dim]
    if act_max_batch.dtype != target_dtype:
        act_max_batch = act_max_batch.to(dtype=target_dtype)
    return act_max_batch


def prepare_group_weight_data(
    *,
    group_refs: Sequence[LinearPrepRef],
    intra_parallel: Union[int, Sequence[int]],
    codebook_dim: int,
    batch_size: int,
    normalize_weight: bool,
    recon_loss_type: str,
    activation_weight_by_linear: Optional[Dict[str, torch.Tensor]],
    train_device: str,
    intra_part_sort_mode: Union[str, Sequence[str]] = "none",
    outlier_protect_count: int = 0,
    outlier_protect_axis: str = "input",
) -> GroupDataPrepResult:
    row_parts, col_parts = resolve_intra_parallel(intra_parallel)
    parts_per_linear = int(row_parts) * int(col_parts)
    num_linear = len(group_refs)
    num_models = num_linear * parts_per_linear
    protect_count = int(outlier_protect_count)
    if protect_count < 0:
        raise ValueError(f"outlier_protect_count must be >= 0, got {protect_count}")

    resolved_sort_mode = normalize_intra_part_sort_mode(
        intra_part_sort_mode,
        arg_name="intra_part_sort_mode",
    )
    use_wa_mse = str(recon_loss_type).lower() == "wa_mse"
    requires_act = resolved_sort_mode == "act_spectral_cosine"
    needs_activation = requires_act or use_wa_mse or protect_count > 0
    if needs_activation and activation_weight_by_linear is None:
        raise ValueError(
            "Activation vectors are required by outlier protection, wa_mse, or intra_part_sort_mode=act_spectral_cosine. "
            "No activation source was provided for the current group."
        )

    split_list = []
    split_metas: List[LinearSplitMeta] = []
    part_restore_col_indices_by_linear: Dict[str, Optional[torch.Tensor]] = {}
    wa_mse_group_refs: List[LinearPrepRef] = []
    wa_mse_activation_weight_by_linear: Dict[str, torch.Tensor] = {}
    for r in group_refs:
        act_for_linear = None
        if needs_activation:
            if activation_weight_by_linear is None or r.name not in activation_weight_by_linear:
                raise KeyError(
                    f"Missing activation vector for linear '{r.name}' required by outlier protection, wa_mse, or intra_part_sort_mode=act_spectral_cosine."
                )
            act_for_linear = activation_weight_by_linear[r.name]
        prepared_weight = _prepare_linear_weight_for_outlier_protection(
            weight=r.weight,
            linear_name=r.name,
            activation_weight=act_for_linear,
            outlier_protect_count=protect_count,
            outlier_protect_axis=outlier_protect_axis,
        )
        act_for_sort = prepared_weight.activation_weight if requires_act else None
        split_parts, part_restore_cols = split_linear_into_parts_with_sort(
            prepared_weight.split_weight,
            r.transpose,
            (row_parts, col_parts),
            sort_mode=resolved_sort_mode,
            activation_weight=act_for_sort,
            linear_name=r.name,
            codebook_dim=int(codebook_dim),
        )
        split_list.append(split_parts.cpu())
        wa_mse_group_refs.append(
            LinearPrepRef(
                name=r.name,
                weight=prepared_weight.split_weight,
                in_features=int(prepared_weight.compressed_in_features),
                out_features=int(prepared_weight.compressed_out_features),
                transpose=bool(r.transpose),
            )
        )
        if prepared_weight.activation_weight is not None:
            wa_mse_activation_weight_by_linear[r.name] = prepared_weight.activation_weight
        part_restore_cols_cpu = (
            part_restore_cols.detach().to(dtype=torch.long, device="cpu").contiguous()
            if part_restore_cols is not None
            else None
        )
        part_restore_col_indices_by_linear[r.name] = part_restore_cols_cpu
        split_metas.append(
            LinearSplitMeta(
                linear_name=r.name,
                transpose=bool(r.transpose),
                sort_mode=resolved_sort_mode,
                parallel_rows=int(row_parts),
                parallel_cols=int(col_parts),
                restore_row_indices=None,
                restore_col_indices=None,
                part_restore_col_indices=part_restore_cols_cpu,
                compressed_in_features=int(prepared_weight.compressed_in_features),
                compressed_out_features=int(prepared_weight.compressed_out_features),
                protected_input_indices=prepared_weight.protected_input_indices,
                protected_input_weight=prepared_weight.protected_input_weight,
                protected_output_indices=prepared_weight.protected_output_indices,
                protected_output_weight=prepared_weight.protected_output_weight,
            )
        )
    per_linear_flat = torch.stack(split_list, dim=0)  # [num_linear, parts_per_linear, N]
    stacked_flat = per_linear_flat.reshape(num_models, -1)  # [num_models, N]

    d_mean = stacked_flat.mean(dim=1, keepdim=True)
    d_std = stacked_flat.std(dim=1, keepdim=True)
    if normalize_weight:
        stacked_flat = (stacked_flat - d_mean) / (d_std + 1e-6)

    numel = stacked_flat.shape[1]
    if numel % int(codebook_dim) != 0:
        raise ValueError(
            f"flatten_len={numel} not divisible by codebook_dim={int(codebook_dim)}"
        )

    stacked_data = stacked_flat.view(num_models, -1, int(codebook_dim)).permute(1, 0, 2).contiguous()
    block_indices = torch.arange(stacked_data.shape[0], dtype=torch.long)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(stacked_data, block_indices),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    eval_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(stacked_data, block_indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    part_metas: List[WAMSEPartMeta] = []
    if use_wa_mse:
        if not wa_mse_activation_weight_by_linear:
            raise ValueError("recon_loss_type=wa_mse requires activation vectors for the current group.")
        part_metas = _build_wa_mse_part_metas(
            group_refs=wa_mse_group_refs,
            intra_parallel=(row_parts, col_parts),
            activation_weight_by_linear=wa_mse_activation_weight_by_linear,
            train_device=train_device,
            part_restore_col_indices_by_linear=part_restore_col_indices_by_linear,
        )
        if len(part_metas) != num_models:
            raise RuntimeError(
                f"wa_mse internal mismatch: len(part_metas)={len(part_metas)} vs num_models={num_models}"
            )
        expected_flat_len = int(stacked_data.shape[0]) * int(codebook_dim)
        for meta in part_metas:
            if int(meta.rows_part) * int(meta.cols) != expected_flat_len:
                raise ValueError(
                    f"wa_mse index map mismatch for {meta.linear_name}: "
                    f"rows_part*cols={int(meta.rows_part) * int(meta.cols)} vs expected={expected_flat_len}"
                )

    return GroupDataPrepResult(
        num_models=num_models,
        codebook_dim=int(codebook_dim),
        stacked_data=stacked_data,
        d_mean=d_mean,
        d_std=d_std,
        train_loader=train_loader,
        eval_loader=eval_loader,
        use_wa_mse=use_wa_mse,
        part_metas=part_metas,
        split_metas=split_metas,
    )
