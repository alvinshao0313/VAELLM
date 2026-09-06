from __future__ import annotations

import heapq
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

CHANNEL_ALLOCATION_ARTIFACT_FILENAME = "channel_allocation_artifact.json"


def _resolve_intra_parallel(value: object) -> Tuple[int, int]:
    if isinstance(value, int):
        if int(value) < 1:
            raise ValueError(f"intra_parallel must be >= 1, got {value}")
        return int(value), 1
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"intra_parallel must be a (row_parts, col_parts) pair, got {value!r}.")
    row_parts = int(value[0])
    col_parts = int(value[1])
    if row_parts < 1 or col_parts < 1:
        raise ValueError(f"intra_parallel factors must be >= 1, got {value}.")
    return row_parts, col_parts


def compressed_features_after_protection(
    *,
    in_features: int,
    out_features: int,
    protect_count: int,
    axis: str,
) -> Tuple[int, int]:
    resolved_axis = str(axis).strip().lower()
    count = int(protect_count)
    if count < 0:
        raise ValueError(f"protect_count must be >= 0, got {protect_count}.")
    if resolved_axis == "input":
        if count >= int(in_features):
            raise ValueError(
                f"protect_count={count} is not smaller than in_features={in_features}."
            )
        return int(out_features), int(in_features) - count
    if resolved_axis == "output":
        if count >= int(out_features):
            raise ValueError(
                f"protect_count={count} is not smaller than out_features={out_features}."
            )
        return int(out_features) - count, int(in_features)
    raise ValueError(f"Unsupported channel axis={axis!r}. Expected input or output.")


def vae_layout_rows_cols(
    *,
    compressed_out: int,
    compressed_in: int,
    transpose: bool,
) -> Tuple[int, int]:
    if bool(transpose):
        return int(compressed_in), int(compressed_out)
    return int(compressed_out), int(compressed_in)


def check_compressed_vae_part_legality(
    *,
    compressed_out: int,
    compressed_in: int,
    transpose: bool,
    intra_parallel: object,
    codebook_dim: Optional[int] = None,
    linear_name: str = "<unknown>",
) -> int:
    rows, cols = vae_layout_rows_cols(
        compressed_out=int(compressed_out),
        compressed_in=int(compressed_in),
        transpose=bool(transpose),
    )
    row_parts, col_parts = _resolve_intra_parallel(intra_parallel)
    if rows % row_parts != 0:
        raise ValueError(
            f"{linear_name}: weight dim0={rows} not divisible by row_parts={row_parts} "
            f"(transpose={bool(transpose)})"
        )
    if cols % col_parts != 0:
        raise ValueError(
            f"{linear_name}: weight dim1={cols} not divisible by col_parts={col_parts} "
            f"(transpose={bool(transpose)})"
        )
    n_part = int(rows // row_parts) * int(cols // col_parts)
    if codebook_dim is not None and n_part % int(codebook_dim) != 0:
        raise ValueError(
            f"{linear_name}: flatten_len={n_part} not divisible by codebook_dim={int(codebook_dim)}"
        )
    return n_part


def check_vae_part_legality(
    *,
    in_features: int,
    out_features: int,
    protect_count: int,
    axis: str,
    transpose: bool,
    intra_parallel: object,
    codebook_dim: int,
    linear_name: str = "<unknown>",
) -> int:
    compressed_out, compressed_in = compressed_features_after_protection(
        in_features=int(in_features),
        out_features=int(out_features),
        protect_count=int(protect_count),
        axis=axis,
    )
    return check_compressed_vae_part_legality(
        compressed_out=compressed_out,
        compressed_in=compressed_in,
        transpose=bool(transpose),
        intra_parallel=intra_parallel,
        codebook_dim=int(codebook_dim),
        linear_name=linear_name,
    )


def is_legal_protect_count(
    *,
    in_features: int,
    out_features: int,
    protect_count: int,
    axis: str,
    transpose: bool,
    intra_parallel: object,
    codebook_dim: int,
    linear_name: str = "<unknown>",
) -> bool:
    try:
        check_vae_part_legality(
            in_features=in_features,
            out_features=out_features,
            protect_count=protect_count,
            axis=axis,
            transpose=transpose,
            intra_parallel=intra_parallel,
            codebook_dim=codebook_dim,
            linear_name=linear_name,
        )
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class ChannelLinearSpec:
    name: str
    in_features: int
    out_features: int
    codebook_dim: int
    transpose: bool
    intra_parallel: Tuple[int, int]
    ref_position: int
    scores: Optional[torch.Tensor] = None
    axis: str = "input"
    category: Optional[str] = None


def legal_protect_counts(spec: ChannelLinearSpec) -> Tuple[int, ...]:
    axis_features = int(spec.in_features) if str(spec.axis) == "input" else int(spec.out_features)
    codebook_dim = int(spec.codebook_dim)
    if codebook_dim < 1:
        raise ValueError(f"{spec.name}: codebook_dim must be >= 1, got {spec.codebook_dim}.")
    legal: List[int] = []
    count = 0
    while count < axis_features:
        if is_legal_protect_count(
            in_features=spec.in_features,
            out_features=spec.out_features,
            protect_count=count,
            axis=spec.axis,
            transpose=spec.transpose,
            intra_parallel=spec.intra_parallel,
            codebook_dim=spec.codebook_dim,
            linear_name=spec.name,
        ):
            legal.append(int(count))
        count = codebook_dim if count == 0 else count + codebook_dim
    return tuple(legal)


def stack_compatible_signature(spec: ChannelLinearSpec, protect_count: int) -> Tuple[int, int, bool, int, int, int, int]:
    compressed_out, compressed_in = compressed_features_after_protection(
        in_features=spec.in_features,
        out_features=spec.out_features,
        protect_count=int(protect_count),
        axis=spec.axis,
    )
    n_part = check_compressed_vae_part_legality(
        compressed_out=compressed_out,
        compressed_in=compressed_in,
        transpose=spec.transpose,
        intra_parallel=spec.intra_parallel,
        codebook_dim=spec.codebook_dim,
        linear_name=spec.name,
    )
    row_parts, col_parts = _resolve_intra_parallel(spec.intra_parallel)
    return (
        int(compressed_out),
        int(compressed_in),
        bool(spec.transpose),
        int(row_parts),
        int(col_parts),
        int(n_part),
        int(spec.codebook_dim),
    )


def select_channel_indices(scores: torch.Tensor, count: int) -> torch.Tensor:
    protect_count = int(count)
    if protect_count < 0:
        raise ValueError(f"protect_count must be >= 0, got {count}.")
    scores_cpu = scores.detach().to(device="cpu", dtype=torch.float32).reshape(-1).contiguous()
    if protect_count == 0:
        return torch.empty(0, dtype=torch.long)
    if protect_count > int(scores_cpu.numel()):
        raise ValueError(
            f"protect_count={protect_count} exceeds score channels={int(scores_cpu.numel())}."
        )
    order = torch.argsort(-scores_cpu, stable=True)
    return torch.sort(order[:protect_count].to(dtype=torch.long)).values.contiguous()


def _prefix_utility(scores: torch.Tensor, count: int) -> float:
    protect_count = int(count)
    if protect_count <= 0:
        return 0.0
    scores_cpu = scores.detach().to(device="cpu", dtype=torch.float32).reshape(-1).contiguous()
    order = torch.argsort(-scores_cpu, stable=True)
    return float(scores_cpu.index_select(0, order[:protect_count]).sum().item())


def _next_legal_count(legal_counts: Sequence[int], current: int) -> Optional[int]:
    for count in legal_counts:
        if int(count) > int(current):
            return int(count)
    return None


def allocate_codebook_aligned_counts(
    specs: Sequence[ChannelLinearSpec],
    *,
    raw_budget: int,
    min_per_layer: int,
) -> Dict[str, int]:
    budget = int(raw_budget)
    floor = int(min_per_layer)
    if budget < 0:
        raise ValueError(f"raw_budget must be >= 0, got {raw_budget}.")
    if floor < 0:
        raise ValueError(f"channel_min_per_layer must be >= 0, got {min_per_layer}.")
    if not specs:
        return {}

    legal_by_name = {spec.name: legal_protect_counts(spec) for spec in specs}
    for spec in specs:
        if spec.scores is None:
            raise ValueError(f"{spec.name}: channel scores are required for adaptive allocation.")
        if 0 not in legal_by_name[spec.name]:
            raise ValueError(
                f"{spec.name}: K=0 is not a legal VAE part count under the current "
                f"intra_parallel/codebook_dim constraints."
            )
        if floor > 0 and floor not in legal_by_name[spec.name]:
            raise ValueError(
                f"{spec.name}: channel_min_per_layer={floor} is not a legal protect count. "
                f"Legal counts: {legal_by_name[spec.name]}."
            )

    if floor > 0:
        floor_sum = int(floor) * int(len(specs))
        if floor_sum > budget:
            raise ValueError(
                f"channel_min_per_layer floor sum {floor_sum} exceeds raw budget {budget}."
            )
        current = {spec.name: int(floor) for spec in specs}
        remaining = budget - floor_sum
    else:
        current = {spec.name: 0 for spec in specs}
        remaining = budget

    utilities: Dict[str, Dict[int, float]] = {}
    for spec in specs:
        utilities[spec.name] = {
            int(count): _prefix_utility(spec.scores, int(count))
            for count in legal_by_name[spec.name]
        }

    heap: List[Tuple[float, float, int, int, str]] = []

    def push(spec: ChannelLinearSpec) -> None:
        nxt = _next_legal_count(legal_by_name[spec.name], current[spec.name])
        if nxt is None:
            return
        prev = current[spec.name]
        cost = int(nxt) - int(prev)
        marginal = float(utilities[spec.name][nxt] - utilities[spec.name][prev])
        density = marginal / float(cost)
        heapq.heappush(
            heap,
            (-float(density), -float(marginal), int(spec.ref_position), int(nxt), spec.name),
        )

    spec_by_name = {spec.name: spec for spec in specs}
    for spec in specs:
        push(spec)

    while heap and remaining > 0:
        _neg_density, _neg_marginal, _ref_position, k_next, name = heapq.heappop(heap)
        prev = current[name]
        expected = _next_legal_count(legal_by_name[name], prev)
        if expected != int(k_next):
            continue
        cost = int(k_next) - int(prev)
        if cost > remaining:
            continue
        current[name] = int(k_next)
        remaining -= cost
        push(spec_by_name[name])
    return current


def group_adaptive_inventory(
    specs: Sequence[ChannelLinearSpec],
    counts: Dict[str, int],
    linear_group_size: int,
) -> List[List[str]]:
    group_size = int(linear_group_size)
    if group_size < 1:
        raise ValueError(f"linear_group_size must be >= 1, got {linear_group_size}.")
    buckets: Dict[Tuple[int, int, bool, int, int, int, int], List[str]] = {}
    signature_order: List[Tuple[int, int, bool, int, int, int, int]] = []
    for spec in specs:
        if spec.name not in counts:
            raise KeyError(f"missing allocated protect count for {spec.name}.")
        signature = stack_compatible_signature(spec, int(counts[spec.name]))
        if signature not in buckets:
            buckets[signature] = []
            signature_order.append(signature)
        buckets[signature].append(spec.name)

    assigned = [name for names in buckets.values() for name in names]
    if sorted(assigned) != sorted(spec.name for spec in specs):
        raise ValueError("adaptive grouping must place every planner target into exactly one group.")
    groups: List[List[str]] = []
    for signature in signature_order:
        names = buckets[signature]
        for start in range(0, len(names), group_size):
            groups.append(list(names[start : start + group_size]))
    return groups


def group_layer_scope_inventory(
    names: Sequence[str],
    *,
    linear_group_size: int,
    allow_tail_group: bool,
) -> List[List[str]]:
    group_size = int(linear_group_size)
    if group_size < 1:
        raise ValueError(f"linear_group_size must be >= 1, got {linear_group_size}.")
    planned = list(names)
    if not bool(allow_tail_group):
        planned = planned[: (len(planned) // group_size) * group_size]
    return [planned[start : start + group_size] for start in range(0, len(planned), group_size)]


def layer_scope_group_seed_offsets(
    names: Sequence[str],
    *,
    linear_group_size: int,
    allow_tail_group: bool,
) -> List[int]:
    groups = group_layer_scope_inventory(
        names,
        linear_group_size=int(linear_group_size),
        allow_tail_group=bool(allow_tail_group),
    )
    return [int(index) * int(linear_group_size) for index in range(len(groups))]


def adaptive_scope_group_seed_offsets(num_groups: int) -> List[int]:
    count = int(num_groups)
    if count < 0:
        raise ValueError(f"num_groups must be >= 0, got {num_groups}.")
    return list(range(count))


def vae_group_shuffle_seed(base_seed: int, category_idx: int, group_seed_offset: int) -> int:
    return int(base_seed) + int(category_idx) * 100000 + int(group_seed_offset)


def validate_global_channel_runtime(
    *,
    channel_scope: str,
    channel_protect_mode: str,
    channel_axis: str,
) -> None:
    scope = str(channel_scope).strip().lower()
    if scope != "global":
        return
    mode = str(channel_protect_mode).strip().lower()
    axis = str(channel_axis).strip().lower()
    if mode != "channel" or axis != "input":
        raise ValueError(
            "channel_scope=global requires channel_protect_mode=channel and channel_axis=input, "
            f"got mode={mode!r} axis={axis!r}."
        )


def validate_adaptive_channel_tail_policy(
    *,
    scope: str,
    budget: int,
    allow_tail_group: bool,
) -> None:
    resolved_scope = str(scope).strip().lower()
    if resolved_scope in {"category", "global"} and int(budget) > 0 and not bool(allow_tail_group):
        raise ValueError(
            "channel_scope=category/global with a positive protection budget requires "
            "allow_tail_group=true; refusing to drop signature-bucket tails after allocation."
        )


def category_raw_budget(specs: Sequence[ChannelLinearSpec], protect_count: int) -> int:
    max_legal_total = 0
    for spec in specs:
        legal = legal_protect_counts(spec)
        if legal:
            max_legal_total += int(legal[-1])
    return min(int(protect_count) * int(len(specs)), int(max_legal_total))


def global_raw_budget(specs: Sequence[ChannelLinearSpec], ratio: float) -> int:
    total_input = sum(int(spec.in_features) for spec in specs)
    return int(float(ratio) * float(total_input))


def build_channel_allocation_artifact(
    specs: Sequence[ChannelLinearSpec],
    counts: Dict[str, int],
    groups: Sequence[Sequence[str]],
    *,
    raw_budget: int,
    metric: str,
    axis: str,
) -> Dict[str, object]:
    used = 0
    per_linear: Dict[str, Dict[str, object]] = {}
    group_id_by_name: Dict[str, int] = {}
    for group_id, group in enumerate(groups):
        for name in group:
            if name in group_id_by_name:
                raise ValueError(f"{name} was assigned to multiple VAE groups.")
            group_id_by_name[name] = int(group_id)
    if sorted(group_id_by_name) != sorted(spec.name for spec in specs):
        raise ValueError("allocation artifact requires every planner target in exactly one VAE group.")

    total_input = 0
    total_params = 0
    protected_params = 0
    for spec in specs:
        count = int(counts[spec.name])
        used += count
        signature = stack_compatible_signature(spec, count)
        compressed_out, compressed_in, transpose, row_parts, col_parts, n_part, codebook_dim = signature
        indices = (
            select_channel_indices(spec.scores, count)
            if spec.scores is not None
            else torch.empty(0, dtype=torch.long)
        )
        per_linear[spec.name] = {
            "count": count,
            "indices": indices.tolist(),
            "codebook_dim": int(codebook_dim),
            "compressed_out": int(compressed_out),
            "compressed_in": int(compressed_in),
            "intra_parallel": (int(row_parts), int(col_parts)),
            "n_part": int(n_part),
            "group_signature": signature,
            "group_id": int(group_id_by_name[spec.name]),
            "transpose": bool(transpose),
        }
        total_input += int(spec.in_features)
        total_params += int(spec.in_features) * int(spec.out_features)
        if str(spec.axis) == "input":
            protected_params += int(count) * int(spec.out_features)
        else:
            protected_params += int(count) * int(spec.in_features)
    return {
        "raw_budget": int(raw_budget),
        "used_channels": int(used),
        "score_metric": str(metric),
        "axis": str(axis),
        "achieved_channel_ratio": (float(used) / float(total_input)) if total_input else 0.0,
        "achieved_parameter_ratio": (float(protected_params) / float(total_params)) if total_params else 0.0,
        "per_linear": per_linear,
    }


def _jsonable_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported adaptive plan value type: {type(value)!r}.")


@dataclass(frozen=True)
class AdaptiveChannelPlan:
    scope: str
    axis: str
    score_metric: str
    raw_budget: int
    used_channels: int
    counts: Dict[str, int]
    selected_indices: Dict[str, List[int]]
    groups: List[List[str]]
    signatures: List[Tuple[int, int, bool, int, int, int, int]]
    group_seed_offsets: List[int]
    artifact: Dict[str, object]
    groups_by_category: Dict[str, List[List[str]]]
    signatures_by_category: Dict[str, List[Tuple[int, int, bool, int, int, int, int]]]
    group_seed_offsets_by_category: Dict[str, List[int]]


def serialize_adaptive_channel_plan(plan: AdaptiveChannelPlan) -> Dict[str, object]:
    return {
        "format": "vaellm_adaptive_channel_plan",
        "version": 1,
        "scope": str(plan.scope),
        "axis": str(plan.axis),
        "score_metric": str(plan.score_metric),
        "raw_budget": int(plan.raw_budget),
        "used_channels": int(plan.used_channels),
        "counts": {str(name): int(count) for name, count in plan.counts.items()},
        "selected_indices": {
            str(name): [int(index) for index in indices]
            for name, indices in plan.selected_indices.items()
        },
        "groups": [[str(name) for name in group] for group in plan.groups],
        "signatures": [_jsonable_value(signature) for signature in plan.signatures],
        "group_seed_offsets": [int(offset) for offset in plan.group_seed_offsets],
        "artifact": _jsonable_value(plan.artifact),
        "groups_by_category": {
            str(category): [[str(name) for name in group] for group in groups]
            for category, groups in plan.groups_by_category.items()
        },
        "signatures_by_category": {
            str(category): [_jsonable_value(signature) for signature in signatures]
            for category, signatures in plan.signatures_by_category.items()
        },
        "group_seed_offsets_by_category": {
            str(category): [int(offset) for offset in offsets]
            for category, offsets in plan.group_seed_offsets_by_category.items()
        },
    }


def deserialize_adaptive_channel_plan(payload: Dict[str, object]) -> AdaptiveChannelPlan:
    if not isinstance(payload, dict):
        raise TypeError(f"adaptive channel plan payload must be a dict, got {type(payload)}.")
    if payload.get("format") != "vaellm_adaptive_channel_plan" or int(payload.get("version", 0)) != 1:
        raise ValueError("Received invalid adaptive channel plan format/version.")

    def _signature(raw: object) -> Tuple[int, int, bool, int, int, int, int]:
        values = tuple(raw)
        return (
            int(values[0]),
            int(values[1]),
            bool(values[2]),
            int(values[3]),
            int(values[4]),
            int(values[5]),
            int(values[6]),
        )

    groups = [[str(name) for name in group] for group in payload["groups"]]
    signatures = [_signature(item) for item in payload["signatures"]]
    groups_by_category = {
        str(category): [[str(name) for name in group] for group in groups]
        for category, groups in dict(payload["groups_by_category"]).items()
    }
    signatures_by_category = {
        str(category): [_signature(item) for item in signatures]
        for category, signatures in dict(payload["signatures_by_category"]).items()
    }
    return AdaptiveChannelPlan(
        scope=str(payload["scope"]),
        axis=str(payload["axis"]),
        score_metric=str(payload["score_metric"]),
        raw_budget=int(payload["raw_budget"]),
        used_channels=int(payload["used_channels"]),
        counts={str(name): int(count) for name, count in dict(payload["counts"]).items()},
        selected_indices={
            str(name): [int(index) for index in indices]
            for name, indices in dict(payload["selected_indices"]).items()
        },
        groups=groups,
        signatures=signatures,
        group_seed_offsets=[int(offset) for offset in payload["group_seed_offsets"]],
        artifact=dict(payload["artifact"]),
        groups_by_category=groups_by_category,
        signatures_by_category=signatures_by_category,
        group_seed_offsets_by_category={
            str(category): [int(offset) for offset in offsets]
            for category, offsets in dict(payload["group_seed_offsets_by_category"]).items()
        },
    )


def persist_channel_allocation_artifact(
    run_output_dir: str,
    artifact: Dict[str, object],
    *,
    scope: str,
    category: Optional[str] = None,
) -> str:
    root = str(run_output_dir).strip()
    if not root:
        raise ValueError("run_output_dir must be a non-empty path.")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, CHANNEL_ALLOCATION_ARTIFACT_FILENAME)
    payload: Dict[str, object] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise TypeError("existing channel allocation artifact must be a JSON object.")
        payload = loaded
    key = "global" if str(scope).strip().lower() == "global" else str(category or scope)
    plans = payload.get("plans")
    if not isinstance(plans, dict):
        plans = {}
    plans[key] = _jsonable_value(artifact)
    payload["scope"] = str(scope)
    payload["plans"] = plans
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def _group_signatures_for_names(
    specs: Sequence[ChannelLinearSpec],
    counts: Dict[str, int],
    groups: Sequence[Sequence[str]],
) -> List[Tuple[int, int, bool, int, int, int, int]]:
    spec_by_name = {spec.name: spec for spec in specs}
    signatures: List[Tuple[int, int, bool, int, int, int, int]] = []
    for group in groups:
        if not group:
            raise ValueError("adaptive channel group must not be empty.")
        first = spec_by_name[str(group[0])]
        signatures.append(stack_compatible_signature(first, int(counts[first.name])))
    return signatures


def build_adaptive_channel_plan(
    specs: Sequence[ChannelLinearSpec],
    *,
    raw_budget: int,
    min_per_layer: int,
    linear_group_size: int,
    metric: str,
    axis: str,
    scope: str,
    category: Optional[str] = None,
    group_by_category: bool = False,
    activation_view_fn: Callable[[Sequence[ChannelLinearSpec]], Tuple[object, object]],
    score_fn: Callable[[Sequence[ChannelLinearSpec], object, object], Sequence[ChannelLinearSpec]],
) -> AdaptiveChannelPlan:
    validate_global_channel_runtime(
        channel_scope=scope,
        channel_protect_mode="channel",
        channel_axis=axis,
    )
    act_weight, act_mean = activation_view_fn(specs)
    scored = list(score_fn(specs, act_weight, act_mean))
    counts = allocate_codebook_aligned_counts(
        scored,
        raw_budget=int(raw_budget),
        min_per_layer=int(min_per_layer),
    )
    selected_indices = {
        spec.name: select_channel_indices(spec.scores, int(counts[spec.name])).tolist()
        for spec in scored
    }
    groups_by_category: Dict[str, List[List[str]]] = {}
    signatures_by_category: Dict[str, List[Tuple[int, int, bool, int, int, int, int]]] = {}
    if bool(group_by_category):
        seen_categories: List[str] = []
        for spec in scored:
            cat = str(spec.category or "")
            if cat not in groups_by_category:
                groups_by_category[cat] = []
                seen_categories.append(cat)
        for cat in seen_categories:
            cat_specs = [spec for spec in scored if str(spec.category or "") == cat]
            cat_counts = {spec.name: int(counts[spec.name]) for spec in cat_specs}
            cat_groups = group_adaptive_inventory(cat_specs, cat_counts, int(linear_group_size))
            groups_by_category[cat] = cat_groups
            signatures_by_category[cat] = _group_signatures_for_names(cat_specs, cat_counts, cat_groups)
        groups = [group for cat in seen_categories for group in groups_by_category[cat]]
        signatures = [sig for cat in seen_categories for sig in signatures_by_category[cat]]
    else:
        groups = group_adaptive_inventory(scored, counts, int(linear_group_size))
        signatures = _group_signatures_for_names(scored, counts, groups)
        key = str(category or scope)
        groups_by_category = {key: groups}
        signatures_by_category = {key: signatures}
    group_seed_offsets = adaptive_scope_group_seed_offsets(len(groups))
    group_seed_offsets_by_category = {
        cat: adaptive_scope_group_seed_offsets(len(cat_groups))
        for cat, cat_groups in groups_by_category.items()
    }
    artifact = build_channel_allocation_artifact(
        scored,
        counts,
        [name for groups in groups_by_category.values() for name in groups],
        raw_budget=int(raw_budget),
        metric=str(metric),
        axis=str(axis),
    )
    artifact = _jsonable_value(artifact)
    used_channels = int(artifact["used_channels"])
    return AdaptiveChannelPlan(
        scope=str(scope),
        axis=str(axis),
        score_metric=str(metric),
        raw_budget=int(raw_budget),
        used_channels=used_channels,
        counts={name: int(count) for name, count in counts.items()},
        selected_indices=selected_indices,
        groups=groups,
        signatures=signatures,
        group_seed_offsets=group_seed_offsets,
        artifact=artifact,
        groups_by_category=groups_by_category,
        signatures_by_category=signatures_by_category,
        group_seed_offsets_by_category=group_seed_offsets_by_category,
    )


def resolve_adaptive_channel_plan(
    specs: Sequence[ChannelLinearSpec],
    *,
    raw_budget: int,
    min_per_layer: int,
    linear_group_size: int,
    metric: str,
    axis: str,
    scope: str,
    category: Optional[str] = None,
    group_by_category: bool = False,
    is_main: bool,
    world_size: int,
    broadcast_fn: Callable[[Optional[Dict[str, object]]], Dict[str, object]],
    activation_view_fn: Callable[[Sequence[ChannelLinearSpec]], Tuple[object, object]],
    score_fn: Callable[[Sequence[ChannelLinearSpec], object, object], Sequence[ChannelLinearSpec]],
    run_output_dir: Optional[str] = None,
) -> AdaptiveChannelPlan:
    validate_global_channel_runtime(
        channel_scope=scope,
        channel_protect_mode="channel",
        channel_axis=axis,
    )
    payload: Optional[Dict[str, object]] = None
    if bool(is_main):
        plan = build_adaptive_channel_plan(
            specs,
            raw_budget=int(raw_budget),
            min_per_layer=int(min_per_layer),
            linear_group_size=int(linear_group_size),
            metric=str(metric),
            axis=str(axis),
            scope=str(scope),
            category=category,
            group_by_category=bool(group_by_category),
            activation_view_fn=activation_view_fn,
            score_fn=score_fn,
        )
        if run_output_dir:
            persist_channel_allocation_artifact(
                str(run_output_dir),
                plan.artifact,
                scope=str(scope),
                category=category,
            )
        payload = serialize_adaptive_channel_plan(plan)
        if int(world_size) <= 1:
            return plan
    elif int(world_size) <= 1:
        raise ValueError("single-rank adaptive plan requires the main rank to generate the plan.")
    received = broadcast_fn(payload)
    return deserialize_adaptive_channel_plan(received)
