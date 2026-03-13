import argparse
import ast
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tools.cat_train as cat_train_mod
from tools.cat_train import (
    LinearRef,
    _collect_linears,
    _eval_ppl_after_category,
    _extract_layer_idx,
    _format_namespace,
    _split_csv,
    _train_group_vae_and_replace,
)
from train_utils.cat_data_prep import load_activation_weight_dict, resolve_intra_parallel
from train_utils.model_checkpoint_io import (
    _build_run_output_dir,
    save_model_checkpoint,
)
from train_utils.train_args import (
    build_cat_train_parser,
    process_args_from,
    resolve_codebook_int_for_category,
    resolve_intra_parallel_for_category,
    resolve_skip_layer_matches,
)
from train_utils.utils import get_logger, set_seed


log = get_logger("linear_by_schedule")


def _clone_namespace(ns, **overrides):
    out = argparse.Namespace(**vars(ns))
    for key, value in overrides.items():
        setattr(out, key, value)
    return out


def _format_intra_parallel_desc(row_parts: int, col_parts: int) -> str:
    row_parts = int(row_parts)
    col_parts = int(col_parts)
    if col_parts == 1:
        return str(row_parts)
    return f"[{row_parts},{col_parts}]"


def _strip_optional_quotes(token: str) -> str:
    t = token.strip()
    if len(t) >= 2 and ((t[0] == "'" and t[-1] == "'") or (t[0] == '"' and t[-1] == '"')):
        t = t[1:-1].strip()
    return t


def _parse_parallel_schedule(value: str) -> List[List[str]]:
    raw = str(value).strip()
    if not raw:
        raise ValueError("--parallel_schedule 不能为空")

    parsed = None
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        parsed = None

    if isinstance(parsed, (list, tuple)):
        groups: List[List[str]] = []
        for g in parsed:
            if not isinstance(g, (list, tuple)):
                raise ValueError(f"--parallel_schedule 必须是二维列表，收到元素类型: {type(g)}")
            cats = []
            for item in g:
                cat = _strip_optional_quotes(str(item))
                if cat:
                    cats.append(cat)
            if not cats:
                raise ValueError("--parallel_schedule 中存在空分组")
            groups.append(cats)
        if not groups:
            raise ValueError("--parallel_schedule 解析后为空")
        return groups

    # 兼容未加引号写法，例如:
    # [[q_proj, k_proj, v_proj, o_proj],[gate_proj, up_proj, down_proj]]
    inner_groups = re.findall(r"\[([^\[\]]*)\]", raw)
    groups: List[List[str]] = []
    for group_raw in inner_groups:
        cats = []
        for token in group_raw.split(","):
            cat = _strip_optional_quotes(token)
            if cat:
                cats.append(cat)
        if cats:
            groups.append(cats)
    if not groups:
        raise ValueError(
            "无法解析 --parallel_schedule。示例: "
            "[[q_proj,k_proj,v_proj,o_proj],[gate_proj,up_proj,down_proj]]"
        )
    return groups


def _flatten_unique(schedule: Sequence[Sequence[str]]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for group in schedule:
        for cat in group:
            if cat not in seen:
                seen.add(cat)
                out.append(cat)
    return out


def _validate_schedule_categories(
    parallel_schedule: Sequence[Sequence[str]],
    discovered_categories: Sequence[str],
) -> List[str]:
    flat = _flatten_unique(parallel_schedule)
    duplicated = []
    seen: Set[str] = set()
    for cat in [c for g in parallel_schedule for c in g]:
        if cat in seen and cat not in duplicated:
            duplicated.append(cat)
        seen.add(cat)
    if duplicated:
        raise ValueError(
            "--parallel_schedule 中类别重复出现: " + ",".join(sorted(duplicated))
        )

    discovered_set = set(discovered_categories)
    flat_set = set(flat)
    missing = sorted(discovered_set - flat_set)
    extra = sorted(flat_set - discovered_set)
    if missing or extra:
        raise ValueError(
            "parallel_schedule 与模型可训练类别不一致。"
            f" missing_in_schedule={missing}, unknown_in_schedule={extra}, "
            f" discovered={sorted(discovered_set)}, schedule={flat}"
        )
    return flat


def _format_group_shapes(group_refs: Sequence[LinearRef]) -> str:
    lines: List[str] = []
    for r in group_refs:
        w = r.module.weight
        raw_shape = tuple(int(x) for x in w.shape)
        effective_shape = (int(raw_shape[1]), int(raw_shape[0])) if r.transpose else raw_shape
        lines.append(
            f"{r.name} | category={r.category} | transpose={r.transpose} "
            f"| raw_shape={raw_shape} | effective_shape={effective_shape} | numel={int(w.numel())}"
        )
    return "\n".join(lines)


def _validate_group_parallel_compatibility(
    *,
    group_refs: Sequence[LinearRef],
    group_tag: str,
    intra_parallel,
    codebook_dim: int,
) -> None:
    row_parts, col_parts = resolve_intra_parallel(intra_parallel)
    if not group_refs:
        raise ValueError(f"[{group_tag}] empty group")

    numels: Set[int] = set()
    for r in group_refs:
        w = r.module.weight
        raw_shape = tuple(int(x) for x in w.shape)
        eff_rows = int(raw_shape[1]) if r.transpose else int(raw_shape[0])
        eff_cols = int(raw_shape[0]) if r.transpose else int(raw_shape[1])
        numels.add(int(w.numel()))
        if eff_rows % int(row_parts) != 0:
            details = _format_group_shapes(group_refs)
            raise ValueError(
                f"[{group_tag}] 分组不满足行方向切分要求: "
                f"{r.name} effective_rows={eff_rows} 不能被 row_parts={int(row_parts)} 整除。\n"
                f"组内权重详情:\n{details}"
            )
        if eff_cols % int(col_parts) != 0:
            details = _format_group_shapes(group_refs)
            raise ValueError(
                f"[{group_tag}] 分组不满足列方向切分要求: "
                f"{r.name} effective_cols={eff_cols} 不能被 col_parts={int(col_parts)} 整除。\n"
                f"组内权重详情:\n{details}"
            )
        part_numel = (eff_rows // int(row_parts)) * (eff_cols // int(col_parts))
        if part_numel % int(codebook_dim) != 0:
            details = _format_group_shapes(group_refs)
            raise ValueError(
                f"[{group_tag}] 分组不满足 codebook_dim 切分要求: "
                f"{r.name} part_numel={part_numel} 不能被 codebook_dim={int(codebook_dim)} 整除。\n"
                f"组内权重详情:\n{details}"
            )

    if len(numels) != 1:
        details = _format_group_shapes(group_refs)
        raise ValueError(
            f"[{group_tag}] 并行内优化的权重参数量不一致，无法拼接训练。"
            f" numel_set={sorted(numels)}\n组内权重详情:\n{details}"
        )


def _build_schedule_train_parser() -> argparse.ArgumentParser:
    parser = build_cat_train_parser()
    parser.add_argument(
        "--parallel_schedule",
        type=str,
        default="[[q_proj,k_proj,v_proj,o_proj],[gate_proj,up_proj,down_proj]]",
        help=(
            "层内并行分组规则。格式为二维 list，支持带引号或不带引号写法。"
            "示例: [[q_proj,k_proj,v_proj,o_proj],[gate_proj,up_proj,down_proj]]"
        ),
    )
    return parser


def process_schedule_train_args(argv: Optional[Sequence[str]]):
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_schedule_train_parser()
    script_args, remaining = parser.parse_known_args(list(argv))
    hf_args, training_args, vae_args = process_args_from(remaining)
    return script_args, hf_args, training_args, vae_args


def _collect_refs_for_layer_group(
    *,
    model: nn.Module,
    layer_idx: int,
    categories_in_group: Sequence[str],
    transpose_modules: Sequence[str],
    only_decoder_projections: bool,
    projection_suffixes: Sequence[str],
) -> List[LinearRef]:
    # 每个 stage 动态重采样，避免 LoRA merge/模块替换后引用失效。
    current_linears = _collect_linears(
        model,
        transpose_modules,
        only_decoder_projections=only_decoder_projections,
        projection_suffixes=projection_suffixes,
    )
    grouped: Dict[str, List[LinearRef]] = {}
    wanted = set(categories_in_group)
    for r in current_linears:
        li = _extract_layer_idx(r.name)
        if li != int(layer_idx):
            continue
        if r.category not in wanted:
            continue
        grouped.setdefault(r.category, []).append(r)
    out: List[LinearRef] = []
    for cat in categories_in_group:
        refs = grouped.get(cat, [])
        refs = sorted(refs, key=lambda x: x.name)
        out.extend(refs)
    return out


def _summarize_category_counts(refs: Sequence[LinearRef]) -> str:
    count_by_cat: Dict[str, int] = {}
    for r in refs:
        count_by_cat[r.category] = count_by_cat.get(r.category, 0) + 1
    if not count_by_cat:
        return "none"
    return ",".join(f"{cat}:{count_by_cat[cat]}" for cat in sorted(count_by_cat))


def _collect_remaining_target_refs(
    *,
    model: nn.Module,
    target_categories: Sequence[str],
    transpose_modules: Sequence[str],
    only_decoder_projections: bool,
    projection_suffixes: Sequence[str],
) -> List[LinearRef]:
    remain = _collect_linears(
        model,
        transpose_modules,
        only_decoder_projections=only_decoder_projections,
        projection_suffixes=projection_suffixes,
    )
    target_set = set(target_categories)
    return [r for r in remain if r.category in target_set]


def main(argv: Optional[Sequence[str]] = None) -> None:
    global log
    cat_args, hf_args, training_args, vae_args = process_schedule_train_args(argv)
    set_seed(cat_args.seed)

    os.makedirs(cat_args.output_dir, exist_ok=True)
    run_output_dir = _build_run_output_dir(cat_args.output_dir, vae_args.model_path)
    os.environ["LOG_FILE"] = os.path.join(run_output_dir, "linear_by_schedule.log")
    log = get_logger("linear_by_schedule")
    # 复用 cat_train 内部函数时，统一到同一个 logger，避免 step/eval 日志落到别处。
    cat_train_mod.log = log
    log.info("Bound tools.cat_train logger to %s", log.name)
    cat_args.output_dir = run_output_dir

    log.info("Run output directory: %s", run_output_dir)
    log.info(
        "Args:\nscript=%s\nvae=%s\ntraining=%s",
        _format_namespace(cat_args),
        _format_namespace(vae_args),
        _format_namespace(training_args),
    )

    parallel_schedule = _parse_parallel_schedule(cat_args.parallel_schedule)
    log.info("parallel_schedule=%s", json.dumps(parallel_schedule, ensure_ascii=False))

    log.info("Loading model: %s", vae_args.model_path)
    from rotation.model_utils import get_model

    model = get_model(vae_args.model_path, hf_args.access_token)
    activation_weight_by_linear: Optional[Dict[str, torch.Tensor]] = None
    wa_mse_runtime: Optional[Dict[str, object]] = None
    if str(getattr(vae_args, "recon_loss_type", "")).lower() == "wa_mse":
        act_path = getattr(cat_args, "activation_weight_path", None)
        if act_path:
            activation_weight_by_linear = load_activation_weight_dict(str(act_path))
            log.info(
                "Loaded static activation abs-max dict: %s (entries=%d)",
                act_path,
                len(activation_weight_by_linear),
            )
        wa_mse_runtime = {
            "dynamic": str(getattr(cat_args, "wa_mse_act_mode", "dynamic")).strip().lower() == "dynamic",
            "cache": None,
            "dataset": str(getattr(cat_args, "wa_mse_calib_dataset", "wikitext2")),
            "nsamples": int(getattr(cat_args, "wa_mse_calib_nsamples", 512)),
            "seqlen": int(getattr(cat_args, "wa_mse_calib_seqlen", 512)),
            "seed": int(getattr(cat_args, "wa_mse_calib_seed", 0)),
            "device": str(getattr(cat_args, "wa_mse_calib_device", "")).strip() or str(cat_args.train_device),
            "log_every": int(getattr(cat_args, "wa_mse_calib_log_every", 0)),
            "model_path": str(vae_args.model_path),
            "access_token": hf_args.access_token,
            "static_dict": activation_weight_by_linear,
        }
        if not bool(wa_mse_runtime["dynamic"]) and activation_weight_by_linear is None:
            raise ValueError(
                "wa_mse requires either --wa_mse_act_mode dynamic or --activation_weight_path in static mode."
            )
        if bool(wa_mse_runtime["dynamic"]):
            log.info(
                "wa_mse dynamic act_max enabled: dataset=%s nsamples=%d seqlen=%d seed=%d device=%s",
                str(wa_mse_runtime["dataset"]),
                int(wa_mse_runtime["nsamples"]),
                int(wa_mse_runtime["seqlen"]),
                int(wa_mse_runtime["seed"]),
                str(wa_mse_runtime["device"]),
            )

    transpose_modules = _split_csv(cat_args.transpose_modules)
    projection_suffixes = _split_csv(cat_args.projection_suffixes)
    only_decoder_projections = bool(cat_args.only_decoder_projections) and not bool(cat_args.include_all_linears)

    all_linears = _collect_linears(
        model,
        transpose_modules,
        only_decoder_projections=only_decoder_projections,
        projection_suffixes=projection_suffixes,
    )
    discovered_categories = sorted({r.category for r in all_linears})
    schedule_categories = _validate_schedule_categories(parallel_schedule, discovered_categories)
    schedule_category_set = set(schedule_categories)
    log.info("discovered_categories=%s", ",".join(discovered_categories))
    log.info("all candidate linears=%d, by_category=%s", len(all_linears), _summarize_category_counts(all_linears))

    discovered_skip_keys = []
    layer_indices_set: Set[int] = set()
    for r in all_linears:
        li = _extract_layer_idx(r.name)
        if li is not None:
            discovered_skip_keys.append((li, r.category))
            layer_indices_set.add(int(li))
    layer_indices = sorted(layer_indices_set)
    if not layer_indices:
        raise ValueError("未发现可训练的 decoder projection linear（含有效 layer idx）")
    log.info("layers discovered=%d, min=%d, max=%d", len(layer_indices), layer_indices[0], layer_indices[-1])

    # 启动前做一次层覆盖检查，防止 schedule 不匹配特定层结构。
    for li in layer_indices:
        cats_this_layer = sorted(
            {
                r.category
                for r in all_linears
                if _extract_layer_idx(r.name) == int(li)
            }
        )
        missing = [c for c in schedule_categories if c not in set(cats_this_layer)]
        if missing:
            raise ValueError(
                f"layer={li} 缺少 schedule 所需类别: {missing}, "
                f"当前层可用类别: {cats_this_layer}"
            )

    skip_layer_keys, matched, missing = resolve_skip_layer_matches(
        getattr(cat_args, "skip_layers", ""),
        discovered_skip_keys,
    )
    if skip_layer_keys:
        if matched:
            log.info(
                "skip_layers 生效: %s",
                ",".join(f"{li}.{cat}" for li, cat in matched),
            )
        if missing:
            log.warning(
                "skip_layers 未匹配到任何 Linear: %s",
                ",".join(f"{li}.{cat}" for li, cat in missing),
            )

    steps_per_group = int(cat_args.steps_per_group) if cat_args.steps_per_group is not None else int(
        cat_args.steps_per_category)
    intra_parallel_raw = getattr(cat_args, "intra_parallel", 1)
    category_intra_parallel: Dict[str, Tuple[int, int]] = {}
    category_codebook: Dict[str, Tuple[int, int]] = {}
    for cat in schedule_categories:
        category_intra_parallel[cat] = resolve_intra_parallel_for_category(intra_parallel_raw, cat)
        category_codebook[cat] = (
            resolve_codebook_int_for_category(
                getattr(vae_args, "codebook_bits"),
                cat,
                arg_name="codebook_bits",
            ),
            resolve_codebook_int_for_category(
                getattr(vae_args, "codebook_dim"),
                cat,
                arg_name="codebook_dim",
            ),
        )

    unique_parallel = sorted(set(category_intra_parallel.values()))
    if int(getattr(vae_args, "parallel_layers", 1)) != 1:
        log.warning(
            "检测到 --parallel_layers=%d，但当前脚本不使用该参数；请使用 --intra_parallel。",
            int(vae_args.parallel_layers),
        )
    if unique_parallel:
        if len(unique_parallel) == 1:
            intra_row_parts, intra_col_parts = unique_parallel[0]
            intra_parallel_desc = _format_intra_parallel_desc(intra_row_parts, intra_col_parts)
            log.info(
                "并行配置: intra_parallel=%s (rows=%d, cols=%d)",
                intra_parallel_desc,
                intra_row_parts,
                intra_col_parts,
            )
        else:
            per_cat_desc = ",".join(
                f"{cat}:{_format_intra_parallel_desc(*category_intra_parallel[cat])}"
                for cat in schedule_categories
            )
            log.info(
                "并行配置: intra_parallel=per_category{%s}",
                per_cat_desc,
            )
    unique_codebook = sorted(set(category_codebook.values()))
    if unique_codebook:
        if len(unique_codebook) == 1:
            cb_bits, cb_dim = unique_codebook[0]
            log.info("codebook 配置: bits=%d, dim=%d", cb_bits, cb_dim)
        else:
            per_cat_cb_desc = ",".join(
                f"{cat}:[bits={category_codebook[cat][0]},dim={category_codebook[cat][1]}]"
                for cat in schedule_categories
            )
            log.info("codebook 配置: per_category{%s}", per_cat_cb_desc)

    stage_counter = 0
    lora_round_idx = 0
    total_stages = len(layer_indices) * len(parallel_schedule)
    log.info("schedule stage total=%d (%d layers x %d groups)",
             total_stages, len(layer_indices), len(parallel_schedule))
    for li in layer_indices:
        for stage_idx, cat_group in enumerate(parallel_schedule):
            group_tag = f"L{li}.S{stage_idx}.{'-'.join(cat_group)}"
            log.info("===== Stage %d/%d: %s =====", stage_counter + 1, total_stages, group_tag)
            group_refs = _collect_refs_for_layer_group(
                model=model,
                layer_idx=li,
                categories_in_group=cat_group,
                transpose_modules=transpose_modules,
                only_decoder_projections=only_decoder_projections,
                projection_suffixes=projection_suffixes,
            )
            if not group_refs:
                log.info("[%s] no remaining nn.Linear in this stage, skip.", group_tag)
                stage_counter += 1
                continue
            stage_parallel_set = {category_intra_parallel[cat] for cat in cat_group}
            if len(stage_parallel_set) != 1:
                stage_parallel_detail = ",".join(
                    f"{cat}:{_format_intra_parallel_desc(*category_intra_parallel[cat])}"
                    for cat in cat_group
                )
                raise ValueError(
                    f"[{group_tag}] 当前 stage 中并行训练的类别必须共享同一 intra_parallel。"
                    f" 实际配置: {stage_parallel_detail}"
                )
            stage_row_parts, stage_col_parts = next(iter(stage_parallel_set))
            stage_codebook_set = {category_codebook[cat] for cat in cat_group}
            if len(stage_codebook_set) != 1:
                stage_codebook_detail = ",".join(
                    f"{cat}:[bits={category_codebook[cat][0]},dim={category_codebook[cat][1]}]"
                    for cat in cat_group
                )
                raise ValueError(
                    f"[{group_tag}] 当前 stage 中并行训练的类别必须共享同一 codebook 配置。"
                    f" 实际配置: {stage_codebook_detail}"
                )
            stage_codebook_bits, stage_codebook_dim = next(iter(stage_codebook_set))
            stage_vae_args = _clone_namespace(
                vae_args,
                codebook_bits=int(stage_codebook_bits),
                codebook_dim=int(stage_codebook_dim),
            )
            stage_parts_per_linear = int(stage_row_parts) * int(stage_col_parts)
            stage_intra_parallel_desc = _format_intra_parallel_desc(stage_row_parts, stage_col_parts)

            shape_summary = "; ".join(
                f"{r.name}:shape={tuple(int(x) for x in r.module.weight.shape)},transpose={r.transpose}"
                for r in group_refs
            )
            log.info("[%s] group_refs=%d | %s", group_tag, len(group_refs), shape_summary)
            remain_before = _collect_remaining_target_refs(
                model=model,
                target_categories=schedule_categories,
                transpose_modules=transpose_modules,
                only_decoder_projections=only_decoder_projections,
                projection_suffixes=projection_suffixes,
            )
            log.info(
                "[%s] remaining target nn.Linear before stage=%d, by_category=%s",
                group_tag,
                len(remain_before),
                _summarize_category_counts(remain_before),
            )
            _validate_group_parallel_compatibility(
                group_refs=group_refs,
                group_tag=group_tag,
                intra_parallel=(stage_row_parts, stage_col_parts),
                codebook_dim=int(stage_codebook_dim),
            )
            log.info(
                "---- Stage: %s (linears=%d, intra_parallel=%s, codebook_bits=%d, codebook_dim=%d, num_models=%d) ----",
                group_tag,
                len(group_refs),
                stage_intra_parallel_desc,
                int(stage_codebook_bits),
                int(stage_codebook_dim),
                len(group_refs) * stage_parts_per_linear,
            )
            _train_group_vae_and_replace(
                model=model,
                group_refs=group_refs,
                group_tag=group_tag,
                vae_args=stage_vae_args,
                training_args=training_args,
                train_device=cat_args.train_device,
                convert_device=cat_args.convert_device,
                do_convert=bool(cat_args.convert),
                steps=steps_per_group,
                batch_size=cat_args.batch_size,
                log_every=cat_args.log_every,
                eval_every=cat_args.eval_every,
                eval_blocks=cat_args.eval_blocks,
                output_dir=cat_args.output_dir,
                intra_parallel=(stage_row_parts, stage_col_parts),
                intra_part_sort_mode=str(getattr(cat_args, "intra_part_sort_mode", "row_l2")),
                skip_layer_keys=skip_layer_keys,
                activation_weight_by_linear=activation_weight_by_linear,
                wa_mse_runtime=wa_mse_runtime,
            )

            if cat_args.lora_after_category and stage_idx == len(parallel_schedule) - 1:
                from train_utils.lora_utils import lora_finetune_remaining_categories

                log.info("LoRA 微调前评估...")
                _eval_ppl_after_category(
                    model=model,
                    vae_args=vae_args,
                    ppl_limit=cat_args.ppl_limit,
                    category=f"{group_tag}.before_lora",
                    eval_device=cat_args.train_device,
                )

                model = lora_finetune_remaining_categories(
                    model=model,
                    remaining_categories=schedule_categories,
                    collect_linears_fn=_collect_linears,
                    transpose_modules=transpose_modules,
                    projection_suffixes=projection_suffixes,
                    only_decoder_projections=only_decoder_projections,
                    cat_args=cat_args,
                    vae_args=vae_args,
                    training_args=training_args,
                    logger=log,
                    lora_round_idx=lora_round_idx,
                )
                lora_round_idx += 1
                remain_after_lora = _collect_remaining_target_refs(
                    model=model,
                    target_categories=schedule_categories,
                    transpose_modules=transpose_modules,
                    only_decoder_projections=only_decoder_projections,
                    projection_suffixes=projection_suffixes,
                )
                log.info(
                    "[%s] remaining target nn.Linear after LoRA=%d, by_category=%s",
                    group_tag,
                    len(remain_after_lora),
                    _summarize_category_counts(remain_after_lora),
                )

            _eval_ppl_after_category(
                model=model,
                vae_args=vae_args,
                ppl_limit=cat_args.ppl_limit,
                category=group_tag,
                eval_device=cat_args.train_device,
            )
            # stage_dir_name = _safe_path_token(group_tag)
            # stage_model_dir = os.path.join(run_output_dir, stage_dir_name)
            # save_paths = save_model_checkpoint(
            #     model,
            #     stage_model_dir,
            #     base_model_path=vae_args.model_path,
            #     tokenizer=None,
            #     save_config=True,
            #     extra_meta={
            #         "stage": "after_schedule_stage",
            #         "layer_index": int(li),
            #         "schedule_stage_index": int(stage_idx),
            #         "group_tag": group_tag,
            #         "stage_counter": int(stage_counter),
            #         "lora_after_category": bool(cat_args.lora_after_category),
            #     },
            # )
            # log.info("Saved stage checkpoint (%s): %s", group_tag, save_paths["output_dir"])
            stage_counter += 1

    if cat_args.save_model:
        if not cat_args.convert:
            raise ValueError("--save_model requires --convert")
        from transformers import AutoTokenizer
        from litebsq.vae_linear import clear_model_vae_linear_cache

        model_out = os.path.join(run_output_dir, "final_model")
        tok = AutoTokenizer.from_pretrained(vae_args.model_path, use_fast=True, token=hf_args.access_token)
        cleared = clear_model_vae_linear_cache(model)
        log.info("Final save: cleared decoded cache for %d VAELinear modules.", cleared)
        save_paths = save_model_checkpoint(
            model,
            model_out,
            base_model_path=vae_args.model_path,
            tokenizer=tok,
            save_config=True,
            extra_meta={"stage": "final"},
            unload_vae_original_weights=bool(cat_args.unload_vae_original_weights_on_final_save),
        )
        log.info("Saved final model to %s", save_paths["output_dir"])

    # 结束前检查：目标类别是否还有 nn.Linear 未替换。
    remain = _collect_linears(
        model,
        transpose_modules,
        only_decoder_projections=only_decoder_projections,
        projection_suffixes=projection_suffixes,
    )
    remain_target = [r for r in remain if r.category in schedule_category_set]
    if remain_target:
        log.warning(
            "训练结束后仍有未替换目标 Linear: %d (示例: %s)",
            len(remain_target),
            ",".join(r.name for r in remain_target[:5]),
        )
    else:
        log.info("训练结束：schedule 覆盖类别均已替换。")

    log.info("Done.")


if __name__ == "__main__":
    main()
