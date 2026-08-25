import json
import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from train_utils.cat_arg_overrides import OverrideTable
from train_utils.cat_train_pipeline import run_cat_train
from train_utils.cat_train_runtime import load_cat_resume_distill_progress


def _write_meta(path, payload):
    path.mkdir(parents=True, exist_ok=True)
    meta_path = path / "checkpoint_meta.json"
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    return meta_path


def test_empty_resume_progress():
    progress = load_cat_resume_distill_progress(None)

    assert progress.completed_categories == ()
    assert progress.distill_stage_history == ()


def test_resume_progress_reads_completed_and_full_history(tmp_path):
    _write_meta(
        tmp_path,
        {
            "completed_categories": ["q_proj", "k_proj"],
            "distill_stage_history": [
                {"category": "q_proj", "did_train": True},
                {"category": "k_proj", "did_train": True},
            ],
        },
    )

    progress = load_cat_resume_distill_progress(str(tmp_path))

    assert progress.completed_categories == ("q_proj", "k_proj")
    assert len(progress.distill_stage_history) == 2
    assert progress.distill_stage_history[0]["category"] == "q_proj"


def test_resume_progress_rejects_duplicate_completed_categories(tmp_path):
    _write_meta(tmp_path, {"completed_categories": ["q_proj", "q_proj"]})

    with pytest.raises(ValueError, match="duplicate"):
        load_cat_resume_distill_progress(str(tmp_path))


def test_resume_progress_uses_single_distill_stage_when_full_history_missing(tmp_path):
    _write_meta(
        tmp_path,
        {
            "completed_categories": ["q_proj"],
            "distill_stage": {"category": "q_proj", "did_train": True},
        },
    )

    progress = load_cat_resume_distill_progress(str(tmp_path))

    assert progress.completed_categories == ("q_proj",)
    assert progress.distill_stage_history == ({"category": "q_proj", "did_train": True},)


def test_resume_progress_reads_extra_meta_checkpoint_format(tmp_path):
    _write_meta(
        tmp_path,
        {
            "format": "vaellm_state_dict_with_meta",
            "extra_meta": {
                "completed_categories": ["q_proj"],
                "distill_stage_history": [{"category": "q_proj", "did_train": True}],
            },
        },
    )

    progress = load_cat_resume_distill_progress(str(tmp_path))

    assert progress.completed_categories == ("q_proj",)
    assert progress.distill_stage_history[0]["category"] == "q_proj"


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 2)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyLayer()])
        self.config = SimpleNamespace(use_cache=False)


def _override(name, value):
    return OverrideTable(
        arg_name=name,
        allowed_selectors=("default", "after"),
        has_default=True,
        default=value,
    )


def _cat_args(tmp_path, resume_path):
    return SimpleNamespace(
        activation_calib_dataset="",
        activation_calib_device="",
        activation_calib_log_every=0,
        activation_calib_nsamples=1,
        activation_calib_seed=0,
        activation_calib_seqlen=8,
        allow_tail_group=True,
        batch_size=1,
        candidate_artifact_output_dir=str(tmp_path / "candidate"),
        candidate_artifact_spec="",
        convert=True,
        convert_device="cpu",
        deterministic=False,
        distill_after_category="remaining_lora",
        distill_batch_size=_override("distill_batch_size", 1),
        distill_dataset="wiki=1.0",
        distill_eakld_confidence_k=16,
        distill_hidden_alignment_layer_weighting="uniform",
        distill_hidden_loss_weight=_override("distill_hidden_loss_weight", 0.0),
        distill_independent_categories=False,
        distill_log_every=_override("distill_log_every", 1),
        distill_loss_alpha=_override("distill_loss_alpha", 1.0),
        distill_loss_type=_override("distill_loss_type", "sft"),
        distill_lr=_override("distill_lr", 1e-4),
        distill_pre_mlp_hidden_loss_weight=_override("distill_pre_mlp_hidden_loss_weight", 0.0),
        distill_prompt_kd_weight=_override("distill_prompt_kd_weight", 0.0),
        distill_steps=_override("distill_steps", 0),
        distill_temperature=_override("distill_temperature", 1.0),
        distill_weight_decay=_override("distill_weight_decay", 0.0),
        eval_blocks=1,
        eval_every=0,
        eval_hif4_act=False,
        eval_ppl=False,
        eval_tasks="",
        gpu_resident_data=False,
        include_all_linears=False,
        linear_group_size=1,
        log_every=1,
        lora_alpha=_override("lora_alpha", 4.0),
        lora_dropout=_override("lora_dropout", 0.0),
        lora_rank=_override("lora_rank", 2),
        lora_use_dora=_override("lora_use_dora", False),
        outlier_channel_scope="global",
        outlier_mlp_fuse_weights="",
        outlier_mlp_rank_metric="none",
        outlier_protect_axis="input",
        outlier_protect_channel_quant="none",
        outlier_protect_min_per_layer=0,
        outlier_protect_mode="channel",
        outlier_rank_metric="weight_abs",
        outlier_residual_block_shape=(1, 1),
        outlier_residual_codec="coo_fp16",
        outlier_residual_index_bits=16,
        outlier_residual_min_abs=0.0,
        outlier_residual_value_bits=8,
        outlier_residual_vae_batch_multiplier=1,
        outlier_residual_vae_decoder_share_scope="none",
        outlier_residual_vae_lr=0.0,
        outlier_residual_vae_steps=0,
        output_dir=str(tmp_path / "out"),
        ppl_limit=-1,
        resume_from_checkpoint=str(resume_path),
        rot_llm=False,
        save_candidate_artifact=False,
        save_model=False,
        seed=0,
        skip_layers="",
        target_categories="q_proj",
        train_device="cpu",
        transpose_modules="q_proj",
    )


def _cfg():
    return SimpleNamespace(
        category="q_proj",
        codebook_bits=8,
        codebook_dim=4,
        intra_part_sort_mode="none",
        outlier_protect_count=0,
        outlier_residual_top_p=0.0,
        outlier_residual_vae_codebook_bits=8,
        outlier_residual_vae_codebook_dim=4,
        outlier_residual_vae_stages=1,
        recon_loss_type="mse",
        residual_stages=1,
        steps=1,
    )


def test_full_skip_completed_category_is_not_reentered(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "resume"
    _write_meta(
        checkpoint_dir,
        {
            "extra_meta": {
                "completed_categories": ["q_proj"],
                "distill_stage": {"category": "q_proj", "did_train": True},
            }
        },
    )
    cat_args = _cat_args(tmp_path, checkpoint_dir)
    vae_args = SimpleNamespace(model_path="dummy")
    hf_args = SimpleNamespace(access_token=None)
    training_args = SimpleNamespace(
        bf16=False,
        distill_hif4_act=False,
        distill_teacher_model_offload="none",
        fp16=False,
    )
    collect_calls = {"count": 0}

    def fail_if_category_processing_reentered(*_args, **_kwargs):
        collect_calls["count"] += 1
        raise AssertionError("completed category should skip before category collection")

    monkeypatch.setattr("train_utils.cat_train_pipeline._load_model_for_cat_train", lambda **_kwargs: _TinyModel())
    monkeypatch.setattr(
        "train_utils.cat_train_pipeline.resolve_category_runtime_configs",
        lambda *_args, **_kwargs: {"q_proj": _cfg()},
    )
    monkeypatch.setattr("train_utils.cat_train_pipeline._save_normalized_cat_train_snapshot", lambda **_kwargs: "snapshot")
    monkeypatch.setattr("train_utils.cat_train_pipeline.resolve_distill_teacher_dtype", lambda *_args, **_kwargs: torch.float32)
    monkeypatch.setattr("train_utils.cat_train_pipeline.DistillTeacherRuntime", lambda **_kwargs: object())
    monkeypatch.setattr("train_utils.cat_train_pipeline._collect_sorted_category_refs", fail_if_category_processing_reentered)
    monkeypatch.setattr("e2e_common.post_norm_head.fuse_post_norm_head_linear", lambda _model: False)

    run_cat_train(
        cat_args=cat_args,
        hf_args=hf_args,
        training_args=training_args,
        vae_args=vae_args,
    )

    assert collect_calls["count"] == 0


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _progress_worker(rank, port, checkpoint_dir, queue):
    os.environ.update({"RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": str(rank)})
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    try:
        progress = load_cat_resume_distill_progress(str(checkpoint_dir))
        queue.put(
            (
                rank,
                list(progress.completed_categories),
                len(progress.distill_stage_history),
                len(progress.completed_categories),
            )
        )
    finally:
        torch.distributed.destroy_process_group()


def test_ddp_ranks_read_same_resume_progress(tmp_path):
    checkpoint_dir = tmp_path / "resume"
    _write_meta(
        checkpoint_dir,
        {
            "extra_meta": {
                "completed_categories": ["q_proj", "k_proj"],
                "distill_stage_history": [
                    {"category": "q_proj", "did_train": True},
                    {"category": "k_proj", "did_train": True},
                ],
            }
        },
    )
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    port = _free_port()
    workers = [
        context.Process(target=_progress_worker, args=(rank, port, str(checkpoint_dir), queue))
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0
    received = sorted(queue.get() for _ in workers)

    assert received == [
        (0, ["q_proj", "k_proj"], 2, 2),
        (1, ["q_proj", "k_proj"], 2, 2),
    ]
