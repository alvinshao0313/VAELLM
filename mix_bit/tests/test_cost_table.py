from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from litebsq.llm_vae import Decoder
from litebsq.vae_linear import VAELinear
from mix_bit.assembler import (
    build_model_from_assignments,
    build_uniform_assignments,
    write_uniform_baseline_overlay,
)
from mix_bit.calibration import CalibrationExample, build_causal_kl_mask
from mix_bit.candidate_artifact import save_candidate_artifact_from_model
from mix_bit.candidate_pool import generate_candidate_trials, write_trial_spec
from mix_bit.checkpoint_pool import CandidatePoolIndex, build_candidate_pool_index
from mix_bit.kl_metric import (
    KL_MODE_EXACT_FULL_VOCAB,
    KL_MODE_TEACHER_TOPK,
    METRIC_NAME_EXACT_FULL_VOCAB,
    METRIC_NAME_TEACHER_TOPK,
    per_sample_exact_forward_kl,
)
from mix_bit.model_adapter import get_model_adapter
from mix_bit.model_inventory import ModelInventory, inventory_from_targets
from mix_bit.schema import (
    CalibrationConfig,
    CandidateMode,
    CandidateSpaceConfig,
    CandidateTrainingSpec,
    CategorySpec,
    MixBitRunConfig,
    ModelProfile,
    ResolvedRunConfig,
    TrainingRecipeConfig,
    sha256_file,
)
from mix_bit.teacher_cache import build_teacher_topk_chunk, write_teacher_cache_chunk


HIDDEN = 8
VOCAB = 16
BASELINE_MODE = "b4d4s2"
CANDIDATE_MODE = "b4d4s1"


def _make_decoder(cdim: int) -> Decoder:
    return Decoder(
        in_dim=cdim,
        out_dim=cdim,
        hidden_dim=16,
        num_res_blocks=0,
        decoder_type="linear",
        norm_type="group",
        activation_type="swish",
    )


def _make_vae_linear(
    *,
    in_features: int = HIDDEN,
    out_features: int = HIDDEN,
    cdim: int = 4,
    residual_stages: int = 2,
    with_bias: bool = False,
    transpose: bool = False,
) -> VAELinear:
    n_blocks = (in_features * out_features) // cdim
    logical = (n_blocks, 1, cdim)
    stages = []
    decoders = []
    for _ in range(residual_stages):
        stages.append(torch.randint(0, 2, logical, dtype=torch.bool))
        decoders.append(_make_decoder(cdim))
    return VAELinear(
        in_features=in_features,
        out_features=out_features,
        bias=None if not with_bias else nn.Parameter(torch.zeros(out_features)),
        original_weight=None,
        stage_vq_weights=stages,
        stage_decoders=decoders,
        codebook_dim=cdim,
        stage_codebook_dims=[cdim] * residual_stages,
        transpose=transpose,
        parallel_parts=1,
    )


class _ToyLM(nn.Module):
    def __init__(self, n_layers: int = 2, hidden: int = HIDDEN, vocab: int = VOCAB):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        layers = []
        for _ in range(n_layers):
            layer = nn.Module()
            layer.q_proj = nn.Linear(hidden, hidden, bias=False)
            layer.k_proj = nn.Linear(hidden, hidden, bias=False)
            layers.append(layer)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        class _Cfg:
            model_type = "toy"
            _name_or_path = "toy-model"
            vocab_size = vocab
            use_cache = True

            def save_pretrained(self, path: str) -> None:
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "config.json").write_text(
                    json.dumps({"model_type": "toy", "vocab_size": vocab}),
                    encoding="utf-8",
                )

        self.config = _Cfg()
        self.forward_calls = 0
        self.seen_training = []
        self.seen_use_cache = []
        self.seen_inference_mode = []

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
        self.forward_calls += 1
        self.seen_training.append(bool(self.training))
        self.seen_use_cache.append(bool(getattr(self.config, "use_cache", True)))
        self.seen_inference_mode.append(not torch.is_grad_enabled())
        x = self.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = x + layer.q_proj(x) + layer.k_proj(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits)


def _toy_profile() -> ModelProfile:
    return ModelProfile(
        model_id="toy",
        model_path="toy-model",
        adapter="generic_decoder",
        only_decoder_projections=True,
        candidate_training=CandidateTrainingSpec(linear_group_size="all", allow_tail_group=True),
        layer_index_patterns=(r"(?:^|\.)model\.layers\.(\d+)\.",),
        categories=(
            CategorySpec("q_proj", "q_proj", True),
            CategorySpec("k_proj", "k_proj", False),
        ),
        regression_expectations={},
    )


def _toy_modes() -> tuple[CandidateMode, ...]:
    # codebook_bits == codebook_dim == 4 so nominal_bit == residual_stages.
    return (
        CandidateMode(
            name=BASELINE_MODE,
            nominal_bit=2.0,
            codebook_bits=4,
            codebook_dim=4,
            residual_stages=2,
        ),
        CandidateMode(
            name=CANDIDATE_MODE,
            nominal_bit=1.0,
            codebook_bits=4,
            codebook_dim=4,
            residual_stages=1,
        ),
    )


def _make_resolved(tmp_path: Path, profile: ModelProfile, modes: tuple[CandidateMode, ...]) -> ResolvedRunConfig:
    recipe = TrainingRecipeConfig(
        recipe_id="toy_recipe",
        values={
            "seed": 31,
            "deterministic": True,
            "vae_steps": 10,
            "vae_batch_size": 8,
            "base_ch": 8,
            "num_res_blocks": 0,
            "decoder_base_ch": 8,
            "decoder_num_res_blocks": 0,
            "norm_type": "layer",
            "activation_type": "swish",
            "decoder_type": "linear",
            "recon_loss_type": "mse",
            "quantizer_type": "BSQ",
            "gamma0": 1.0,
            "gamma": 1.0,
            "zeta": 1.0,
            "inv_temperature": 100.0,
            "vae_learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.95,
            "vae_weight_decay": 0.0,
            "vae_optim": "adamw",
            "vae_lr_scheduler_type": "linear",
            "vae_warmup_ratio": 0,
            "l1_weight": 1.0,
            "lfq_weight": 1.0,
            "commitment_loss_weight": 0.25,
            "entropy_loss_weight": 0.01,
            "normalize_weight": True,
            "vae_decoder_checkpoint": True,
            "new_quant": True,
            "log_every": 1,
            "eval_every": 0,
            "eval_blocks": 8,
            "channel_protect_mode": "channel",
            "channel_protect_count": 0,
            "channel_min_per_layer": 0,
            "after_category_mode": "none",
            "skip_ppl_eval": True,
            "eval_tasks": "",
            "rot_llm": False,
            "fp16": False,
            "bf16": True,
        },
    )
    config = MixBitRunConfig(
        run_id="toy_run",
        model_profile=profile,
        candidate_space=CandidateSpaceConfig(
            candidate_space_id="toy_space",
            baseline_mode=BASELINE_MODE,
            target_average_bit=2.0,
            modes=modes,
        ),
        training_recipe=recipe,
        calibration=CalibrationConfig(
            source_jsonl=str(tmp_path / "calib.jsonl"),
            input_format="text",
            max_samples=4,
            max_length=8,
            seed=0,
            label_mode="all_nonpad",
        ),
    )
    root = tmp_path / "result" / "mix_bit" / profile.model_id
    return ResolvedRunConfig(
        config=config,
        run_config_path=str(tmp_path / "run.json"),
        run_config_sha256="r" * 64,
        model_profile_path=str(tmp_path / "profile.json"),
        model_profile_sha256="p" * 64,
        candidate_space_path=str(tmp_path / "space.json"),
        candidate_space_sha256="c" * 64,
        training_recipe_path=str(tmp_path / "recipe.json"),
        training_recipe_sha256="t" * 64,
        canonical_model_root=str(root),
        canonical_run_root=str(root / "runs" / config.run_id),
    )


def _inventory_for(profile: ModelProfile, model: nn.Module) -> ModelInventory:
    adapter = get_model_adapter("generic_decoder")
    targets = adapter.discover_target_linears(model, profile)
    return inventory_from_targets(
        profile=profile,
        model=model,
        targets=targets,
        model_profile_sha256="p" * 64,
    )


def _export_pool(resolved: ResolvedRunConfig, inventory: ModelInventory) -> CandidatePoolIndex:
    trials = generate_candidate_trials(resolved, inventory)
    for trial in trials:
        host = nn.Module()
        host.model = nn.Module()
        max_block = max(t.block_index for t in inventory.targets)
        layers = []
        for block_idx in range(max_block + 1):
            layer = nn.Module()
            for target in inventory.targets:
                if target.block_index != block_idx:
                    continue
                if target.category != trial.category_name:
                    continue
                setattr(
                    layer,
                    target.module_suffix,
                    _make_vae_linear(
                        cdim=trial.mode.codebook_dim,
                        residual_stages=trial.mode.residual_stages,
                        transpose=bool(target.transpose),
                    ),
                )
            layers.append(layer)
        host.model.layers = nn.ModuleList(layers)
        host.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        host.norm = nn.LayerNorm(HIDDEN)
        host.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        trial_spec_path = write_trial_spec(trial, command=["python", "-c", "pass"])
        save_candidate_artifact_from_model(
            model=host,
            trial_spec_path=trial_spec_path,
            output_dir=str(Path(trial.trial_root) / "artifact"),
            source_run_dir=str(Path(trial.trial_root) / "runs" / "fake"),
        )
    return build_candidate_pool_index(resolved, inventory)


def _make_examples(n: int = 3) -> list[CalibrationExample]:
    examples = []
    for i in range(n):
        length = 4 + (i % 2)
        ids = torch.arange(1, length + 1, dtype=torch.long)
        mask = torch.ones(length, dtype=torch.long)
        labels = ids.clone()
        examples.append(
            CalibrationExample(
                sample_id=100 + i,
                input_ids=ids,
                attention_mask=mask,
                labels=labels,
            )
        )
    return examples


def _write_dataset(tmp_path: Path, examples: list[CalibrationExample], resolved: ResolvedRunConfig, inventory: ModelInventory) -> tuple[Path, Path]:
    dataset_path = tmp_path / "dataset.pt"
    payload = [
        {
            "sample_id": ex.sample_id,
            "input_ids": ex.input_ids,
            "attention_mask": ex.attention_mask,
            "labels": ex.labels,
        }
        for ex in examples
    ]
    torch.save(payload, dataset_path)
    manifest = {
        "kind": "mix_bit_calibration_dataset_manifest",
        "run_config_sha256": resolved.run_config_sha256,
        "model_profile_sha256": resolved.model_profile_sha256,
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "training_recipe_sha256": resolved.training_recipe_sha256,
        "model_inventory_fingerprint": inventory.fingerprint_sha256,
        "sample_count": len(examples),
        "dataset_file": str(dataset_path),
        "dataset_file_sha256": sha256_file(dataset_path),
        "pad_token_id": 0,
    }
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dataset_path, manifest_path


def _write_teacher_cache(
    cache_dir: Path,
    *,
    examples: list[CalibrationExample],
    teacher: nn.Module,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    dataset_sha: str,
    teacher_topk: int = 4,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    teacher.eval()
    chunks = []
    all_ids = []
    with torch.inference_mode():
        for idx, ex in enumerate(examples):
            input_ids = ex.input_ids.unsqueeze(0)
            attention_mask = ex.attention_mask.unsqueeze(0)
            labels = ex.labels.unsqueeze(0) if ex.labels is not None else None
            outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
            shifted = outputs.logits[:, :-1, :]
            valid = build_causal_kl_mask(attention_mask, labels)
            chunk = build_teacher_topk_chunk(
                sample_ids=[ex.sample_id],
                shifted_teacher_logits=shifted,
                valid_mask=valid,
                teacher_topk=teacher_topk,
                cache_prob_dtype="float32",
            )
            rel = f"chunk_{idx:04d}.pt"
            digest = write_teacher_cache_chunk(cache_dir / rel, chunk)
            chunks.append(
                {
                    "path": rel,
                    "sample_start": idx,
                    "sample_end": idx + 1,
                    "sample_ids": [ex.sample_id],
                    "n_valid": int(chunk["teacher_topk_indices"].shape[0]),
                    "sha256": digest,
                }
            )
            all_ids.append(ex.sample_id)
    index = {
        "kind": "mix_bit_teacher_topk_cache_index",
        "kl_mode": KL_MODE_TEACHER_TOPK,
        "metric_name": METRIC_NAME_TEACHER_TOPK,
        "teacher_topk": teacher_topk,
        "vocab_size": VOCAB,
        "cache_prob_dtype": "float32",
        "run_config_sha256": resolved.run_config_sha256,
        "model_profile_sha256": resolved.model_profile_sha256,
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "training_recipe_sha256": resolved.training_recipe_sha256,
        "model_inventory_fingerprint": inventory.fingerprint_sha256,
        "dataset_file_sha256": dataset_sha,
        "model_id": resolved.config.model_profile.model_id,
        "model_path": resolved.config.model_profile.model_path,
        "sample_count": len(examples),
        "sample_ids": all_ids,
        "chunks": chunks,
        "cache_dir": str(cache_dir.resolve()),
    }
    index_path = cache_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache_dir


@pytest.fixture()
def cost_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    torch.manual_seed(0)
    profile = _toy_profile()
    modes = _toy_modes()
    resolved = _make_resolved(tmp_path, profile, modes)
    template = _ToyLM(n_layers=2)
    inventory = _inventory_for(profile, template)
    pool_index = _export_pool(resolved, inventory)
    assignments = build_uniform_assignments(pool_index, BASELINE_MODE)
    overlay_dir = Path(resolved.canonical_run_root) / "baseline" / BASELINE_MODE
    overlay_path = write_uniform_baseline_overlay(
        output_dir=str(overlay_dir),
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        mode_name=BASELINE_MODE,
    )
    examples = _make_examples(3)
    dataset_path, manifest_path = _write_dataset(tmp_path, examples, resolved, inventory)

    load_counts = {"student": 0, "teacher": 0}

    def _load_model(self, _profile, *, access_token=None):
        # Cost search marks teacher loads via a dedicated helper; student uses adapter.
        model = copy.deepcopy(template)
        load_counts["student"] += 1
        return model

    monkeypatch.setattr(
        "mix_bit.model_adapter.GenericDecoderAdapter.load_model",
        _load_model,
    )

    teacher_template = _ToyLM(n_layers=2)
    # Make teacher logits differ from a randomly initialized student backbone.
    with torch.no_grad():
        teacher_template.lm_head.weight.add_(0.35)

    cache_dir = _write_teacher_cache(
        tmp_path / "teacher_cache",
        examples=examples,
        teacher=teacher_template,
        resolved=resolved,
        inventory=inventory,
        dataset_sha=sha256_file(dataset_path),
        teacher_topk=4,
    )
    cost_run_root = Path(resolved.canonical_run_root) / "costs" / "topk_k4"
    cost_run_root.mkdir(parents=True, exist_ok=True)

    return {
        "tmp_path": tmp_path,
        "resolved": resolved,
        "inventory": inventory,
        "pool_index": pool_index,
        "overlay_path": overlay_path,
        "examples": examples,
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "cache_dir": cache_dir,
        "cost_run_root": cost_run_root,
        "template": template,
        "teacher_template": teacher_template,
        "load_counts": load_counts,
        "assignments": assignments,
    }


def test_candidate_cost_is_mean_of_paired_sample_deltas():
    from mix_bit.cost_search import summarize_paired_deltas

    deltas = np.array([0.5, -0.25, 1.0], dtype=np.float64)
    stats = summarize_paired_deltas(deltas)
    assert stats["mean_delta_kl"] == pytest.approx(float(deltas.mean()))
    assert stats["mean_delta_kl"] == pytest.approx((0.5 - 0.25 + 1.0) / 3.0)


def test_delta_std_uses_sample_ddof_one_and_se_is_std_over_sqrt_n():
    from mix_bit.cost_search import summarize_paired_deltas

    deltas = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    stats = summarize_paired_deltas(deltas)
    expected_std = float(np.std(deltas, ddof=1))
    expected_se = expected_std / math.sqrt(len(deltas))
    assert stats["std_delta_kl"] == pytest.approx(expected_std)
    assert stats["standard_error_delta_kl"] == pytest.approx(expected_se)


def test_single_sample_std_and_se_are_zero():
    from mix_bit.cost_search import summarize_paired_deltas

    stats = summarize_paired_deltas(np.array([-0.7], dtype=np.float64))
    assert stats["mean_delta_kl"] == pytest.approx(-0.7)
    assert stats["std_delta_kl"] == 0.0
    assert stats["standard_error_delta_kl"] == 0.0


def test_negative_delta_is_preserved(cost_world, monkeypatch):
    from mix_bit import cost_search

    # Force candidate KL below baseline so mean delta is negative.
    baseline = np.array([1.0, 1.2, 0.8], dtype=np.float64)
    candidate = np.array([0.5, 0.4, 0.9], dtype=np.float64)

    def _fake_eval(ctx, **_kwargs):
        return {
            "sample_ids": np.array([100, 101, 102], dtype=np.int64),
            "per_sample_kl": candidate.copy(),
        }

    monkeypatch.setattr(cost_search, "evaluate_student_per_sample_kl", _fake_eval)
    monkeypatch.setattr(
        cost_search,
        "build_candidate_module",
        lambda *args, **kwargs: nn.Linear(HIDDEN, HIDDEN, bias=False),
    )
    monkeypatch.setattr(
        cost_search,
        "load_compact_state_mmap",
        lambda source: {},
    )
    monkeypatch.setattr(
        cost_search,
        "extract_prefixed_module_state",
        lambda state, name: {f"{name}.weight": torch.zeros(HIDDEN, HIDDEN)},
    )

    ctx = cost_search.create_cost_worker(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_path=cost_world["dataset_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        cost_run_root=cost_world["cost_run_root"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
        device="cpu",
        batch_size=1,
        skip_baseline_build=True,
    )
    ctx.sample_ids = np.array([100, 101, 102], dtype=np.int64)
    ctx.baseline_kl = baseline
    ctx.model = _ToyLM()
    ctx.model.eval()

    rows = cost_search.run_category_mode_job(
        ctx,
        category="q_proj",
        mode=CANDIDATE_MODE,
    )
    assert rows
    for row in rows:
        assert row["mean_delta_kl"] < 0
        npz = np.load(row["per_sample_file"])
        deltas = npz["candidate_kl"].astype(np.float64) - npz["baseline_kl"].astype(np.float64)
        assert np.any(deltas < 0)
        assert float(deltas.mean()) == pytest.approx(row["mean_delta_kl"])


def test_param_count_comes_from_inventory_shape(cost_world, monkeypatch):
    from mix_bit import cost_search

    target = cost_world["inventory"].targets[0]
    expected = int(target.param_count)
    assert expected == target.in_features * target.out_features + (1 if target.has_bias else 0)

    monkeypatch.setattr(
        cost_search,
        "evaluate_student_per_sample_kl",
        lambda ctx, **kwargs: {
            "sample_ids": np.array([100, 101, 102], dtype=np.int64),
            "per_sample_kl": np.zeros(3, dtype=np.float64),
        },
    )
    monkeypatch.setattr(
        cost_search,
        "build_candidate_module",
        lambda *args, **kwargs: nn.Linear(HIDDEN, HIDDEN, bias=False),
    )
    monkeypatch.setattr(cost_search, "load_compact_state_mmap", lambda source: {})
    monkeypatch.setattr(
        cost_search,
        "extract_prefixed_module_state",
        lambda state, name: {f"{name}.weight": torch.zeros(HIDDEN, HIDDEN)},
    )

    ctx = cost_search.create_cost_worker(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_path=cost_world["dataset_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        cost_run_root=cost_world["cost_run_root"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
        device="cpu",
        batch_size=1,
        skip_baseline_build=True,
    )
    ctx.sample_ids = np.array([100, 101, 102], dtype=np.int64)
    ctx.baseline_kl = np.zeros(3, dtype=np.float64)
    ctx.model = _ToyLM()
    ctx.model.eval()

    rows = cost_search.run_category_mode_job(
        ctx,
        category=target.category,
        mode=CANDIDATE_MODE,
    )
    by_name = {row["module_name"]: row for row in rows}
    assert by_name[target.module_name]["param_count"] == expected


def test_forward_uses_eval_inference_mode_no_cache_and_one_causal_shift(cost_world):
    from mix_bit import cost_search

    result = cost_search.evaluate_and_write_baseline_per_sample(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_path=cost_world["dataset_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        cost_run_root=cost_world["cost_run_root"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
        device="cpu",
        batch_size=2,
    )
    npz_path = Path(result["baseline_per_sample_path"])
    assert npz_path.is_file()
    data = np.load(npz_path, allow_pickle=False)
    assert data["sample_ids"].dtype == np.int64
    assert data["baseline_kl"].dtype == np.float64
    assert str(data["kl_mode"]) == KL_MODE_TEACHER_TOPK
    assert str(data["metric_name"]) == METRIC_NAME_TEACHER_TOPK

    # Rebuild a student and wrap forward to assert contract on a second evaluation.
    model = build_model_from_assignments(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        assignments=cost_world["assignments"],
        device="cpu",
    )
    model.train()
    model.config.use_cache = True
    original_forward = model.forward
    call_meta = {"n": 0, "training": [], "use_cache": [], "inference": [], "logit_lens": []}

    def _wrapped(input_ids, attention_mask=None, **kwargs):
        call_meta["n"] += 1
        call_meta["training"].append(bool(model.training))
        call_meta["use_cache"].append(bool(model.config.use_cache))
        call_meta["inference"].append(not torch.is_grad_enabled())
        out = original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        call_meta["logit_lens"].append(int(out.logits.shape[1]))
        return out

    model.forward = _wrapped  # type: ignore[method-assign]
    ctx = cost_search.CostWorkerContext.__new__(cost_search.CostWorkerContext)
    ctx.model = model
    ctx.examples = cost_world["examples"]
    ctx.device = torch.device("cpu")
    ctx.batch_size = 1
    ctx.pad_token_id = 0
    ctx.kl_mode = KL_MODE_TEACHER_TOPK
    ctx.metric_name = METRIC_NAME_TEACHER_TOPK
    ctx.teacher_topk = 4
    ctx.teacher_cache_index = cost_search.load_teacher_cache_for_worker(cost_world["cache_dir"])
    ctx.teacher_model = None
    ctx.teacher_model_id = cost_world["resolved"].config.model_profile.model_id

    out = cost_search.evaluate_student_per_sample_kl(ctx)
    assert call_meta["n"] == len(cost_world["examples"])
    assert all(t is False for t in call_meta["training"])
    assert all(c is False for c in call_meta["use_cache"])
    assert all(inf is True for inf in call_meta["inference"])
    # One causal shift: model sees full T, metric uses T-1 internally.
    for ex, logit_len in zip(cost_world["examples"], call_meta["logit_lens"]):
        assert logit_len == int(ex.input_ids.numel())
    assert model.config.use_cache is True  # restored
    assert len(out["sample_ids"]) == len(cost_world["examples"])


def test_topk_worker_never_loads_teacher_model(cost_world, monkeypatch):
    from mix_bit import cost_search

    teacher_loads = {"n": 0}

    def _boom(*_args, **_kwargs):
        teacher_loads["n"] += 1
        raise AssertionError("teacher model must not be loaded in teacher_topk worker")

    monkeypatch.setattr(cost_search, "load_teacher_model", _boom)

    ctx = cost_search.create_cost_worker(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_path=cost_world["dataset_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        cost_run_root=cost_world["cost_run_root"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
        device="cpu",
        batch_size=1,
    )
    assert ctx.teacher_model is None
    assert teacher_loads["n"] == 0
    cost_search.run_category_mode_job(ctx, category="q_proj", mode=CANDIDATE_MODE)
    assert teacher_loads["n"] == 0
    assert ctx.teacher_model is None


def test_exact_worker_loads_one_teacher_per_process(cost_world, monkeypatch):
    from mix_bit import cost_search

    loads = {"n": 0}
    teacher = cost_world["teacher_template"]

    def _load_teacher(*_args, **_kwargs):
        loads["n"] += 1
        return copy.deepcopy(teacher)

    monkeypatch.setattr(cost_search, "load_teacher_model", _load_teacher)

    exact_root = cost_world["cost_run_root"].parent / "exact_full_vocab"
    exact_root.mkdir(parents=True, exist_ok=True)
    ctx = cost_search.create_cost_worker(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_path=cost_world["dataset_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        cost_run_root=exact_root,
        kl_mode=KL_MODE_EXACT_FULL_VOCAB,
        teacher_topk=None,
        teacher_cache=None,
        device="cpu",
        batch_size=1,
    )
    assert loads["n"] == 1
    assert ctx.teacher_model is not None
    cost_search.run_category_mode_job(ctx, category="q_proj", mode=CANDIDATE_MODE)
    cost_search.run_category_mode_job(ctx, category="k_proj", mode=CANDIDATE_MODE)
    assert loads["n"] == 1


def test_worker_restores_baseline_module_after_each_job(cost_world):
    from mix_bit import cost_search

    ctx = cost_search.create_cost_worker(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_path=cost_world["dataset_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        cost_run_root=cost_world["cost_run_root"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
        device="cpu",
        batch_size=1,
    )
    originals = {
        name: ctx.model.get_submodule(name)
        for name, mode in ctx.assignments.items()
        if mode == BASELINE_MODE
    }
    cost_search.run_category_mode_job(ctx, category="q_proj", mode=CANDIDATE_MODE)
    for name, module in originals.items():
        assert ctx.model.get_submodule(name) is module


def test_worker_rejects_metric_or_provenance_mismatch(cost_world):
    from mix_bit import cost_search

    with pytest.raises(ValueError, match="teacher_topk|teacher_cache|kl_mode"):
        cost_search.create_cost_worker(
            resolved=cost_world["resolved"],
            inventory=cost_world["inventory"],
            pool_index=cost_world["pool_index"],
            baseline_overlay_path=cost_world["overlay_path"],
            dataset_path=cost_world["dataset_path"],
            dataset_manifest_path=cost_world["manifest_path"],
            cost_run_root=cost_world["cost_run_root"],
            kl_mode=KL_MODE_TEACHER_TOPK,
            teacher_topk=None,
            teacher_cache=cost_world["cache_dir"],
            device="cpu",
            batch_size=1,
        )

    bad_manifest = json.loads(cost_world["manifest_path"].read_text(encoding="utf-8"))
    bad_manifest["run_config_sha256"] = "0" * 64
    bad_path = cost_world["tmp_path"] / "bad_manifest.json"
    bad_path.write_text(json.dumps(bad_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch|run_config"):
        cost_search.create_cost_worker(
            resolved=cost_world["resolved"],
            inventory=cost_world["inventory"],
            pool_index=cost_world["pool_index"],
            baseline_overlay_path=cost_world["overlay_path"],
            dataset_path=cost_world["dataset_path"],
            dataset_manifest_path=bad_path,
            cost_run_root=cost_world["cost_run_root"],
            kl_mode=KL_MODE_TEACHER_TOPK,
            teacher_topk=4,
            teacher_cache=cost_world["cache_dir"],
            device="cpu",
            batch_size=1,
        )


def test_per_sample_npz_and_row_json_are_written_atomically(cost_world, monkeypatch):
    from mix_bit import cost_search

    replace_targets: list[str] = []
    real_replace = cost_search.os.replace

    def _spy_replace(src, dst):
        replace_targets.append(str(dst))
        assert str(src).endswith(".tmp") or Path(src).name.endswith(".tmp")
        return real_replace(src, dst)

    monkeypatch.setattr(cost_search.os, "replace", _spy_replace)

    ctx = cost_search.create_cost_worker(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_path=cost_world["dataset_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        cost_run_root=cost_world["cost_run_root"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
        device="cpu",
        batch_size=1,
    )
    rows = cost_search.run_category_mode_job(ctx, category="q_proj", mode=CANDIDATE_MODE)
    assert rows
    for row in rows:
        npz = Path(row["per_sample_file"])
        json_path = Path(cost_world["cost_run_root"]) / "rows" / f"{cost_search.module_safe_name(row['module_name'])}__{row['mode']}.json"
        assert npz.is_file()
        assert json_path.is_file()
        assert str(npz) in replace_targets
        assert str(json_path) in replace_targets
        # NPZ written before JSON for each module.
        assert replace_targets.index(str(npz)) < replace_targets.index(str(json_path))
        assert row["per_sample_sha256"] == sha256_file(npz)
        assert row["status"] == "complete"
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["per_sample_sha256"] == row["per_sample_sha256"]


def test_baseline_mode_cost_is_exact_zero_after_equivalence_check(cost_world):
    from mix_bit import cost_search

    ctx = cost_search.create_cost_worker(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_path=cost_world["dataset_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        cost_run_root=cost_world["cost_run_root"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
        device="cpu",
        batch_size=1,
    )
    audit = cost_search.audit_baseline_self_swap(ctx)
    assert audit["passed"] is True
    audit_path = Path(cost_world["cost_run_root"]) / "baseline_self_swap_audit.json"
    assert audit_path.is_file()
    rows = cost_search.write_baseline_mode_zero_rows(ctx, audit=audit)
    assert rows
    assert len(rows) == len(cost_world["inventory"].targets)
    for row in rows:
        assert row["mode"] == BASELINE_MODE
        assert row["mean_delta_kl"] == 0.0
        assert row["std_delta_kl"] == 0.0
        assert row["standard_error_delta_kl"] == 0.0
        assert row["status"] == "complete"
        npz = np.load(row["per_sample_file"])
        assert np.allclose(npz["delta_kl"], 0.0)
        assert np.allclose(npz["candidate_kl"], npz["baseline_kl"])


def test_failed_self_swap_still_writes_audit_then_aborts(cost_world, monkeypatch):
    from mix_bit import cost_search

    ctx = cost_search.create_cost_worker(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_path=cost_world["dataset_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        cost_run_root=cost_world["cost_run_root"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
        device="cpu",
        batch_size=1,
    )
    audit_path = Path(cost_world["cost_run_root"]) / "baseline_self_swap_audit.json"
    assert not audit_path.exists()

    original_short = cost_search._short_batch_logits
    call_n = {"n": 0}

    def _biased_short(model, *, seed: int = 0):
        logits = original_short(model, seed=seed)
        call_n["n"] += 1
        # First call is the resident reference; poison the first swapped evaluation.
        if call_n["n"] == 2:
            return logits + 10.0
        return logits

    monkeypatch.setattr(cost_search, "_short_batch_logits", _biased_short)

    with pytest.raises(ValueError, match="self-swap audit failed"):
        cost_search.audit_baseline_self_swap(ctx)

    assert audit_path.is_file()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["module_count"] >= 1
    failed = [m for m in payload["modules"] if not m["passed"]]
    assert failed
    assert failed[0]["max_abs_error"] > 1e-4
    assert "max_rel_error" in failed[0]
    assert failed[0]["module_name"]

    with pytest.raises(ValueError, match="self-swap audit did not pass"):
        cost_search.write_baseline_mode_zero_rows(ctx, audit=payload)


# ---------------------------------------------------------------------------
# Task 9: multi-GPU resumable cost search and table finalization
# ---------------------------------------------------------------------------


def _task9_counts(inventory, modes):
    from mix_bit.cost_table import compute_search_counts

    return compute_search_counts(
        category_count=len(inventory.category_order),
        target_linear_count=len(inventory.targets),
        mode_count=len(modes),
    )


def _write_fake_complete_row(
    cost_run_root: Path,
    *,
    module_name: str,
    mode: str,
    category: str,
    module_suffix: str,
    block_index: int,
    sample_ids: np.ndarray,
    provenance: dict[str, Any],
    mean_delta: float = 0.1,
) -> dict[str, Any]:
    from mix_bit.cost_search import module_safe_name, summarize_paired_deltas, write_json_atomic, write_npz_atomic

    n = int(sample_ids.size)
    baseline_kl = np.linspace(0.1, 0.2, n, dtype=np.float64)
    delta = np.full(n, mean_delta, dtype=np.float64)
    candidate_kl = baseline_kl + delta
    stats = summarize_paired_deltas(delta)
    safe = module_safe_name(module_name)
    stem = f"{safe}__{mode}"
    npz_path = cost_run_root / "per_sample" / f"{stem}.npz"
    row_path = cost_run_root / "rows" / f"{stem}.json"
    digest = write_npz_atomic(
        npz_path,
        sample_ids=sample_ids.astype(np.int64),
        baseline_kl=baseline_kl,
        candidate_kl=candidate_kl,
        delta_kl=delta,
        kl_mode=np.asarray(provenance["kl_mode"]),
        metric_name=np.asarray(provenance["metric_name"]),
        teacher_topk=np.int64(
            -1 if provenance.get("teacher_topk") is None else int(provenance["teacher_topk"])
        ),
        module_name=np.asarray(module_name),
        mode=np.asarray(mode),
    )
    row = {
        "module_name": module_name,
        "category": category,
        "module_suffix": module_suffix,
        "block_index": int(block_index),
        "mode": mode,
        "nominal_bit": 1.0,
        "param_count": 64,
        "kl_mode": provenance["kl_mode"],
        "metric_name": provenance["metric_name"],
        "teacher_topk": provenance.get("teacher_topk"),
        "sample_count": n,
        "baseline_kl_mean": float(baseline_kl.mean()),
        "candidate_kl_mean": float(candidate_kl.mean()),
        "mean_delta_kl": stats["mean_delta_kl"],
        "std_delta_kl": stats["std_delta_kl"],
        "standard_error_delta_kl": stats["standard_error_delta_kl"],
        "run_config_sha256": provenance["run_config_sha256"],
        "model_inventory_sha256": provenance["model_inventory_sha256"],
        "candidate_manifest_sha256": provenance["candidate_manifest_sha256"],
        "calibration_manifest_sha256": provenance["calibration_manifest_sha256"],
        "baseline_overlay_sha256": provenance["baseline_overlay_sha256"],
        "teacher_cache_index_sha256": provenance.get("teacher_cache_index_sha256", ""),
        "source_compact_state": "fake.pt",
        "source_compact_state_sha256": "a" * 64,
        "per_sample_file": str(npz_path.resolve()),
        "per_sample_sha256": digest,
        "status": "complete",
    }
    write_json_atomic(row_path, row)
    return row


def _task9_provenance(cost_world, *, kl_mode=KL_MODE_TEACHER_TOPK, teacher_topk=4):
    from mix_bit.cost_search import load_teacher_cache_for_worker
    from mix_bit.schema import sha256_file

    cache_sha = ""
    if kl_mode == KL_MODE_TEACHER_TOPK:
        cache_sha = load_teacher_cache_for_worker(cost_world["cache_dir"]).index_sha256
    return {
        "kl_mode": kl_mode,
        "metric_name": (
            METRIC_NAME_TEACHER_TOPK if kl_mode == KL_MODE_TEACHER_TOPK else METRIC_NAME_EXACT_FULL_VOCAB
        ),
        "teacher_topk": teacher_topk if kl_mode == KL_MODE_TEACHER_TOPK else None,
        "run_config_sha256": cost_world["resolved"].run_config_sha256,
        "model_inventory_sha256": cost_world["inventory"].fingerprint_sha256,
        "candidate_manifest_sha256": sha256_file(cost_world["pool_index"].manifest_path),
        "calibration_manifest_sha256": sha256_file(cost_world["manifest_path"]),
        "baseline_overlay_sha256": sha256_file(cost_world["overlay_path"]),
        "teacher_cache_index_sha256": cache_sha,
    }


def test_job_count_is_categories_times_nonbaseline_modes(cost_world):
    counts = _task9_counts(cost_world["inventory"], cost_world["resolved"].config.candidate_space.modes)
    c = len(cost_world["inventory"].category_order)
    r = len(cost_world["resolved"].config.candidate_space.modes)
    assert counts["source_job_count"] == c * (r - 1)
    assert counts["C"] == c
    assert counts["R"] == r


def test_module_evaluation_count_is_inventory_targets_times_nonbaseline_modes(cost_world):
    counts = _task9_counts(cost_world["inventory"], cost_world["resolved"].config.candidate_space.modes)
    l = len(cost_world["inventory"].targets)
    r = len(cost_world["resolved"].config.candidate_space.modes)
    assert counts["non_baseline_module_evaluation_count"] == l * (r - 1)
    assert counts["L"] == l


def test_complete_row_count_is_inventory_targets_times_all_modes(cost_world):
    counts = _task9_counts(cost_world["inventory"], cost_world["resolved"].config.candidate_space.modes)
    l = len(cost_world["inventory"].targets)
    r = len(cost_world["resolved"].config.candidate_space.modes)
    assert counts["complete_row_count"] == l * r


def test_qwen3_regression_counts_are_28_1008_1260():
    from mix_bit.candidate_space import load_candidate_space
    from mix_bit.cost_table import compute_search_counts
    from mix_bit.model_inventory import load_model_inventory
    from mix_bit.schema import load_model_profile

    repo = Path(__file__).resolve().parents[2]
    profile = load_model_profile(str(repo / "mix_bit/configs/models/qwen3_8b.json"))
    space = load_candidate_space(str(repo / "mix_bit/configs/candidate_spaces/vae_1to3bit.json"))
    inventory = load_model_inventory(str(repo / ".result/mix_bit/qwen3_8b/model_inventory.json"))
    counts = compute_search_counts(
        category_count=len(inventory.category_order),
        target_linear_count=len(inventory.targets),
        mode_count=len(space.modes),
    )
    # Profile regression only — must not drive generic control flow.
    assert profile.regression_expectations["category_count"] == 7
    assert profile.regression_expectations["target_linear_count"] == 252
    assert counts["source_job_count"] == 28
    assert counts["non_baseline_module_evaluation_count"] == 1008
    assert counts["complete_row_count"] == 1260
    assert counts["source_job_count"] == counts["C"] * (counts["R"] - 1)
    assert counts["non_baseline_module_evaluation_count"] == counts["L"] * (counts["R"] - 1)
    assert counts["complete_row_count"] == counts["L"] * counts["R"]


def test_job_partition_is_deterministic(cost_world):
    from mix_bit.cost_table import plan_cost_jobs

    kwargs = dict(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        cost_run_root=cost_world["cost_run_root"],
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
    )
    a = plan_cost_jobs(**kwargs)
    b = plan_cost_jobs(**kwargs)
    assert json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(
        b, sort_keys=True, separators=(",", ":")
    )
    assert len(a["jobs"]) == 2  # C*(R-1) for toy
    job_ids = [job["job_id"] for job in a["jobs"]]
    assert job_ids == sorted(job_ids) or job_ids == [
        f"{cat}__{CANDIDATE_MODE}" for cat in cost_world["inventory"].category_order
    ]
    # Ordered by category_order then mode order.
    assert [job["category"] for job in a["jobs"]] == list(cost_world["inventory"].category_order)


def test_completed_atomic_rows_are_skipped_on_resume(cost_world):
    from mix_bit.cost_table import pending_jobs, plan_cost_jobs

    manifest = plan_cost_jobs(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        cost_run_root=cost_world["cost_run_root"],
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
    )
    provenance = _task9_provenance(cost_world)
    sample_ids = np.array([ex.sample_id for ex in cost_world["examples"]], dtype=np.int64)
    first_job = manifest["jobs"][0]
    for module_name in first_job["module_names"]:
        target = next(t for t in cost_world["inventory"].targets if t.module_name == module_name)
        _write_fake_complete_row(
            cost_world["cost_run_root"],
            module_name=module_name,
            mode=first_job["mode"],
            category=target.category,
            module_suffix=target.module_suffix,
            block_index=target.block_index,
            sample_ids=sample_ids,
            provenance=provenance,
        )
    pending = pending_jobs(manifest, cost_world["cost_run_root"], expected_provenance=provenance)
    pending_ids = {job["job_id"] for job in pending}
    assert first_job["job_id"] not in pending_ids
    assert len(pending) == len(manifest["jobs"]) - 1


def test_incomplete_row_or_npz_is_recomputed(cost_world):
    from mix_bit.cost_search import module_safe_name
    from mix_bit.cost_table import is_atomic_row_complete, pending_jobs, plan_cost_jobs

    manifest = plan_cost_jobs(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        cost_run_root=cost_world["cost_run_root"],
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
    )
    provenance = _task9_provenance(cost_world)
    sample_ids = np.array([ex.sample_id for ex in cost_world["examples"]], dtype=np.int64)
    job = manifest["jobs"][0]
    module_name = job["module_names"][0]
    target = next(t for t in cost_world["inventory"].targets if t.module_name == module_name)
    row = _write_fake_complete_row(
        cost_world["cost_run_root"],
        module_name=module_name,
        mode=job["mode"],
        category=target.category,
        module_suffix=target.module_suffix,
        block_index=target.block_index,
        sample_ids=sample_ids,
        provenance=provenance,
    )
    # Corrupt NPZ digest in row JSON → incomplete.
    row_path = Path(cost_world["cost_run_root"]) / "rows" / f"{module_safe_name(module_name)}__{job['mode']}.json"
    payload = json.loads(row_path.read_text(encoding="utf-8"))
    payload["per_sample_sha256"] = "0" * 64
    row_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert not is_atomic_row_complete(
        cost_world["cost_run_root"],
        module_name=module_name,
        mode=job["mode"],
        expected_provenance=provenance,
        baseline_sample_ids=sample_ids,
    )
    # Remaining modules of the job are still missing → job pending.
    pending = pending_jobs(manifest, cost_world["cost_run_root"], expected_provenance=provenance)
    assert any(p["job_id"] == job["job_id"] for p in pending)
    assert row["status"] == "complete"  # on-disk status alone is not enough


def test_finalizer_rejects_duplicate_module_mode_rows(cost_world):
    from mix_bit.cost_table import finalize_cost_table

    provenance = _task9_provenance(cost_world)
    sample_ids = np.array([ex.sample_id for ex in cost_world["examples"]], dtype=np.int64)
    target = cost_world["inventory"].targets[0]
    rows = []
    for _ in range(2):
        rows.append(
            {
                "module_name": target.module_name,
                "category": target.category,
                "module_suffix": target.module_suffix,
                "block_index": target.block_index,
                "mode": CANDIDATE_MODE,
                "nominal_bit": 1.0,
                "param_count": int(target.param_count),
                "mean_delta_kl": 0.1,
                "std_delta_kl": 0.0,
                "standard_error_delta_kl": 0.0,
                "kl_mode": provenance["kl_mode"],
                "metric_name": provenance["metric_name"],
                "teacher_topk": provenance["teacher_topk"],
                **{k: provenance[k] for k in (
                    "run_config_sha256",
                    "model_inventory_sha256",
                    "candidate_manifest_sha256",
                    "calibration_manifest_sha256",
                    "baseline_overlay_sha256",
                    "teacher_cache_index_sha256",
                )},
            }
        )
    with pytest.raises(ValueError, match="duplicate"):
        finalize_cost_table(
            rows=rows,
            resolved=cost_world["resolved"],
            inventory=cost_world["inventory"],
            pool_index=cost_world["pool_index"],
            cost_run_root=cost_world["cost_run_root"],
            expected_provenance=provenance,
            self_swap_audit={"passed": True, "audit_sha256": "b" * 64},
            source_job_count=2,
            baseline_kl_mean=0.0,
        )


def test_finalizer_requires_inventory_targets_times_modes(cost_world):
    from mix_bit.cost_table import finalize_cost_table

    provenance = _task9_provenance(cost_world)
    with pytest.raises(ValueError, match="complete_row_count|expected.*rows|L \\* R"):
        finalize_cost_table(
            rows=[],
            resolved=cost_world["resolved"],
            inventory=cost_world["inventory"],
            pool_index=cost_world["pool_index"],
            cost_run_root=cost_world["cost_run_root"],
            expected_provenance=provenance,
            self_swap_audit={"passed": True, "audit_sha256": "b" * 64},
            source_job_count=2,
            baseline_kl_mean=0.0,
        )


def test_finalizer_rejects_inventory_metadata_mismatch(cost_world):
    from mix_bit.cost_table import finalize_cost_table

    provenance = _task9_provenance(cost_world)
    inventory = cost_world["inventory"]
    modes = [m.name for m in cost_world["resolved"].config.candidate_space.modes]
    rows = []
    for target in inventory.targets:
        for mode in modes:
            rows.append(
                {
                    "module_name": target.module_name,
                    "category": target.category,
                    "module_suffix": target.module_suffix,
                    "block_index": int(target.block_index),
                    "mode": mode,
                    "nominal_bit": 1.0 if mode == CANDIDATE_MODE else 2.0,
                    "param_count": int(target.param_count),
                    "mean_delta_kl": 0.0 if mode == BASELINE_MODE else 0.1,
                    "std_delta_kl": 0.0,
                    "standard_error_delta_kl": 0.0,
                    "kl_mode": provenance["kl_mode"],
                    "metric_name": provenance["metric_name"],
                    "teacher_topk": provenance["teacher_topk"],
                    **{
                        k: provenance[k]
                        for k in (
                            "run_config_sha256",
                            "model_inventory_sha256",
                            "candidate_manifest_sha256",
                            "calibration_manifest_sha256",
                            "baseline_overlay_sha256",
                            "teacher_cache_index_sha256",
                        )
                    },
                }
            )
    # Corrupt one row's inventory metadata.
    rows[0]["category"] = "not_a_real_category"
    rows[0]["param_count"] = int(rows[0]["param_count"]) + 7
    with pytest.raises(ValueError, match="inventory|metadata|category|param_count"):
        finalize_cost_table(
            rows=rows,
            resolved=cost_world["resolved"],
            inventory=inventory,
            pool_index=cost_world["pool_index"],
            cost_run_root=cost_world["cost_run_root"],
            expected_provenance=provenance,
            self_swap_audit={"passed": True, "audit_sha256": "b" * 64},
            source_job_count=2,
            baseline_kl_mean=0.0,
        )


def test_topk_run_requires_one_consistent_k_and_cache_hash(cost_world):
    from mix_bit.cost_table import validate_cost_run_arguments

    with pytest.raises(ValueError, match="teacher_topk|teacher_cache"):
        validate_cost_run_arguments(
            kl_mode=KL_MODE_TEACHER_TOPK,
            teacher_topk=None,
            teacher_cache=cost_world["cache_dir"],
        )
    # Existing jobs.json with different K must reject resume.
    from mix_bit.cost_table import persist_jobs_manifest, plan_cost_jobs

    manifest = plan_cost_jobs(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        cost_run_root=cost_world["cost_run_root"],
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
    )
    persist_jobs_manifest(cost_world["cost_run_root"], manifest)
    altered = dict(manifest)
    altered["teacher_topk"] = 8
    with pytest.raises(ValueError, match="jobs.json|byte-for-byte|manifest mismatch|differ"):
        persist_jobs_manifest(cost_world["cost_run_root"], altered)


def test_exact_run_rejects_teacher_cache(cost_world):
    from mix_bit.cost_table import validate_cost_run_arguments

    with pytest.raises(ValueError, match="teacher_cache"):
        validate_cost_run_arguments(
            kl_mode=KL_MODE_EXACT_FULL_VOCAB,
            teacher_topk=None,
            teacher_cache=cost_world["cache_dir"],
        )


def test_resume_rejects_different_metric_or_provenance(cost_world):
    from mix_bit.cost_table import persist_jobs_manifest, plan_cost_jobs

    root = cost_world["cost_run_root"]
    manifest = plan_cost_jobs(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        cost_run_root=root,
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
    )
    persist_jobs_manifest(root, manifest)
    other = plan_cost_jobs(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        cost_run_root=root,
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
    )
    other = dict(other)
    other["run_config_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="jobs.json|byte-for-byte|manifest mismatch|differ|provenance"):
        persist_jobs_manifest(root, other)


def test_worker_crash_does_not_mark_job_complete(cost_world, monkeypatch):
    from mix_bit import cost_table

    manifest = cost_table.plan_cost_jobs(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        cost_run_root=cost_world["cost_run_root"],
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
    )
    cost_table.persist_jobs_manifest(cost_world["cost_run_root"], manifest)
    provenance = _task9_provenance(cost_world)

    def _crashing_execute(job, worker_state):
        raise RuntimeError("simulated worker crash")

    monkeypatch.setattr(cost_table, "execute_category_mode_job", _crashing_execute)

    with pytest.raises(RuntimeError, match="worker|crash|simulated"):
        cost_table.run_cost_search_scheduler(
            manifest=manifest,
            resolved=cost_world["resolved"],
            inventory=cost_world["inventory"],
            pool_index=cost_world["pool_index"],
            baseline_overlay_path=cost_world["overlay_path"],
            dataset_path=cost_world["dataset_path"],
            dataset_manifest_path=cost_world["manifest_path"],
            cost_run_root=cost_world["cost_run_root"],
            kl_mode=KL_MODE_TEACHER_TOPK,
            teacher_topk=4,
            teacher_cache=cost_world["cache_dir"],
            gpus=["0"],
            batch_size=1,
            device_override="cpu",
            in_process=True,
        )

    pending = cost_table.pending_jobs(
        manifest, cost_world["cost_run_root"], expected_provenance=provenance
    )
    assert len(pending) == len(manifest["jobs"])
    # Diagnostic logs must not be treated as completion.
    log_dir = Path(cost_world["cost_run_root"]) / "worker_logs"
    if log_dir.is_dir():
        for path in log_dir.glob("*.jsonl"):
            text = path.read_text(encoding="utf-8")
            assert "job_complete" not in text
    # Explicit: no atomic rows written for crashed work.
    rows_dir = Path(cost_world["cost_run_root"]) / "rows"
    if rows_dir.is_dir():
        assert list(rows_dir.glob("*.json")) == []


def test_failure_drain_does_not_block_on_bounded_job_queue():
    """Simulated bounded queue: blocking sentinel put would hang; drain must not."""
    import queue as queue_mod
    import threading

    from mix_bit.cost_table import drain_job_queue_and_stop_workers

    job_queue: queue_mod.Queue = queue_mod.Queue(maxsize=2)
    job_queue.put({"job_id": "queued_a"})
    job_queue.put({"job_id": "queued_b"})

    class _FakeProc:
        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout=None) -> None:
            self._alive = False

        def terminate(self) -> None:
            self._alive = False

    procs = [_FakeProc(), _FakeProc()]
    done = {"ok": False}

    def _run() -> None:
        drain_job_queue_and_stop_workers(job_queue, procs, join_timeout=1.0)
        done["ok"] = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=3.0)
    assert not thread.is_alive(), "drain_job_queue_and_stop_workers hung on full bounded queue"
    assert done["ok"] is True
    assert all(not p.is_alive() for p in procs)
    # Queued jobs cleared; only sentinels (or empty after consumers) remain acceptable.
    leftover = []
    while True:
        try:
            leftover.append(job_queue.get_nowait())
        except queue_mod.Empty:
            break
    assert all(item is None for item in leftover)


# ---------------------------------------------------------------------------
# Task 5: cost worker must not transfer full student logits to CPU
# ---------------------------------------------------------------------------


def test_evaluate_student_per_sample_kl_source_does_not_cpu_full_logits():
    """Guard against full-logits CPU transfer regression in the cost worker."""
    import inspect

    from mix_bit.cost_search import evaluate_student_per_sample_kl

    source = inspect.getsource(evaluate_student_per_sample_kl)
    assert "shifted_student.detach().cpu()" not in source
    assert "shifted_student.cpu()" not in source
    # The production path must pass the on-device logits into the KL helper.
    assert "shifted_student_logits=shifted_student" in source or (
        "shifted_student_logits=shifted_student.detach()" in source
    )


# ---------------------------------------------------------------------------
# Task 7: spawn worker startup and runtime fail-fast behavior
# ---------------------------------------------------------------------------

import queue as _queue_mod
import threading as _threading_mod


def test_process_start_inherits_requested_cuda_visibility_and_restores_parent(monkeypatch):
    from mix_bit import cost_table

    seen: list[str | None] = []

    class _RecordingProc:
        def start(self) -> None:
            seen.append(os.environ.get("CUDA_VISIBLE_DEVICES"))

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    cost_table._start_process_on_physical_gpu(_RecordingProc(), "3")

    assert seen == ["3"]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "6,7"


def test_process_start_removes_temporary_cuda_visibility_when_parent_had_none(monkeypatch):
    from mix_bit import cost_table

    seen: list[str | None] = []

    class _RecordingProc:
        def start(self) -> None:
            seen.append(os.environ.get("CUDA_VISIBLE_DEVICES"))

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cost_table._start_process_on_physical_gpu(_RecordingProc(), "2")

    assert seen == ["2"]
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_isolated_cuda_init_rejects_more_than_one_visible_device(monkeypatch):
    from mix_bit import cost_table

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(RuntimeError, match="exactly one device"):
        cost_table._initialize_isolated_cuda_device(physical_gpu="3")


def test_isolated_cuda_init_uses_logical_cuda_zero(monkeypatch):
    from mix_bit import cost_table

    selected: list[int] = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda idx: selected.append(int(idx)))

    device, visible_count = cost_table._initialize_isolated_cuda_device(physical_gpu="2")

    assert device == "cuda:0"
    assert visible_count == 1
    assert selected == [0]


class _FakeProc:
    """Minimal mp.Process stand-in: pid/exitcode/is_alive/terminate/join."""

    def __init__(self, *, pid: int, exitcode: int | None = None, alive: bool = True) -> None:
        self._pid = int(pid)
        self._exitcode = exitcode
        self._alive = bool(alive)
        self._terminated = False
        self._joined = False
        self._started = False

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def exitcode(self) -> int | None:
        return self._exitcode

    def is_alive(self) -> bool:
        return self._alive

    def start(self) -> None:
        self._started = True

    def terminate(self) -> None:
        self._alive = False
        self._terminated = True
        if self._exitcode is None:
            self._exitcode = -15

    def join(self, timeout: float | None = None) -> None:
        self._joined = True
        self._alive = False
        if self._exitcode is None:
            self._exitcode = 0


class _FakeQueue:
    """Minimal mp.Queue stand-in: get(timeout=) raises queue.Empty when empty."""

    def __init__(self, messages: list | None = None) -> None:
        self._messages: list[Any] = list(messages or [])

    def get(self, *, timeout: float | None = None) -> Any:
        if self._messages:
            return self._messages.pop(0)
        raise _queue_mod.Empty()

    def put(self, item: Any) -> None:
        self._messages.append(item)

    def put_nowait(self, item: Any) -> None:
        self.put(item)

    def get_nowait(self) -> Any:
        if self._messages:
            return self._messages.pop(0)
        raise _queue_mod.Empty()

    def qsize(self) -> int:
        return len(self._messages)


class _ScriptedQueue:
    """Queue that serves a fixed message list once, then raises Empty.

    On the first Empty after scripted messages are exhausted, an optional
    callback fires (used to simulate a worker dying once runtime polling
    begins).
    """

    def __init__(self, messages: list[Any], on_empty=None) -> None:
        self._messages: list[Any] = list(messages)
        self._on_empty = on_empty
        self._fired = False

    def get(self, *, timeout: float | None = None) -> Any:
        if self._messages:
            return self._messages.pop(0)
        if self._on_empty is not None and not self._fired:
            self._fired = True
            self._on_empty()
        raise _queue_mod.Empty()

    def put(self, item: Any) -> None:
        pass

    def put_nowait(self, item: Any) -> None:
        pass

    def get_nowait(self) -> Any:
        raise _queue_mod.Empty()

    def qsize(self) -> int:
        return 0


def _baseline_wait_helper():
    from mix_bit.cost_table import _wait_for_single_process_message

    return _wait_for_single_process_message


def test_baseline_wait_fails_when_child_exits_without_message():
    wait = _baseline_wait_helper()

    proc = _FakeProc(pid=123, exitcode=1, alive=False)
    q = _FakeQueue(messages=[])

    with pytest.raises(RuntimeError, match="123.*exitcode.*1"):
        wait(
            process=proc,
            result_queue=q,
            expected_type="baseline_ready",
            timeout_seconds=0.01,
            label="baseline",
        )


def test_baseline_wait_surfaces_failure_traceback():
    wait = _baseline_wait_helper()

    proc = _FakeProc(pid=10, alive=True)
    failure_msg = {
        "type": "failure",
        "error": "RuntimeError: boom",
        "traceback": "Traceback (most recent call last):\n  boom",
    }
    q = _FakeQueue(messages=[failure_msg])

    with pytest.raises(RuntimeError, match="boom") as exc_info:
        wait(
            process=proc,
            result_queue=q,
            expected_type="baseline_ready",
            timeout_seconds=0.01,
            label="baseline",
        )
    assert "Traceback" in str(exc_info.value)


def test_baseline_wait_times_out_and_terminates_child():
    wait = _baseline_wait_helper()

    proc = _FakeProc(pid=42, alive=True)
    q = _FakeQueue(messages=[])

    with pytest.raises(TimeoutError, match="baseline"):
        wait(
            process=proc,
            result_queue=q,
            expected_type="baseline_ready",
            timeout_seconds=0.01,
            label="baseline",
        )
    assert proc._terminated is True
    assert proc.is_alive() is False


def test_baseline_wait_accepts_exact_ready_message():
    wait = _baseline_wait_helper()

    proc = _FakeProc(pid=7, alive=True)
    ready_msg = {"type": "baseline_ready", "baseline_per_sample_path": "/tmp/x.npz"}
    q = _FakeQueue(messages=[ready_msg])

    msg = wait(
        process=proc,
        result_queue=q,
        expected_type="baseline_ready",
        timeout_seconds=0.01,
        label="baseline",
    )
    assert msg is ready_msg
    assert proc._joined is True
    assert proc.is_alive() is False
    assert proc.exitcode == 0


def test_baseline_wait_fails_when_process_exits_nonzero_after_ready():
    wait = _baseline_wait_helper()

    proc = _FakeProc(pid=9, alive=True)
    proc._exitcode = None  # will be set to nonzero by join override below

    class _NonZeroJoinProc(_FakeProc):
        def join(self, timeout: float | None = None) -> None:
            self._joined = True
            self._alive = False
            self._exitcode = 3

    proc = _NonZeroJoinProc(pid=9, alive=True)
    ready_msg = {"type": "baseline_ready", "baseline_per_sample_path": "/tmp/y.npz"}
    q = _FakeQueue(messages=[ready_msg])

    with pytest.raises(RuntimeError, match="exitcode.*3"):
        wait(
            process=proc,
            result_queue=q,
            expected_type="baseline_ready",
            timeout_seconds=0.01,
            label="baseline",
        )


def _worker_ready_helper():
    from mix_bit.cost_table import _wait_for_workers_ready

    return _wait_for_workers_ready


def test_worker_ready_wait_rejects_duplicate_logical_id():
    wait = _worker_ready_helper()

    procs = [_FakeProc(pid=1, alive=True), _FakeProc(pid=2, alive=True)]
    q = _FakeQueue(
        messages=[
            {"type": "ready", "logical_id": 0, "physical_gpu": "0"},
            {"type": "ready", "logical_id": 0, "physical_gpu": "0"},
        ]
    )

    with pytest.raises(RuntimeError, match="duplicate.*logical_id.*0"):
        wait(processes=procs, result_queue=q, timeout_seconds=0.01)


def test_worker_ready_wait_fails_if_one_worker_dies():
    wait = _worker_ready_helper()

    procs = [_FakeProc(pid=1, alive=False, exitcode=1), _FakeProc(pid=2, alive=True)]
    q = _FakeQueue(messages=[])

    with pytest.raises(RuntimeError, match="died.*pid=1.*exitcode=1"):
        wait(processes=procs, result_queue=q, timeout_seconds=0.01)


def test_worker_ready_wait_times_out_and_terminates_all():
    wait = _worker_ready_helper()

    procs = [_FakeProc(pid=1, alive=True), _FakeProc(pid=2, alive=True)]
    q = _FakeQueue(messages=[])

    with pytest.raises(TimeoutError, match="ready"):
        wait(processes=procs, result_queue=q, timeout_seconds=0.01)
    assert all(p._terminated for p in procs)
    assert all(not p.is_alive() for p in procs)


def test_worker_ready_wait_accepts_all_unique_workers():
    wait = _worker_ready_helper()

    procs = [_FakeProc(pid=1, alive=True), _FakeProc(pid=2, alive=True)]
    q = _FakeQueue(
        messages=[
            {"type": "ready", "logical_id": 0, "physical_gpu": "0"},
            {"type": "ready", "logical_id": 1, "physical_gpu": "1"},
        ]
    )

    wait(processes=procs, result_queue=q, timeout_seconds=0.01)


def test_worker_ready_wait_surfaces_failure_message():
    wait = _worker_ready_helper()

    procs = [_FakeProc(pid=1, alive=True), _FakeProc(pid=2, alive=True)]
    q = _FakeQueue(
        messages=[
            {"type": "ready", "logical_id": 0, "physical_gpu": "0"},
            {
                "type": "failure",
                "logical_id": 1,
                "physical_gpu": "1",
                "error": "RuntimeError: load failed",
                "traceback": "Traceback: load failed",
            },
        ]
    )

    with pytest.raises(RuntimeError, match="load failed") as exc_info:
        wait(processes=procs, result_queue=q, timeout_seconds=0.01)
    assert "Traceback" in str(exc_info.value)


def test_runtime_partial_death_fails_and_does_not_keep_polling(cost_world, monkeypatch):
    """One worker dies mid-flight while another stays alive: scheduler must fail."""
    from mix_bit import cost_table

    # Pre-create baseline so _ensure_baseline_per_sample_spawn returns early.
    cost_root = Path(cost_world["cost_run_root"])
    sample_ids = np.array([ex.sample_id for ex in cost_world["examples"]], dtype=np.int64)
    baseline_kl = np.zeros(sample_ids.shape, dtype=np.float64)
    np.savez(
        cost_root / "baseline_per_sample.npz",
        sample_ids=sample_ids,
        baseline_kl=baseline_kl,
    )

    manifest = cost_table.plan_cost_jobs(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        cost_run_root=cost_root,
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
    )
    cost_table.persist_jobs_manifest(cost_root, manifest)

    procs = [_FakeProc(pid=100, alive=True), _FakeProc(pid=200, alive=True)]

    def _on_first_empty():
        # Worker 0 crashes once runtime polling starts.
        procs[0]._alive = False
        procs[0]._exitcode = 137

    # Two ready messages (one per worker), then Empty forever.
    result_queue = _ScriptedQueue(
        messages=[
            {"type": "ready", "logical_id": 0, "physical_gpu": "0"},
            {"type": "ready", "logical_id": 1, "physical_gpu": "1"},
        ],
        on_empty=_on_first_empty,
    )

    class _FakeJobQueue:
        def __init__(self) -> None:
            self._items: list = []

        def put(self, item) -> None:
            self._items.append(item)

        def put_nowait(self, item) -> None:
            self._items.append(item)

        def get_nowait(self):
            if self._items:
                return self._items.pop(0)
            raise _queue_mod.Empty()

    class _FakeSpawnCtx:
        def __init__(self) -> None:
            self._proc_idx = 0

        def Queue(self, maxsize: int = 0):
            # First Queue() call is the result_queue (already built above); the
            # second is the job_queue.
            return _FakeJobQueue()

        def Process(self, target=None, args=(), daemon=None):
            proc = procs[self._proc_idx]
            self._proc_idx += 1
            return proc

    fake_ctx = _FakeSpawnCtx()

    def _fake_get_context(name):
        assert name == "spawn"
        return fake_ctx

    # Replace the result_queue that the scheduler would create with our
    # scripted one by intercepting the second Queue() call (job_queue first,
    # result_queue second).
    queue_calls = {"n": 0}

    def _make_queue(maxsize: int = 0):
        queue_calls["n"] += 1
        if queue_calls["n"] == 2:
            return result_queue
        return _FakeJobQueue()

    class _ScriptedSpawnCtx:
        def __init__(self) -> None:
            self._proc_idx = 0

        def Queue(self, maxsize: int = 0):
            return _make_queue(maxsize)

        def Process(self, target=None, args=(), daemon=None):
            proc = procs[self._proc_idx]
            self._proc_idx += 1
            return proc

    monkeypatch.setattr(cost_table.mp, "get_context", lambda name: _ScriptedSpawnCtx())

    with pytest.raises(RuntimeError) as exc_info:
        cost_table.run_cost_search_scheduler(
            manifest=manifest,
            resolved=cost_world["resolved"],
            inventory=cost_world["inventory"],
            pool_index=cost_world["pool_index"],
            baseline_overlay_path=cost_world["overlay_path"],
            dataset_path=cost_world["dataset_path"],
            dataset_manifest_path=cost_world["manifest_path"],
            cost_run_root=cost_root,
            kl_mode=KL_MODE_TEACHER_TOPK,
            teacher_topk=4,
            teacher_cache=cost_world["cache_dir"],
            gpus=["0", "1"],
            batch_size=1,
            inventory_path=str(cost_world["tmp_path"] / "inventory.json"),
        )
    msg = str(exc_info.value)
    assert "pid=100" in msg
    assert "exitcode=137" in msg


def test_startup_failure_terminates_remaining_workers(cost_world, monkeypatch):
    """If one worker dies before ready, the scheduler must drain/terminate the
    still-alive sibling so no GPU-holding daemon lingers."""
    from mix_bit import cost_table

    cost_root = Path(cost_world["cost_run_root"])
    sample_ids = np.array([ex.sample_id for ex in cost_world["examples"]], dtype=np.int64)
    np.savez(
        cost_root / "baseline_per_sample.npz",
        sample_ids=sample_ids,
        baseline_kl=np.zeros(sample_ids.shape, dtype=np.float64),
    )

    manifest = cost_table.plan_cost_jobs(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        cost_run_root=cost_root,
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
    )
    cost_table.persist_jobs_manifest(cost_root, manifest)

    # Worker 0 already dead before ready; worker 1 still alive (must be drained).
    procs = [_FakeProc(pid=11, alive=False, exitcode=1), _FakeProc(pid=22, alive=True)]

    class _FakeJobQueue:
        def __init__(self) -> None:
            self._items: list = []

        def put(self, item) -> None:
            self._items.append(item)

        def put_nowait(self, item) -> None:
            self._items.append(item)

        def get_nowait(self):
            if self._items:
                return self._items.pop(0)
            raise _queue_mod.Empty()

    result_queue = _FakeQueue(messages=[])  # never produces a ready message

    queue_calls = {"n": 0}

    def _make_queue(maxsize: int = 0):
        queue_calls["n"] += 1
        if queue_calls["n"] == 2:
            return result_queue
        return _FakeJobQueue()

    class _ScriptedSpawnCtx:
        def __init__(self) -> None:
            self._proc_idx = 0

        def Queue(self, maxsize: int = 0):
            return _make_queue(maxsize)

        def Process(self, target=None, args=(), daemon=None):
            proc = procs[self._proc_idx]
            self._proc_idx += 1
            return proc

    monkeypatch.setattr(cost_table.mp, "get_context", lambda name: _ScriptedSpawnCtx())

    with pytest.raises(RuntimeError, match="died before ready"):
        cost_table.run_cost_search_scheduler(
            manifest=manifest,
            resolved=cost_world["resolved"],
            inventory=cost_world["inventory"],
            pool_index=cost_world["pool_index"],
            baseline_overlay_path=cost_world["overlay_path"],
            dataset_path=cost_world["dataset_path"],
            dataset_manifest_path=cost_world["manifest_path"],
            cost_run_root=cost_root,
            kl_mode=KL_MODE_TEACHER_TOPK,
            teacher_topk=4,
            teacher_cache=cost_world["cache_dir"],
            gpus=["0", "1"],
            batch_size=1,
            inventory_path=str(cost_world["tmp_path"] / "inventory.json"),
        )

    # The still-alive sibling must have been drained (joined/terminated).
    assert procs[1].is_alive() is False
    assert procs[1]._joined is True


# ---------------------------------------------------------------------------
# Task 8: spawned worker args must preserve the authoritative pool manifest path
# ---------------------------------------------------------------------------


def test_spawned_worker_args_preserve_pool_manifest_path(cost_world, monkeypatch):
    """Baseline init and every worker arg must carry the same absolute pool_manifest_path.

    Exercises the real baseline spawn path (does NOT pre-create baseline_per_sample.npz)
    so the args passed into _baseline_init_process_main are captured too, then asserts
    baseline init_args and every worker_args share one absolute pool_manifest_path.
    """
    from mix_bit import cost_table

    cost_root = Path(cost_world["cost_run_root"])
    baseline_npz = cost_root / "baseline_per_sample.npz"
    # Do NOT pre-create baseline so _ensure_baseline_per_sample_spawn spawns.
    assert not baseline_npz.is_file()
    sample_ids = np.array([ex.sample_id for ex in cost_world["examples"]], dtype=np.int64)

    manifest = cost_table.plan_cost_jobs(
        resolved=cost_world["resolved"],
        inventory=cost_world["inventory"],
        pool_index=cost_world["pool_index"],
        cost_run_root=cost_root,
        baseline_overlay_path=cost_world["overlay_path"],
        dataset_manifest_path=cost_world["manifest_path"],
        kl_mode=KL_MODE_TEACHER_TOPK,
        teacher_topk=4,
        teacher_cache=cost_world["cache_dir"],
    )
    cost_table.persist_jobs_manifest(cost_root, manifest)

    captured_args: list[dict] = []

    class _CapturingJobQueue:
        def __init__(self) -> None:
            self._items: list = []

        def put(self, item) -> None:
            self._items.append(item)

        def put_nowait(self, item) -> None:
            self._items.append(item)

        def get_nowait(self):
            if self._items:
                return self._items.pop(0)
            raise _queue_mod.Empty()

    class _BaselineResultQueue:
        """Serve baseline_ready on first get, materializing the npz as a side effect.

        The fake baseline process never runs, so the parent must create the file
        that _ensure_baseline_per_sample_spawn checks after the ready message.
        """

        def __init__(self) -> None:
            self._served = False

        def get(self, *, timeout: float | None = None):
            if not self._served:
                self._served = True
                np.savez(
                    baseline_npz,
                    sample_ids=sample_ids,
                    baseline_kl=np.zeros(sample_ids.shape, dtype=np.float64),
                )
                return {
                    "type": "baseline_ready",
                    "baseline_per_sample_path": str(baseline_npz),
                }
            raise _queue_mod.Empty()

        def put(self, item) -> None:
            pass

        def put_nowait(self, item) -> None:
            pass

        def get_nowait(self):
            raise _queue_mod.Empty()

    baseline_proc = _FakeProc(pid=99, alive=True)
    worker_procs = [_FakeProc(pid=1, alive=True), _FakeProc(pid=2, alive=True)]

    def _on_first_empty():
        # Once runtime polling starts, kill both workers so the scheduler
        # raises RuntimeError instead of looping forever.
        for p in worker_procs:
            p._alive = False
            p._exitcode = 137

    scheduler_result_queue = _ScriptedQueue(
        messages=[
            {"type": "ready", "logical_id": 0, "physical_gpu": "0"},
            {"type": "ready", "logical_id": 1, "physical_gpu": "1"},
        ],
        on_empty=_on_first_empty,
    )

    # Queue() call order across both spawn contexts (baseline then scheduler):
    #   1) baseline result_queue  2) job_queue  3) scheduler result_queue
    queue_calls = {"n": 0}

    def _make_queue(maxsize: int = 0):
        queue_calls["n"] += 1
        if queue_calls["n"] == 1:
            return _BaselineResultQueue()
        if queue_calls["n"] == 3:
            return scheduler_result_queue
        return _CapturingJobQueue()

    # Process() call order: 1) baseline  2,3) workers
    proc_idx = {"n": 0}

    class _CapturingSpawnCtx:
        def Queue(self, maxsize: int = 0):
            return _make_queue(maxsize)

        def Process(self, target=None, args=(), daemon=None):
            captured_args.append(args[0])
            i = proc_idx["n"]
            proc_idx["n"] += 1
            if i == 0:
                return baseline_proc
            return worker_procs[i - 1]

    shared_ctx = _CapturingSpawnCtx()
    monkeypatch.setattr(cost_table.mp, "get_context", lambda name: shared_ctx)

    with pytest.raises(RuntimeError):
        cost_table.run_cost_search_scheduler(
            manifest=manifest,
            resolved=cost_world["resolved"],
            inventory=cost_world["inventory"],
            pool_index=cost_world["pool_index"],
            baseline_overlay_path=cost_world["overlay_path"],
            dataset_path=cost_world["dataset_path"],
            dataset_manifest_path=cost_world["manifest_path"],
            cost_run_root=cost_root,
            kl_mode=KL_MODE_TEACHER_TOPK,
            teacher_topk=4,
            teacher_cache=cost_world["cache_dir"],
            gpus=["0", "1"],
            batch_size=1,
            inventory_path=str(cost_world["tmp_path"] / "inventory.json"),
        )

    # One baseline init + two workers were spawned and captured.
    assert len(captured_args) == 3
    baseline_args = captured_args[0]
    worker_args_list = captured_args[1:]
    assert "pool_manifest_path" in baseline_args
    assert "pool_manifest_path" in worker_args_list[0]
    assert "pool_manifest_path" in worker_args_list[1]

    expected_manifest = str(Path(cost_world["pool_index"].manifest_path).resolve())
    assert Path(baseline_args["pool_manifest_path"]).resolve() == Path(expected_manifest).resolve()
    for worker_args in worker_args_list:
        assert Path(worker_args["pool_manifest_path"]).resolve() == Path(expected_manifest).resolve()

    # Baseline and all workers share one absolute manifest path.
    paths = {Path(a["pool_manifest_path"]).resolve() for a in captured_args}
    assert len(paths) == 1
    assert next(iter(paths)) == Path(expected_manifest).resolve()


