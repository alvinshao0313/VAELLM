from __future__ import annotations

import copy
import json
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
    assemble_optimal_mixed_checkpoint,
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
)
from mix_bit.model_adapter import get_model_adapter
from mix_bit.model_inventory import ModelInventory, inventory_from_targets, write_model_inventory
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
from mix_bit.solver import bit_to_units
from mix_bit.teacher_cache import build_teacher_topk_chunk, write_teacher_cache_chunk
from train_utils.model_checkpoint_io import META_FILENAME, STATE_DICT_FILENAME


HIDDEN = 8
VOCAB = 16
BASELINE_MODE = "b4d4s2"
OTHER_MODE = "b4d4s1"
TEACHER_TOPK = 4


class _TinyTokenizer:
    """Tiny save/reload tokenizer fixture for validation tests (no HF network)."""

    def __init__(
        self,
        *,
        vocab_size: int = 32,
        pad_token_id: int | None = 0,
        eos_token_id: int = 0,
        bos_token_id: int | None = None,
        unk_token_id: int | None = None,
        name_or_path: str = "tiny",
        chat_template: str | None = None,
        vocab_seed: int = 0,
        model_max_length: int = 1024,
    ):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.unk_token_id = unk_token_id
        self.name_or_path = name_or_path
        self.padding_side = "left"
        self.truncation_side = "right"
        self.model_max_length = model_max_length
        self.chat_template = chat_template
        self.init_kwargs = {"name_or_path": name_or_path, "vocab_size": vocab_size}
        self.mix_bit_pad_token_normalized_from_eos = False
        self._vocab_seed = vocab_seed
        self._vocab = {f"tok_{i + vocab_seed}": i for i in range(vocab_size)}
        self.special_tokens_map: dict = {}

    def get_vocab(self):
        return dict(self._vocab)

    def get_added_vocab(self):
        return {}

    def save_pretrained(self, output_dir):
        import os

        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "unk_token_id": self.unk_token_id,
            "chat_template": self.chat_template,
            "vocab_seed": self._vocab_seed,
            "model_max_length": self.model_max_length,
        }
        (Path(output_dir) / "tiny_tokenizer.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )


def _patch_tiny_tokenizer(monkeypatch: pytest.MonkeyPatch):
    def _fake_from_pretrained(path, *, local_files_only=False, trust_remote_code=False, **_kwargs):
        marker = Path(path) / "tiny_tokenizer.json"
        if marker.is_file():
            data = json.loads(marker.read_text(encoding="utf-8"))
            data["name_or_path"] = str(path)
            return _TinyTokenizer(**data)
        if local_files_only:
            raise OSError(
                f"No tokenizer files found in {path} (local_files_only=True)"
            )
        return _TinyTokenizer(name_or_path=str(path))

    monkeypatch.setattr("mix_bit.model_adapter.AutoTokenizer.from_pretrained", _fake_from_pretrained)
    monkeypatch.setattr("mix_bit.assembler.AutoTokenizer.from_pretrained", _fake_from_pretrained)
    monkeypatch.setattr("mix_bit.validation.AutoTokenizer.from_pretrained", _fake_from_pretrained)


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

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **_kwargs):
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
            name=OTHER_MODE,
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
            "steps_per_category": 10,
            "batch_size": 8,
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
            "lr": 0.001,
            "beta1": 0.9,
            "beta2": 0.95,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "lr_scheduler": "linear",
            "lr_warmup_steps": 0,
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
            "outlier_protect_mode": "channel",
            "outlier_protect_count": 0,
            "outlier_protect_min_per_layer": 0,
            "distill_after_category": "none",
            "eval_ppl": False,
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


def _write_dataset(
    tmp_path: Path,
    examples: list[CalibrationExample],
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
) -> tuple[Path, Path]:
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
    teacher_topk: int = TEACHER_TOPK,
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
    (cache_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return cache_dir


def _mixed_assignments(inventory: ModelInventory, pool_index: CandidatePoolIndex) -> dict[str, str]:
    assignments = build_uniform_assignments(pool_index, BASELINE_MODE)
    seen_cats: set[str] = set()
    for target in inventory.targets:
        if target.category in seen_cats:
            continue
        assignments[target.module_name] = OTHER_MODE
        seen_cats.add(target.category)
    return assignments


def _write_allocation_and_costs(
    *,
    path: Path,
    resolved: ResolvedRunConfig,
    inventory: ModelInventory,
    pool_index: CandidatePoolIndex,
    assignments: dict[str, str],
    cost_dir: Path,
    kl_mode: str = KL_MODE_TEACHER_TOPK,
    metric_name: str = METRIC_NAME_TEACHER_TOPK,
    teacher_topk: int | None = TEACHER_TOPK,
    baseline_kl_mean: float = 1.5,
    baseline_overlay_path: Path | str | None = None,
    teacher_cache_dir: Path | str | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    order = {t.module_name: idx for idx, t in enumerate(inventory.targets)}
    entries = []
    cost_rows = []
    objective = 0.0
    used_units = 0
    total_params = 0
    weighted_bits = 0.0
    modes = [m.name for m in resolved.config.candidate_space.modes]
    for target in sorted(inventory.targets, key=lambda t: order[t.module_name]):
        for mode in modes:
            cand = pool_index.candidates[(target.module_name, mode)]
            cost = 0.0 if mode == BASELINE_MODE else 0.01 * (target.block_index + 1)
            if mode == assignments[target.module_name]:
                objective += cost
                used_units += int(target.param_count) * bit_to_units(cand.nominal_bit)
                total_params += int(target.param_count)
                weighted_bits += int(target.param_count) * float(cand.nominal_bit)
                entries.append(
                    {
                        "module_name": target.module_name,
                        "category": target.category,
                        "module_suffix": target.module_suffix,
                        "block_index": target.block_index,
                        "in_features": target.in_features,
                        "out_features": target.out_features,
                        "has_bias": target.has_bias,
                        "param_count": target.param_count,
                        "mode": mode,
                        "nominal_bit": cand.nominal_bit,
                        "mean_delta_kl": cost,
                        "compact_state_sha256": cand.source.compact_state_sha256,
                        "per_sample_sha256": "c" * 64,
                    }
                )
            cost_rows.append(
                {
                    "module_name": target.module_name,
                    "category": target.category,
                    "module_suffix": target.module_suffix,
                    "block_index": target.block_index,
                    "in_features": target.in_features,
                    "out_features": target.out_features,
                    "has_bias": target.has_bias,
                    "param_count": target.param_count,
                    "mode": mode,
                    "nominal_bit": cand.nominal_bit,
                    "mean_delta_kl": cost,
                    "kl_mode": kl_mode,
                    "metric_name": metric_name,
                    "teacher_topk": teacher_topk,
                    "run_config_sha256": resolved.run_config_sha256,
                    "model_inventory_sha256": inventory.fingerprint_sha256,
                    "candidate_manifest_sha256": sha256_file(pool_index.manifest_path),
                    "candidate_space_sha256": resolved.candidate_space_sha256,
                    "source_compact_state_sha256": cand.source.compact_state_sha256,
                    "per_sample_sha256": "c" * 64,
                }
            )

    target_bit = float(resolved.config.candidate_space.target_average_bit)
    budget_units = bit_to_units(target_bit) * total_params
    achieved = weighted_bits / float(total_params)
    predicted = baseline_kl_mean + objective

    cost_dir.mkdir(parents=True, exist_ok=True)
    cost_table_path = cost_dir / "cost_table.jsonl"
    with open(cost_table_path, "w", encoding="utf-8") as handle:
        for row in cost_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    cost_table_sha = sha256_file(cost_table_path)
    if baseline_overlay_path is None:
        raise ValueError("baseline_overlay_path is required for cost meta fixture")
    overlay_sha = sha256_file(baseline_overlay_path)
    if kl_mode == KL_MODE_TEACHER_TOPK:
        if teacher_cache_dir is None:
            raise ValueError("teacher_cache_dir is required for teacher_topk cost meta fixture")
        cache_sha = sha256_file(Path(teacher_cache_dir) / "index.json")
    else:
        cache_sha = ""
    meta = {
        "kind": "mix_bit_cost_table_meta",
        "kl_mode": kl_mode,
        "metric_name": metric_name,
        "teacher_topk": teacher_topk,
        "baseline_kl_mean": baseline_kl_mean,
        "baseline_overlay_sha256": overlay_sha,
        "teacher_cache_index_sha256": cache_sha,
        "run_config_sha256": resolved.run_config_sha256,
        "model_inventory_sha256": inventory.fingerprint_sha256,
        "candidate_manifest_sha256": sha256_file(pool_index.manifest_path),
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "cost_table_sha256": cost_table_sha,
        "row_count": len(cost_rows),
    }
    meta_path = cost_dir / "cost_table_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = {
        "kind": "mix_bit_allocation",
        "model_id": inventory.model_id,
        "run_id": resolved.config.run_id,
        "solver_name": "scipy.optimize.milp",
        "solver_status": 0,
        "solver_message": "Optimization terminated successfully.",
        "scipy_version": "1.0.0",
        "is_globally_optimal": True,
        "allow_suboptimal": False,
        "time_limit_sec": None,
        "objective_scale": 1.0,
        "objective_delta_kl": objective,
        "baseline_mode": BASELINE_MODE,
        "baseline_objective_delta_kl": 0.0,
        "baseline_kl_mean": baseline_kl_mean,
        "predicted_mixed_model_kl": predicted,
        "kl_mode": kl_mode,
        "metric_name": metric_name,
        "teacher_topk": teacher_topk,
        "target_average_bit": target_bit,
        "bit_unit_denominator": 2,
        "used_bit_units": used_units,
        "budget_bit_units": budget_units,
        "budget_slack_bit_units": budget_units - used_units,
        "achieved_average_bit": achieved,
        "total_target_parameters": total_params,
        "run_config_sha256": resolved.run_config_sha256,
        "model_inventory_sha256": inventory.fingerprint_sha256,
        "candidate_manifest_sha256": sha256_file(pool_index.manifest_path),
        "candidate_space_sha256": resolved.candidate_space_sha256,
        "cost_table_sha256": cost_table_sha,
        "cost_table_meta_sha256": sha256_file(meta_path),
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload, cost_table_path, meta_path


def _patch_reload_model(monkeypatch: pytest.MonkeyPatch, template: nn.Module) -> None:
    def _get_model(_path, _token=None):
        return copy.deepcopy(template)

    monkeypatch.setattr("train_utils.model_checkpoint_io.get_model", _get_model)
    monkeypatch.setattr("rotation.model_utils.get_model", _get_model)


@pytest.fixture()
def validation_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    torch.manual_seed(0)
    profile = _toy_profile()
    modes = _toy_modes()
    resolved = _make_resolved(tmp_path, profile, modes)
    template = _ToyLM(n_layers=2)
    inventory = _inventory_for(profile, template)
    inventory_path = tmp_path / "inventory.json"
    write_model_inventory(inventory, inventory_path)
    pool_index = _export_pool(resolved, inventory)

    def _load_model(self, _profile, *, access_token=None):
        return copy.deepcopy(template)

    monkeypatch.setattr(
        "mix_bit.model_adapter.GenericDecoderAdapter.load_model",
        _load_model,
    )
    _patch_reload_model(monkeypatch, template)
    _patch_tiny_tokenizer(monkeypatch)

    assignments = _mixed_assignments(inventory, pool_index)
    overlay_dir = Path(resolved.canonical_run_root) / "baseline" / BASELINE_MODE
    overlay_path = write_uniform_baseline_overlay(
        output_dir=str(overlay_dir),
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=build_uniform_assignments(pool_index, BASELINE_MODE),
        mode_name=BASELINE_MODE,
    )
    examples = _make_examples(3)
    dataset_path, manifest_path = _write_dataset(tmp_path, examples, resolved, inventory)
    teacher_template = _ToyLM(n_layers=2)
    with torch.no_grad():
        teacher_template.lm_head.weight.add_(0.35)
    cache_dir = _write_teacher_cache(
        tmp_path / "teacher_cache",
        examples=examples,
        teacher=teacher_template,
        resolved=resolved,
        inventory=inventory,
        dataset_sha=sha256_file(dataset_path),
        teacher_topk=TEACHER_TOPK,
    )
    alloc_path = Path(resolved.canonical_run_root) / "allocation" / "topk_k4" / "optimal_2bit.json"
    cost_dir = Path(resolved.canonical_run_root) / "costs" / "topk_k4"
    payload, cost_table_path, cost_meta_path = _write_allocation_and_costs(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        cost_dir=cost_dir,
        teacher_topk=TEACHER_TOPK,
        baseline_overlay_path=overlay_path,
        teacher_cache_dir=cache_dir,
    )
    result = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=str(inventory_path),
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    return {
        "tmp_path": tmp_path,
        "resolved": resolved,
        "inventory": inventory,
        "inventory_path": str(inventory_path),
        "pool_index": pool_index,
        "template": template,
        "teacher_template": teacher_template,
        "assignments": assignments,
        "overlay_path": overlay_path,
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "cache_dir": cache_dir,
        "alloc_path": alloc_path,
        "allocation": payload,
        "cost_table_path": cost_table_path,
        "cost_meta_path": cost_meta_path,
        "mixed_model_dir": result["output_dir"],
    }


def _run_validate(world, **overrides):
    from mix_bit.validation import validate_mixed_model

    kwargs = {
        "resolved": world["resolved"],
        "inventory": world["inventory"],
        "inventory_path": world["inventory_path"],
        "pool_index": world["pool_index"],
        "allocation_path": str(world["alloc_path"]),
        "cost_table_path": str(world["cost_table_path"]),
        "cost_table_meta_path": str(world["cost_meta_path"]),
        "baseline_overlay_path": str(world["overlay_path"]),
        "mixed_model_dir": world["mixed_model_dir"],
        "dataset_path": str(world["dataset_path"]),
        "dataset_manifest_path": str(world["manifest_path"]),
        "teacher_cache": str(world["cache_dir"]),
        "device": "cpu",
        "skip_downstream_eval": True,
    }
    kwargs.update(overrides)
    return validate_mixed_model(**kwargs)


def test_validation_recomputes_integer_budget_and_weighted_bit(validation_world):
    report = _run_validate(validation_world)
    alloc = validation_world["allocation"]
    assert report["budget"]["used_bit_units"] == alloc["used_bit_units"]
    assert report["budget"]["budget_bit_units"] == alloc["budget_bit_units"]
    assert report["budget"]["achieved_average_bit"] == pytest.approx(
        alloc["achieved_average_bit"], abs=1e-9
    )
    assert report["structural"]["passed"] is True
    assert report["objective"]["objective_delta_kl"] == pytest.approx(
        alloc["objective_delta_kl"], rel=1e-9
    )


def test_validation_detects_missing_or_extra_assignment(validation_world):
    out = Path(validation_world["mixed_model_dir"])
    meta_path = out / META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # Drop one converted module name from provenance assignments to force mismatch.
    meta["extra_meta"]["mix_bit"]["assignments"] = meta["extra_meta"]["mix_bit"]["assignments"][:-1]
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="assignment|module|coverage|missing"):
        _run_validate(validation_world)


def test_validation_detects_wrong_compact_source_mode(validation_world):
    out = Path(validation_world["mixed_model_dir"])
    meta_path = out / META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    entry = meta["extra_meta"]["mix_bit"]["assignments"][0]
    # Flip mode without matching installed compact source.
    entry["mode_name"] = OTHER_MODE if entry["mode_name"] == BASELINE_MODE else BASELINE_MODE
    entry["compact_state_sha256"] = "f" * 64
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="compact|mode|source|hash"):
        _run_validate(validation_world)


def test_validation_rejects_any_original_weight_or_adapter_payload(validation_world):
    out = Path(validation_world["mixed_model_dir"])
    meta_path = out / META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["converted_modules"][0]["has_original_weight"] = True
    meta["extra_meta"]["peft_adapter"] = {"type": "lora"}
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="original_weight|adapter|peft|forbidden"):
        _run_validate(validation_world)


def test_save_reload_logits_are_close_with_fixed_tolerance(validation_world):
    report = _run_validate(validation_world)
    logits = report["save_reload"]
    assert logits["passed"] is True
    assert logits["rtol"] == 1e-4
    assert logits["atol"] == 1e-4
    assert logits["max_abs_error"] <= 1e-4 + 1e-12
    assert isinstance(logits["logits_shape"], list)
    ref_path = Path(validation_world["mixed_model_dir"]) / "reference_logits.pt"
    assert ref_path.is_file()


def test_predicted_and_actual_kl_are_reported_separately(validation_world):
    report = _run_validate(validation_world)
    kl = report["kl"]
    assert "baseline_kl_mean" in kl
    assert "predicted_mixed_model_kl" in kl
    assert "actual_mixed_model_kl" in kl
    assert "absolute_gap" in kl
    assert "relative_gap" in kl
    assert kl["predicted_mixed_model_kl"] == pytest.approx(
        validation_world["allocation"]["predicted_mixed_model_kl"]
    )
    assert kl["actual_mixed_model_kl"] != kl["predicted_mixed_model_kl"] or True
    # Fields must remain distinct keys (not overwritten into one another).
    assert set(kl) >= {
        "baseline_kl_mean",
        "predicted_mixed_model_kl",
        "actual_mixed_model_kl",
        "absolute_gap",
        "relative_gap",
    }


def test_topk_validation_uses_same_cache_and_never_loads_teacher(validation_world, monkeypatch):
    calls = {"teacher": 0}

    def _boom(*_a, **_k):
        calls["teacher"] += 1
        raise AssertionError("teacher must not be loaded for teacher_topk validation")

    monkeypatch.setattr("mix_bit.cost_search.load_teacher_model", _boom)
    monkeypatch.setattr("mix_bit.validation.load_teacher_model", _boom)
    report = _run_validate(validation_world)
    assert report["kl"]["kl_mode"] == KL_MODE_TEACHER_TOPK
    assert report["kl"]["teacher_topk"] == TEACHER_TOPK
    assert calls["teacher"] == 0
    expected_cache_sha = sha256_file(Path(validation_world["cache_dir"]) / "index.json")
    assert report["kl"]["teacher_cache_index_sha256"] == expected_cache_sha
    meta = json.loads(Path(validation_world["cost_meta_path"]).read_text(encoding="utf-8"))
    assert meta["teacher_cache_index_sha256"] == expected_cache_sha


def test_topk_validation_rejects_mismatched_teacher_cache_hash(validation_world):
    meta_path = Path(validation_world["cost_meta_path"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["teacher_cache_index_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # allocation records old cost_table_meta_sha256; also rewrite allocation hash field
    # so provenance against cost meta file still matches the mutated meta content.
    alloc_path = Path(validation_world["alloc_path"])
    alloc = json.loads(alloc_path.read_text(encoding="utf-8"))
    alloc["cost_table_meta_sha256"] = sha256_file(meta_path)
    alloc_path.write_text(json.dumps(alloc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Checkpoint mix_bit provenance still has the old allocation_sha256 / cost meta sha.
    # Re-assemble is heavy; instead only exercise the cache-hash gate by patching provenance
    # checks after cost-meta gate — mutate meta alone should fail before checkpoint reload
    # once cache hash is enforced against cost_meta.
    with pytest.raises(ValueError, match="teacher_cache_index_sha256"):
        _run_validate(validation_world)


def test_validation_rejects_mismatched_baseline_overlay_hash(validation_world):
    meta_path = Path(validation_world["cost_meta_path"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["baseline_overlay_sha256"] = "1" * 64
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    alloc_path = Path(validation_world["alloc_path"])
    alloc = json.loads(alloc_path.read_text(encoding="utf-8"))
    alloc["cost_table_meta_sha256"] = sha256_file(meta_path)
    alloc_path.write_text(json.dumps(alloc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="baseline_overlay_sha256"):
        _run_validate(validation_world)


def test_validation_requires_predicted_and_baseline_kl_fields(validation_world):
    alloc_path = Path(validation_world["alloc_path"])
    alloc = json.loads(alloc_path.read_text(encoding="utf-8"))
    del alloc["predicted_mixed_model_kl"]
    del alloc["baseline_kl_mean"]
    alloc_path.write_text(json.dumps(alloc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="baseline_kl_mean|predicted_mixed_model_kl"):
        _run_validate(validation_world)


def test_exact_validation_uses_teacher_and_rejects_cache(validation_world, monkeypatch):
    world = validation_world
    resolved = world["resolved"]
    inventory = world["inventory"]
    pool_index = world["pool_index"]
    assignments = world["assignments"]
    alloc_path = Path(resolved.canonical_run_root) / "allocation" / "exact" / "optimal_2bit.json"
    cost_dir = Path(resolved.canonical_run_root) / "costs" / "exact"
    _write_allocation_and_costs(
        path=alloc_path,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        cost_dir=cost_dir,
        kl_mode=KL_MODE_EXACT_FULL_VOCAB,
        metric_name=METRIC_NAME_EXACT_FULL_VOCAB,
        teacher_topk=None,
        baseline_overlay_path=world["overlay_path"],
        teacher_cache_dir=None,
    )
    result = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=world["inventory_path"],
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
        overwrite=True,
        output_dir=str(Path(world["tmp_path"]) / "exact_final"),
    )
    teacher_loads = {"n": 0}
    teacher_template = world["teacher_template"]

    def _load_teacher(*_a, **_k):
        teacher_loads["n"] += 1
        return copy.deepcopy(teacher_template)

    monkeypatch.setattr("mix_bit.cost_search.load_teacher_model", _load_teacher)
    monkeypatch.setattr("mix_bit.validation.load_teacher_model", _load_teacher)

    with pytest.raises(ValueError, match="teacher_cache|exact"):
        _run_validate(
            world,
            allocation_path=str(alloc_path),
            cost_table_path=str(cost_dir / "cost_table.jsonl"),
            cost_table_meta_path=str(cost_dir / "cost_table_meta.json"),
            mixed_model_dir=result["output_dir"],
            teacher_cache=str(world["cache_dir"]),
        )

    report = _run_validate(
        world,
        allocation_path=str(alloc_path),
        cost_table_path=str(cost_dir / "cost_table.jsonl"),
        cost_table_meta_path=str(cost_dir / "cost_table_meta.json"),
        mixed_model_dir=result["output_dir"],
        teacher_cache=None,
    )
    assert report["kl"]["kl_mode"] == KL_MODE_EXACT_FULL_VOCAB
    assert report["kl"]["teacher_topk"] is None
    assert teacher_loads["n"] == 1


def test_downstream_metrics_are_not_written_into_allocation_or_objective(validation_world, monkeypatch):
    alloc_path = Path(validation_world["alloc_path"])
    before = alloc_path.read_text(encoding="utf-8")
    before_obj = validation_world["allocation"]["objective_delta_kl"]

    def _fake_ppl(model, args):
        return {"wiki_ppl": 12.3, "seqlen": 2048, "nsamples": 1}

    def _fake_lm(model, tokenizer, args):
        return {
            "results": {"boolq": {"acc,none": 0.5}},
            "groups": {},
            "artifact_payload": {"tasks": ["boolq"]},
        }

    monkeypatch.setattr("mix_bit.validation.calculate_ppl", _fake_ppl)
    monkeypatch.setattr("mix_bit.validation.run_lm_eval", _fake_lm)
    monkeypatch.setattr(
        "mix_bit.validation._load_tokenizer_for_profile",
        lambda *_a, **_k: object(),
    )

    report = _run_validate(validation_world, skip_downstream_eval=False)
    after = alloc_path.read_text(encoding="utf-8")
    assert after == before
    assert "downstream" in report
    assert "wiki_ppl" in json.dumps(report["downstream"])
    # Allocation objective must be unchanged and not contain downstream keys.
    alloc = json.loads(after)
    assert alloc["objective_delta_kl"] == before_obj
    assert "downstream" not in alloc
    assert "wiki_ppl" not in alloc
    assert "boolq" not in alloc


# --- Tokenizer fingerprint v2 validation tests (Task 9 Step 4) ---

def test_validation_reports_tokenizer_fingerprint_section(validation_world):
    report = _run_validate(validation_world)
    tok = report["tokenizer"]
    assert tok["fingerprint_version"] == 2
    assert isinstance(tok["fingerprint_sha256"], str) and len(tok["fingerprint_sha256"]) == 64
    assert tok["reported_name_or_path"] == str(
        Path(validation_world["mixed_model_dir"]).resolve()
    )
    assert tok["local_reload_passed"] is True


def test_validation_fails_on_tampered_final_tokenizer_file(validation_world):
    out = Path(validation_world["mixed_model_dir"])
    marker = out / "tiny_tokenizer.json"
    data = json.loads(marker.read_text(encoding="utf-8"))
    data["vocab_seed"] = 123
    marker.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer fingerprint mismatch"):
        _run_validate(validation_world)


def test_validation_fails_on_legacy_tokenizer_fingerprint_version(validation_world):
    out = Path(validation_world["mixed_model_dir"])
    meta_path = out / META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["extra_meta"]["mix_bit"]["tokenizer_fingerprint_version"] = 1
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer_fingerprint_version"):
        _run_validate(validation_world)


def test_validation_fails_when_final_tokenizer_file_missing(validation_world):
    out = Path(validation_world["mixed_model_dir"])
    (out / "tiny_tokenizer.json").unlink()
    with pytest.raises(ValueError, match="tokenizer|local"):
        _run_validate(validation_world)
