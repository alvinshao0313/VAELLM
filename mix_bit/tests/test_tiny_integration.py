"""Offline tiny end-to-end integration for mixed-bit VAE allocation handoff.

Fixture shape: 2 blocks × 2 categories × 3 candidate modes.
Exercises candidate-only export, tensor-free baseline overlay + dual in-memory
builds, both KL metrics on tiny tensors, atomic cost finalization, MILP vs
brute-force, and final assemble + strict reload — no network.
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from litebsq.llm_vae import Decoder
from litebsq.vae_linear import VAELinear
from mix_bit.candidate_artifact import save_candidate_artifact_from_model
from mix_bit.candidate_pool import generate_candidate_trials, write_trial_spec
from mix_bit.checkpoint_pool import CandidatePoolIndex, build_candidate_pool_index
from mix_bit.kl_metric import (
    KL_MODE_EXACT_FULL_VOCAB,
    KL_MODE_TEACHER_TOPK,
    METRIC_NAME_EXACT_FULL_VOCAB,
    METRIC_NAME_TEACHER_TOPK,
    paired_delta_kl,
    per_sample_exact_forward_kl,
    per_sample_teacher_topk_forward_kl,
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
from mix_bit.solver import BIT_UNIT_DENOMINATOR, bit_to_units


HIDDEN = 8
VOCAB = 16
BASELINE_MODE = "b4d4s2"
MODE_LOW = "b4d4s1"
MODE_HIGH = "b4d4s3"


class _TinyTokenizer:
    """Tiny save/reload tokenizer fixture for integration tests (no HF network)."""

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
        bias=None,
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

            def save_pretrained(self, path: str) -> None:
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "config.json").write_text(
                    json.dumps({"model_type": "toy", "vocab_size": vocab}),
                    encoding="utf-8",
                )

        self.config = _Cfg()

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids: torch.Tensor, **_kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = x + layer.q_proj(x) + layer.k_proj(x)
        x = self.norm(x)
        return SimpleNamespace(logits=self.lm_head(x))


def _toy_profile() -> ModelProfile:
    return ModelProfile(
        model_id="tiny_integration",
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


def _three_modes() -> tuple[CandidateMode, ...]:
    # codebook_bits == codebook_dim == 4 so nominal_bit == residual_stages;
    # varying residual_stages gives the 1.0 / 2.0 / 3.0 low / baseline / high tradeoff.
    return (
        CandidateMode(
            name=MODE_LOW,
            nominal_bit=1.0,
            codebook_bits=4,
            codebook_dim=4,
            residual_stages=1,
        ),
        CandidateMode(
            name=BASELINE_MODE,
            nominal_bit=2.0,
            codebook_bits=4,
            codebook_dim=4,
            residual_stages=2,
        ),
        CandidateMode(
            name=MODE_HIGH,
            nominal_bit=3.0,
            codebook_bits=4,
            codebook_dim=4,
            residual_stages=3,
        ),
    )


def _make_resolved(tmp_path: Path, profile: ModelProfile, modes: tuple[CandidateMode, ...]) -> ResolvedRunConfig:
    recipe = TrainingRecipeConfig(
        recipe_id="tiny_recipe",
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
        run_id="tiny_run",
        model_profile=profile,
        candidate_space=CandidateSpaceConfig(
            candidate_space_id="tiny_space",
            baseline_mode=BASELINE_MODE,
            target_average_bit=2.0,
            modes=modes,
        ),
        training_recipe=recipe,
        calibration=CalibrationConfig(
            source_jsonl=str(tmp_path / "calib.jsonl"),
            input_format="text",
            max_samples=1,
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


def _patch_reload_model(monkeypatch: pytest.MonkeyPatch, template: nn.Module) -> None:
    def _get_model(_path, _token=None):
        return copy.deepcopy(template)

    monkeypatch.setattr("train_utils.v6_model_loader.get_model", _get_model)
    monkeypatch.setattr("rotation.model_utils.get_model", _get_model)


def _synthetic_costs(inventory: ModelInventory, modes: tuple[CandidateMode, ...]) -> dict[tuple[str, str], float]:
    """Deterministic finite costs with non-trivial tradeoffs for MILP vs brute-force."""
    costs: dict[tuple[str, str], float] = {}
    for target in inventory.targets:
        costs[(target.module_name, BASELINE_MODE)] = 0.0
        # Prefer low-bit on even blocks when budget allows; high-bit is expensive.
        if target.block_index % 2 == 0:
            costs[(target.module_name, MODE_LOW)] = -0.20 - 0.01 * target.block_index
            costs[(target.module_name, MODE_HIGH)] = 0.40
        else:
            costs[(target.module_name, MODE_LOW)] = 0.15
            costs[(target.module_name, MODE_HIGH)] = -0.05
        for mode in modes:
            costs.setdefault((target.module_name, mode.name), 0.0)
    return costs


def _bruteforce_optimal(
    inventory: ModelInventory,
    modes: tuple[CandidateMode, ...],
    costs: dict[tuple[str, str], float],
    *,
    target_average_bit: float,
) -> tuple[dict[str, str], float]:
    mode_names = [m.name for m in modes]
    bit_of = {m.name: m.nominal_bit for m in modes}
    total_n = sum(t.param_count for t in inventory.targets)
    target_units = bit_to_units(target_average_bit)
    best_obj: float | None = None
    best_assign: dict[str, str] | None = None
    for assign in itertools.product(mode_names, repeat=len(inventory.targets)):
        used = 0
        obj = 0.0
        mapping: dict[str, str] = {}
        for target, mode in zip(inventory.targets, assign, strict=True):
            mapping[target.module_name] = mode
            used += target.param_count * bit_to_units(bit_of[mode])
            obj += costs[(target.module_name, mode)]
        if used > target_units * total_n:
            continue
        if best_obj is None or obj < best_obj - 1e-15:
            best_obj = obj
            best_assign = mapping
    assert best_assign is not None and best_obj is not None
    return best_assign, best_obj


def test_tiny_offline_end_to_end_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mix_bit.assembler import (
        assemble_optimal_mixed_checkpoint,
        build_model_from_assignments,
        build_uniform_assignments,
        prepare_uniform_baseline_overlay,
    )
    from mix_bit.cost_table import compute_search_counts, finalize_cost_table
    from mix_bit.solver import (
        derive_allocation_dir,
        load_cost_table_for_solve,
        solve_mixed_bit_allocation,
        write_allocation_outputs,
    )
    from train_utils.checkpoint_v6 import META_FILENAME, load_v6_full_checkpoint_into_model

    torch.manual_seed(0)
    profile = _toy_profile()
    modes = _three_modes()
    resolved = _make_resolved(tmp_path, profile, modes)
    template = _ToyLM(n_layers=2)
    inventory = _inventory_for(profile, template)

    # Shape gate: 2 blocks, 2 categories, 3 modes → C×R=6 trials, L×R=12 rows.
    assert inventory.block_count == 2
    assert len(inventory.category_order) == 2
    assert len(inventory.targets) == 4
    assert len(modes) == 3
    counts = compute_search_counts(
        category_count=len(inventory.category_order),
        target_linear_count=len(inventory.targets),
        mode_count=len(modes),
    )
    assert counts["source_job_count"] == 4  # C*(R-1)
    assert counts["non_baseline_module_evaluation_count"] == 8  # L*(R-1)
    assert counts["complete_row_count"] == 12  # L*R

    inventory_path = tmp_path / "inventory.json"
    write_model_inventory(inventory, inventory_path)

    def _load_model(self, _profile, *, access_token=None):
        return copy.deepcopy(template)

    monkeypatch.setattr(
        "mix_bit.model_adapter.GenericDecoderAdapter.load_model",
        _load_model,
    )
    _patch_reload_model(monkeypatch, template)
    _patch_tiny_tokenizer(monkeypatch)

    # 1) Candidate-only artifact export + L×R pool coverage.
    pool_index = _export_pool(resolved, inventory)
    trials = generate_candidate_trials(resolved, inventory)
    assert len(trials) == 6
    assert len(pool_index.candidates) == 12
    for (_module, _mode), cand in pool_index.candidates.items():
        assert Path(cand.source.compact_state_path).is_file()
        meta = json.loads(Path(cand.source.candidate_meta_path).read_text(encoding="utf-8"))
        assert "embed_tokens" not in json.dumps(meta)
        assert "lm_head" not in json.dumps(meta)

    # 2) Tensor-free baseline overlay + two independent in-memory builds.
    baseline = prepare_uniform_baseline_overlay(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        device="cpu",
        skip_audit=False,
    )
    overlay_path = Path(baseline["baseline_overlay"])
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert overlay["kind"] == "uniform_baseline_overlay"
    assert "tensor" not in overlay_path.read_text(encoding="utf-8").lower()
    banned = {
        "pytorch_model.bin",
        "model.safetensors",
        "checkpoint_meta.json",
        "module_state.pt",
    }
    baseline_files = {p.name for p in Path(baseline["baseline_dir"]).rglob("*") if p.is_file()}
    assert baseline_files.isdisjoint(banned)

    assignments = build_uniform_assignments(pool_index, BASELINE_MODE)
    model_a = build_model_from_assignments(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        device="cpu",
    )
    model_b = build_model_from_assignments(
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        assignments=assignments,
        device="cpu",
    )
    with torch.inference_mode():
        ids = torch.arange(8, dtype=torch.long).view(1, 8) % VOCAB
        logits_a = model_a(input_ids=ids).logits
        logits_b = model_b(input_ids=ids).logits
    torch.testing.assert_close(logits_a, logits_b, rtol=1e-4, atol=1e-4)
    del model_a, model_b

    # 3) Both KL metrics on tiny tensors (offline, no teacher model load).
    teacher = torch.randn(2, 4, VOCAB)
    student = teacher + 0.1 * torch.randn_like(teacher)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]],
        dtype=torch.bool,
    )
    exact = per_sample_exact_forward_kl(teacher, student, mask)
    assert exact.shape == (2,)
    assert torch.isfinite(exact).all()

    flat_teacher = teacher[mask]
    k = 3
    topk_prob, topk_idx = torch.topk(torch.softmax(flat_teacher.float(), dim=-1), k=k, dim=-1)
    offsets = torch.tensor([0, int(mask[0].sum()), int(mask.sum())], dtype=torch.long)
    topk_kl = per_sample_teacher_topk_forward_kl(
        teacher_topk_indices=topk_idx,
        teacher_topk_probs=topk_prob,
        token_offsets=offsets,
        shifted_student_logits=student,
        valid_mask=mask,
    )
    assert topk_kl.shape == (2,)
    assert torch.isfinite(topk_kl).all()
    deltas = paired_delta_kl(
        sample_ids_a=[10, 11],
        kl_a=exact,
        sample_ids_b=[10, 11],
        kl_b=torch.zeros_like(exact),
    )
    assert torch.allclose(deltas, exact)

    # 4) Atomic cost finalization → L×R rows.
    cost_run_root = Path(resolved.canonical_run_root) / "costs" / "exact_full_vocab"
    cost_run_root.mkdir(parents=True, exist_ok=True)
    costs = _synthetic_costs(inventory, modes)
    provenance = {
        "kl_mode": KL_MODE_EXACT_FULL_VOCAB,
        "metric_name": METRIC_NAME_EXACT_FULL_VOCAB,
        "teacher_topk": None,
        "run_config_sha256": resolved.run_config_sha256,
        "model_inventory_sha256": inventory.fingerprint_sha256,
        "candidate_manifest_sha256": sha256_file(pool_index.manifest_path),
        "calibration_manifest_sha256": "d" * 64,
        "baseline_overlay_sha256": sha256_file(overlay_path),
        "teacher_cache_index_sha256": "",
    }
    bit_of = {m.name: m.nominal_bit for m in modes}
    rows: list[dict[str, Any]] = []
    for target in inventory.targets:
        for mode in modes:
            cand = pool_index.candidates[(target.module_name, mode.name)]
            mean = float(costs[(target.module_name, mode.name)])
            rows.append(
                {
                    "module_name": target.module_name,
                    "category": target.category,
                    "module_suffix": target.module_suffix,
                    "block_index": int(target.block_index),
                    "in_features": int(target.in_features),
                    "out_features": int(target.out_features),
                    "has_bias": bool(target.has_bias),
                    "mode": mode.name,
                    "nominal_bit": float(bit_of[mode.name]),
                    "param_count": int(target.param_count),
                    "mean_delta_kl": mean,
                    "std_delta_kl": 0.0,
                    "standard_error_delta_kl": 0.0,
                    "kl_mode": provenance["kl_mode"],
                    "metric_name": provenance["metric_name"],
                    "teacher_topk": provenance["teacher_topk"],
                    "source_compact_state_sha256": cand.source.compact_state_sha256,
                    "per_sample_sha256": f"ps-{target.module_name}-{mode.name}",
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

    finalized = finalize_cost_table(
        rows=rows,
        resolved=resolved,
        inventory=inventory,
        pool_index=pool_index,
        cost_run_root=cost_run_root,
        expected_provenance=provenance,
        self_swap_audit={"passed": True, "audit_sha256": "b" * 64},
        source_job_count=int(counts["source_job_count"]),
        baseline_kl_mean=1.25,
    )
    assert finalized["row_count"] == 12
    assert Path(finalized["cost_table_jsonl"]).is_file()
    assert Path(finalized["cost_table_meta"]).is_file()

    # Also assert teacher_topk metric contract remains available offline.
    assert KL_MODE_TEACHER_TOPK == "teacher_topk"
    assert METRIC_NAME_TEACHER_TOPK

    # 5) MILP vs brute-force.
    expected_hashes = {
        "run_config_sha256": resolved.run_config_sha256,
        "model_inventory_sha256": inventory.fingerprint_sha256,
        "candidate_manifest_sha256": sha256_file(pool_index.manifest_path),
        "candidate_space_sha256": resolved.candidate_space_sha256,
    }
    cost_rows = load_cost_table_for_solve(
        finalized["cost_table_jsonl"],
        finalized["cost_table_meta"],
        inventory=inventory,
        candidate_space=resolved.config.candidate_space,
        expected_hashes=expected_hashes,
    )
    result = solve_mixed_bit_allocation(
        cost_rows,
        inventory=inventory,
        candidate_space=resolved.config.candidate_space,
        target_average_bit=2.0,
    )
    best_assign, best_obj = _bruteforce_optimal(
        inventory, modes, costs, target_average_bit=2.0
    )
    selected = {e.module_name: e.mode for e in result.entries}
    assert selected == best_assign
    assert result.objective_delta_kl == pytest.approx(best_obj, abs=1e-12)
    assert result.is_globally_optimal is True
    assert result.bit_unit_denominator == BIT_UNIT_DENOMINATOR
    assert result.achieved_average_bit <= 2.0 + 1e-12

    alloc_dir = derive_allocation_dir(finalized["cost_table_jsonl"])
    paths = write_allocation_outputs(
        result,
        output_dir=alloc_dir,
        model_id=inventory.model_id,
        run_id=resolved.config.run_id,
        provenance={
            **expected_hashes,
            "cost_table_sha256": sha256_file(finalized["cost_table_jsonl"]),
            "cost_table_meta_sha256": sha256_file(finalized["cost_table_meta"]),
            "kl_mode": provenance["kl_mode"],
            "metric_name": provenance["metric_name"],
            "teacher_topk": provenance["teacher_topk"],
            "baseline_kl_mean": 1.25,
        },
    )
    alloc_path = Path(paths["json"])
    assert alloc_path.is_file()

    # 6) Final full-model assembly + strict reload.
    assembled = assemble_optimal_mixed_checkpoint(
        resolved=resolved,
        inventory=inventory,
        inventory_path=str(inventory_path),
        pool_index=pool_index,
        allocation_path=str(alloc_path),
        device="cpu",
    )
    out = Path(assembled["output_dir"])
    assert (out / META_FILENAME).is_file()
    assert (out / "reference_logits.pt").is_file()
    assert (out / "state_fingerprint.json").is_file()
    assert assembled["state_fingerprint"] == str((out / "state_fingerprint.json").resolve())

    adapter = get_model_adapter(resolved.config.model_profile.adapter)
    reloaded = adapter.load_model(resolved.config.model_profile)
    load_v6_full_checkpoint_into_model(reloaded, str(out), map_location="cpu", strict=True)
    for name, mode in selected.items():
        module = reloaded.get_submodule(name)
        assert isinstance(module, VAELinear)
        assert module.original_weight is None
        expected_sha = pool_index.candidates[(name, mode)].source.compact_state_sha256
        meta = json.loads((out / META_FILENAME).read_text(encoding="utf-8"))
        assign_meta = {
            e["module_name"]: e
            for e in meta["extra_meta"]["mix_bit"]["assignments"]
        }
        assert assign_meta[name]["mode_name"] == mode
        assert assign_meta[name]["compact_state_sha256"] == expected_sha


def test_tiny_teacher_topk_chunk_stays_compact_offline():
    """Integration guard: toy-model logits flow through build_teacher_topk_chunk
    and produce compact [N_valid, K] CPU tensors with no full-logits CPU transfer."""
    from mix_bit.teacher_cache import build_teacher_topk_chunk

    torch.manual_seed(7)
    profile = _toy_profile()
    template = _ToyLM(n_layers=1)
    with torch.inference_mode():
        ids = torch.arange(8, dtype=torch.long).view(1, 8) % VOCAB
        logits = template(input_ids=ids).logits[:, :-1, :]
    mask = torch.ones(1, logits.shape[1], dtype=torch.bool)
    chunk = build_teacher_topk_chunk(
        sample_ids=[0],
        shifted_teacher_logits=logits,
        valid_mask=mask,
        teacher_topk=3,
        cache_prob_dtype="float32",
    )
    assert chunk["teacher_topk_indices"].ndim == 2
    assert chunk["teacher_topk_probs"].ndim == 2
    assert chunk["teacher_topk_indices"].shape == (logits.shape[1], 3)
    assert chunk["teacher_topk_probs"].shape == (logits.shape[1], 3)
    assert chunk["teacher_topk_indices"].device.type == "cpu"
    assert chunk["teacher_topk_probs"].device.type == "cpu"
