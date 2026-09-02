import torch

from sparse_bit_tuning.module import BankSpec, SparseBitTuningModule, layout_bank_specs


def _spec(module, stage, part, active, device="cpu"):
    logical = (8, 1, 32)
    return BankSpec(
        canonical_key=f"{module}|stage={stage}|part={part}",
        module_path=module,
        stage_idx=stage,
        part_idx=part,
        logical_shape=logical,
        n_bits=logical[0] * logical[2],
        n_active=active,
        device=torch.device(device),
    )


def test_module_banks_are_contiguous_and_not_split_across_chunks():
    specs = [
        _spec("model.layers.0.mlp.down_proj", 0, 0, 5),
        _spec("model.layers.0.mlp.down_proj", 1, 0, 7),
        _spec("model.layers.1.mlp.down_proj", 0, 0, 6),
    ]
    laid, chunks = layout_bank_specs(specs, target_chunk_bytes=24)
    by_module = {}
    for spec in laid:
        by_module.setdefault(spec.module_path, []).append(spec)
    first = by_module["model.layers.0.mlp.down_proj"]
    assert len({x.chunk_id for x in first}) == 1
    assert max(x.score_end for x in first) - min(x.score_start for x in first) == 12
    assert len(chunks) >= 1


def test_sparse_bit_module_owns_only_flat_score_parameters():
    specs = [
        _spec("model.layers.0.self_attn.q_proj", 0, 0, 4),
        _spec("model.layers.0.self_attn.k_proj", 0, 0, 3),
    ]
    module = SparseBitTuningModule(specs, target_chunk_bytes=1024)
    params = list(module.parameters())
    assert params
    assert all(p.dtype == torch.float16 for p in params)
    assert sum(p.numel() for p in params) == 7
    assert all("score_chunks" in name for name, _ in module.named_parameters())
    assert not any(hasattr(spec, "active_indices") for spec in module.bank_specs)


def test_module_score_span_covers_all_banks_of_module():
    specs = [
        _spec("model.layers.0.mlp.down_proj", 0, 0, 5),
        _spec("model.layers.0.mlp.down_proj", 0, 1, 6),
        _spec("model.layers.0.mlp.down_proj", 1, 0, 7),
    ]
    module = SparseBitTuningModule(specs)
    span, banks = module.module_score_span("model.layers.0.mlp.down_proj")
    assert span.numel() == 18
    assert [(b.stage_idx, b.part_idx) for b in banks] == [(0, 0), (0, 1), (1, 0)]
