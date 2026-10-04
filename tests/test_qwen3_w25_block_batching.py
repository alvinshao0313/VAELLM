from copy import deepcopy
import json

import pytest
import torch
from torch import nn
from transformers import Qwen3ForCausalLM

from experiments.qwen3_w25.block_batching import batch_block_documents
from experiments.qwen3_w25.block_calibration import capture_block_batches
from experiments.qwen3_w25.block_weight_graph import block_weight_graph
from experiments.qwen3_w25.local_block import PROJECTIONS, _squared_error
from test_qwen3_w25_local_block import _fixture
from test_qwen3_w25_block_weight_graph import _prepare, _relative


def _documents():
    block, _, _ = _fixture()
    torch.manual_seed(176)
    teacher = Qwen3ForCausalLM(deepcopy(block.self_attn.config)).bfloat16().eval().requires_grad_(False)
    teacher.model.layers[1] = deepcopy(block)
    for name in PROJECTIONS:
        source = block.get_submodule(name)
        dense = nn.Linear(source.in_features, source.out_features, bias=source.bias is not None).bfloat16()
        with torch.no_grad():
            weight = source._decode_weight(dtype=torch.bfloat16)
            dense.weight.copy_(weight + 0.002 * torch.randn_like(weight))
            if source.bias is not None:
                dense.bias.copy_(source.bias)
        teacher.model.layers[1].set_submodule(name, dense)
    teacher.eval().requires_grad_(False)
    ids = torch.randint(3, 60, (8, 6))
    mask = torch.ones_like(ids)
    for index, length in enumerate((6, 5, 4, 3, 6, 5, 4, 3)):
        if index % 2:
            mask[index, length:] = 0
        else:
            mask[index, :6 - length] = 0
    ids[mask == 0] = 0
    batch = {"input_ids": ids, "attention_mask": mask}
    return teacher, block, batch, capture_block_batches(teacher, 1, batch)


def test_real_eager_batch_preserves_teacher_documents_masks_rope_and_outputs():
    teacher, block, raw, documents = _documents()
    before = deepcopy(documents)
    merged = batch_block_documents(documents, microbatch_size=8)
    assert len(merged) == 1 and merged[0].hidden_states.shape == (8, 6, 16)
    batch = merged[0]
    assert torch.equal(batch.valid_mask, raw["attention_mask"].bool())
    assert batch.forward_kwargs["attention_mask"].shape == (8, 1, 6, 6)
    assert batch.forward_kwargs["position_ids"].shape == (8, 6)
    assert batch.forward_kwargs["position_embeddings"][0].shape == (8, 6, 4)
    assert torch.equal(batch.forward_kwargs["cache_position"], torch.arange(6))
    with torch.no_grad():
        teacher_output = teacher.model.layers[1](batch.hidden_states, **batch.forward_kwargs)[0]
        single = torch.cat([block(item.hidden_states, **item.forward_kwargs)[0] for item in documents])
        together = block(batch.hidden_states, **batch.forward_kwargs)[0]
        # These are native BF16 blocks, not a replacement attention implementation.
        torch.testing.assert_close(teacher_output, batch.target_hidden_states, rtol=0.01, atol=0.00025)
        torch.testing.assert_close(together, single, rtol=0.01, atol=0.00025)
        old_loss = sum(float(_squared_error(block, item, torch.device("cpu"))) for item in documents)
        new_loss = float(_squared_error(block, batch, torch.device("cpu")))
        assert abs(new_loss - old_loss) / old_loss < 0.01
        batch.target_hidden_states[~batch.valid_mask] = 1000
        assert float(_squared_error(block, batch, torch.device("cpu"))) == new_loss
    for original, snapshot in zip(documents, before):
        assert torch.equal(original.hidden_states, snapshot.hidden_states)
        assert torch.equal(original.target_hidden_states, snapshot.target_hidden_states)
        assert torch.equal(original.forward_kwargs["attention_mask"], snapshot.forward_kwargs["attention_mask"])
    print(json.dumps({"native_output_relative_l2": _relative(together, single),
                      "valid_loss_relative_difference": abs(new_loss - old_loss) / old_loss}))


def _gradient_step(block, batches):
    targets, parameters = _prepare(block, True)
    optimizer = torch.optim.AdamW(parameters.values(), lr=0.0002, weight_decay=0.0)
    norm = sum(int(batch.valid_mask.sum()) for batch in batches) * block.hidden_size
    initial = {name: value.detach().clone() for name, value in parameters.items()}
    frozen = {name: value.clone() for name, value in block.state_dict().items() if name not in parameters}
    value = 0.0
    with block_weight_graph(targets, requires_grad=True) as graph:
        for batch in batches:
            loss = _squared_error(block, batch, torch.device("cpu")) / norm
            loss.backward()
            value += float(loss.detach())
        graph.backward_to_decoders()
    gradients = {name: parameter.grad.clone() for name, parameter in parameters.items()}
    optimizer.step()
    updates = {name: parameter.detach() - initial[name] for name, parameter in parameters.items()}
    assert all(torch.equal(block.state_dict()[name], tensor) for name, tensor in frozen.items())
    assert all(parameter.grad is None for name, parameter in block.named_parameters() if name not in parameters)
    return value, gradients, updates


def test_real_bf16_block_gradients_and_one_adam_update_for_batch_one_and_eight():
    _, original, _, documents = _documents()
    reference, candidate = deepcopy(original), deepcopy(original)
    loss1, gradients1, updates1 = _gradient_step(reference, batch_block_documents(documents, microbatch_size=1))
    loss8, gradients8, updates8 = _gradient_step(candidate, batch_block_documents(documents, microbatch_size=8))
    per_projection = {}
    for prefix in PROJECTIONS:
        names = [name for name in gradients1 if name.startswith(prefix + ".")]
        g1 = torch.cat([gradients1[name].flatten() for name in names])
        g8 = torch.cat([gradients8[name].flatten() for name in names])
        u1 = torch.cat([updates1[name].flatten() for name in names])
        u8 = torch.cat([updates8[name].flatten() for name in names])
        relative, cosine = _relative(g8, g1), float(torch.nn.functional.cosine_similarity(g8.double(), g1.double(), dim=0))
        # A batch changes BF16 GEMM gradient reduction/rounding. Test aggregate
        # direction and size rather than bit equality of near-zero components.
        assert relative < 0.01 and cosine > 0.9999
        assert _relative(u8, u1) < 0.01
        per_projection[prefix] = {"gradient_relative_l2": relative, "gradient_cosine": cosine,
                                  "adam_update_relative_l2": _relative(u8, u1),
                                  "adam_update_max_abs_difference": float((u8 - u1).abs().max())}
    assert abs(loss8 - loss1) / loss1 < 0.01
    assert all(batch.hidden_states.grad is None and batch.target_hidden_states.grad is None for batch in documents)
    print(json.dumps({"valid_loss_relative_difference": abs(loss8 - loss1) / loss1,
                      "per_projection": per_projection}))


def test_native_fa2_mask_preparation_is_grouped_without_claiming_cpu_fa2_execution():
    teacher, _, raw, documents = _documents()
    teacher.config._attn_implementation = "flash_attention_2"
    for index, item in enumerate(documents):
        # Exercise native CPU mask preparation only; no FA2 kernel is replaced.
        item.forward_kwargs["attention_mask"] = teacher.model._update_causal_mask(
            raw["attention_mask"][index:index + 1], item.hidden_states,
            item.forward_kwargs["cache_position"], None, False)
    grouped = batch_block_documents(documents)
    assert [batch.hidden_states.shape[0] for batch in grouped] == [2, 6]
    assert [batch.forward_kwargs["attention_mask"] is None for batch in grouped] == [True, False]
    ordered = [documents[index] for index in (0, 4, 1, 2, 3, 5, 6, 7)]
    for attribute in ("hidden_states", "target_hidden_states", "valid_mask"):
        assert torch.equal(torch.cat([getattr(batch, attribute) for batch in grouped]),
                           torch.cat([getattr(batch, attribute) for batch in ordered]))
    for group, originals in zip(grouped, (ordered[:2], ordered[2:])):
        for name in ("position_ids",):
            assert torch.equal(group.forward_kwargs[name],
                               torch.cat([batch.forward_kwargs[name] for batch in originals]))
        for index in range(2):
            assert torch.equal(group.forward_kwargs["position_embeddings"][index],
                               torch.cat([batch.forward_kwargs["position_embeddings"][index] for batch in originals]))
    capped = batch_block_documents(documents, microbatch_size=3)
    assert [batch.hidden_states.shape[0] for batch in capped] == [2, 3, 3]
    assert torch.equal(torch.cat([batch.hidden_states for batch in capped]),
                       torch.cat([batch.hidden_states for batch in ordered]))
    single = batch_block_documents(documents, microbatch_size=1)
    assert torch.equal(torch.cat([batch.hidden_states for batch in single]),
                       torch.cat([batch.hidden_states for batch in documents]))
    for item in grouped:
        mask = item.forward_kwargs["attention_mask"]
        if mask is not None:
            assert torch.equal(mask, item.valid_mask)
    full = batch_block_documents([documents[0], documents[4]])[0]
    assert full.hidden_states.shape[0] == 2 and full.forward_kwargs["attention_mask"] is None


def test_position_values_are_preserved_and_incompatible_shared_kwargs_split_or_reject():
    teacher, _, raw, documents = _documents()
    # Generate alternate positions with the native teacher's rotary embedding.
    changed = deepcopy(documents[0])
    positions = torch.arange(3, 9).unsqueeze(0)
    changed.forward_kwargs["position_ids"] = positions
    changed.forward_kwargs["position_embeddings"] = teacher.model.rotary_emb(changed.hidden_states, positions)
    combined = batch_block_documents([documents[0], changed])[0]
    assert torch.equal(combined.forward_kwargs["position_ids"][1], positions[0])
    assert torch.equal(combined.forward_kwargs["position_embeddings"][0][1], changed.forward_kwargs["position_embeddings"][0][0])
    changed.forward_kwargs["cache_position"] += 3
    assert len(batch_block_documents([documents[0], changed])) == 2
    separated = batch_block_documents([documents[0], changed, documents[1]])
    assert [batch.hidden_states.shape[0] for batch in separated] == [2, 1]
    assert torch.equal(separated[0].hidden_states, torch.cat([documents[0].hidden_states, documents[1].hidden_states]))
    assert torch.equal(separated[1].forward_kwargs["cache_position"], changed.forward_kwargs["cache_position"])
    short = capture_block_batches(teacher, 1, {key: value[:1, :4] for key, value in raw.items()})[0]
    assert len(batch_block_documents([documents[0], short])) == 2
    changed = deepcopy(documents[0])
    changed.forward_kwargs["position_embeddings"] = list(changed.forward_kwargs["position_embeddings"])
    assert len(batch_block_documents([documents[0], changed])) == 2
    changed.forward_kwargs["unexpected_argument"] = False
    with pytest.raises(ValueError, match="seven actually captured"):
        batch_block_documents([documents[0], changed])
    changed = deepcopy(documents[1])
    changed.forward_kwargs["attention_mask"] = None
    with pytest.raises(ValueError, match="entirely valid"):
        batch_block_documents([changed])
    for invalid in (0, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            batch_block_documents(documents, microbatch_size=invalid)
