import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.bit_pilot import (
    BASE_MODEL, TRAIN_DOCUMENTS, _digest, _restore, pilot_document_plan,
)
from experiments.qwen3_w25.calibration import build_calibration_batches
from litebsq.autoencoder import Decoder
from litebsq.vae_linear import VAELinear
from rotation.weight_preprocess import rotation_spec


def test_real_calibration_document_prefixes_and_disjoint_selection_confirmation():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train, heldout, manifest = build_calibration_batches(
        tokenizer, seed=31, train_documents=512, heldout_documents=64, max_length=256)
    plan = pilot_document_plan(train, heldout, manifest)
    flat_train = train["input_ids"][train["attention_mask"].bool()]
    for count in TRAIN_DOCUMENTS:
        direct = train["input_ids"][:count][train["attention_mask"][:count].bool()]
        assert torch.equal(flat_train[:plan["arms"][str(count)]["valid_tokens"]], direct)
    flat_heldout = heldout["input_ids"][heldout["attention_mask"].bool()]
    cut = plan["selection"]["valid_tokens"]
    assert torch.equal(flat_heldout[:cut], heldout["input_ids"][:32][heldout["attention_mask"][:32].bool()])
    assert torch.equal(flat_heldout[cut:], heldout["input_ids"][32:][heldout["attention_mask"][32:].bool()])
    training_ids = {row["row_index"] for row in manifest["train"]["documents"]}
    selection_ids = {row["row_index"] for row in manifest["heldout"]["documents"][:32]}
    confirmation_ids = {row["row_index"] for row in manifest["heldout"]["documents"][32:]}
    assert len(training_ids) == 512 and len(selection_ids) == len(confirmation_ids) == 32
    assert not training_ids & (selection_ids | confirmation_ids) and not selection_ids & confirmation_ids
    print({"train_tokens": {count: plan["arms"][str(count)]["valid_tokens"] for count in TRAIN_DOCUMENTS},
           "selection_tokens": cut, "confirmation_tokens": len(flat_heldout) - cut,
           "manifest_sha256": plan["manifest_sha256"]})


def test_complete_real_vae_state_reset_rebuilds_packed_plan_and_native_output():
    layer = VAELinear(
        in_features=4, out_features=4, bias=None, codebook_dim=4, transpose=True,
        stage_vq_weights=[torch.zeros(4, 1, 9, dtype=torch.bool) for _ in range(2)],
        stage_decoders=[Decoder(in_dim=9, out_dim=4, decoder_type="linear").bfloat16() for _ in range(2)],
        weight_rotation_spec=rotation_spec("two_sided", 0, 31, "reset_test"),
    ).eval()
    layer.pack_parallel_stage_decoder_(trainable=False)
    initial = {key: value.clone() for key, value in layer.state_dict().items()}
    expected_digest = _digest(initial, set())
    with torch.no_grad():
        expected = layer._decode_weight(dtype=torch.bfloat16).clone()
        for stage in range(2):
            layer.get_stage_part_vq_storage(stage, 0).bitwise_xor_(1)
        next(layer._parallel_stage_decoder.parameters()).add_(0.125)
    layer._build_parallel_stage_decode_plan()
    _restore(layer, initial, expected_digest)
    assert layer._cached_weight is None
    assert _digest(layer.state_dict(), set()) == expected_digest
    with torch.no_grad():
        torch.testing.assert_close(layer._decode_weight(dtype=torch.bfloat16), expected, atol=0, rtol=0)
