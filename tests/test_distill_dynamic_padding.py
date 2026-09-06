import copy

import torch
import transformers

from e2e_common.dense_loss import compute_dense_loss_from_offloaded_teacher
from e2e_common.lazy_datasets import build_edgerazor_data_collator
from train_utils.cat_category_runtime import CatTrainHFTrainingArguments
from train_utils.distill_loss_core import compute_model_level_loss
from train_utils.lora_training import compute_distill_hidden_alignment_loss


class _PaddingTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def pad(
        self,
        encoded_inputs,
        *,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
        **_kwargs,
    ):
        padding_value = getattr(padding, "value", padding)
        lengths = [len(item["input_ids"]) for item in encoded_inputs]
        if padding_value == "max_length":
            target_length = int(max_length)
        else:
            target_length = max(lengths)
        if pad_to_multiple_of is not None:
            multiple = int(pad_to_multiple_of)
            target_length = ((target_length + multiple - 1) // multiple) * multiple

        input_ids = []
        attention_mask = []
        for item in encoded_inputs:
            ids = list(item["input_ids"])
            mask = list(item.get("attention_mask", [1] * len(ids)))
            pad_len = target_length - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_tensors == "pt":
            batch = {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in batch.items()
            }
        return batch


_FEATURES = [
    {
        "input_ids": [1, 2, 3, 4, 5],
        "attention_mask": [1, 1, 1, 1, 1],
        "labels": [-100, -100, 3, 4, 5],
    },
    {
        "input_ids": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        "attention_mask": [1] * 11,
        "labels": [-100, -100, -100, -100, 15, 16, 17, 18, 19, 20, 21],
    },
]

_VALID_LENGTHS = (5, 11)


def _make_fixed_and_dynamic_batches():
    tokenizer = _PaddingTokenizer()
    fixed_collator = build_edgerazor_data_collator(
        tokenizer,
        max_seq_len=32,
        dynamic_padding=False,
    )
    dynamic_collator = build_edgerazor_data_collator(
        tokenizer,
        max_seq_len=32,
        dynamic_padding=True,
    )
    fixed_batch = fixed_collator(copy.deepcopy(_FEATURES))
    dynamic_batch = dynamic_collator(copy.deepcopy(_FEATURES))
    return fixed_batch, dynamic_batch


def _make_logit_pairs():
    torch.manual_seed(1234)
    teacher_dynamic = torch.randn(2, 16, 128, dtype=torch.float32)
    student_dynamic = torch.randn(2, 16, 128, dtype=torch.float32)
    teacher_fixed = torch.randn(2, 32, 128, dtype=torch.float32)
    student_fixed = torch.randn(2, 32, 128, dtype=torch.float32)

    for row, valid_len in enumerate(_VALID_LENGTHS):
        teacher_fixed[row, :valid_len].copy_(teacher_dynamic[row, :valid_len])
        student_fixed[row, :valid_len].copy_(student_dynamic[row, :valid_len])

    return teacher_fixed, student_fixed, teacher_dynamic, student_dynamic


def _make_hidden_state_pairs():
    torch.manual_seed(4321)
    teacher_fixed_states = []
    student_fixed_states = []
    teacher_dynamic_states = []
    student_dynamic_states = []

    # 5 states = embedding output + 4 transformer block outputs.
    for _state_idx in range(5):
        teacher_dynamic = torch.randn(2, 16, 8, dtype=torch.float32)
        student_dynamic = torch.randn(2, 16, 8, dtype=torch.float32)
        teacher_fixed = torch.randn(2, 32, 8, dtype=torch.float32)
        student_fixed = torch.randn(2, 32, 8, dtype=torch.float32)

        for row, valid_len in enumerate(_VALID_LENGTHS):
            teacher_fixed[row, :valid_len].copy_(teacher_dynamic[row, :valid_len])
            student_fixed[row, :valid_len].copy_(student_dynamic[row, :valid_len])

        teacher_fixed_states.append(teacher_fixed)
        student_fixed_states.append(student_fixed)
        teacher_dynamic_states.append(teacher_dynamic)
        student_dynamic_states.append(student_dynamic)

    return (
        tuple(teacher_fixed_states),
        tuple(student_fixed_states),
        tuple(teacher_dynamic_states),
        tuple(student_dynamic_states),
    )


def _model_level_loss(
    *,
    loss_type: str,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    batch: dict,
    prompt_loss_weight: float = 0.03,
    top_k: int = 100,
    alpha: float = 0.5,
) -> torch.Tensor:
    return compute_model_level_loss(
        loss_type=loss_type,
        student_logits=student_logits,
        input_ids=batch["input_ids"],
        labels=batch["labels"],
        attention_mask=batch["attention_mask"],
        teacher_logits=teacher_logits,
        temperature=1.0,
        alpha=alpha,
        top_k=top_k,
        prompt_loss_weight=prompt_loss_weight,
    )


def _offloaded_model_level_loss(
    *,
    loss_type: str,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    batch: dict,
    prompt_loss_weight: float = 0.0,
    top_k: int = 100,
    alpha: float = 0.5,
) -> torch.Tensor:
    return compute_dense_loss_from_offloaded_teacher(
        loss_type=loss_type,
        student_logits=student_logits,
        teacher_logits_cpu=teacher_logits.detach().cpu(),
        input_ids=batch["input_ids"],
        labels=batch["labels"],
        attention_mask=batch["attention_mask"],
        temperature=1.0,
        alpha=alpha,
        top_k=top_k,
        prompt_loss_weight=prompt_loss_weight,
    )


def test_fixed_padding_preserves_max_length_behavior():
    tokenizer = _PaddingTokenizer()
    collator = build_edgerazor_data_collator(
        tokenizer,
        max_seq_len=32,
        dynamic_padding=False,
    )
    batch = collator(copy.deepcopy(_FEATURES))

    assert batch["input_ids"].shape == (2, 32)
    assert batch["attention_mask"].shape == (2, 32)
    assert batch["labels"].shape == (2, 32)

    for row, feature in enumerate(_FEATURES):
        valid_len = len(feature["input_ids"])
        assert torch.equal(
            batch["input_ids"][row, :valid_len],
            torch.tensor(feature["input_ids"], dtype=torch.long),
        )
        assert torch.equal(
            batch["attention_mask"][row, :valid_len],
            torch.tensor(feature["attention_mask"], dtype=torch.long),
        )
        assert torch.equal(
            batch["labels"][row, :valid_len],
            torch.tensor(feature["labels"], dtype=torch.long),
        )
        assert torch.all(batch["attention_mask"][row, valid_len:] == 0)
        assert torch.all(batch["labels"][row, valid_len:] == -100)
        assert torch.all(batch["input_ids"][row, valid_len:] == tokenizer.pad_token_id)


def test_dynamic_padding_uses_longest_rounded_to_multiple_of_8():
    tokenizer = _PaddingTokenizer()
    collator = build_edgerazor_data_collator(
        tokenizer,
        max_seq_len=32,
        dynamic_padding=True,
    )
    batch = collator(copy.deepcopy(_FEATURES))
    assert batch["input_ids"].shape == (2, 16)
    assert batch["attention_mask"].shape == (2, 16)
    assert batch["labels"].shape == (2, 16)
    assert torch.all(batch["attention_mask"][0, 5:] == 0)
    assert torch.all(batch["labels"][0, 5:] == -100)
    assert torch.all(batch["input_ids"][0, 5:] == tokenizer.pad_token_id)


def test_dynamic_padding_does_not_add_extra_block_when_already_aligned():
    tokenizer = _PaddingTokenizer()
    collator = build_edgerazor_data_collator(
        tokenizer,
        max_seq_len=32,
        dynamic_padding=True,
    )
    features = [
        {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [-100, -100, 3, 4, 5],
        },
        {
            "input_ids": list(range(100, 116)),
            "attention_mask": [1] * 16,
            "labels": [-100] * 4 + list(range(104, 116)),
        },
    ]
    batch = collator(features)
    assert batch["input_ids"].shape == (2, 16)


def test_dynamic_padding_keeps_full_length_sample_at_configured_max():
    tokenizer = _PaddingTokenizer()
    collator = build_edgerazor_data_collator(
        tokenizer,
        max_seq_len=32,
        dynamic_padding=True,
    )
    full = list(range(1, 33))
    features = [
        {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [-100, -100, 3, 4, 5],
        },
        {
            "input_ids": full,
            "attention_mask": [1] * 32,
            "labels": [-100] * 4 + full[4:],
        },
    ]
    batch = collator(features)
    assert batch["input_ids"].shape == (2, 32)
    assert torch.equal(
        batch["input_ids"][1],
        torch.tensor(full, dtype=torch.long),
    )


def test_dynamic_padding_rejects_max_length_not_divisible_by_8():
    tokenizer = _PaddingTokenizer()
    try:
        build_edgerazor_data_collator(
            tokenizer,
            max_seq_len=30,
            dynamic_padding=True,
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "dynamic padding requires max_seq_len to be divisible by 8" in str(exc)

    collator = build_edgerazor_data_collator(
        tokenizer,
        max_seq_len=30,
        dynamic_padding=False,
    )
    assert collator is not None


def test_collator_rejects_non_positive_max_length():
    tokenizer = _PaddingTokenizer()
    for bad in (0, -1):
        try:
            build_edgerazor_data_collator(
                tokenizer,
                max_seq_len=bad,
                dynamic_padding=False,
            )
            raise AssertionError(f"expected ValueError for max_seq_len={bad}")
        except ValueError:
            pass


def test_cat_hf_argument_parses_distill_dynamic_padding():
    parser = transformers.HfArgumentParser((CatTrainHFTrainingArguments,))
    (parsed,) = parser.parse_args_into_dataclasses(
        args=["--distill_dynamic_padding", "true"]
    )
    assert parsed.distill_dynamic_padding is True

    (default_parsed,) = parser.parse_args_into_dataclasses(args=[])
    assert default_parsed.distill_dynamic_padding is False


def test_prediction_masks_are_invariant_to_removed_padding():
    from train_utils.distill_loss_core import build_prediction_token_masks

    fixed_batch, dynamic_batch = _make_fixed_and_dynamic_batches()

    resp_f, prompt_f = build_prediction_token_masks(
        labels=fixed_batch["labels"],
        attention_mask=fixed_batch["attention_mask"],
    )
    resp_d, prompt_d = build_prediction_token_masks(
        labels=dynamic_batch["labels"],
        attention_mask=dynamic_batch["attention_mask"],
    )

    for row, valid_len in enumerate(_VALID_LENGTHS):
        causal_valid_len = max(valid_len - 1, 0)
        torch.testing.assert_close(
            resp_f[row, :causal_valid_len],
            resp_d[row, :causal_valid_len],
        )
        torch.testing.assert_close(
            prompt_f[row, :causal_valid_len],
            prompt_d[row, :causal_valid_len],
        )
        assert torch.count_nonzero(resp_f[row, causal_valid_len:]).item() == 0
        assert torch.count_nonzero(prompt_f[row, causal_valid_len:]).item() == 0
        assert torch.count_nonzero(resp_d[row, causal_valid_len:]).item() == 0
        assert torch.count_nonzero(prompt_d[row, causal_valid_len:]).item() == 0


def test_cat_kl_top_with_top_k_100_prompt_loss_is_padding_invariant():
    fixed_batch, dynamic_batch = _make_fixed_and_dynamic_batches()
    (
        teacher_logits_fixed,
        student_logits_fixed,
        teacher_logits_dynamic,
        student_logits_dynamic,
    ) = _make_logit_pairs()

    loss_fixed = _model_level_loss(
        loss_type="kl_top",
        student_logits=student_logits_fixed,
        teacher_logits=teacher_logits_fixed,
        batch=fixed_batch,
        top_k=100,
        prompt_loss_weight=0.03,
    )
    loss_dynamic = _model_level_loss(
        loss_type="kl_top",
        student_logits=student_logits_dynamic,
        teacher_logits=teacher_logits_dynamic,
        batch=dynamic_batch,
        top_k=100,
        prompt_loss_weight=0.03,
    )
    torch.testing.assert_close(
        loss_fixed,
        loss_dynamic,
        rtol=1e-5,
        atol=1e-6,
    )


def test_adaptive_top3_hidden_alignment_is_padding_invariant():
    fixed_batch, dynamic_batch = _make_fixed_and_dynamic_batches()
    (
        teacher_logits_fixed,
        student_logits_fixed,
        teacher_logits_dynamic,
        student_logits_dynamic,
    ) = _make_logit_pairs()
    del teacher_logits_fixed, student_logits_fixed, teacher_logits_dynamic, student_logits_dynamic
    (
        teacher_hiddens_fixed,
        student_hiddens_fixed,
        teacher_hiddens_dynamic,
        student_hiddens_dynamic,
    ) = _make_hidden_state_pairs()

    fixed_hidden_loss = compute_distill_hidden_alignment_loss(
        teacher_hidden_states=teacher_hiddens_fixed,
        student_hidden_states=student_hiddens_fixed,
        attention_mask=fixed_batch["attention_mask"],
        layer_weighting="adaptive_top_3",
    )
    dynamic_hidden_loss = compute_distill_hidden_alignment_loss(
        teacher_hidden_states=teacher_hiddens_dynamic,
        student_hidden_states=student_hiddens_dynamic,
        attention_mask=dynamic_batch["attention_mask"],
        layer_weighting="adaptive_top_3",
    )
    torch.testing.assert_close(
        fixed_hidden_loss,
        dynamic_hidden_loss,
        rtol=1e-5,
        atol=1e-6,
    )


def test_e2e_sft_ce_is_padding_invariant():
    fixed_batch, dynamic_batch = _make_fixed_and_dynamic_batches()
    (
        teacher_logits_fixed,
        student_logits_fixed,
        teacher_logits_dynamic,
        student_logits_dynamic,
    ) = _make_logit_pairs()
    del teacher_logits_fixed, teacher_logits_dynamic

    ce_fixed = _model_level_loss(
        loss_type="sft",
        student_logits=student_logits_fixed,
        teacher_logits=None,
        batch=fixed_batch,
        prompt_loss_weight=0.0,
    )
    ce_dynamic = _model_level_loss(
        loss_type="sft",
        student_logits=student_logits_dynamic,
        teacher_logits=None,
        batch=dynamic_batch,
        prompt_loss_weight=0.0,
    )
    torch.testing.assert_close(
        ce_fixed,
        ce_dynamic,
        rtol=1e-5,
        atol=1e-6,
    )


def test_e2e_offloaded_kl_is_padding_invariant():
    fixed_batch, dynamic_batch = _make_fixed_and_dynamic_batches()
    (
        teacher_logits_fixed,
        student_logits_fixed,
        teacher_logits_dynamic,
        student_logits_dynamic,
    ) = _make_logit_pairs()

    loss_fixed = _offloaded_model_level_loss(
        loss_type="kl",
        student_logits=student_logits_fixed,
        teacher_logits=teacher_logits_fixed,
        batch=fixed_batch,
    )
    loss_dynamic = _offloaded_model_level_loss(
        loss_type="kl",
        student_logits=student_logits_dynamic,
        teacher_logits=teacher_logits_dynamic,
        batch=dynamic_batch,
    )
    torch.testing.assert_close(
        loss_fixed,
        loss_dynamic,
        rtol=1e-5,
        atol=1e-6,
    )


def test_e2e_offloaded_kl_top_with_top_k_100_is_padding_invariant():
    fixed_batch, dynamic_batch = _make_fixed_and_dynamic_batches()
    (
        teacher_logits_fixed,
        student_logits_fixed,
        teacher_logits_dynamic,
        student_logits_dynamic,
    ) = _make_logit_pairs()

    loss_fixed = _offloaded_model_level_loss(
        loss_type="kl_top",
        student_logits=student_logits_fixed,
        teacher_logits=teacher_logits_fixed,
        batch=fixed_batch,
        top_k=100,
    )
    loss_dynamic = _offloaded_model_level_loss(
        loss_type="kl_top",
        student_logits=student_logits_dynamic,
        teacher_logits=teacher_logits_dynamic,
        batch=dynamic_batch,
        top_k=100,
    )
    torch.testing.assert_close(
        loss_fixed,
        loss_dynamic,
        rtol=1e-5,
        atol=1e-6,
    )


def test_e2e_offloaded_kd_is_padding_invariant():
    fixed_batch, dynamic_batch = _make_fixed_and_dynamic_batches()
    (
        teacher_logits_fixed,
        student_logits_fixed,
        teacher_logits_dynamic,
        student_logits_dynamic,
    ) = _make_logit_pairs()

    loss_fixed = _offloaded_model_level_loss(
        loss_type="kd",
        student_logits=student_logits_fixed,
        teacher_logits=teacher_logits_fixed,
        batch=fixed_batch,
        alpha=0.5,
        prompt_loss_weight=0.0,
    )
    loss_dynamic = _offloaded_model_level_loss(
        loss_type="kd",
        student_logits=student_logits_dynamic,
        teacher_logits=teacher_logits_dynamic,
        batch=dynamic_batch,
        alpha=0.5,
        prompt_loss_weight=0.0,
    )
    torch.testing.assert_close(
        loss_fixed,
        loss_dynamic,
        rtol=1e-5,
        atol=1e-6,
    )
