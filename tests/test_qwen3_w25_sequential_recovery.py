from copy import deepcopy

import pytest
import torch
from transformers import AutoTokenizer

from experiments.qwen3_w25.calibration import build_calibration_batches
from experiments.qwen3_w25.local_block import PROJECTIONS
from experiments.qwen3_w25.local_recovery import BASE_MODEL
from experiments.qwen3_w25.sequential_recovery import (
    _expected_decoder_state, _initialize_streams, _recover_blocks,
    _require_core_input_metadata, recovery_configuration, select_calibration_prefix,
)
from test_qwen3_w25_sequential_block_calibration import _fixture, _reference


def test_configuration_preserves_explicit_local_budget_and_rejects_unselected_settings():
    _require_core_input_metadata()
    for documents in (64, 512):
        for steps in (64, 192):
            config = recovery_configuration(train_documents=documents, steps=steps, lr=1.25e-5)
            assert config["layer_order"] == list(range(36))
            assert config["microbatch_size"] == 8
            assert config["maximum_train_documents"] == 512 and config["heldout_documents"] == 64
            assert not config["cross_layer_gradients"] and not config["bias_added"]
            assert not config["heldout_used_for_selection"]
            assert config["native_advance_dtype"] == "torch.bfloat16"
    for options in ({"train_documents": 32, "steps": 64, "lr": 1.25e-5},
                    {"train_documents": 64, "steps": 128, "lr": 1.25e-5},
                    {"train_documents": 64, "steps": 64, "lr": 2.5e-5}):
        with pytest.raises(ValueError):
            recovery_configuration(**options)


def test_real_512_plus_64_manifest_keeps_the_same_heldout_after_prefix_selection():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train, heldout, data = build_calibration_batches(
        tokenizer, seed=31, train_documents=512, heldout_documents=64, max_length=256)
    plans = []
    for count in (64, 512):
        selected, selected_heldout, plan = select_calibration_prefix(train, heldout, data, count)
        assert selected_heldout is heldout
        assert torch.equal(selected["input_ids"], train["input_ids"][:count])
        assert torch.equal(selected["attention_mask"], train["attention_mask"][:count])
        assert plan["train_document_indices"] == list(range(count))
        assert plan["heldout_document_indices"] == list(range(512, 576))
        assert plan["train_tokens"] == int(train["attention_mask"][:count].sum())
        assert plan["heldout_tokens"] == int(heldout["attention_mask"].sum())
        plans.append(plan)
    for key in ("manifest_sha256", "heldout_documents_sha256", "heldout_lengths", "heldout_tokens"):
        assert plans[0][key] == plans[1][key]
    assert plans[0]["train_documents_sha256"] != plans[1]["train_documents_sha256"]
    with pytest.raises(ValueError, match="64 or 512"):
        select_calibration_prefix(train, heldout, data, 128)


def test_real_two_block_optimizer_advances_updated_native_streams_and_freezes_the_rest():
    # Real tiny Qwen3 and real optimization exercise the orchestration helper;
    # production entry still requires every original 8B projection and 36 blocks.
    teacher, student, batch = _fixture()
    train = {name: value[:2] for name, value in batch.items()}
    heldout = {name: value[2:] for name, value in batch.items()}
    before_student, before_teacher = deepcopy(student.state_dict()), deepcopy(teacher.state_dict())
    train_stream, heldout_stream = _initialize_streams(teacher, student, train, heldout, torch.device("cpu"))
    reports, completed_states = [], []

    def complete(index, report):
        assert index == len(reports)
        assert train_stream.layer_index == heldout_stream.layer_index == index + 1
        assert report["native_streams_advanced"]
        assert report["input_source"] == "rolling_native_student_hidden_states"
        assert report["target_source"] == "independent_original_teacher_next_hidden_states_before_final_norm"
        assert report["execution_batch_sizes"] == {"train": [2], "heldout": [1]}
        assert report["settings"]["train_tokens"] == 7
        assert report["settings"]["heldout_tokens"] == 6
        assert all(not p.requires_grad and p.grad is None for p in student.parameters())
        for previous, frozen_state in enumerate(completed_states):
            assert all(torch.equal(value, frozen_state[name])
                       for name, value in student.model.layers[previous].state_dict().items())
        completed_states.append(deepcopy(student.model.layers[index].state_dict()))
        reports.append(report)

    _recover_blocks(teacher, student, train_stream, heldout_stream, device=torch.device("cpu"),
                    steps=24, lr=0.0002, microbatch_size=8, on_block=complete)
    assert len(reports) == 2 and train_stream.complete and heldout_stream.complete
    for stream, split in ((train_stream, train), (heldout_stream, heldout)):
        native_student, _ = _reference(student, split)
        native_teacher, _ = _reference(teacher, split)
        for doc, hidden in enumerate(stream._student_hidden):
            assert torch.equal(hidden, native_student[-1][doc]["output"])
            assert hidden.dtype == torch.bfloat16 and hidden.grad_fn is None and not hidden.requires_grad
        for doc, hidden in enumerate(stream._teacher_hidden):
            assert torch.equal(hidden, native_teacher[-1][doc]["output"])
    assert all(torch.equal(value, before_teacher[name]) for name, value in teacher.state_dict().items())
    targets = tuple(f"model.layers.{index}.{name}" for index in range(2) for name in PROJECTIONS)
    expected, changed = _expected_decoder_state(before_student, student.state_dict(), targets)
    assert changed and all("._parallel_stage_decoder." in name for name in changed)
    assert all(torch.equal(value, expected[name]) for name, value in student.state_dict().items())
    assert all(value.dtype == before_student[name].dtype for name, value in expected.items())

    broken = deepcopy(expected)
    frozen_key = next(name for name in broken if name.endswith(".vq_weight"))
    broken[frozen_key].reshape(-1)[0].bitwise_xor_(1)
    with pytest.raises(ValueError, match="Frozen sequential payload"):
        _expected_decoder_state(before_student, broken, targets)
    broken = deepcopy(expected)
    broken[changed[0]] = broken[changed[0]].float()
    with pytest.raises(RuntimeError, match="shape/dtype"):
        _expected_decoder_state(before_student, broken, targets)
    with pytest.raises(RuntimeError, match="No decoder update"):
        _expected_decoder_state(before_student, before_student, targets)
