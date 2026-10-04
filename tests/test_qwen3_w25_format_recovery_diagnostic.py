"""CPU tests for the newly added read-only answer/heldout statistics."""
from copy import deepcopy
import hashlib
import json
from uuid import uuid4

import pytest
import torch

from experiments.qwen3_w25.format_recovery_diagnostic import answer_ce, check_inputs, summarize_heldout
from test_qwen3_w25_kl_joint_export import TARGET, _model
from train_utils.checkpoint_v6 import load_v6_meta, save_v6_full_checkpoint


def test_answer_ce_single_shift_respects_declared_positions_and_real_response_labels():
    batch = {"input_ids": torch.tensor([[1,2,3,4,0]]), "labels": torch.tensor([[-100,-100,3,4,-100]]),
             "attention_mask": torch.tensor([[1,1,1,1,0]])}
    logits = torch.arange(30).reshape(1,5,6).float()/7
    manifest = {"samples": [{"answer_positions": [2,3], "input_and_label_sha256": "observed", "iterator_sample_ordinal": 1024}]}
    result = answer_ce(logits, batch, manifest)[0]
    expected = -torch.log_softmax(logits[0,[1,2]], -1)[[0,1],[3,4]]
    assert result["answer_tokens"] == 2 and result["answer_ce_sum"] == pytest.approx(float(expected.sum()), abs=1e-6)
    assert result["first_answer_token_ce"] == pytest.approx(float(expected[0]), abs=1e-6)
    bad = deepcopy(manifest);bad["samples"][0]["answer_positions"] = [1,2]
    with pytest.raises(ValueError, match="real response labels"):
        answer_ce(logits, batch, bad)


def test_heldout_keeps_batch_kl_and_answer_sequence_token_reductions_distinct():
    rows=[]
    for i in range(8):
        samples=[{"answer_tokens": j+1, "answer_ce_sum": float((j+1)**2), "answer_ce_mean": float(j+1), "first_answer_token_ce": float(j+1)} for j in range(4)]
        rows.append({"loss": float(i+1), "denominator": float(i+2), "student_answer_ce": samples, "teacher_answer_ce": samples})
    result = summarize_heldout(rows)
    assert result["equal_batch_loss"] == 4.5
    assert result["loss_denominator_weighted"] == pytest.approx(sum((i+1)*(i+2) for i in range(8))/sum(i+2 for i in range(8)))
    assert result["student_answer_ce"]["sequences"] == 32
    assert result["student_answer_ce"]["sequence_mean"] == 2.5
    assert result["student_answer_ce"]["token_mean"] == 3.
    with pytest.raises(ValueError, match="eight finite"):
        summarize_heldout(rows[:-1])


def _input_record(tmp_path):
    source = tmp_path / "source"
    save_v6_full_checkpoint(_model(), str(source), checkpoint_kind="final_model",
        compressed_targets=(TARGET,), base_model_path="cpu-format-recovery-fixture", save_config=False)
    archive = tmp_path / "archive.pt"
    torch.save({"format": "input-pinning-test"}, archive)
    formal = tmp_path / "reference.json"
    formal.write_text(json.dumps({"purpose": "input-pinning-test"}))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    weight = source / "pytorch_model.bin"
    stat = weight.stat()
    return {"training_format": "chatml", "mode": "joint", "steps": 256,
        "source_checkpoint": str(source), "source_checkpoint_id": load_v6_meta(str(source))["checkpoint_id"],
        "source_sha256": {}, "input_metadata_sha256": {str(source / "checkpoint_meta.json"): digest(source / "checkpoint_meta.json")},
        "input_evidence_sha256": {}, "data_archive": {"path": str(archive), "sha256": digest(archive)},
        "formal_reference": {"path": str(formal), "sha256": digest(formal)},
        "input_stats": [{"path": str(weight), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}]}


def test_fresh_format_source_identity_and_pins_are_checked(tmp_path):
    record = _input_record(tmp_path)
    check_inputs(record)
    changed = deepcopy(record)
    changed["source_checkpoint_id"] = str(uuid4())
    with pytest.raises(ValueError, match="checkpoint identity differs"):
        check_inputs(changed)
    changed = deepcopy(record)
    changed["data_archive"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Pinned source/data/evidence changed"):
        check_inputs(changed)
    changed = deepcopy(record)
    changed["input_stats"][0]["size"] += 1
    with pytest.raises(ValueError, match="Protected input stat changed"):
        check_inputs(changed)
