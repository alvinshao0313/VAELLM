"""CPU regression for completed-block persistence and strict restart identity.

Uses actual torch Linear modules, optimization, RNG and disk serialization. It
checks persistence semantics only; the real VAELLM/GPU equivalence is a separate
integration check.
"""
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from experiments.liftquant_recovery.recovery_resume import (
    boundary_identity, checkpoint_fingerprint, load_boundary, mutable_names_by_block,
    restore_rng, save_boundary, teacher_fingerprint,
)


def model():
    result = torch.nn.Module()
    result.model = torch.nn.Module()
    result.model.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)])
    return result


def train(block):
    optimizer = torch.optim.AdamW(block.parameters(), lr=0.01, weight_decay=0.0)
    for _ in range(2):
        x = torch.randn(2, 3) * float(np.random.uniform(0.5, 1.5))
        y = torch.randn(2, 3) + random.random()
        optimizer.zero_grad(set_to_none=True)
        (block(x) - y).square().mean().backward()
        optimizer.step()
    with torch.no_grad():
        return x, block(x).clone()


def assert_state(a, b):
    for name, value in a.items():
        torch.testing.assert_close(value, b[name], rtol=0, atol=0)


def expect_rejection(operation):
    try:
        operation()
    except (ValueError, RuntimeError, EOFError, OSError, KeyError, IndexError):
        return
    raise AssertionError("Invalid boundary was accepted.")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    student = model()
    initial = copy.deepcopy(student.state_dict())
    args = SimpleNamespace(nsamples=4, holdout=0, epochs=1, batch_size=2,
                           seed=42, seqlen=3, code_lr=2e-5, decoder_lr=1.25e-5,
                           output="unused", resume=None, checkpoint="unused",
                           blocks="0,1", gpu_memory_gib=12,
                           smoke_rows=None, redpajama_arrow_dir=None)
    ids = torch.arange(12, dtype=torch.long).reshape(4, 3)
    names = mutable_names_by_block(set(student.state_dict()), [0, 1])
    with tempfile.TemporaryDirectory(prefix="recovery_boundary_") as directory:
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        (source / "checkpoint_meta.json").write_text('{"checkpoint_id":"cpu-regression"}')
        (source / "config.json").write_text('{}')
        torch.save(initial, source / "pytorch_model.bin")
        fingerprint = checkpoint_fingerprint(source)
        teacher_dir = root / "teacher"
        (teacher_dir / "shards").mkdir(parents=True)
        (teacher_dir / "config.json").write_text("{}")
        torch.save(initial, teacher_dir / "shards" / "weights.bin")
        (teacher_dir / "weights.index.json").write_text(json.dumps(
            {"weight_map": {name: "shards/weights.bin" for name in initial}}))
        teacher_payload = teacher_fingerprint(teacher_dir)
        identity = boundary_identity(args, [0, 1], fingerprint, {"actual_training_source": "cpu-regression"},
                                     ids, teacher_fingerprint=teacher_payload)
        x0, y0 = train(student.model.layers[0])
        records = {"0": {"optimizer_steps": 2}}
        samples, outputs = {0: x0}, {0: y0}
        path = root / "latest_boundary.pt"
        save_boundary(path, student, identity, names, [0], records, samples, outputs)
        completed_native = copy.deepcopy(student.state_dict())
        train(student.model.layers[1])
        uninterrupted = copy.deepcopy(student.state_dict())
        expected_rng = (random.random(), float(np.random.random()), torch.rand(4))

        restarted = model()  # Model construction deliberately consumes Torch RNG.
        restarted.load_state_dict(initial, strict=True)
        restored = load_boundary(path, restarted, identity, names)
        assert_state(restarted.state_dict(), completed_native)
        random.random(), np.random.random(), torch.rand(19)  # Prefix replay can consume RNG.
        restore_rng(restored["rng"])
        train(restarted.model.layers[1])
        assert_state(restarted.state_dict(), uninterrupted)
        actual_rng = (random.random(), float(np.random.random()), torch.rand(4))
        assert actual_rng[:2] == expected_rng[:2]
        torch.testing.assert_close(actual_rng[2], expected_rng[2], rtol=0, atol=0)

        rejected = []
        for key in ("source", "teacher", "training_code", "config", "calibration"):
            changed = copy.deepcopy(identity)
            changed[key] = {"different": True}
            before = copy.deepcopy(restarted.state_dict())
            expect_rejection(lambda changed=changed: load_boundary(path, restarted, changed, names))
            assert_state(restarted.state_dict(), before)
            rejected.append(key)
        payload = torch.load(path, weights_only=True)
        for label, modify in (
            ("nonprefix", lambda p: p.update(completed=[1])),
            ("extra_frozen_tensor", lambda p: p["state"].update({"model.layers.1.weight": initial["model.layers.1.weight"]})),
            ("wrong_shape", lambda p: p["state"].update({"model.layers.0.weight": torch.zeros(1)})),
            ("nonfinite_state", lambda p: p["state"].update({"model.layers.0.weight": torch.full((3, 3), float("nan"))})),
            ("incomplete_block", lambda p: p["records"]["0"].update(optimizer_steps=1)),
        ):
            broken = copy.deepcopy(payload)
            modify(broken)
            broken_path = root / f"{label}.pt"
            torch.save(broken, broken_path)
            before = copy.deepcopy(restarted.state_dict())
            expect_rejection(lambda: load_boundary(broken_path, restarted, identity, names))
            assert_state(restarted.state_dict(), before)
            rejected.append(label)

        # Replacing the latest boundary cannot leave multiple historical weights.
        records["1"] = {"optimizer_steps": 2}
        samples[1], outputs[1] = x0, restarted.model.layers[1](x0).detach()
        save_boundary(path, restarted, identity, names, [0, 1], records, samples, outputs)
        newest = load_boundary(path, model(), identity, names)
        assert newest["completed"] == [0, 1]
        assert not list(root.glob(".latest_boundary.pt.*.tmp"))

        changed_weights = copy.deepcopy(initial)
        changed_weights["model.layers.0.weight"][0, 0] += 1
        torch.save(changed_weights, source / "pytorch_model.bin")
        assert checkpoint_fingerprint(source) != fingerprint
        torch.save(changed_weights, teacher_dir / "shards" / "weights.bin")
        assert teacher_fingerprint(teacher_dir) != teacher_payload
        (source / "pytorch_model.bin").rename(source / "native_payload.custom")
        (source / "checkpoint_meta.json").write_text(json.dumps({"state_dict_file": "native_payload.custom"}))
        assert "native_payload.custom" in checkpoint_fingerprint(source)
        print(json.dumps(dict(status="PASS", device="cpu", uninterrupted_equals_resumed=True,
                              torch_python_numpy_rng_exact=True, rejected=rejected,
                              actual_weight_payload_change_detected=True, teacher_indexed_payload_change_detected=True,
                              custom_native_payload_filename=True, atomic_latest_replaced=True,
                              gpu_validation="not performed by this CPU regression"), indent=2))


if __name__ == "__main__":
    main()
