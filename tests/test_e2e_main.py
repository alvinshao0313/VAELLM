from __future__ import annotations

from types import SimpleNamespace

from compressed_e2e_fintuning import main as e2e_main


def test_main_destroys_initialized_process_group(monkeypatch):
    parsed = (object(), object(), SimpleNamespace(full_determinism=False, seed=3))
    calls = []
    monkeypatch.setattr(e2e_main, "parse_args", lambda argv: parsed)
    monkeypatch.setattr(e2e_main, "configure_e2e_determinism", lambda value: None)
    monkeypatch.setattr(e2e_main, "set_e2e_seed", lambda value: None)
    monkeypatch.setattr(e2e_main, "run", lambda *args: calls.append("run"))
    monkeypatch.setattr(e2e_main.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(e2e_main.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        e2e_main.torch.distributed,
        "destroy_process_group",
        lambda: calls.append("destroy"),
    )

    e2e_main.main([])

    assert calls == ["run", "destroy"]


def test_main_destroys_process_group_when_run_fails(monkeypatch):
    parsed = (object(), object(), SimpleNamespace(full_determinism=False, seed=3))
    calls = []
    monkeypatch.setattr(e2e_main, "parse_args", lambda argv: parsed)
    monkeypatch.setattr(e2e_main, "configure_e2e_determinism", lambda value: None)
    monkeypatch.setattr(e2e_main, "set_e2e_seed", lambda value: None)

    def fail(*args):
        calls.append("run")
        raise RuntimeError("expected")

    monkeypatch.setattr(e2e_main, "run", fail)
    monkeypatch.setattr(e2e_main.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(e2e_main.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        e2e_main.torch.distributed,
        "destroy_process_group",
        lambda: calls.append("destroy"),
    )

    try:
        e2e_main.main([])
    except RuntimeError as exc:
        assert str(exc) == "expected"
    else:
        raise AssertionError("expected RuntimeError")

    assert calls == ["run", "destroy"]
