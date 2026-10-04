from pathlib import Path

import pytest

from tools.prepare_edgerazor_distill_jsonl import _atomic_jsonl_writer


def test_interrupted_download_does_not_publish_partial_dataset(tmp_path: Path):
    target = tmp_path / "dataset.jsonl"
    with pytest.raises(RuntimeError, match="interrupted"):
        with _atomic_jsonl_writer(target) as handle:
            handle.write('{"incomplete": true}\n')
            raise RuntimeError("interrupted")
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_replacement_preserves_completed_dataset_on_failure(tmp_path: Path):
    target = tmp_path / "dataset.jsonl"
    original = '{"complete": true}\n'
    with _atomic_jsonl_writer(target) as handle:
        handle.write(original)
    with pytest.raises(RuntimeError, match="interrupted"):
        with _atomic_jsonl_writer(target) as handle:
            handle.write('{"incomplete": true}\n')
            assert target.read_text(encoding="utf-8") == original
            raise RuntimeError("interrupted")
    assert target.read_text(encoding="utf-8") == original
    with _atomic_jsonl_writer(target) as handle:
        handle.write('{"new_complete": true}\n')
    assert target.read_text(encoding="utf-8") == '{"new_complete": true}\n'
    assert list(tmp_path.iterdir()) == [target]
