import os
import tempfile
import unittest

import torch

from tools.cat_eval import _compute_checkpoint_fingerprint, _validate_adapter_checkpoint_match


class CatEvalAdapterMatchTest(unittest.TestCase):
    def _build_checkpoint_dir(self, root: str) -> tuple[str, dict]:
        ckpt_dir = os.path.join(root, "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        meta = {
            "format": "vaellm_state_dict_with_meta",
            "version": 4,
            "state_dict_file": "pytorch_model.bin",
            "base_model_path": "meta-llama/Llama-3.1-8B",
            "converted_modules": [],
        }
        with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as handle:
            import json

            json.dump(meta, handle, ensure_ascii=False, indent=2)
        torch.save({"x": torch.tensor([1, 2, 3])}, os.path.join(ckpt_dir, "pytorch_model.bin"))
        return ckpt_dir, meta

    def test_validate_adapter_checkpoint_match_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir, meta = self._build_checkpoint_dir(tmpdir)
            fingerprint = _compute_checkpoint_fingerprint(ckpt_dir, meta)
            adapter_meta = {
                "source_checkpoint_meta_sha256": fingerprint["meta_sha256"],
                "source_checkpoint_state_sha256": fingerprint["state_sha256"],
            }
            resolved = _validate_adapter_checkpoint_match(
                checkpoint_dir=ckpt_dir,
                checkpoint_meta=meta,
                adapter_meta=adapter_meta,
            )
            self.assertEqual(resolved["meta_sha256"], fingerprint["meta_sha256"])
            self.assertEqual(resolved["state_sha256"], fingerprint["state_sha256"])

    def test_validate_adapter_checkpoint_match_fail_on_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir, meta = self._build_checkpoint_dir(tmpdir)
            adapter_meta = {
                "source_checkpoint_meta_sha256": "0" * 64,
                "source_checkpoint_state_sha256": "1" * 64,
            }
            with self.assertRaises(ValueError):
                _validate_adapter_checkpoint_match(
                    checkpoint_dir=ckpt_dir,
                    checkpoint_meta=meta,
                    adapter_meta=adapter_meta,
                )


if __name__ == "__main__":
    unittest.main()
