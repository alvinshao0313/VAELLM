import os
import socket
import subprocess
import time
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
from torch import nn

import train_utils.cat_inline_distributed as cat_inline_distributed
from train_utils.cat_train_pipeline import _validate_inline_after_category_mode
from train_utils.cat_inline_distributed import (
    _pack_bool_tensors_for_transport,
    _resolve_cat_inline_vae_wait_timeout_sec,
    _unpack_bool_tensors_from_transport,
    broadcast_group_vae_payload,
    initialize_cat_payload_group,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _broadcast_worker(rank: int, port: int, queue, delay_source: bool = False) -> None:
    os.environ.update({"RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": str(rank)})
    os.environ["CAT_INLINE_VAE_WAIT_TIMEOUT_SEC"] = "10" if delay_source else "7200"
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    try:
        if delay_source and rank == 0:
            time.sleep(2)
        payload = (
            {
                "format": "vaellm_group_vae_payload",
                "version": 1,
                "all_stage_bits": [torch.tensor([[True, False, True]], dtype=torch.bool)],
            }
            if rank == 0
            else None
        )
        received = broadcast_group_vae_payload(payload)
        queue.put((rank, received["all_stage_bits"][0].tolist()))
    finally:
        torch.distributed.destroy_process_group()


class CatInlineDistributedTests(unittest.TestCase):
    def _with_timeout_env(self, value):
        old = os.environ.get("CAT_INLINE_VAE_WAIT_TIMEOUT_SEC")
        if value is None:
            os.environ.pop("CAT_INLINE_VAE_WAIT_TIMEOUT_SEC", None)
        else:
            os.environ["CAT_INLINE_VAE_WAIT_TIMEOUT_SEC"] = value
        self.addCleanup(self._restore_timeout_env, old)

    @staticmethod
    def _restore_timeout_env(old):
        if old is None:
            os.environ.pop("CAT_INLINE_VAE_WAIT_TIMEOUT_SEC", None)
        else:
            os.environ["CAT_INLINE_VAE_WAIT_TIMEOUT_SEC"] = old

    def test_timeout_resolver_defaults_to_7200_when_unset(self):
        self._with_timeout_env(None)
        self.assertEqual(_resolve_cat_inline_vae_wait_timeout_sec(), 7200)

    def test_timeout_resolver_accepts_positive_integer(self):
        for value, expected in (("10800", 10800), ("1", 1)):
            self._with_timeout_env(value)
            self.assertEqual(_resolve_cat_inline_vae_wait_timeout_sec(), expected)
            self.doCleanups()

    def test_timeout_resolver_rejects_invalid_values(self):
        for value in ("abc", "0", "-1"):
            self._with_timeout_env(value)
            with self.assertRaises(ValueError):
                _resolve_cat_inline_vae_wait_timeout_sec()
            self.doCleanups()

    def test_initialize_payload_group_wires_gloo_timeout(self):
        self._with_timeout_env("12345")
        old_payload_group = cat_inline_distributed._PAYLOAD_GROUP
        cat_inline_distributed._PAYLOAD_GROUP = None
        self.addCleanup(setattr, cat_inline_distributed, "_PAYLOAD_GROUP", old_payload_group)
        with (
            patch("train_utils.cat_inline_distributed.ensure_distill_process_group_initialized"),
            patch("train_utils.cat_inline_distributed.distill_world_size", return_value=2),
            patch("train_utils.cat_inline_distributed.torch.distributed.new_group", return_value="group") as new_group,
        ):
            self.assertEqual(initialize_cat_payload_group(), "group")
        new_group.assert_called_once_with(
            backend="gloo",
            timeout=timedelta(seconds=12345),
        )

    def test_transport_bool_pack_round_trip_preserves_opaque_module(self):
        decoder = nn.Linear(3, 2)
        original_decoder_state = {name: value.detach().clone() for name, value in decoder.state_dict().items()}
        payload = {
            "format": "vaellm_group_vae_payload",
            "version": 1,
            "all_stage_bits": [torch.tensor([[True, False, True, False, True, False, True, True, False]], dtype=torch.bool)],
            "nested": {"x": [torch.tensor([[False, True]], dtype=torch.bool)]},
            "float_tensor": torch.tensor([1.0, 2.0]),
            "decoder": decoder,
        }

        packed = _pack_bool_tensors_for_transport(payload)
        packed_bits = packed["all_stage_bits"][0]["data"]
        self.assertEqual(packed_bits.dtype, torch.uint8)
        self.assertEqual(packed_bits.numel(), 2)
        restored = _unpack_bool_tensors_from_transport(packed)

        self.assertTrue(torch.equal(restored["all_stage_bits"][0], payload["all_stage_bits"][0]))
        self.assertEqual(restored["all_stage_bits"][0].dtype, torch.bool)
        self.assertTrue(torch.equal(restored["nested"]["x"][0], payload["nested"]["x"][0]))
        self.assertTrue(torch.equal(restored["float_tensor"], payload["float_tensor"]))
        for name, value in decoder.state_dict().items():
            self.assertTrue(torch.equal(value, original_decoder_state[name]))

    def test_two_process_gloo_payload_broadcast(self):
        context = mp.get_context("spawn")
        queue = context.SimpleQueue()
        port = _free_port()
        workers = [context.Process(target=_broadcast_worker, args=(rank, port, queue)) for rank in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30)
            self.assertEqual(worker.exitcode, 0)
        received = sorted(queue.get() for _ in workers)
        self.assertEqual(received, [(0, [[True, False, True]]), (1, [[True, False, True]])])

    def test_two_process_gloo_payload_broadcast_with_delayed_source(self):
        context = mp.get_context("spawn")
        queue = context.SimpleQueue()
        port = _free_port()
        workers = [
            context.Process(target=_broadcast_worker, args=(rank, port, queue, True))
            for rank in range(2)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30)
            self.assertEqual(worker.exitcode, 0)
        received = sorted(queue.get() for _ in workers)
        self.assertEqual(received, [(0, [[True, False, True]]), (1, [[True, False, True]])])

    def test_gpu_launcher_validation_and_count(self):
        script = "scripts/catlora_simple2.sh"
        text = open(script, "r", encoding="utf-8").read()
        self.assertIn("export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", text)
        self.assertIn("torchrun --standalone --nproc_per_node=8 tools/cat_train.py", text)
        result = subprocess.run(["bash", "-n", script], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_inline_after_category_mode_allows_remaining_family(self):
        for mode in (
            "remaining_lora",
            "remaining_lora_current_decoder",
            "remaining_lora_prefix_decoder",
        ):
            _validate_inline_after_category_mode(mode)

    def test_inline_after_category_mode_rejects_compressed_modes(self):
        for mode in (
            "current_lora",
            "current_decoder",
            "current_lora_decoder",
            "compressed_lora",
            "decoder",
            "both",
            "remaining_lora_decoder",
            "remaining_lora_all_decoder",
        ):
            with self.assertRaises(ValueError):
                _validate_inline_after_category_mode(mode)
