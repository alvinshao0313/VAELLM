import os
import socket
import subprocess
import unittest

import torch
import torch.multiprocessing as mp
from torch import nn

from train_utils.cat_inline_distributed import (
    _pack_bool_tensors_for_transport,
    _unpack_bool_tensors_from_transport,
    broadcast_group_vae_payload,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _broadcast_worker(rank: int, port: int, queue) -> None:
    os.environ.update({"RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": str(rank)})
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    try:
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

    def test_gpu_launcher_validation_and_count(self):
        script = "scripts/catlora_simple copy.sh"
        prefix = "source <(sed -n '3,27p' \"$1\"); printf '%s:%s' \"$CUDA_VISIBLE_DEVICES\" \"$NPROC_PER_NODE\""
        for value, expected in (("5", "5:1"), ("5,6,7,8", "5,6,7,8:4"), ("0,2,4", "0,2,4:3")):
            result = subprocess.run(
                ["bash", "-c", prefix, "bash", script],
                env={**os.environ, "DISTILL_GPUS": value},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, expected)
        for value in ("5,", "5, 6", "5,5"):
            result = subprocess.run(
                ["bash", "-c", prefix, "bash", script],
                env={**os.environ, "DISTILL_GPUS": value},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)

