from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils import checkpoint as torch_checkpoint

from rotation.model_utils import get_layers


def _move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


@dataclass
class _SavedTensorHandle:
    cpu_tensor: torch.Tensor
    original_device: torch.device
    copy_event: Optional[torch.cuda.Event]


class SavedTensorOffloadContext:
    def __init__(
        self,
        *,
        enabled: bool,
        min_tensor_bytes: int,
        pin_memory: bool,
    ):
        self.enabled = bool(enabled)
        self.min_tensor_bytes = int(min_tensor_bytes)
        self.pin_memory = bool(pin_memory)
        self._d2h_streams: Dict[int, torch.cuda.Stream] = {}
        self._h2d_streams: Dict[int, torch.cuda.Stream] = {}

    def _stream_for(self, device: torch.device, streams: Dict[int, torch.cuda.Stream]) -> torch.cuda.Stream:
        if device.type != "cuda":
            raise ValueError(f"Expected CUDA device, got {device!s}.")
        idx = 0 if device.index is None else int(device.index)
        stream = streams.get(idx)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            streams[idx] = stream
        return stream

    def _pack(self, tensor: torch.Tensor):
        if (
            not self.enabled
            or not torch.is_tensor(tensor)
            or tensor.device.type != "cuda"
            or _tensor_nbytes(tensor) < self.min_tensor_bytes
        ):
            return tensor

        cpu_tensor = torch.empty(
            tuple(tensor.shape),
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=bool(self.pin_memory),
        )
        copy_stream = self._stream_for(tensor.device, self._d2h_streams)
        with torch.cuda.stream(copy_stream):
            cpu_tensor.copy_(tensor.detach(), non_blocking=True)
            event = torch.cuda.Event()
            event.record(copy_stream)
        return _SavedTensorHandle(
            cpu_tensor=cpu_tensor,
            original_device=tensor.device,
            copy_event=event,
        )

    def _unpack(self, packed):
        if not isinstance(packed, _SavedTensorHandle):
            return packed
        if packed.copy_event is not None:
            packed.copy_event.synchronize()
        target_device = packed.original_device
        copy_stream = self._stream_for(target_device, self._h2d_streams)
        with torch.cuda.device(target_device), torch.cuda.stream(copy_stream):
            restored = packed.cpu_tensor.to(device=target_device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(copy_stream)
        torch.cuda.current_stream(target_device).wait_event(event)
        return restored

    def context(self):
        if not self.enabled:
            return nullcontext()
        return torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)


class OffloadStreamPool:
    def __init__(self):
        self._h2d_streams: Dict[int, torch.cuda.Stream] = {}
        self._d2h_streams: Dict[int, torch.cuda.Stream] = {}

    @staticmethod
    def _device_index(device: torch.device) -> int:
        if device.type != "cuda":
            raise ValueError(f"Expected CUDA device, got {device!s}.")
        return 0 if device.index is None else int(device.index)

    def h2d(self, device: torch.device) -> torch.cuda.Stream:
        idx = self._device_index(device)
        stream = self._h2d_streams.get(idx)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._h2d_streams[idx] = stream
        return stream

    def d2h(self, device: torch.device) -> torch.cuda.Stream:
        idx = self._device_index(device)
        stream = self._d2h_streams.get(idx)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._d2h_streams[idx] = stream
        return stream


class StreamingOffloadManager:
    def __init__(
        self,
        *,
        layer_devices: Dict[int, torch.device],
        prefetch_distance: int,
        checkpoint_layers: bool = True,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("offload_mode=streaming requires CUDA.")
        self.layer_devices = {int(idx): torch.device(device) for idx, device in layer_devices.items()}
        self.prefetch_distance = int(prefetch_distance)
        if self.prefetch_distance < 0:
            raise ValueError(f"offload_prefetch_distance must be >= 0, got {self.prefetch_distance}.")
        self.checkpoint_layers = bool(checkpoint_layers)
        self.streams = OffloadStreamPool()
        self._events: Dict[int, torch.cuda.Event] = {}
        self._resident: Dict[int, torch.device] = {}
        self.layers: List[OffloadedCheckpointLayer] = []

    def register(self, wrapper: "OffloadedCheckpointLayer") -> None:
        self.layers.append(wrapper)

    def prefetch(self, layer_idx: int) -> None:
        idx = int(layer_idx)
        if idx < 0 or idx >= len(self.layers):
            return
        target_device = self.layer_devices[idx]
        if self._resident.get(idx) == target_device and idx not in self._events:
            return
        previous_event = self._events.pop(idx, None)
        if previous_event is not None:
            previous_event.synchronize()
        wrapper = self.layers[idx]
        stream = self.streams.h2d(target_device)
        with torch.cuda.device(target_device), torch.cuda.stream(stream):
            wrapper.layer.to(device=target_device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        self._events[idx] = event
        self._resident[idx] = target_device

    def wait_for(self, layer_idx: int) -> None:
        idx = int(layer_idx)
        event = self._events.pop(idx, None)
        target_device = self.layer_devices[idx]
        if event is not None:
            torch.cuda.current_stream(target_device).wait_event(event)

    def prefetch_neighbors(self, layer_idx: int, *, direction: int) -> None:
        if self.prefetch_distance < 1:
            return
        for distance in range(1, self.prefetch_distance + 1):
            self.prefetch(int(layer_idx) + int(direction) * distance)

    def offload(self, layer_idx: int) -> None:
        idx = int(layer_idx)
        if idx < 0 or idx >= len(self.layers):
            return
        resident_device = self._resident.get(idx)
        if resident_device is None:
            return
        event = self._events.pop(idx, None)
        if event is not None:
            event.synchronize()
        wrapper = self.layers[idx]
        stream = self.streams.d2h(resident_device)
        with torch.cuda.device(resident_device), torch.cuda.stream(stream):
            wrapper.layer.to(device="cpu", non_blocking=True)
            offload_event = torch.cuda.Event()
            offload_event.record(stream)
        self._events[idx] = offload_event
        self._resident.pop(idx, None)

    def offload_far_forward(self, layer_idx: int) -> None:
        self.offload(int(layer_idx) - self.prefetch_distance - 1)

    def offload_far_backward(self, layer_idx: int) -> None:
        self.offload(int(layer_idx) + self.prefetch_distance + 1)

    def offload_all(self, *, synchronize: bool = False) -> None:
        for idx in range(len(self.layers)):
            self.offload(idx)
        if synchronize:
            self.synchronize()

    def synchronize(self) -> None:
        for event in list(self._events.values()):
            event.synchronize()
        self._events.clear()


class OffloadedCheckpointLayer(nn.Module):
    def __init__(
        self,
        *,
        layer: nn.Module,
        layer_idx: int,
        manager: StreamingOffloadManager,
    ):
        super().__init__()
        self.layer = layer
        self.layer_idx = int(layer_idx)
        self.manager = manager
        self.manager.register(self)
        self.layer.to("cpu")
        self.register_full_backward_hook(self._backward_hook)

    def _backward_hook(self, _module, _grad_input, _grad_output):
        if self.manager.checkpoint_layers:
            self.manager.prefetch_neighbors(self.layer_idx, direction=-1)
        self.manager.offload(self.layer_idx)
        self.manager.offload_far_backward(self.layer_idx)

    def _run_layer(self, *args, **kwargs):
        return self.layer(*args, **kwargs)

    def forward(self, *args, **kwargs):
        device = self.manager.layer_devices[self.layer_idx]
        self.manager.prefetch(self.layer_idx)
        self.manager.wait_for(self.layer_idx)
        moved_args = _move_to_device(args, device)
        moved_kwargs = _move_to_device(kwargs, device)
        self.manager.prefetch_neighbors(self.layer_idx, direction=1)

        if self.manager.checkpoint_layers and torch.is_grad_enabled():
            def _closure(*inner_args):
                self.manager.prefetch(self.layer_idx)
                self.manager.wait_for(self.layer_idx)
                if torch.is_grad_enabled():
                    self.manager.prefetch_neighbors(self.layer_idx, direction=-1)
                active_args = _move_to_device(inner_args, device)
                active_kwargs = _move_to_device(moved_kwargs, device)
                return self._run_layer(*active_args, **active_kwargs)

            out = torch_checkpoint.checkpoint(
                _closure,
                *moved_args,
                use_reentrant=True,
                preserve_rng_state=True,
            )
        else:
            out = self._run_layer(*moved_args, **moved_kwargs)

        if self.manager.checkpoint_layers:
            self.manager.offload_far_forward(self.layer_idx)
        return out


def wrap_model_layers_for_streaming_offload(
    model: nn.Module,
    *,
    layer_devices: Dict[int, torch.device],
    prefetch_distance: int,
    checkpoint_layers: bool,
) -> Tuple[StreamingOffloadManager, Dict[str, str]]:
    layers = get_layers(model)
    manager = StreamingOffloadManager(
        layer_devices=layer_devices,
        prefetch_distance=int(prefetch_distance),
        checkpoint_layers=bool(checkpoint_layers),
    )
    for idx, layer in enumerate(list(layers)):
        layers[idx] = OffloadedCheckpointLayer(
            layer=layer,
            layer_idx=int(idx),
            manager=manager,
        )
    mode = "streaming_checkpoint" if bool(checkpoint_layers) else "streaming_no_checkpoint"
    hf_device_map = {f"model.layers.{idx}": f"{mode}:{str(device)}" for idx, device in layer_devices.items()}
    setattr(model, "hf_device_map", hf_device_map)
    setattr(model, "is_parallelizable", True)
    setattr(model, "model_parallel", True)
    return manager, hf_device_map


def unwrap_streaming_offload_layers(model: nn.Module) -> int:
    layers = get_layers(model)
    count = 0
    for idx, layer in enumerate(list(layers)):
        if isinstance(layer, OffloadedCheckpointLayer):
            layer.manager.synchronize()
            layer.layer.to("cpu")
            layers[idx] = layer.layer
            count += 1
    return count


def validate_streaming_layer_devices(layer_devices: Dict[int, torch.device]) -> None:
    for idx, device in layer_devices.items():
        if torch.device(device).type != "cuda":
            raise ValueError(
                f"offload_mode=streaming requires every Transformer layer on CUDA target, "
                f"got layer {int(idx)} -> {device!s}."
            )
