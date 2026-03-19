from typing import Optional, Tuple

import torch
from torch import nn


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x:
        return _first_tensor(x[0])
    if isinstance(x, dict) and x:
        for v in x.values():
            try:
                return _first_tensor(v)
            except TypeError:
                continue
    raise TypeError(f"Unsupported hook payload type: {type(x)}")


class LayerIOHook:
    def __init__(self, layer: nn.Module):
        self.inp: Optional[torch.Tensor] = None
        self.out: Optional[torch.Tensor] = None
        self._with_kwargs = False
        try:
            self._pre_handle = layer.register_forward_pre_hook(self._pre_hook_with_kwargs, with_kwargs=True)
            self._post_handle = layer.register_forward_hook(self._post_hook_with_kwargs, with_kwargs=True)
            self._with_kwargs = True
        except TypeError:
            # Old torch versions do not support with_kwargs.
            self._pre_handle = layer.register_forward_pre_hook(self._pre_hook)
            self._post_handle = layer.register_forward_hook(self._post_hook)

    def _extract_input_tensor(self, args, kwargs=None):
        if args:
            return _first_tensor(args[0])
        if kwargs:
            # Common kw names for transformer blocks / attention blocks.
            for key in ("hidden_states", "x", "input", "inputs_embeds"):
                if key in kwargs:
                    return _first_tensor(kwargs[key])
            # Fallback: first tensor-like value in kwargs.
            for value in kwargs.values():
                try:
                    return _first_tensor(value)
                except TypeError:
                    continue
        return None

    def _pre_hook(self, _module, args):
        self.inp = self._extract_input_tensor(args, None)

    def _pre_hook_with_kwargs(self, _module, args, kwargs):
        self.inp = self._extract_input_tensor(args, kwargs)

    def _post_hook(self, _module, _args, output):
        self.out = _first_tensor(output)

    def _post_hook_with_kwargs(self, _module, _args, _kwargs, output):
        self.out = _first_tensor(output)

    def pop(self, detach: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.inp is None or self.out is None:
            raise RuntimeError("Layer hook did not capture both input/output tensors.")
        inp = self.inp.detach() if detach else self.inp
        out = self.out.detach() if detach else self.out
        self.inp = None
        self.out = None
        return inp, out

    def pop_output(self, detach: bool) -> torch.Tensor:
        if self.out is None:
            raise RuntimeError("Layer hook did not capture output tensor.")
        out = self.out.detach() if detach else self.out
        self.inp = None
        self.out = None
        return out

    def clear(self) -> None:
        self.inp = None
        self.out = None

    def remove(self) -> None:
        self._pre_handle.remove()
        self._post_handle.remove()
