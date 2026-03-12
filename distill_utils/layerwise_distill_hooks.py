from typing import Optional, Tuple

import torch
from torch import nn


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x:
        return _first_tensor(x[0])
    raise TypeError(f"Unsupported hook payload type: {type(x)}")


class LayerIOHook:
    def __init__(self, layer: nn.Module):
        self.inp: Optional[torch.Tensor] = None
        self.out: Optional[torch.Tensor] = None
        self._pre_handle = layer.register_forward_pre_hook(self._pre_hook)
        self._post_handle = layer.register_forward_hook(self._post_hook)

    def _pre_hook(self, _module, args):
        if not args:
            raise RuntimeError("Layer forward args is empty; cannot capture input hidden states.")
        self.inp = _first_tensor(args[0])

    def _post_hook(self, _module, _args, output):
        self.out = _first_tensor(output)

    def pop(self, detach: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.inp is None or self.out is None:
            raise RuntimeError("Layer hook did not capture both input/output tensors.")
        inp = self.inp.detach() if detach else self.inp
        out = self.out.detach() if detach else self.out
        self.inp = None
        self.out = None
        return inp, out

    def clear(self) -> None:
        self.inp = None
        self.out = None

    def remove(self) -> None:
        self._pre_handle.remove()
        self._post_handle.remove()
