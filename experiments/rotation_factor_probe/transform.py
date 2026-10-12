"""Experimental factors with the unchanged native two-sided transform arithmetic."""
from __future__ import annotations

import torch
from torch import nn

from experiments.rotation_factor_probe.factors import probe_factor
from rotation.weight_preprocess import TwoSidedHadamard, validate_rotation_spec


class ProbeRotation(TwoSidedHadamard):
    """Use isolated factors without changing native dispatch or writing v6 metadata."""

    def __init__(self, out_features: int, in_features: int, spec: dict, factors: dict):
        nn.Module.__init__(self)
        validate_rotation_spec(spec)
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.block_size = int(spec["block_size"])
        self.seed = int(spec["seed"])
        self.output_block = self.block_size or self.out_features
        self.input_block = self.block_size or self.in_features
        for size, block in ((self.out_features, self.output_block),
                            (self.in_features, self.input_block)):
            if size < 1 or block < 1 or size % block:
                raise ValueError("The explicit rotation blocks must divide the unprotected dimensions.")
        out_had, self.output_factor = probe_factor(self.output_block, factors)
        in_had, self.input_factor = probe_factor(self.input_block, factors)
        self.register_buffer("output_had", out_had, persistent=False)
        self.register_buffer("input_had", in_had, persistent=False)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.register_buffer("output_signs", torch.randint(
            0, 2, (self.out_features,), generator=generator, dtype=torch.int8) * 2 - 1)
        self.register_buffer("input_signs", torch.randint(
            0, 2, (self.in_features,), generator=generator, dtype=torch.int8) * 2 - 1)
        self.validate_state()

    def to_spec(self):
        raise RuntimeError("This isolated prototype must not be serialized as a production v6 rotation.")
