"""Random sparse bit tuning for packed VAELLM VQ payloads.

The package is intentionally independent from the compressed E2E trainer.  The
trainer/VAELinear integration imports it only when sparse bit tuning is enabled.
"""

from .config import SparseBitTuningConfig
from .module import BankSpec, SparseBitTuningModule
from .sampler import AffineSamplerState

__all__ = [
    "AffineSamplerState",
    "BankSpec",
    "SparseBitTuningConfig",
    "SparseBitTuningModule",
]
