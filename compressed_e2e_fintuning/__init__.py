from typing import Any

__all__ = ["run"]


def __getattr__(name: str) -> Any:
    if name == "run":
        from compressed_e2e_fintuning.runtime_v6 import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
