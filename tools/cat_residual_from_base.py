import os
import sys
from typing import Optional, Sequence


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train_utils.cat_residual_from_base import main as _main


def main(argv: Optional[Sequence[str]] = None) -> None:
    _main(argv)


if __name__ == "__main__":
    main()
