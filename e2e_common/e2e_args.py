import argparse
from typing import Optional, List


_TARGET_MODULE_ALIASES = {
    "q": "q_proj",
    "query": "q_proj",
    "k": "k_proj",
    "key": "k_proj",
    "v": "v_proj",
    "value": "v_proj",
    "o": "o_proj",
    "out": "o_proj",
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}


def parse_decoder_layers(value: Optional[str]) -> Optional[List[int]]:
    raw = str(value or "").strip().lower()
    if raw in {"", "all", "*"}:
        return None

    out = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            parts = [p.strip() for p in token.split("-", 1)]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise argparse.ArgumentTypeError(
                    f"Invalid --decoder_layers token '{token}'. Expected <idx> or <begin>-<end>."
                )
            begin = int(parts[0])
            end = int(parts[1])
            if begin < 0 or end < 0 or end < begin:
                raise argparse.ArgumentTypeError(
                    f"Invalid --decoder_layers range '{token}'. Expected non-negative begin <= end."
                )
            out.update(range(begin, end + 1))
            continue

        idx = int(token)
        if idx < 0:
            raise argparse.ArgumentTypeError(
                f"Invalid --decoder_layers token '{token}'. Expected non-negative layer index."
            )
        out.add(idx)

    if not out:
        raise argparse.ArgumentTypeError("--decoder_layers cannot be empty.")
    return sorted(out)


def parse_target_modules(value: Optional[str]) -> Optional[List[str]]:
    raw = str(value or "").strip().lower()
    if raw in {"", "all", "*"}:
        return None

    out = []
    seen = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        normalized = _TARGET_MODULE_ALIASES.get(token, token)
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)

    if not out:
        raise argparse.ArgumentTypeError("--target_modules cannot be empty.")
    return out


def needs_teacher(loss_type: str) -> bool:
    norm = str(loss_type or "").strip().lower()
    return norm not in {"", "sft", "origin"}
