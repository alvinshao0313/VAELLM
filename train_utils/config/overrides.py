import argparse
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, Optional, Sequence, Tuple, TypeVar


T = TypeVar("T")

SELECTOR_DEFAULT = "default"
SELECTOR_CATEGORY = "cat"
SELECTOR_AFTER = "after"
KNOWN_SELECTORS = (SELECTOR_DEFAULT, SELECTOR_CATEGORY, SELECTOR_AFTER)
_INTRA_PARALLEL_PATTERN = re.compile(r"^(\d+)[xX](\d+)$")


@dataclass(frozen=True)
class OverrideKey:
    selector: str
    value: Optional[str] = None

    def as_token(self) -> str:
        if self.selector == SELECTOR_DEFAULT:
            return SELECTOR_DEFAULT
        return f"{self.selector}:{self.value}"


@dataclass(frozen=True)
class OverrideSpec(Generic[T]):
    arg_name: str
    parse_value: Callable[[str], T]
    allowed_selectors: Tuple[str, ...]
    example: str


@dataclass
class OverrideTable(Generic[T]):
    arg_name: str
    allowed_selectors: Tuple[str, ...]
    has_default: bool = False
    default: Optional[T] = None
    by_category: Dict[str, T] = field(default_factory=dict)
    by_after_category: Dict[str, T] = field(default_factory=dict)

    def is_override_enabled(self) -> bool:
        return bool(self.by_category or self.by_after_category)

    def to_jsonable(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        if self.has_default:
            out["default"] = self.default
        if self.by_category:
            out["by_category"] = dict(self.by_category)
        if self.by_after_category:
            out["by_after_category"] = dict(self.by_after_category)
        return out


def looks_like_override_string(raw: object) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    if "=" not in text:
        return False
    token = text.split("=", 1)[0].strip()
    if token == SELECTOR_DEFAULT:
        return True
    if token.startswith(f"{SELECTOR_CATEGORY}:") or token.startswith(f"{SELECTOR_AFTER}:"):
        return True
    return "," in text and any(
        part.strip().startswith(("default=", "cat:", "after:"))
        for part in text.split(",")
    )


def _parse_selector(selector_token: str, spec: OverrideSpec[T]) -> OverrideKey:
    raw = str(selector_token).strip()
    if not raw:
        raise argparse.ArgumentTypeError(f"{spec.arg_name} selector cannot be empty.")
    if raw == SELECTOR_DEFAULT:
        selector = OverrideKey(selector=SELECTOR_DEFAULT, value=None)
    elif raw.startswith(f"{SELECTOR_CATEGORY}:"):
        category = raw.split(":", 1)[1].strip()
        if not category:
            raise argparse.ArgumentTypeError(
                f"Invalid {spec.arg_name} selector '{raw}'. Category name cannot be empty."
            )
        selector = OverrideKey(selector=SELECTOR_CATEGORY, value=category)
    elif raw.startswith(f"{SELECTOR_AFTER}:"):
        category = raw.split(":", 1)[1].strip()
        if not category:
            raise argparse.ArgumentTypeError(
                f"Invalid {spec.arg_name} selector '{raw}'. after-category name cannot be empty."
            )
        selector = OverrideKey(selector=SELECTOR_AFTER, value=category)
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid {spec.arg_name} selector '{raw}'. "
            f"Supported selectors: {','.join(spec.allowed_selectors)}. Example: {spec.example}"
        )
    if selector.selector not in spec.allowed_selectors:
        raise argparse.ArgumentTypeError(
            f"{spec.arg_name} does not support selector '{selector.as_token()}'. "
            f"Supported selectors: {','.join(spec.allowed_selectors)}. Example: {spec.example}"
        )
    return selector


def parse_override_table(raw: str, spec: OverrideSpec[T]) -> OverrideTable[T]:
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise argparse.ArgumentTypeError(
            f"{spec.arg_name} cannot be empty. Use selector=value entries, for example: {spec.example}"
        )
    if text.startswith("{") or text.startswith("["):
        raise argparse.ArgumentTypeError(
            f"{spec.arg_name} no longer accepts JSON/dict/list syntax. "
            f"Use selector=value entries instead, for example: {spec.example}"
        )

    table: OverrideTable[T] = OverrideTable(
        arg_name=spec.arg_name,
        allowed_selectors=tuple(spec.allowed_selectors),
    )
    seen = set()
    items = [part.strip() for part in text.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            f"{spec.arg_name} cannot be empty. Use selector=value entries, for example: {spec.example}"
        )

    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid {spec.arg_name} entry '{item}'. "
                f"Expected selector=value, for example: {spec.example}"
            )
        selector_token, value_token = item.split("=", 1)
        selector = _parse_selector(selector_token, spec)
        selector_key = selector.as_token()
        if selector_key in seen:
            raise argparse.ArgumentTypeError(
                f"Duplicate {spec.arg_name} selector '{selector_key}'. "
                f"Each selector may only appear once."
            )
        seen.add(selector_key)

        value_raw = str(value_token).strip()
        if not value_raw:
            raise argparse.ArgumentTypeError(
                f"Invalid {spec.arg_name} entry '{item}'. Value cannot be empty."
            )
        try:
            parsed_value = spec.parse_value(value_raw)
        except argparse.ArgumentTypeError:
            raise
        except Exception as e:  # pragma: no cover - defensive wrapper
            raise argparse.ArgumentTypeError(f"Invalid {spec.arg_name} value '{value_raw}': {e}") from e

        if selector.selector == SELECTOR_DEFAULT:
            table.has_default = True
            table.default = parsed_value
        elif selector.selector == SELECTOR_CATEGORY:
            table.by_category[str(selector.value)] = parsed_value
        elif selector.selector == SELECTOR_AFTER:
            table.by_after_category[str(selector.value)] = parsed_value
        else:  # pragma: no cover - impossible with _parse_selector
            raise RuntimeError(f"Unexpected selector: {selector.selector}")

    return table


def resolve_category_value(table: OverrideTable[T], category: str) -> T:
    category_key = str(category)
    if category_key in table.by_category:
        return table.by_category[category_key]
    if table.has_default:
        return table.default  # type: ignore[return-value]
    raise ValueError(
        f"{table.arg_name} has no value for category '{category_key}' and no default. "
        f"Expected syntax like: default=...,cat:{category_key}=..."
    )


def resolve_after_category_value(table: OverrideTable[T], after_category: Optional[str]) -> T:
    if after_category is not None:
        category_key = str(after_category)
        if category_key in table.by_after_category:
            return table.by_after_category[category_key]
    if table.has_default:
        return table.default  # type: ignore[return-value]
    if after_category is None:
        raise ValueError(f"{table.arg_name} has no default value.")
    raise ValueError(
        f"{table.arg_name} has no value for after-category '{after_category}' and no default. "
        f"Expected syntax like: default=...,after:{after_category}=..."
    )


def validate_category_keys(table: OverrideTable[T], active_categories: Sequence[str], arg_name: str) -> None:
    active_set = {str(category) for category in active_categories}
    invalid_cat_keys = sorted(key for key in table.by_category if key not in active_set)
    invalid_after_keys = sorted(key for key in table.by_after_category if key not in active_set)
    if invalid_cat_keys:
        raise ValueError(
            f"{arg_name} contains unknown category override keys: {invalid_cat_keys}. "
            f"Available categories: {sorted(active_set)}"
        )
    if invalid_after_keys:
        raise ValueError(
            f"{arg_name} contains unknown after-category override keys: {invalid_after_keys}. "
            f"Available categories: {sorted(active_set)}"
        )


def parse_bool_text(raw: str, *, arg_name: str) -> bool:
    if isinstance(raw, bool):
        return bool(raw)
    value = str(raw).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid {arg_name} value '{raw}'. Expected true/false.")


def parse_int_text(
    raw: str,
    *,
    arg_name: str,
    min_value: int,
) -> int:
    try:
        out = int(raw)
    except (TypeError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"Invalid {arg_name} value '{raw}'. Expected integer.") from e
    if out < int(min_value):
        raise argparse.ArgumentTypeError(f"{arg_name} must be >= {min_value}, got {out}.")
    return int(out)


def parse_optional_int_text(
    raw: str,
    *,
    arg_name: str,
    min_value: int,
) -> Optional[int]:
    text = str(raw).strip().lower()
    if text == "none":
        return None
    return parse_int_text(text, arg_name=arg_name, min_value=min_value)


def parse_float_text(
    raw: str,
    *,
    arg_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
    require_finite: bool = True,
) -> float:
    try:
        out = float(raw)
    except (TypeError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"Invalid {arg_name} value '{raw}'. Expected float.") from e
    if require_finite and not _is_finite(out):
        raise argparse.ArgumentTypeError(f"{arg_name} must be finite, got {out}.")
    if min_value is not None:
        min_value = float(min_value)
        if inclusive_min and out < min_value:
            raise argparse.ArgumentTypeError(f"{arg_name} must be >= {min_value}, got {out}.")
        if not inclusive_min and out <= min_value:
            raise argparse.ArgumentTypeError(f"{arg_name} must be > {min_value}, got {out}.")
    if max_value is not None:
        max_value = float(max_value)
        if inclusive_max and out > max_value:
            raise argparse.ArgumentTypeError(f"{arg_name} must be <= {max_value}, got {out}.")
        if not inclusive_max and out >= max_value:
            raise argparse.ArgumentTypeError(f"{arg_name} must be < {max_value}, got {out}.")
    return float(out)


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def make_choice_parser(*, arg_name: str, choices: Sequence[str]) -> Callable[[str], str]:
    allowed = {str(choice).strip().lower() for choice in choices}

    def _parse_choice(raw: str) -> str:
        value = str(raw).strip().lower()
        if value not in allowed:
            raise argparse.ArgumentTypeError(
                f"Invalid {arg_name} value '{raw}'. Supported: {','.join(sorted(allowed))}."
            )
        return value

    return _parse_choice


def parse_intra_parallel_text(raw: str, *, arg_name: str) -> Tuple[int, int]:
    value = str(raw).strip()
    if not value:
        raise argparse.ArgumentTypeError(f"{arg_name} cannot be empty.")
    match = _INTRA_PARALLEL_PATTERN.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"Invalid {arg_name} value '{raw}'. Expected RxC, for example 1x1 or 4x1."
        )
    row_parts = parse_int_text(match.group(1), arg_name=arg_name, min_value=1)
    col_parts = parse_int_text(match.group(2), arg_name=arg_name, min_value=1)
    return int(row_parts), int(col_parts)


def parse_intra_part_sort_mode_text(
    raw: str,
    *,
    arg_name: str,
) -> str:
    value = str(raw).strip().lower()
    if not value:
        raise argparse.ArgumentTypeError(f"{arg_name} cannot be empty.")
    if value != "none":
        raise argparse.ArgumentTypeError(
            f"Invalid {arg_name} value '{raw}'. 排序代码已关闭；只允许 none."
        )
    return value


def make_override_spec(
    *,
    arg_name: str,
    parse_value,
    allowed_selectors: Sequence[str],
    example: str,
) -> OverrideSpec:
    return OverrideSpec(
        arg_name=arg_name,
        parse_value=parse_value,
        allowed_selectors=tuple(str(selector) for selector in allowed_selectors),
        example=example,
    )


def make_positive_int_override_spec(
    *,
    arg_name: str,
    allowed_selectors: Sequence[str],
    example: str,
    min_value: int = 1,
) -> OverrideSpec:
    return make_override_spec(
        arg_name=arg_name,
        parse_value=lambda raw: parse_int_text(raw, arg_name=arg_name, min_value=min_value),
        allowed_selectors=allowed_selectors,
        example=example,
    )


def make_optional_int_override_spec(
    *,
    arg_name: str,
    allowed_selectors: Sequence[str],
    example: str,
    min_value: int,
) -> OverrideSpec:
    return make_override_spec(
        arg_name=arg_name,
        parse_value=lambda raw: parse_optional_int_text(raw, arg_name=arg_name, min_value=min_value),
        allowed_selectors=allowed_selectors,
        example=example,
    )


def make_choice_override_spec(
    *,
    arg_name: str,
    allowed_selectors: Sequence[str],
    example: str,
    choices: Sequence[str],
) -> OverrideSpec:
    return make_override_spec(
        arg_name=arg_name,
        parse_value=make_choice_parser(arg_name=arg_name, choices=choices),
        allowed_selectors=allowed_selectors,
        example=example,
    )
