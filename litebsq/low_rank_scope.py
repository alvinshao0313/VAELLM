LOW_RANK_SCOPE_FULL = "full"
LOW_RANK_SCOPE_COMPRESSED_SUBSPACE = "compressed_subspace"
VALID_LOW_RANK_SCOPES = frozenset({
    LOW_RANK_SCOPE_FULL,
    LOW_RANK_SCOPE_COMPRESSED_SUBSPACE,
})


def normalize_low_rank_scope(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in VALID_LOW_RANK_SCOPES:
        raise ValueError(
            f"Unsupported low_rank_scope={value!r}; expected one of "
            f"{sorted(VALID_LOW_RANK_SCOPES)}."
        )
    return normalized
