from __future__ import annotations

from typing import Any, Sequence


def shape_hint_only_adds_singleton_or_dynamic_axes(
    *,
    resolved_shape: Sequence[Any],
    shape_hint: Sequence[Any],
) -> bool:
    """Return whether a higher-rank hint only inserts singleton/dynamic axes."""
    try:
        resolved = [int(value) for value in resolved_shape]
        hint = [int(value) for value in shape_hint]
    except (TypeError, ValueError):
        return False
    if (
        len(hint) <= len(resolved)
        or not resolved
        or any(value <= 0 for value in resolved)
    ):
        return False
    matched_counts = {0}
    for hinted_dimension in hint:
        next_counts: set[int] = set()
        for matched_count in matched_counts:
            if int(hinted_dimension) <= 1:
                next_counts.add(int(matched_count))
            if (
                int(matched_count) < len(resolved)
                and (
                    int(hinted_dimension) <= 0
                    or int(hinted_dimension) == int(resolved[matched_count])
                )
            ):
                next_counts.add(int(matched_count) + 1)
        matched_counts = next_counts
        if not matched_counts:
            return False
    return len(resolved) in matched_counts
