from __future__ import annotations

from typing import Any, Optional, Sequence


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


def static_shape_vector_length(tensor: Any) -> Optional[int]:
    """Return the statically known element count of a rank-1 shape tensor."""
    if tensor is None:
        return None
    shape = [int(value) for value in list(tensor.shape)]
    signature = (
        [int(value) for value in list(tensor.shape_signature)]
        if tensor.shape_signature is not None
        else list(shape)
    )
    if (
        len(shape) != 1
        or len(signature) != 1
        or int(shape[0]) <= 0
        or int(signature[0]) <= 0
        or int(shape[0]) != int(signature[0])
    ):
        return None
    return int(shape[0])


def preserve_rewritten_output_dynamic_axes(
    *,
    source_tensor: Any,
    target_tensor: Any,
) -> bool:
    """Preserve dynamic axes when an optimizer retargets a producer output."""
    if source_tensor is None or target_tensor is None:
        return False
    source_shape = [int(value) for value in list(source_tensor.shape)]
    target_shape = [int(value) for value in list(target_tensor.shape)]
    if len(source_shape) == 0 or len(source_shape) != len(target_shape):
        return False
    source_signature = (
        [int(value) for value in list(source_tensor.shape_signature)]
        if source_tensor.shape_signature is not None
        else list(source_shape)
    )
    target_signature = (
        [int(value) for value in list(target_tensor.shape_signature)]
        if target_tensor.shape_signature is not None
        else list(target_shape)
    )
    if (
        len(source_signature) != len(source_shape)
        or len(target_signature) != len(target_shape)
    ):
        return False
    merged = [
        -1 if int(source_dim) <= 0 else int(target_dim)
        for source_dim, target_dim in zip(source_signature, target_signature)
    ]
    if merged == target_signature:
        return False
    target_tensor.shape_signature = merged
    return True
