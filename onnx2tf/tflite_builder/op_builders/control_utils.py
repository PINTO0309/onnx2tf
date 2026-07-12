from __future__ import annotations

from typing import Any


def tensor_shape_is_statically_proven(
    *,
    ctx: Any,
    tensor_name: str,
) -> bool:
    """Require tensor metadata and any ONNX shape hint to prove a static shape."""
    tensor = ctx.model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return False
    shape = [int(value) for value in list(tensor.shape)]
    signature = (
        [int(value) for value in list(tensor.shape_signature)]
        if tensor.shape_signature is not None
        else list(shape)
    )
    if (
        not shape
        or len(signature) != len(shape)
        or any(value <= 0 for value in shape)
        or any(value <= 0 for value in signature)
    ):
        return False
    raw_hint = (
        ctx.shape_map.get(str(tensor_name), None)
        if hasattr(ctx, "shape_map") and isinstance(ctx.shape_map, dict)
        else None
    )
    if raw_hint is None:
        return True
    try:
        raw_shape = [int(value) for value in raw_hint]
    except (TypeError, ValueError):
        return False
    return len(raw_shape) == len(shape) and all(value > 0 for value in raw_shape)
