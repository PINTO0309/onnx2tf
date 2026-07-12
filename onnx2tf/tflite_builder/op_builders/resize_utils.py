from __future__ import annotations

from typing import Any


def default_domain_opset(onnx_model: Any) -> int | None:
    if onnx_model is None:
        return None
    for opset in getattr(onnx_model, "opset_import", []):
        if str(getattr(opset, "domain", "")) in {"", "ai.onnx"}:
            return int(getattr(opset, "version", 0))
    return None


def resolve_resize_flags(
    *,
    node: Any,
    onnx_model: Any,
) -> tuple[str, str, bool, bool]:
    """Resolve Resize coordinate flags with the correct opset default."""
    mode = str(node.attrs.get("mode", "nearest")).lower()
    op_type = str(getattr(node, "op", ""))
    opset = default_domain_opset(onnx_model)
    legacy_asymmetric_default = bool(
        (op_type == "Upsample" and mode == "nearest")
        or (op_type == "Resize" and opset is not None and int(opset) <= 10)
    )
    default_ctm = "asymmetric" if legacy_asymmetric_default else "half_pixel"
    ctm = str(node.attrs.get("coordinate_transformation_mode", default_ctm)).lower()
    align_corners = bool(ctm == "align_corners")
    half_pixel_centers = bool(ctm in {"half_pixel", "pytorch_half_pixel"})
    if mode == "nearest" and ctm == "asymmetric":
        align_corners = False
        half_pixel_centers = False
    return mode, ctm, align_corners, half_pixel_centers
