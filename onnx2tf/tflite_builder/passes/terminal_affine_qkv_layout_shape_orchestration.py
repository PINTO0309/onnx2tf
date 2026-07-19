from __future__ import annotations

from typing import Any, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.pre_terminal_affine_slice_spp_orchestration import (
    run_pre_terminal_affine_slice_spp_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_qkv_activation_layout_shape_orchestration import (
    run_terminal_qkv_activation_layout_shape_cleanup,
)


TerminalAffineQKVLayoutShapeContext = ModelIRPassContext
CleanupResults = Tuple[Any, ...]


def run_terminal_affine_qkv_layout_shape_cleanup(
    context: TerminalAffineQKVLayoutShapeContext,
    *,
    include_layout_transpose: bool,
) -> Tuple[CleanupResults, CleanupResults]:
    """Run terminal affine and QKV/layout-shape cleanup in fixed order."""

    return (
        run_pre_terminal_affine_slice_spp_cleanup(context),
        run_terminal_qkv_activation_layout_shape_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
    )
