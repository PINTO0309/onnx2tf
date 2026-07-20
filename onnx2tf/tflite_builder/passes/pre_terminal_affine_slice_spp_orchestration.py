from __future__ import annotations

from typing import Any, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.pre_terminal_cleanup_orchestration import (
    run_pre_terminal_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_affine_slice_spp_orchestration import (
    run_terminal_affine_slice_spp_cleanup,
)


PreTerminalAffineSliceSPPContext = ModelIRPassContext
CleanupResults = Tuple[Any, ...]


def run_pre_terminal_affine_slice_spp_cleanup(
    context: PreTerminalAffineSliceSPPContext,
) -> Tuple[CleanupResults, CleanupResults]:
    """Run pre-terminal and affine/Slice/SPP cleanup in fixed order."""

    return (
        run_pre_terminal_cleanup(context),
        run_terminal_affine_slice_spp_cleanup(context),
    )
