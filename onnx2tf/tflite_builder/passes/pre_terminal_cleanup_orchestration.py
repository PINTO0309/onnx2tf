from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.channel_slice_pad_mul_orchestration import (
    run_channel_slice_pad_mul_summary,
)
from onnx2tf.tflite_builder.passes.pre_terminal_affine_tail_orchestration import (
    run_pre_terminal_affine_tail_cleanup,
)
from onnx2tf.tflite_builder.passes.pre_terminal_instancenorm_layout_orchestration import (
    run_pre_terminal_instancenorm_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.pre_terminal_pre_add_orchestration import (
    run_pre_terminal_pre_add_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_affine_concat_split_recovery_orchestration import (
    run_terminal_affine_concat_split_recovery_summary,
)


PreTerminalCleanupContext = ModelIRPassContext
InstanceNormCleanupResults = Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]
AffineTailCleanupResults = Tuple[Dict[str, int], Dict[str, int]]


def run_pre_terminal_cleanup(
    context: PreTerminalCleanupContext,
) -> Tuple[
    InstanceNormCleanupResults,
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    AffineTailCleanupResults,
]:
    """Run the complete pre-terminal cleanup stage in fixed order."""

    return (
        run_pre_terminal_instancenorm_layout_cleanup(context),
        run_terminal_affine_concat_split_recovery_summary(context),
        run_pre_terminal_pre_add_cleanup(context),
        run_channel_slice_pad_mul_summary(context),
        run_pre_terminal_affine_tail_cleanup(context),
    )
