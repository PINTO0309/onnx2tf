from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.conv_mul_affine_fold_compat import (
    optimize_fold_conv_mul_add_affine_chains,
)
from onnx2tf.tflite_builder.passes.late_concat_layout_orchestration import (
    run_late_concat_layout_cleanup,
)


LateAffineConcatContext = ModelIRPassContext


def run_late_affine_concat_cleanup(
    context: LateAffineConcatContext,
) -> Tuple[Dict[str, int], Tuple[Dict[str, int], ...]]:
    """Run late Conv-affine folding and Concat/layout cleanup in order."""

    return (
        optimize_fold_conv_mul_add_affine_chains(
            context.model_ir,
            enable_conv_add_only_fold=True,
            layout_state=context.layout_state,
        ),
        run_late_concat_layout_cleanup(context),
    )
