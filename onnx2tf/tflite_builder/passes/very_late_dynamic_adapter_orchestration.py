from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.channel_shuffle import (
    run_stale_nchw_channel_shuffle_repair,
)
from onnx2tf.tflite_builder.passes.concat_global_pool_layout import (
    _repair_nchw_concat_global_pool_conv_axes as repair_nchw_concat_global_pool_conv_axes,
)
from onnx2tf.tflite_builder.passes.concat_transpose_conv_layout import (
    _repair_nchw_concat_transpose_conv_axes as repair_nchw_concat_transpose_conv_axes,
)
from onnx2tf.tflite_builder.passes.conv_input_adapter_repair import (
    run_indexed_conv_input_adapter_repairs_summary,
)
from onnx2tf.tflite_builder.passes.dynamic_reshape import (
    rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs,
)
from onnx2tf.tflite_builder.passes.dynamic_reshape_resolution import (
    resolve_dynamic_reshape_shapes,
)


VeryLateDynamicAdapterContext = ModelIRPassContext


def run_very_late_dynamic_adapter_cleanup(
    context: VeryLateDynamicAdapterContext,
) -> Tuple[
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
]:
    return (
        resolve_dynamic_reshape_shapes(
            context.model_ir,
            prefer_runtime_inferable_from_onnx_raw=True,
        ),
        run_indexed_conv_input_adapter_repairs_summary(context.model_ir),
        run_stale_nchw_channel_shuffle_repair(
            context.model_ir,
            layout_state=context.layout_state,
            diagnostics=context.diagnostics,
        ),
        repair_nchw_concat_transpose_conv_axes(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        repair_nchw_concat_global_pool_conv_axes(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs(
            context.model_ir,
            layout_state=context.layout_state,
        ),
    )
