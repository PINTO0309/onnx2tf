from __future__ import annotations

from typing import Any, Dict, List

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.nhwc_concat_layout import (
    run_nhwc_concat_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.nhwc_concat_legacy_layout import (
    optimize_transpose_pre_concat_nhwc_chains_legacy,
)
from onnx2tf.tflite_builder.passes.nhwc_concat_quantized_layout import (
    run_nhwc_concat_quantized_layout_cleanup,
)


_STATS_KEY = "optimized_transpose_pre_concat_nhwc_chains"
_INDEXED_STATS_KEYS = (
    "optimized_transpose_pre_concat_nhwc_direct_chains",
    "optimized_transpose_pre_concat_nhwc_unary_chains",
    "optimized_transpose_pre_concat_nhwc_pad_chains",
    "optimized_transpose_pre_concat_nhwc_dequantize_chains",
    "optimized_transpose_pre_concat_nhwc_prelu_chains",
    "optimized_transpose_pre_concat_nhwc_softmax_chains",
    "optimized_transpose_pre_concat_nhwc_swish_chains",
    "optimized_transpose_pre_concat_nhwc_slice_chains",
    "optimized_transpose_pre_concat_nhwc_split_chains",
    "optimized_transpose_pre_concat_nhwc_add_chains",
    "optimized_transpose_pre_concat_nhwc_leaky_chains",
)
_QUANTIZED_INDEXED_STATS_KEYS = (
    "optimized_transpose_pre_concat_nhwc_quantized_direct_chains",
    "optimized_transpose_pre_concat_nhwc_quantized_unary_chains",
    "optimized_transpose_pre_concat_nhwc_quantized_pad_chains",
    "optimized_transpose_pre_concat_nhwc_quantized_unary_pad_chains",
    "optimized_transpose_pre_concat_nhwc_quantized_all_pad_chains",
    "optimized_transpose_pre_concat_nhwc_quantized_swish_chains",
    "optimized_transpose_pre_concat_nhwc_quantized_dequantize_chains",
    "optimized_transpose_pre_concat_nhwc_quantized_prelu_chains",
    "optimized_transpose_pre_concat_nhwc_quantized_softmax_chains",
    "optimized_transpose_pre_concat_nhwc_quantized_leaky_chains",
    "optimized_transpose_pre_concat_nhwc_quantized_slice_chains",
    "optimized_transpose_pre_concat_nhwc_quantized_split_chains",
    "optimized_transpose_pre_concat_nhwc_quantized_add_chains",
)


def optimize_transpose_pre_concat_nhwc_chains(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Run indexed, quantized indexed, then legacy pre-Concat repairs."""

    indexed_stats = run_nhwc_concat_layout_cleanup(
        model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )
    quantized_indexed_stats = run_nhwc_concat_quantized_layout_cleanup(
        model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )
    legacy_stats = optimize_transpose_pre_concat_nhwc_chains_legacy(model_ir)
    optimized = sum(
        int(indexed_stats.get(stats_key, 0))
        for stats_key in _INDEXED_STATS_KEYS
    )
    optimized += sum(
        int(quantized_indexed_stats.get(stats_key, 0))
        for stats_key in _QUANTIZED_INDEXED_STATS_KEYS
    )
    optimized += int(legacy_stats.get(_STATS_KEY, 0))
    return {_STATS_KEY: int(optimized)}
