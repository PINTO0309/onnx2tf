from __future__ import annotations

from typing import Any, List, Tuple

import onnx

from onnx2tf.tflite_builder.pytorch_onnx_bridge_passes import (
    _onnx_fold_binary_layout_bridges_in_place,
    _onnx_fold_channel_front_concat_layout_bridges_in_place,
    _onnx_fold_channel_front_gathernd_transpose_bridges_in_place,
    _onnx_fold_concat_layout_bridges_in_place,
    _onnx_fold_mul_add_clip_to_hardsigmoid_in_place,
    _onnx_fold_pad_concat_layout_bridges_in_place,
    _onnx_fold_singleton_binary_layout_bridges_in_place,
    _onnx_fold_singleton_concat_layout_bridges_in_place,
    _onnx_fold_singleton_concat_slice_layout_bridges_in_place,
    _onnx_fold_singleton_resize_matmul_transpose_bridges_in_place,
    _onnx_fold_singleton_slice_layout_bridges_in_place,
    _onnx_fold_softmax_layout_bridges_in_place,
    _onnx_fold_unary_binary_layout_bridges_in_place,
    _onnx_fold_unary_binary_reduce_mean_layout_bridges_in_place,
)
from onnx2tf.tflite_builder.pytorch_onnx_layout_passes import (
    _onnx_fold_inverse_transpose_pairs_in_place,
    _onnx_fold_mul_reducesum_sigmoid_layout_bridges_in_place,
    _onnx_fold_reducesum_sigmoid_layout_bridges_in_place,
    _onnx_fold_relu_layout_bridges_in_place,
    _onnx_remove_passthrough_identity_nodes_in_place,
)
from onnx2tf.tflite_builder.pytorch_onnx_model_passes import (
    _onnx_optimize_pidnet_spp_transpose_bridges_in_place,
    _onnx_optimize_pphumanseg_add_resize_layout_bridges_in_place,
)
from onnx2tf.tflite_builder.pytorch_onnx_utils import (
    _onnx_fold_constant_binary_elementwise_in_place,
    _onnx_fold_constant_reshape_in_place,
    _onnx_fold_constant_scatter_nd_in_place,
    _onnx_repair_inferred_shapes_in_place,
)


def _optimize_dynamo_exported_onnx_in_place(model: onnx.ModelProto) -> None:
    def _value_info_signature(graph: onnx.GraphProto) -> Tuple[Any, ...]:
        signature: List[Tuple[str, Tuple[Any, ...]]] = []
        for value_info in list(graph.value_info) + list(graph.input) + list(graph.output):
            dims: List[Any] = []
            for dim in value_info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(int(dim.dim_value))
                elif dim.HasField("dim_param"):
                    dims.append(str(dim.dim_param))
                else:
                    dims.append("?")
            signature.append((str(value_info.name), tuple(dims)))
        return tuple(signature)

    def _graph_signature(graph: onnx.GraphProto) -> Tuple[Any, ...]:
        return (
            tuple(
            (
                str(node.name),
                str(node.op_type),
                tuple(str(v) for v in node.input),
                tuple(str(v) for v in node.output),
                tuple(
                    (
                        str(attr.name),
                        repr(onnx.helper.get_attribute_value(attr)),
                    )
                    for attr in node.attribute
                ),
            )
            for node in graph.node
            ),
            _value_info_signature(graph),
        )

    for _ in range(4):
        before_signature = _graph_signature(model.graph)
        _onnx_remove_passthrough_identity_nodes_in_place(model.graph)
        _onnx_fold_constant_scatter_nd_in_place(model.graph)
        _onnx_fold_constant_reshape_in_place(model.graph)
        _onnx_fold_constant_binary_elementwise_in_place(model.graph)
        _onnx_fold_relu_layout_bridges_in_place(model.graph)
        _onnx_fold_mul_reducesum_sigmoid_layout_bridges_in_place(model.graph)
        _onnx_fold_reducesum_sigmoid_layout_bridges_in_place(model.graph)
        _onnx_fold_inverse_transpose_pairs_in_place(model.graph)
        _onnx_fold_singleton_resize_matmul_transpose_bridges_in_place(model.graph)
        _onnx_fold_singleton_slice_layout_bridges_in_place(model.graph)
        _onnx_optimize_pidnet_spp_transpose_bridges_in_place(model.graph)
        _onnx_optimize_pphumanseg_add_resize_layout_bridges_in_place(model.graph)
        _onnx_fold_singleton_concat_slice_layout_bridges_in_place(model.graph)
        _onnx_fold_unary_binary_reduce_mean_layout_bridges_in_place(model.graph)
        _onnx_fold_unary_binary_layout_bridges_in_place(model.graph)
        _onnx_fold_binary_layout_bridges_in_place(model.graph)
        _onnx_fold_mul_add_clip_to_hardsigmoid_in_place(model.graph)
        _onnx_fold_singleton_binary_layout_bridges_in_place(model.graph)
        _onnx_fold_concat_layout_bridges_in_place(model.graph)
        _onnx_fold_channel_front_concat_layout_bridges_in_place(model.graph)
        _onnx_fold_channel_front_gathernd_transpose_bridges_in_place(model.graph)
        _onnx_fold_pad_concat_layout_bridges_in_place(model.graph)
        _onnx_fold_singleton_concat_layout_bridges_in_place(model.graph)
        _onnx_fold_softmax_layout_bridges_in_place(model.graph)
        _onnx_repair_inferred_shapes_in_place(model)
        if _graph_signature(model.graph) == before_signature:
            break
